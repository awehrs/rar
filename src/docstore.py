from contextlib import contextmanager
import functools
from importlib.metadata import metadata
import os
from typing import Dict, List

from einops import rearrange
import numpy as np
import pandas as pd
import torch

SOS_ID = 101
EOS_ID = 102

# Helper functions.


@contextmanager
def memmap(*args, **kwargs):
    pointer = np.memmap(*args, **kwargs)
    yield pointer
    del pointer


def range_chunked(max_value, *, batch_size):
    counter = 0
    while counter < max_value:
        curr = counter + batch_size
        curr = min(curr, max_value)
        yield slice(counter, curr)
        counter = curr


# Class defintion(s).


class Docstore:
    def __init__(
        self,
        folder: str,
        document_encoder,
        metadata_memmap_path: str,
        values_memmap_path: str,
        dates_memmap_path: str,
        chunks_idx_memmap_path: str,
        max_series: int,
        chunk_size: int,
        max_chunks: int,
        metadata_fields: List[str:int] = [
            "Description",
            "Units",
            "Source",
            "Start_Date",
            "End_Date",
        ],
        chunks_to_embeddings_batch_size: int = 16,
        embed_dim: int = 768,
        fields_to_embed: dict[str:int] = {"Description": 0, "Units": 1},
        index_file: str = "knn.index",
        max_metadata_seq_len: int = 100,
        max_rows_per_embedding_file: int = 500,
        pad_id: int = -1,
        use_cls_repr: bool = False,
        **index_kwargs,
    ) -> None:

        self.chunk_len = chunk_size
        self.max_chunks = max_chunks
        self.max_series = max_series
        self.metadata_fields = metadata_fields

        metadata_shape = (max_series, len(metadata_fields))
        data_shape = (max_chunks, chunk_size)
        chunks_idx_shape = (max_chunks,)

        # Lazily construct data/metadata memmap context managers.
        self.get_metadata = functools.partial(
            memmap, metadata_memmap_path, dtype=np.object_, shape=metadata_shape
        )
        self.get_values = functools.partial(
            memmap, values_memmap_path, dtype=np.float32, shape=data_shape
        )
        self.get_dates = functools.partial(
            memmap, dates_memmap_path, dtype=np.int64, shape=data_shape
        )
        self.get_chunks_idx = functools.partial(
            memmap, chunks_idx_memmap_path, dtype=np.int64, shape=chunks_idx_shape
        )

        # Create data/metadata memory maps.
        self.stats = memory_map_folder_contents(
            folder=folder,
            metadata_memmap_fn=self.get_metadata,
            dates_memmap_fn=self.get_dates,
            values_memmap_fn=self.get_dates,
            chunks_idx_memmap_fn=self.get_chunks_idx,
            chunk_size=chunk_size,
            pad_id=pad_id,
        )

        # Lazily construct embeddings memmap context manager.
        num_chunks = self.stats["chunks"]
        embeddings_path = f"{metadata_memmap_path}.embedded"
        embed_shape = (num_chunks, embed_dim)

        self.get_embeddings = functools.partial(
            memmap, embeddings_path, dtype=np.float32, shape=embed_shape
        )

        # Embed metadata and build index.
        self.index, self.embeddings = embed_and_index_metadata(
            document_encoder=document_encoder,
            num_series=self.stats["series"],
            max_metadata_seq_len=max_metadata_seq_len,
            embeddings_memmap_fn=self.get_embeddings,
            metadata_memmap_fn=self.get_metadata,
            metadata_fields=fields_to_embed,
            use_cls_repr=use_cls_repr,
            max_rows_per_embedding_file=max_rows_per_embedding_file,
            metadata_to_embeddings_batch_size=chunks_to_embeddings_batch_size,
            embed_dim=embed_dim,
            index_file=index_file,
            **index_kwargs,
        )

    def search_index(self):
        raise NotImplementedError

    def convert_to_torch(self):
        raise NotImplementedError


# Main functions.


def memory_map_folder_contents(
    *,
    folder: str,
    metadata_memmap_fn: functools.partial,
    dates_memmap_fn: functools.partial,
    values_memmap_fn: functools.partial,
    chunks_idx_memmap_fn: functools.partial,
    chunk_size: int,
    pad_id: int,
):
    """
    Iterate over series in data folder, and:
        - Memory map its metadata,
        - Memory map its chunked data,
        - Create mapping between series and indices of its
        chunked data within data memory map.

    Args:
        folder: Directory containing series subdirectories,
            each containing "data.csv" and "metadata.csv" files.
        metadata_memmap_fn: Partially constructed context manager for metadata memmap.
        dates_memmap_fn: Partially constructed context manager for dates memmap.
        values_memmap_fn: Partially constructed context manager for values memmap.
        chunks_idx_memmap_fn: Partially constructed context manager for
            memmap that maps series to their chunks' indices
        chunk_size: Chunk length for data
        pad_id: Pad value chunked data
    Returns:
        Dictionary containing total number of series and chunks
        in created memmory maps.
    """

    total_chunks = 0
    total_series = 0

    paths = [f.path for f in os.scandir(folder) if f.is_dir()]

    with (
        metadata_memmap_fn(mode="w+") as metadata,
        dates_memmap_fn(mode="w+") as dates,
        values_memmap_fn(mode="w+") as values,
        chunks_idx_memmap_fn(mode="w+") as chunks_idx,
    ):

        for path in paths:
            meta = pd.read_csv(os.path.join(path, "metadata.csv"))
            data = pd.read_csv(os.path.join(path, "data.csv"))
            date_array = data["Dates"]
            value_array = data["Values"]

            # Memory map the metadata fields.
            metadata[total_series] = meta.iloc[0]

            # Break data array into equal length chunks.
            date_chunks, value_chunks = chunk_data_arrays(
                date_array, value_array, chunk_size, pad_id
            )

            data_chunk_len = date_chunks.shape[0]
            chunk_slice = slice[total_chunks : (total_chunks + data_chunk_len)]

            # Memory map the data chunks and their indices.
            dates[chunk_slice] = date_chunks
            values[chunk_slice] = value_chunks
            chunks_idx[chunk_slice] = np.full((data_chunk_len,), total_series)

            total_chunks += data_chunk_len
            total_series += 1

    return dict(
        chunks=total_chunks,
        series=total_series,
    )


def chunk_data_arrays(
    date_array: np.ndarray,
    value_array: np.ndarray,
    chunk_size: int,
    pad_id: int,
):
    """
    Break arrays into equal size chunks.

    Args:
        date_array: Array of strings with shape [data_len,]
        value_array: Array of floats with shape [data_len,]
        chunk_size: Length of chunks
        pad_id: Int to pad with
    Returns:
        Chunked date and value arrays, each with shape:
            [num_chunks, chunk_size]
    """

    # Ensure arrays have equal length.
    if len(date_array) != len(value_array):
        raise ValueError(
            f"Length of dates array ({len(date_array)}) must equal"
            f" length of values array ({len(value_array)})"
        )

    # Pad to make arrays lengths multiple of chunk size.
    pad_len = chunk_size - (len(date_array) % chunk_size)

    if pad_len != chunk_size:
        date_array = np.pad(
            date_array, (0, pad_len), mode="constant", constant_values=pad_id
        )
        value_array = np.pad(
            value_array, (0, pad_len), mode="constant", constant_values=pad_id
        )

    # Reshape arrays.
    date_chunks = rearrange(date_array, "(n c) -> n c", c=chunk_size)
    value_chunks = rearrange(value_array, "(n c) -> n c", c=chunk_size)

    return date_chunks, value_chunks


def embed_and_index_metadata(
    *,
    document_encoder,
    num_series: int,
    max_metadata_seq_len: int,
    embeddings_memmap_fn: functools.partial,
    metadata_memmap_fn: functools.partial,
    metadata_fields: List[str],
    use_cls_repr: bool,
    max_rows_per_embedding_file: int,
    metadata_to_embeddings_batch_size: int,
    embed_dim: int,
    index_file: str,
    **index_kwargs,
):

    # Embed desired metadata fields.
    embed_metadata(
        document_encoder=document_encoder,
        num_series=num_series,
        max_metadata_seq_len=max_metadata_seq_len,
        embeddings_memmap_fn=embeddings_memmap_fn,
        meta_memmap_fn=metadata_memmap_fn,
        metadata_fields=metadata_fields,
        use_cls_repr=use_cls_repr,
        batch_size=metadata_to_embeddings_batch_size,
        embed_dim=embed_dim,
        pad_id=None,  # diff pad id?
    )

    # Memory map the embeddings.
    memmap_file_to_chunks_(
        embedding_path,
        shape=embed_shape,
        dtype=np.float32,
        folder=EMBEDDING_TMP_SUBFOLDER,
        max_rows_per_file=max_rows_per_file,
    )

    # Create index.
    index = index_embeddings(
        embeddings_folder=EMBEDDING_TMP_SUBFOLDER, index_file=index_file, **index_kwargs
    )

    # Memory map the embeddings.
    embeddings = np.memmap(
        embedding_path, shape=embed_shape, dtype=np.float32, mode="r"
    )

    return index, embeddings


def embed_metadata(
    *,
    document_encoder,
    num_series: int,
    max_metadata_seq_len: int,
    embeddings_memmap_fn: functools.partial,
    metadata_memmap_fn: functools.partial,
    metadata_fields: dict[str:int],
    use_cls_repr: bool,
    batch_size: int,
    embed_dim: int,
    pad_id=None,  # diff pad id?
):

    with (
        embeddings_memmap_fn(mode="w+") as embeddings,
        metadata_memmap_fn() as metadata,
    ):

        for dim_slice in range_chunked(num_series, batch_size=batch_size):
            # Get all metadata fields.
            batch_chunk_npy = metadata[dim_slice]

            # Filter for desired metadata fields.
            batch_chunk_npy = batch_chunk_npy[:, [metadata_fields.values()]]

            batch_chunk = torch.from_numpy(batch_chunk_npy)

            cls_tokens = torch.full((batch_chunk.shape[0], 1), SOS_ID)
            batch_chunk = torch.cat((cls_tokens, batch_chunk), dim=1)

            batch_chunk = batch_chunk[
                :, :-1
            ]  # omit last token, the first token of the next chunk, used for autoregressive training

            batch_embed = bert_embed(batch_chunk, return_cls_repr=use_cls_repr)

            embeddings[dim_slice] = batch_embed.detach().cpu().numpy()
            print(f"embedded {dim_slice.stop} / {num_chunks}")


# Map series index to chunks

# Concatenate and embed selected fields

# Break up embeddings into temp files, for autofaiss indexing

# Build date array
