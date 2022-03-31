from src.utils import embed, reset_folder_, tokenize

from contextlib import contextmanager
import functools
import logging
import os
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple

from autofaiss import build_index
from einops import rearrange
import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# Logging.

logger = logging.getLogger(name=__name__)
logger.setLevel(logging.INFO)


# Constants.

SOS_ID = 101
EOS_ID = 102

TMP_PATH = Path("./.tmp")
INDEX_FOLDER_PATH = Path("./data/.index")
EMBEDDING_TMP_SUBFOLDER = "embeddings"


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


def faiss_read_index(path):
    return faiss.read_index(str(path), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)


# Class defintion(s).


class Docstore:
    def __init__(
        self,
        folder: str,
        metadata_memmap_path: str = "metadata.dat",
        tokens_memmap_path: str = "tokens.dat",
        values_memmap_path: str = "values.dat",
        dates_memmap_path: str = "dates.dat",
        chunks_idx_memmap_path: str = "chunks_idx.dat",
        max_series: int = 10_000,
        chunk_size: int = 100,
        max_chunks: int = 10_000,
        metadata_fields: List[str] = [
            "Description",
            "Units",
            "Source",
            "Start_Date",
            "End_Date",
            "Publisher",
        ],
        chunks_to_embeddings_batch_size: int = 16,
        embed_dim: int = 768,
        fields_to_embed: List[str] = ["Description", "Units"],
        save_index_to_disk: bool = False,
        index_folder: str = "data/index",
        max_metadata_seq_len: int = 100,
        max_rows_per_embedding_file: int = 500,
        doc_encoder_model: str = "bert-base-uncased",
        pad_id: int = -1,
        use_cls_repr: bool = False,
        **index_kwargs,
    ) -> None:

        """
        Memory map metadata, chunks of dates and values, and
        tokens and embeddings of selected metadata fields.
        """

        metadata_shape = (max_series, len(metadata_fields))
        tokens_shape = (max_series, max_metadata_seq_len)
        data_shape = (max_chunks, chunk_size)
        chunks_idx_shape = (max_chunks,)

        # Lazily construct data/metadata memmap context managers.
        get_metadata = functools.partial(
            memmap, metadata_memmap_path, dtype=np.object_, shape=metadata_shape
        )
        get_tokens = functools.partial(
            memmap, tokens_memmap_path, dtype=np.int32, shape=tokens_shape
        )
        get_values = functools.partial(
            memmap, values_memmap_path, dtype=np.float32, shape=data_shape
        )
        get_dates = functools.partial(
            memmap, dates_memmap_path, dtype=np.int64, shape=data_shape
        )
        get_chunks_idx = functools.partial(
            memmap, chunks_idx_memmap_path, dtype=np.int64, shape=chunks_idx_shape
        )

        # Create data/metadata memory maps.
        logger.info("Memory mapping folder contents...")

        self.num_chunks, self.num_series = memory_map_folder_contents(
            folder=folder,
            metadata_memmap_fn=get_metadata,
            tokens_memmap_fn=get_tokens,
            dates_memmap_fn=get_dates,
            values_memmap_fn=get_values,
            chunks_idx_memmap_fn=get_chunks_idx,
            doc_encoder_model=doc_encoder_model,
            fields_to_embed=fields_to_embed,
            max_metadata_seq_len=max_metadata_seq_len,
            chunk_size=chunk_size,
            pad_id=pad_id,
        )

        # Lazily construct embeddings memmap context manager.
        embeddings_path = f"{tokens_memmap_path}.embedded"
        embed_shape = (self.num_series, embed_dim)

        get_embeddings = functools.partial(
            memmap, embeddings_path, dtype=np.float32, shape=embed_shape
        )

        # Embed desired metadata fields.
        logger.info("Embedding metadata fields...")

        embed_metadata(
            doc_encoder_model=doc_encoder_model,
            num_series=self.num_series,
            embeddings_memmap_fn=get_embeddings,
            tokens_memmap_fn=get_tokens,
            use_cls_repr=use_cls_repr,
            batch_size=chunks_to_embeddings_batch_size,
        )

        # Save embeddings to temporary files.
        logger.info("Preparing embeddings for indexing.")

        chunk_embeddings_to_tmp_files(
            embeddings_memmap_fn=get_embeddings,
            folder=EMBEDDING_TMP_SUBFOLDER,
            shape=embed_shape,
            max_rows_per_file=max_rows_per_embedding_file,
        )

        # Create index.
        logger.info("Building index...")

        self.index = index_embeddings(
            embeddings_folder=EMBEDDING_TMP_SUBFOLDER,
            index_folder=index_folder,
            save_index_to_disk=save_index_to_disk,
            **index_kwargs,
        )

        # Get memory map of embeddings directly.
        self.embeddings = np.memmap(
            embeddings_path, shape=embed_shape, dtype=np.float32, mode="r"
        )

    def search_index(self):
        raise NotImplementedError

    def save_index(self):
        raise NotImplementedError

    def convert_to_torch(self):
        raise NotImplementedError

    def convert_to_jnp(self):
        raise NotImplementedError


# Main functions.


def memory_map_folder_contents(
    *,
    folder: str,
    metadata_memmap_fn: functools.partial,
    tokens_memmap_fn: functools.partial,
    dates_memmap_fn: functools.partial,
    values_memmap_fn: functools.partial,
    chunks_idx_memmap_fn: functools.partial,
    doc_encoder_model: str,
    fields_to_embed: List[str],
    max_metadata_seq_len: int,
    chunk_size: int,
    pad_id: int,
) -> Tuple[int, int]:

    """

    Iterate over series in data folder, and:
        - Memory map its metadata,
        - Memory map its chunked data,
        - Create mapping between series and indices of its
        chunked data within data memory map.

    Args:
        folder: Directory containing series subdirectories,
            each containing "data.csv" and "metadata.csv" files.
        metadata_memmap_fn: Partially constructed context manager
            for metadata memmap.
        dates_memmap_fn: Partially constructed context manager for
            dates memmap.
        values_memmap_fn: Partially constructed context manager for
            values memmap.
        chunks_idx_memmap_fn: Partially constructed context manager
            for memmap that maps series to their chunks' indices.
        doc_encoder_model: Name of huggingface pretrained transformer.
        fields_to_embed: List of the names of fields to embed.
        max_metadata_seq_len: Length of token sequence to embed.
        chunk_size: Chunk length for data.
        pad_id: Pad value chunked data.
    Returns:
        Total number of chunks, total number of series.
    """

    total_chunks = 0
    total_series = 0

    paths = [f.path for f in os.scandir(folder) if f.is_dir()]

    with (
        metadata_memmap_fn(mode="w+") as metadata,
        tokens_memmap_fn(mode="w+") as tokens,
        dates_memmap_fn(mode="w+") as dates,
        values_memmap_fn(mode="w+") as values,
        chunks_idx_memmap_fn(mode="w+") as chunks_idx,
    ):

        for path in tqdm(paths):
            meta = pd.read_csv(os.path.join(path, "metadata.csv"))
            data = pd.read_csv(os.path.join(path, "data.csv"))
            date_array = data["Date"]
            value_array = data["Values"]

            # Memory map the metadata fields.
            metadata[total_series] = meta.iloc[0]

            # Memory map tokens of metadata fields to be embedded.
            text = meta.iloc[0][fields_to_embed].str.cat(sep=" ")

            ids = tokenize(text, doc_encoder_model)

            text_len = ids.shape[-1]

            padding = max_metadata_seq_len - text_len

            ids = F.pad(ids, (0, padding))

            tokens[total_series] = ids

            # Break data array into equal length chunks.
            date_chunks, value_chunks = chunk_data_arrays(
                date_array, value_array, chunk_size, pad_id
            )

            data_chunk_len = date_chunks.shape[0]
            chunk_slice = slice(total_chunks, (total_chunks + data_chunk_len))

            # Memory map the data chunks and their indices.
            dates[chunk_slice] = date_chunks
            values[chunk_slice] = value_chunks
            chunks_idx[chunk_slice] = np.full((data_chunk_len,), total_series)

            total_chunks += data_chunk_len
            total_series += 1

    return total_chunks, total_series


def chunk_data_arrays(
    date_array: np.ndarray,
    value_array: np.ndarray,
    chunk_size: int,
    pad_id: int,
) -> Tuple[np.ndarray, np.ndarray]:

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


def embed_metadata(
    *,
    doc_encoder_model: str,
    num_series: int,
    embeddings_memmap_fn: functools.partial,
    tokens_memmap_fn: functools.partial,
    use_cls_repr: bool,
    batch_size: int,
) -> None:

    """
    Break memory mapped token sequences into chunks.
    Batch encode the chunks.

    Args:
        doc_encoder_model: The name of the HuggingFace transformer
            model to use for document encoder.
        num_series: The number of token sequences to embed.
        embeddings_memmap_fn: Partially constructed context manager for
            embeddings memmap.
        tokens_memmap_fn: Partially constructed context manager for
            tokens memmap.
        use_cls_repr: Whether the document encoder should return mean of
            entire hidden state tensor, or just [CLS] vector.
        batch_size: Batch size for encoding token sequences.
    Returns:
        None.
    """

    with (
        embeddings_memmap_fn(mode="w+") as embeddings,
        tokens_memmap_fn() as tokens,
    ):

        for dim_slice in range_chunked(num_series, batch_size=batch_size):

            batch_chunk_npy = tokens[dim_slice]

            batch_chunk = torch.from_numpy(batch_chunk_npy)

            cls_tokens = torch.full((batch_chunk.shape[0], 1), SOS_ID)

            batch_chunk = torch.cat((cls_tokens, batch_chunk), dim=1)

            batch_embed = embed(
                model=doc_encoder_model,
                token_ids=batch_chunk,
                return_cls_repr=use_cls_repr,
            )

            embeddings[dim_slice] = batch_embed.detach().cpu().numpy()


def chunk_embeddings_to_tmp_files(
    embeddings_memmap_fn: functools.partial,
    *,
    folder: str,
    shape: Tuple[int, int],
    max_rows_per_file: int,
) -> None:

    """
    Use memory map of embeddings to create temporary
    .npy files, to use for creating faiss index.

    Args:
        embeddings_memmap_fn: Partially constructed context manager for
            embeddings memmap.
        folder: Path to folder where .npy files will be placed.
        shape: The shape of the embedding memmap.
        max_rows_per_file: Number of rows from memmap to put .npy file.
    Returns:
        None.
    """
    rows, _ = shape

    with embeddings_memmap_fn(mode="r") as f:
        root_path = TMP_PATH / folder
        reset_folder_(root_path)

        for ind, dim_slice in enumerate(
            range_chunked(rows, batch_size=max_rows_per_file)
        ):
            filename = root_path / f"{ind}.npy"
            np.save(str(filename), f[dim_slice])


def index_embeddings(
    embeddings_folder: str,
    save_index_to_disk: bool,
    index_file="knn.index",
    index_infos_file="index_infos.json",
    max_index_memory_usage="100m",
    current_memory_available="1G",
    **index_kwargs,
) -> faiss.Index:

    """
    Build index from temporary .npy files of embeddings.

    Args:
        embeddings_folder: Temporary folder with .npy files.
        save_index_to_disk: Whether to persist index after knn calculations.
        index_file: File to save index to.
        index_infos_file: File to save index info to.
        max_index_memory_usage: Maximum amount of memory index can use.
        current_memory_available: Current memory available.
    Returns:
        Faiss index.
    """

    embeddings_path = TMP_PATH / embeddings_folder

    index_path = INDEX_FOLDER_PATH / index_file

    if save_index_to_disk == True:
        reset_folder_(INDEX_FOLDER_PATH)

    index = build_index(
        embeddings=str(embeddings_path),
        index_path=str(index_path),
        index_infos_path=str(INDEX_FOLDER_PATH / index_infos_file),
        save_on_disk=save_index_to_disk,
        max_index_memory_usage=max_index_memory_usage,
        current_memory_available=current_memory_available,
        should_be_memory_mappable=True,
        use_gpu=torch.cuda.is_available(),
    )

    return index
