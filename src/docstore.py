from contextlib import contextmanager
from functools import partial
import os
from typing import List

from einops import rearrange
import numpy as np
import pandas as pd


@contextmanager
def memmap(*args, **kwargs):
    pointer = np.memmap(*args, **kwargs)
    yield pointer
    del pointer


class Docstore:
    def __init__(
        self,
        folder: str,
        metadata_memmap_path: str,
        values_memmap_path: str,
        dates_memmap_path: str,
        chunks_idx_memmap_path: str,
        max_series: int,
        chunk_len: int,
        max_chunks: int,
        metadata_fields: List[str],
        pad_id: int = 0,
    ) -> None:

        self.chunk_len = chunk_len
        self.max_chunks = max_chunks
        self.max_series = max_series
        self.metadata_fields = metadata_fields

        metadata_shape = (max_series, len(metadata_fields))
        data_shape = (max_chunks, chunk_len)
        chunks_idx_shape = (max_chunks,)

        # Lazily construct memmap context managers.
        self.get_metadata = partial(
            memmap, metadata_memmap_path, dtype=np.object_, shape=metadata_shape
        )
        self.get_values = partial(
            memmap, values_memmap_path, dtype=np.float32, shape=data_shape
        )
        self.get_dates = partial(
            memmap, dates_memmap_path, dtype=np.object_, shape=data_shape
        )
        self.get_chunks_idx = partial(
            memmap, chunks_idx_memmap_path, dtype=np.int32, shape=chunks_idx_shape
        )

        # Fill in the memmaps.

        total_chunks = 0
        total_series = 0

        paths = [f.path for f in os.scandir(folder) if f.is_dir()]

        with (
            self.get_metadata(mode="w+") as metadata,
            self.get_values(mode="w+") as values,
            self.get_dates(mode="w+") as dates,
            self.get_chunks_idx(mode="w+") as chunks_idx,
        ):

            for path in paths:
                meta = pd.read_csv(os.path.join(path, "metadata.csv"))
                data = pd.read_csv(os.path.join(path, "data.csv"))
                date_array = data["Dates"]
                value_array = data["Values"]

                # Memory map the metadata fields.
                metadata[total_series] = meta.iloc[0]

                # Break data array into equal length chunks.
                date_chunks, value_chunks = chunk_data_array(
                    date_array, value_array, chunk_len, pad_id
                )

                data_chunk_len = date_chunks.shape[0]
                chunk_slice = slice[total_chunks : (total_chunks + data_chunk_len)]

                # Memory map the data chunks and their indices.
                dates[chunk_slice] = date_chunks
                values[chunk_slice] = value_chunks
                chunks_idx[chunk_slice] = np.full((data_chunk_len,), total_series)

                total_chunks += data_chunk_len
                total_series += 1

    # Attributes
    # Data directory
    # Metadata fields to embed
    # Max length of chunks
    # Max description length
    # Paths to various memory maps
    # Series
    # Description
    # Units
    # Source
    # Start / End date.
    # Dates
    # Values
    # Chunks
    # IDX
    # Metadata tokenizer
    # Document encoder
    # Max number of chunks per series
    # Dim of returned data vectors
    # Max number of chunks
    # Max number of series
    # Embedding dimension
    # Metadata padding id
    # Data padding id
    # Batch size
    # Index location
    # Stats (number of series, number of chunks, etc.)

    # Init
    # Counters:
    # total_chunks = 0
    # total_series = 0
    # Iterate through data directory
    # For each publisher/series:
    # Enter metadata field in respective metadata memmaps
    # Chunk the data, note the number of chunks
    # Create chunk indices associated with series
    # Put chunks in chunks memmap
    # Put indices in IDX memmap
    # Concatentate and embed desired fields
    # Build index (so that idx = series memmap idx = idx)

    # Methods
    # Search index
    # Takes: query (tokenized/embedded?), lookback period (date array)
    # Retrieves embeddings and idx of closest docs
    # Get associated metadata fields (e.g., units, source)
    # Get associated chunks that fall within the lookback period
    # Align those chunks with date array
    # Optionally perform normalization right here
    # Fill missing values
    # Optionally: sample
    # Shaped into fixed length vectors
    # Return [(key embedding, <value vectors>), (key embedding, <value vectors>)]


# IN LINE FUNCTIONS


def chunk_data_array(
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
    """

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
    date_chunks = rearrange(date_array, "(c n) -> c n", c=chunk_size)
    value_chunks = rearrange(value_array, "(c n) -> c n", c=chunk_size)

    return date_chunks, value_chunks


# Map series index to chunks

# Concatenate and embed selected fields

# Break up embeddings into temp files, for autofaiss indexing

# Build date array
