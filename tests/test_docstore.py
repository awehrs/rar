from src.docstore import (
    Docstore,
    chunk_data_array,
    memmap,
    memory_map_folder_contents,
)

import functools
import os
import numpy as np
import pandas as pd
import pytest
from tempfile import mkdtemp


# chunk_data_array()


@pytest.fixture
def chunk_input():

    array_len = 100
    start_date = "20000101"

    return {
        "pad_id": -1,
        "chunk_size": 11,
        "array_len": array_len,
        "value_array": np.random.uniform(size=(array_len,)),
        "date_array": np.array(
            [
                int(d.strftime("%Y%m%d"))
                for d in pd.date_range(start_date, periods=array_len)
            ]
        )[::-1],
    }


def test_chunk_data_array_pad_location(chunk_input):

    """Test that pad_id appears only in last row."""

    pad_id = chunk_input["pad_id"]
    date_chunks, value_chunks = chunk_data_array(
        chunk_input["date_array"],
        chunk_input["value_array"],
        chunk_input["chunk_size"],
        pad_id,
    )
    assert np.any(date_chunks[:-1] != pad_id) and np.any(value_chunks[:-1] != pad_id)


def test_chunk_data_array_pad_count(chunk_input):

    """Test that number of pad_ids < chunk_size"""

    chunk_size = chunk_input["chunk_size"]
    pad_id = chunk_input["pad_id"]
    date_chunks, value_chunks = chunk_data_array(
        chunk_input["date_array"],
        chunk_input["value_array"],
        chunk_size,
        pad_id,
    )
    date_count = np.count_nonzero(date_chunks[-1] == pad_id)
    value_count = np.count_nonzero(value_chunks[-1] == pad_id)
    assert (date_count == value_count) and (date_count < chunk_size)


def test_chunk_data_output_shapes(chunk_input):

    """Test that chunk tensor has proper shape."""

    array_len = chunk_input["array_len"]
    chunk_size = chunk_input["chunk_size"]
    date_chunks, value_chunks = chunk_data_array(
        chunk_input["date_array"],
        chunk_input["value_array"],
        chunk_size,
        chunk_input["pad_id"],
    )
    dim_1 = (
        (array_len // chunk_size) + 1
        if array_len % chunk_size != 0
        else array_len / chunk_size
    )
    assert date_chunks.shape == value_chunks.shape == (dim_1, chunk_size)


def test_chunk_data_array_unequal_len_arrays(chunk_input):

    """Ensure that date and value arrays with unequal length throws error."""

    date_array = chunk_input["date_array"]
    date_array = np.append(date_array, date_array[-1])
    with pytest.raises(ValueError):
        chunk_data_array(
            date_array,
            chunk_input["value_array"],
            chunk_input["chunk_size"],
            chunk_input["pad_id"],
        )


# memory_map_folder_contents()


@pytest.fixture
def memmap_input():

    # Declare constants.
    metadata_fields = [
        "Description",
        "Units",
        "Source",
        "Start_Date",
        "End_Date",
        "Publisher",
    ]

    fields_to_embed = ["Description", "Units"]

    data_dir = os.path.join("data", "mock")

    max_series = 100
    max_chunks = 10000
    max_metadata_seq_len = 200
    chunk_size = 100
    pad_id = 0

    # Create memmap context managers.
    root_dir = mkdtemp()

    metadata_memmap_path = os.path.join(root_dir, "metadata.dat")
    tokens_memmap_path = os.path.join(root_dir, "tokens.dat")
    values_memmap_path = os.path.join(root_dir, "values.dat")
    dates_memmap_path = os.path.join(root_dir, "dates.dat")
    chunks_idx_memmap_path = os.path.join(root_dir, "chunks_idx.dat")

    metadata_shape = (max_series, len(metadata_fields))
    tokens_shape = (max_series, max_metadata_seq_len)
    data_shape = (max_chunks, chunk_size)
    chunks_idx_shape = (max_chunks,)

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

    # doc encoder model

    return {
        "folder": data_dir,
        "metadata_memmap_fn": get_metadata,
        "tokens_memmap_fn": get_tokens,
        "dates_memmap_fn": get_dates,
        "values_memmap_fn": get_values,
        "chunks_idx_memmap_fn": get_chunks_idx,
        "doc_encoder_model": None,
        "fields_to_embed": fields_to_embed,
        "max_metadata_seq_len": max_metadata_seq_len,
        "pad_id": pad_id,
        "chunk_size": chunk_size,
    }


def test_memory_map_folder_contents(memmap_input):
    "Test that return values are accurate"


# chunk_embeddings_to_tmp_files()

# index_embeddings()

#
