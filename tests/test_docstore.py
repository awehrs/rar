from curses import meta
from src.docstore import (
    Docstore,
    chunk_data_arrays,
)

import os
import numpy as np
import pandas as pd
from pathlib import Path
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


def test_chunk_data_arrays_pad_location(chunk_input):

    """Test that pad_id appears only in last row."""

    pad_id = chunk_input["pad_id"]
    date_chunks, value_chunks = chunk_data_arrays(
        chunk_input["date_array"],
        chunk_input["value_array"],
        chunk_input["chunk_size"],
        pad_id,
    )
    assert np.any(date_chunks[:-1] != pad_id) and np.any(value_chunks[:-1] != pad_id)


def test_chunk_data_arrays_pad_count(chunk_input):

    """Test that number of pad_ids < chunk_size"""

    chunk_size = chunk_input["chunk_size"]
    pad_id = chunk_input["pad_id"]
    date_chunks, value_chunks = chunk_data_arrays(
        chunk_input["date_array"],
        chunk_input["value_array"],
        chunk_size,
        pad_id,
    )
    date_count = np.count_nonzero(date_chunks[-1] == pad_id)
    value_count = np.count_nonzero(value_chunks[-1] == pad_id)
    assert (date_count == value_count) and (date_count < chunk_size)


def test_chunk_data_arrays_output_shapes(chunk_input):

    """Test that chunk tensor has proper shape."""

    array_len = chunk_input["array_len"]
    chunk_size = chunk_input["chunk_size"]
    date_chunks, value_chunks = chunk_data_arrays(
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


def test_chunk_data_arrays_unequal_len_arrays(chunk_input):

    """Ensure that date and value arrays with unequal length throws error."""

    date_array = chunk_input["date_array"]
    date_array = np.append(date_array, date_array[-1])
    with pytest.raises(ValueError):
        chunk_data_arrays(
            date_array,
            chunk_input["value_array"],
            chunk_input["chunk_size"],
            chunk_input["pad_id"],
        )


# memory_map_folder_contents()


@pytest.fixture
def docstore_inputs():
    root_dir = Path(".") / TemporaryDirectory
    return Docstore(
        folder=os.path.join("data", "mock"),
        metadata_memmap_path=root_dir / "metadata.dat",
        tokens_memmap_path=root_dir / "tokens.dat",
        values_memmap_path=root_dir / "values.dat",
        dates_memmap_path=root_dir / "dates.dat",
        chunks_idx_memmap_path=root_dir / "chunks_idx.dat",
        max_series=200,
        chunk_size=100,
        max_chunks=10000,
    )


def test_docstore_init(docstore_inputs):
    assert True


root_dir = Path(mkdtemp())

Docstore(
    folder=os.path.join("data", "mock"),
    metadata_memmap_path=root_dir / "metadata.dat",
    tokens_memmap_path=root_dir / "tokens.dat",
    values_memmap_path=root_dir / "values.dat",
    dates_memmap_path=root_dir / "dates.dat",
    chunks_idx_memmap_path=root_dir / "chunks_idx.dat",
    save_index_to_disk=False,
    max_series=200,
    chunk_size=100,
    max_chunks=10000,
)
