from src.docstore import Docstore, chunk_data_array

import pytest
import numpy as np
import pandas as pd

# Class tests.

# Pure function tests.


@pytest.fixture
def chunk_input():
    array_len = 100
    start_date = "20000101"

    return {
        "pad_id": -1,
        "chunk_size": 10,
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


def test_chunk_data_array_unequal_len_arrays(chunk_input):
    date_array = chunk_input["date_array"]
    date_array = np.append(date_array, date_array[-1])
    with pytest.raises(ValueError):
        chunk_data_array(
            date_array,
            chunk_input["value_array"],
            chunk_input["chunk_size"],
            chunk_input["pad_id"],
        )
