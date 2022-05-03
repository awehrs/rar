from src.docstore import Docstore

from pathlib import Path
import pytest
import random
import shutil
import tempfile

import numpy as np


@pytest.fixture
def mockstore():
    return {
        "model": "bert-base-uncased",
        "build_dir": Path("data", "mock"),
        "load_dir": Path("tests", "test_artifacts", "test_docstore"),
        "n_neighbors": 5,
        "metadata_fields": ["Description", "Units"],
    }


def test_docstore_build_and_save(mockstore):
    """Build Docstore from files and save to disk."""
    docstore = Docstore.build(mockstore["build_dir"])
    tmp_dir = tempfile.mkdtemp()
    docstore.save(tmp_dir)
    shutil.rmtree(tmp_dir)
    assert True


def test_docstore_load_from_disk(mockstore):
    """Load existing Docstore from disk."""
    Docstore.load(mockstore["load_dir"])
    assert True


def test_docstore_search(mockstore):
    """Ensure the top search result of a docstore element is itself."""
    docstore = Docstore.load(mockstore["load_dir"])
    n_neighbors = mockstore["n_neighbors"]
    query_idx = random.choices(
        list(range(len(docstore._huggingface_dataset))), k=n_neighbors
    )
    queries = docstore._huggingface_dataset[query_idx]["Embeddings"]
    queries = np.array(queries).astype(np.float32)
    _, result_idx = docstore.search(queries, n_neighbors)
    assert np.array_equal(np.array(query_idx), result_idx[:, 0])


def test_docstore_get_nearest_examples(mockstore):
    """Ensure only desired metadata fields are returned for nearest examples."""
    docstore = Docstore.load(mockstore["load_dir"])
    n_neighbors = mockstore["n_neighbors"]
    metadata_fields = mockstore["metadata_fields"]
    query_idx = random.choices(
        list(range(len(docstore._huggingface_dataset))), k=n_neighbors
    )
    queries = docstore._huggingface_dataset[query_idx]["Embeddings"]
    queries = np.array(queries).astype(np.float32)
    _, examples = docstore.get_nearest_examples(queries, n_neighbors, metadata_fields)
    keys = [e.keys() for e in examples]
    assert (keys.count(keys[0]) == len(keys)) and (list(keys[0]) == metadata_fields)
