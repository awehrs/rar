from src.docstore import Docstore
from src.utils import embed, tokenize

from pathlib import Path
import pytest
import shutil
import tempfile


@pytest.fixture
def mockstore():
    return {
        "model": "bert-base-uncased",
        "build_dir": Path("data", "mock"),
        "load_dir": Path("tests", "test_artifacts", "test_docstore"),
        "queries": [],
        "n_neighbors": 5,
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


def test_docstore_query_by_embeddings(mockstore):
    """Search batch of queries."""
    docstore = Docstore.load[mockstore["load_dir"]]
    token_ids = tokenize(mockstore["queries"], mockstore["model"])
    embedded_queries = embed(mockstore["model"], token_ids)
    idx = docstore.search()
    assert True


# docstore = Docstore.build(Path("data", "mock"), max_series=10000)
# docstore.save(Path("tests", "test_artifacts", "test_docstore"))
docstore = Docstore.load(Path("tests", "test_artifacts", "test_docstore"))
queries = ["blah blah blah", "blah, blah, blah"]
ids = tokenize(queries, model="bert-base-uncased", max_length=100)
queries = embed("bert-base-uncased", ids).numpy()

docstore.search_index(queries, 10)
