from src.docstore import (
    Docstore,
)

import os
import numpy as np
import pandas as pd
from pathlib import Path
import pytest
from tempfile import mkdtemp


@pytest.fixture
def docstore_inputs():

    root_dir = Path(mkdtemp())

    return Docstore(
        folder=os.path.join("data", "mock"),
        tokens_memmap_path=root_dir / "tokens.dat",
        data_memmap_path=root_dir / "data.dat",
        chunks_idx_memmap_path=root_dir / "chunks_idx.dat",
    )


def test_docstore_init_from_files(docstore_inputs):
    docstore = docstore_inputs
    assert True


def test_docstore_load_from_path(docstore_inputs):
    raise NotImplementedError


def test_docstore_search(docstore_inputs):
    raise NotImplementedError


def test_docstore_convert_to_torch(docstroe_inputs):
    raise NotImplementedError
