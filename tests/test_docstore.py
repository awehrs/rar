from src.docstore import Docstore

import itertools
from pathlib import Path
import pytest
import random
from typing import Dict, List

import numpy as np
import pandas as pd
import torch


random.seed(42)


@pytest.fixture
def mockstore(autouse=True):
    """Basic docstore."""
    return Docstore.load(Path("tests", "test_artifacts", "test_docstore"))


@pytest.fixture
def query_idx(mockstore, query_parameters):
    """Random docstore indices to use as queries."""
    return random.choices(
        list(range(len(mockstore._huggingface_dataset))),
        k=(query_parameters["batch_size"] * query_parameters["num_queries"]),
    )


@pytest.fixture
def query_array(mockstore, query_idx):
    """An array of queries."""
    queries = mockstore._huggingface_dataset[query_idx]["Embeddings"]
    return np.array(query_idx), np.array(queries).astype(np.float32)


@pytest.fixture
def query_parameters():
    return {
        "batch_size": 4,
        "k_neighbors": 5,
        "extra_neighbors": 2,
        "num_queries": 10,
        "search_date": "1/1/2020",
        "coverage_ratio": 0.6,
        "num_samples": 100,
        "resample_freq": "M",
        "aggregation_fn": "mean",
        "metadata_field": "Embeddings",
        "pad_val": 0.0,
    }


@pytest.fixture
def data_dfs(query_parameters):
    """Mock data inputs to resample."""
    k_neighbors = query_parameters["k_neighbors"]
    batch_size = query_parameters["batch_size"]
    num_queries = query_parameters["num_queries"]
    num_samples = query_parameters["num_samples"]
    search_date = query_parameters["search_date"]
    date_range = pd.date_range(end=search_date, freq="M", periods=num_samples)
    return [
        pd.DataFrame(np.full((k_neighbors, num_samples), i), columns=date_range)
        for i in range(batch_size * num_queries)
    ]


@pytest.fixture
def meta_dfs(query_parameters):
    """Mock metadata inputs to resample."""
    k_neighbors = query_parameters["k_neighbors"]
    batch_size = query_parameters["batch_size"]
    num_queries = query_parameters["num_queries"]
    return [
        pd.DataFrame(np.full((k_neighbors, 768), i))
        for i in range(batch_size * num_queries)
    ]


def test_docstore_build_and_save(tmp_path):
    """Build Docstore from files and save to disk."""
    data_dir = Path("data", "mock")
    docstore = Docstore.build(data_dir)
    docstore.save(tmp_path)
    assert True


def test_docstore_load_from_disk():
    """Load existing Docstore from disk."""
    docstore_dir = Path("tests", "test_artifacts", "test_docstore")
    Docstore.load(docstore_dir)
    assert True


def test_docstore_search(mockstore, query_array, query_parameters):
    """Ensure the top search result of a docstore element is itself."""
    query_idx, query_embeddings = query_array
    _, result_idx = mockstore.search(query_embeddings, query_parameters["k_neighbors"])
    assert np.array_equal(query_idx, result_idx[:, 0])


def test_docstore_get_nearest_examples(mockstore, query_array, query_parameters):
    """Ensure only desired metadata column is returned for nearest examples."""
    _, query_embeddings = query_array
    metadata_column = "Embeddings"
    _, examples = mockstore.get_nearest_examples(
        query_embeddings, query_parameters["k_neighbors"], metadata_column
    )
    keys = [e.keys() for e in examples]
    assert (keys.count(keys[0]) == len(keys)) and (list(keys[0])[-1] == metadata_column)


class TestResampleData:
    def test_order_preservation(self, mockstore, data_dfs, query_parameters):
        """Test that results aren't shuffled and no indices masked."""
        data, _ = mockstore.resample_data(
            data_dfs=data_dfs,
            search_date=query_parameters["search_date"],
            num_samples=query_parameters["num_samples"],
            resample_freq=query_parameters["resample_freq"],
            aggregation_fn=query_parameters["aggregation_fn"],
            coverage_ratio=query_parameters["coverage_ratio"],
        )
        ordered = all(
            data[i].iat[0, 0] < data[i + 1].iat[0, 0] for i in range(len(data) - 1)
        )
        assert ordered

    def test_coverage_filter(self, mockstore, data_dfs, query_parameters):
        """All indices should be invalid."""
        _, valid_indices = mockstore.resample_data(
            data_dfs=data_dfs,
            search_date=query_parameters["search_date"],
            num_samples=int(
                query_parameters["num_samples"]
                * (1 / query_parameters["coverage_ratio"])
            )
            + 1,
            resample_freq=query_parameters["resample_freq"],
            aggregation_fn=query_parameters["aggregation_fn"],
            coverage_ratio=query_parameters["coverage_ratio"],
        )
        assert all([df.values.sum() != len(df) for df in valid_indices])


class TestRestructureDateAligned:
    def valid_indices(
        self, batch_size: int, index_len: int, num_null_rows: int, num_queries: int
    ) -> List[pd.Series]:
        """Construct mock valid index filters, each with specifid number of "False" entries."""
        indices_list = []
        for _ in range(batch_size * num_queries):
            bool_list = ((index_len - num_null_rows) * [True]) + (
                num_null_rows * [False]
            )
            random.shuffle(bool_list)
            indices_list.append(pd.Series(bool_list))
        return indices_list

    def execute_test_function(
        self,
        batch_size: int,
        k_neighbors: int,
        num_null_rows: int,
        num_queries: int,
        num_samples: int,
        pad_val: float,
        docstore: Docstore,
        data_dfs: List[pd.DataFrame],
        meta_dfs: List[pd.DataFrame],
    ) -> Dict[str, torch.Tensor]:
        """Boilerplate for calling the function being tested."""

        valid_indices = self.valid_indices(
            batch_size=batch_size,
            index_len=k_neighbors,
            num_null_rows=num_null_rows,
            num_queries=num_queries,
        )

        return docstore.restructure_date_aligned(
            data=data_dfs,
            meta=meta_dfs,
            valid_indices=valid_indices,
            k_neighbors=k_neighbors,
            num_queries=num_queries,
            num_samples=num_samples,
            pad_val=pad_val,
        )

    def test_expected_shape(
        self,
        data_dfs,
        meta_dfs,
        mockstore,
        query_parameters,
    ):
        """Ensure output tensors have correct shape."""

        batch_size = query_parameters["batch_size"]
        num_queries = query_parameters["num_queries"]
        k_neighbors = query_parameters["k_neighbors"]
        num_samples = query_parameters["num_samples"]
        pad_val = query_parameters["pad_val"]

        output_dict = self.execute_test_function(
            batch_size=batch_size,
            k_neighbors=k_neighbors,
            num_null_rows=0,
            num_queries=num_queries,
            num_samples=num_samples,
            pad_val=pad_val,
            docstore=mockstore,
            data_dfs=data_dfs,
            meta_dfs=meta_dfs,
        )
        data = output_dict["data"]
        meta = output_dict["metadata"]
        mask = output_dict["mask"]

        assert (
            data.shape
            == (
                batch_size,
                num_queries,
                k_neighbors,
                num_samples,
            )
            and meta.shape[:-1] == (batch_size, num_queries, k_neighbors)
            and mask.shape == (batch_size, num_queries, k_neighbors)
        )

    def test_order_preservation(self, data_dfs, meta_dfs, mockstore, query_parameters):
        """Ensure order of data isn't shuffled."""
        batch_size = query_parameters["batch_size"]
        num_queries = query_parameters["num_queries"]
        k_neighbors = query_parameters["k_neighbors"]
        num_samples = query_parameters["num_samples"]
        pad_val = query_parameters["pad_val"]

        output_dict = self.execute_test_function(
            batch_size=batch_size,
            k_neighbors=k_neighbors,
            num_null_rows=0,
            num_queries=num_queries,
            num_samples=num_samples,
            pad_val=pad_val,
            docstore=mockstore,
            data_dfs=data_dfs,
            meta_dfs=meta_dfs,
        )
        data = output_dict["data"]
        meta = output_dict["metadata"]

        ordered = [
            all(
                _tensor[
                    i,
                    j,
                    :,
                    :,
                ].sum()
                < _tensor[i + 1, j + 1, :, :].sum()
                for (i, j) in itertools.product(
                    range(batch_size - 1), range(num_queries - 1)
                )
            )
            for _tensor in [data, meta]
        ]

        assert all(ordered)

    def test_all_indices_invalid(self, data_dfs, meta_dfs, mockstore, query_parameters):
        batch_size = query_parameters["batch_size"]
        num_queries = query_parameters["num_queries"]
        k_neighbors = query_parameters["k_neighbors"]
        num_samples = query_parameters["num_samples"]
        pad_val = query_parameters["pad_val"]

        output_dict = self.execute_test_function(
            batch_size=batch_size,
            k_neighbors=k_neighbors,
            num_null_rows=k_neighbors,
            num_queries=num_queries,
            num_samples=num_samples,
            pad_val=pad_val,
            docstore=mockstore,
            data_dfs=data_dfs,
            meta_dfs=meta_dfs,
        )
        data = output_dict["data"]
        meta = output_dict["metadata"]
        mask = output_dict["mask"]

        assert (
            torch.equal(data, torch.full(data.shape, pad_val))
            and torch.equal(meta, torch.full(meta.shape, pad_val))
            and torch.equal(mask, torch.ones(mask.shape))
        )

    def test_some_indices_invalid(
        self, mockstore, data_dfs, meta_dfs, query_parameters
    ):
        """Ensure that only invalid indices are padded."""
        batch_size = query_parameters["batch_size"]
        num_queries = query_parameters["num_queries"]
        k_neighbors = query_parameters["k_neighbors"]
        num_samples = query_parameters["num_samples"]
        pad_val = -1.0

        output_dict = self.execute_test_function(
            batch_size=batch_size,
            k_neighbors=k_neighbors,
            num_null_rows=k_neighbors // 2,
            num_queries=num_queries,
            num_samples=num_samples,
            pad_val=pad_val,
            docstore=mockstore,
            data_dfs=data_dfs,
            meta_dfs=meta_dfs,
        )
        data = output_dict["data"]
        meta = output_dict["metadata"]
        mask = output_dict["mask"]

        data_tests = []
        meta_tests = []
        mask_tests = []

        for (i, j, k) in itertools.product(
            range(batch_size), range(num_queries), range(k_neighbors)
        ):
            if k <= k_neighbors // 2:
                data_tests.append(
                    not torch.equal(
                        data[i, j, k, :],
                        torch.full((data.shape[-1],), pad_val),
                    )
                )
                meta_tests.append(
                    not torch.equal(
                        meta[i, j, k, :],
                        torch.full((meta.shape[-1],), pad_val),
                    )
                )
                mask_tests.append(mask[i, j, k].item() == 0)
            else:
                data_tests.append(
                    torch.equal(
                        data[i, j, k, :], torch.full((data.shape[-1],), pad_val)
                    )
                )
                meta_tests.append(
                    torch.equal(
                        meta[i, j, k, :], torch.full((meta.shape[-1],), pad_val)
                    )
                )
                mask_tests.append(mask[i, j, k].item() == 1)

        assert all(data_tests) and all(meta_tests) and all(mask_tests)
