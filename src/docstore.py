from concurrent.futures import thread
from src.utils import embed, get_model, get_tokenizer, tokenize

from inspect import signature
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable, Dict, List, Tuple, Union

from autofaiss import build_index
import datasets
import faiss
import numpy as np
import pandas as pd
from pyarrow import dataset
import torch


# Logging.

logger = logging.getLogger(name=__name__)
logger.setLevel(logging.INFO)


# Constants.

SOS_ID = 101
EOS_ID = 102


# Docstore Class.


class Docstore:
    def __init__(
        self,
        *,
        faiss_index: faiss.swigfaiss.Index,
        huggingface_dataset: datasets.Dataset,
        config_dict: Dict,
    ) -> None:
        """
        Docstore object.

        Args:
            faiss_index: A faiss index object.
            huggingface_dataset: A huggingface dataset wrapper around
                a pyarrow table. Rows are unique time series. Columns
                are either (a) metadata: the same for each series,
                with various value types, and (b) timestamps: with
                observations being either a float or null, for a given
                series and timestamp.
            config_dict: Dictionary containing metadata about the docstore:
                faiss parameters, encoding model, etc.
        Returns:
            Docstore instance.
        """

        self._faiss_index = faiss_index
        self._huggingface_dataset = huggingface_dataset
        self._config_dict = config_dict

    def __getitem__(self, indices: Union[int, list]):
        """Slice huggingface datset by index."""
        return self._huggingface_dataset[indices]

    @property
    def num_metadata_fields(self) -> int:
        """Get the number of metadata columns in dataset."""
        return self._config_dict["num_metadata_fields"]

    @property
    def time_index(self) -> List[str]:
        """Get the common time index for series' values."""
        # Exclude last two columns ("Tokens", "Embeddings").
        return self._huggingface_dataset.column_names[self.num_metadata_fields : -2]

    # Factory functions.

    @classmethod
    def load(
        cls,
        docstore_dir: Union[str, Path],
    ) -> object:
        """
        Load index, dataset, and config from disk.

        Args:
            docstore_dir: Directory holding memory maps, index, and
                configuation dictionary.
        Returns:
            Docstore instance.
        """
        # Check that relevant files exist.
        root_dir = Path(docstore_dir)

        _paths = dict(
            index_path=cls.index_path(root_dir),
            config_path=cls.config_path(root_dir),
        )
        for _path in _paths.values():
            if not os.path.isfile(_path):
                raise ValueError(
                    f"The file {_path} files does not exist in indicated directory."
                )
        # Load object configuration.
        _config_dict: dict = {}

        with open(_paths["config_path"], "r") as config_file:
            _config_dict = json.load(config_file)

        init_params = dict(config_dict=_config_dict)

        # Fetch index and dataset.
        init_params["faiss_index"] = faiss.read_index(_paths["index_path"])
        init_params["huggingface_dataset"] = datasets.load_from_disk(root_dir)

        return cls(**init_params)

    @classmethod
    def build(
        cls,
        data_dir: Union[str, Path],
        doc_encoder_model: str = "bert-base-uncased",
        fields_to_embed: List[str] = ["Description", "Units"],
        num_metadata_fields: int = 6,
        max_metadata_seq_len: int = 100,
        max_rows_per_embedding_file: int = 500,
        metadata_to_embedding_batch_size: int = 16,
        use_cls_repr: bool = False,
        **index_kwargs,
    ) -> object:
        """
        Build Docstore from local files.

        Args:
            data_dir: Local directory with data.
            doc_encoder_model: Model name of document encoder.
            fields_to_embed: The metadata fields to concatenate, embed,
                and index.
            num_metadata_fields: The number of columns of metadata in
                data files.
            max_metadata_seq_len: Maximum number of tokens in metadata
                sequence to be embedded.
            max_rows_per_embedding_file: Maximum number of embeddings per
                temporary .npy file.
            metadata_to_embeddings_batch_size: Batch size for tokenizing and
                embedding metadata fields.
            use_cls_repr: Whether to represent metadata embedding by
                [CLS] token. If false, mean of final latent values is
                used.
        Returns:
            Docstore instance.
        """
        logger.info("Building Docstore from files...")

        # Start with arrow dataset for automatic schema inference.
        arrow_dataset = dataset.dataset(data_dir, format="csv")
        tmp_arrow_dir = tempfile.mkdtemp()
        dataset.write_dataset(arrow_dataset, tmp_arrow_dir, format="parquet")

        # Convert to huggingface dataset for batching capabilities.
        hf_dataset = datasets.load_dataset("parquet", data_dir=tmp_arrow_dir)
        hf_dataset = hf_dataset["train"]

        del arrow_dataset
        shutil.rmtree(tmp_arrow_dir)

        # Tokenize and embed desired metadata fields.
        logger.info("Tokenizing and embedding metadata fields...")

        tokenizer = get_tokenizer(doc_encoder_model)

        encoder = get_model(doc_encoder_model)

        hf_dataset = hf_dataset.map(
            tokenize_and_embed,
            batched=True,
            batch_size=metadata_to_embedding_batch_size,
            fn_kwargs={
                "tokenizer": tokenizer,
                "encoder": encoder,
                "columns_to_embed": fields_to_embed,
                "max_metadata_seq_len": max_metadata_seq_len,
                "use_cls_repr": use_cls_repr,
            },
        )

        logger.info("Preparing embeddings for indexing...")

        # Save embeddings to temporary files.
        tmp_embeddings_dir = tempfile.mkdtemp()

        hf_dataset.map(
            chunk_embeddings_to_tmp_files,
            batched=True,
            batch_size=max_rows_per_embedding_file,
            with_indices=True,
            fn_kwargs={
                "embeddings_dir": tmp_embeddings_dir,
                "batch_size": max_rows_per_embedding_file,
            },
        )

        # Create index.
        logger.info("Building index...")

        faiss_index, index_info = index_embeddings(
            embeddings_folder=tmp_embeddings_dir,
            **index_kwargs,
        )

        shutil.rmtree(tmp_embeddings_dir)

        # Create configuration dictionary.
        config_dict = {"index_info": index_info}

        sig = signature(cls.build)
        _locals = locals()

        for param in sig.parameters.values():
            if param.name in _locals:
                config_dict[param.name] = _locals[param.name]

        # Instantiate object.
        return cls(
            faiss_index=faiss_index,
            huggingface_dataset=hf_dataset,
            config_dict=config_dict,
        )

    def save(
        self,
        directory: Union[str, Path],
    ) -> None:

        """
        Save Docstore built from files.

        Args:
            directory: Directory in which to save dataset,
                index, and configuration dictionary.
        Returns:
            None
        """
        logging.info("Saving index, memory maps, and Docstore config...")

        # Get/create relevant paths.
        if not Path(directory).exists():
            Path.mkdir(directory, parents=True)

        # Save dataset.
        self._huggingface_dataset.save_to_disk(directory)

        # Save index.
        index_path = self.index_path(directory)
        faiss.write_index(self._faiss_index, str(index_path))

        # Save object configuration.
        config_path = self.config_path(directory)

        with open(config_path, "w") as f:
            json.dump(self._config_dict, f, default=str)

    # Search / filter methods.

    def search(
        self,
        query_embeddings: np.array,
        n_neighbors: int,
    ) -> Tuple[List[List[float]], List[List[int]]]:
        """
        Return the scores and indices of the nearest examples to a
            given batch of queries.

        Args:
            query_embeddings: Array of shape [batch_size, num_queries, embedding_dim].
            n_neighbors: the number of results to fetch per query in batch.
        Returns:
            scores: The retrieval scores of the retrieved examples per query.
            indices: The indices of the retrieved examples per query.
        """
        if len(query_embeddings.shape) != 2:
            raise ValueError("Shape of query must be 2D")

        if not query_embeddings.flags.c_contiguous:
            query_embeddings = np.asarray(query_embeddings, order="C")

        scores, indices = self._faiss_index.search(query_embeddings, n_neighbors)

        return scores, indices.astype(int)

    def transform_data(
        self,
        data_dfs: List,
        search_date: str,
        num_samples: int,
        resample_freq: str,
        aggregation_fn: str,
        dropna_coverage_ratio: int,
    ):
        # Concat dataframes to single dataframe.
        batch_size = len(data_dfs)
        df = pd.concat(data_dfs)

        # Convert columns to datetime.
        df.columns = pd.to_datetime(df.columns, format="%Y-%m-%d")

        # Resample.
        df = df.resample(rule=resample_freq, axis=1).agg(aggregation_fn).ffill(axis=1)

        # Slice by date.
        df = df.loc[:, :search_date]
        df = df.iloc[:, len(df.columns) - num_samples :]

        # Split dataframe back into batches.
        dropna_threshold = int(len(df.columns) * (dropna_coverage_ratio / 100))

        transformed_dfs = [
            _df.dropna(axis=0, thresh=dropna_threshold)
            for _df in np.split(df, batch_size, axis=0)
        ]

        # Normalize.

        raise NotImplementedError

    def filter_by_date(self):
        raise NotImplementedError

    def fill_na(self):
        raise NotImplementedError

    def normalize(self):
        raise NotImplementedError

    def get_nearest_examples(
        self,
        query_embedding: np.array,
        n_neighbors: int,
        search_date,
        num_samples: int,
        frequency: str = "M",
        aggregation_fn: str = "mean",
        dropna_coverage_ratio: int = 80,
        metadata_field: str = "Embeddings",
    ) -> Tuple[List[List[float]], List[dict]]:
        """
        Return the rows in the huggingface_dataset (filtered by
            requested columns) of the nearest examples for a batch
            of search queries.

        Returns:
            total_scores: The retrieval scores of the retrieved examples per query.
            total_examples: The retrieved examples per query.

        """
        if metadata_field not in ["Embeddings", "Tokens"]:
            raise ValueError  # TODO specify ValueError

        if aggregation_fn not in ["min", "max", "mean", "last"]:
            raise ValueError  # TODO specify ValueError

        if not 0 < dropna_coverage_ratio <= 100:
            raise ValueError  # TODO specify ValueError

        total_scores, total_indices = self.search(query_embedding, n_neighbors)

        total_scores = [
            scores_i[: len([i for i in indices_i if i >= 0])]
            for scores_i, indices_i in zip(total_scores, total_indices)
        ]
        total_samples = [
            self._huggingface_dataset[[i for i in indices if i >= 0]]
            for indices in total_indices
        ]

        # Break up retrieved samples into data and metadata.
        sample_dfs = [pd.DataFrame(sample) for sample in total_samples]

        data = [df[self.time_index] for df in sample_dfs]

        metadata = [df[metadata_field] for df in sample_dfs]

        # Filter / fill / sample data.
        # transformed_data = self.transform_data(
        #     data, search_date, start_date, num_samples, na_fill_type
        # )

        return total_scores, total_samples

    # Static utility methods.

    @staticmethod
    def dataset_path(docstore_dir: Union[str, Path]) -> Path:
        """Return path of huggingface dataset object."""
        return Path(docstore_dir / "hf_dataset")

    @staticmethod
    def index_path(docstore_dir: Union[str, Path]) -> str:
        """Return path of faiss index, as a string."""
        return str(Path(docstore_dir) / "index.faiss")

    @staticmethod
    def config_path(docstore_dir: Union[str, Path]) -> Path:
        """Return path of configuration dictionary."""
        return Path(docstore_dir, "docstore_config").with_suffix(".json")


# Docstore.build() helper methods.


def tokenize_and_embed(
    example_batch: list[dict],
    tokenizer: Callable,
    encoder: Callable,
    columns_to_embed: List[int],
    max_metadata_seq_len: int,
    use_cls_repr: bool,
) -> Dict:

    """ """
    # Concatenate desired metadata fields.
    concatenated_text = list(
        map(" ".join, zip(*[example_batch[column] for column in columns_to_embed]))
    )

    # Tokenize concatenated fields.
    token_ids = tokenize(
        texts=concatenated_text,
        tokenizer=tokenizer,
        max_length=max_metadata_seq_len,
        add_special_tokens=use_cls_repr,
    )

    # Encode token ids.
    embeddings = embed(
        token_ids=token_ids,
        encoder=encoder,
        return_cls_repr=use_cls_repr,
    )

    # Update dataset batch.
    example_batch["Tokens"] = token_ids.numpy()
    example_batch["Embeddings"] = embeddings.numpy()

    return example_batch


def chunk_embeddings_to_tmp_files(
    example_batch: list[np.ndarray],
    indicies: list[int],
    embeddings_dir: Union[str, Path],
    batch_size: int,
) -> None:

    """
    Create temporary .npy files, to use for creating faiss index.

    Args:
        embeddings_dir: Directory where .npy files will be placed.
        batch_size: The batch_size with which the huggingface dataset's
            "map" method will be called.
    Returns:
        None.
    """
    data = np.vstack(example_batch["Embeddings"])
    file_num = indicies[0] % batch_size
    filename = Path(embeddings_dir) / f"{file_num}.npy"
    np.save(str(filename), data)

    return example_batch


def index_embeddings(
    *,
    embeddings_folder: Union[str, Path],
    max_index_memory_usage="100m",
    current_memory_available="1G",
    **index_kwargs,
) -> faiss.Index:

    """
    Build index from temporary .npy files of embeddings.

    Args:
        embeddings_folder: Temporary folder with .npy files.
        index_infos_file: File to save index info to.
        max_index_memory_usage: Maximum amount of memory index can use.
        current_memory_available: Current memory available.
    Returns:
        Faiss index.
    """
    index, index_info = build_index(
        embeddings=str(embeddings_folder),
        save_on_disk=False,
        max_index_memory_usage=max_index_memory_usage,
        current_memory_available=current_memory_available,
        should_be_memory_mappable=True,
        use_gpu=torch.cuda.is_available(),
    )
    return index, index_info
