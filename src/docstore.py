from src.utils import embed, reset_folder_, tokenize

from contextlib import contextmanager
import functools
from inspect import signature
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable, List, Optional, Sequence, Tuple, Union

from autofaiss import build_index
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


# Class defintion.


class Docstore:
    def __init__(
        self,
        *,
        faiss_index: faiss.swigfaiss.Index,
        data_memmap: functools.partial,
        tokens_memmap: functools.partial,
        chunks_idx_memmap: functools.partial,
        config_dict: dict,
    ) -> None:

        self._faiss_index = faiss_index
        self._data_memmap = data_memmap
        self._chunks_idx_memmap = chunks_idx_memmap
        self._tokens_memmap = tokens_memmap
        self._config_dict = config_dict

    # Factory functions.

    @classmethod
    def load(
        cls,
        docstore_dir: Union[str, Path],
    ) -> object:
        """
        Load index, memory maps, and config from disk.

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
            data_path=cls.memmap_path(root_dir, "data"),
            tokens_path=cls.memmap_path(root_dir, "tokens"),
            chunks_idx_path=cls.memmap_path(root_dir, "chunks_idx"),
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

        # Fetch index and memory maps.
        init_params["faiss_index"] = faiss.read_index(_paths["index_path"])

        init_params["data_memmap"] = functools.partial(
            memmap,
            _paths["data_path"],
            dtype=cls.memmap_dtype("data"),
            shape=(_config_dict["max_data_len"], 2),
        )
        init_params["tokens_memmap"] = functools.partial(
            memmap,
            _paths["tokens_path"],
            dtype=cls.memmap_dtype("tokens"),
            shape=(_config_dict["max_series"], _config_dict["max_metadata_seq_len"]),
        )
        init_params["chunks_idx_memmap"] = functools.partial(
            memmap,
            _paths["chunks_idx_path"],
            dtype=cls.memmap_dtype("chunks_idx"),
            shape=(_config_dict["max_data_len"],),
        )

        return cls(**init_params)

    @classmethod
    def build(
        cls,
        data_dir: Union[str, Path],
        embed_dim: int = 768,
        doc_encoder_model: str = "bert-base-uncased",
        use_cls_repr: bool = False,
        max_series: int = 10_000,
        max_data_len: int = 10_000_000,
        max_metadata_seq_len: int = 100,
        max_rows_per_embedding_file: int = 500,
        metadata_fields: List[str] = [
            "Description",
            "Units",
            "Source",
            "Start_Date",
            "End_Date",
            "Publisher",
        ],
        fields_to_embed: List[str] = ["Description", "Units"],
        metadata_to_tokens_batch_size: int = 16,
        chunks_to_embeddings_batch_size: int = 16,
        **index_kwargs,
    ) -> object:
        """
        Build Docstore from local files.

        Args:
            data_dir: Local directory with data.
            embed_dim: Embedding dimension of document encoder/index.
            doc_encoder_model: Model name of document encoder.
            use_cls_repr: Whether to represent metadata embedding by
                [CLS] token. If false, mean of final latent values is
                used.
            max_series: Maximum number of series the Docstore can hold.
            max_data_len: Maximum length allowed for a time series.
            max_metadata_seq_len: Maximum number of tokens in metadata
                sequence to be embedded.
            max_rows_per_embedding_file: Maximum number of embeddings per
                temporary .npy file.
            metadata_fields: Column names of metadata files.
            fields_to_embed: The metadata fields to concatenate, embed,
                and index.
            metadata_to_tokens_batch_size: Batch size for tokenizing metadata
                fields.
            chunks_to_embeddings_batch_size: Batch size for embedding
                metadata fields.
        Returns:
            Docstore instance.
        """
        logger.info("Loading exisiting docstore from disk...")

        # Lazily construct data/metadata context managers.
        _memmap_dir = Path(tempfile.mkdtemp())

        get_metadata = functools.partial(
            memmap,
            cls.memmap_path(_memmap_dir, "metadata"),
            dtype=cls.memmap_dtype("metadata"),
            shape=(max_series, len(fields_to_embed)),
        )
        get_tokens = functools.partial(
            memmap,
            cls.memmap_path(_memmap_dir, "tokens"),
            dtype=cls.memmap_dtype("tokens"),
            shape=(max_series, max_metadata_seq_len),
        )
        get_data = functools.partial(
            memmap,
            cls.memmap_path(_memmap_dir, "data"),
            dtype=cls.memmap_dtype("data"),
            shape=(max_data_len, 2),
        )
        get_chunks_idx = functools.partial(
            memmap,
            cls.memmap_path(_memmap_dir, "chunks_idx"),
            dtype=cls.memmap_dtype("chunks_idx"),
            shape=(max_data_len,),
        )

        # Create data/metadata memory maps.
        logger.info("Memory mapping folder contents...")

        columns_to_embed = [metadata_fields.index(_) for _ in fields_to_embed]

        num_series = memory_map_folder_contents(
            folder=data_dir,
            metadata_memmap_fn=get_metadata,
            data_memmap_fn=get_data,
            tokens_memmap_fn=get_tokens,
            chunks_idx_memmap_fn=get_chunks_idx,
            doc_encoder_model=doc_encoder_model,
            columns_to_embed=columns_to_embed,
            max_metadata_seq_len=max_metadata_seq_len,
            batch_size=metadata_to_tokens_batch_size,
        )

        # Lazily construct embeddings memmap context manager.
        get_embeddings = functools.partial(
            memmap,
            cls.memmap_path(_memmap_dir, "embeddings"),
            dtype=cls.memmap_dtype("embeddings"),
            shape=(num_series, embed_dim),
        )

        # Embed desired metadata fields.
        logger.info("Embedding metadata fields...")

        embed_metadata(
            doc_encoder_model=doc_encoder_model,
            num_series=num_series,
            embeddings_memmap_fn=get_embeddings,
            tokens_memmap_fn=get_tokens,
            use_cls_repr=use_cls_repr,
            batch_size=chunks_to_embeddings_batch_size,
        )

        # Save embeddings to temporary files.
        logger.info("Preparing embeddings for indexing...")

        embeddings_tmp_folder = Path(tempfile.mkdtemp())

        chunk_embeddings_to_tmp_files(
            embeddings_memmap_fn=get_embeddings,
            embeddings_dir=embeddings_tmp_folder,
            num_series=num_series,
            max_rows_per_file=max_rows_per_embedding_file,
        )

        # Create index.
        logger.info("Building index...")

        faiss_index = index_embeddings(
            embeddings_folder=embeddings_tmp_folder,
            **index_kwargs,
        )

        shutil.rmtree(embeddings_tmp_folder)

        # Create configuration dictionary.
        config_dict = dict(memmap_dir=_memmap_dir)

        sig = signature(cls.build)
        _locals = locals()

        for param in sig.parameters.values():
            if param.name in _locals:
                config_dict[param.name] = _locals[param.name]

        # Instantiate object.
        return cls(
            faiss_index=faiss_index,
            chunks_idx_memmap=get_chunks_idx,
            data_memmap=get_data,
            tokens_memmap=get_tokens,
            config_dict=config_dict,
        )

    def save(
        self,
        directory: Union[str, Path],
    ) -> None:

        """
        Save Docstore built from files.

        Args:
            directory: Directory in which to save memory maps,
                index, and configuration dictionary.
        Returns:
            None
        """
        logging.info("Saving index, memory maps, and Docstore config...")

        # Get/create relevant paths.
        if not Path(directory).exists():
            Path.mkdir(directory, parents=True)

        if "memmap_dir" not in self._config_dict:
            raise ValueError("Can't save docstore loaded from disk")
        else:
            memmap_dir = self._config_dict["memmap_dir"]

        # Move memory maps from temporary folder to destination_dir.
        for memmap in ["data", "metadata", "tokens", "chunks_idx"]:
            shutil.move(
                self.memmap_path(memmap_dir, memmap),
                self.memmap_path(directory, memmap),
            )

        # Save index.
        index_path = self.index_path(directory)
        faiss.write_index(self._faiss_index, str(index_path))

        # Save object configuration.
        config_path = self.config_path(directory)

        with open(config_path, "w") as f:
            json.dump(self._config_dict, f, default=str)

        shutil.rmtree(memmap_dir)

    # Search / filter methods.

    def search_index(
        self,
        query_embedding: np.ndarray,
        n_neighbors: int,
        return_tokens: bool = False,
        return_embedding: bool = False,
    ):
        distances, idx = self._faiss_index.search(query_embedding, k=n_neighbors)

        with (
            self._chunks_idx_memmap(mode="r") as chunks_idx,
            self._data_memmap(mode="r") as data,
            self._tokens_memmap(mode="r") as tokens,
        ):
            print(np.where(chunks_idx == idx, data))

    # Static utility methods.

    @staticmethod
    def memmap_dtype(memmap_name: str) -> np.dtype:
        """Return datatype of relevant memory map."""

        if memmap_name == "data":
            return np.float32
        if memmap_name == "chunks_idx":
            return np.int64
        if memmap_name == "embeddings":
            return np.float32
        if memmap_name == "metadata":
            return np.object_
        if memmap_name == "tokens":
            return np.int32
        else:
            raise ValueError(
                f"memmap name: `{memmap_name}` must be one of:"
                " data, chunks_idx, embeddings, metadata, tokens."
            )

    @staticmethod
    def memmap_path(memmap_dir: Union[Path, str], memmap_name: str) -> Path:
        """
        Return canoncial file path of relevant memory map.

        Args:
            memmap_dir: Directory containing memory maps.
            memmap_name: Name of memory map.
        Returns:
            Path of memory map.
        """

        if memmap_name not in [
            "data",
            "chunks_idx",
            "embeddings",
            "metadata",
            "tokens",
        ]:
            raise ValueError(
                f"Name of memory map: `{memmap_name}` must be one of:"
                " data, chunks_idx, embeddings, metadata, tokens."
            )

        return Path(memmap_dir, memmap_name).with_suffix(".dat")

    @staticmethod
    def index_path(docstore_dir: Union[str, Path]) -> str:
        """Return path of faiss index, as a string."""
        return str(Path(docstore_dir) / "index")

    @staticmethod
    def config_path(docstore_dir: Union[str, Path]) -> Path:
        """Return path of configuration dictionary."""
        return Path(docstore_dir, "config").with_suffix(".json")


# Docstore.build() helper methods.


def memory_map_folder_contents(
    *,
    folder: str,
    metadata_memmap_fn: functools.partial,
    data_memmap_fn: functools.partial,
    tokens_memmap_fn: functools.partial,
    chunks_idx_memmap_fn: functools.partial,
    doc_encoder_model: str,
    columns_to_embed: List[int],
    max_metadata_seq_len: int,
    batch_size: int,
) -> int:

    """
    Iterate over series in data folder, and:
        - Memory map its metadata,
        - Memory map its data,
        - Create mapping between series and indices of its
        data within data memory map.

    Args:
        folder: Directory containing series subdirectories,
            each containing "data.csv" and "metadata.csv" files.
        metadata_memmap_fn: Partially constructed context manager
            for metadata memmap.
        data_memmap_fn: Partially constructed context manager for
            data memmap.
        tokens_memmap_fn: Partially constructed context manager for
            tokens memmap.
        chunks_idx_memmap_fn: Partially constructed context manager
            for memmap that maps series to their indices in the data
            memmap.
        doc_encoder_model: Name of huggingface pretrained transformer.
        columns_to_embed: List of the column indices of fields to
            embed.
        max_metadata_seq_len: Max xength of token sequence to embed.
        batch_size: Batch size to tokenize metadata.
    Returns:
        Total number of series.
    """

    total_series = 0

    paths = [f.path for f in os.scandir(folder) if f.is_dir()]

    # Memory map the metadata and data.
    with (
        metadata_memmap_fn(mode="w+") as metadata,
        data_memmap_fn(mode="w+") as data,
        chunks_idx_memmap_fn(mode="w+") as chunks_idx,
    ):
        for path in paths:
            meta = pd.read_csv(os.path.join(path, "metadata.csv"))
            vals = pd.read_csv(os.path.join(path, "data.csv")).to_numpy()

            # Memory map the required metadata fields.
            metadata[total_series] = meta.iloc[[0], columns_to_embed]

            # Memory map data and its indices.
            data_len = vals.shape[0]
            chunk_slice = slice(total_series, (total_series + data_len))
            data[chunk_slice] = vals
            chunks_idx[chunk_slice] = np.full((data_len,), total_series)

            total_series += 1

    # Batch tokenize and memory map metadata.
    with (
        tokens_memmap_fn(mode="w+") as tokens,
        metadata_memmap_fn(mode="r") as metadata,
    ):
        for row_slice in range_chunked(total_series, batch_size=batch_size):

            # Tokenize.
            text_batch = metadata[row_slice]
            text_batch = [" ".join(col) for col in text_batch]
            ids = tokenize(
                text_batch, model=doc_encoder_model, max_length=max_metadata_seq_len
            )

            # Memory map.
            tokens[row_slice] = ids.numpy()

    return total_series


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
    embeddings_dir: Union[Path, str],
    num_series: int,
    max_rows_per_file: int,
) -> None:

    """
    Use memory map of embeddings to create temporary
    .npy files, to use for creating faiss index.

    Args:
        embeddings_memmap_fn: Partially constructed context manager for
            embeddings memmap.
        embeddings_dir: Directory where .npy files will be placed.
        num_series: The total number of sequences to embed.
        max_rows_per_file: Number of rows from memmap to put .npy file.
    Returns:
        None.
    """

    with embeddings_memmap_fn(mode="r") as f:

        reset_folder_(embeddings_dir)

        for ind, dim_slice in enumerate(
            range_chunked(num_series, batch_size=max_rows_per_file)
        ):
            filename = embeddings_dir / f"{ind}.npy"
            np.save(str(filename), f[dim_slice])


def index_embeddings(
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

    index, _ = build_index(
        embeddings=str(embeddings_folder),
        save_on_disk=False,
        max_index_memory_usage=max_index_memory_usage,
        current_memory_available=current_memory_available,
        should_be_memory_mappable=True,
        use_gpu=torch.cuda.is_available(),
    )

    return index
