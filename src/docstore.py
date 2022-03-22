
from contextlib import contextmanager
from functools import partial
from turtle import shape

import numpy as np

@contextmanager
def memmap(*args, **kwargs):
    pointer = np.memmap(*args, **kwargs)
    yield pointer
    del pointer

class Docstore: 
    def __init__(
        self, 
        folder: str,
        description_memmap_path: str,
        units_memmap_path: str,
        source_memmap_path: str, 
        start_memmap_path: str,
        end_memmap_path: str, 
        values_memmap_path: str,
        dates_memmap_path: str,
        chunks_idx_memmap_path: str,
        max_series: int, 
        chunk_len: int,  
        max_chunks: int,
        max_chunks_per_series: int,       
    ) -> None:
        
        self.chunk_len = chunk_len 
        self.max_chunks = max_chunks
        self.max_series = max_series
        self.max_chunks_per_series = max_chunks_per_series

        metadata_shape = (max_series,)
        data_shape = (max_chunks, chunk_len)
        chunks_idx_shape = (max_series, max_chunks_per_series)

        # Lazily construct memmap context managers. 
        self.get_descriptions = partial(memmap, description_memmap_path, dtype=np.object_, shape=metadata_shape)
        self.get_units = partial(memmap, units_memmap_path, dtype=np.object_, shape=metadata_shape)
        self.get_source = partial(memmap, source_memmap_path, dtype=np.object_, shape=metadata_shape)
        self.get_start = partial(memmap, start_memmap_path, dtype=np.object_, shape=metadata_shape)
        self.get_end = partial(memmap, end_memmap_path, dtype=np.object, shape=metadata_shape)
        self.get_values = partial(memmap, values_memmap_path, dtype=np.float32, shape=data_shape)
        self.get_dates = partial(memmap, dates_memmap_path, dtype=np.object_, shape=data_shape)
        self.get_chunks_idx = partial(memmap, chunks_idx_memmap_path, dtype=int, shape=chunks_idx_shape)

        # Fill in the memmaps. 
        # For files in folder: 
            # Extract 


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

# Break up series data into chunks

# Map series index to chunks 

# Concatenate and embed selected fields 

# Break up embeddings into temp files, for autofaiss indexing 

# Build date array 
