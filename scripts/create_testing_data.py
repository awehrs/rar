"""Script to create small dataset for testing suite."""

import os
from pathlib import Path
import random
import shutil

# Constants.

NUM_SERIES = 100

ROOT_DIR = Path("data")
MOCK_DIR = ROOT_DIR / "mock"
REAL_DIR = ROOT_DIR / "processed"

# Randomly select NUM_SERIES from REAL_DIR.

series_paths = [Path(entry) for entry in os.scandir(REAL_DIR)]
selected_series = random.sample(series_paths, NUM_SERIES)

# Copy them to MOCK_DIR.

if MOCK_DIR.is_dir():
    shutil.rmtree(MOCK_DIR)
os.mkdir(MOCK_DIR)

for f in selected_series:
    f = Path(f)
    src = f
    dst = MOCK_DIR / f.name
    shutil.copyfile(src, dst)
