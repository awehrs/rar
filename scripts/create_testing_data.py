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

# Randomly select NUM_SERIES folders in REAL_DIR.

series_folders = [Path(f.path) for f in os.scandir(REAL_DIR) if f.is_dir()]

selected_folders = random.sample(series_folders, NUM_SERIES)

# Copy them to MOCK_DIR.

for f in selected_folders:
    src = f
    dst = MOCK_DIR / f.stem
    shutil.copytree(src, dst)
