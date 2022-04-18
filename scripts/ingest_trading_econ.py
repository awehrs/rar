"""Script to bulk download and process Trading Economics data."""

import dotenv
import nasdaqdatalink
import os
import pandas as pd
from pathlib import Path
import shutil
from tqdm import tqdm
import zipfile

import tempfile

from pyparsing import col

pd.options.mode.chained_assignment = None

# Connect to API.
dotenv.load_dotenv()
NASDAQ_KEY = os.getenv("NASDAQ_KEY")
nasdaqdatalink.ApiConfig.api_key = NASDAQ_KEY

# Declare constants.
DATABASE_CODE = "SGE"
RAW_DIR = Path("data", "raw")
PROCESSED_DIR = Path("data", "processed")
FIELD_MAPPING = {"name": "Description", "units": "Units", "source": "Source"}

# Create database directories.
if RAW_DIR.is_dir():
    shutil.rmtree(RAW_DIR)
os.mkdir(RAW_DIR)

if PROCESSED_DIR.is_dir():
    shutil.rmtree(PROCESSED_DIR)
os.mkdir(PROCESSED_DIR)

# Get Trading Economics metadata.
codes = pd.read_csv("https://static.quandl.com/coverage/SGE_codes.csv")

# Bulk download Trading Economics data.
zip_file = tempfile.mktemp()
nasdaqdatalink.Database(DATABASE_CODE).bulk_download_to_file(zip_file)
with zipfile.ZipFile(zip_file, "r") as zip_ref:
    csv_file = RAW_DIR / zip_ref.namelist()[0]
    zip_ref.extractall(RAW_DIR)
    os.remove(zip_file)

## Process bulk data into intermediate representation. ##

# Read data into dataframe.
values = pd.read_csv(csv_file, names=["Code", "Date", "Values"])
values[values.columns[0]] = values[values.columns[0]].map(
    lambda x: DATABASE_CODE + "/" + x
)

# Split dataframe by series code.
values = values.groupby("Code")
code_dfs = [values.get_group(x) for x in values.groups]

# Process individual series.
for df in tqdm(code_dfs):
    # Create series path.
    series_code = df.iat[0, 0]
    series_path = Path(PROCESSED_DIR, series_code.replace("/", "")).with_suffix(".csv")

    # Create dataframe with dates and values.
    data = df[["Date", "Values"]]
    data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d")
    start_date = data["Date"].tolist()[0]
    end_date = data["Date"].tolist()[-1]
    data = data.set_index("Date").transpose().reset_index()
    data.columns.name = None
    data.drop(columns=["index"], inplace=True)

    # Add metadata to frame.
    metadata = codes.loc[codes["code"] == series_code]
    metadata = metadata[list(FIELD_MAPPING.keys())]
    metadata.rename(columns=FIELD_MAPPING, inplace=True)
    metadata.reset_index(inplace=True)
    metadata.drop(columns=["index"], inplace=True)
    metadata["Start_Date"] = start_date
    metadata["End_Date"] = end_date
    metadata["Publisher"] = DATABASE_CODE

    # Concatenate data and metadata.
    series = pd.concat([metadata, data], axis=1, join="inner")
    series.to_csv(series_path, index=False)

# Save Trading Economics metadata file.
codes.to_csv(os.path.join(RAW_DIR, "SGE_metadata.csv"), index=False)
