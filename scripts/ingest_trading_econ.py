import dotenv
import os
import pandas as pd
import numpy as np
import nasdaqdatalink
import zipfile

pd.options.mode.chained_assignment = None

# Connect to API.
dotenv.load_dotenv()
NASDAQ_KEY = os.getenv("NASDAQ_KEY")
nasdaqdatalink.ApiConfig.api_key = NASDAQ_KEY

# Declare constants.
DATABASE_CODE = "SGE"
RAW_DIR = os.path.join("data", "raw", DATABASE_CODE)
ZIP_FILE = os.path.join(RAW_DIR, "SGE.zip")
PROCESSED_DIR = os.path.join("data", "processed")
FIELD_MAPPING = {"name": "Description", "units": "Units", "source": "Source"}

# Create database directories.
if not os.path.isdir(RAW_DIR):
    os.mkdir(RAW_DIR)

if not os.path.isdir(PROCESSED_DIR):
    os.mkdir(PROCESSED_DIR)

# Get Trading Economics metadata.
codes = pd.read_csv("https://static.quandl.com/coverage/SGE_codes.csv")

# Bulk download Trading Economics data.
nasdaqdatalink.Database(DATABASE_CODE).bulk_download_to_file(ZIP_FILE)
with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
    csv_file = os.path.join(RAW_DIR, zip_ref.namelist()[0])
    zip_ref.extractall(RAW_DIR)
    os.remove(ZIP_FILE)

# Process bulk data into intermediate representation:

# Read data into dataframe.
values = pd.read_csv(csv_file, names=["Code", "Date", "Values"])
values[values.columns[0]] = values[values.columns[0]].map(
    lambda x: DATABASE_CODE + "/" + x
)

# Split dataframe by series code.
values = values.groupby("Code")
code_dfs = [values.get_group(x) for x in values.groups]

# Process individual series.
for df in code_dfs:
    # Create series subdirectory.
    sge_code = df.iat[0, 0]
    series_dir = os.path.join(PROCESSED_DIR, sge_code.replace("/", ""))
    if not os.path.isdir(series_dir):
        os.mkdir(series_dir)

    # Create data file.
    data = df[["Date", "Values"]]
    data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d")
    data["Date"] = data["Date"].dt.strftime("%Y%m%d")
    pd.to_numeric(data["Date"], errors="coerce").fillna(0).astype(np.int64)
    start_date = data["Date"].tolist()[0]
    end_date = data["Date"].tolist()[-1]
    data.to_csv(os.path.join(series_dir, "data.csv"), index=False)

    # Create metadata file.
    metadata = codes.loc[codes["code"] == sge_code]
    metadata = metadata[list(FIELD_MAPPING.keys())]
    metadata.rename(columns=FIELD_MAPPING, inplace=True)
    metadata["Start_Date"] = start_date
    metadata["End_Date"] = end_date
    metadata["Publisher"] = DATABASE_CODE
    metadata.to_csv(os.path.join(series_dir, "metadata.csv"), index=False)

# Save Trading Economics metadata file.
codes.to_csv(os.path.join(RAW_DIR, "SGE_metadata.csv"), index=False)
