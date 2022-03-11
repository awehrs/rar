import dotenv
import os
import pandas as pd
import nasdaqdatalink
import zipfile

# Connect to API.
dotenv.load_dotenv()
NASDAQ_KEY = os.getenv("NASDAQ_KEY")
nasdaqdatalink.ApiConfig.api_key = NASDAQ_KEY

# Declare constants.
DATABASE_CODE = "SGE"
RAW_DIR = os.path.join("data", "raw", DATABASE_CODE)
ZIP_FILE = os.path.join(RAW_DIR, "SGE.zip")
PROCESSED_DIR = os.path.join("data", "processed", DATABASE_CODE)

# Create database directories.
if not os.path.isdir(RAW_DIR):
    os.mkdir(RAW_DIR)

if not os.path.isdir(PROCESSED_DIR):
    os.mkdir(PROCESSED_DIR)

# Get Trading Economics metadata
codes = pd.read_csv("https://static.quandl.com/coverage/SGE_codes.csv")

# Bulk download Trading Economics Data.
nasdaqdatalink.Database(DATABASE_CODE).bulk_download_to_file(ZIP_FILE)
with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
    csv_file = os.path.join(RAW_DIR, zip_ref.namelist()[0])
    zip_ref.extractall(path=RAW_DIR)
    os.remove(ZIP_FILE)

# Process bulk data into intermediate representation.
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

    # Create metadata file.
    metadata = codes.loc[codes["code"] == sge_code]
    metadata.to_csv(os.path.join(series_dir, "metadata.csv"), index=False)
    
    # Create data file.
    data = df[["Date", "Values"]]
    data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d")
    data["Date"] = data["Date"].dt.strftime("%m-%d-%Y")
    data.to_csv(os.path.join(series_dir, "data.csv"), index=False)
    data = None  # F(df) [slice off code, format dates]
   