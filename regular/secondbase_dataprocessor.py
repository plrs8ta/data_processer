# %%
import pandas as pd
import pyarrow as pa
# %%
PARQUET_FILE_PATH = r"C:\Users\EDY\Desktop\PROJECT\PI\cloudsail\database\aggregated_database\aggregated_data_60s(test_51turbines_ori).parquet"

# %%
df = pd.read_parquet(PARQUET_FILE_PATH, engine='pyarrow')

# %%
