import pandas as pd

# Load the data
df = pd.read_csv("../data/BTCUSDT_1m.csv", parse_dates=["timestamp"], index_col="timestamp")

# Preview the data
print(df.head())  # Check the first 5 rows
print(df.tail())  # Check the last 5 rows
print(f"Total rows: {len(df)}")  # Check the number of rows
print(f"Columns: {df.columns.tolist()}")  # Check the column names

# Check for missing or NaN values
print(df.isnull().sum())