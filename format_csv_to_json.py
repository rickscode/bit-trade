import os
import pandas as pd
import json

# Define the directory containing the CSV files
DATA_DIR = "data/"
OUTPUT_DIR = "formatted_data/"

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def format_csv_to_json(input_path, output_path):
    """
    Convert a CSV file to a JSON file.
    """
    df = pd.read_csv(input_path, parse_dates=["timestamp"])
    # Convert timestamp column to string
    df["timestamp"] = df["timestamp"].astype(str)
    # Select necessary columns and convert to list of dictionaries
    data = df[["timestamp", "open", "high", "low", "close", "volume"]].to_dict(orient="records")
    # Save as JSON
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Formatted: {input_path} -> {output_path}")

def process_all_files():
    """
    Process all CSV files in the data directory and convert them to JSON.
    """
    for file_name in os.listdir(DATA_DIR):
        if file_name.endswith(".csv"):
            input_path = os.path.join(DATA_DIR, file_name)
            output_path = os.path.join(OUTPUT_DIR, file_name.replace(".csv", ".json"))
            format_csv_to_json(input_path, output_path)

if __name__ == "__main__":
    process_all_files()
