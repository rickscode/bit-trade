# data_fetch.py
import os
import pandas as pd
from binance.client import Client
from dotenv import load_dotenv

load_dotenv()  # Reads variables from .env into the environment

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

def fetch_and_save_historical_data(
    symbol="BTCUSDT",
    interval=Client.KLINE_INTERVAL_1HOUR,
    lookback="1 year ago UTC",   # changed from "1 day ago UTC"
    csv_path="data/BTCUSDT_1h.csv"
):
    """
    Fetch historical kline data from Binance and save as CSV.
    symbol: e.g. 'BTCUSDT'
    interval: e.g. Client.KLINE_INTERVAL_1MINUTE
    lookback: e.g. '1 day ago UTC' or '7 day ago UTC'
    csv_path: file path to save CSV
    """
    # Check that keys are not None
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        print("Error: Binance API keys not set in environment variables.")
        print("Set BINANCE_API_KEY and BINANCE_API_SECRET and try again.")
        return

    client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

    print(f"Fetching {symbol} data from Binance...")
    raw_klines = client.get_historical_klines(symbol, interval, lookback)

    # Create DataFrame
    df = pd.DataFrame(raw_klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume',
        'number_of_trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    # Convert numeric columns
    float_cols = ['open', 'high', 'low', 'close', 'volume',
                  'quote_asset_volume', 'taker_buy_base', 'taker_buy_quote']
    df[float_cols] = df[float_cols].astype(float)

    df = df[['open', 'high', 'low', 'close', 'volume']]

    # Ensure data folder exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    df.to_csv(csv_path)
    print(f"Data saved to {csv_path}")

if __name__ == "__main__":
    fetch_and_save_historical_data()
