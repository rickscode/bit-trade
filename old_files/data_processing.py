# data_processing.py

import pandas as pd

def process_data(raw_data):
    """
    Converts raw binance kline data into a DataFrame with:
      - Index: timestamp
      - Columns: open, high, low, close, volume
    """
    df = pd.DataFrame(raw_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume',
        'number_of_trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    # Convert the timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    # Convert relevant columns to float
    float_cols = ['open', 'high', 'low', 'close', 'volume']
    df[float_cols] = df[float_cols].astype(float)

    # Keep only the columns we need
    df = df[['open', 'high', 'low', 'close', 'volume']]

    return df
