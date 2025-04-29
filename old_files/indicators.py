# indicators.py

import talib as ta

def add_indicators(df):
    """
    Adds technical indicators (e.g., RSI) to the DataFrame in place.
    """
    df['RSI'] = ta.RSI(df['close'], timeperiod=14)
    return df
