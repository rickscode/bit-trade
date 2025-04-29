import pandas as pd
import vectorbt as vbt

def run_vectorbt_backtest(csv_path="data/BTCUSDT_15m.csv"):
    """
    Breakout strategy backtest using vectorbt with 15-minute intervals.
    """
    # 1. Load data
    df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")
    price = df['close']

    # 2. Breakout strategy logic
    lookback = 50  # Lookback period for the breakout
    range_high = price.rolling(lookback).max().shift(1)  # Highest high over lookback period
    range_low = price.rolling(lookback).min().shift(1)   # Lowest low over lookback period

    entries = price > range_high  # Buy signal
    exits = price < range_low     # Sell signal

    # 3. Build portfolio with stop-loss and take-profit
    pf = vbt.Portfolio.from_signals(
        close=price,
        entries=entries,
        exits=exits,
        sl_stop=0.05,  # 5% stop-loss
        tp_stop=0.005,  # 0.5% take-profit
        init_cash=10000.0,
        fees=0.001,  # 0.1% fees
        slippage=0.001,  # 0.1% slippage
        freq='15min'  # Using 15-minute intervals
    )

    # 4. Print stats
    print(pf.stats())

    # 5. Plot results
    pf.plot().show()

if __name__ == "__main__":
    run_vectorbt_backtest()
