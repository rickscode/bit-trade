import json
import vectorbt as vbt
import pandas as pd

def run_backtest(data_path="data/BTCUSDT_1day.csv", strategy_path="strategies/BTCUSDT_1day_strategy.json", output_path="backtest_metrics.json"):
    # Load data
    df = pd.read_csv(data_path, parse_dates=["timestamp"], index_col="timestamp")
    price = df["close"]

    # Load LLM-generated strategy
    with open(strategy_path, "r") as f:
        strategy = json.load(f)

    # Parse indicators and rules from the strategy
    # Example: Extract MA50, MA200, and RSI rules
    df["MA50"] = price.rolling(50).mean()
    df["MA200"] = price.rolling(200).mean()
    df["RSI"] = vbt.indicators.RSI.run(price).rsi

    # Apply entry and exit conditions based on LLM strategy
    entries = (df["MA50"] > df["MA200"]) & (df["RSI"] < 30)
    exits = (df["MA50"] < df["MA200"]) & (df["RSI"] > 70)

    # Backtest the strategy
    pf = vbt.Portfolio.from_signals(
        close=price,
        entries=entries,
        exits=exits,
        size=1,
        fees=0.001,  # 0.1% trading fees
        freq="1D"
    )

    # Extract key metrics
    metrics = {
        "Start": str(pf.wrapper.index[0]),
        "End": str(pf.wrapper.index[-1]),
        "Total Return [%]": pf.total_return(),
        "Sharpe Ratio": pf.sharpe_ratio(),
        "Win Rate [%]": pf.win_rate(),
        "Total Trades": pf.total_trades(),
        "Max Drawdown [%]": pf.max_drawdown(),
    }

    # Save metrics to a JSON file
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Backtesting metrics saved to {output_path}")

if __name__ == "__main__":
    run_backtest()
