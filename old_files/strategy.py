# strategy.py

def simple_strategy(df, profit_target=0.005):
    """
    Example strategy:
      - Buy when RSI < 30 (oversold)
      - Sell when price increases by profit_target from the buy price
    """
    position = False
    buy_price = 0.0
    trades = []

    for i in range(len(df)):
        rsi_value = df['RSI'].iloc[i]
        current_price = df['close'].iloc[i]
        timestamp = df.index[i]

        # Entry Condition
        if not position and rsi_value < 30:
            buy_price = current_price
            position = True
            trades.append(f"[{timestamp}] BUY at {buy_price:.2f}")

        # Exit Condition
        if position and current_price >= buy_price * (1 + profit_target):
            sell_price = current_price
            position = False
            profit = sell_price - buy_price
            trades.append(f"[{timestamp}] SELL at {sell_price:.2f}, Profit: {profit:.2f}")

    return trades
