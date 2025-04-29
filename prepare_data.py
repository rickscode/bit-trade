import pandas as pd

def prepare_data(input_path):
    # Load data
    df = pd.read_json(input_path, orient="records")

    # Calculate moving averages
    df['MA50'] = df['close'].rolling(window=50).mean()
    df['MA200'] = df['close'].rolling(window=200).mean()

    # Calculate RSI
    def calculate_rsi(data, n=14):
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=n).mean()
        avg_loss = loss.rolling(window=n).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    df['RSI'] = calculate_rsi(df['close'])

    return df

if __name__ == "__main__":
    input_path = "formatted_data/BTCUSDT_1day.json"
    output_path = "prepared_data/BTCUSDT_1day_prepared.csv"
    data = prepare_data(input_path)
    data.to_csv(output_path, index=False)
    print(f"Prepared data saved to {output_path}")
