from supabase import create_client, Client
from dotenv import load_dotenv
import json
import os

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def save_to_supabase(strategy_path, metrics_path, llm_notes="", is_successful=False):
    with open(strategy_path, "r") as f:
        strategy_data = json.load(f)

    with open(metrics_path, "r") as f:
        metrics_data = json.load(f)

    print("Strategy JSON:", strategy_data)
    print("Metrics JSON:", metrics_data)


    record = {
        "symbol": strategy_data.get("symbol"),
        "interval": strategy_data.get("timeframe"),  # match your "interval" column
        "strategy_code": strategy_data.get("code"),  # full code block
        "metrics": metrics_data,
        "llm_notes": llm_notes,
        "is_successful": is_successful
    }

    response = supabase.table("trading-strategies").insert(record).execute()

    if response.status_code == 201:
        print("✅ Strategy saved to Supabase successfully!")
    else:
        print(f"❌ Supabase save failed: {response.data}")

if __name__ == "__main__":
    save_to_supabase("strategies/BTCUSDT_1day_strategy.json", "backtest_metrics.json")
