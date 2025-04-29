# config.py

import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Binance Keys (pulled from .env)
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

# Default config variables
SYMBOL = "BTCUSDT"        # Trading pair
INTERVAL = "1m"           # 1-minute candles
LOOKBACK = "1 day ago UTC"
