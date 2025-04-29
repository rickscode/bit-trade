# binance_client.py

from binance.client import Client
import config

def get_binance_client():
    """
    Returns a Binance client instance (mainnet by default).
    To use Testnet, you'd add: Client(..., testnet=True).
    """
    client = Client(
        api_key=config.BINANCE_API_KEY,
        api_secret=config.BINANCE_API_SECRET
    )
    return client
