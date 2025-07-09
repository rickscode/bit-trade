import os
import pandas as pd
import numpy as np
from binance.client import Client
from dotenv import load_dotenv
import time
from datetime import datetime, timedelta
import json

load_dotenv()

class EnhancedDataCollector:
    def __init__(self):
        self.binance_client = Client(
            os.getenv("BINANCE_API_KEY"),
            os.getenv("BINANCE_API_SECRET")
        )
        
        # Multiple symbols for diversification
        self.symbols = [
            "BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT",
            "BNBUSDT", "SOLUSDT", "MATICUSDT", "AVAXUSDT", "ATOMUSDT"
        ]
        
        # Multiple timeframes for multi-timeframe analysis
        self.timeframes = {
            "1m": Client.KLINE_INTERVAL_1MINUTE,
            "5m": Client.KLINE_INTERVAL_5MINUTE,
            "15m": Client.KLINE_INTERVAL_15MINUTE,
            "30m": Client.KLINE_INTERVAL_30MINUTE,
            "1h": Client.KLINE_INTERVAL_1HOUR,
            "4h": Client.KLINE_INTERVAL_4HOUR,
            "12h": Client.KLINE_INTERVAL_12HOUR,
            "1d": Client.KLINE_INTERVAL_1DAY,
            "3d": Client.KLINE_INTERVAL_3DAY,
            "1w": Client.KLINE_INTERVAL_1WEEK
        }
        
        # Different lookback periods for varied datasets
        self.lookback_periods = [
            "30 days ago UTC",
            "90 days ago UTC", 
            "180 days ago UTC",
            "1 year ago UTC",
            "2 years ago UTC"
        ]

    def fetch_comprehensive_data(self, symbol="BTCUSDT", timeframe="1d", lookback="1 year ago UTC"):
        """Fetch comprehensive market data with additional features"""
        try:
            print(f"Fetching {symbol} data for {timeframe} timeframe...")
            
            # Get basic OHLCV data
            klines = self.binance_client.get_historical_klines(
                symbol, 
                self.timeframes[timeframe], 
                lookback
            )
            
            # Create DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            
            # Process timestamps
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Convert to numeric
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 
                          'quote_asset_volume', 'number_of_trades', 
                          'taker_buy_base', 'taker_buy_quote']
            df[numeric_cols] = df[numeric_cols].astype(float)
            
            # Add enhanced features
            df = self.add_technical_features(df)
            df = self.add_market_features(df)
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None

    def add_technical_features(self, df):
        """Add comprehensive technical indicators"""
        # Price-based indicators
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'ma_{period}'] = df['close'].rolling(period).mean()
            df[f'ma_{period}_slope'] = df[f'ma_{period}'].diff()
        
        # Volatility indicators
        df['volatility_10'] = df['returns'].rolling(10).std()
        df['volatility_20'] = df['returns'].rolling(20).std()
        df['volatility_50'] = df['returns'].rolling(50).std()
        
        # Price channels
        df['high_20'] = df['high'].rolling(20).max()
        df['low_20'] = df['low'].rolling(20).min()
        df['price_position'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'])
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df

    def add_market_features(self, df):
        """Add market microstructure and advanced features"""
        # Volume features
        df['volume_ma_10'] = df['volume'].rolling(10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_10']
        df['volume_price_trend'] = df['volume'] * df['returns']
        
        # Price action features
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['total_range'] = df['high'] - df['low']
        df['body_ratio'] = df['body_size'] / df['total_range']
        
        # Market sentiment proxies
        df['buy_pressure'] = df['taker_buy_base'] / df['volume']
        df['sell_pressure'] = (df['volume'] - df['taker_buy_base']) / df['volume']
        df['trade_intensity'] = df['number_of_trades'] / df['volume']
        
        # Time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        
        # Gap analysis
        df['gap_up'] = (df['open'] > df['close'].shift(1)) & (df['low'] > df['close'].shift(1))
        df['gap_down'] = (df['open'] < df['close'].shift(1)) & (df['high'] < df['close'].shift(1))
        
        return df

    def collect_multi_asset_data(self, symbols=None, timeframes=None, lookback="1 year ago UTC"):
        """Collect data for multiple assets and timeframes"""
        if symbols is None:
            symbols = self.symbols[:5]  # Use top 5 symbols
        if timeframes is None:
            timeframes = ["1h", "4h", "1d"]
        
        all_data = {}
        
        for symbol in symbols:
            all_data[symbol] = {}
            for timeframe in timeframes:
                print(f"Collecting {symbol} {timeframe} data...")
                data = self.fetch_comprehensive_data(symbol, timeframe, lookback)
                if data is not None:
                    all_data[symbol][timeframe] = data
                    
                    # Save to CSV
                    filename = f"data/{symbol}_{timeframe}_enhanced.csv"
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    data.to_csv(filename)
                    
                    # Also save as JSON for LLM consumption
                    json_filename = f"formatted_data/{symbol}_{timeframe}_enhanced.json"
                    os.makedirs(os.path.dirname(json_filename), exist_ok=True)
                    data.head(500).to_json(json_filename, orient='records', date_format='iso')
                
                # Rate limiting
                time.sleep(0.5)
        
        return all_data

    def create_market_regime_data(self, df):
        """Identify different market regimes for diverse strategy testing"""
        df = df.copy()
        
        # Trend identification
        df['trend_short'] = np.where(df['ma_20'] > df['ma_50'], 1, -1)
        df['trend_long'] = np.where(df['ma_50'] > df['ma_200'], 1, -1)
        
        # Volatility regime
        vol_median = df['volatility_20'].median()
        df['vol_regime'] = np.where(df['volatility_20'] > vol_median, 'high_vol', 'low_vol')
        
        # Market regime classification
        conditions = [
            (df['trend_short'] == 1) & (df['trend_long'] == 1) & (df['vol_regime'] == 'low_vol'),
            (df['trend_short'] == 1) & (df['trend_long'] == 1) & (df['vol_regime'] == 'high_vol'),
            (df['trend_short'] == -1) & (df['trend_long'] == -1) & (df['vol_regime'] == 'low_vol'),
            (df['trend_short'] == -1) & (df['trend_long'] == -1) & (df['vol_regime'] == 'high_vol'),
            (df['trend_short'] != df['trend_long']) & (df['vol_regime'] == 'low_vol'),
            (df['trend_short'] != df['trend_long']) & (df['vol_regime'] == 'high_vol')
        ]
        
        choices = [
            'bull_stable', 'bull_volatile', 'bear_stable', 
            'bear_volatile', 'sideways_stable', 'sideways_volatile'
        ]
        
        df['market_regime'] = np.select(conditions, choices, default='undefined')
        
        return df

    def generate_synthetic_scenarios(self, base_data, num_scenarios=10):
        """Generate synthetic market scenarios for stress testing"""
        scenarios = []
        
        for i in range(num_scenarios):
            synthetic_data = base_data.copy()
            
            # Apply random transformations
            scenario_type = np.random.choice(['crash', 'bubble', 'sideways', 'recovery'])
            
            if scenario_type == 'crash':
                # Simulate market crash
                crash_magnitude = np.random.uniform(0.3, 0.7)
                synthetic_data['close'] *= (1 - crash_magnitude * np.random.random(len(synthetic_data)))
                
            elif scenario_type == 'bubble':
                # Simulate bubble formation
                bubble_magnitude = np.random.uniform(0.5, 2.0)
                synthetic_data['close'] *= (1 + bubble_magnitude * np.random.random(len(synthetic_data)))
                
            elif scenario_type == 'sideways':
                # Simulate sideways market
                noise_level = np.random.uniform(0.02, 0.05)
                synthetic_data['close'] += np.random.normal(0, noise_level, len(synthetic_data))
                
            elif scenario_type == 'recovery':
                # Simulate recovery pattern
                recovery_rate = np.random.uniform(0.1, 0.3)
                synthetic_data['close'] *= np.linspace(1, 1 + recovery_rate, len(synthetic_data))
            
            # Recalculate features
            synthetic_data = self.add_technical_features(synthetic_data)
            synthetic_data['scenario_type'] = scenario_type
            
            scenarios.append(synthetic_data)
        
        return scenarios

    def export_training_datasets(self, output_dir="training_data"):
        """Export structured datasets for strategy training"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Collect comprehensive data
        all_data = self.collect_multi_asset_data()
        
        # Create regime-specific datasets
        regime_datasets = {}
        for symbol, timeframe_data in all_data.items():
            regime_datasets[symbol] = {}
            for timeframe, data in timeframe_data.items():
                if data is not None:
                    enhanced_data = self.create_market_regime_data(data)
                    regime_datasets[symbol][timeframe] = enhanced_data
                    
                    # Export regime-specific data
                    for regime in enhanced_data['market_regime'].unique():
                        if regime != 'undefined':
                            regime_data = enhanced_data[enhanced_data['market_regime'] == regime]
                            if len(regime_data) > 50:  # Minimum data requirement
                                filename = f"{output_dir}/{symbol}_{timeframe}_{regime}.csv"
                                regime_data.to_csv(filename)
        
        # Generate synthetic scenarios
        print("Generating synthetic scenarios...")
        for symbol, timeframe_data in all_data.items():
            for timeframe, data in timeframe_data.items():
                if data is not None:
                    scenarios = self.generate_synthetic_scenarios(data)
                    for i, scenario in enumerate(scenarios):
                        filename = f"{output_dir}/{symbol}_{timeframe}_synthetic_{i}.csv"
                        scenario.to_csv(filename)
        
        print(f"Training datasets exported to {output_dir}/")
        return regime_datasets

if __name__ == "__main__":
    collector = EnhancedDataCollector()
    
    # Collect comprehensive training data
    print("Starting comprehensive data collection...")
    datasets = collector.export_training_datasets()
    
    print("✅ Enhanced data collection complete!")
    print("📊 Generated multiple datasets for:")
    print("   - Multiple assets and timeframes")
    print("   - Different market regimes")
    print("   - Synthetic stress-test scenarios")
    print("   - Comprehensive technical indicators")
    print("   - Market microstructure features")