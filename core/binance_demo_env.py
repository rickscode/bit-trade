#!/usr/bin/env python3
"""
Binance Demo Trading Environment
Real trading environment using Binance Testnet for paper trading with real market data
"""

import os
import time
import json
import asyncio
import websockets
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np
from binance.client import Client
from binance.enums import *
import gymnasium as gym
from gymnasium import spaces
from dotenv import load_dotenv

try:
    from .enhanced_logger import logger
except ImportError:
    from enhanced_logger import logger

load_dotenv()

@dataclass
class Trade:
    """Represents a single trade"""
    id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    timestamp: datetime
    order_id: str
    status: str = 'FILLED'

@dataclass
class Position:
    """Represents current position"""
    symbol: str
    quantity: float
    avg_price: float
    unrealized_pnl: float
    realized_pnl: float

class BinanceDemoEnv(gym.Env):
    """
    Binance Demo Trading Environment using Testnet
    Provides real market data with simulated trading for RL training
    """
    
    def __init__(self, 
                 symbols: List[str] = ["BTCUSDT"],
                 initial_balance: float = 10000.0,
                 testnet: bool = True,
                 risk_limit: float = 0.02):  # 2% risk per trade
        
        super().__init__()
        
        # Environment settings
        self.symbols = symbols
        self.current_symbol = symbols[0]
        self.initial_balance = initial_balance
        self.risk_limit = risk_limit
        self.testnet = testnet
        
        # Initialize Binance client
        self._init_binance_client()
        
        # Portfolio state
        self.balance = initial_balance
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.portfolio_value = initial_balance
        
        # Market data
        self.current_prices: Dict[str, float] = {}
        self.price_history: Dict[str, List[float]] = {symbol: [] for symbol in symbols}
        self.market_data: Dict[str, pd.DataFrame] = {}
        
        # RL Environment spaces
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(10,),  # price features + portfolio state
            dtype=np.float32
        )
        
        # WebSocket connection for real-time data
        self.ws_thread = None
        self.running = False
        
        # Episode tracking
        self.episode_start_time = None
        self.episode_duration = timedelta(hours=24)  # 24-hour episodes
        self.step_count = 0
        self.max_steps = 1440  # 1 minute steps for 24 hours
        
        logger.info(f"🏗️ Binance Demo Environment initialized")
        logger.info(f"   Symbols: {symbols}")
        logger.info(f"   Initial Balance: ${initial_balance:,.2f}")
        logger.info(f"   Testnet: {'Enabled' if testnet else 'Disabled'}")
    
    def _init_binance_client(self):
        """Initialize Binance client for testnet or live market data"""
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        
        if not api_key or not api_secret:
            logger.warning("⚠️ Binance API credentials not found. Using market data only mode.")
            self.client = None
            return
        
        if self.testnet:
            # Testnet for safe paper trading
            self.client = Client(api_key, api_secret, testnet=True)
            logger.info("🧪 Connected to Binance Testnet")
        else:
            # Live client for market data only (no trading)
            self.client = Client(api_key, api_secret)
            logger.info("📊 Connected to Binance for market data")
    
    def reset(self) -> np.ndarray:
        """Reset environment for new episode"""
        # Reset portfolio
        self.balance = self.initial_balance
        self.positions = {}
        self.trades = []
        self.portfolio_value = self.initial_balance
        
        # Reset episode tracking
        self.episode_start_time = datetime.now()
        self.step_count = 0
        
        # Get initial market data
        self._fetch_market_data()
        
        # Start real-time data feed
        self._start_realtime_data()
        
        observation = self._get_observation()
        logger.info(f"🔄 Environment reset. Starting episode at {self.episode_start_time}")
        
        return observation
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment"""
        self.step_count += 1
        
        # Get current market state
        observation = self._get_observation()
        current_price = self.current_prices.get(self.current_symbol, 0)
        
        # Execute action
        reward = 0.0
        info = {"action": action, "price": current_price}
        
        if action == 1:  # Buy
            reward = self._execute_buy()
            info["action_taken"] = "BUY"
        elif action == 2:  # Sell
            reward = self._execute_sell()
            info["action_taken"] = "SELL"
        else:  # Hold
            info["action_taken"] = "HOLD"
        
        # Update portfolio value
        self._update_portfolio_value()
        
        # Check if episode is done
        done = self._is_episode_done()
        
        # Calculate additional reward components
        portfolio_return = (self.portfolio_value - self.initial_balance) / self.initial_balance
        info["portfolio_value"] = self.portfolio_value
        info["portfolio_return"] = portfolio_return
        info["total_trades"] = len(self.trades)
        
        # Add return-based reward
        reward += portfolio_return * 100  # Scale for RL
        
        if done:
            self._stop_realtime_data()
            final_return = portfolio_return * 100
            logger.info(f"📊 Episode completed. Final return: {final_return:.2f}%")
        
        return observation, reward, done, info
    
    def _execute_buy(self) -> float:
        """Execute buy order"""
        current_price = self.current_prices.get(self.current_symbol, 0)
        if current_price == 0:
            return 0
        
        # Calculate position size based on risk limit
        risk_amount = self.balance * self.risk_limit
        quantity = risk_amount / current_price
        
        if quantity * current_price > self.balance:
            return 0  # Insufficient balance
        
        # Execute simulated trade
        trade_id = f"trade_{len(self.trades)}_{int(time.time())}"
        trade = Trade(
            id=trade_id,
            symbol=self.current_symbol,
            side="BUY",
            quantity=quantity,
            price=current_price,
            timestamp=datetime.now(),
            order_id=trade_id
        )
        
        # Update balance and positions
        self.balance -= quantity * current_price
        self.trades.append(trade)
        
        # Update position
        if self.current_symbol in self.positions:
            pos = self.positions[self.current_symbol]
            total_quantity = pos.quantity + quantity
            avg_price = ((pos.quantity * pos.avg_price) + (quantity * current_price)) / total_quantity
            pos.quantity = total_quantity
            pos.avg_price = avg_price
        else:
            self.positions[self.current_symbol] = Position(
                symbol=self.current_symbol,
                quantity=quantity,
                avg_price=current_price,
                unrealized_pnl=0,
                realized_pnl=0
            )
        
        logger.info(f"💰 BUY: {quantity:.6f} {self.current_symbol} @ ${current_price:.2f}")
        return 1.0  # Positive reward for action execution
    
    def _execute_sell(self) -> float:
        """Execute sell order"""
        current_price = self.current_prices.get(self.current_symbol, 0)
        if current_price == 0 or self.current_symbol not in self.positions:
            return 0
        
        position = self.positions[self.current_symbol]
        if position.quantity <= 0:
            return 0
        
        # Sell entire position for simplicity
        quantity = position.quantity
        
        # Execute simulated trade
        trade_id = f"trade_{len(self.trades)}_{int(time.time())}"
        trade = Trade(
            id=trade_id,
            symbol=self.current_symbol,
            side="SELL",
            quantity=quantity,
            price=current_price,
            timestamp=datetime.now(),
            order_id=trade_id
        )
        
        # Update balance and positions
        sale_amount = quantity * current_price
        self.balance += sale_amount
        self.trades.append(trade)
        
        # Calculate realized PnL
        pnl = (current_price - position.avg_price) * quantity
        position.realized_pnl += pnl
        position.quantity = 0
        
        logger.info(f"💸 SELL: {quantity:.6f} {self.current_symbol} @ ${current_price:.2f} (PnL: ${pnl:.2f})")
        
        # Return reward based on PnL
        return pnl / self.initial_balance * 100  # Normalized PnL as reward
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation state"""
        current_price = self.current_prices.get(self.current_symbol, 0)
        
        # Price features (technical indicators)
        price_features = self._calculate_technical_indicators()
        
        # Portfolio features
        portfolio_features = [
            self.balance / self.initial_balance,  # Normalized balance
            self.portfolio_value / self.initial_balance,  # Normalized portfolio value
            len(self.positions),  # Number of positions
            len(self.trades),  # Number of trades
        ]
        
        # Combine features
        observation = np.array(price_features + portfolio_features, dtype=np.float32)
        
        # Ensure observation matches expected shape
        if len(observation) < 10:
            # Pad with zeros if needed
            observation = np.pad(observation, (0, 10 - len(observation)))
        
        return observation[:10]  # Ensure exactly 10 features
    
    def _calculate_technical_indicators(self) -> List[float]:
        """Calculate technical indicators from price history"""
        symbol_prices = self.price_history.get(self.current_symbol, [])
        
        if len(symbol_prices) < 20:
            return [0.0] * 6  # Return zeros if insufficient data
        
        prices = np.array(symbol_prices[-20:])  # Last 20 prices
        
        # Simple technical indicators
        current_price = prices[-1]
        ma5 = np.mean(prices[-5:])
        ma10 = np.mean(prices[-10:])
        ma20 = np.mean(prices)
        
        # Price momentum
        momentum = (current_price - prices[-5]) / prices[-5] if prices[-5] > 0 else 0
        
        # Volatility (standard deviation)
        volatility = np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0
        
        return [
            current_price / 50000,  # Normalized price (assuming ~$50k for BTC)
            (current_price - ma5) / current_price if current_price > 0 else 0,
            (current_price - ma10) / current_price if current_price > 0 else 0,
            (current_price - ma20) / current_price if current_price > 0 else 0,
            momentum,
            volatility
        ]
    
    def _update_portfolio_value(self):
        """Update total portfolio value"""
        total_value = self.balance
        
        for symbol, position in self.positions.items():
            if position.quantity > 0:
                current_price = self.current_prices.get(symbol, 0)
                position_value = position.quantity * current_price
                position.unrealized_pnl = (current_price - position.avg_price) * position.quantity
                total_value += position_value
        
        self.portfolio_value = total_value
    
    def _is_episode_done(self) -> bool:
        """Check if episode should end"""
        # End if max steps reached
        if self.step_count >= self.max_steps:
            return True
        
        # End if portfolio lost too much (risk management)
        if self.portfolio_value < self.initial_balance * 0.5:  # 50% drawdown limit
            logger.warning("⚠️ Episode ended due to excessive drawdown")
            return True
        
        # End if episode duration exceeded
        if self.episode_start_time:
            elapsed = datetime.now() - self.episode_start_time
            if elapsed > self.episode_duration:
                return True
        
        return False
    
    def _fetch_market_data(self):
        """Fetch initial market data"""
        if not self.client:
            # Generate dummy data if no client
            for symbol in self.symbols:
                self.current_prices[symbol] = 50000.0  # Dummy BTC price
                self.price_history[symbol] = [50000.0] * 20
            return
        
        try:
            for symbol in self.symbols:
                # Get current price
                ticker = self.client.get_symbol_ticker(symbol=symbol)
                price = float(ticker['price'])
                self.current_prices[symbol] = price
                
                # Get recent price history
                klines = self.client.get_klines(
                    symbol=symbol,
                    interval=Client.KLINE_INTERVAL_1MINUTE,
                    limit=100
                )
                
                prices = [float(kline[4]) for kline in klines]  # Close prices
                self.price_history[symbol] = prices
                
                logger.info(f"📊 {symbol}: ${price:,.2f}")
                
        except Exception as e:
            logger.error(f"❌ Error fetching market data: {e}")
    
    def _start_realtime_data(self):
        """Start real-time WebSocket data feed"""
        if not self.client:
            return
        
        self.running = True
        self.ws_thread = threading.Thread(target=self._run_websocket)
        self.ws_thread.daemon = True
        self.ws_thread.start()
        
        logger.info("🔄 Real-time data feed started")
    
    def _stop_realtime_data(self):
        """Stop real-time data feed"""
        self.running = False
        if self.ws_thread:
            self.ws_thread.join(timeout=1)
        
        logger.info("⏹️ Real-time data feed stopped")
    
    def _run_websocket(self):
        """Run WebSocket connection for real-time data"""
        # Simplified - would implement actual WebSocket connection
        while self.running:
            try:
                # Simulate price updates
                for symbol in self.symbols:
                    if symbol in self.current_prices:
                        # Add small random movement
                        current = self.current_prices[symbol]
                        change = np.random.normal(0, current * 0.001)  # 0.1% volatility
                        new_price = max(current + change, 0.01)
                        
                        self.current_prices[symbol] = new_price
                        self.price_history[symbol].append(new_price)
                        
                        # Keep only last 100 prices
                        if len(self.price_history[symbol]) > 100:
                            self.price_history[symbol] = self.price_history[symbol][-100:]
                
                time.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"❌ WebSocket error: {e}")
                time.sleep(5)
    
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary"""
        summary = {
            "balance": self.balance,
            "portfolio_value": self.portfolio_value,
            "total_return": (self.portfolio_value - self.initial_balance) / self.initial_balance,
            "total_trades": len(self.trades),
            "positions": {},
            "recent_trades": self.trades[-5:]  # Last 5 trades
        }
        
        for symbol, position in self.positions.items():
            if position.quantity > 0:
                current_price = self.current_prices.get(symbol, 0)
                summary["positions"][symbol] = {
                    "quantity": position.quantity,
                    "avg_price": position.avg_price,
                    "current_price": current_price,
                    "market_value": position.quantity * current_price,
                    "unrealized_pnl": position.unrealized_pnl,
                    "realized_pnl": position.realized_pnl
                }
        
        return summary
    
    def close(self):
        """Close environment and cleanup"""
        self._stop_realtime_data()
        logger.info("🏁 Binance Demo Environment closed")

if __name__ == "__main__":
    # Test the environment
    env = BinanceDemoEnv(symbols=["BTCUSDT"], initial_balance=10000.0)
    
    print("🧪 Testing Binance Demo Environment")
    
    # Reset environment
    obs = env.reset()
    print(f"Initial observation: {obs}")
    
    # Run a few steps
    for i in range(10):
        action = np.random.choice([0, 1, 2])  # Random action
        obs, reward, done, info = env.step(action)
        
        print(f"Step {i+1}: Action={action}, Reward={reward:.4f}, Done={done}")
        print(f"   Info: {info}")
        
        if done:
            break
    
    # Print portfolio summary
    summary = env.get_portfolio_summary()
    print(f"\\n📊 Portfolio Summary:")
    print(f"   Balance: ${summary['balance']:,.2f}")
    print(f"   Portfolio Value: ${summary['portfolio_value']:,.2f}")
    print(f"   Total Return: {summary['total_return']:.2%}")
    print(f"   Total Trades: {summary['total_trades']}")
    
    env.close()
    print("✅ Test completed")