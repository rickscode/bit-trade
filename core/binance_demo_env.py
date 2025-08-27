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
                 initial_balance: float = None,  # Will be set from API
                 testnet: bool = True,
                 risk_limit: float = 0.02,  # 2% risk per trade
                 trading_capital: float = 10000.0):  # Allocated capital for HFT rewards
        
        super().__init__()
        
        # Environment settings
        self.symbols = symbols
        self.current_symbol = symbols[0]
        self.risk_limit = risk_limit
        self.base_trading_capital = trading_capital  # Base allocation, will be dynamic
        
        # HFT Risk Management
        self.daily_pnl = 0.0  # Track daily P&L for stop loss
        self.trading_halted = False  # Daily stop loss protection
        self.active_positions = {}  # Track open positions with OCO orders
        self.daily_trades_count = 0  # Track number of trades today
        self.stop_loss_percentage = 0.30  # 30% stop loss
        self.min_take_profit_percentage = 0.02  # 2% minimum take profit
        self.testnet = testnet
        
        # Initialize Binance client first
        self._init_binance_client()
        
        # Get real account balance from API
        real_balance = self._get_real_account_balance()
        self.initial_balance = real_balance
        self.balance = real_balance
        self.portfolio_value = real_balance
        
        # Portfolio state
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        
        # Market data
        self.current_prices: Dict[str, float] = {}
        self.price_history: Dict[str, List[float]] = {symbol: [] for symbol in symbols}
        self.market_data: Dict[str, pd.DataFrame] = {}
        
        # RL Environment spaces
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(25,),  # Enhanced: price features + detailed portfolio state
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
        
        # Set dynamic trading capital allocation (10% of total balance)
        self.trading_capital = self.initial_balance * 0.10
        
        logger.info(f"🏗️ Binance Demo Environment initialized")
        logger.info(f"   Symbols: {symbols}")
        logger.info(f"   Total Balance: ${self.initial_balance:,.2f}")
        logger.info(f"   Trading Capital (10%): ${self.trading_capital:,.2f}")
        logger.info(f"   Daily Target: 15% (${self.trading_capital * 0.15:,.2f})")
        logger.info(f"   Stop Loss: 30% (${self.trading_capital * 0.30:,.2f})")
        logger.info(f"   Testnet: {'Enabled' if testnet else 'Disabled'}")
        
        # Sync with real account on startup
        if self.client:
            self._sync_account_data()
    
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
    
    def _get_real_account_balance(self) -> float:
        """Get the real USDT balance from Binance API"""
        if not self.client:
            logger.warning("No Binance client available, using default balance")
            return 10000.0  # Fallback
        
        try:
            account_info = self.client.get_account()
            for balance in account_info['balances']:
                if balance['asset'] == 'USDT':
                    usdt_balance = float(balance['free'])
                    logger.info(f"Real Testnet USDT balance: ${usdt_balance:,.2f}")
                    return usdt_balance
            
            # If no USDT found, return 0
            logger.warning("No USDT balance found in account")
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting real balance: {e}")
            return 10000.0  # Fallback
    
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
        
        # Check if trading is halted due to stop loss
        if self.trading_halted:
            reward = -5.0  # Penalty for attempting to trade when halted
            info["action_taken"] = "HALTED"
            info["halt_reason"] = "Daily stop loss triggered"
        elif action == 1:  # Buy
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
        
        # Log portfolio changes for debugging
        if hasattr(self, '_last_portfolio_value'):
            portfolio_change = self.portfolio_value - self._last_portfolio_value
            if abs(portfolio_change) > 0.01:  # Significant change
                logger.info(f"💹 Portfolio change: ${self._last_portfolio_value:,.2f} → ${self.portfolio_value:,.2f} ({portfolio_return:+.4f}%)")
        self._last_portfolio_value = self.portfolio_value
        
        # Note: Reward now comes only from trade execution P&L, not portfolio return
        # This prevents double-counting and focuses on per-trade performance
        
        if done:
            self._stop_realtime_data()
            final_return = portfolio_return * 100
            logger.info(f"📊 Episode completed. Final return: {final_return:.2f}%")
        
        # Check for completed OCO orders (take profit or stop loss fills)
        oco_reward = self._check_oco_completions()
        if oco_reward != 0:
            reward += oco_reward
            info["oco_completion"] = True
        
        return observation, reward, done, info
    
    def _apply_order_filters(self, quantity: float, price: float, symbol_info: dict) -> float:
        """Apply Binance order filters to ensure valid quantity"""
        # Apply lot size filter (minimum quantity)
        for filter_rule in symbol_info['filters']:
            if filter_rule['filterType'] == 'LOT_SIZE':
                min_qty = float(filter_rule['minQty'])
                step_size = float(filter_rule['stepSize'])
                
                # Round quantity to step size
                quantity = round(quantity / step_size) * step_size
                
                if quantity < min_qty:
                    logger.warning(f"⚠️ Quantity {quantity} below minimum {min_qty}")
                    return 0
                break
        
        # Apply notional filter (minimum order value)
        for filter_rule in symbol_info['filters']:
            if filter_rule['filterType'] == 'NOTIONAL':
                min_notional = float(filter_rule['minNotional'])
                order_value = quantity * price
                
                if order_value < min_notional:
                    logger.warning(f"⚠️ Order value ${order_value:.2f} below minimum ${min_notional:.2f}")
                    return 0
                break
        
        return quantity
    
    def _reset_daily_trading(self):
        """Reset daily trading state for new trading day"""
        self.daily_pnl = 0.0
        self.trading_halted = False
        self.daily_trades_count = 0
        logger.info("🌅 New trading day started - daily state reset")
    
    def _check_oco_completions(self) -> float:
        """Check for completed OCO orders and calculate rewards"""
        total_reward = 0.0
        completed_positions = []
        
        for symbol, position in self.active_positions.items():
            try:
                # Check OCO order status
                oco_status = self.client.get_order_list(orderListId=position['oco_order_id'])
                
                if oco_status['listStatusType'] == 'ALL_DONE':
                    # OCO completed - determine if take profit or stop loss
                    entry_price = position['entry_price']
                    quantity = position['quantity']
                    
                    # Find which order filled
                    for order in oco_status['orders']:
                        if order['status'] == 'FILLED':
                            fill_price = float(order['price'])
                            
                            # Calculate P&L
                            trade_pnl = (fill_price - entry_price) * quantity
                            pnl_percentage = trade_pnl / (entry_price * quantity)
                            
                            # Calculate reward based on P&L percentage
                            reward = min(max(pnl_percentage * 100, -30.0), 15.0)  # Cap at +15%/-30%
                            
                            # Update daily P&L tracking
                            self.daily_pnl += pnl_percentage
                            
                            # Check if stop loss hit (negative reward)
                            if reward < -25.0:  # Close to -30% stop loss
                                self.trading_halted = True
                                logger.warning(f"🛑 STOP LOSS HIT! Daily trading halted. Loss: {pnl_percentage:.2%}")
                                reward = -30.0  # Max penalty
                            
                            total_reward += reward
                            
                            logger.info(f"🎯 OCO COMPLETED: {symbol} @ ${fill_price:.2f} | P&L: {pnl_percentage:+.2%} | Reward: {reward:+.1f}")
                            break
                    
                    completed_positions.append(symbol)
                    
            except Exception as e:
                logger.error(f"❌ Error checking OCO for {symbol}: {e}")
        
        # Remove completed positions
        for symbol in completed_positions:
            del self.active_positions[symbol]
        
        return total_reward
    
    def _execute_buy(self) -> float:
        """Execute buy order with automatic OCO (stop loss + take profit)"""
        if not self.client:
            logger.warning("⚠️ No Binance client available. Cannot execute real trades.")
            return -1.0
            
        # Check if we already have an active position
        if self.current_symbol in self.active_positions:
            logger.warning(f"⚠️ Already have active position in {self.current_symbol}")
            return -0.5
            
        current_price = self.current_prices.get(self.current_symbol, 0)
        if current_price == 0:
            logger.error("❌ No current price available")
            return -1.0
        
        try:
            # Calculate position size from trading capital allocation (not total balance)
            # Use smaller position sizes to allow multiple trades per day
            position_size = self.trading_capital * 0.10  # 10% of trading capital per trade
            quantity = position_size / current_price
            
            # Get symbol info for order requirements
            exchange_info = self.client.get_exchange_info()
            symbol_info = None
            for s in exchange_info['symbols']:
                if s['symbol'] == self.current_symbol:
                    symbol_info = s
                    break
            
            if not symbol_info:
                logger.error(f"❌ Symbol {self.current_symbol} not found")
                return -1.0
            
            # Apply lot size and notional filters
            quantity = self._apply_order_filters(quantity, current_price, symbol_info)
            if quantity <= 0:
                return -0.5
            
            # Execute market buy order first
            buy_order = self.client.order_market_buy(
                symbol=self.current_symbol,
                quantity=f"{quantity:.8f}"
            )
            
            if buy_order['status'] not in ['FILLED', 'PARTIALLY_FILLED']:
                logger.error(f"❌ Buy order failed: {buy_order['status']}")
                return -1.0
            
            # Get actual fill data
            filled_qty = float(buy_order['executedQty'])
            avg_fill_price = float(buy_order['fills'][0]['price']) if buy_order['fills'] else current_price
            
            # Calculate OCO order prices
            stop_price = avg_fill_price * (1 - self.stop_loss_percentage)  # 30% stop loss
            take_profit_price = avg_fill_price * (1 + self.min_take_profit_percentage)  # 2% take profit minimum
            
            # Place OCO (One-Cancels-Other) order for risk management
            try:
                oco_order = self.client.order_oco_sell(
                    symbol=self.current_symbol,
                    quantity=f"{filled_qty:.8f}",
                    price=f"{take_profit_price:.2f}",  # Take profit limit order
                    stopPrice=f"{stop_price:.2f}",     # Stop loss trigger price  
                    stopLimitPrice=f"{stop_price * 0.99:.2f}"  # Stop limit price (1% below stop)
                )
                
                # Track active position with OCO details
                self.active_positions[self.current_symbol] = {
                    'entry_price': avg_fill_price,
                    'quantity': filled_qty,
                    'entry_order_id': buy_order['orderId'],
                    'oco_order_id': oco_order['orderListId'],
                    'stop_price': stop_price,
                    'take_profit_price': take_profit_price,
                    'timestamp': datetime.now()
                }
                
                self.daily_trades_count += 1
                
                # Small positive reward for successful entry with protection
                entry_reward = 0.5  # Fixed small reward for successful protected entry
                
                logger.info(f"🎯 PROTECTED BUY: {filled_qty:.6f} {self.current_symbol} @ ${avg_fill_price:.2f}")
                logger.info(f"   📈 Take Profit: ${take_profit_price:.2f} (+{self.min_take_profit_percentage*100:.1f}%)")
                logger.info(f"   🛑 Stop Loss: ${stop_price:.2f} (-{self.stop_loss_percentage*100:.1f}%)")
                logger.info(f"   📋 OCO Order ID: {oco_order['orderListId']}")
                
                return entry_reward
                
            except Exception as oco_error:
                # If OCO fails, immediately market sell to close position
                logger.error(f"❌ OCO order failed, closing position: {oco_error}")
                try:
                    close_order = self.client.order_market_sell(
                        symbol=self.current_symbol,
                        quantity=f"{filled_qty:.8f}"
                    )
                    logger.info(f"🚨 Emergency close executed: {close_order['orderId']}")
                except Exception as close_error:
                    logger.error(f"❌ Emergency close failed: {close_error}")
                
                return -2.0  # Penalty for failed risk management
            
        except Exception as e:
            logger.error(f"❌ Buy execution error: {e}")
            return -1.0
    
    def _execute_sell(self) -> float:
        """Handle manual sell actions (discouraged with OCO system)"""
        # With OCO system, manual sells should be rare
        # OCO orders handle exits automatically with proper risk management
        
        if self.current_symbol in self.active_positions:
            # Already have OCO protection in place - discourage manual intervention
            logger.warning(f"⚠️ Manual sell blocked: OCO orders active for {self.current_symbol}")
            logger.info(f"   Take Profit: ${self.active_positions[self.current_symbol]['take_profit_price']:.2f}")
            logger.info(f"   Stop Loss: ${self.active_positions[self.current_symbol]['stop_price']:.2f}")
            return -1.0  # Penalty for trying to override OCO system
        if not self.client:
            logger.warning("⚠️ No Binance client available. Cannot execute real trades.")
            return 0
            
        current_price = self.current_prices.get(self.current_symbol, 0)
        if current_price == 0:
            return 0
        
        try:
            # Get account info to check real asset balance
            account_info = self.client.get_account()
            asset_balance = 0
            base_asset = self.current_symbol.replace('USDT', '')  # Get base asset (BTC from BTCUSDT)
            
            for balance in account_info['balances']:
                if balance['asset'] == base_asset:
                    asset_balance = float(balance['free'])
                    break
            
            if asset_balance <= 0:
                logger.warning(f"⚠️ No {base_asset} balance to sell: {asset_balance:.8f}")
                return 0
            
            # Get symbol info for minimum order requirements
            exchange_info = self.client.get_exchange_info()
            symbol_info = None
            for s in exchange_info['symbols']:
                if s['symbol'] == self.current_symbol:
                    symbol_info = s
                    break
            
            if not symbol_info:
                logger.error(f"❌ Symbol {self.current_symbol} not found")
                return 0
            
            # Apply lot size filter and sell available balance
            quantity = asset_balance
            for filter_rule in symbol_info['filters']:
                if filter_rule['filterType'] == 'LOT_SIZE':
                    min_qty = float(filter_rule['minQty'])
                    step_size = float(filter_rule['stepSize'])
                    
                    # Round quantity to step size
                    quantity = round(quantity / step_size) * step_size
                    
                    if quantity < min_qty:
                        logger.warning(f"⚠️ Quantity {quantity} below minimum {min_qty}")
                        return 0
                    break
            
            # Apply notional filter (minimum order value)
            for filter_rule in symbol_info['filters']:
                if filter_rule['filterType'] == 'NOTIONAL':
                    min_notional = float(filter_rule['minNotional'])
                    order_value = quantity * current_price
                    
                    if order_value < min_notional:
                        logger.warning(f"⚠️ Order value ${order_value:.2f} below minimum ${min_notional:.2f}")
                        return 0
                    break
            
            # Execute real market sell order
            order = self.client.order_market_sell(
                symbol=self.current_symbol,
                quantity=f"{quantity:.8f}"
            )
            
            if order['status'] in ['FILLED', 'PARTIALLY_FILLED']:
                # Create trade record from real order
                filled_qty = float(order['executedQty'])
                avg_price = float(order['fills'][0]['price']) if order['fills'] else current_price
                
                trade = Trade(
                    id=str(order['orderId']),
                    symbol=self.current_symbol,
                    side="SELL",
                    quantity=filled_qty,
                    price=avg_price,
                    timestamp=datetime.fromtimestamp(order['transactTime'] / 1000),
                    order_id=str(order['orderId']),
                    status=order['status']
                )
                
                self.trades.append(trade)
                
                # Update positions (will be synced with real account)
                self._sync_account_data()
                
                # Calculate realized PnL (simplified - would need to track cost basis properly)
                sale_amount = filled_qty * avg_price
                
                logger.info(f"💸 REAL SELL: {filled_qty:.6f} {self.current_symbol} @ ${avg_price:.2f} = ${sale_amount:.2f} (Order ID: {order['orderId']})")
                return 2.0  # Higher reward for successful real trade
            else:
                logger.error(f"❌ Sell order failed: {order['status']}")
                return -0.5
                
        except Exception as e:
            logger.error(f"❌ Sell order failed: {e}")
            return -1.0
    
    def _get_observation(self) -> np.ndarray:
        """Get comprehensive observation state for AI decision making"""
        current_price = self.current_prices.get(self.current_symbol, 0)
        
        # 1. Market/Price Features (6 features)
        price_features = self._calculate_technical_indicators()
        
        # 2. Portfolio Financial State (5 features)
        portfolio_financial = [
            self.balance / self.initial_balance,  # Normalized available balance
            self.portfolio_value / self.initial_balance,  # Normalized total portfolio value
            (self.portfolio_value - self.initial_balance) / self.initial_balance,  # Total return %
            self.balance / self.portfolio_value if self.portfolio_value > 0 else 1.0,  # Cash allocation %
            (self.portfolio_value - self.balance) / self.portfolio_value if self.portfolio_value > 0 else 0.0  # Position allocation %
        ]
        
        # 3. Position Details (6 features)
        position_features = self._get_position_features()
        
        # 4. Trade Performance (4 features) 
        trade_performance = self._get_trade_performance_features()
        
        # 5. Risk Metrics (4 features)
        risk_features = self._get_risk_features()
        
        # Combine all features (6 + 5 + 6 + 4 + 4 = 25 features)
        all_features = price_features + portfolio_financial + position_features + trade_performance + risk_features
        
        # Ensure exactly 25 features
        observation = np.array(all_features, dtype=np.float32)
        if len(observation) < 25:
            observation = np.pad(observation, (0, 25 - len(observation)))
        
        return observation[:25]
    
    def _get_position_features(self) -> List[float]:
        """Get detailed position features"""
        current_price = self.current_prices.get(self.current_symbol, 0)
        
        if self.current_symbol not in self.positions or current_price == 0:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # No position
        
        pos = self.positions[self.current_symbol]
        position_value = pos.quantity * current_price
        unrealized_pnl = position_value - (pos.quantity * pos.avg_price)
        unrealized_pnl_pct = unrealized_pnl / (pos.quantity * pos.avg_price) if pos.quantity > 0 else 0
        
        return [
            pos.quantity,  # Position size (absolute)
            position_value / self.portfolio_value if self.portfolio_value > 0 else 0,  # Position weight %
            pos.avg_price / current_price if current_price > 0 else 1.0,  # Avg price vs current (entry quality)
            unrealized_pnl / self.initial_balance,  # Normalized unrealized P&L
            unrealized_pnl_pct,  # Unrealized P&L %
            1.0 if pos.quantity > 0 else 0.0  # Position exists (boolean)
        ]
    
    def _get_trade_performance_features(self) -> List[float]:
        """Get recent trade performance metrics"""
        if len(self.trades) == 0:
            return [0.0, 0.0, 0.0, 0.0]
        
        recent_trades = self.trades[-20:]  # Last 20 trades
        profitable_trades = 0
        total_pnl = 0.0
        
        # Calculate basic performance from recent trades
        # (This is simplified - in real implementation we'd track realized P&L properly)
        for i in range(1, len(recent_trades)):
            if recent_trades[i].side == "SELL" and i > 0:
                # Simple P&L estimation (sell price vs previous buy)
                buy_trade = None
                for j in range(i-1, -1, -1):
                    if recent_trades[j].side == "BUY":
                        buy_trade = recent_trades[j]
                        break
                
                if buy_trade:
                    pnl = (recent_trades[i].price - buy_trade.price) * recent_trades[i].quantity
                    total_pnl += pnl
                    if pnl > 0:
                        profitable_trades += 1
        
        trade_count = max(len(recent_trades), 1)
        win_rate = profitable_trades / max(len([t for t in recent_trades if t.side == "SELL"]), 1)
        
        return [
            len(self.trades) / 100.0,  # Total trade count (normalized)
            win_rate,  # Win rate from recent trades  
            total_pnl / self.initial_balance,  # Recent P&L normalized
            len(recent_trades) / 20.0  # Recent activity level
        ]
    
    def _get_risk_features(self) -> List[float]:
        """Get risk management metrics"""
        current_price = self.current_prices.get(self.current_symbol, 0)
        
        # Position risk
        position_exposure = 0.0
        if self.current_symbol in self.positions and current_price > 0:
            pos = self.positions[self.current_symbol]
            position_exposure = (pos.quantity * current_price) / self.portfolio_value
        
        # Drawdown from peak
        max_value = max(self.initial_balance, self.portfolio_value)
        current_drawdown = (max_value - self.portfolio_value) / max_value
        
        return [
            position_exposure,  # Single position exposure %
            self.balance / self.portfolio_value if self.portfolio_value > 0 else 1.0,  # Cash buffer %
            current_drawdown,  # Current drawdown from peak
            min(self.portfolio_value / self.initial_balance, 2.0)  # Account growth (capped at 2x)
        ]
    
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
                if current_price > 0:
                    position_value = position.quantity * current_price
                    position.unrealized_pnl = (current_price - position.avg_price) * position.quantity
                    total_value += position_value
                    
                    # Debug logging for significant P&L
                    if abs(position.unrealized_pnl) > 1.0:
                        pnl_pct = (current_price - position.avg_price) / position.avg_price * 100
                        logger.debug(f"📈 {symbol}: {position.quantity:.6f} @ ${position.avg_price:.2f} → ${current_price:.2f} = P&L: ${position.unrealized_pnl:+.2f} ({pnl_pct:+.2f}%)")
        
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
        """Run real Binance WebSocket connection with reconnect logic"""
        import asyncio
        import json
        import time
        
        async def connect_binance_ws():
            """Connect to Binance WebSocket streams with retry logic"""
            max_retries = 5
            base_delay = 1
            max_delay = 60
            retry_count = 0
            
            while self.running and retry_count < max_retries:
                try:
                    # Exponential backoff delay
                    if retry_count > 0:
                        delay = min(base_delay * (2 ** (retry_count - 1)), max_delay)
                        logger.info(f"⏳ Reconnecting in {delay} seconds... (attempt {retry_count + 1})")
                        await asyncio.sleep(delay)
                    
                    # Binance WebSocket URLs (from official docs)
                    if self.testnet:
                        ws_base = "wss://stream.testnet.binance.vision/ws"  # Official testnet WebSocket
                    else:
                        ws_base = "wss://stream.binance.com:9443/ws"  # Production WebSocket
                    
                    # Create stream names for all symbols (ticker data)
                    streams = []
                    for symbol in self.symbols:
                        streams.append(f"{symbol.lower()}@ticker")
                    
                    stream_string = "/".join(streams)
                    ws_url = f"{ws_base}/{stream_string}"
                    
                    logger.info(f"🔗 Connecting to Binance WebSocket: {ws_url}")
                    
                    # Connect with timeout
                    websocket = await asyncio.wait_for(
                        websockets.connect(ws_url), 
                        timeout=10
                    )
                    
                    logger.info("✅ Connected to Binance WebSocket stream")
                    retry_count = 0  # Reset retry count on successful connection
                    
                    # Connection success - start receiving messages
                    while self.running:
                        try:
                            # Receive message from WebSocket with timeout
                            message = await asyncio.wait_for(websocket.recv(), timeout=30)
                            data = json.loads(message)
                            
                            # Handle single stream or combined stream
                            if 'stream' in data:
                                # Combined stream format
                                stream_data = data['data']
                                symbol = stream_data['s']  # Symbol
                            else:
                                # Single stream format  
                                stream_data = data
                                symbol = data['s']
                            
                            # Update current price
                            current_price = float(stream_data['c'])  # Current price
                            self.current_prices[symbol] = current_price
                            
                            # Update price history
                            if symbol in self.price_history:
                                self.price_history[symbol].append(current_price)
                                
                                # Keep only last 100 prices
                                if len(self.price_history[symbol]) > 100:
                                    self.price_history[symbol] = self.price_history[symbol][-100:]
                            
                            # Update portfolio value with new prices
                            self._update_portfolio_value()
                            
                        except asyncio.TimeoutError:
                            # Send ping to keep connection alive
                            try:
                                await websocket.ping()
                                logger.debug("📡 WebSocket ping sent")
                            except Exception:
                                logger.warning("⚠️ WebSocket ping failed - connection may be lost")
                                break  # Exit to trigger reconnection
                        
                        except websockets.exceptions.ConnectionClosed:
                            logger.warning("⚠️ WebSocket connection closed by server")
                            break  # Exit to trigger reconnection
                            
                        except Exception as e:
                            logger.error(f"❌ WebSocket message error: {e}")
                            break  # Exit to trigger reconnection
                    
                    # Close websocket cleanly
                    await websocket.close()
                    
                except asyncio.TimeoutError:
                    retry_count += 1
                    logger.error(f"❌ WebSocket connection timeout (attempt {retry_count})")
                    
                except Exception as e:
                    retry_count += 1
                    logger.error(f"❌ WebSocket connection error (attempt {retry_count}): {e}")
            
            # If we reach here, either stopped or max retries exceeded
            if retry_count >= max_retries:
                logger.error("❌ Max WebSocket reconnection attempts reached, falling back to REST API")
                await self._fallback_price_updates()
            else:
                logger.info("⏹️ WebSocket connection stopped")
        
        async def _fallback_price_updates():
            """Fallback to REST API updates if WebSocket fails"""
            logger.info("📡 Using fallback REST API updates")
            
            while self.running:
                try:
                    if self.client:
                        for symbol in self.symbols:
                            ticker = self.client.get_symbol_ticker(symbol=symbol)
                            price = float(ticker['price'])
                            self.current_prices[symbol] = price
                            
                            # Initialize price history if not exists
                            if symbol not in self.price_history:
                                self.price_history[symbol] = []
                            
                            self.price_history[symbol].append(price)
                            
                            if len(self.price_history[symbol]) > 100:
                                self.price_history[symbol] = self.price_history[symbol][-100:]
                        
                        # Update portfolio value
                        self._update_portfolio_value()
                    
                    await asyncio.sleep(2)  # Update every 2 seconds via REST
                    
                except Exception as e:
                    logger.error(f"❌ REST API fallback error: {e}")
                    await asyncio.sleep(5)
        
        # Run the WebSocket connection
        try:
            asyncio.run(connect_binance_ws())
        except Exception as e:
            logger.error(f"❌ WebSocket runtime error: {e}")
            # Final fallback - use simulation
            self._run_simulation_fallback()
    
    def _run_simulation_fallback(self):
        """Final fallback to price simulation if everything else fails"""
        logger.warning("⚠️ Using price simulation as final fallback")
        
        while self.running:
            try:
                for symbol in self.symbols:
                    if symbol in self.current_prices:
                        # Add small random movement (same as before)
                        current = self.current_prices[symbol]
                        change = np.random.normal(0, current * 0.001)
                        new_price = max(current + change, 0.01)
                        
                        self.current_prices[symbol] = new_price
                        self.price_history[symbol].append(new_price)
                        
                        if len(self.price_history[symbol]) > 100:
                            self.price_history[symbol] = self.price_history[symbol][-100:]
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Simulation fallback error: {e}")
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
    
    def get_detailed_portfolio_context(self) -> Dict[str, Any]:
        """Get comprehensive portfolio context for AI decision making"""
        current_price = self.current_prices.get(self.current_symbol, 0)
        
        # Portfolio overview
        context = {
            "account_overview": {
                "balance": self.balance,
                "portfolio_value": self.portfolio_value,
                "total_return": (self.portfolio_value - self.initial_balance) / self.initial_balance,
                "total_return_usd": self.portfolio_value - self.initial_balance,
                "cash_allocation": self.balance / self.portfolio_value if self.portfolio_value > 0 else 1.0
            }
        }
        
        # Current positions
        positions_detail = {}
        for symbol, pos in self.positions.items():
            symbol_price = self.current_prices.get(symbol, 0)
            if symbol_price > 0 and pos.quantity > 0:
                position_value = pos.quantity * symbol_price
                unrealized_pnl = position_value - (pos.quantity * pos.avg_price)
                unrealized_pnl_pct = unrealized_pnl / (pos.quantity * pos.avg_price)
                
                positions_detail[symbol] = {
                    "quantity": pos.quantity,
                    "avg_price": pos.avg_price,
                    "current_price": symbol_price,
                    "position_value": position_value,
                    "unrealized_pnl": unrealized_pnl,
                    "unrealized_pnl_pct": unrealized_pnl_pct,
                    "weight": position_value / self.portfolio_value,
                    "days_held": 0  # Could be calculated with trade timestamps
                }
        
        context["positions"] = positions_detail
        
        # Recent trading performance
        recent_trades = self.trades[-10:] if len(self.trades) >= 10 else self.trades
        trade_summary = {
            "total_trades": len(self.trades),
            "recent_trades_count": len(recent_trades),
            "last_trade": recent_trades[-1].__dict__ if recent_trades else None
        }
        
        # Calculate simple win rate from recent sells
        sell_trades = [t for t in recent_trades if t.side == "SELL"]
        if sell_trades:
            # Simplified win calculation - could be enhanced
            wins = sum(1 for t in sell_trades if t.price > (t.price * 0.98))  # Placeholder
            trade_summary["recent_win_rate"] = wins / len(sell_trades) if sell_trades else 0
        else:
            trade_summary["recent_win_rate"] = 0
        
        context["trading_performance"] = trade_summary
        
        # Risk metrics
        context["risk_metrics"] = {
            "max_position_weight": max([pos["weight"] for pos in positions_detail.values()]) if positions_detail else 0,
            "total_position_weight": 1.0 - context["account_overview"]["cash_allocation"],
            "drawdown_from_peak": max(0, (max(self.initial_balance, self.portfolio_value) - self.portfolio_value) / max(self.initial_balance, self.portfolio_value)),
            "account_growth_multiple": self.portfolio_value / self.initial_balance
        }
        
        # Market context
        context["market_context"] = {
            "current_symbol": self.current_symbol,
            "current_price": current_price,
            "price_change_24h": 0.0,  # Could be calculated from price history
            "market_session": "active" if self.running else "closed"
        }
        
        return context
    
    def _sync_account_data(self):
        """Sync internal state with real Binance Testnet account"""
        if not self.client:
            return
        
        try:
            # Get real account info
            account_info = self.client.get_account()
            
            # Update balance from real USDT balance
            for balance in account_info['balances']:
                if balance['asset'] == 'USDT':
                    real_usdt_balance = float(balance['free'])
                    logger.info(f"💰 Real USDT balance: ${real_usdt_balance:,.2f}")
                    self.balance = real_usdt_balance
                    break
            
            # Update positions from real account, preserving cost basis
            for balance in account_info['balances']:
                asset = balance['asset']
                free_qty = float(balance['free'])
                
                # Skip USDT (that's our cash) and zero balances
                if asset == 'USDT' or free_qty <= 0:
                    continue
                
                # Find corresponding symbol (e.g., BTC -> BTCUSDT)
                symbol = f"{asset}USDT"
                if symbol in self.symbols:
                    current_price = self.current_prices.get(symbol, 0)
                    if current_price > 0:
                        # Preserve existing cost basis if position exists, otherwise calculate from trades
                        if symbol in self.positions and self.positions[symbol].quantity > 0:
                            # Update quantity but preserve avg_price for P&L calculation
                            existing_pos = self.positions[symbol]
                            self.positions[symbol] = Position(
                                symbol=symbol,
                                quantity=free_qty,
                                avg_price=existing_pos.avg_price,  # Preserve cost basis
                                unrealized_pnl=(current_price - existing_pos.avg_price) * free_qty,
                                realized_pnl=existing_pos.realized_pnl
                            )
                        else:
                            # New position - calculate avg_price from recent trades if available
                            avg_price = current_price  # Fallback
                            if len(self.trades) > 0:
                                # Find most recent buy trades for this symbol to get cost basis
                                recent_buys = [t for t in reversed(self.trades) 
                                              if t.symbol == symbol and t.side == "BUY"]
                                if recent_buys:
                                    # Use weighted average of recent buy trades
                                    total_cost = sum(t.price * t.quantity for t in recent_buys[:5])  # Last 5 buys
                                    total_qty = sum(t.quantity for t in recent_buys[:5])
                                    if total_qty > 0:
                                        avg_price = total_cost / total_qty
                            
                            self.positions[symbol] = Position(
                                symbol=symbol,
                                quantity=free_qty,
                                avg_price=avg_price,
                                unrealized_pnl=(current_price - avg_price) * free_qty,
                                realized_pnl=0
                            )
                        
                        logger.info(f"📊 Real position: {free_qty:.6f} {asset} @ ${self.positions[symbol].avg_price:.2f} (Current: ${current_price:.2f}, P&L: ${self.positions[symbol].unrealized_pnl:.2f})")
            
            # Remove positions that no longer exist in real account
            symbols_in_account = set()
            for balance in account_info['balances']:
                asset = balance['asset']
                if asset != 'USDT' and float(balance['free']) > 0:
                    symbol = f"{asset}USDT"
                    if symbol in self.symbols:
                        symbols_in_account.add(symbol)
            
            # Remove positions not in account
            self.positions = {k: v for k, v in self.positions.items() if k in symbols_in_account}
            
            # Update portfolio value
            self._update_portfolio_value()
            
            logger.info(f"🔄 Account synced. Portfolio value: ${self.portfolio_value:,.2f}")
            
        except Exception as e:
            logger.error(f"❌ Failed to sync account data: {e}")
    
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