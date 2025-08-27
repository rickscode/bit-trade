#!/usr/bin/env python3
"""
Portfolio Persistence System
Saves and loads portfolio state between trading sessions for continuous learning
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

@dataclass
class PortfolioState:
    """Portfolio state for persistence"""
    balance: float
    portfolio_value: float
    positions: Dict[str, Dict]  # symbol -> position data
    trades: List[Dict]  # recent trades
    total_return: float
    win_rate: float
    total_trades: int
    sessions_played: int
    created_at: str
    last_updated: str
    
class PortfolioPersistence:
    """Manages portfolio state persistence across sessions"""
    
    def __init__(self, persistence_file: str = "outputs/portfolio_state.json"):
        self.persistence_file = persistence_file
        self.ensure_directory_exists()
    
    def ensure_directory_exists(self):
        """Ensure the outputs directory exists"""
        os.makedirs(os.path.dirname(self.persistence_file), exist_ok=True)
    
    def save_portfolio_state(self, 
                           balance: float,
                           portfolio_value: float,
                           positions: Dict,
                           trades: List,
                           total_return: float = 0.0,
                           win_rate: float = 0.0,
                           total_trades: int = 0) -> None:
        """Save current portfolio state to file"""
        
        # Load existing state to preserve history
        existing_state = self.load_portfolio_state()
        sessions_played = existing_state.sessions_played + 1 if existing_state else 1
        created_at = existing_state.created_at if existing_state else datetime.now().isoformat()
        
        # Create new state
        state = PortfolioState(
            balance=balance,
            portfolio_value=portfolio_value,
            positions=self._serialize_positions(positions),
            trades=self._serialize_trades(trades[-10:]),  # Keep last 10 trades
            total_return=total_return,
            win_rate=win_rate,
            total_trades=total_trades,
            sessions_played=sessions_played,
            created_at=created_at,
            last_updated=datetime.now().isoformat()
        )
        
        # Save to file
        try:
            with open(self.persistence_file, 'w') as f:
                json.dump(asdict(state), f, indent=2)
            print(f"Portfolio state saved: ${balance:,.2f} (Session #{sessions_played})")
        except Exception as e:
            print(f"Error saving portfolio state: {e}")
    
    def load_portfolio_state(self) -> Optional[PortfolioState]:
        """Load portfolio state from file"""
        try:
            if os.path.exists(self.persistence_file):
                with open(self.persistence_file, 'r') as f:
                    data = json.load(f)
                return PortfolioState(**data)
        except Exception as e:
            print(f"Error loading portfolio state: {e}")
        return None
    
    def should_reset_portfolio(self, current_balance: float, min_balance: float = 100.0) -> bool:
        """Determine if portfolio should be reset (account blown)"""
        return current_balance < min_balance
    
    def get_initial_balance(self, default_balance: float = 10000.0) -> Dict[str, Any]:
        """Get initial balance for new session"""
        state = self.load_portfolio_state()
        
        if state is None:
            # First time running
            return {
                "balance": default_balance,
                "is_new_account": True,
                "session_number": 1,
                "message": "Starting fresh with new account"
            }
        
        if self.should_reset_portfolio(state.balance):
            # Account blown - reset
            return {
                "balance": default_balance,
                "is_new_account": True,
                "session_number": state.sessions_played + 1,
                "previous_balance": state.balance,
                "message": f"Account blown (${state.balance:.2f}). Starting fresh with ${default_balance:,.2f}"
            }
        else:
            # Continue with existing balance
            return {
                "balance": state.balance,
                "is_new_account": False,
                "session_number": state.sessions_played + 1,
                "message": f"Continuing with ${state.balance:,.2f} (Session #{state.sessions_played + 1})"
            }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for current account"""
        state = self.load_portfolio_state()
        if not state:
            return {"status": "new_account"}
        
        return {
            "current_balance": state.balance,
            "portfolio_value": state.portfolio_value,
            "total_return": state.total_return,
            "win_rate": state.win_rate,
            "total_trades": state.total_trades,
            "sessions_played": state.sessions_played,
            "account_age": state.created_at,
            "last_session": state.last_updated,
            "is_profitable": state.balance > 10000.0
        }
    
    def _serialize_positions(self, positions: Dict) -> Dict:
        """Convert positions to JSON-serializable format"""
        serialized = {}
        for symbol, position in positions.items():
            if hasattr(position, '__dict__'):
                serialized[symbol] = position.__dict__
            else:
                serialized[symbol] = position
        return serialized
    
    def _serialize_trades(self, trades: List) -> List[Dict]:
        """Convert trades to JSON-serializable format"""
        serialized = []
        for trade in trades:
            if hasattr(trade, '__dict__'):
                trade_dict = trade.__dict__.copy()
                # Convert datetime to string if present
                if 'timestamp' in trade_dict and hasattr(trade_dict['timestamp'], 'isoformat'):
                    trade_dict['timestamp'] = trade_dict['timestamp'].isoformat()
                serialized.append(trade_dict)
            else:
                serialized.append(trade)
        return serialized
    
    def reset_account(self, new_balance: float = 10000.0) -> None:
        """Manually reset account (for testing or user request)"""
        if os.path.exists(self.persistence_file):
            # Backup old state
            backup_file = f"{self.persistence_file}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.rename(self.persistence_file, backup_file)
            print(f"Previous account backed up to: {backup_file}")
        
        print(f"Account manually reset to ${new_balance:,.2f}")