#!/usr/bin/env python3
"""
Real-time Web Dashboard for Bit-Trade
Provides live monitoring and visualization of the autonomous trading agent
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
import pickle
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Optional

# Set page configuration
st.set_page_config(
    page_title="Bit-Trade Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DashboardData:
    """Manages dashboard data and updates"""
    
    def __init__(self):
        self.portfolio_history = []
        self.trades_history = []
        self.model_performance = {}
        self.agent_memory = None
        self.last_update = datetime.now()
        
    def load_trading_results(self) -> Optional[Dict]:
        """Load latest trading results from agent memory with real portfolio data"""
        try:
            # First try to load from agent memory (contains real trading data)
            agent_memory_data = self.load_agent_memory()
            if agent_memory_data and 'experience_buffer' in agent_memory_data:
                experience_buffer = agent_memory_data['experience_buffer']
                
                # Extract portfolio values and returns from agent experiences
                portfolio_values = []
                rewards = []
                
                for experience in experience_buffer:
                    # Portfolio value is in market_context
                    if 'market_context' in experience:
                        market_ctx = experience['market_context']
                        if 'portfolio_value' in market_ctx:
                            portfolio_values.append(market_ctx['portfolio_value'])
                    
                    # Rewards are directly available
                    if 'reward' in experience:
                        rewards.append(experience['reward'])
                
                # Use stored metrics from agent memory if available
                avg_return = agent_memory_data.get('avg_return', 0.0)
                win_rate = agent_memory_data.get('win_rate', 0.0)
                
                # If we have portfolio values, calculate returns
                if portfolio_values and len(portfolio_values) > 1:
                    initial_value = portfolio_values[0]
                    episode_returns = [(pv - initial_value) / initial_value for pv in portfolio_values]
                    
                    # Update metrics if we have better data
                    if len(episode_returns) > 0:
                        avg_return = np.mean(episode_returns)
                        win_rate = len([r for r in episode_returns if r > 0]) / len(episode_returns)
                else:
                    # Use rewards as returns if no portfolio values
                    episode_returns = rewards
                
                # Create results structure with real data
                results = {
                    "session": {
                        "total_return": episode_returns[-1] if episode_returns else 0.0,
                        "win_rate": win_rate,
                        "total_steps": len(experience_buffer),
                        "status": "completed"
                    },
                    "results": {
                        "episode_returns": episode_returns,
                        "avg_return": avg_return,
                        "win_rate": win_rate,
                        "portfolio_values": portfolio_values,
                        "total_return": episode_returns[-1] if episode_returns else 0.0
                    }
                }
                return results
            
            # Fallback to JSON file if agent memory unavailable
            if os.path.exists("outputs/demo_trading_results.json"):
                with open("outputs/demo_trading_results.json", 'r') as f:
                    return json.load(f)
                    
        except Exception as e:
            st.error(f"Error loading trading results: {e}")
        return None
    
    def load_model_performance(self) -> Optional[Dict]:
        """Load model performance data from multiple sources"""
        try:
            # Try to load from model performance file first
            if os.path.exists("outputs/model_performance.json"):
                with open("outputs/model_performance.json", 'r') as f:
                    return json.load(f)
            
            # Extract model performance from AI reasoning history
            if os.path.exists("outputs/ai_reasoning_history.json"):
                with open("outputs/ai_reasoning_history.json", 'r') as f:
                    reasoning_data = json.load(f)
                
                # Analyze model usage and decision quality
                model_stats = {
                    "versatile": {"decisions": 0, "avg_confidence": 0, "mcts_sims": [], "successful_trades": 0},
                    "analytical": {"decisions": 0, "avg_confidence": 0, "mcts_sims": [], "successful_trades": 0},
                    "maverick": {"decisions": 0, "avg_confidence": 0, "mcts_sims": [], "successful_trades": 0},
                    "scout": {"decisions": 0, "avg_confidence": 0, "mcts_sims": [], "successful_trades": 0},
                    "diverse": {"decisions": 0, "avg_confidence": 0, "mcts_sims": [], "successful_trades": 0}
                }
                
                total_decisions = len(reasoning_data)
                
                for decision in reasoning_data:
                    ai_reasoning = decision.get('ai_reasoning', {})
                    confidence = ai_reasoning.get('confidence', 0.5)
                    mcts_stats = decision.get('mcts_stats', {})
                    
                    # Assign decision to a model based on confidence and MCTS patterns
                    if confidence > 0.8:
                        model = "analytical"  # High confidence = analytical model
                    elif confidence < 0.4:
                        model = "maverick"    # Low confidence = maverick model  
                    elif mcts_stats.get('total_simulations', 0) > 5:
                        model = "scout"       # High simulations = scout model
                    elif len(ai_reasoning.get('primary_factors', [])) > 3:
                        model = "diverse"     # Many factors = diverse model
                    else:
                        model = "versatile"   # Default = versatile model
                    
                    if model in model_stats:
                        model_stats[model]["decisions"] += 1
                        model_stats[model]["avg_confidence"] += confidence
                        model_stats[model]["mcts_sims"].append(mcts_stats.get('total_simulations', 0))
                        
                        # Count as successful if decision was executed (has market context)
                        if decision.get('market_context'):
                            model_stats[model]["successful_trades"] += 1
                
                # Calculate final metrics
                model_performance = {}
                for model, stats in model_stats.items():
                    if stats["decisions"] > 0:
                        avg_conf = stats["avg_confidence"] / stats["decisions"]
                        avg_sims = np.mean(stats["mcts_sims"]) if stats["mcts_sims"] else 0
                        success_rate = stats["successful_trades"] / stats["decisions"] if stats["decisions"] > 0 else 0
                        
                        model_performance[model] = {
                            "decisions_made": stats["decisions"],
                            "success_rate": success_rate,
                            "avg_confidence": avg_conf,
                            "avg_mcts_sims": avg_sims,
                            "usage_percentage": (stats["decisions"] / total_decisions) * 100 if total_decisions > 0 else 0,
                            "strategies_generated": stats["successful_trades"],
                            "avg_return": (avg_conf - 0.5) * 0.1,  # Approximate return based on confidence
                            "best_strategy_return": avg_conf * 0.05  # Approximate best return
                        }
                    else:
                        model_performance[model] = {
                            "decisions_made": 0,
                            "success_rate": 0,
                            "avg_confidence": 0,
                            "avg_mcts_sims": 0,
                            "usage_percentage": 0,
                            "strategies_generated": 0,
                            "avg_return": 0,
                            "best_strategy_return": 0
                        }
                
                # Add overall stats
                model_performance["overall_stats"] = {
                    "total_decisions": total_decisions,
                    "total_successful": sum(stats["successful_trades"] for stats in model_stats.values()),
                    "overall_success_rate": sum(stats["successful_trades"] for stats in model_stats.values()) / total_decisions if total_decisions > 0 else 0
                }
                
                return model_performance
                
        except Exception as e:
            st.error(f"Error loading model performance: {e}")
        return None
    
    def load_agent_memory(self) -> Optional[Dict]:
        """Load agent memory data"""
        try:
            if os.path.exists("outputs/agent_memory.pkl"):
                with open("outputs/agent_memory.pkl", 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            st.error(f"Error loading agent memory: {e}")
        return None
    
    def load_portfolio_state(self) -> Optional[Dict]:
        """Load current portfolio state with comprehensive real account data"""
        try:
            # Try to load from portfolio state file first
            if os.path.exists("outputs/portfolio_state.json"):
                with open("outputs/portfolio_state.json", 'r') as f:
                    return json.load(f)
            
            # Extract from agent memory with full portfolio context
            agent_memory = self.load_agent_memory()
            if agent_memory and 'experience_buffer' in agent_memory:
                experiences = agent_memory['experience_buffer']
                if experiences:
                    # Get latest experience for portfolio state
                    latest_exp = experiences[-1]
                    if 'market_context' in latest_exp:
                        market_ctx = latest_exp['market_context']
                        
                        # Extract comprehensive portfolio data
                        portfolio_value = market_ctx.get('portfolio_value', 0)
                        balance = market_ctx.get('balance', 0)
                        
                        # Get detailed position and account info
                        portfolio_ctx = market_ctx.get('portfolio_context', {})
                        account_overview = portfolio_ctx.get('account_overview', {})
                        positions_data = portfolio_ctx.get('positions', {})
                        
                        # Calculate total returns
                        initial_balance = 117492.49  # Known starting balance
                        total_pnl = portfolio_value - initial_balance
                        total_return_pct = (total_pnl / initial_balance) if initial_balance > 0 else 0
                        
                        return {
                            'portfolio_value': portfolio_value,
                            'balance': balance,
                            'positions': positions_data,
                            'total_return': total_return_pct,
                            'total_pnl': total_pnl,
                            'initial_balance': initial_balance,
                            'current_price': market_ctx.get('current_price', 0),
                            'timestamp': 'Live from Binance Testnet',
                            'account_overview': account_overview,
                            'trading_performance': portfolio_ctx.get('trading_performance', {}),
                            'risk_metrics': portfolio_ctx.get('risk_metrics', {})
                        }
        except Exception as e:
            st.error(f"Error loading portfolio state: {e}")
        return None
    
    def load_ai_reasoning(self) -> Optional[List]:
        """Load AI reasoning history"""
        try:
            if os.path.exists("outputs/ai_reasoning_history.json"):
                with open("outputs/ai_reasoning_history.json", 'r') as f:
                    return json.load(f)
        except Exception as e:
            st.error(f"Error loading AI reasoning: {e}")
        return None
    
    def extract_trading_history(self) -> List[Dict]:
        """Extract trading history from agent memory"""
        trades = []
        try:
            agent_memory = self.load_agent_memory()
            if agent_memory and 'experience_buffer' in agent_memory:
                for i, exp in enumerate(agent_memory['experience_buffer']):
                    if 'market_context' in exp:
                        ctx = exp['market_context']
                        portfolio_ctx = ctx.get('portfolio_context', {})
                        trading_perf = portfolio_ctx.get('trading_performance', {})
                        
                        # Check if this experience contains a trade
                        if 'last_trade' in trading_perf and trading_perf['last_trade']:
                            trade = trading_perf['last_trade']
                            trades.append({
                                'experience_id': i,
                                'timestamp': trade.get('timestamp'),
                                'symbol': trade.get('symbol', 'BTCUSDT'),
                                'side': trade.get('side'),
                                'quantity': trade.get('quantity', 0),
                                'price': trade.get('price', 0),
                                'order_id': trade.get('order_id'),
                                'portfolio_value': portfolio_ctx.get('account_overview', {}).get('portfolio_value', 0),
                                'action': exp.get('action_probs', {}).get(exp.get('reward', 0), 'UNKNOWN'),
                                'reward': exp.get('reward', 0),
                                'market_price': ctx.get('current_price', 0)
                            })
        except Exception as e:
            st.error(f"Error extracting trading history: {e}")
        return trades

def calculate_learning_progression(agent_memory: Dict, trading_results: Dict) -> Dict:
    """Calculate learning progression and confidence metrics"""
    if not agent_memory or not trading_results:
        return {}
    
    try:
        experience_buffer = agent_memory.get("experience_buffer", [])
        episode_returns = trading_results.get("episode_returns", [])
        
        # Calculate progression metrics
        total_experiences = len(experience_buffer)
        recent_experiences = experience_buffer[-10:] if len(experience_buffer) > 10 else experience_buffer
        early_experiences = experience_buffer[:10] if len(experience_buffer) > 10 else []
        
        # Extract portfolio performance instead of RL rewards
        portfolio_returns = []
        for exp in experience_buffer:
            if 'market_context' in exp and 'portfolio_value' in exp['market_context']:
                portfolio_returns.append(exp['market_context']['portfolio_value'])
        
        # Calculate actual returns from portfolio values
        actual_returns = []
        if len(portfolio_returns) > 1:
            initial_value = portfolio_returns[0]
            actual_returns = [(pv - initial_value) / initial_value for pv in portfolio_returns]
        
        # Learning trajectory based on real portfolio performance
        learning_metrics = {
            'total_experiences': total_experiences,
            'learning_phase': 'Initial' if total_experiences < 10 else 'Growing' if total_experiences < 50 else 'Experienced',
            'confidence_level': min(total_experiences / 100.0, 1.0),  # 0-1 scale based on experience
            'recent_performance': np.mean(actual_returns[-10:]) if len(actual_returns) > 10 else np.mean(actual_returns) if actual_returns else 0,
            'overall_performance': np.mean(actual_returns) if actual_returns else 0,
            'improvement_rate': 0,
            'decision_consistency': 0,
            'risk_management': 0
        }
        
        # Calculate improvement rate
        if early_experiences and recent_experiences:
            early_avg = np.mean([exp.get('reward', 0) for exp in early_experiences])
            recent_avg = learning_metrics['recent_performance']
            if abs(early_avg) > 1e-8:
                learning_metrics['improvement_rate'] = ((recent_avg - early_avg) / abs(early_avg)) * 100
        
        # Decision consistency (based on action distribution)
        actions = [exp.get('action', 0) for exp in experience_buffer if 'action' in exp]
        if actions:
            action_entropy = -sum(p * np.log(p + 1e-8) for p in np.bincount(actions) / len(actions) if p > 0)
            learning_metrics['decision_consistency'] = max(0, 1 - action_entropy / np.log(3))  # Normalize to 0-1
        
        # Risk management score (based on portfolio volatility)
        if len(episode_returns) > 1:
            returns_std = np.std(episode_returns)
            returns_mean = np.mean(episode_returns)
            if returns_std > 0:
                sharpe_like = abs(returns_mean) / returns_std
                learning_metrics['risk_management'] = min(sharpe_like / 2.0, 1.0)  # Normalize to 0-1
        
        return learning_metrics
        
    except Exception as e:
        st.error(f"Error calculating learning progression: {e}")
        return {}

def create_learning_journey_chart(learning_metrics: Dict, experience_buffer: List) -> go.Figure:
    """Create learning journey visualization"""
    fig = go.Figure()
    
    if not experience_buffer:
        fig.add_annotation(text="No learning data available", 
                          xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    # Extract rewards over time
    rewards = [exp.get('reward', 0) for exp in experience_buffer]
    episodes = list(range(1, len(rewards) + 1))
    
    # Calculate rolling average
    window = min(5, len(rewards))
    if len(rewards) >= window:
        rolling_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        rolling_episodes = list(range(window, len(rewards) + 1))
    else:
        rolling_avg = rewards
        rolling_episodes = episodes
    
    # Individual episode rewards
    fig.add_trace(go.Scatter(
        x=episodes, y=rewards,
        mode='markers',
        name='Episode Rewards',
        marker=dict(color='lightblue', size=6),
        opacity=0.6
    ))
    
    # Rolling average trend
    fig.add_trace(go.Scatter(
        x=rolling_episodes, y=rolling_avg,
        mode='lines',
        name=f'Trend ({window}-ep avg)',
        line=dict(color='red', width=3)
    ))
    
    fig.update_layout(
        title="Learning Journey - Reward Progression",
        xaxis_title="Episode",
        yaxis_title="Reward",
        height=400
    )
    
    return fig

def create_confidence_gauge(confidence_level: float) -> go.Figure:
    """Create confidence level gauge"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence_level * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "AI Confidence Level"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkgreen"},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "green"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    
    fig.update_layout(height=300)
    return fig

def create_portfolio_chart(trading_results: Dict) -> go.Figure:
    """Create portfolio performance chart from real trading data"""
    if not trading_results:
        fig = go.Figure()
        fig.add_annotation(text="No portfolio data available", 
                          xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    # Extract portfolio values and returns from results
    results = trading_results.get("results", {})
    portfolio_values = results.get("portfolio_values", [])
    episode_returns = results.get("episode_returns", [])
    
    if not portfolio_values and not episode_returns:
        fig = go.Figure()
        fig.add_annotation(text="No trading data available", 
                          xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    # Use portfolio values if available, otherwise use episode returns
    if portfolio_values:
        episodes = list(range(1, len(portfolio_values) + 1))
        # Calculate returns from portfolio values
        initial_value = portfolio_values[0] if portfolio_values else 100000
        portfolio_returns = [(pv - initial_value) / initial_value for pv in portfolio_values]
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Portfolio Value ($)', 'Portfolio Returns (%)'),
            vertical_spacing=0.1,
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )
        
        # Portfolio value line (absolute dollars)
        fig.add_trace(
            go.Scatter(
                x=episodes,
                y=portfolio_values,
                mode='lines+markers',
                name='Portfolio Value ($)',
                line=dict(color='#00ff88', width=3),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # Portfolio returns percentage (scale for better visibility of small changes)
        scaled_returns = [r * 10000 for r in portfolio_returns]  # Show in basis points (0.01%)
        colors = ['#ff4444' if r < 0 else '#00ff88' for r in portfolio_returns]
        fig.add_trace(
            go.Scatter(
                x=episodes,
                y=scaled_returns,
                mode='lines+markers',
                name='Return (basis points)',
                line=dict(color='#ffa500', width=2),
                marker=dict(size=4, color=colors)
            ),
            row=2, col=1
        )
        
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Return (basis points)", row=2, col=1)
        
    else:
        # Fallback to episode returns if no portfolio values
        episodes = list(range(1, len(episode_returns) + 1))
        cumulative_returns = np.cumprod([1 + r for r in episode_returns]) - 1
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Cumulative Returns', 'Episode Returns'),
            vertical_spacing=0.1,
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )
        
        fig.add_trace(
            go.Scatter(
                x=episodes,
                y=[r * 100 for r in cumulative_returns],
                mode='lines+markers',
                name='Cumulative Return %',
                line=dict(color='#00ff88', width=3),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        colors = ['#ff4444' if r < 0 else '#00ff88' for r in episode_returns]
        fig.add_trace(
            go.Bar(
                x=episodes,
                y=[r * 100 for r in episode_returns],
                name='Episode Return %',
                marker_color=colors
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        title_text="Trading Performance",
        showlegend=False,
        height=600,
        plot_bgcolor='rgba(0,0,0,0.1)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    fig.update_xaxes(title_text="Trading Steps", row=2, col=1)
    
    return fig

def create_model_performance_chart(model_data: Dict) -> go.Figure:
    """Create model performance comparison chart"""
    if not model_data:
        fig = go.Figure()
        fig.add_annotation(text="No model performance data available", 
                          xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    models = list(model_data.keys())
    success_rates = [model_data[model].get("success_rate", 0) * 100 for model in models]
    avg_returns = [model_data[model].get("avg_return", 0) * 100 for model in models]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Success Rate %', 'Average Return %'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    fig.add_trace(
        go.Bar(x=models, y=success_rates, name="Success Rate %",
               marker_color='#4CAF50'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=models, y=avg_returns, name="Avg Return %",
               marker_color='#2196F3'),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text="AI Model Performance Comparison",
        showlegend=False,
        height=400,
        plot_bgcolor='rgba(0,0,0,0.1)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_experience_metrics(agent_memory: Dict) -> Dict:
    """Extract key metrics from agent memory"""
    if not agent_memory:
        return {}
    
    experience_buffer = agent_memory.get("experience_buffer", [])
    
    metrics = {
        "total_experiences": len(experience_buffer),
        "training_episodes": agent_memory.get("training_episodes", 0),
        "avg_return": agent_memory.get("avg_return", 0),
        "win_rate": agent_memory.get("win_rate", 0),
        "recent_decisions": len([exp for exp in experience_buffer[-100:] if exp.get("reward", 0) > 0]) if experience_buffer else 0
    }
    
    return metrics

def main():
    """Main dashboard function"""
    st.title("🚀 Bit-Trade Autonomous Trading Dashboard")
    st.markdown("Real-time monitoring and analysis of AI trading agent performance")
    
    # Initialize dashboard data
    dashboard_data = DashboardData()
    
    # Sidebar controls
    st.sidebar.header("Dashboard Controls")
    st.sidebar.checkbox("Auto-refresh (30s)", value=True, disabled=True, help="Manual refresh only for now")
    refresh_button = st.sidebar.button("Refresh Now")
    
    # Manual refresh or auto-refresh trigger
    if refresh_button:
        st.rerun()
    
    # Load all data sources
    trading_results = dashboard_data.load_trading_results()
    model_performance = dashboard_data.load_model_performance()
    agent_memory = dashboard_data.load_agent_memory()
    portfolio_state = dashboard_data.load_portfolio_state()
    
    # Real Account Overview
    st.subheader("🏦 Live Binance Testnet Account")
    
    # Get current portfolio state
    if portfolio_state:
        account_col1, account_col2, account_col3, account_col4 = st.columns(4)
        
        with account_col1:
            balance = portfolio_state.get("balance", 0)
            st.metric("💰 USDT Balance", f"${balance:,.2f}")
        
        with account_col2:
            positions = portfolio_state.get("positions", {})
            btc_qty = 0
            btc_value = 0
            for symbol, pos in positions.items():
                if "BTC" in symbol:
                    btc_qty = pos.get("quantity", 0)
                    btc_value = pos.get("position_value", 0)
            
            if btc_qty > 0:
                avg_price = btc_value / btc_qty if btc_qty > 0 else 0
                current_price = portfolio_state.get("current_price", avg_price)
                pnl = (current_price - avg_price) * btc_qty if avg_price > 0 else 0
                st.metric("₿ BTC Holdings", f"{btc_qty:.6f} BTC", f"${btc_value:,.2f}")
            else:
                st.metric("₿ BTC Holdings", "0.000000 BTC", "$0.00")
        
        with account_col3:
            total_value = portfolio_state.get("portfolio_value", balance)
            st.metric("💼 Total Portfolio", f"${total_value:,.2f}")
        
        with account_col4:
            total_pnl = portfolio_state.get("total_pnl", 0)
            total_return_pct = portfolio_state.get("total_return", 0) * 100
            st.metric("📈 Total P&L", f"${total_pnl:+.2f}", f"{total_return_pct:+.3f}%")
            
        # Add additional context row
        st.markdown("---")
        context_col1, context_col2, context_col3, context_col4 = st.columns(4)
        
        with context_col1:
            trading_perf = portfolio_state.get("trading_performance", {})
            total_trades = trading_perf.get("total_trades", 0)
            st.metric("🔄 Total Trades", total_trades)
        
        with context_col2:
            current_price = portfolio_state.get("current_price", 0)
            st.metric("📊 BTC Price", f"${current_price:,.2f}")
        
        with context_col3:
            risk_metrics = portfolio_state.get("risk_metrics", {})
            max_position = risk_metrics.get("max_position_weight", 0) * 100
            st.metric("⚖️ Max Position", f"{max_position:.2f}%")
        
        with context_col4:
            timestamp = portfolio_state.get("timestamp", "Unknown")
            st.metric("🕒 Data Source", timestamp)
    else:
        st.warning("⚠️ Unable to load current portfolio state")
        st.info("💡 Run a trading session to see real portfolio data here")
    
    st.divider()
    
    # Trading Performance Metrics
    st.subheader("📊 Trading Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if trading_results:
            # Handle both old and new data structure
            results = trading_results.get("results", trading_results)
            avg_return = results.get("avg_return", 0)
            st.metric("Avg Return", f"{avg_return:.4%}", 
                     delta=f"{avg_return:.4%}" if avg_return != 0 else None)
        else:
            st.metric("Avg Return", "No data")
    
    with col2:
        if trading_results:
            # Handle both old and new data structure  
            results = trading_results.get("results", trading_results)
            win_rate = results.get("win_rate", 0)
            st.metric("Win Rate", f"{win_rate:.1%}",
                     delta=f"{win_rate:.1%}" if win_rate != 0 else None)
        else:
            st.metric("Win Rate", "No data")
    
    with col3:
        if agent_memory:
            episodes = agent_memory.get("training_episodes", 0)
            st.metric("Training Episodes", episodes)
        else:
            st.metric("Training Episodes", "No data")
    
    with col4:
        if agent_memory:
            experiences = len(agent_memory.get("experience_buffer", []))
            st.metric("Experiences", experiences)
        else:
            st.metric("Experiences", "No data")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Portfolio", "🧠 Learning Journey", "🤖 AI Models", "🔍 Agent Memory", "🤖 AI Reasoning", "📊 Live Data"])
    
    with tab1:
        st.subheader("Portfolio Performance")
        
        if trading_results:
            # Portfolio chart
            portfolio_fig = create_portfolio_chart(trading_results)
            st.plotly_chart(portfolio_fig, use_container_width=True)
            
            # Summary statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Episode Statistics")
                returns = trading_results.get("episode_returns", [])
                if returns:
                    stats_df = pd.DataFrame({
                        "Metric": ["Best Episode", "Worst Episode", "Total Episodes", "Profitable Episodes"],
                        "Value": [f"{max(returns):.2%}", f"{min(returns):.2%}", 
                                len(returns), sum(1 for r in returns if r > 0)]
                    })
                    st.table(stats_df)
            
            with col2:
                st.subheader("Training Statistics")
                training_stats = trading_results.get("training_stats", {})
                if training_stats:
                    training_df = pd.DataFrame([
                        {"Metric": k, "Value": v} for k, v in training_stats.items()
                    ])
                    st.table(training_df)
            
            # Diagnostic Plots Section
            st.subheader("📊 Advanced Portfolio Analytics")
            
            # Check for generated diagnostic plots
            plot_files = []
            if os.path.exists("reports/plots"):
                for filename in os.listdir("reports/plots"):
                    if filename.endswith('.png'):
                        plot_files.append(os.path.join("reports/plots", filename))
            
            if plot_files:
                # Sort by modification time (newest first)
                plot_files.sort(key=os.path.getmtime, reverse=True)
                
                # Display plots
                col1, col2 = st.columns(2)
                
                portfolio_plots = [f for f in plot_files if 'portfolio_analysis' in f]
                system_plots = [f for f in plot_files if 'system_diagnostics' in f]
                learning_plots = [f for f in plot_files if 'learning_curve' in f]
                
                with col1:
                    if portfolio_plots:
                        st.subheader("Portfolio Analysis")
                        st.image(portfolio_plots[0], caption="Portfolio Performance Analysis", use_container_width=True)
                    
                    if learning_plots:
                        st.subheader("Learning Curve Analysis")  
                        st.image(learning_plots[0], caption="AI Learning Progression", use_container_width=True)
                
                with col2:
                    if system_plots:
                        st.subheader("System Diagnostics")
                        st.image(system_plots[0], caption="System Health & Data Distribution", use_container_width=True)
                
                # Refresh plots button
                if st.button("🔄 Refresh Diagnostic Plots"):
                    with st.spinner("Generating fresh diagnostic plots..."):
                        try:
                            # Run the plot generation script
                            import subprocess
                            result = subprocess.run(
                                ["python3", "generate_diagnostic_plots.py"], 
                                capture_output=True, 
                                text=True,
                                cwd="/Users/rick/Desktop/apps/bit-trade"
                            )
                            if result.returncode == 0:
                                st.success("✅ Diagnostic plots refreshed successfully!")
                                st.rerun()
                            else:
                                st.error(f"Error generating plots: {result.stderr}")
                        except Exception as e:
                            st.error(f"Error running plot generation: {e}")
            else:
                st.info("No diagnostic plots available. Click the button below to generate them.")
                
                if st.button("📊 Generate Diagnostic Plots"):
                    with st.spinner("Generating diagnostic plots..."):
                        try:
                            import subprocess
                            result = subprocess.run(
                                ["python3", "generate_diagnostic_plots.py"], 
                                capture_output=True, 
                                text=True,
                                cwd="/Users/rick/Desktop/apps/bit-trade"
                            )
                            if result.returncode == 0:
                                st.success("✅ Diagnostic plots generated successfully!")
                                st.rerun()
                            else:
                                st.error(f"Error generating plots: {result.stderr}")
                        except Exception as e:
                            st.error(f"Error running plot generation: {e}")
        else:
            st.warning("No portfolio data available. Run a trading session to see results.")
    
    with tab2:
        st.subheader("🧠 AI Learning Journey")
        
        # Calculate learning progression metrics
        learning_metrics = calculate_learning_progression(agent_memory, trading_results)
        
        if learning_metrics:
            # Learning phase and confidence indicators
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Learning Phase", learning_metrics.get('learning_phase', 'Unknown'))
            
            with col2:
                confidence = learning_metrics.get('confidence_level', 0) * 100
                st.metric("AI Confidence", f"{confidence:.1f}%", 
                         delta=f"{confidence - 50:.1f}%" if confidence != 50 else None)
            
            with col3:
                improvement = learning_metrics.get('improvement_rate', 0)
                st.metric("Improvement Rate", f"{improvement:+.1f}%",
                         delta=f"{improvement:.1f}%" if improvement != 0 else None)
            
            # Learning journey visualization
            st.subheader("Learning Progress Over Time")
            if agent_memory and agent_memory.get("experience_buffer"):
                journey_fig = create_learning_journey_chart(learning_metrics, agent_memory["experience_buffer"])
                st.plotly_chart(journey_fig, use_container_width=True)
            else:
                st.info("No learning data available. Run trading episodes to see progress.")
            
            # Confidence gauge
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("AI Confidence Level")
                confidence_fig = create_confidence_gauge(learning_metrics.get('confidence_level', 0))
                st.plotly_chart(confidence_fig, use_container_width=True)
            
            with col2:
                st.subheader("Learning Quality Metrics")
                quality_metrics = {
                    "Decision Consistency": f"{learning_metrics.get('decision_consistency', 0) * 100:.1f}%",
                    "Risk Management": f"{learning_metrics.get('risk_management', 0) * 100:.1f}%",
                    "Recent Performance": f"{learning_metrics.get('recent_performance', 0):.4f}",
                    "Overall Performance": f"{learning_metrics.get('overall_performance', 0):.4f}"
                }
                
                for metric, value in quality_metrics.items():
                    st.write(f"**{metric}:** {value}")
            
            # Live trading readiness assessment
            st.subheader("🚦 Live Trading Readiness Assessment")
            
            readiness_score = 0
            readiness_factors = []
            
            # Experience factor (0-30 points)
            experience_count = learning_metrics.get('total_experiences', 0)
            if experience_count >= 100:
                exp_score = 30
                exp_status = "✅ Excellent"
            elif experience_count >= 50:
                exp_score = 20
                exp_status = "⚠️ Good"
            elif experience_count >= 20:
                exp_score = 10
                exp_status = "⚠️ Fair"
            else:
                exp_score = 0
                exp_status = "❌ Insufficient"
            
            readiness_score += exp_score
            readiness_factors.append(f"Experience ({experience_count} episodes): {exp_status} ({exp_score}/30 pts)")
            
            # Consistency factor (0-25 points)
            consistency = learning_metrics.get('decision_consistency', 0) * 25
            if consistency >= 20:
                cons_status = "✅ Excellent"
            elif consistency >= 15:
                cons_status = "⚠️ Good"
            elif consistency >= 10:
                cons_status = "⚠️ Fair"
            else:
                cons_status = "❌ Poor"
            
            readiness_score += consistency
            readiness_factors.append(f"Decision Consistency: {cons_status} ({consistency:.1f}/25 pts)")
            
            # Performance factor (0-25 points)
            recent_perf = learning_metrics.get('recent_performance', 0)
            if recent_perf > 0.01:
                perf_score = 25
                perf_status = "✅ Excellent"
            elif recent_perf > 0:
                perf_score = 15
                perf_status = "⚠️ Good"
            elif recent_perf > -0.01:
                perf_score = 10
                perf_status = "⚠️ Fair"
            else:
                perf_score = 0
                perf_status = "❌ Poor"
            
            readiness_score += perf_score
            readiness_factors.append(f"Recent Performance: {perf_status} ({perf_score}/25 pts)")
            
            # Risk Management factor (0-20 points)
            risk_mgmt = learning_metrics.get('risk_management', 0) * 20
            if risk_mgmt >= 15:
                risk_status = "✅ Excellent"
            elif risk_mgmt >= 10:
                risk_status = "⚠️ Good"
            elif risk_mgmt >= 5:
                risk_status = "⚠️ Fair"
            else:
                risk_status = "❌ Poor"
            
            readiness_score += risk_mgmt
            readiness_factors.append(f"Risk Management: {risk_status} ({risk_mgmt:.1f}/20 pts)")
            
            # Overall readiness assessment
            if readiness_score >= 80:
                readiness_level = "🟢 READY for Live Trading"
                readiness_color = "success"
            elif readiness_score >= 60:
                readiness_level = "🟡 CAUTION - More training recommended"
                readiness_color = "warning"
            elif readiness_score >= 40:
                readiness_level = "🟠 NOT READY - Significant training needed"
                readiness_color = "warning"
            else:
                readiness_level = "🔴 NOT READY - Extensive training required"
                readiness_color = "error"
            
            # Display readiness assessment
            st.success(f"**Overall Readiness Score: {readiness_score:.1f}/100**") if readiness_score >= 80 else \
            st.warning(f"**Overall Readiness Score: {readiness_score:.1f}/100**") if readiness_score >= 40 else \
            st.error(f"**Overall Readiness Score: {readiness_score:.1f}/100**")
            
            st.write(f"**Assessment: {readiness_level}**")
            
            with st.expander("Detailed Readiness Breakdown"):
                for factor in readiness_factors:
                    st.write(f"• {factor}")
            
        else:
            st.info("No learning data available. Run trading episodes to see learning journey.")
    
    with tab3:
        st.subheader("🤖 AI Model Performance & Q-Learning Analysis")
        
        if model_performance:
            # Overview metrics
            overall_stats = model_performance.get("overall_stats", {})
            if overall_stats:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Decisions", overall_stats.get("total_decisions", 0))
                
                with col2:
                    st.metric("Successful Executions", overall_stats.get("total_successful", 0))
                
                with col3:
                    success_rate = overall_stats.get("overall_success_rate", 0) * 100
                    st.metric("Overall Success Rate", f"{success_rate:.1f}%")
            
            # Model performance comparison chart
            st.subheader("Model Usage & Performance Comparison")
            
            # Prepare data for visualization
            models = []
            usage_percentages = []
            success_rates = []
            avg_confidences = []
            decisions_made = []
            
            for model, stats in model_performance.items():
                if model != "overall_stats" and isinstance(stats, dict):
                    models.append(model.title())
                    usage_percentages.append(stats.get("usage_percentage", 0))
                    success_rates.append(stats.get("success_rate", 0) * 100)
                    avg_confidences.append(stats.get("avg_confidence", 0) * 100)
                    decisions_made.append(stats.get("decisions_made", 0))
            
            if models:
                # Create multi-metric chart
                col1, col2 = st.columns(2)
                
                with col1:
                    # Usage distribution
                    fig_usage = px.pie(
                        values=usage_percentages, 
                        names=models, 
                        title="Model Usage Distribution",
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig_usage.update_layout(height=400)
                    st.plotly_chart(fig_usage, use_container_width=True)
                
                with col2:
                    # Success rates comparison
                    fig_success = px.bar(
                        x=models, 
                        y=success_rates,
                        title="Model Success Rates",
                        color=success_rates,
                        color_continuous_scale="RdYlGn"
                    )
                    fig_success.update_layout(
                        yaxis_title="Success Rate (%)",
                        xaxis_title="AI Model",
                        height=400
                    )
                    st.plotly_chart(fig_success, use_container_width=True)
            
            # Model performance radar chart
            if len(models) > 0:
                st.subheader("Model Performance Radar Chart")
                
                # Create radar chart data
                fig_radar = go.Figure()
                
                for i, model in enumerate(models):
                    stats = model_performance.get(model.lower(), {})
                    
                    values = [
                        stats.get("success_rate", 0) * 100,
                        stats.get("avg_confidence", 0) * 100,
                        min(stats.get("usage_percentage", 0), 100),
                        min(stats.get("decisions_made", 0) / 10, 100),  # Scale decisions to 0-100
                        min(stats.get("avg_mcts_sims", 0) * 10, 100)   # Scale MCTS sims to 0-100
                    ]
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values + [values[0]],  # Close the polygon
                        theta=['Success Rate', 'Confidence', 'Usage %', 'Decisions', 'MCTS Depth'] + ['Success Rate'],
                        fill='toself',
                        name=model,
                        opacity=0.6
                    ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )
                    ),
                    height=500,
                    title="Multi-Dimensional Model Performance"
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
            
            # Detailed model statistics table
            st.subheader("Detailed Model Statistics")
            model_data = []
            for model, stats in model_performance.items():
                if model != "overall_stats" and isinstance(stats, dict):
                    model_data.append({
                        "Model": model.title(),
                        "Decisions Made": stats.get("decisions_made", 0),
                        "Success Rate": f"{stats.get('success_rate', 0):.1%}",
                        "Avg Confidence": f"{stats.get('avg_confidence', 0):.1%}",
                        "Usage %": f"{stats.get('usage_percentage', 0):.1f}%",
                        "Avg MCTS Sims": f"{stats.get('avg_mcts_sims', 0):.1f}",
                        "Strategies Generated": stats.get("strategies_generated", 0),
                        "Est. Avg Return": f"{stats.get('avg_return', 0):.2%}"
                    })
            
            if model_data:
                st.dataframe(pd.DataFrame(model_data), use_container_width=True)
            
            # Q-Learning Analysis
            st.subheader("🧠 Q-Learning Model Selection Analysis")
            
            # Model selection trends over time
            if len(models) > 0:
                st.write("**Model Selection Strategy:**")
                st.write("- **Analytical**: High confidence decisions (>80%)")
                st.write("- **Maverick**: Low confidence, experimental decisions (<40%)")  
                st.write("- **Scout**: Extensive MCTS exploration (>5 simulations)")
                st.write("- **Diverse**: Multi-factor analysis (>3 decision factors)")
                st.write("- **Versatile**: Balanced, general-purpose decisions")
                
                # Performance insights
                best_model = max(models, key=lambda m: model_performance.get(m.lower(), {}).get("success_rate", 0))
                most_used = max(models, key=lambda m: model_performance.get(m.lower(), {}).get("usage_percentage", 0))
                highest_confidence = max(models, key=lambda m: model_performance.get(m.lower(), {}).get("avg_confidence", 0))
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.success(f"**Best Performer:** {best_model}")
                
                with col2:
                    st.info(f"**Most Used:** {most_used}")
                
                with col3:
                    st.warning(f"**Highest Confidence:** {highest_confidence}")
            
        else:
            st.warning("No model performance data available. Run trading episodes to collect model performance data.")
    
    with tab4:
        st.subheader("Agent Memory & Learning")
        
        if agent_memory:
            # Experience metrics
            metrics = create_experience_metrics(agent_memory)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Learning Progress")
                progress_data = {
                    "Training Episodes": metrics.get("training_episodes", 0),
                    "Total Experiences": metrics.get("total_experiences", 0),
                    "Average Return": f"{metrics.get('avg_return', 0):.2%}",
                    "Win Rate": f"{metrics.get('win_rate', 0):.1%}"
                }
                
                for key, value in progress_data.items():
                    st.metric(key, value)
            
            with col2:
                st.subheader("Experience Distribution")
                experience_buffer = agent_memory.get("experience_buffer", [])
                
                if experience_buffer:
                    # Create experience reward distribution
                    rewards = [exp.get("reward", 0) for exp in experience_buffer[-100:]]  # Last 100 experiences
                    fig = px.histogram(x=rewards, title="Recent Experience Rewards Distribution")
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Recent experiences
            st.subheader("Recent Trading Decisions")
            if experience_buffer:
                recent_exp = experience_buffer[-10:]  # Last 10 experiences
                exp_data = []
                for i, exp in enumerate(recent_exp):
                    exp_data.append({
                        "Experience": len(experience_buffer) - len(recent_exp) + i + 1,
                        "Reward": f"{exp.get('reward', 0):.4f}",
                        "Done": "✅" if exp.get('done', False) else "➡️"
                    })
                
                st.table(pd.DataFrame(exp_data))
        else:
            st.warning("No agent memory data available.")
    
    with tab5:
        st.subheader("🤖 AI Decision Reasoning & MCTS Analysis")
        
        # Load AI reasoning data
        ai_reasoning = dashboard_data.load_ai_reasoning()
        
        if ai_reasoning and len(ai_reasoning) > 0:
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Decisions", len(ai_reasoning))
            
            with col2:
                avg_confidence = np.mean([d.get('ai_reasoning', {}).get('confidence', 0) for d in ai_reasoning])
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            
            with col3:
                avg_mcts_sims = np.mean([d.get('mcts_stats', {}).get('total_simulations', 0) for d in ai_reasoning])
                st.metric("Avg MCTS Sims", f"{avg_mcts_sims:.1f}")
            
            with col4:
                recent_decisions = ai_reasoning[-10:]
                action_counts = {}
                for d in recent_decisions:
                    action = d.get('action_name', 'UNKNOWN')
                    action_counts[action] = action_counts.get(action, 0) + 1
                most_common_action = max(action_counts, key=action_counts.get) if action_counts else "N/A"
                st.metric("Recent Trend", most_common_action)
            
            # Decision analysis over time
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Decision Distribution Analysis")
                
                # All decisions distribution
                all_decision_summary = {}
                for decision in ai_reasoning:
                    action = decision.get('action_name', 'UNKNOWN')
                    all_decision_summary[action] = all_decision_summary.get(action, 0) + 1
                
                if all_decision_summary:
                    fig = px.pie(
                        values=list(all_decision_summary.values()),
                        names=list(all_decision_summary.keys()),
                        title=f"All Decisions Distribution ({len(ai_reasoning)} total)",
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📈 Decision Confidence Over Time")
                
                confidences = []
                decision_numbers = []
                actions = []
                
                for i, decision in enumerate(ai_reasoning[-50:]):  # Last 50 decisions
                    reasoning = decision.get('ai_reasoning', {})
                    confidences.append(reasoning.get('confidence', 0.5) * 100)
                    decision_numbers.append(i + 1)
                    actions.append(decision.get('action_name', 'UNKNOWN'))
                
                if confidences:
                    fig = go.Figure()
                    
                    # Color points by action type
                    colors = {'BUY': '#2E8B57', 'SELL': '#DC143C', 'HOLD': '#4169E1'}
                    for action in set(actions):
                        action_confidences = [conf for i, conf in enumerate(confidences) if actions[i] == action]
                        action_decisions = [num for i, num in enumerate(decision_numbers) if actions[i] == action]
                        
                        fig.add_trace(go.Scatter(
                            x=action_decisions,
                            y=action_confidences,
                            mode='markers+lines',
                            name=f'{action} Decisions',
                            marker=dict(color=colors.get(action, '#888888'), size=8),
                            line=dict(color=colors.get(action, '#888888'), width=2)
                        ))
                    
                    fig.update_layout(
                        title="Decision Confidence Progression",
                        xaxis_title="Decision Number (Recent 50)",
                        yaxis_title="Confidence (%)",
                        height=400,
                        yaxis=dict(range=[0, 100])
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # MCTS Analysis Section
            st.subheader("🌳 MCTS (Monte Carlo Tree Search) Analysis")
            
            # MCTS statistics
            mcts_sims = [d.get('mcts_stats', {}).get('total_simulations', 0) for d in ai_reasoning]
            mcts_values = [d.get('mcts_stats', {}).get('root_value', 0) for d in ai_reasoning]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # MCTS simulation distribution
                fig_mcts = px.histogram(
                    x=mcts_sims, 
                    title="MCTS Simulation Count Distribution",
                    nbins=20
                )
                fig_mcts.update_layout(
                    xaxis_title="Simulations per Decision",
                    yaxis_title="Frequency",
                    height=350
                )
                st.plotly_chart(fig_mcts, use_container_width=True)
            
            with col2:
                # MCTS value progression
                fig_values = go.Figure()
                fig_values.add_trace(go.Scatter(
                    x=list(range(len(mcts_values[-30:]))),
                    y=mcts_values[-30:],
                    mode='lines+markers',
                    name='Root Value',
                    line=dict(color='purple', width=3)
                ))
                fig_values.update_layout(
                    title="MCTS Root Value Trend (Last 30)",
                    xaxis_title="Decision",
                    yaxis_title="Root Value",
                    height=350
                )
                st.plotly_chart(fig_values, use_container_width=True)
            
            # MCTS Tree Analysis for Recent Decision
            st.subheader("🔍 Deep Dive: MCTS Tree Analysis")
            
            # Select a decision to analyze
            decision_index = st.selectbox(
                "Select Decision to Analyze:",
                range(len(ai_reasoning)),
                index=len(ai_reasoning)-1,  # Default to most recent
                format_func=lambda x: f"Decision {x+1}: {ai_reasoning[x].get('action_name', 'UNKNOWN')} at {ai_reasoning[x].get('timestamp', '')[:19]}"
            )
            
            selected_decision = ai_reasoning[decision_index]
            reasoning = selected_decision.get('ai_reasoning', {})
            mcts_stats = selected_decision.get('mcts_stats', {})
            market_context = selected_decision.get('market_context', {})
            
            # Decision overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Final Decision", reasoning.get('decision', 'N/A'))
                st.metric("Confidence", f"{reasoning.get('confidence', 0):.1%}")
            
            with col2:
                st.metric("MCTS Simulations", mcts_stats.get('total_simulations', 0))
                st.metric("Root Value", f"{mcts_stats.get('root_value', 0):.4f}")
            
            with col3:
                st.metric("Current Price", f"${market_context.get('current_price', 0):,.2f}")
                st.metric("24h Change", f"{market_context.get('price_change_24h', 0):.2%}")
            
            with col4:
                portfolio_analysis = reasoning.get('portfolio_analysis', {})
                st.metric("Portfolio Value", f"${portfolio_analysis.get('current_value', 0):,.2f}")
                st.metric("Risk Level", portfolio_analysis.get('risk_level', 'N/A'))
            
            # MCTS Children Analysis
            children_stats = mcts_stats.get('children_stats', {})
            if children_stats:
                st.subheader("MCTS Action Evaluation")
                
                # Create MCTS action comparison
                actions = list(children_stats.keys())
                visits = [children_stats[action].get('visits', 0) for action in actions]
                q_values = [children_stats[action].get('q_value', 0) for action in actions]
                prior_probs = [children_stats[action].get('prior_prob', 0) * 100 for action in actions]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Action visits comparison
                    fig_visits = px.bar(
                        x=actions, 
                        y=visits,
                        title="MCTS Action Exploration (Visits)",
                        color=visits,
                        color_continuous_scale="Viridis"
                    )
                    fig_visits.update_layout(height=300)
                    st.plotly_chart(fig_visits, use_container_width=True)
                
                with col2:
                    # Prior probabilities vs Q-values
                    fig_comparison = go.Figure()
                    
                    fig_comparison.add_trace(go.Bar(
                        x=actions,
                        y=prior_probs,
                        name='Prior Probability (%)',
                        marker_color='lightblue'
                    ))
                    
                    fig_comparison.add_trace(go.Bar(
                        x=actions,
                        y=[q * 100 for q in q_values],  # Scale Q-values for visualization
                        name='Q-Value (scaled)',
                        marker_color='orange'
                    ))
                    
                    fig_comparison.update_layout(
                        title="Prior vs Learned Values",
                        height=300,
                        barmode='group'
                    )
                    st.plotly_chart(fig_comparison, use_container_width=True)
                
                # MCTS Decision Table
                mcts_df = pd.DataFrame({
                    'Action': actions,
                    'Visits': visits,
                    'Q-Value': [f"{q:.4f}" for q in q_values],
                    'Prior Prob': [f"{p:.1f}%" for p in prior_probs],
                    'Selection Score': [f"{v * q:.4f}" for v, q in zip(visits, q_values)]
                })
                
                st.write("**MCTS Action Analysis:**")
                st.dataframe(mcts_df, use_container_width=True)
            
            # Market Context Analysis
            st.subheader("📊 Market Context & Decision Factors")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Market Conditions:**")
                market_conditions = reasoning.get('market_conditions', {})
                if market_conditions:
                    for key, value in market_conditions.items():
                        st.write(f"- **{key.title()}**: {value}")
                else:
                    st.write("No market conditions recorded")
                
                # Primary factors
                primary_factors = reasoning.get('primary_factors', [])
                if primary_factors:
                    st.write("**Primary Decision Factors:**")
                    for i, factor in enumerate(primary_factors, 1):
                        st.write(f"{i}. {factor}")
                else:
                    st.write("**Primary Decision Factors:** Not explicitly listed")
            
            with col2:
                st.write("**Portfolio Analysis:**")
                if portfolio_analysis:
                    st.write(f"- **Current Value**: ${portfolio_analysis.get('current_value', 0):,.2f}")
                    st.write(f"- **Cash Allocation**: {portfolio_analysis.get('cash_allocation', 0):.1%}")
                    st.write(f"- **Total Return**: {portfolio_analysis.get('total_return', 0):.2%}")
                    st.write(f"- **Risk Level**: {portfolio_analysis.get('risk_level', 'N/A')}")
                
                # MCTS consensus
                mcts_analysis = reasoning.get('mcts_analysis', {})
                if mcts_analysis:
                    st.write("**MCTS Analysis:**")
                    st.write(f"- **Simulations**: {mcts_analysis.get('simulations_run', 0)}")
                    st.write(f"- **Consensus Strength**: {mcts_analysis.get('consensus_strength', 'N/A')}")
            
            # Decision Timeline
            st.subheader("⏰ Recent Decision Timeline")
            
            recent_decisions = ai_reasoning[-20:]  # Last 20 decisions
            timeline_data = []
            
            for i, decision in enumerate(recent_decisions):
                timeline_data.append({
                    'Decision': len(ai_reasoning) - len(recent_decisions) + i + 1,
                    'Timestamp': decision.get('timestamp', '')[:19],
                    'Action': decision.get('action_name', 'UNKNOWN'),
                    'Confidence': f"{decision.get('ai_reasoning', {}).get('confidence', 0):.1%}",
                    'MCTS Sims': decision.get('mcts_stats', {}).get('total_simulations', 0),
                    'Price': f"${decision.get('market_context', {}).get('current_price', 0):,.2f}"
                })
            
            st.dataframe(pd.DataFrame(timeline_data), use_container_width=True)
        
        else:
            st.warning("No AI reasoning data available. Run trading episodes to see AI decision analysis.")
    
    with tab6:
        st.subheader("Live System Data")
        
        # System status
        st.subheader("System Status")
        
        files_status = {
            "Trading Results": "✅" if os.path.exists("outputs/demo_trading_results.json") else "❌",
            "Model Performance": "✅" if os.path.exists("outputs/model_performance.json") else "❌", 
            "Agent Memory": "✅" if os.path.exists("outputs/agent_memory.pkl") else "❌",
            "Portfolio State": "✅" if os.path.exists("outputs/portfolio_state.json") else "❌"
        }
        
        status_df = pd.DataFrame([
            {"Component": k, "Status": v} for k, v in files_status.items()
        ])
        st.table(status_df)
        
        # Recent log entries (if available)
        st.subheader("Recent System Activity")
        if os.path.exists("logs/bit_trade.log"):
            try:
                with open("logs/bit_trade.log", 'r') as f:
                    lines = f.readlines()
                    recent_lines = lines[-20:]  # Last 20 log lines
                    
                log_text = "".join(recent_lines)
                st.text_area("Recent Logs", log_text, height=300)
            except Exception as e:
                st.error(f"Error reading logs: {e}")
        else:
            st.warning("No log file found.")
    
    # Footer
    st.markdown("---")
    st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown("🚀 Bit-Trade Autonomous Trading System")

if __name__ == "__main__":
    main()