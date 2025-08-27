#!/usr/bin/env python3
"""
AlphaZero-Style Trading Agent
Combines MCTS with neural networks for trading strategy optimization using Groq models
"""

import json
import time
import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import threading
import math

try:
    from .enhanced_logger import logger
    from .multi_llm_manager import MultiLLMManager
    from .binance_demo_env import BinanceDemoEnv
except ImportError:
    from enhanced_logger import logger
    from multi_llm_manager import MultiLLMManager
    from binance_demo_env import BinanceDemoEnv

@dataclass
class MCTSNode:
    """Monte Carlo Tree Search Node"""
    state: Dict[str, Any]
    parent: Optional['MCTSNode'] = None
    children: Dict[int, 'MCTSNode'] = None
    visits: int = 0
    value_sum: float = 0.0
    prior_prob: float = 0.0
    action: Optional[int] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = {}
    
    @property
    def q_value(self) -> float:
        """Average value of this node"""
        return self.value_sum / self.visits if self.visits > 0 else 0.0
    
    def ucb_score(self, c_puct: float = 1.0) -> float:
        """Upper Confidence Bound score for node selection"""
        if self.visits == 0:
            return float('inf')
        
        exploration = c_puct * self.prior_prob * math.sqrt(self.parent.visits) / (1 + self.visits)
        return self.q_value + exploration
    
    def select_child(self, c_puct: float = 1.0) -> 'MCTSNode':
        """Select best child based on UCB score"""
        return max(self.children.values(), key=lambda child: child.ucb_score(c_puct))
    
    def expand(self, action_probs: Dict[int, float], env_state: Dict):
        """Expand node with new children"""
        for action, prob in action_probs.items():
            if action not in self.children:
                self.children[action] = MCTSNode(
                    state=env_state,
                    parent=self,
                    prior_prob=prob,
                    action=action
                )
    
    def backup(self, value: float):
        """Backup value through the tree"""
        self.visits += 1
        self.value_sum += value
        if self.parent:
            self.parent.backup(-value)  # Flip value for adversarial perspective

@dataclass
class TradingExperience:
    """Experience tuple for training"""
    state: np.ndarray
    action_probs: Dict[int, float]
    reward: float
    next_state: np.ndarray
    done: bool
    market_context: Dict[str, Any]

class AlphaZeroTradingAgent:
    """
    AlphaZero-style agent for autonomous trading
    Combines MCTS with LLM-based policy and value estimation
    """
    
    def __init__(self, 
                 llm_manager: MultiLLMManager,
                 mcts_simulations: int = 50,
                 c_puct: float = 1.0,
                 temperature: float = 1.0,
                 experience_buffer_size: int = 10000):
        
        self.llm_manager = llm_manager
        self.mcts_simulations = mcts_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        
        # Cache for LLM responses during MCTS to prevent repeated calls
        self.action_prob_cache = {}
        
        # Experience replay for training
        self.experience_buffer: List[TradingExperience] = []
        self.max_buffer_size = experience_buffer_size
        
        # Performance tracking
        self.training_episodes = 0
        self.win_rate = 0.0
        self.avg_return = 0.0
        self.model_performance = defaultdict(list)
        
        # AlphaZero components
        self.root_node: Optional[MCTSNode] = None
        self.game_history: List[Dict] = []
        
        # Model ensemble for policy and value estimation (removed scout due to rate limits)
        self.policy_models = ["analytical", "diverse"]  # Strategic thinking
        self.value_models = ["versatile", "analytical"]  # Risk assessment
        self.creative_models = ["maverick", "diverse"]  # Exploration
        
        # Persistent memory file
        self.memory_file = "outputs/agent_memory.pkl"
        
        # Load existing experience if available
        self._load_experience_buffer()
        
        logger.info("🧠 AlphaZero Trading Agent initialized")
        logger.info(f"   MCTS Simulations: {mcts_simulations}")
        logger.info(f"   C-PUCT: {c_puct}")
        logger.info(f"   Temperature: {temperature}")
    
    def select_action(self, env: BinanceDemoEnv, observation: np.ndarray) -> int:
        """
        Select action using MCTS with LLM-based policy
        
        Args:
            env: Trading environment
            observation: Current market observation
            
        Returns:
            Action (0: Hold, 1: Buy, 2: Sell)
        """
        # Store environment reference for context access
        self.current_env = env
        
        # Clear cache at start of new action to prevent stale data
        self.action_prob_cache.clear()
        
        # Convert observation to market context
        market_context = self._observation_to_context(env, observation)
        
        # Initialize root node for MCTS
        self.root_node = MCTSNode(state=market_context)
        
        # Run MCTS simulations
        for _ in range(self.mcts_simulations):
            self._run_mcts_simulation(env, observation)
        
        # Select action based on visit counts
        action = self._select_final_action()
        
        # Generate AI reasoning explanation (simplified for performance)
        reasoning = {
            "decision": ["HOLD", "BUY", "SELL"][action],
            "confidence": 0.8,
            "simplified": True
        }
        
        # Store decision in game history with AI reasoning
        decision_data = {
            "observation": observation.copy(),
            "market_context": market_context,
            "action": action,
            "action_name": ["HOLD", "BUY", "SELL"][action],
            "timestamp": datetime.now().isoformat(),
            "mcts_stats": self._get_mcts_stats(),
            "ai_reasoning": reasoning
        }
        
        self.game_history.append(decision_data)
        
        # Save reasoning data for dashboard
        self._save_decision_reasoning(decision_data)
        
        return action
    
    def _run_mcts_simulation(self, env: BinanceDemoEnv, observation: np.ndarray):
        """Run a single MCTS simulation"""
        node = self.root_node
        path = []
        
        # Selection: Navigate to leaf node
        while node.children and all(child.visits > 0 for child in node.children.values()):
            node = node.select_child(self.c_puct)
            path.append(node)
        
        # Expansion: Add new children if not terminal
        if not self._is_terminal_state(node.state):
            action_probs = self._get_action_probabilities(node.state)
            node.expand(action_probs, node.state)
            
            # Select a child for simulation
            if node.children:
                action = random.choices(
                    list(node.children.keys()),
                    weights=[child.prior_prob for child in node.children.values()]
                )[0]
                node = node.children[action]
                path.append(node)
        
        # Simulation: Evaluate leaf node
        value = self._evaluate_position(node.state)
        
        # Backpropagation: Update all nodes in path
        for node_in_path in reversed(path):
            node_in_path.backup(value)
            value = -value  # Alternate perspective
    
    def _get_action_probabilities(self, observation) -> Dict[int, float]:
        """
        Use LLM ensemble to estimate action probabilities (with caching)
        
        Args:
            observation: Current market observation
            
        Returns:
            Dictionary of action probabilities
        """
        # Create cache key from observation
        obs_key = tuple(observation.tolist()) if hasattr(observation, 'tolist') else str(observation)
        
        # Check cache first to avoid repeated LLM calls during MCTS
        if obs_key in self.action_prob_cache:
            return self.action_prob_cache[obs_key]
        
        # Get comprehensive market context from environment
        if hasattr(self, 'current_env') and self.current_env:
            market_context = {
                'portfolio_context': self.current_env.get_detailed_portfolio_context(),
                'current_prices': self.current_env.current_prices,
                'observation': observation
            }
        else:
            # Fallback context if no environment available
            market_context = {
                'portfolio_context': {
                    'account_overview': {},
                    'positions': {},
                    'trading_performance': {},
                    'risk_metrics': {}
                },
                'current_prices': {},
                'observation': observation
            }
        
        # Generate policy prompt
        policy_prompt = self._create_policy_prompt(market_context)
        
        # Get predictions from policy models
        model_predictions = []
        available_policy_models = [m for m in self.policy_models if self.llm_manager.available_models.get(m, False)]
        
        # If no policy models available, use any available model as fallback
        if not available_policy_models:
            available_policy_models = [m for m, available in self.llm_manager.available_models.items() if available]
            if available_policy_models:
                logger.warning(f"⚠️ Policy models unavailable, using fallback: {available_policy_models[0]}")
        
        for model_key in available_policy_models:
            try:
                response, metadata = self.llm_manager.generate_with_model(
                    policy_prompt, model_key, temperature=0.3, max_tokens=512
                )
                
                # Parse action probabilities from LLM response
                action_probs = self._parse_action_probabilities(response)
                model_predictions.append(action_probs)
                
                logger.debug(f"🎯 {model_key} policy: {action_probs}")
                
            except Exception as e:
                logger.error(f"❌ Policy error with {model_key}: {e}")
                continue
        
        # Ensemble average of predictions
        if model_predictions:
            ensemble_probs = {}
            for action in [0, 1, 2]:  # Hold, Buy, Sell
                ensemble_probs[action] = np.mean([pred.get(action, 0.33) for pred in model_predictions])
            
            # Normalize probabilities
            total_prob = sum(ensemble_probs.values())
            if total_prob > 0:
                ensemble_probs = {k: v/total_prob for k, v in ensemble_probs.items()}
            else:
                ensemble_probs = {0: 0.33, 1: 0.33, 2: 0.34}  # Default uniform
        else:
            ensemble_probs = {0: 0.33, 1: 0.33, 2: 0.34}  # Fallback uniform
        
        # Cache the result to avoid repeated LLM calls during MCTS
        self.action_prob_cache[obs_key] = ensemble_probs
        
        return ensemble_probs
    
    def _evaluate_position(self, market_context: Dict) -> float:
        """
        Use LLM ensemble to evaluate position value
        
        Args:
            market_context: Current market state
            
        Returns:
            Position value estimate (-1 to 1)
        """
        # Generate value prompt
        value_prompt = self._create_value_prompt(market_context)
        
        # Get value estimates from value models
        value_estimates = []
        available_value_models = [m for m in self.value_models if self.llm_manager.available_models.get(m, False)]
        
        # If no value models available, use any available model as fallback
        if not available_value_models:
            available_value_models = [m for m, available in self.llm_manager.available_models.items() if available]
            if available_value_models:
                logger.warning(f"⚠️ Value models unavailable, using fallback: {available_value_models[0]}")
        
        for model_key in available_value_models:
            try:
                response, metadata = self.llm_manager.generate_with_model(
                    value_prompt, model_key, temperature=0.2, max_tokens=256
                )
                
                # Parse value estimate from LLM response
                value = self._parse_value_estimate(response)
                value_estimates.append(value)
                
                logger.debug(f"💰 {model_key} value: {value:.3f}")
                
            except Exception as e:
                logger.error(f"❌ Value error with {model_key}: {e}")
                continue
        
        # Return ensemble average
        return np.mean(value_estimates) if value_estimates else 0.0
    
    def _create_policy_prompt(self, market_context: Dict) -> str:
        """Create comprehensive prompt for policy network with full portfolio visibility"""
        
        # Extract portfolio context from environment with safe defaults
        portfolio_ctx = market_context.get('portfolio_context', {})
        account = portfolio_ctx.get('account_overview', {}) if portfolio_ctx else {}
        positions = portfolio_ctx.get('positions', {}) if portfolio_ctx else {}
        trading_perf = portfolio_ctx.get('trading_performance', {}) if portfolio_ctx else {}
        risk_metrics = portfolio_ctx.get('risk_metrics', {}) if portfolio_ctx else {}
        market_ctx = portfolio_ctx.get('market_context', {}) if portfolio_ctx else {}
        
        
        # Current market data
        current_price = market_ctx.get('current_price', 0)
        price_change = market_context.get('price_change_24h', 0)
        
        prompt = f"""You are a professional portfolio manager with complete visibility into your trading account. 
Make an informed decision based on your current holdings, performance, and market conditions.

=== ACCOUNT OVERVIEW ===
Portfolio Value: ${account.get('portfolio_value', 0):,.2f}
Available Cash: ${account.get('balance', 0):,.2f}
Total Return: {account.get('total_return', 0):.2%} (${account.get('total_return_usd', 0):+,.2f})
Cash Allocation: {account.get('cash_allocation', 0):.1%}

=== CURRENT POSITIONS ==="""

        if positions:
            for symbol, pos in positions.items():
                prompt += f"""
{symbol}: {pos['quantity']:.6f} units @ ${pos['avg_price']:,.2f} avg
  Current: ${pos['current_price']:,.2f} | P&L: ${pos['unrealized_pnl']:+,.2f} ({pos['unrealized_pnl_pct']:+.2%})
  Weight: {pos['weight']:.1%} | Value: ${pos['position_value']:,.2f}"""
        else:
            prompt += "\nNo current positions - fully in cash"
        
        trading_total = trading_perf.get('total_trades', 0)
        trading_win_rate = trading_perf.get('recent_win_rate', 0)
        last_trade = trading_perf.get('last_trade', {})
        last_trade_side = last_trade.get('side', 'None') if last_trade else 'None'
        last_trade_price = last_trade.get('price', 0) if last_trade else 0
        max_pos_weight = risk_metrics.get('max_position_weight', 0)
        total_invested = risk_metrics.get('total_position_weight', 0)
        account_growth = risk_metrics.get('account_growth_multiple', 1.0)
        drawdown = risk_metrics.get('drawdown_from_peak', 0)
        
        prompt += f"""

=== TRADING PERFORMANCE ===
Total Trades: {trading_total}
Recent Win Rate: {trading_win_rate:.1%}
Last Trade: {last_trade_side} @ ${last_trade_price:,.2f}

=== RISK ANALYSIS ===
Max Position Weight: {max_pos_weight:.1%}
Total Invested: {total_invested:.1%}
Account Growth: {account_growth:.2f}x
Drawdown: {drawdown:.2%}

=== MARKET CONDITIONS ===
Symbol: {market_ctx.get('current_symbol', 'BTCUSDT')}
Current Price: ${current_price:,.2f}
24h Change: {price_change:.2f}%
Technical Indicators:
- Momentum: {market_context.get('momentum', 0):.3f}
- Volatility: {market_context.get('volatility', 0):.3f}
- Trend: {market_context.get('ma_trend', 'neutral')}

=== DECISION REQUIRED ===
Based on your complete portfolio situation above, what action should you take?

Actions Available:
0: HOLD - Maintain current positions and cash levels
1: BUY - Add to position or establish new position (risk: concentration, reward: upside)
2: SELL - Reduce or close position (risk: missing gains, reward: lock profits/cut losses)

Consider:
- Your current position size and risk exposure
- Recent trading performance and patterns
- Portfolio balance and cash allocation
- Market momentum and technical setup
- Risk management principles

Respond with probabilities (must sum to 1.0):
HOLD: [probability]
BUY: [probability]  
SELL: [probability]

Strategic Reasoning: [explain your decision based on portfolio analysis]"""
        
        return prompt
    
    def _create_value_prompt(self, market_context: Dict) -> str:
        """Create prompt for value network"""
        current_price = market_context.get('current_price', 0)
        portfolio_return = market_context.get('portfolio_return', 0)
        
        prompt = f"""You are a risk assessment expert evaluating the current trading position.

Position Analysis:
- Current Price: ${current_price:,.2f}
- Portfolio Return: {portfolio_return:.2f}%
- Market Momentum: {market_context.get('momentum', 0):.3f}
- Volatility Level: {market_context.get('volatility', 0):.3f}

Risk Factors:
- Position exposure: {market_context.get('position_exposure', 0):.2f}%
- Market trend: {market_context.get('trend', 'neutral')}
- Recent performance: {market_context.get('recent_performance', 0):.2f}%

Evaluate the position value from -1.0 (very bearish) to +1.0 (very bullish).
Consider risk-adjusted returns and market conditions.

Position Value: [value between -1.0 and +1.0]
Confidence: [high/medium/low]
Reasoning: [brief explanation]"""
        
        return prompt
    
    def _parse_action_probabilities(self, llm_response: str) -> Dict[int, float]:
        """Parse action probabilities from LLM response"""
        try:
            # Extract probabilities using simple regex patterns
            import re
            
            hold_match = re.search(r'HOLD:\s*([0-9.]+)', llm_response, re.IGNORECASE)
            buy_match = re.search(r'BUY:\s*([0-9.]+)', llm_response, re.IGNORECASE)
            sell_match = re.search(r'SELL:\s*([0-9.]+)', llm_response, re.IGNORECASE)
            
            hold_prob = float(hold_match.group(1)) if hold_match else 0.33
            buy_prob = float(buy_match.group(1)) if buy_match else 0.33
            sell_prob = float(sell_match.group(1)) if sell_match else 0.34
            
            # Normalize probabilities
            total = hold_prob + buy_prob + sell_prob
            if total > 0:
                return {
                    0: hold_prob / total,
                    1: buy_prob / total,
                    2: sell_prob / total
                }
            
        except Exception as e:
            logger.debug(f"⚠️ Failed to parse probabilities: {e}")
        
        # Fallback to uniform distribution
        return {0: 0.33, 1: 0.33, 2: 0.34}
    
    def _parse_value_estimate(self, llm_response: str) -> float:
        """Parse value estimate from LLM response"""
        try:
            import re
            
            # Look for value between -1.0 and 1.0
            value_match = re.search(r'Position Value:\s*([+-]?[0-9]*\.?[0-9]+)', llm_response, re.IGNORECASE)
            if value_match:
                value = float(value_match.group(1))
                return max(-1.0, min(1.0, value))  # Clamp to [-1, 1]
            
            # Alternative patterns
            value_patterns = [
                r'value[:\s]*([+-]?[0-9]*\.?[0-9]+)',
                r'([+-]?[0-9]*\.?[0-9]+)\s*(?:rating|score|value)',
                r'(?:bullish|bearish)[:\s]*([+-]?[0-9]*\.?[0-9]+)'
            ]
            
            for pattern in value_patterns:
                match = re.search(pattern, llm_response, re.IGNORECASE)
                if match:
                    value = float(match.group(1))
                    return max(-1.0, min(1.0, value))
            
        except Exception as e:
            logger.debug(f"⚠️ Failed to parse value: {e}")
        
        return 0.0  # Neutral fallback
    
    def _observation_to_context(self, env: BinanceDemoEnv, observation: np.ndarray) -> Dict:
        """Convert environment observation to market context"""
        portfolio_summary = env.get_portfolio_summary()
        current_price = env.current_prices.get(env.current_symbol, 0)
        
        # Calculate additional metrics
        price_history = env.price_history.get(env.current_symbol, [])
        price_change_24h = 0
        if len(price_history) >= 2:
            price_change_24h = ((current_price - price_history[0]) / price_history[0]) * 100
        
        momentum = observation[4] if len(observation) > 4 else 0
        volatility = observation[5] if len(observation) > 5 else 0
        
        # Get detailed portfolio context for AI decision making
        detailed_portfolio = env.get_detailed_portfolio_context()
        
        context = {
            "current_price": current_price,
            "price_change_24h": price_change_24h,
            "momentum": momentum,
            "volatility": volatility,
            "portfolio_value": portfolio_summary["portfolio_value"],
            "balance": portfolio_summary["balance"],
            "portfolio_return": portfolio_summary["total_return"],
            "total_trades": portfolio_summary["total_trades"],
            "position_size": 0,
            "position_exposure": 0,
            "trend": "neutral",
            "recent_performance": 0,
            "ma_trend": "neutral",
            "portfolio_context": detailed_portfolio  # Full portfolio context for AI
        }
        
        # Add position information
        if env.current_symbol in portfolio_summary["positions"]:
            pos = portfolio_summary["positions"][env.current_symbol]
            context["position_size"] = pos["quantity"]
            context["position_exposure"] = (pos["market_value"] / portfolio_summary["portfolio_value"]) * 100
        
        return context
    
    def _select_final_action(self) -> int:
        """Select final action based on MCTS visit counts"""
        if not self.root_node or not self.root_node.children:
            return 0  # Default to hold
        
        # Calculate visit-based probabilities
        total_visits = sum(child.visits for child in self.root_node.children.values())
        if total_visits == 0:
            return 0
        
        # Temperature-based selection
        if self.temperature == 0:
            # Greedy selection
            return max(self.root_node.children.keys(), 
                      key=lambda a: self.root_node.children[a].visits)
        else:
            # Stochastic selection based on visit counts
            visit_counts = []
            actions = []
            for action, child in self.root_node.children.items():
                actions.append(action)
                visit_counts.append(child.visits)
            
            # Apply temperature
            if self.temperature != 1.0:
                visit_counts = [(v / total_visits) ** (1.0 / self.temperature) for v in visit_counts]
                total = sum(visit_counts)
                visit_counts = [v / total for v in visit_counts]
            else:
                visit_counts = [v / total_visits for v in visit_counts]
            
            return np.random.choice(actions, p=visit_counts)
    
    def _get_mcts_stats(self) -> Dict:
        """Get MCTS statistics for analysis"""
        if not self.root_node:
            return {}
        
        stats = {
            "total_simulations": self.root_node.visits,
            "root_value": self.root_node.q_value,
            "children_stats": {}
        }
        
        for action, child in self.root_node.children.items():
            action_name = ["HOLD", "BUY", "SELL"][action]
            stats["children_stats"][action_name] = {
                "visits": child.visits,
                "q_value": child.q_value,
                "prior_prob": child.prior_prob
            }
        
        return stats
    
    def _is_terminal_state(self, state: Dict) -> bool:
        """Check if state is terminal"""
        # For trading, we don't have clear terminal states in MCTS
        # Could add conditions like extreme portfolio loss
        portfolio_return = state.get("portfolio_return", 0)
        return portfolio_return < -0.5  # 50% loss
    
    def add_experience(self, experience: TradingExperience):
        """Add experience to replay buffer"""
        self.experience_buffer.append(experience)
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)
    
    def train_from_episode(self, episode_data: List[Dict]):
        """Train from completed episode data with enhanced learning"""
        if not episode_data:
            return
            
        # Extract episode outcomes for learning
        total_return = episode_data[-1].get("portfolio_return", 0) if episode_data else 0
        episode_reward = sum(step.get("reward", 0) for step in episode_data)
        
        # Create comprehensive experience for each decision point
        for i, step_data in enumerate(episode_data):
            market_context = step_data.get("market_context", {})
            observation = step_data.get("observation")
            action = step_data.get("action", 0)
            
            # Score this decision based on eventual outcome
            decision_score = self._score_decision(i, len(episode_data), total_return, episode_reward)
            
            # Create experience tuple
            experience = TradingExperience(
                state=observation if observation is not None else np.zeros(25),
                action_probs={action: 1.0},  # Record actual action taken
                reward=decision_score,
                next_state=episode_data[i+1].get("observation") if i+1 < len(episode_data) else np.zeros(25),
                done=(i == len(episode_data) - 1),
                market_context=market_context
            )
            
            self.add_experience(experience)
        
        # Update training statistics
        self.training_episodes += 1
        self.avg_return = (self.avg_return * (self.training_episodes - 1) + total_return) / self.training_episodes
        
        if total_return > 0:
            self.win_rate = (self.win_rate * (self.training_episodes - 1) + 1) / self.training_episodes
        else:
            self.win_rate = (self.win_rate * (self.training_episodes - 1)) / self.training_episodes
        
        # Save experience buffer to persistent memory
        self._save_experience_buffer()
        
        logger.info(f"📚 Training Episode {self.training_episodes}")
        logger.info(f"   Return: {total_return:.2%}")
        logger.info(f"   Decisions: {len(episode_data)}")
        logger.info(f"   Experience Buffer: {len(self.experience_buffer)}")
        logger.info(f"   Avg Return: {self.avg_return:.2%}")
        logger.info(f"   Win Rate: {self.win_rate:.2%}")
    
    def get_training_stats(self) -> Dict:
        """Get training statistics"""
        return {
            "training_episodes": self.training_episodes,
            "avg_return": self.avg_return,
            "win_rate": self.win_rate,
            "experience_buffer_size": len(self.experience_buffer),
            "model_performance": dict(self.model_performance)
        }
    
    def _score_decision(self, step_index: int, total_steps: int, final_return: float, episode_reward: float) -> float:
        """Score individual trading decisions based on episode outcome"""
        # Weight decisions more heavily if they're closer to the end (recency)
        recency_weight = (step_index + 1) / total_steps
        
        # Base score from final episode performance
        base_score = final_return * 0.7 + episode_reward * 0.3
        
        # Apply recency weighting (recent decisions matter more)
        decision_score = base_score * (0.5 + 0.5 * recency_weight)
        
        return decision_score
    
    def _save_experience_buffer(self):
        """Save experience buffer to persistent storage"""
        import pickle
        import os
        
        try:
            os.makedirs("outputs", exist_ok=True)
            
            # Prepare data for pickling (convert numpy arrays to lists if needed)
            serializable_buffer = []
            for exp in self.experience_buffer:
                exp_dict = {
                    'state': exp.state.tolist() if hasattr(exp.state, 'tolist') else exp.state,
                    'action_probs': exp.action_probs,
                    'reward': exp.reward,
                    'next_state': exp.next_state.tolist() if hasattr(exp.next_state, 'tolist') else exp.next_state,
                    'done': exp.done,
                    'market_context': exp.market_context
                }
                serializable_buffer.append(exp_dict)
            
            save_data = {
                'experience_buffer': serializable_buffer,
                'training_episodes': self.training_episodes,
                'avg_return': self.avg_return,
                'win_rate': self.win_rate,
                'model_performance': dict(self.model_performance)
            }
            
            with open(self.memory_file, 'wb') as f:
                pickle.dump(save_data, f)
                
            logger.debug(f"💾 Experience buffer saved: {len(self.experience_buffer)} experiences")
            
        except Exception as e:
            logger.error(f"❌ Failed to save experience buffer: {e}")
    
    def _load_experience_buffer(self):
        """Load experience buffer from persistent storage"""
        import pickle
        import os
        
        if not os.path.exists(self.memory_file):
            logger.info("📝 No existing agent memory found, starting fresh")
            return
            
        try:
            with open(self.memory_file, 'rb') as f:
                save_data = pickle.load(f)
            
            # Restore experience buffer
            self.experience_buffer = []
            for exp_dict in save_data.get('experience_buffer', []):
                experience = TradingExperience(
                    state=np.array(exp_dict['state']),
                    action_probs=exp_dict['action_probs'],
                    reward=exp_dict['reward'],
                    next_state=np.array(exp_dict['next_state']),
                    done=exp_dict['done'],
                    market_context=exp_dict['market_context']
                )
                self.experience_buffer.append(experience)
            
            # Restore training statistics
            self.training_episodes = save_data.get('training_episodes', 0)
            self.avg_return = save_data.get('avg_return', 0.0)
            self.win_rate = save_data.get('win_rate', 0.0)
            self.model_performance = defaultdict(list, save_data.get('model_performance', {}))
            
            logger.info(f"🧠 Agent memory loaded: {len(self.experience_buffer)} experiences, {self.training_episodes} episodes")
            logger.info(f"   Historical performance: {self.avg_return:.2%} avg return, {self.win_rate:.2%} win rate")
            
        except Exception as e:
            logger.error(f"❌ Failed to load experience buffer: {e}")
    
    def _generate_decision_reasoning(self, market_context: Dict, action: int) -> Dict:
        """Generate human-readable explanation for AI trading decision"""
        action_name = ["HOLD", "BUY", "SELL"][action]
        
        # Extract key market factors
        portfolio_ctx = market_context.get('portfolio_context', {})
        account = portfolio_ctx.get('account_overview', {})
        positions = portfolio_ctx.get('positions', {})
        
        # MCTS analysis
        mcts_stats = self._get_mcts_stats()
        
        # Generate reasoning
        reasoning = {
            "decision": action_name,
            "confidence": self._calculate_decision_confidence(mcts_stats),
            "primary_factors": self._identify_key_factors(market_context, action),
            "portfolio_analysis": {
                "current_value": account.get('portfolio_value', 0),
                "cash_allocation": account.get('cash_allocation', 0),
                "total_return": account.get('total_return', 0),
                "risk_level": self._assess_risk_level(market_context)
            },
            "mcts_analysis": {
                "simulations_run": mcts_stats.get("total_simulations", 0),
                "alternative_actions": mcts_stats.get("children_stats", {}),
                "consensus_strength": self._calculate_consensus_strength(mcts_stats)
            },
            "market_conditions": {
                "trend": market_context.get('ma_trend', 'neutral'),
                "momentum": market_context.get('momentum', 0),
                "volatility": market_context.get('volatility', 0)
            }
        }
        
        return reasoning
    
    def _calculate_decision_confidence(self, mcts_stats: Dict) -> float:
        """Calculate confidence level for the decision based on MCTS consensus"""
        if not mcts_stats or "children_stats" not in mcts_stats:
            return 0.5
        
        children = mcts_stats["children_stats"]
        if not children:
            return 0.5
        
        # Get visit counts for all actions
        visit_counts = [child.get("visits", 0) for child in children.values()]
        total_visits = sum(visit_counts)
        
        if total_visits == 0:
            return 0.5
        
        # Confidence is based on how dominant the top choice is
        max_visits = max(visit_counts) if visit_counts else 0
        confidence = max_visits / total_visits if total_visits > 0 else 0.5
        
        return min(max(confidence, 0.0), 1.0)
    
    def _identify_key_factors(self, market_context: Dict, action: int) -> List[str]:
        """Identify the key factors that influenced the decision"""
        factors = []
        
        # Portfolio factors
        portfolio_ctx = market_context.get('portfolio_context', {})
        account = portfolio_ctx.get('account_overview', {})
        
        cash_allocation = account.get('cash_allocation', 0)
        total_return = account.get('total_return', 0)
        
        if action == 1:  # BUY
            if cash_allocation > 0.7:
                factors.append("High cash allocation suggests buying opportunity")
            if total_return < -0.05:
                factors.append("Portfolio underperforming, seeking recovery")
            if market_context.get('momentum', 0) > 0:
                factors.append("Positive momentum supports buy decision")
        
        elif action == 2:  # SELL
            if cash_allocation < 0.3:
                factors.append("Low cash allocation suggests taking profits")
            if total_return > 0.05:
                factors.append("Strong portfolio performance, securing gains")
            if market_context.get('volatility', 0) > 0.5:
                factors.append("High volatility suggests risk reduction")
        
        else:  # HOLD
            factors.append("Balanced portfolio state supports holding")
            if abs(market_context.get('momentum', 0)) < 0.1:
                factors.append("Neutral momentum supports wait-and-see approach")
        
        return factors[:3]  # Top 3 factors
    
    def _assess_risk_level(self, market_context: Dict) -> str:
        """Assess current risk level"""
        volatility = market_context.get('volatility', 0)
        portfolio_ctx = market_context.get('portfolio_context', {})
        account = portfolio_ctx.get('account_overview', {})
        total_return = account.get('total_return', 0)
        
        risk_score = abs(volatility) + abs(total_return)
        
        if risk_score > 0.5:
            return "HIGH"
        elif risk_score > 0.2:
            return "MEDIUM" 
        else:
            return "LOW"
    
    def _calculate_consensus_strength(self, mcts_stats: Dict) -> str:
        """Calculate how strong the consensus is among MCTS simulations"""
        confidence = self._calculate_decision_confidence(mcts_stats)
        
        if confidence > 0.8:
            return "STRONG"
        elif confidence > 0.6:
            return "MODERATE"
        else:
            return "WEAK"
    
    def _save_decision_reasoning(self, decision_data: Dict):
        """Save decision reasoning for dashboard display"""
        import json
        import os
        
        try:
            os.makedirs("outputs", exist_ok=True)
            
            # Load existing reasoning history
            reasoning_file = "outputs/ai_reasoning_history.json"
            reasoning_history = []
            
            if os.path.exists(reasoning_file):
                with open(reasoning_file, 'r') as f:
                    reasoning_history = json.load(f)
            
            # Add new decision
            reasoning_history.append(decision_data)
            
            # Keep only last 100 decisions
            reasoning_history = reasoning_history[-100:]
            
            # Save updated history
            with open(reasoning_file, 'w') as f:
                json.dump(reasoning_history, f, indent=2, default=str)
            
            logger.debug(f"🧠 AI reasoning saved for {decision_data.get('action_name', 'UNKNOWN')} decision")
            
        except Exception as e:
            logger.error(f"❌ Failed to save decision reasoning: {e}")

if __name__ == "__main__":
    # Test the AlphaZero agent
    from multi_llm_manager import MultiLLMManager
    from binance_demo_env import BinanceDemoEnv
    
    print("🧪 Testing AlphaZero Trading Agent")
    
    # Initialize components
    llm_manager = MultiLLMManager()
    env = BinanceDemoEnv(symbols=["BTCUSDT"], initial_balance=10000.0)
    agent = AlphaZeroTradingAgent(llm_manager, mcts_simulations=10)  # Reduced for testing
    
    # Run test episode
    obs = env.reset()
    done = False
    step = 0
    
    while not done and step < 5:  # Short test
        action = agent.select_action(env, obs)
        obs, reward, done, info = env.step(action)
        
        print(f"Step {step + 1}: Action={action}, Reward={reward:.4f}")
        print(f"   Portfolio Value: ${info.get('portfolio_value', 0):,.2f}")
        
        step += 1
    
    # Get final statistics
    training_stats = agent.get_training_stats()
    print(f"\\n📊 Training Stats: {training_stats}")
    
    env.close()
    print("✅ AlphaZero agent test completed")