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
    
    @property
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
        
        # Model ensemble for policy and value estimation
        self.policy_models = ["analytical", "scout", "diverse"]  # Strategic thinking
        self.value_models = ["versatile", "analytical"]  # Risk assessment
        self.creative_models = ["maverick", "scout"]  # Exploration
        
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
        # Convert observation to market context
        market_context = self._observation_to_context(env, observation)
        
        # Initialize root node for MCTS
        self.root_node = MCTSNode(state=market_context)
        
        # Run MCTS simulations
        for _ in range(self.mcts_simulations):
            self._run_mcts_simulation(env, observation)
        
        # Select action based on visit counts
        action = self._select_final_action()
        
        # Store decision in game history
        self.game_history.append({
            "observation": observation.copy(),
            "market_context": market_context,
            "action": action,
            "timestamp": datetime.now().isoformat(),
            "mcts_stats": self._get_mcts_stats()
        })
        
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
    
    def _get_action_probabilities(self, market_context: Dict) -> Dict[int, float]:
        """
        Use LLM ensemble to estimate action probabilities
        
        Args:
            market_context: Current market state
            
        Returns:
            Dictionary of action probabilities
        """
        # Generate policy prompt
        policy_prompt = self._create_policy_prompt(market_context)
        
        # Get predictions from policy models
        model_predictions = []
        for model_key in self.policy_models:
            try:
                if not self.llm_manager.available_models.get(model_key, False):
                    continue
                
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
        for model_key in self.value_models:
            try:
                if not self.llm_manager.available_models.get(model_key, False):
                    continue
                
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
        """Create prompt for policy network"""
        current_price = market_context.get('current_price', 0)
        price_change = market_context.get('price_change_24h', 0)
        portfolio_value = market_context.get('portfolio_value', 0)
        balance = market_context.get('balance', 0)
        position_size = market_context.get('position_size', 0)
        
        prompt = f"""You are an expert trading AI analyzing market conditions for optimal action selection.

Market Analysis:
- Current Price: ${current_price:,.2f}
- 24h Price Change: {price_change:.2f}%
- Portfolio Value: ${portfolio_value:,.2f}
- Available Balance: ${balance:,.2f}
- Current Position: {position_size:.6f}

Technical Indicators:
- Price momentum: {market_context.get('momentum', 0):.3f}
- Volatility: {market_context.get('volatility', 0):.3f}
- Moving averages trend: {market_context.get('ma_trend', 'neutral')}

Based on this analysis, what action probabilities would you assign?

Actions:
0: HOLD - Maintain current positions
1: BUY - Increase position size  
2: SELL - Reduce or close positions

Respond with probabilities (must sum to 1.0):
HOLD: [probability]
BUY: [probability]
SELL: [probability]

Reasoning: [brief explanation]"""
        
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
            "ma_trend": "neutral"
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
        """Train from completed episode data"""
        # In AlphaZero, we would train neural networks here
        # For LLM-based system, we update model performance tracking
        
        total_return = episode_data[-1].get("portfolio_return", 0) if episode_data else 0
        
        # Update training statistics
        self.training_episodes += 1
        self.avg_return = (self.avg_return * (self.training_episodes - 1) + total_return) / self.training_episodes
        
        if total_return > 0:
            self.win_rate = (self.win_rate * (self.training_episodes - 1) + 1) / self.training_episodes
        else:
            self.win_rate = (self.win_rate * (self.training_episodes - 1)) / self.training_episodes
        
        logger.info(f"📚 Training Episode {self.training_episodes}")
        logger.info(f"   Return: {total_return:.2%}")
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