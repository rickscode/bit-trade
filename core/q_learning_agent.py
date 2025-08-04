import numpy as np
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import pickle

class QLearningAgent:
    """Q-Learning agent for strategy selection and optimization"""
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.95, 
                 epsilon: float = 0.1, epsilon_decay: float = 0.995, 
                 min_epsilon: float = 0.01):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Q-table: state -> action -> Q-value
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Experience replay buffer
        self.experience_buffer = []
        self.max_buffer_size = 10000
        
        # State and action spaces (expanded to 23 models: Groq + Cloudflare + OpenRouter)
        self.actions = [
            # Groq models (5)
            "versatile", "analytical", "maverick", "scout", "diverse",
            # Cloudflare models (3) 
            "reasoning", "coder", "questioner",
            # OpenRouter Tier 1: High-performance (4)
            "horizon", "glm_air", "kimi_k2", "kimi_dev",
            # OpenRouter Tier 2: Specialized (7)
            "qwen_coder", "deepseek_r1", "deepseek_0528", "deepseek_qwen", 
            "chimera", "mistral_small", "mistral_devstral",
            # OpenRouter Tier 3: Efficient/Diverse (4)
            "qwen_qwq", "gemma3_4b", "sarvam_m", "hunyuan"
        ]
        self.states = self._initialize_states()
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_count = 0
        
        # Load existing Q-table if available
        self.q_table_file = "outputs/q_table.pkl"
        self._load_q_table()
    
    def _initialize_states(self) -> List[str]:
        """Initialize possible market states"""
        # Market conditions
        volatility_states = ["low_vol", "med_vol", "high_vol"]
        trend_states = ["bull", "bear", "sideways"]
        performance_states = ["good", "average", "poor"]
        
        states = []
        for vol in volatility_states:
            for trend in trend_states:
                for perf in performance_states:
                    states.append(f"{vol}_{trend}_{perf}")
        
        return states
    
    def _get_market_state(self, market_data: Dict) -> str:
        """Convert market data to discrete state"""
        # Analyze market conditions
        volatility = market_data.get('volatility', 0.02)
        trend = market_data.get('trend', 0.0)
        recent_performance = market_data.get('recent_performance', 0.0)
        
        # Categorize volatility
        if volatility < 0.015:
            vol_state = "low_vol"
        elif volatility < 0.035:
            vol_state = "med_vol"
        else:
            vol_state = "high_vol"
        
        # Categorize trend
        if trend > 0.02:
            trend_state = "bull"
        elif trend < -0.02:
            trend_state = "bear"
        else:
            trend_state = "sideways"
        
        # Categorize recent performance
        if recent_performance > 0.05:
            perf_state = "good"
        elif recent_performance > -0.05:
            perf_state = "average"
        else:
            perf_state = "poor"
        
        return f"{vol_state}_{trend_state}_{perf_state}"
    
    def select_action(self, state: str, available_actions: List[str] = None) -> str:
        """Select action using epsilon-greedy strategy"""
        if available_actions is None:
            available_actions = self.actions
        
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            # Explore: random action
            action = np.random.choice(available_actions)
        else:
            # Exploit: best known action
            q_values = {action: self.q_table[state][action] for action in available_actions}
            action = max(q_values, key=q_values.get)
        
        return action
    
    def update_q_value(self, state: str, action: str, reward: float, next_state: str, done: bool = False):
        """Update Q-value using Q-learning update rule"""
        current_q = self.q_table[state][action]
        
        if done:
            next_q_max = 0
        else:
            next_q_max = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0
        
        # Q-learning update rule
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_q_max - current_q)
        self.q_table[state][action] = new_q
        
        # Store experience for replay
        experience = (state, action, reward, next_state, done)
        self.experience_buffer.append(experience)
        
        # Limit buffer size
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)
    
    def calculate_reward(self, strategy_metrics: Dict) -> float:
        """Calculate reward based on strategy performance"""
        # Multi-objective reward function
        total_return = strategy_metrics.get("Total Return [%]", 0) / 100
        sharpe_ratio = strategy_metrics.get("Sharpe Ratio", 0)
        max_drawdown = strategy_metrics.get("Max Drawdown [%]", 0) / 100
        win_rate = strategy_metrics.get("Win Rate [%]", 0) / 100
        
        # Weighted reward combining multiple factors
        reward = (
            0.4 * total_return +          # 40% weight on returns
            0.3 * sharpe_ratio +          # 30% weight on risk-adjusted returns
            0.2 * win_rate +              # 20% weight on consistency
            0.1 * (1 - max_drawdown)      # 10% weight on drawdown protection
        )
        
        # Bonus for exceptional performance
        if total_return > 0.15 and sharpe_ratio > 1.5:
            reward += 0.5
        
        # Penalty for poor performance
        if total_return < -0.05 or sharpe_ratio < 0:
            reward -= 0.3
        
        return reward
    
    def experience_replay(self, batch_size: int = 32):
        """Perform experience replay to improve learning"""
        if len(self.experience_buffer) < batch_size:
            return
        
        # Sample random batch
        batch = np.random.choice(self.experience_buffer, batch_size, replace=False)
        
        for state, action, reward, next_state, done in batch:
            self.update_q_value(state, action, reward, next_state, done)
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def get_action_values(self, state: str) -> Dict[str, float]:
        """Get Q-values for all actions in given state"""
        return dict(self.q_table[state])
    
    def get_best_action(self, state: str, available_actions: List[str] = None) -> str:
        """Get best action for given state (no exploration)"""
        if available_actions is None:
            available_actions = self.actions
        
        q_values = {action: self.q_table[state][action] for action in available_actions}
        return max(q_values, key=q_values.get)
    
    def save_q_table(self):
        """Save Q-table to file"""
        try:
            os.makedirs(os.path.dirname(self.q_table_file), exist_ok=True)
            
            # Convert defaultdict to regular dict for serialization
            q_table_dict = {
                state: dict(actions) for state, actions in self.q_table.items()
            }
            
            save_data = {
                "q_table": q_table_dict,
                "epsilon": self.epsilon,
                "episode_count": self.episode_count,
                "episode_rewards": self.episode_rewards[-100:],  # Keep last 100 episodes
                "timestamp": datetime.now().isoformat()
            }
            
            with open(self.q_table_file, 'wb') as f:
                pickle.dump(save_data, f)
                
        except Exception as e:
            print(f"Error saving Q-table: {e}")
    
    def _load_q_table(self):
        """Load Q-table from file"""
        try:
            if os.path.exists(self.q_table_file):
                with open(self.q_table_file, 'rb') as f:
                    save_data = pickle.load(f)
                
                # Restore Q-table
                q_table_dict = save_data.get("q_table", {})
                self.q_table = defaultdict(lambda: defaultdict(float))
                
                for state, actions in q_table_dict.items():
                    for action, q_value in actions.items():
                        self.q_table[state][action] = q_value
                
                # Restore other parameters
                self.epsilon = save_data.get("epsilon", self.epsilon)
                self.episode_count = save_data.get("episode_count", 0)
                self.episode_rewards = save_data.get("episode_rewards", [])
                
                print(f"✅ Loaded Q-table with {len(self.q_table)} states")
                
        except Exception as e:
            print(f"⚠️  Could not load Q-table: {e}")
    
    def get_learning_stats(self) -> Dict:
        """Get learning statistics"""
        if not self.episode_rewards:
            return {"episodes": 0, "avg_reward": 0, "epsilon": self.epsilon}
        
        recent_rewards = self.episode_rewards[-20:] if len(self.episode_rewards) >= 20 else self.episode_rewards
        
        return {
            "episodes": len(self.episode_rewards),
            "avg_reward": np.mean(recent_rewards),
            "max_reward": max(self.episode_rewards),
            "min_reward": min(self.episode_rewards),
            "epsilon": self.epsilon,
            "q_table_size": len(self.q_table),
            "recent_trend": "improving" if len(recent_rewards) > 10 and 
                           np.mean(recent_rewards[-10:]) > np.mean(recent_rewards[-20:-10]) else "stable"
        }
    
    def end_episode(self, total_reward: float):
        """End current episode and update statistics"""
        self.episode_rewards.append(total_reward)
        self.episode_count += 1
        self.decay_epsilon()
        
        # Perform experience replay
        self.experience_replay()
        
        # Save Q-table periodically
        if self.episode_count % 10 == 0:
            self.save_q_table()

class QLearningModelSelector:
    """Q-Learning based model selector for the trading system"""
    
    def __init__(self, multi_llm_manager):
        self.multi_llm_manager = multi_llm_manager
        self.q_agent = QLearningAgent()
        self.current_state = None
        self.current_action = None
        self.episode_rewards = []
    
    def select_model(self, market_data: Dict, available_models: List[str]) -> str:
        """Select model using Q-learning"""
        # Get current market state
        self.current_state = self.q_agent._get_market_state(market_data)
        
        # Select action (model) using Q-learning
        self.current_action = self.q_agent.select_action(self.current_state, available_models)
        
        return self.current_action
    
    def update_performance(self, strategy_metrics: Dict, next_market_data: Dict = None):
        """Update Q-learning agent based on strategy performance"""
        if self.current_state is None or self.current_action is None:
            return
        
        # Calculate reward
        reward = self.q_agent.calculate_reward(strategy_metrics)
        self.episode_rewards.append(reward)
        
        # Determine next state
        if next_market_data:
            next_state = self.q_agent._get_market_state(next_market_data)
        else:
            next_state = self.current_state  # Assume same state if no new data
        
        # Update Q-value
        self.q_agent.update_q_value(self.current_state, self.current_action, reward, next_state)
        
        # Reset for next strategy
        self.current_state = None
        self.current_action = None
    
    def end_learning_cycle(self):
        """End current learning cycle"""
        if self.episode_rewards:
            total_reward = sum(self.episode_rewards)
            self.q_agent.end_episode(total_reward)
            self.episode_rewards = []
    
    def get_learning_stats(self) -> Dict:
        """Get Q-learning statistics"""
        return self.q_agent.get_learning_stats()

if __name__ == "__main__":
    # Test Q-learning agent
    agent = QLearningAgent()
    
    # Simulate some learning
    for episode in range(10):
        state = "med_vol_bull_average"
        action = agent.select_action(state)
        
        # Simulate strategy result
        mock_metrics = {
            "Total Return [%]": np.random.normal(10, 5),
            "Sharpe Ratio": np.random.normal(1.0, 0.3),
            "Max Drawdown [%]": np.random.normal(5, 2),
            "Win Rate [%]": np.random.normal(60, 10)
        }
        
        reward = agent.calculate_reward(mock_metrics)
        next_state = "med_vol_bull_good"
        
        agent.update_q_value(state, action, reward, next_state)
        agent.end_episode(reward)
    
    # Print learning stats
    stats = agent.get_learning_stats()
    print("Q-Learning Agent Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nQ-Learning Agent Test Complete!")