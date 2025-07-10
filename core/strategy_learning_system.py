import json
import os
import pandas as pd
from datetime import datetime, timedelta
from groq import Groq
from supabase import create_client, Client
from dotenv import load_dotenv
import random
import numpy as np
try:
    from .multi_llm_manager import MultiLLMManager
except ImportError:
    from multi_llm_manager import MultiLLMManager

load_dotenv()

class StrategyLearningSystem:
    def __init__(self):
        self.groq_client = Groq()
        self.supabase: Client = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )
        
        # Initialize multi-LLM manager
        self.llm_manager = MultiLLMManager()
        print("🤖 Multi-LLM Manager initialized with 5 free Groq models")
        
        # Strategy templates for diversification
        self.strategy_templates = [
            "momentum_based",
            "mean_reversion",
            "breakout_detection",
            "trend_following",
            "volatility_trading",
            "support_resistance",
            "multi_timeframe",
            "volume_analysis"
        ]
        
        # Technical indicators pool
        self.indicators = [
            "RSI", "MACD", "Bollinger Bands", "Moving Average", "Stochastic",
            "ADX", "CCI", "Williams %R", "Ichimoku", "Parabolic SAR",
            "Volume Profile", "OBV", "Money Flow Index", "ATR"
        ]
        
        # Risk management styles
        self.risk_styles = [
            "fixed_percentage", "volatility_based", "kelly_criterion",
            "position_sizing", "stop_loss_trailing", "time_based_exit"
        ]

    def get_successful_strategies(self, min_return=5.0, min_sharpe=0.5):
        """Retrieve successful strategies from database for learning"""
        try:
            response = self.supabase.table("trading-strategies").select("*").filter(
                "is_successful", "eq", True
            ).execute()
            
            successful_strategies = []
            for strategy in response.data:
                metrics = strategy.get("metrics", {})
                if (metrics.get("Total Return [%]", 0) >= min_return and 
                    metrics.get("Sharpe Ratio", 0) >= min_sharpe):
                    successful_strategies.append(strategy)
            
            return successful_strategies
        except Exception as e:
            print(f"Error fetching successful strategies: {e}")
            return []

    def analyze_successful_patterns(self, strategies):
        """Analyze patterns in successful strategies"""
        if not strategies:
            return {}
        
        patterns = {
            "common_indicators": {},
            "risk_management": {},
            "timeframes": {},
            "performance_metrics": []
        }
        
        for strategy in strategies:
            metrics = strategy.get("metrics", {})
            patterns["performance_metrics"].append(metrics)
            
            # Extract patterns from strategy code (simplified)
            code = strategy.get("strategy_code", "")
            if "RSI" in code:
                patterns["common_indicators"]["RSI"] = patterns["common_indicators"].get("RSI", 0) + 1
            if "MA" in code or "moving" in code.lower():
                patterns["common_indicators"]["MA"] = patterns["common_indicators"].get("MA", 0) + 1
            if "MACD" in code:
                patterns["common_indicators"]["MACD"] = patterns["common_indicators"].get("MACD", 0) + 1
        
        return patterns

    def _calculate_market_conditions(self, market_data):
        """Calculate market conditions for Q-learning agent"""
        try:
            # Calculate volatility
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.std()
            
            # Calculate trend (price change over period)
            price_change = (market_data['close'].iloc[-1] - market_data['close'].iloc[0]) / market_data['close'].iloc[0]
            
            # Calculate recent performance (last 20 periods)
            recent_data = market_data.tail(min(20, len(market_data)))
            recent_performance = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
            
            return {
                'volatility': volatility,
                'trend': price_change,
                'recent_performance': recent_performance
            }
        except Exception as e:
            print(f"Error calculating market conditions: {e}")
            return {'volatility': 0.02, 'trend': 0.0, 'recent_performance': 0.0}

    def generate_enhanced_strategy(self, market_data, learning_insights=None):
        """Generate strategy with learning insights and diversification"""
        
        # Select random strategy template for diversification
        strategy_type = random.choice(self.strategy_templates)
        selected_indicators = random.sample(self.indicators, random.randint(2, 4))
        risk_style = random.choice(self.risk_styles)
        
        # Calculate market conditions for Q-learning
        market_conditions = self._calculate_market_conditions(market_data)
        
        # Build enhanced prompt with learning insights
        prompt = f"""
        Generate a {strategy_type} trading strategy using the following market data.
        
        LEARNING INSIGHTS:
        {json.dumps(learning_insights, indent=2) if learning_insights else "No previous learning data"}
        
        REQUIREMENTS:
        1. Use these technical indicators: {', '.join(selected_indicators)}
        2. Implement {risk_style} risk management
        3. Include specific entry/exit rules
        4. Add position sizing logic
        5. Consider market volatility
        6. Include stop-loss and take-profit levels
        
        MARKET DATA SUMMARY:
        - Total periods: {len(market_data)}
        - Price range: {market_data['low'].min():.2f} - {market_data['high'].max():.2f}
        - Average volume: {market_data['volume'].mean():.0f}
        - Volatility: {market_data['close'].pct_change().std():.4f}
        
        Generate a complete Python strategy with clear logic and explain the reasoning.
        Format as JSON with fields: strategy_type, indicators, entry_rules, exit_rules, risk_management, code
        """
        
        try:
            # Select model using multi-LLM manager with market data for Q-learning
            selected_model = self.llm_manager.select_model(
                task_type="strategy_generation", 
                strategy_index=self.llm_manager.strategy_counter,
                market_data=market_conditions
            )
            
            print(f"🎯 Generating strategy with model: {selected_model}")
            print(f"📊 Market conditions: {market_conditions}")
            
            # Generate strategy using selected model
            strategy_content, metadata = self.llm_manager.generate_with_model(
                prompt, selected_model
            )
            
            # Enhanced strategy metadata
            strategy_data = {
                "strategy_type": strategy_type,
                "indicators": selected_indicators,
                "risk_style": risk_style,
                "generation_timestamp": datetime.now().isoformat(),
                "learning_version": "v2.2_q_learning",
                "model_used": selected_model,
                "market_conditions": market_conditions,
                "model_metadata": {
                    "model_key": metadata.get("model_key"),
                    "model_name": metadata.get("model_name"),
                    "timestamp": metadata.get("timestamp")
                },
                "content": strategy_content
            }
            
            # Increment counter for round-robin selection
            self.llm_manager.increment_strategy_counter()
            
            return strategy_data
            
        except Exception as e:
            print(f"Error generating strategy with multi-LLM: {e}")
            return None

    def batch_generate_strategies(self, market_data, num_strategies=10):
        """Generate multiple diverse strategies"""
        successful_strategies = self.get_successful_strategies()
        learning_insights = self.analyze_successful_patterns(successful_strategies)
        
        strategies = []
        for i in range(num_strategies):
            print(f"Generating strategy {i+1}/{num_strategies}...")
            strategy = self.generate_enhanced_strategy(market_data, learning_insights)
            if strategy:
                strategies.append(strategy)
                
                # Save strategy to file
                filename = f"strategies/enhanced_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.json"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, 'w') as f:
                    json.dump(strategy, f, indent=2)
        
        return strategies

    def recursive_learning_cycle(self, market_data, cycles=5, strategies_per_cycle=5):
        """Implement recursive learning with multiple cycles"""
        print(f"Starting recursive learning cycle with {cycles} cycles...")
        
        all_results = []
        
        for cycle in range(cycles):
            print(f"\n--- Cycle {cycle + 1}/{cycles} ---")
            
            # Generate strategies for this cycle
            strategies = self.batch_generate_strategies(market_data, strategies_per_cycle)
            
            # Backtest each strategy (simplified - you'd integrate with your backtest_strategy.py)
            cycle_results = []
            for i, strategy in enumerate(strategies):
                print(f"Backtesting strategy {i+1}...")
                
                # Placeholder for backtesting - integrate with your existing backtest system
                # result = backtest_strategy(strategy, market_data)
                
                # Simulate results for demonstration
                simulated_result = {
                    "strategy": strategy,
                    "metrics": {
                        "Total Return [%]": random.uniform(-10, 25),
                        "Sharpe Ratio": random.uniform(-0.5, 2.0),
                        "Win Rate [%]": random.uniform(30, 70),
                        "Max Drawdown [%]": random.uniform(5, 30)
                    }
                }
                cycle_results.append(simulated_result)
            
            # Evaluate and save successful strategies
            for result in cycle_results:
                metrics = result["metrics"]
                is_successful = (metrics["Total Return [%]"] > 8 and 
                               metrics["Sharpe Ratio"] > 0.8 and
                               metrics["Max Drawdown [%]"] < 20)
                
                if is_successful:
                    print(f"✅ Successful strategy found! Return: {metrics['Total Return [%]']:.2f}%")
                    # Save to database
                    self.save_strategy_to_db(result["strategy"], metrics, is_successful)
                else:
                    print(f"❌ Strategy underperformed. Return: {metrics['Total Return [%]']:.2f}%")
            
            all_results.extend(cycle_results)
            
            # Wait between cycles for rate limiting
            if cycle < cycles - 1:
                print("Waiting before next cycle...")
                # time.sleep(30)  # Uncomment for actual rate limiting
        
        # End Q-learning cycle after all strategies are processed
        self.llm_manager.end_q_learning_cycle()
        
        print(f"\n🎯 Recursive learning complete! Generated {len(all_results)} strategies across {cycles} cycles.")
        
        # Show Q-learning statistics
        q_stats = self.llm_manager.get_q_learning_stats()
        if q_stats:
            print(f"🧠 Q-learning stats: {q_stats}")
        
        return all_results

    def save_strategy_to_db(self, strategy, metrics, is_successful, next_market_data=None):
        """Save strategy to Supabase database and update model performance"""
        try:
            # Get model information
            model_used = strategy.get('model_used', 'unknown')
            
            record = {
                "symbol": "BTCUSDT",  # Default symbol
                "interval": "1day",   # Default interval
                "strategy_code": json.dumps(strategy),
                "metrics": metrics,
                "llm_notes": f"Generated by enhanced learning system v2.2 Q-learning - {strategy.get('strategy_type', 'unknown')} - Model: {model_used}",
                "is_successful": is_successful
            }
            
            response = self.supabase.table("trading-strategies").insert(record).execute()
            
            # Update model performance tracking with Q-learning
            if model_used != 'unknown':
                self.llm_manager.update_model_performance(model_used, metrics, is_successful, next_market_data)
                print(f"📊 Updated performance for model: {model_used}")
            
            return response.data
            
        except Exception as e:
            print(f"Error saving to database: {e}")
            return None

    def get_learning_statistics(self):
        """Get statistics about the learning system performance"""
        try:
            response = self.supabase.table("trading-strategies").select("*").execute()
            strategies = response.data
            
            total_strategies = len(strategies)
            successful_strategies = len([s for s in strategies if s.get("is_successful", False)])
            
            # Get model performance report
            model_performance = self.llm_manager.get_performance_report()
            
            if total_strategies > 0:
                success_rate = (successful_strategies / total_strategies) * 100
                
                # Calculate average performance metrics
                successful_metrics = [s["metrics"] for s in strategies if s.get("is_successful", False)]
                if successful_metrics:
                    avg_return = np.mean([m.get("Total Return [%]", 0) for m in successful_metrics])
                    avg_sharpe = np.mean([m.get("Sharpe Ratio", 0) for m in successful_metrics])
                    avg_winrate = np.mean([m.get("Win Rate [%]", 0) for m in successful_metrics])
                    
                    return {
                        "total_strategies": total_strategies,
                        "successful_strategies": successful_strategies,
                        "success_rate": success_rate,
                        "avg_return": avg_return,
                        "avg_sharpe": avg_sharpe,
                        "avg_winrate": avg_winrate,
                        "model_performance": model_performance,
                        "q_learning_stats": self.llm_manager.get_q_learning_stats()
                    }
            
            return {
                "total_strategies": total_strategies, 
                "successful_strategies": 0, 
                "success_rate": 0,
                "model_performance": model_performance,
                "q_learning_stats": self.llm_manager.get_q_learning_stats()
            }
            
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {}
    
    def enable_q_learning(self, enable: bool = True):
        """Enable or disable Q-learning for model selection"""
        self.llm_manager.enable_q_learning(enable)
        print(f"🧠 Q-learning {'enabled' if enable else 'disabled'} for strategy learning system")
    
    def set_model_selection_mode(self, mode: str):
        """Set model selection mode (round_robin, performance_based, weighted_random, market_adaptive)"""
        self.llm_manager.switch_selection_mode(mode)
        print(f"🔄 Model selection mode set to: {mode}")

if __name__ == "__main__":
    # Initialize the learning system
    learning_system = StrategyLearningSystem()
    
    # Load market data
    market_data = pd.read_csv("data/BTCUSDT_1day.csv", parse_dates=["timestamp"], index_col="timestamp")
    
    # Run recursive learning cycle
    results = learning_system.recursive_learning_cycle(market_data, cycles=3, strategies_per_cycle=5)
    
    # Show learning statistics
    stats = learning_system.get_learning_statistics()
    print(f"\n📊 Learning System Statistics:")
    print(f"Total Strategies: {stats.get('total_strategies', 0)}")
    print(f"Successful Strategies: {stats.get('successful_strategies', 0)}")
    print(f"Success Rate: {stats.get('success_rate', 0):.2f}%")
    
    if stats.get('successful_strategies', 0) > 0:
        print(f"Average Return: {stats.get('avg_return', 0):.2f}%")
        print(f"Average Sharpe: {stats.get('avg_sharpe', 0):.2f}")
        print(f"Average Win Rate: {stats.get('avg_winrate', 0):.2f}%")