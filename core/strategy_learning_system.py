import json
import os
import pandas as pd
from datetime import datetime, timedelta
from groq import Groq
from supabase import create_client, Client
from dotenv import load_dotenv
import random
import numpy as np

load_dotenv()

class StrategyLearningSystem:
    def __init__(self):
        self.groq_client = Groq()
        self.supabase: Client = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )
        
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

    def generate_enhanced_strategy(self, market_data, learning_insights=None):
        """Generate strategy with learning insights and diversification"""
        
        # Select random strategy template for diversification
        strategy_type = random.choice(self.strategy_templates)
        selected_indicators = random.sample(self.indicators, random.randint(2, 4))
        risk_style = random.choice(self.risk_styles)
        
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
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.7,
                max_tokens=2048,
            )
            
            strategy_content = response.choices[0].message.content
            
            # Enhanced strategy metadata
            strategy_data = {
                "strategy_type": strategy_type,
                "indicators": selected_indicators,
                "risk_style": risk_style,
                "generation_timestamp": datetime.now().isoformat(),
                "learning_version": "v2.0",
                "content": strategy_content
            }
            
            return strategy_data
            
        except Exception as e:
            print(f"Error generating strategy: {e}")
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
        
        print(f"\n🎯 Recursive learning complete! Generated {len(all_results)} strategies across {cycles} cycles.")
        return all_results

    def save_strategy_to_db(self, strategy, metrics, is_successful):
        """Save strategy to Supabase database"""
        try:
            record = {
                "symbol": "BTCUSDT",  # Default symbol
                "interval": "1day",   # Default interval
                "strategy_code": json.dumps(strategy),
                "metrics": metrics,
                "llm_notes": f"Generated by enhanced learning system v2.0 - {strategy.get('strategy_type', 'unknown')}",
                "is_successful": is_successful
            }
            
            response = self.supabase.table("trading-strategies").insert(record).execute()
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
                        "avg_winrate": avg_winrate
                    }
            
            return {"total_strategies": total_strategies, "successful_strategies": 0, "success_rate": 0}
            
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {}

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