#!/usr/bin/env python3
"""
Bit-Trade: Autonomous Crypto Trading Agent
Main entry point for the enhanced trading system with recursive learning.
"""

import sys
import os
import argparse
from datetime import datetime

# Add core modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from strategy_learning_system import StrategyLearningSystem
from enhanced_data_collector import EnhancedDataCollector
from enhanced_backtest_system import EnhancedBacktestSystem

def main():
    parser = argparse.ArgumentParser(description="Bit-Trade: Autonomous Crypto Trading Agent")
    parser.add_argument("--mode", choices=["collect", "learn", "backtest", "full"], 
                       default="full", help="Operation mode")
    parser.add_argument("--cycles", type=int, default=3, help="Number of learning cycles")
    parser.add_argument("--strategies", type=int, default=5, help="Strategies per cycle")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT"], help="Trading symbols")
    parser.add_argument("--timeframes", nargs="+", default=["1d"], help="Timeframes")
    
    args = parser.parse_args()
    
    print(f"🚀 Bit-Trade Enhanced System Starting - Mode: {args.mode}")
    print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    
    if args.mode in ["collect", "full"]:
        print("📊 Phase 1: Enhanced Data Collection")
        collector = EnhancedDataCollector()
        collector.export_training_datasets()
        print("✅ Data collection complete!\n")
    
    if args.mode in ["learn", "full"]:
        print("🧠 Phase 2: Recursive Learning System")
        learning_system = StrategyLearningSystem()
        
        # Load market data
        import pandas as pd
        market_data = pd.read_csv("data/BTCUSDT_1day.csv", parse_dates=["timestamp"], index_col="timestamp")
        
        # Run learning cycles
        results = learning_system.recursive_learning_cycle(
            market_data, 
            cycles=args.cycles, 
            strategies_per_cycle=args.strategies
        )
        
        # Show statistics
        stats = learning_system.get_learning_statistics()
        print(f"\n📈 Learning Results:")
        print(f"   Total Strategies: {stats.get('total_strategies', 0)}")
        print(f"   Successful: {stats.get('successful_strategies', 0)}")
        print(f"   Success Rate: {stats.get('success_rate', 0):.2f}%")
        
        # Show model performance
        model_perf = stats.get('model_performance', {})
        if model_perf.get('model_rankings'):
            print(f"\n🤖 Model Performance Rankings:")
            for i, model_stats in enumerate(model_perf['model_rankings'][:3]):  # Top 3
                print(f"   {i+1}. {model_stats['model']}: {model_stats['success_rate']:.2f}% success, {model_stats['avg_return']:.2f}% avg return")
        
        print("✅ Learning phase complete!\n")
    
    if args.mode in ["backtest", "full"]:
        print("🔬 Phase 3: Enhanced Backtesting")
        backtest_system = EnhancedBacktestSystem()
        
        # Example backtest with sample strategy
        strategy_data = {
            "strategy_type": "momentum_based",
            "indicators": ["RSI", "MACD", "MA"],
            "risk_style": "fixed_percentage"
        }
        
        import pandas as pd
        market_data = pd.read_csv("data/BTCUSDT_1day.csv", parse_dates=["timestamp"], index_col="timestamp")
        
        results = backtest_system.generate_comprehensive_report(
            strategy_data, 
            market_data, 
            "outputs/comprehensive_backtest_report.json"
        )
        
        print(f"📊 Backtest Results:")
        print(f"   Total Return: {results.get('Total Return [%]', 0):.2f}%")
        print(f"   Sharpe Ratio: {results.get('Sharpe Ratio', 0):.2f}")
        print(f"   Win Rate: {results.get('Win Rate [%]', 0):.2f}%")
        print("✅ Backtesting complete!\n")
    
    print("🎯 Bit-Trade Enhanced System Complete!")
    print("💡 Check the outputs/ directory for detailed results")

if __name__ == "__main__":
    main()