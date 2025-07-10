#!/usr/bin/env python3
"""
Demo script to showcase the Multi-LLM Reinforcement Learning System
This demo shows how the system works without requiring API keys
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add core modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

def demo_multi_llm_system():
    """Demo the multi-LLM system functionality"""
    
    print("🚀 Bit-Trade Multi-LLM Reinforcement Learning System Demo")
    print("=" * 60)
    
    # 1. Initialize Multi-LLM Manager
    print("\n🤖 Phase 1: Multi-LLM Manager Initialization")
    try:
        from multi_llm_manager import MultiLLMManager
        manager = MultiLLMManager()
        
        print(f"✅ Initialized with {len(manager.get_available_models())} AI models:")
        for i, model in enumerate(manager.get_available_models(), 1):
            info = manager.get_model_info(model)
            print(f"   {i}. {model}: {info.get('description', 'No description')}")
        
    except Exception as e:
        print(f"❌ Error initializing Multi-LLM Manager: {e}")
        return False
    
    # 2. Demo Round-Robin Model Selection
    print("\n🎯 Phase 2: Round-Robin Model Selection Demo")
    print("Simulating strategy generation with different models:")
    
    for i in range(8):
        selected_model = manager.select_model("strategy_generation", i)
        model_info = manager.get_model_info(selected_model)
        print(f"   Strategy {i+1}: {selected_model} - {model_info.get('strengths', ['general'])[0]}")
    
    # 3. Demo Performance Tracking
    print("\n📊 Phase 3: Performance Tracking System")
    
    # Simulate some strategy results
    test_strategies = [
        {"model": "versatile", "return": 12.5, "sharpe": 1.2, "success": True},
        {"model": "analytical", "return": 15.3, "sharpe": 1.4, "success": True},
        {"model": "maverick", "return": -2.1, "sharpe": -0.3, "success": False},
        {"model": "scout", "return": 8.7, "sharpe": 0.9, "success": True},
        {"model": "diverse", "return": 20.1, "sharpe": 1.8, "success": True}
    ]
    
    print("Simulating strategy performance updates...")
    for strategy in test_strategies:
        metrics = {
            "Total Return [%]": strategy["return"],
            "Sharpe Ratio": strategy["sharpe"]
        }
        manager.update_model_performance(strategy["model"], metrics, strategy["success"])
        status = "✅ Success" if strategy["success"] else "❌ Failed"
        print(f"   {strategy['model']}: {strategy['return']:.1f}% return, {strategy['sharpe']:.1f} Sharpe {status}")
    
    # 4. Demo Performance Report
    print("\n📈 Phase 4: Performance Report")
    performance_report = manager.get_performance_report()
    
    print("Model Performance Rankings:")
    for i, model_stats in enumerate(performance_report.get('model_rankings', []), 1):
        print(f"   {i}. {model_stats['model']}: "
              f"{model_stats['success_rate']:.1%} success rate, "
              f"{model_stats['avg_return']:.1f}% avg return")
    
    # 5. Demo Adaptive Selection
    print("\n🧠 Phase 5: Adaptive Model Selection")
    print("Testing different selection modes:")
    
    modes = ["round_robin", "performance_based", "weighted_random"]
    for mode in modes:
        manager.switch_selection_mode(mode)
        selected = manager.select_model("strategy_generation", 0)
        print(f"   Mode: {mode:<15} → Selected: {selected}")
    
    # 6. Demo Market Data Integration
    print("\n📊 Phase 6: Market Data Integration")
    try:
        data = pd.read_csv("data/BTCUSDT_1day.csv", parse_dates=["timestamp"], index_col="timestamp")
        
        print(f"✅ Loaded market data: {len(data)} days of BTCUSDT data")
        print(f"📅 Period: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        print(f"💰 Price range: ${data['low'].min():,.0f} - ${data['high'].max():,.0f}")
        print(f"📈 Final price: ${data['close'].iloc[-1]:,.0f}")
        
        # Calculate some basic metrics
        volatility = data['close'].pct_change().std()
        trend = (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100
        
        print(f"📊 Volatility: {volatility:.3f}")
        print(f"📈 Total trend: {trend:+.1f}%")
        
    except Exception as e:
        print(f"❌ Error loading market data: {e}")
    
    # 7. Demo Strategy Generation Framework
    print("\n🎯 Phase 7: Strategy Generation Framework")
    
    strategy_templates = [
        "momentum_based", "mean_reversion", "breakout_detection", 
        "trend_following", "volatility_trading", "support_resistance"
    ]
    
    indicators = ["RSI", "MACD", "Bollinger Bands", "Moving Average", "ADX"]
    risk_styles = ["fixed_percentage", "volatility_based", "kelly_criterion"]
    
    print("Available strategy components:")
    print(f"   📋 Strategy Types: {', '.join(strategy_templates)}")
    print(f"   📊 Indicators: {', '.join(indicators)}")
    print(f"   🛡️  Risk Styles: {', '.join(risk_styles)}")
    
    # 8. Demo Learning Insights
    print("\n🧠 Phase 8: Learning Insights Simulation")
    
    # Simulate learning patterns
    learning_patterns = {
        "common_indicators": {"RSI": 8, "MACD": 6, "MA": 10},
        "successful_combinations": ["RSI + MA", "MACD + Bollinger Bands"],
        "best_timeframes": ["1h", "4h", "1d"],
        "avg_successful_return": 12.4
    }
    
    print("Simulated learning insights from previous strategies:")
    print(f"   📊 Most used indicators: {', '.join(learning_patterns['common_indicators'].keys())}")
    print(f"   🏆 Best combinations: {', '.join(learning_patterns['successful_combinations'])}")
    print(f"   ⏰ Optimal timeframes: {', '.join(learning_patterns['best_timeframes'])}")
    print(f"   📈 Average successful return: {learning_patterns['avg_successful_return']:.1f}%")
    
    # 9. Demo Complete
    print("\n🎉 Demo Complete!")
    print("=" * 60)
    print("The Multi-LLM Reinforcement Learning System is ready!")
    print("\n🚀 Next steps:")
    print("1. Add your API keys to .env file")
    print("2. Run: python main.py --mode full --cycles 3 --strategies 5")
    print("3. Watch the system learn and adapt!")
    print("\n💡 Key Benefits:")
    print("   • 5x strategy diversity with different AI models")
    print("   • Reinforcement learning improves over time")
    print("   • Comprehensive backtesting and risk analysis")
    print("   • Fully autonomous operation")
    
    return True

if __name__ == "__main__":
    success = demo_multi_llm_system()
    sys.exit(0 if success else 1)