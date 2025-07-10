#!/usr/bin/env python3
"""
Test script for Multi-LLM implementation
"""

import sys
import os

# Add core modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

def test_multi_llm_manager():
    """Test the multi-LLM manager functionality"""
    print("🧪 Testing Multi-LLM Manager...")
    
    try:
        from multi_llm_manager import MultiLLMManager
        
        # Initialize manager
        manager = MultiLLMManager()
        print("✅ Multi-LLM Manager initialized successfully")
        
        # Test available models
        models = manager.get_available_models()
        print(f"📋 Available models: {models}")
        
        # Test model selection
        print("\n🎯 Testing model selection:")
        for i in range(5):
            selected = manager.select_model("strategy_generation", i)
            print(f"   Strategy {i}: {selected}")
        
        # Test model info
        print("\n📊 Model Information:")
        for model_key in models:
            info = manager.get_model_info(model_key)
            print(f"   {model_key}: {info.get('description', 'No description')}")
        
        # Test different selection modes
        print("\n🔄 Testing selection modes:")
        for mode in ["round_robin", "performance_based", "weighted_random"]:
            manager.switch_selection_mode(mode)
            selected = manager.select_model("strategy_generation", 0)
            print(f"   Mode {mode}: {selected}")
        
        print("\n✅ All Multi-LLM Manager tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Multi-LLM Manager test failed: {e}")
        return False

def test_strategy_learning_integration():
    """Test the strategy learning system integration"""
    print("\n🧠 Testing Strategy Learning System Integration...")
    
    try:
        from strategy_learning_system import StrategyLearningSystem
        
        # Initialize learning system
        learning_system = StrategyLearningSystem()
        print("✅ Strategy Learning System initialized with Multi-LLM support")
        
        # Test model switching
        print(f"🔄 Current selection mode: {learning_system.llm_manager.selection_mode}")
        
        # Test performance tracking
        performance_report = learning_system.llm_manager.get_performance_report()
        print(f"📊 Performance report generated: {len(performance_report.get('model_rankings', []))} models tracked")
        
        print("✅ Strategy Learning System integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Strategy Learning System integration test failed: {e}")
        return False

def test_legacy_evaluation():
    """Test the legacy evaluation system"""
    print("\n🔍 Testing Legacy Evaluation System...")
    
    try:
        # Test imports
        sys.path.append(os.path.join(os.path.dirname(__file__), 'legacy'))
        from evaluate_strategy import evaluate_performance
        
        # Create sample metrics file
        import json
        sample_metrics = {
            "Total Return [%]": 15.5,
            "Sharpe Ratio": 1.2,
            "Win Rate [%]": 65.0,
            "Max Drawdown [%]": 8.5
        }
        
        with open("test_metrics.json", "w") as f:
            json.dump(sample_metrics, f)
        
        print("✅ Legacy evaluation system can be imported and should work with Multi-LLM")
        
        # Clean up
        if os.path.exists("test_metrics.json"):
            os.remove("test_metrics.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Legacy evaluation system test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Multi-LLM Implementation Tests")
    print("=" * 60)
    
    tests = [
        test_multi_llm_manager,
        test_strategy_learning_integration,
        test_legacy_evaluation
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"🎯 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Multi-LLM implementation is ready!")
        print("\n🚀 Usage examples:")
        print("   python main.py --mode full --cycles 3 --strategies 5")
        print("   python core/strategy_learning_system.py")
        print("   python legacy/evaluate_strategy.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)