#!/usr/bin/env python3
"""
Test Cloudflare Workers AI Integration with Multi-LLM Trading System
"""

import os
import sys
sys.path.append('core')

import pandas as pd
import numpy as np
from datetime import datetime
from core.multi_llm_manager import MultiLLMManager
from core.cloudflare_ai_client import CloudflareAIClient

def test_cloudflare_client():
    """Test Cloudflare AI client independently"""
    print("🔮 Testing Cloudflare Workers AI Client")
    print("=" * 50)
    
    try:
        client = CloudflareAIClient()
        print(f"✅ Cloudflare client initialized")
        print(f"Available models: {client.get_available_models()}")
        
        # Test each available model
        for model_key in client.get_available_models():
            print(f"\n🧪 Testing {model_key} model...")
            try:
                content, metadata = client.generate_with_model(
                    "Generate a simple trading strategy for Bitcoin",
                    model_key,
                    max_tokens=50
                )
                print(f"✅ {model_key}: Generated {len(content)} characters")
                print(f"   Provider: {metadata['provider']}")
                print(f"   Model: {metadata['model_name']}")
            except Exception as e:
                print(f"❌ {model_key}: {e}")
                
        return True
        
    except Exception as e:
        print(f"❌ Cloudflare client test failed: {e}")
        return False

def test_multi_llm_integration():
    """Test integrated Multi-LLM manager with Cloudflare models"""
    print("\n🤖 Testing Multi-LLM Manager with Cloudflare Integration")
    print("=" * 60)
    
    try:
        manager = MultiLLMManager()
        
        # Show all available models
        available_models = manager.get_available_models()
        print(f"Available models: {available_models}")
        
        # Categorize by provider
        groq_models = []
        cloudflare_models = []
        
        for model_key in available_models:
            model_info = manager.get_model_info(model_key)
            if model_info.get('provider') == 'groq':
                groq_models.append(model_key)
            elif model_info.get('provider') == 'cloudflare':
                cloudflare_models.append(model_key)
        
        print(f"\n📊 Model Distribution:")
        print(f"   Groq models: {groq_models}")
        print(f"   Cloudflare models: {cloudflare_models}")
        
        # Test strategy generation with different models
        test_prompt = "Generate a momentum-based trading strategy for BTCUSDT with RSI and MACD indicators"
        
        print(f"\n🎯 Testing Strategy Generation:")
        for model_key in available_models[:6]:  # Test first 6 models
            try:
                print(f"\n🔧 Testing {model_key}...")
                content, metadata = manager.generate_with_model(
                    test_prompt,
                    model_key,
                    max_tokens=100
                )
                
                provider = metadata.get('provider', 'unknown')
                print(f"✅ {model_key} ({provider}): Generated {len(content)} characters")
                print(f"   Content preview: {content[:100]}...")
                
            except Exception as e:
                print(f"❌ {model_key}: {e}")
        
        # Test model selection modes
        print(f"\n🔄 Testing Model Selection Modes:")
        for mode in ["round_robin", "performance_based", "weighted_random"]:
            try:
                manager.switch_selection_mode(mode)
                selected = manager.select_model("strategy_generation")
                model_info = manager.get_model_info(selected)
                provider = model_info.get('provider', 'unknown')
                print(f"✅ {mode}: Selected {selected} ({provider})")
            except Exception as e:
                print(f"❌ {mode}: {e}")
        
        # Test Q-learning with market data
        print(f"\n🧠 Testing Q-Learning with Mixed Providers:")
        manager.enable_q_learning(True)
        
        # Create mock market data
        market_data = {
            'volatility': 0.025,
            'trend': 0.15,
            'recent_performance': 0.08
        }
        
        try:
            selected = manager.select_model(
                task_type="strategy_generation",
                market_data=market_data
            )
            model_info = manager.get_model_info(selected)
            provider = model_info.get('provider', 'unknown')
            print(f"✅ Q-learning selected: {selected} ({provider})")
        except Exception as e:
            print(f"❌ Q-learning selection failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Multi-LLM integration test failed: {e}")
        return False

def test_performance_tracking():
    """Test performance tracking across providers"""
    print("\n📊 Testing Performance Tracking Across Providers")
    print("=" * 50)
    
    try:
        manager = MultiLLMManager()
        
        # Simulate performance data for different models
        test_results = [
            ("versatile", {"Total Return [%]": 12.5, "Sharpe Ratio": 1.2}, True),
            ("reasoning", {"Total Return [%]": 15.8, "Sharpe Ratio": 1.5}, True),
            ("coder", {"Total Return [%]": 8.3, "Sharpe Ratio": 0.9}, False),
            ("analytical", {"Total Return [%]": 18.2, "Sharpe Ratio": 1.8}, True),
            ("questioner", {"Total Return [%]": 11.1, "Sharpe Ratio": 1.1}, True),
        ]
        
        # Update performance for each model
        for model_key, metrics, is_successful in test_results:
            if model_key in manager.get_available_models():
                manager.update_model_performance(model_key, metrics, is_successful)
                model_info = manager.get_model_info(model_key)
                provider = model_info.get('provider', 'unknown')
                status = "✅" if is_successful else "❌"
                print(f"{status} {model_key} ({provider}): {metrics['Total Return [%]']:.1f}% return")
        
        # Get performance report
        report = manager.get_performance_report()
        
        if report.get('model_rankings'):
            print(f"\n🏆 Top Performing Models:")
            for i, ranking in enumerate(report['model_rankings'][:5]):  # Top 5
                model_key = ranking['model']
                model_info = manager.get_model_info(model_key)
                provider = model_info.get('provider', 'unknown')
                success_rate = ranking['success_rate']
                avg_return = ranking['avg_return']
                print(f"   {i+1}. {model_key} ({provider}): {success_rate:.1%} success, {avg_return:.1f}% avg return")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance tracking test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    print("🔮 Cloudflare Workers AI Integration Test Suite")
    print("=" * 60)
    
    # Check environment variables
    required_vars = ["CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_AUTH_TOKEN"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        print("Please set these in your .env file:")
        print("CLOUDFLARE_ACCOUNT_ID=your-account-id")
        print("CLOUDFLARE_AUTH_TOKEN=your-auth-token")
        return
    
    # Run tests
    tests = [
        ("Cloudflare Client", test_cloudflare_client),
        ("Multi-LLM Integration", test_multi_llm_integration), 
        ("Performance Tracking", test_performance_tracking)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("🎯 Test Summary:")
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")
    
    if total_passed == len(results):
        print("🎉 All tests passed! Cloudflare integration is ready.")
    else:
        print("⚠️  Some tests failed. Check configuration and try again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()