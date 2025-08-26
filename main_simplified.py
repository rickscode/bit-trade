#!/usr/bin/env python3
"""
Bit-Trade: Simplified Autonomous Crypto Trading Agent
Main entry point for AlphaZero-style RL trading with Groq models and Binance demo trading.
"""

# Removed unused sys and os imports
import argparse
import time
import numpy as np
from datetime import datetime

# Core modules imported directly

from core.multi_llm_manager import MultiLLMManager
from core.binance_demo_env import BinanceDemoEnv
from core.alphazero_trading_agent import AlphaZeroTradingAgent
from core.enhanced_logger import logger

def run_demo_trading(args):
    """Run demo trading with AlphaZero agent"""
    logger.info("🚀 Starting Binance Demo Trading with AlphaZero Agent")
    
    # Initialize components
    llm_manager = MultiLLMManager()
    env = BinanceDemoEnv(
        symbols=args.symbols,
        initial_balance=args.balance,
        testnet=True
    )
    agent = AlphaZeroTradingAgent(
        llm_manager=llm_manager,
        mcts_simulations=args.mcts_sims,
        temperature=args.temperature
    )
    
    logger.info(f"📊 Demo Trading Configuration:")
    logger.info(f"   Symbols: {args.symbols}")
    logger.info(f"   Initial Balance: ${args.balance:,.2f}")
    logger.info(f"   Episodes: {args.episodes}")
    logger.info(f"   MCTS Simulations: {args.mcts_sims}")
    
    total_returns = []
    
    try:
        for episode in range(args.episodes):
            logger.info(f"\n🎯 Episode {episode + 1}/{args.episodes}")
            
            # Reset environment
            observation = env.reset()
            done = False
            step = 0
            episode_reward = 0
            
            while not done and step < args.max_steps:
                # Agent selects action using MCTS + LLM ensemble
                action = agent.select_action(env, observation)
                
                # Execute action in environment
                next_observation, reward, done, info = env.step(action)
                
                episode_reward += reward
                step += 1
                
                # Log important steps
                if step % 100 == 0 or done:
                    portfolio_value = info.get('portfolio_value', 0)
                    portfolio_return = info.get('portfolio_return', 0)
                    action_name = ["HOLD", "BUY", "SELL"][action]
                    
                    logger.info(f"   Step {step}: {action_name} | "
                              f"Portfolio: ${portfolio_value:,.2f} | "
                              f"Return: {portfolio_return:.2%} | "
                              f"Reward: {reward:.4f}")
                
                observation = next_observation
                
                # Small delay to avoid overwhelming APIs
                time.sleep(0.1)
            
            # Episode completed
            final_summary = env.get_portfolio_summary()
            final_return = final_summary['total_return']
            total_returns.append(final_return)
            
            logger.info(f"\n📊 Episode {episode + 1} Summary:")
            logger.info(f"   Final Portfolio Value: ${final_summary['portfolio_value']:,.2f}")
            logger.info(f"   Total Return: {final_return:.2%}")
            logger.info(f"   Total Trades: {final_summary['total_trades']}")
            logger.info(f"   Steps Taken: {step}")
            
            # Train agent from episode data
            agent.train_from_episode(agent.game_history)
            agent.game_history = []  # Reset for next episode
    
    except KeyboardInterrupt:
        logger.info("\n⏹️ Training interrupted by user")
    
    finally:
        env.close()
    
    # Final statistics
    if total_returns:
        avg_return = np.mean(total_returns)
        win_rate = sum(1 for r in total_returns if r > 0) / len(total_returns)
        
        logger.info(f"\n🎯 Final Results ({len(total_returns)} episodes):")
        logger.info(f"   Average Return: {avg_return:.2%}")
        logger.info(f"   Win Rate: {win_rate:.2%}")
        logger.info(f"   Best Episode: {max(total_returns):.2%}")
        logger.info(f"   Worst Episode: {min(total_returns):.2%}")
        
        # Save results
        results = {
            "episode_returns": total_returns,
            "avg_return": avg_return,
            "win_rate": win_rate,
            "training_stats": agent.get_training_stats(),
            "model_performance": llm_manager.get_performance_report()
        }
        
        import json
        with open("outputs/demo_trading_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info("💾 Results saved to outputs/demo_trading_results.json")

def test_system_components():
    """Test system components individually"""
    logger.info("🧪 Testing System Components")
    
    # Test 1: Groq Models
    logger.info("\n1️⃣ Testing Groq Models...")
    try:
        llm_manager = MultiLLMManager()
        available_models = [m for m, available in llm_manager.available_models.items() if available]
        logger.info(f"✅ Available models: {available_models}")
        
        # Test ensemble generation
        test_prompt = "Analyze current BTC market conditions for trading."
        consensus = llm_manager.generate_ensemble_consensus(test_prompt, num_models=3)
        logger.info(f"✅ Ensemble consensus generated with {consensus['ensemble_size']} models")
        
    except Exception as e:
        logger.error(f"❌ Groq models test failed: {e}")
    
    # Test 2: Demo Environment
    logger.info("\n2️⃣ Testing Demo Environment...")
    try:
        env = BinanceDemoEnv(symbols=["BTCUSDT"], initial_balance=10000.0)
        obs = env.reset()
        logger.info(f"✅ Environment initialized, observation shape: {obs.shape}")
        
        # Test a few steps
        for i in range(3):
            action = np.random.choice([0, 1, 2])
            obs, reward, _, _ = env.step(action)
            logger.info(f"   Step {i+1}: Action={action}, Reward={reward:.4f}")
        
        summary = env.get_portfolio_summary()
        logger.info(f"✅ Portfolio tracking working: ${summary['portfolio_value']:,.2f}")
        env.close()
        
    except Exception as e:
        logger.error(f"❌ Demo environment test failed: {e}")
    
    # Test 3: AlphaZero Agent
    logger.info("\n3️⃣ Testing AlphaZero Agent...")
    try:
        llm_manager = MultiLLMManager()
        agent = AlphaZeroTradingAgent(llm_manager, mcts_simulations=5)  # Reduced for testing
        
        # Create mock environment state
        mock_env = BinanceDemoEnv(symbols=["BTCUSDT"], initial_balance=10000.0)
        mock_obs = mock_env.reset()
        
        # Test action selection
        action = agent.select_action(mock_env, mock_obs)
        logger.info(f"✅ Agent selected action: {action} ({['HOLD', 'BUY', 'SELL'][action]})")
        
        stats = agent.get_training_stats()
        logger.info(f"✅ Training stats accessible: {stats}")
        
        mock_env.close()
        
    except Exception as e:
        logger.error(f"❌ AlphaZero agent test failed: {e}")
    
    logger.info("\n✅ Component testing completed")

def main():
    parser = argparse.ArgumentParser(description="Bit-Trade: Simplified Autonomous Crypto Trading Agent")
    parser.add_argument("--mode", choices=["demo", "test", "train"], 
                       default="demo", help="Operation mode")
    parser.add_argument("--episodes", type=int, default=5, help="Number of trading episodes")
    parser.add_argument("--max_steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT"], help="Trading symbols")
    parser.add_argument("--balance", type=float, default=10000.0, help="Initial balance")
    parser.add_argument("--mcts_sims", type=int, default=25, help="MCTS simulations per action")
    parser.add_argument("--temperature", type=float, default=1.0, help="Action selection temperature")
    
    args = parser.parse_args()
    
    print(f"🚀 Bit-Trade Simplified System Starting - Mode: {args.mode}")
    print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    
    if args.mode == "test":
        test_system_components()
    elif args.mode in ["demo", "train"]:
        run_demo_trading(args)
    
    print("\n🎯 Bit-Trade System Complete!")
    print("💡 Check the outputs/ directory for detailed results")

if __name__ == "__main__":
    main()