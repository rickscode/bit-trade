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
from core.session_manager import SessionManager

def run_demo_trading(args):
    """Run demo trading with AlphaZero agent"""
    logger.info("🚀 Starting Binance Demo Trading with AlphaZero Agent")
    
    # Initialize session manager
    session_manager = SessionManager()
    
    # Create session configuration
    session_config = {
        'episodes': args.episodes,
        'max_steps': args.max_steps,
        'symbols': args.symbols,
        'mcts_sims': args.mcts_sims if not args.fast else 5,
        'temperature': args.temperature,
        'fast_mode': args.fast,
        'dashboard': args.dashboard
    }
    
    # Create new trading session
    session_id = session_manager.create_session(args.mode, session_config)
    
    # Initialize components
    llm_manager = MultiLLMManager()
    env = BinanceDemoEnv(
        symbols=args.symbols,
        testnet=True
    )
    # Optimize for fast mode
    mcts_sims = 5 if args.fast else args.mcts_sims
    
    agent = AlphaZeroTradingAgent(
        llm_manager=llm_manager,
        mcts_simulations=mcts_sims,
        temperature=args.temperature
    )
    
    logger.info(f"📊 Demo Trading Configuration:")
    logger.info(f"   Symbols: {args.symbols}")
    logger.info(f"   Real Account Balance: ${env.initial_balance:,.2f}")
    logger.info(f"   Episodes: {args.episodes}")
    logger.info(f"   MCTS Simulations: {args.mcts_sims}")
    
    total_returns = []
    
    try:
        for episode in range(args.episodes):
            logger.info(f"\n🎯 Episode {episode + 1}/{args.episodes}")
            
            # Update session progress
            session_manager.update_session(completed_episodes=episode)
            
            # Create checkpoint every 2 episodes
            if episode > 0 and episode % 2 == 0:
                agent_state = {
                    'experience_buffer_size': len(agent.experience_buffer),
                    'training_episodes': agent.training_episodes,
                    'avg_return': agent.avg_return,
                    'win_rate': agent.win_rate
                }
                env_state = {
                    'portfolio_value': env.portfolio_value,
                    'balance': env.balance,
                    'total_trades': len(env.trades)
                }
                session_manager.checkpoint_session(agent_state, env_state)
            
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
                
                # Log important steps with structured data
                if step % 100 == 0 or done:
                    portfolio_value = info.get('portfolio_value', 0)
                    portfolio_return = info.get('portfolio_return', 0)
                    action_name = ["HOLD", "BUY", "SELL"][action]
                    
                    # Structured logging for analysis
                    step_data = {
                        "episode": episode + 1,
                        "step": step,
                        "action": action,
                        "action_name": action_name,
                        "portfolio_value": portfolio_value,
                        "portfolio_return": portfolio_return,
                        "reward": reward,
                        "observation": observation.tolist() if hasattr(observation, 'tolist') else observation,
                        "market_context": env.get_detailed_portfolio_context(),
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Save to game history for AI reasoning analysis
                    agent.game_history.append(step_data)
                    
                    logger.info(f"   Step {step}: {action_name} | "
                              f"Portfolio: ${portfolio_value:,.2f} | "
                              f"Return: {portfolio_return:.2%} | "
                              f"Reward: {reward:.4f}")
                    
                    # Log structured data for analysis
                    logger.performance(step_data)
                
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
        logger.info("\n⏹️ Trading interrupted by user")
        session_manager.pause_session()
    
    except Exception as e:
        logger.error(f"❌ Trading session failed: {e}")
        session_manager.fail_session(str(e))
        raise
    
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
            "session_id": session_id,
            "episode_returns": total_returns,
            "avg_return": avg_return,
            "win_rate": win_rate,
            "training_stats": agent.get_training_stats(),
            "model_performance": llm_manager.get_performance_report(),
            "completed_episodes": len(total_returns),
            "total_return": avg_return
        }
        
        import json
        import os
        
        # Ensure outputs directory exists
        os.makedirs("outputs", exist_ok=True)
        
        try:
            with open("outputs/demo_trading_results.json", "w") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info("💾 Results saved to outputs/demo_trading_results.json")
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")
        
        # Complete session
        session_manager.complete_session(results)

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
    parser.add_argument("--mcts_sims", type=int, default=10, help="MCTS simulations per action (reduced from 25)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Action selection temperature")
    parser.add_argument("--fast", action="store_true", help="Fast mode: reduced MCTS sims and timeouts")
    parser.add_argument("--dashboard", action="store_true", help="Launch real-time dashboard")
    parser.add_argument("--resume", type=str, help="Resume session by session_id")
    parser.add_argument("--list_sessions", action="store_true", help="List available sessions")
    parser.add_argument("--session_stats", action="store_true", help="Show session statistics")
    
    args = parser.parse_args()
    
    # Handle session management commands
    if args.list_sessions or args.session_stats:
        session_manager = SessionManager()
        
        if args.list_sessions:
            print("\n📋 Available Trading Sessions:")
            sessions = session_manager.list_sessions(limit=20)
            
            if not sessions:
                print("   No sessions found")
            else:
                for session in sessions:
                    status_emoji = {
                        'completed': '✅',
                        'active': '🔄', 
                        'paused': '⏸️',
                        'failed': '❌'
                    }.get(session.get('status', 'unknown'), '❓')
                    
                    print(f"   {status_emoji} {session['session_id'][:20]}...")
                    print(f"      Status: {session.get('status', 'unknown')}")
                    print(f"      Started: {session.get('start_time', 'unknown')[:19]}")
                    print(f"      Episodes: {session.get('completed_episodes', 0)}/{session.get('episodes', 0)}")
                    print(f"      Return: {session.get('total_return', 0):.2%}")
                    print()
        
        if args.session_stats:
            print("\n📊 Session Statistics:")
            stats = session_manager.get_session_stats()
            
            for key, value in stats.items():
                if 'return' in key and isinstance(value, (int, float)):
                    print(f"   {key.replace('_', ' ').title()}: {value:.2%}")
                else:
                    print(f"   {key.replace('_', ' ').title()}: {value}")
        
        return
    
    # Launch dashboard if requested
    if args.dashboard and args.mode != "test":
        import subprocess
        import threading
        import time
        
        def launch_dashboard():
            try:
                subprocess.run(["streamlit", "run", "dashboard.py", "--server.port=5000", "--server.headless=true"])
            except Exception as e:
                print(f"❌ Dashboard launch failed: {e}")
        
        dashboard_thread = threading.Thread(target=launch_dashboard, daemon=True)
        dashboard_thread.start()
        print("🌐 Dashboard launching at http://localhost:5000")
        time.sleep(3)  # Give dashboard time to start
    
    print(f"🚀 Bit-Trade Simplified System Starting - Mode: {args.mode}")
    if args.fast:
        print("⚡ Fast mode enabled: Reduced MCTS simulations")
    if args.dashboard and args.mode != "test":
        print("🌐 Dashboard available at http://localhost:5000")
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