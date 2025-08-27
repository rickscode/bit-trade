#!/usr/bin/env python3
"""
Simple test to see the real Binance trading in action
"""

import numpy as np
from core.binance_demo_env import BinanceDemoEnv

def main():
    print("Testing Real Binance Testnet Trading")
    print("=" * 50)
    
    # Create environment
    env = BinanceDemoEnv(symbols=["BTCUSDT"], initial_balance=10000.0)
    
    # Reset environment
    observation = env.reset()
    print(f"Initial observation shape: {observation.shape}")
    print(f"Initial portfolio value: ${env.portfolio_value:,.2f}")
    print(f"Initial balance: ${env.balance:,.2f}")
    
    # Show current market price
    if env.current_prices:
        price = env.current_prices.get("BTCUSDT", 0)
        print(f"Current BTC price: ${price:,.2f}")
    
    print("\nTesting 5 random actions...")
    for i in range(5):
        # Random action
        action = np.random.choice([0, 1, 2])  # 0=Hold, 1=Buy, 2=Sell
        action_name = ["HOLD", "BUY", "SELL"][action]
        
        print(f"\nStep {i+1}: {action_name}")
        
        # Execute action
        observation, reward, done, info = env.step(action)
        
        print(f"  Reward: {reward:.4f}")
        print(f"  Portfolio Value: ${info.get('portfolio_value', 0):,.2f}")
        print(f"  Balance: ${env.balance:,.2f}")
        
        # Show positions
        if env.positions:
            for symbol, pos in env.positions.items():
                if pos.quantity > 0:
                    current_price = env.current_prices.get(symbol, 0)
                    value = pos.quantity * current_price
                    print(f"  Position: {pos.quantity:.6f} {symbol} = ${value:,.2f}")
        
        if done:
            print("Episode completed!")
            break
    
    # Final summary
    summary = env.get_portfolio_summary()
    print(f"\nFinal Summary:")
    print(f"  Portfolio Value: ${summary['portfolio_value']:,.2f}")
    print(f"  Total Return: {summary['total_return']:.2%}")
    print(f"  Total Trades: {summary['total_trades']}")
    
    env.close()
    print("\nTest completed!")

if __name__ == "__main__":
    main()