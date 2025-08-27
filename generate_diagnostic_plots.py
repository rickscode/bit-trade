#!/usr/bin/env python3
"""
Generate diagnostic plots for the Bit-Trade system
"""
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

def load_trading_data() -> Dict:
    """Load trading results and agent memory"""
    data = {}
    
    # Load trading results
    if os.path.exists("outputs/demo_trading_results.json"):
        with open("outputs/demo_trading_results.json", 'r') as f:
            data['trading_results'] = json.load(f)
    
    # Load agent memory
    if os.path.exists("outputs/agent_memory.json"):
        with open("outputs/agent_memory.json", 'r') as f:
            data['agent_memory'] = json.load(f)
    
    # Load model performance
    if os.path.exists("outputs/model_performance.json"):
        with open("outputs/model_performance.json", 'r') as f:
            data['model_performance'] = json.load(f)
    
    return data

def create_learning_curve_plot(data: Dict) -> str:
    """Create learning curve plot from experience buffer"""
    if 'agent_memory' not in data or not data['agent_memory'].get('experience_buffer'):
        return "No experience data available"
    
    experiences = data['agent_memory']['experience_buffer']
    rewards = [exp.get('reward', 0) for exp in experiences]
    episodes = list(range(1, len(rewards) + 1))
    
    plt.figure(figsize=(12, 8))
    
    # Individual episode rewards
    plt.subplot(2, 2, 1)
    plt.plot(episodes, rewards, 'o-', alpha=0.7, markersize=4)
    plt.title('Episode Rewards Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    
    # Cumulative rewards
    plt.subplot(2, 2, 2)
    cumulative_rewards = np.cumsum(rewards)
    plt.plot(episodes, cumulative_rewards, 'g-', linewidth=2)
    plt.title('Cumulative Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.grid(True, alpha=0.3)
    
    # Rolling average (window=5)
    plt.subplot(2, 2, 3)
    if len(rewards) >= 5:
        rolling_mean = pd.Series(rewards).rolling(window=5).mean()
        plt.plot(episodes, rolling_mean, 'r-', linewidth=2, label='5-episode average')
        plt.plot(episodes, rewards, 'b-', alpha=0.3, label='Individual episodes')
        plt.legend()
    else:
        plt.plot(episodes, rewards, 'b-', alpha=0.7)
    plt.title('Learning Trend (Rolling Average)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    
    # Reward distribution
    plt.subplot(2, 2, 4)
    plt.hist(rewards, bins=20, alpha=0.7, edgecolor='black')
    plt.title('Reward Distribution')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.axvline(np.mean(rewards), color='red', linestyle='--', label=f'Mean: {np.mean(rewards):.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs("reports/plots", exist_ok=True)
    filename = f"reports/plots/learning_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename

def create_model_performance_plot(data: Dict) -> str:
    """Create model performance comparison plot"""
    if 'model_performance' not in data or not data['model_performance']:
        return "No model performance data available"
    
    model_data = data['model_performance']
    
    plt.figure(figsize=(12, 10))
    
    # Model rankings if available
    if 'model_rankings' in model_data:
        rankings = model_data['model_rankings']
        models = [r['model'] for r in rankings]
        success_rates = [r.get('success_rate', 0) for r in rankings]
        avg_returns = [r.get('avg_return', 0) for r in rankings]
        strategies_generated = [r.get('strategies_generated', 0) for r in rankings]
        
        # Success rates
        plt.subplot(2, 2, 1)
        bars = plt.bar(models, success_rates, color='skyblue', edgecolor='navy', alpha=0.7)
        plt.title('Model Success Rates')
        plt.ylabel('Success Rate')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, success_rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.2f}', ha='center', va='bottom')
        
        # Average returns
        plt.subplot(2, 2, 2)
        colors = ['green' if r >= 0 else 'red' for r in avg_returns]
        bars = plt.bar(models, avg_returns, color=colors, alpha=0.7)
        plt.title('Average Returns by Model')
        plt.ylabel('Average Return')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Strategies generated
        plt.subplot(2, 2, 3)
        bars = plt.bar(models, strategies_generated, color='orange', alpha=0.7)
        plt.title('Strategies Generated by Model')
        plt.ylabel('Number of Strategies')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Model efficiency (return per strategy)
        plt.subplot(2, 2, 4)
        efficiency = []
        for i, model in enumerate(models):
            if strategies_generated[i] > 0:
                efficiency.append(avg_returns[i] / strategies_generated[i])
            else:
                efficiency.append(0)
        
        colors = ['green' if e >= 0 else 'red' for e in efficiency]
        bars = plt.bar(models, efficiency, color=colors, alpha=0.7)
        plt.title('Model Efficiency (Return per Strategy)')
        plt.ylabel('Return per Strategy')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs("reports/plots", exist_ok=True)
    filename = f"reports/plots/model_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename

def create_portfolio_analysis_plot(data: Dict) -> str:
    """Create portfolio performance analysis plot"""
    if 'trading_results' not in data:
        return "No trading results available"
    
    trading_results = data['trading_results']
    episode_returns = trading_results.get('episode_returns', [])
    
    if not episode_returns:
        return "No episode returns data available"
    
    plt.figure(figsize=(12, 10))
    
    # Portfolio growth
    plt.subplot(2, 2, 1)
    portfolio_values = []
    initial_value = 10000  # Assume starting with 10k
    current_value = initial_value
    portfolio_values.append(current_value)
    
    for ret in episode_returns:
        current_value *= (1 + ret)
        portfolio_values.append(current_value)
    
    episodes = list(range(len(portfolio_values)))
    plt.plot(episodes, portfolio_values, 'b-', linewidth=2, marker='o', markersize=4)
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True, alpha=0.3)
    
    # Return distribution
    plt.subplot(2, 2, 2)
    plt.hist(episode_returns, bins=20, alpha=0.7, edgecolor='black', color='lightgreen')
    plt.axvline(np.mean(episode_returns), color='red', linestyle='--', 
                label=f'Mean: {np.mean(episode_returns):.4f}')
    plt.axvline(np.median(episode_returns), color='orange', linestyle='--', 
                label=f'Median: {np.median(episode_returns):.4f}')
    plt.title('Episode Returns Distribution')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Drawdown analysis
    plt.subplot(2, 2, 3)
    peak = portfolio_values[0]
    drawdowns = []
    
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (value - peak) / peak
        drawdowns.append(drawdown)
    
    plt.plot(episodes, [d * 100 for d in drawdowns], 'r-', linewidth=2)
    plt.fill_between(episodes, [d * 100 for d in drawdowns], 0, alpha=0.3, color='red')
    plt.title('Portfolio Drawdown (%)')
    plt.xlabel('Episode')
    plt.ylabel('Drawdown (%)')
    plt.grid(True, alpha=0.3)
    
    # Risk-return analysis
    plt.subplot(2, 2, 4)
    if len(episode_returns) > 1:
        volatility = np.std(episode_returns)
        mean_return = np.mean(episode_returns)
        
        plt.scatter(volatility * 100, mean_return * 100, s=100, color='blue', alpha=0.7)
        plt.xlabel('Volatility (%)')
        plt.ylabel('Mean Return (%)')
        plt.title('Risk-Return Profile')
        
        # Add sharpe ratio info
        if volatility > 0:
            sharpe = mean_return / volatility
            plt.text(0.05, 0.95, f'Sharpe Ratio: {sharpe:.2f}', 
                    transform=plt.gca().transAxes, fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs("reports/plots", exist_ok=True)
    filename = f"reports/plots/portfolio_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename

def create_system_diagnostics_plot(data: Dict) -> str:
    """Create system diagnostics overview"""
    plt.figure(figsize=(12, 8))
    
    # Experience buffer size over time (if available)
    if 'agent_memory' in data and data['agent_memory'].get('experience_buffer'):
        experiences = data['agent_memory']['experience_buffer']
        buffer_sizes = list(range(1, len(experiences) + 1))
        
        plt.subplot(2, 2, 1)
        plt.plot(buffer_sizes, 'g-', linewidth=2)
        plt.title('Experience Buffer Growth')
        plt.xlabel('Episode')
        plt.ylabel('Total Experiences')
        plt.grid(True, alpha=0.3)
    else:
        plt.subplot(2, 2, 1)
        plt.text(0.5, 0.5, 'No experience data', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Experience Buffer Growth')
    
    # System status indicators
    plt.subplot(2, 2, 2)
    status_data = {}
    
    if 'trading_results' in data:
        status_data['Trading Results'] = 'Available' if data['trading_results'] else 'Empty'
    if 'agent_memory' in data:
        status_data['Agent Memory'] = 'Available' if data['agent_memory'] else 'Empty'
    if 'model_performance' in data:
        status_data['Model Performance'] = 'Available' if data['model_performance'] else 'Empty'
    
    if status_data:
        labels = list(status_data.keys())
        colors = ['green' if v == 'Available' else 'red' for v in status_data.values()]
        y_pos = range(len(labels))
        
        plt.barh(y_pos, [1] * len(labels), color=colors, alpha=0.7)
        plt.yticks(y_pos, labels)
        plt.xlabel('Status')
        plt.title('System Data Availability')
        
        # Add status text
        for i, (label, status) in enumerate(status_data.items()):
            plt.text(0.5, i, status, ha='center', va='center', fontweight='bold', color='white')
    
    # Memory usage (approximate)
    plt.subplot(2, 2, 3)
    memory_info = {
        'Experience Buffer': len(data.get('agent_memory', {}).get('experience_buffer', [])),
        'Trading Episodes': len(data.get('trading_results', {}).get('episode_returns', [])),
        'Model Rankings': len(data.get('model_performance', {}).get('model_rankings', []))
    }
    
    labels = list(memory_info.keys())
    sizes = list(memory_info.values())
    colors = ['lightblue', 'lightgreen', 'lightyellow']
    
    if sum(sizes) > 0:
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%', startangle=90)
        plt.title('Data Distribution')
    else:
        plt.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Data Distribution')
    
    # Learning phase indicator
    plt.subplot(2, 2, 4)
    total_experiences = len(data.get('agent_memory', {}).get('experience_buffer', []))
    
    if total_experiences < 10:
        phase = 'Initial'
        color = 'red'
    elif total_experiences < 50:
        phase = 'Growing'
        color = 'orange'
    else:
        phase = 'Experienced'
        color = 'green'
    
    confidence_level = min(total_experiences / 100.0, 1.0) * 100
    
    plt.pie([confidence_level, 100 - confidence_level], 
           labels=[f'{phase} Phase', 'Remaining'], 
           colors=[color, 'lightgray'],
           autopct='%1.1f%%', startangle=90)
    plt.title(f'Learning Phase\n({total_experiences} experiences)')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs("reports/plots", exist_ok=True)
    filename = f"reports/plots/system_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename

def main():
    """Generate all diagnostic plots"""
    print("🔄 Loading trading system data...")
    data = load_trading_data()
    
    generated_plots = []
    
    print("📊 Generating learning curve plot...")
    learning_plot = create_learning_curve_plot(data)
    if learning_plot != "No experience data available":
        generated_plots.append(learning_plot)
        print(f"✅ Created: {learning_plot}")
    else:
        print(f"⚠️ {learning_plot}")
    
    print("🤖 Generating model performance plot...")
    model_plot = create_model_performance_plot(data)
    if model_plot != "No model performance data available":
        generated_plots.append(model_plot)
        print(f"✅ Created: {model_plot}")
    else:
        print(f"⚠️ {model_plot}")
    
    print("💰 Generating portfolio analysis plot...")
    portfolio_plot = create_portfolio_analysis_plot(data)
    if portfolio_plot != "No trading results available" and portfolio_plot != "No episode returns data available":
        generated_plots.append(portfolio_plot)
        print(f"✅ Created: {portfolio_plot}")
    else:
        print(f"⚠️ {portfolio_plot}")
    
    print("🔧 Generating system diagnostics plot...")
    system_plot = create_system_diagnostics_plot(data)
    generated_plots.append(system_plot)
    print(f"✅ Created: {system_plot}")
    
    print(f"\n🎉 Generated {len(generated_plots)} diagnostic plots in reports/plots/")
    for plot in generated_plots:
        print(f"  • {plot}")
    
    return generated_plots

if __name__ == "__main__":
    main()