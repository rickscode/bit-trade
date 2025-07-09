import json
import pandas as pd
import numpy as np
import vectorbt as vbt
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

class EnhancedBacktestSystem:
    def __init__(self):
        self.default_params = {
            "initial_cash": 10000,
            "commission": 0.001,  # 0.1% commission
            "slippage": 0.0005,   # 0.05% slippage
            "min_trade_size": 10,  # Minimum trade size
        }
        
        # Different market conditions for robust testing
        self.market_conditions = [
            "bull_market", "bear_market", "sideways_market", 
            "high_volatility", "low_volatility", "crisis_period"
        ]

    def load_strategy_from_json(self, strategy_path):
        """Load strategy from JSON file"""
        try:
            with open(strategy_path, 'r') as f:
                strategy_data = json.load(f)
            return strategy_data
        except Exception as e:
            print(f"Error loading strategy: {e}")
            return None

    def parse_strategy_signals(self, strategy_data, price_data):
        """Parse strategy and generate trading signals"""
        df = price_data.copy()
        
        # Add technical indicators
        df['MA_20'] = df['close'].rolling(20).mean()
        df['MA_50'] = df['close'].rolling(50).mean()
        df['MA_200'] = df['close'].rolling(200).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        
        # Generate signals based on strategy type
        strategy_type = strategy_data.get('strategy_type', 'momentum_based')
        
        if strategy_type == 'momentum_based':
            entries = (df['MA_20'] > df['MA_50']) & (df['RSI'] < 70) & (df['MACD'] > df['MACD_signal'])
            exits = (df['MA_20'] < df['MA_50']) | (df['RSI'] > 80) | (df['MACD'] < df['MACD_signal'])
            
        elif strategy_type == 'mean_reversion':
            entries = (df['close'] < df['BB_lower']) & (df['RSI'] < 30)
            exits = (df['close'] > df['BB_upper']) | (df['RSI'] > 70)
            
        elif strategy_type == 'breakout_detection':
            df['high_20'] = df['high'].rolling(20).max()
            df['low_20'] = df['low'].rolling(20).min()
            entries = (df['close'] > df['high_20'].shift(1)) & (df['volume'] > df['volume'].rolling(20).mean())
            exits = (df['close'] < df['low_20'].shift(1)) | (df['RSI'] > 80)
            
        elif strategy_type == 'trend_following':
            entries = (df['MA_20'] > df['MA_50']) & (df['MA_50'] > df['MA_200']) & (df['RSI'] > 50)
            exits = (df['MA_20'] < df['MA_50']) | (df['RSI'] < 40)
            
        else:
            # Default momentum strategy
            entries = (df['MA_20'] > df['MA_50']) & (df['RSI'] < 70)
            exits = (df['MA_20'] < df['MA_50']) | (df['RSI'] > 80)
        
        return entries, exits, df

    def calculate_position_size(self, strategy_data, current_price, portfolio_value):
        """Calculate position size based on risk management"""
        risk_style = strategy_data.get('risk_style', 'fixed_percentage')
        
        if risk_style == 'fixed_percentage':
            risk_per_trade = 0.02  # 2% risk per trade
            position_size = (portfolio_value * risk_per_trade) / current_price
            
        elif risk_style == 'volatility_based':
            # Use ATR for position sizing (simplified)
            risk_per_trade = 0.015  # 1.5% for volatility-based
            position_size = (portfolio_value * risk_per_trade) / current_price
            
        elif risk_style == 'kelly_criterion':
            # Simplified Kelly criterion
            win_rate = 0.6  # Assumed win rate
            avg_win = 0.05  # Assumed average win
            avg_loss = 0.03  # Assumed average loss
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0, min(0.25, kelly_fraction))  # Cap at 25%
            position_size = (portfolio_value * kelly_fraction) / current_price
            
        else:
            # Default to 2% risk
            position_size = (portfolio_value * 0.02) / current_price
        
        return max(self.default_params["min_trade_size"], position_size)

    def run_comprehensive_backtest(self, strategy_data, price_data, test_period=None):
        """Run comprehensive backtest with multiple scenarios"""
        
        if test_period:
            price_data = price_data.loc[test_period[0]:test_period[1]]
        
        # Generate signals
        entries, exits, enhanced_df = self.parse_strategy_signals(strategy_data, price_data)
        
        # Run basic backtest
        portfolio = vbt.Portfolio.from_signals(
            close=enhanced_df['close'],
            entries=entries,
            exits=exits,
            size=1000,  # Fixed size for initial test
            fees=self.default_params["commission"],
            slippage=self.default_params["slippage"],
            freq='1D'
        )
        
        # Calculate comprehensive metrics
        metrics = self.calculate_comprehensive_metrics(portfolio, enhanced_df)
        
        # Add trade analysis
        trade_analysis = self.analyze_trades(portfolio)
        metrics.update(trade_analysis)
        
        # Risk analysis
        risk_metrics = self.calculate_risk_metrics(portfolio, enhanced_df)
        metrics.update(risk_metrics)
        
        return metrics, portfolio

    def calculate_comprehensive_metrics(self, portfolio, price_data):
        """Calculate comprehensive performance metrics"""
        
        # Basic metrics
        total_return = portfolio.total_return()
        sharpe_ratio = portfolio.sharpe_ratio()
        max_drawdown = portfolio.max_drawdown()
        
        # Advanced metrics
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
        sortino_ratio = portfolio.sortino_ratio()
        
        # Trade statistics
        trades = portfolio.trades.records_readable
        total_trades = len(trades)
        winning_trades = len(trades[trades['PnL'] > 0]) if total_trades > 0 else 0
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = trades[trades['PnL'] > 0]['PnL'].sum() if winning_trades > 0 else 0
        gross_loss = abs(trades[trades['PnL'] < 0]['PnL'].sum()) if total_trades > winning_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average trade metrics
        avg_trade_return = trades['PnL'].mean() if total_trades > 0 else 0
        avg_winning_trade = trades[trades['PnL'] > 0]['PnL'].mean() if winning_trades > 0 else 0
        avg_losing_trade = trades[trades['PnL'] < 0]['PnL'].mean() if total_trades > winning_trades else 0
        
        # Duration metrics
        avg_trade_duration = trades['Duration'].mean() if total_trades > 0 else 0
        
        return {
            "Start": str(portfolio.wrapper.index[0]),
            "End": str(portfolio.wrapper.index[-1]),
            "Total Return [%]": total_return * 100,
            "Sharpe Ratio": sharpe_ratio,
            "Sortino Ratio": sortino_ratio,
            "Calmar Ratio": calmar_ratio,
            "Max Drawdown [%]": max_drawdown * 100,
            "Win Rate [%]": win_rate,
            "Total Trades": total_trades,
            "Profit Factor": profit_factor,
            "Avg Trade Return": avg_trade_return,
            "Avg Winning Trade": avg_winning_trade,
            "Avg Losing Trade": avg_losing_trade,
            "Avg Trade Duration": avg_trade_duration,
            "Gross Profit": gross_profit,
            "Gross Loss": gross_loss
        }

    def analyze_trades(self, portfolio):
        """Analyze individual trades for patterns"""
        trades = portfolio.trades.records_readable
        
        if len(trades) == 0:
            return {"Trade Analysis": "No trades executed"}
        
        # Consecutive wins/losses
        pnl_series = trades['PnL']
        wins = pnl_series > 0
        
        # Find consecutive streaks
        streaks = []
        current_streak = 1
        current_type = wins.iloc[0]
        
        for i in range(1, len(wins)):
            if wins.iloc[i] == current_type:
                current_streak += 1
            else:
                streaks.append((current_type, current_streak))
                current_streak = 1
                current_type = wins.iloc[i]
        streaks.append((current_type, current_streak))
        
        # Calculate streak statistics
        win_streaks = [s[1] for s in streaks if s[0]]
        loss_streaks = [s[1] for s in streaks if not s[0]]
        
        max_consecutive_wins = max(win_streaks) if win_streaks else 0
        max_consecutive_losses = max(loss_streaks) if loss_streaks else 0
        
        # Monthly performance
        trades['Entry_Date'] = pd.to_datetime(trades['Entry Timestamp'])
        trades['Month'] = trades['Entry_Date'].dt.to_period('M')
        monthly_pnl = trades.groupby('Month')['PnL'].sum()
        
        profitable_months = len(monthly_pnl[monthly_pnl > 0])
        total_months = len(monthly_pnl)
        monthly_win_rate = (profitable_months / total_months * 100) if total_months > 0 else 0
        
        return {
            "Max Consecutive Wins": max_consecutive_wins,
            "Max Consecutive Losses": max_consecutive_losses,
            "Monthly Win Rate [%]": monthly_win_rate,
            "Best Month": monthly_pnl.max(),
            "Worst Month": monthly_pnl.min(),
            "Avg Monthly Return": monthly_pnl.mean()
        }

    def calculate_risk_metrics(self, portfolio, price_data):
        """Calculate risk-adjusted metrics"""
        
        # Value at Risk (VaR)
        returns = portfolio.returns()
        if len(returns) > 0:
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            # Conditional VaR (Expected Shortfall)
            cvar_95 = returns[returns <= var_95].mean()
            cvar_99 = returns[returns <= var_99].mean()
        else:
            var_95 = var_99 = cvar_95 = cvar_99 = 0
        
        # Ulcer Index
        running_max = portfolio.value().expanding().max()
        drawdown = (portfolio.value() - running_max) / running_max
        ulcer_index = np.sqrt((drawdown ** 2).mean())
        
        # Tail ratio
        returns_pos = returns[returns > 0]
        returns_neg = returns[returns < 0]
        
        if len(returns_pos) > 0 and len(returns_neg) > 0:
            tail_ratio = np.percentile(returns_pos, 95) / abs(np.percentile(returns_neg, 5))
        else:
            tail_ratio = 0
        
        return {
            "VaR 95%": var_95,
            "VaR 99%": var_99,
            "CVaR 95%": cvar_95,
            "CVaR 99%": cvar_99,
            "Ulcer Index": ulcer_index,
            "Tail Ratio": tail_ratio
        }

    def run_walk_forward_analysis(self, strategy_data, price_data, train_period=252, test_period=63):
        """Run walk-forward analysis for robustness testing"""
        
        results = []
        total_periods = len(price_data)
        
        for i in range(train_period, total_periods - test_period, test_period):
            # Define training and testing periods
            train_start = i - train_period
            train_end = i
            test_start = i
            test_end = i + test_period
            
            # Test data
            test_data = price_data.iloc[test_start:test_end]
            
            # Run backtest on test period
            metrics, _ = self.run_comprehensive_backtest(
                strategy_data, 
                test_data,
                test_period=(test_data.index[0], test_data.index[-1])
            )
            
            metrics['Period_Start'] = str(test_data.index[0])
            metrics['Period_End'] = str(test_data.index[-1])
            results.append(metrics)
        
        # Aggregate walk-forward results
        if results:
            df_results = pd.DataFrame(results)
            
            avg_return = df_results['Total Return [%]'].mean()
            avg_sharpe = df_results['Sharpe Ratio'].mean()
            avg_win_rate = df_results['Win Rate [%]'].mean()
            consistency = (df_results['Total Return [%]'] > 0).mean() * 100
            
            return {
                "Walk_Forward_Avg_Return": avg_return,
                "Walk_Forward_Avg_Sharpe": avg_sharpe,
                "Walk_Forward_Avg_Win_Rate": avg_win_rate,
                "Walk_Forward_Consistency": consistency,
                "Walk_Forward_Periods": len(results)
            }
        else:
            return {"Walk_Forward_Analysis": "Insufficient data"}

    def monte_carlo_simulation(self, strategy_data, price_data, num_simulations=1000):
        """Run Monte Carlo simulation for strategy robustness"""
        
        # Generate random entry/exit points
        simulation_results = []
        
        for _ in range(num_simulations):
            # Add random noise to prices
            noise_factor = np.random.normal(0, 0.01, len(price_data))
            noisy_prices = price_data.copy()
            noisy_prices['close'] *= (1 + noise_factor)
            
            # Run backtest
            try:
                metrics, _ = self.run_comprehensive_backtest(strategy_data, noisy_prices)
                simulation_results.append(metrics['Total Return [%]'])
            except:
                continue
        
        if simulation_results:
            return {
                "Monte_Carlo_Mean_Return": np.mean(simulation_results),
                "Monte_Carlo_Std_Return": np.std(simulation_results),
                "Monte_Carlo_VaR_95": np.percentile(simulation_results, 5),
                "Monte_Carlo_Probability_Positive": (np.array(simulation_results) > 0).mean() * 100
            }
        else:
            return {"Monte_Carlo_Analysis": "Failed to run simulations"}

    def generate_comprehensive_report(self, strategy_data, price_data, output_path="backtest_report.json"):
        """Generate comprehensive backtest report"""
        
        print("Running comprehensive backtest...")
        
        # Main backtest
        main_metrics, portfolio = self.run_comprehensive_backtest(strategy_data, price_data)
        
        # Walk-forward analysis
        print("Running walk-forward analysis...")
        wf_metrics = self.run_walk_forward_analysis(strategy_data, price_data)
        main_metrics.update(wf_metrics)
        
        # Monte Carlo simulation
        print("Running Monte Carlo simulation...")
        mc_metrics = self.monte_carlo_simulation(strategy_data, price_data)
        main_metrics.update(mc_metrics)
        
        # Add metadata
        main_metrics.update({
            "Strategy_Type": strategy_data.get('strategy_type', 'unknown'),
            "Indicators_Used": strategy_data.get('indicators', []),
            "Risk_Style": strategy_data.get('risk_style', 'unknown'),
            "Backtest_Timestamp": datetime.now().isoformat(),
            "Data_Points": len(price_data),
            "Backtest_Version": "enhanced_v2.0"
        })
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(main_metrics, f, indent=2, default=str)
        
        print(f"Comprehensive backtest report saved to {output_path}")
        return main_metrics

if __name__ == "__main__":
    # Example usage
    backtest_system = EnhancedBacktestSystem()
    
    # Load strategy and data
    strategy_data = {"strategy_type": "momentum_based", "indicators": ["RSI", "MACD", "MA"], "risk_style": "fixed_percentage"}
    price_data = pd.read_csv("data/BTCUSDT_1day.csv", parse_dates=["timestamp"], index_col="timestamp")
    
    # Run comprehensive backtest
    results = backtest_system.generate_comprehensive_report(strategy_data, price_data)
    
    print("✅ Enhanced backtesting complete!")
    print(f"Total Return: {results.get('Total Return [%]', 0):.2f}%")
    print(f"Sharpe Ratio: {results.get('Sharpe Ratio', 0):.2f}")
    print(f"Win Rate: {results.get('Win Rate [%]', 0):.2f}%")