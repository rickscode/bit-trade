# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the complete enhanced system
python main.py --mode full --cycles 3 --strategies 5

# Or run individual phases
python main.py --mode collect    # Data collection only
python main.py --mode learn      # Learning system only
python main.py --mode backtest   # Backtesting only
```

## Environment Setup

Required environment variables in `.env`:
- `GROQ_API_KEY`: For LLM strategy generation and evaluation
- `SUPABASE_URL` and `SUPABASE_KEY`: For database operations
- `BINANCE_API_KEY` and `BINANCE_API_SECRET`: For market data fetching

## Repository Structure

```
bit-trade/
├── main.py                     # Main entry point for enhanced system
├── core/                       # Enhanced core modules
│   ├── strategy_learning_system.py    # Recursive learning system
│   ├── enhanced_data_collector.py     # Multi-asset data collection
│   └── enhanced_backtest_system.py    # Comprehensive backtesting
├── legacy/                     # Legacy modules (for reference)
│   ├── data_fetch.py          # Basic data fetching
│   ├── backtest_strategy.py   # Basic backtesting
│   ├── evaluate_strategy.py   # Basic evaluation
│   └── save_to_supabase.py    # Basic database save
├── data/                       # Raw market data (CSV)
├── strategies/                 # Generated trading strategies
├── outputs/                    # System outputs and reports
├── venv/                      # Virtual environment
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Enhanced System Architecture

### 1. Enhanced Data Collection (`core/enhanced_data_collector.py`)
- **Multi-Asset Support**: 10+ cryptocurrency pairs
- **Multi-Timeframe**: 1m to 1w intervals
- **Market Regime Detection**: Bull/bear/sideways classification
- **50+ Technical Indicators**: RSI, MACD, Bollinger Bands, etc.
- **Synthetic Scenarios**: Crash, bubble, recovery simulations

### 2. Recursive Learning System (`core/strategy_learning_system.py`)
- **Strategy Diversification**: 8 strategy types (momentum, mean reversion, etc.)
- **Learning from Success**: Analyzes profitable strategies for patterns
- **Batch Generation**: Creates multiple strategies per cycle
- **Continuous Improvement**: Each cycle builds on previous successes
- **Performance Tracking**: Monitors success rates and learning progress

### 3. Enhanced Backtesting (`core/enhanced_backtest_system.py`)
- **Comprehensive Metrics**: 25+ performance indicators
- **Walk-Forward Analysis**: Tests across different time periods
- **Monte Carlo Simulation**: 1000+ random scenario tests
- **Risk Analysis**: VaR, CVaR, Ulcer Index, Tail Ratio
- **Trade Pattern Analysis**: Streak detection, monthly performance

## Core Development Commands

```bash
# Collect comprehensive training data
python core/enhanced_data_collector.py

# Run recursive learning cycles
python core/strategy_learning_system.py

# Generate comprehensive backtest reports
python core/enhanced_backtest_system.py

# Legacy commands (still functional)
python legacy/data_fetch.py
python legacy/backtest_strategy.py
python legacy/evaluate_strategy.py
```

## Database Schema

```sql
create table "trading-strategies" (
  id uuid primary key default uuid_generate_v4(),
  symbol text not null,
  interval text not null,
  strategy_code text not null,
  metrics jsonb not null,
  created_at timestamp with time zone default now(),
  llm_notes text,
  is_successful boolean default false
);
```

## Key Features

### Recursive Learning Process
1. **Analyze**: Review successful strategies in database
2. **Learn**: Extract patterns from profitable trades
3. **Generate**: Create diverse new strategies using insights
4. **Test**: Comprehensive backtesting with multiple scenarios
5. **Evolve**: Save winners and iterate with improved knowledge

### Multi-Asset & Multi-Timeframe Support
- **Symbols**: BTCUSDT, ETHUSDT, ADAUSDT, DOTUSDT, LINKUSDT, etc.
- **Timeframes**: 1m, 5m, 15m, 30m, 1h, 4h, 12h, 1d, 3d, 1w
- **Market Regimes**: Bull/bear stable/volatile, sideways markets

### Advanced Risk Management
- **Position Sizing**: Fixed percentage, volatility-based, Kelly criterion
- **Stop Loss**: Trailing, volatility-based, time-based exits
- **Risk Metrics**: Sharpe, Sortino, Calmar ratios, drawdown analysis

## Performance Expectations

- **Volume**: 50+ strategies per day vs. 1-2 with legacy system
- **Diversity**: 8 strategy types across multiple assets/timeframes
- **Quality**: Improving success rate through recursive learning
- **Robustness**: Strategies tested across multiple market scenarios

## Development Notes

- System designed for continuous autonomous operation
- Enhanced modules replace legacy functionality while maintaining compatibility
- All outputs saved to `outputs/` directory for analysis
- Comprehensive logging and error handling throughout
- Rate limiting implemented for API calls