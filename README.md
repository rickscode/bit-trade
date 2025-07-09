# Bit-Trade: Autonomous Crypto Trading Agent

> **Enhanced v2.0** - A fully automated trading research system powered by deep learning and fine-tuned LLMs with **recursive learning capabilities**. Designed to **think, act, evaluate, and adapt** — without human intervention.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Project Vision

Build an agentic system that operates like **Renaissance Technologies** for cryptocurrency trading:

- **🧠 Thinks**: Uses LLMs to generate diverse trading strategies with recursive learning
- **⚡ Acts**: Applies and backtests strategies across multiple assets and timeframes
- **📊 Evaluates**: Comprehensively judges performance with 25+ metrics
- **🔄 Adapts**: Learns from successful strategies and continuously improves

**Continuous autonomous loop with minimal human intervention** - only needed for architecture updates and system monitoring.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- API keys for Binance, Groq, and Supabase

### Setup
```bash
# Clone and setup
git clone <your-repo-url>
cd bit-trade

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Configuration
Create a `.env` file with your API keys:
```env
GROQ_API_KEY=your_groq_key_here
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_service_role_key
BINANCE_API_KEY=your_binance_key
BINANCE_API_SECRET=your_binance_secret
```

### Run the System
```bash
# Run complete enhanced system (recommended)
python main.py --mode full --cycles 3 --strategies 5

# Or run individual phases
python main.py --mode collect    # Data collection only
python main.py --mode learn      # Learning system only
python main.py --mode backtest   # Backtesting only
```

---

## 🏗️ Enhanced System Architecture

### Core Components

| Module | Description | Key Features |
|--------|-------------|--------------|
| **Enhanced Data Collector** | Multi-asset, multi-timeframe data collection | 10+ symbols, 10 timeframes, 50+ indicators |
| **Recursive Learning System** | Learns from successful strategies | 8 strategy types, pattern recognition, batch generation |
| **Enhanced Backtesting** | Comprehensive strategy evaluation | Walk-forward, Monte Carlo, 25+ metrics |
| **Legacy Modules** | Original MVP components | Maintained for compatibility |

### Repository Structure

```
bit-trade/
├── main.py                          # 🚀 Main entry point
├── core/                            # 🎯 Enhanced core modules
│   ├── strategy_learning_system.py      # Recursive learning system
│   ├── enhanced_data_collector.py       # Multi-asset data collection
│   └── enhanced_backtest_system.py      # Comprehensive backtesting
├── legacy/                          # 📦 Legacy modules (reference)
│   ├── data_fetch.py                    # Basic data fetching
│   ├── backtest_strategy.py             # Basic backtesting
│   ├── evaluate_strategy.py             # Basic evaluation
│   └── save_to_supabase.py              # Basic database operations
├── data/                            # 📊 Raw market data (CSV)
├── strategies/                      # 🧠 Generated trading strategies
├── outputs/                         # 📈 System outputs and reports
├── requirements.txt                 # 📋 Python dependencies
└── venv/                           # 🐍 Virtual environment
```

---

## 🔄 Recursive Learning Workflow

### Phase 1: Enhanced Data Collection
- **Multi-Asset Support**: BTCUSDT, ETHUSDT, ADAUSDT, DOTUSDT, LINKUSDT, etc.
- **Multi-Timeframe**: 1m, 5m, 15m, 30m, 1h, 4h, 12h, 1d, 3d, 1w
- **Market Regime Detection**: Bull/bear/sideways classification
- **50+ Technical Indicators**: RSI, MACD, Bollinger Bands, ADX, CCI, etc.
- **Synthetic Scenarios**: Crash, bubble, recovery simulations

### Phase 2: Recursive Learning
1. **Analyze**: Review successful strategies from database
2. **Learn**: Extract patterns from profitable trades
3. **Generate**: Create 8 diverse strategy types using insights
4. **Diversify**: Momentum, mean reversion, breakout, trend following, etc.
5. **Iterate**: Each cycle builds on previous successes

### Phase 3: Enhanced Backtesting
- **Comprehensive Metrics**: 25+ performance indicators
- **Walk-Forward Analysis**: Tests across different time periods
- **Monte Carlo Simulation**: 1000+ random scenario stress tests
- **Risk Analysis**: VaR, CVaR, Ulcer Index, Tail Ratio
- **Trade Pattern Analysis**: Streak detection, monthly performance

---

## 📊 Performance Expectations

| Metric | Legacy System | Enhanced System |
|--------|---------------|-----------------|
| **Daily Strategy Generation** | 1-2 strategies | 50+ strategies |
| **Strategy Diversity** | Single type | 8 different types |
| **Assets Covered** | 1 symbol | 10+ symbols |
| **Timeframes** | 1 timeframe | 10 timeframes |
| **Backtesting Depth** | Basic metrics | 25+ comprehensive metrics |
| **Learning Capability** | None | Recursive pattern learning |
| **Market Conditions** | Single scenario | Multiple regimes + synthetic |

---

## 🛠️ Advanced Usage

### Command Line Options
```bash
# Full system with custom parameters
python main.py --mode full --cycles 5 --strategies 10 --symbols BTCUSDT ETHUSDT --timeframes 1h 1d

# Individual module execution
python core/enhanced_data_collector.py
python core/strategy_learning_system.py
python core/enhanced_backtest_system.py

# Legacy system (original MVP)
python legacy/data_fetch.py
python legacy/backtest_strategy.py
```

### Strategy Types Generated
- **Momentum Based**: Trend following with momentum indicators
- **Mean Reversion**: Price reversal strategies
- **Breakout Detection**: Volume-confirmed breakouts
- **Trend Following**: Multi-timeframe trend analysis
- **Volatility Trading**: Volatility expansion/contraction
- **Support/Resistance**: Key level trading
- **Multi-Timeframe**: Cross-timeframe confirmation
- **Volume Analysis**: Volume-price relationship strategies

### Risk Management Styles
- **Fixed Percentage**: Consistent 2% risk per trade
- **Volatility Based**: ATR-adjusted position sizing
- **Kelly Criterion**: Optimal position sizing based on win rate
- **Position Sizing**: Dynamic sizing based on market conditions
- **Stop Loss Trailing**: Adaptive stop loss management
- **Time-Based Exit**: Duration-based trade management

---

## 🗄️ Database Schema

The system uses Supabase (PostgreSQL) for strategy storage:

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

---

## 📈 Key Features

### 🧠 Recursive Learning
- Analyzes successful strategies to identify winning patterns
- Extracts common indicators and risk management approaches
- Generates new strategies based on learned insights
- Continuously improves success rate over time

### 🎯 Multi-Asset & Multi-Timeframe
- Simultaneous strategy generation across 10+ cryptocurrency pairs
- Tests strategies across 10 different timeframes
- Identifies optimal timeframes for different strategy types
- Cross-asset pattern recognition

### 🔬 Advanced Backtesting
- Walk-forward analysis for robustness testing
- Monte Carlo simulation with 1000+ scenarios
- Comprehensive risk metrics (VaR, CVaR, Ulcer Index)
- Trade pattern analysis and streak detection

### 🛡️ Risk Management
- Multiple position sizing algorithms
- Dynamic stop-loss and take-profit levels
- Volatility-adjusted risk management
- Maximum drawdown controls

---

## 🔧 Development & Contribution

### Running Tests
```bash
# Test individual components
python -m pytest tests/

# Manual testing
python core/enhanced_data_collector.py
python core/strategy_learning_system.py
```

### Adding New Strategy Types
1. Add strategy template to `strategy_learning_system.py`
2. Implement signal generation in `enhanced_backtest_system.py`
3. Update documentation and tests

### Monitoring System Performance
```bash
# Check learning statistics
python -c "from core.strategy_learning_system import StrategyLearningSystem; print(StrategyLearningSystem().get_learning_statistics())"

# View recent strategies
# Check Supabase dashboard or use database queries
```

---

## 📚 Technical Details

### Dependencies
- **VectorBT**: High-performance backtesting framework
- **Groq**: LLM API for strategy generation and evaluation
- **Supabase**: PostgreSQL database for strategy storage
- **Binance API**: Real-time and historical market data
- **Pandas/NumPy**: Data manipulation and analysis
- **TA-Lib**: Technical analysis indicators

### Performance Optimizations
- Parallel strategy generation and backtesting
- Efficient data structures for large datasets
- Rate limiting for API calls
- Memory management for large backtests

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading cryptocurrencies involves substantial risk of loss. Past performance does not guarantee future results. Always do your own research and consider consulting with a financial advisor before making investment decisions.

---

## 🔗 Links

- [VectorBT Documentation](https://vectorbt.dev/)
- [Groq API Documentation](https://console.groq.com/docs)
- [Supabase Documentation](https://supabase.com/docs)
- [Binance API Documentation](https://binance-docs.github.io/apidocs/)

---

**Built with ❤️ for the future of autonomous trading**