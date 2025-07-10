# Bit-Trade: Autonomous Crypto Trading Agent

> **Enhanced v2.2** - A fully automated trading research system powered by **8 AI models** (Groq + Cloudflare) with **Q-learning and multi-provider reinforcement learning**. Designed to **think, act, evaluate, and adapt** — without human intervention.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Project Vision

Build an agentic system that operates like **Renaissance Technologies** for cryptocurrency trading:

- **🧠 Thinks**: Uses 8 AI models (Groq + Cloudflare) to generate diverse trading strategies with Q-learning selection
- **⚡ Acts**: Applies and backtests strategies across multiple assets and timeframes
- **📊 Evaluates**: Comprehensively judges performance with 25+ metrics and cross-provider analytics
- **🔄 Adapts**: Learns from successful strategies and continuously improves model selection

**Continuous autonomous loop with minimal human intervention** - only needed for architecture updates and system monitoring.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- API keys for Binance, Groq, and Supabase
- Optional: Cloudflare Workers AI account (adds 3 more AI models)

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

# Cloudflare Workers AI (optional - adds 3 more AI models)
CLOUDFLARE_ACCOUNT_ID=your-account-id
CLOUDFLARE_AUTH_TOKEN=your-auth-token

# Q-Learning Configuration (optional)
USE_Q_LEARNING=true                    # Enable intelligent model selection
GROQ_MODEL_ROTATION=round_robin        # round_robin, performance_based, weighted_random, market_adaptive
```

### Run the System
```bash
# Test AI model integration first (recommended)
python test_cloudflare_integration.py
python test_q_learning_integration.py

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
| **Multi-LLM Manager** | Orchestrates 8 AI models (Groq + Cloudflare) | Q-learning selection, cross-provider redundancy, performance tracking |
| **Q-Learning Agent** | Intelligent model selection based on market conditions | Market state analysis, experience replay, adaptive learning |
| **Reinforcement Learning System** | Multi-LLM agents learn from market rewards | 8 AI models, pattern recognition, adaptive policy |
| **Enhanced Backtesting** | Comprehensive strategy evaluation & reward calculation | Walk-forward, Monte Carlo, 25+ metrics |
| **Cloudflare AI Client** | Integrates 3 Cloudflare Workers AI models | Advanced reasoning, code generation, problem solving |
| **Legacy Modules** | Original MVP components | Maintained for compatibility |

### Repository Structure

```
bit-trade/
├── main.py                          # 🚀 Main entry point
├── core/                            # 🎯 Enhanced core modules
│   ├── multi_llm_manager.py             # 8-model AI orchestration with Q-learning
│   ├── cloudflare_ai_client.py          # Cloudflare Workers AI integration
│   ├── q_learning_agent.py              # Q-learning model selector
│   ├── strategy_learning_system.py      # Recursive learning system
│   ├── enhanced_data_collector.py       # Multi-asset data collection
│   ├── enhanced_backtest_system.py      # Comprehensive backtesting
│   └── enhanced_logger.py               # Comprehensive logging system
├── legacy/                          # 📦 Legacy modules (reference)
│   ├── data_fetch.py                    # Basic data fetching
│   ├── backtest_strategy.py             # Basic backtesting
│   ├── evaluate_strategy.py             # Basic evaluation
│   └── save_to_supabase.py              # Basic database operations
├── test_cloudflare_integration.py  # 🧪 Cloudflare AI integration tests
├── test_q_learning_integration.py  # 🧪 Q-learning integration tests
├── data/                            # 📊 Raw market data (CSV)
├── strategies/                      # 🧠 Generated trading strategies
├── outputs/                         # 📈 System outputs and reports
├── logs/                            # 📝 System logs and debugging
├── requirements.txt                 # 📋 Python dependencies
└── venv/                           # 🐍 Virtual environment
```

---

## 🔄 Reinforcement Learning Workflow

### Phase 1: Enhanced Data Collection
- **Multi-Asset Support**: BTCUSDT, ETHUSDT, ADAUSDT, DOTUSDT, LINKUSDT, etc.
- **Multi-Timeframe**: 1m, 5m, 15m, 30m, 1h, 4h, 12h, 1d, 3d, 1w
- **Market Regime Detection**: Bull/bear/sideways classification
- **50+ Technical Indicators**: RSI, MACD, Bollinger Bands, ADX, CCI, etc.
- **Synthetic Scenarios**: Crash, bubble, recovery simulations

### Phase 2: Q-Learning Reinforcement Learning Engine
1. **🎯 Agent**: Multi-LLM Strategy Generator with Q-learning model selector
2. **🌍 Environment**: Financial markets + comprehensive backtesting + market state discretization  
3. **⚡ Actions**: Intelligently select AI models based on market conditions
4. **🏆 Rewards**: Multi-objective function (returns + Sharpe + win rate + drawdown protection)
5. **📚 Policy**: Q-table learning optimal model selection for different market states
6. **🔄 Adaptation**: Epsilon-greedy exploration with experience replay and policy improvement

### Phase 3: Enhanced Backtesting & Reward Calculation
- **Comprehensive Metrics**: 25+ performance indicators
- **Walk-Forward Analysis**: Tests across different time periods
- **Monte Carlo Simulation**: 1000+ random scenario stress tests
- **Risk Analysis**: VaR, CVaR, Ulcer Index, Tail Ratio
- **Trade Pattern Analysis**: Streak detection, monthly performance

---

## 📊 Performance Expectations

| Metric | Legacy System | Enhanced System v2.2 |
|--------|---------------|----------------------|
| **AI Models** | 1 model | 8 models (Groq + Cloudflare) |
| **Daily Strategy Generation** | 1-2 strategies | 50+ strategies |
| **Strategy Diversity** | Single type | 8 different types |
| **Assets Covered** | 1 symbol | 10+ symbols |
| **Timeframes** | 1 timeframe | 10 timeframes |
| **Model Selection** | Fixed | Q-learning adaptive selection |
| **Provider Redundancy** | None | Cross-provider fallback |
| **Backtesting Depth** | Basic metrics | 25+ comprehensive metrics |
| **Learning Capability** | None | Q-learning + recursive pattern learning |
| **Market Conditions** | Single scenario | Multiple regimes + synthetic |

---

## 🛠️ Advanced Usage

### Command Line Options
```bash
# Test AI model integration (recommended first step)
python test_cloudflare_integration.py    # Test Cloudflare Workers AI models
python test_q_learning_integration.py    # Test Q-learning system

# Full system with custom parameters
python main.py --mode full --cycles 5 --strategies 10 --symbols BTCUSDT ETHUSDT --timeframes 1h 1d

# Individual module execution
python core/multi_llm_manager.py         # Test 8-model AI orchestration
python core/cloudflare_ai_client.py      # Test Cloudflare AI client
python core/q_learning_agent.py          # Test Q-learning agent
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

### 🧠 Reinforcement Learning Engine
- **Multi-Agent System**: 5 different LLM models acting as diverse trading agents
- **Reward-Based Learning**: Strategies rewarded based on risk-adjusted performance
- **Adaptive Policy**: Learns which models/patterns generate successful strategies
- **Exploration vs Exploitation**: Balances trying new approaches vs using proven winners
- **Continuous Improvement**: Success rate increases with each learning cycle

### 🤖 Multi-LLM Intelligence with Q-Learning

#### 🔥 Groq Models (Free Tier)
- **llama-3.3-70b-versatile**: Balanced & reliable baseline strategies
- **deepseek-r1-distill-llama-70b**: Analytical & mathematical reasoning
- **meta-llama/llama-4-maverick-17b**: Creative & unconventional approaches
- **llama/llama-4-scout-17b**: Pattern exploration & discovery  
- **qwen/qwen3-32b**: Alternative perspectives & diverse thinking

#### ⚡ Cloudflare Workers AI Models (Free Tier - 10,000 Neurons/day)
- **@cf/deepseek-ai/deepseek-r1-distill-qwen-32b**: Advanced reasoning and analysis
- **@cf/qwen/qwen2.5-coder-32b-instruct**: Code-specialized strategy implementation
- **@cf/qwen/qwq-32b**: Question-answering and problem-solving focused

#### 🧠 Q-Learning Model Selection
- **Market State Analysis**: Volatility, trend, and performance discretization
- **Intelligent Model Selection**: Q-learning agent chooses optimal models for market conditions
- **Cross-Provider Optimization**: Learns which provider/model works best for specific market conditions
- **Experience Replay**: Learn from historical model performance patterns
- **Epsilon-Greedy Exploration**: Balance between exploitation and exploration
- **Adaptive Learning**: Model selection improves over time based on reward feedback

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
- **Groq**: Multi-LLM API with 5 free models for diverse strategy generation
- **Cloudflare Workers AI**: 3 additional AI models (reasoning, coding, problem-solving)
- **Supabase**: PostgreSQL database for strategy storage and performance tracking
- **Binance API**: Real-time and historical market data
- **Pandas/NumPy**: Data manipulation and analysis
- **TA-Lib**: Technical analysis indicators
- **Requests**: HTTP client for Cloudflare Workers AI API

### Performance Optimizations
- **Multi-LLM Orchestration**: 8 models (Groq + Cloudflare) generate strategies with intelligent selection
- **Q-Learning Optimization**: Adaptive model selection improves success rates over time
- **Cross-Provider Redundancy**: Automatic fallback between Groq and Cloudflare APIs
- **Efficient Data Structures**: Optimized for large datasets and backtests
- **Rate Limiting**: Smart API call management across multiple providers
- **Memory Management**: Handles large backtests and comprehensive model performance tracking
- **Experience Replay**: Q-learning agent learns from historical performance patterns

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

## 🎯 Next Development Steps

**Priority Tasks**:
1. **Run full system with multiple cycles for production testing**
   - Execute `python main.py --mode full --cycles 5 --strategies 10`
   - Monitor system stability and performance across extended runs
   - Validate Q-learning model selection improvements in real scenarios

2. **Analyze model performance and Q-learning effectiveness**
   - Review `outputs/model_performance.json` for optimization patterns
   - Analyze Q-learning adaptation and model selection accuracy
   - Compare performance metrics across different market conditions

3. **Test Cloudflare AI models integration and fallback**
   - Verify Cloudflare Workers AI models respond correctly
   - Test automatic fallback between Groq and Cloudflare providers
   - Validate rate limiting and error handling across providers

4. **Implement paper trading system with real-time data**
   - Connect to Binance WebSocket for real-time price feeds
   - Implement virtual portfolio management and order simulation
   - Track paper trading performance vs backtested results
   - Validate strategy execution timing and slippage modeling

5. **Implement live trading preparation and risk controls**
   - Add additional safety mechanisms for live trading
   - Implement position sizing limits and maximum drawdown controls
   - Create monitoring dashboard for real-time system health
   - Implement emergency stop mechanisms and circuit breakers

---

**Built with ❤️ for the future of autonomous trading**