# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Overview

**Bit-Trade v2.2** - An autonomous crypto trading agent with:
- **8 AI Models**: 5 Groq + 3 Cloudflare Workers AI models
- **Q-Learning**: Intelligent model selection based on market conditions
- **Reinforcement Learning**: Continuous strategy optimization
- **Multi-Provider Architecture**: Groq + Cloudflare for redundancy and diversity

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
- `GROQ_API_KEY`: For 5 Groq LLM models
- `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_AUTH_TOKEN`: For 3 Cloudflare AI models (optional)
- `SUPABASE_URL` and `SUPABASE_KEY`: For database operations
- `BINANCE_API_KEY` and `BINANCE_API_SECRET`: For market data fetching
- `USE_Q_LEARNING=true`: Enable intelligent model selection (optional)

## Repository Structure

```
bit-trade/
├── main.py                     # Main entry point for enhanced system
├── core/                       # Enhanced core modules
│   ├── multi_llm_manager.py           # 8-model AI orchestration with Q-learning
│   ├── cloudflare_ai_client.py        # Cloudflare Workers AI integration  
│   ├── q_learning_agent.py            # Q-learning model selector
│   ├── strategy_learning_system.py    # Recursive learning system
│   ├── enhanced_data_collector.py     # Multi-asset data collection
│   ├── enhanced_backtest_system.py    # Comprehensive backtesting
│   └── enhanced_logger.py             # Comprehensive logging system
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

### 2. Multi-LLM Orchestration (`core/multi_llm_manager.py`)
- **8 AI Models**: 5 Groq + 3 Cloudflare Workers AI models
- **Q-Learning Selection**: Intelligent model choice based on market conditions
- **Cross-Provider Redundancy**: Automatic fallback between providers
- **Performance Tracking**: Per-model success rate and optimization
- **Adaptive Learning**: Model selection improves over time

### 3. Recursive Learning System (`core/strategy_learning_system.py`)
- **Strategy Diversification**: 8 strategy types (momentum, mean reversion, etc.)
- **Learning from Success**: Analyzes profitable strategies for patterns
- **Multi-Model Generation**: Uses different AI models for diverse approaches
- **Batch Generation**: Creates multiple strategies per cycle
- **Continuous Improvement**: Each cycle builds on previous successes
- **Performance Tracking**: Monitors success rates and learning progress

### 4. Enhanced Backtesting (`core/enhanced_backtest_system.py`)
- **Comprehensive Metrics**: 25+ performance indicators
- **Walk-Forward Analysis**: Tests across different time periods
- **Monte Carlo Simulation**: 1000+ random scenario tests
- **Risk Analysis**: VaR, CVaR, Ulcer Index, Tail Ratio
- **Trade Pattern Analysis**: Streak detection, monthly performance

## Core Development Commands

```bash
# Test AI model integration
python test_cloudflare_integration.py
python test_q_learning_integration.py

# Collect comprehensive training data
python core/enhanced_data_collector.py

# Run recursive learning cycles
python core/strategy_learning_system.py

# Generate comprehensive backtest reports
python core/enhanced_backtest_system.py

# Test individual components
python core/multi_llm_manager.py
python core/cloudflare_ai_client.py
python core/q_learning_agent.py

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

### Multi-LLM AI Models
**Groq Models (Free Tier)**:
- `llama-3.3-70b-versatile`: Balanced & reliable baseline
- `deepseek-r1-distill-llama-70b`: Analytical & mathematical reasoning
- `meta-llama/llama-4-maverick-17b`: Creative & unconventional approaches
- `llama/llama-4-scout-17b`: Pattern exploration & discovery
- `qwen/qwen3-32b`: Alternative perspectives & diverse thinking

**Cloudflare Workers AI Models (Free Tier - 10,000 Neurons/day)**:
- `@cf/deepseek-ai/deepseek-r1-distill-qwen-32b`: Advanced reasoning and analysis
- `@cf/qwen/qwen2.5-coder-32b-instruct`: Code-specialized strategy implementation
- `@cf/qwen/qwq-32b`: Question-answering and problem-solving focused

### Q-Learning Process
1. **Market Analysis**: Calculate volatility, trend, and performance metrics
2. **State Discretization**: Map market conditions to discrete states
3. **Model Selection**: Q-learning agent chooses optimal AI model
4. **Strategy Generation**: Selected model creates trading strategy
5. **Performance Feedback**: Results update Q-learning rewards
6. **Policy Improvement**: Agent learns better model selection over time

### Recursive Learning Process
1. **Analyze**: Review successful strategies in database
2. **Learn**: Extract patterns from profitable trades
3. **Generate**: Create diverse new strategies using AI model insights
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
- **Model Diversity**: 8 AI models (5 Groq + 3 Cloudflare) for varied approaches
- **Strategy Diversity**: 8 strategy types across multiple assets/timeframes
- **Quality**: Q-learning improves model selection; recursive learning improves strategy quality
- **Robustness**: Strategies tested across multiple market scenarios
- **Reliability**: Cross-provider redundancy ensures continuous operation

## Development Notes

- **Autonomous Operation**: System designed for continuous autonomous operation
- **Multi-Provider Architecture**: Groq + Cloudflare for redundancy and cost optimization
- **Q-Learning Integration**: Intelligent model selection based on market conditions
- **Enhanced Modules**: Replace legacy functionality while maintaining compatibility
- **Comprehensive Logging**: All events logged to `logs/` directory for debugging
- **Output Management**: All outputs saved to `outputs/` directory for analysis
- **Error Handling**: Robust fallback mechanisms across providers and models
- **Rate Limiting**: Implemented for both Groq and Cloudflare API calls

## Recent Major Updates (v2.2)

1. **Cloudflare Workers AI Integration** - Added 3 additional AI models
2. **Q-Learning Model Selection** - Intelligent, adaptive model choice
3. **Cross-Provider Redundancy** - Automatic fallback between Groq and Cloudflare
4. **Enhanced Performance Tracking** - Per-model and per-provider analytics
5. **Comprehensive Testing Suite** - Integration tests for all components

## Next Development Steps

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

## Troubleshooting

**Model Availability Issues**:
- Check API keys in `.env` file
- Verify account limits (Groq rate limits, Cloudflare Neurons usage)
- Use `python test_cloudflare_integration.py` to test Cloudflare models
- Use `python test_q_learning_integration.py` to test Q-learning system

**Performance Issues**:
- Enable Q-learning with `USE_Q_LEARNING=true` for intelligent model selection
- Check model performance in `outputs/model_performance.json`
- Review Q-learning statistics in logs for optimization insights