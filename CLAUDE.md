# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Overview

**Bit-Trade v3.1 HFT Enhanced** - An autonomous crypto trading agent with aggressive High-Frequency Trading capabilities:
- **HFT OCO Strategy**: Every trade automatically places One-Cancels-Other orders (30% stop loss, 2% take profit)
- **Dynamic Capital Allocation**: Maximum 10% of total balance allocated to active trading
- **Daily Risk Management**: 15% daily return target with automatic trading halt on 30% loss
- **3 Core Groq Models + Multi-Key**: Optimized AI ensemble with automatic API key rotation (300k tokens/day)
- **AlphaZero-Style RL**: Monte Carlo Tree Search with LLM policy/value networks
- **Real Testnet Trading**: Live Binance Testnet API integration with actual OCO order execution
- **Advanced Reward Scaling**: 1 point per 1% return with ±15%/30% caps for optimal learning
- **Real-time Dashboard**: Web interface with HFT metrics and OCO position monitoring

## Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Test system components
python main_simplified.py --mode test

# Run HFT demo trading with OCO orders
python main_simplified.py --mode demo --episodes 10 --mcts_sims 10

# Extended HFT training targeting 15% daily returns
python main_simplified.py --mode demo --episodes 20 --max_steps 50

# Start with dashboard monitoring HFT metrics
python main_simplified.py --mode demo --episodes 5 --dashboard

# Fast HFT mode for testing OCO system
python main_simplified.py --mode demo --episodes 2 --fast
```

## Environment Setup

Required environment variables in `.env`:
- `GROQ_API_KEY`: Primary Groq API key (100k tokens/day)
- `GROQ_API_KEY_2`: Secondary Groq API key (100k tokens/day) 
- `GROQ_API_KEY_3`: Tertiary Groq API key (100k tokens/day)
- `BINANCE_API_KEY` and `BINANCE_API_SECRET`: For Binance Testnet OCO order execution (REQUIRED)

**3-Key HFT Setup**: System automatically rotates between 3 Groq API keys when rate limits hit, providing 300k tokens/day total capacity for continuous HFT learning without interruption.

## Repository Structure

```
bit-trade/
├── main_simplified.py             # Simplified main entry point
├── dashboard.py                   # Real-time web dashboard
├── core/                          # Core trading system
│   ├── alphazero_trading_agent.py # AlphaZero RL agent with learning
│   ├── binance_demo_env.py        # Real Binance Testnet environment
│   ├── multi_llm_manager.py       # 5-model Groq ensemble
│   ├── q_learning_agent.py        # Q-learning model selection
│   ├── enhanced_logger.py         # Comprehensive logging
│   └── portfolio_persistence.py   # Portfolio state persistence
├── outputs/                       # System outputs and results
│   ├── demo_trading_results.json  # Trading session results
│   ├── agent_memory.pkl           # Agent experience buffer
│   ├── model_performance.json     # AI model performance tracking
│   └── portfolio_state.json       # Portfolio persistence
├── logs/                          # System logs
│   ├── bit_trade.log             # Main system log
│   ├── errors.log                # Error tracking
│   └── performance.log           # Performance metrics
├── simple_test.py                # Simple system test
├── venv/                         # Virtual environment
└── requirements.txt              # Python dependencies
```

## System Architecture

### 1. HFT Binance Trading Environment (`core/binance_demo_env.py`)
- **OCO Order System**: Every buy automatically places One-Cancels-Other orders with 30% stop loss + 2% take profit
- **Dynamic Capital Allocation**: Maximum 10% of total balance allocated to active trading positions
- **Daily Risk Management**: 15% daily return target with automatic trading halt on 30% daily loss
- **Live Market Data**: Real WebSocket feeds from Binance Testnet
- **Real API Trading**: Actual OCO order execution via Binance Testnet API
- **Advanced Reward Scaling**: 1 point per 1% return with ±15%/30% caps for optimal RL learning
- **Portfolio Management**: Complete position tracking with live OCO order status monitoring
- **Account Synchronization**: Real-time sync with Testnet account balances and OCO completions

### 2. AlphaZero Trading Agent (`core/alphazero_trading_agent.py`)
- **MCTS Decision Making**: Monte Carlo Tree Search for action selection (optimized to 10 simulations)
- **Neural Network Policy**: AI-driven trading strategy evaluation
- **Experience Learning**: Continuous improvement from trading experience with persistence
- **Portfolio Context**: Complete visibility into positions and performance
- **Multi-LLM Integration**: Uses 5 Groq models for decision reasoning

### 3. Multi-LLM Manager (`core/multi_llm_manager.py`)
- **5 Groq Models**: Versatile, Analytical, Maverick, Scout, Diverse
- **Ensemble Decision Making**: Consensus from multiple AI perspectives
- **Q-Learning Selection**: Adaptive model selection based on performance
- **Performance Tracking**: Continuous model effectiveness monitoring

### 4. HFT Real-time Dashboard (`dashboard.py`)
- **OCO Position Monitoring**: Live tracking of active OCO orders with take profit/stop loss levels
- **Capital Allocation Display**: Visual meter showing 10% trading capital usage vs total balance
- **Daily P&L Progress**: Real-time tracking toward 15% daily return target
- **Trading Halt Status**: Visual indicator when daily 30% stop loss is triggered
- **AI Decision Visualization**: MCTS results and model consensus with HFT confidence scoring
- **HFT Performance Charts**: Interactive OCO completion tracking and reward progression
- **Risk Management Display**: Real-time OCO order status and daily risk limits

## Core Development Commands

```bash
# Test complete system
python main_simplified.py --mode test

# Run demo trading with dashboard
python main_simplified.py --mode demo --episodes 5 --dashboard

# Fast mode for development (reduced MCTS simulations)
python main_simplified.py --mode demo --episodes 3 --fast

# Extended training with learning persistence
python main_simplified.py --mode train --episodes 10 --mcts_sims 15

# Start dashboard only
python dashboard.py

# Test individual components
python core/multi_llm_manager.py
python simple_test.py
```

## 5-Model Groq Ensemble

The system uses 5 specialized Groq models for trading decisions:

- **Versatile** (`llama-3.3-70b-versatile`): Balanced general analysis
- **Analytical** (`deepseek-r1-distill-llama-70b`): Mathematical and risk analysis  
- **Maverick** (`meta-llama/llama-4-maverick-17b`): Creative and unconventional strategies
- **Scout** (`openai/gpt-oss-20b`): Market pattern exploration
- **Diverse** (`qwen/qwen3-32b`): Alternative perspectives

### Q-Learning Model Selection
- Models are selected based on performance in different market conditions
- Adaptive weighting adjusts model influence based on success rates
- Ensemble consensus combines multiple model perspectives for final decisions
- Performance data persisted to `outputs/model_performance.json`

### HFT Trading Process with OCO Orders
1. **Market Analysis**: All models analyze current market conditions and available trading capital (10% allocation)
2. **Ensemble Decision**: Q-learning selects optimal model combination for current market regime  
3. **Action Selection**: MCTS uses model consensus to choose BUY/SELL/HOLD with confidence scoring
4. **OCO Execution**: 
   - BUY orders automatically place OCO (30% stop loss + 2% take profit)
   - Position size limited to 10% of allocated trading capital
   - Daily P&L tracking toward 15% target
5. **OCO Monitoring**: System monitors OCO completions and calculates real P&L rewards
6. **Experience Storage**: Trade outcomes with actual profit/loss stored in persistent experience buffer
7. **Learning Update**: Model performance and Q-values updated based on real trading results
8. **Risk Control**: Trading halted if daily loss reaches 30% of allocated capital

## Current System Status (Aug 26, 2025 - Multi-Key Enhanced)

**Operational**:
- ✅ Real Binance Testnet API integration with live order execution
- ✅ **3 Groq API keys** with automatic rotation (300k tokens/day total)
- ✅ Live WebSocket market data with reconnection logic
- ✅ Complete portfolio visibility for AI decision making
- ✅ AlphaZero MCTS agent with fallback model support
- ✅ Persistent Q-learning system with experience buffer
- ✅ Real-time web dashboard (Streamlit) for monitoring

**Multi-Key Enhancement**:
- ✅ **Automatic API key rotation** when rate limits hit
- ✅ **3x learning capacity**: 300,000 tokens/day (vs 100k previously)
- ✅ Intelligent key switching with fallback logic
- ✅ Continuous training without daily interruption
- ✅ Core intelligence preserved throughout key transitions

**Today's Learning Results (Aug 26)**:
- **33 real trading experiences** collected in agent memory
- Multiple successful BUY/SELL orders executed on Binance Testnet
- Portfolio tracking: Real P&L, position management, order IDs
- MCTS + Q-learning working seamlessly with 2/3 models available
- Rate limit resilience: System degraded gracefully (3→2→1 models)
- Session persistence: All learning data saved for tomorrow

**Performance Optimizations**:
- Multi-key rotation prevents training interruption
- Timeout handling: 10s max with ThreadPoolExecutor
- WebSocket reconnection: Exponential backoff (5 attempts)
- LLM response caching during MCTS (prevents redundant calls)
- Rate limit detection with automatic key switching
- Dashboard format string fixes

## Dashboard Access

The real-time dashboard is available at `http://localhost:8501` when running:
```bash
python main_simplified.py --mode demo --dashboard
```

Dashboard features:
- Live portfolio value and P&L tracking
- Real-time position monitoring with current prices
- AI decision reasoning display (MCTS simulation results)
- Model performance comparison charts
- Trading history with entry/exit points
- Risk metrics and alerts

## Tomorrow's Enhanced Learning Capacity

**With 3 API Keys (300k tokens/day)**:
- **Extended Training Sessions**: 10-20 episodes without interruption
- **Deeper MCTS Analysis**: More simulations per decision (10-25 vs 3-5 today)
- **Comprehensive Experience Collection**: Target 300+ trading experiences
- **Advanced Strategy Evolution**: Let Q-learning optimize model selection over longer periods
- **Continuous Market Learning**: Trade through different market conditions without rate limit breaks

**Recommended Tomorrow's Commands**:
```bash
# Extended learning session (will use multiple API keys as needed)
python main_simplified.py --mode demo --episodes 20 --max_steps 50 --mcts_sims 15

# Continuous training with dashboard monitoring
python main_simplified.py --mode train --episodes 30 --dashboard

# High-intensity MCTS analysis
python main_simplified.py --mode demo --episodes 10 --mcts_sims 25
```

## Learning System

The agent now implements true continuous learning:

### Experience Collection
- Complete trading episodes with market conditions and outcomes
- Trade profitability scoring and pattern recognition
- Model effectiveness tracking across different market conditions

### Persistent Memory
- Experience buffer saved to `outputs/agent_memory.pkl`
- Q-learning weights persisted across sessions
- Portfolio performance history tracking
- Model selection optimization over time

### Adaptive Improvement
- Models that perform better get higher selection probability
- Successful trading patterns reinforced in decision making
- Risk management parameters adjusted based on outcomes

## Troubleshooting

**API Connection Issues**:
- Check API keys in `.env` file (GROQ_API_KEY, BINANCE_API_KEY, BINANCE_API_SECRET)
- Verify Groq rate limits and account status
- Test Binance Testnet connectivity with `python simple_test.py`

**Performance Issues**:
- Use `--fast` mode for quicker testing (reduces MCTS simulations)
- Check timeout settings if operations are slow

## Latest Updates (2025-08-26) - Dashboard Enhancement Complete

### 🎉 Major Dashboard Enhancements Implemented

**New Learning Journey Visualization**:
- ✅ Complete learning progression tracking with 33 trading experiences
- ✅ AI confidence level gauge and readiness assessment system
- ✅ Live trading readiness scoring (100-point system)
- ✅ Learning quality metrics (consistency, risk management, performance)
- ✅ Interactive charts showing reward progression over time

**Enhanced Dashboard Structure**:
- 📈 **Portfolio Tab**: Portfolio performance and episode statistics
- 🧠 **Learning Journey Tab**: NEW - Complete AI learning analysis
- 🤖 **AI Models Tab**: Model performance comparisons  
- 🔍 **Agent Memory Tab**: Experience buffer and training data
- 🤖 **AI Reasoning Tab**: Decision analysis patterns
- 📊 **Live Data Tab**: Real-time system monitoring

**Data Integration Fixed**:
- ✅ Connected rich trading data (33 experiences) to dashboard
- ✅ Fixed portfolio state loading from agent memory
- ✅ Enhanced trading history extraction
- ✅ Resolved dashboard data visibility issues

**Diagnostic Plot Generation**:
- ✅ Created `generate_diagnostic_plots.py` script
- ✅ Portfolio analysis with drawdown visualization
- ✅ System diagnostics overview charts
- ✅ All plots saved to `reports/plots/` directory

### 🚦 Live Trading Readiness Assessment

**Scoring System (100 points total)**:
- Experience factor (30 pts): Based on trading episode count
- Decision consistency (25 pts): Entropy-based decision analysis
- Performance factor (25 pts): Recent profitability assessment  
- Risk management (20 pts): Portfolio volatility analysis

**Readiness Levels**:
- 🟢 80+ points: READY for Live Trading
- 🟡 60-79 points: CAUTION - More training recommended
- 🟠 40-59 points: NOT READY - Significant training needed
- 🔴 <40 points: NOT READY - Extensive training required

### 🎯 Tomorrow's Priority Tasks

**Dashboard Data Population**:
1. Fix AI Models tab - populate with model performance data from Q-learning system
2. Enhance AI Reasoning tab - add decision analysis from MCTS and LLM reasoning
3. Integrate diagnostic plots directly into dashboard tabs
4. Add time-series prediction analytics visualization

**Dashboard Commands for Tomorrow**:
```bash
# Start enhanced dashboard
streamlit run dashboard.py --server.port 8502

# Generate fresh diagnostic plots  
python generate_diagnostic_plots.py

# Extended training to populate more data
python main_simplified.py --mode demo --episodes 20 --mcts_sims 15
```

### 📊 Current System Status

**Data Sources Working**:
- ✅ Trading results: 33 experiences collected
- ✅ Agent memory: Experience buffer and learning stats
- ✅ Portfolio state: Balance and position tracking
- ⚠️ Model performance: Needs Q-learning data integration
- ⚠️ AI reasoning: Needs MCTS decision data integration

**Files Status**:
- `dashboard.py`: Enhanced with learning journey tab
- `generate_diagnostic_plots.py`: Creates portfolio and system analysis plots
- `outputs/demo_trading_results.json`: Trading session results (working)
- `outputs/agent_memory.json`: Experience buffer (working)
- `outputs/model_performance.json`: Model rankings (needs enhancement)
- `reports/plots/`: Diagnostic visualizations (2 plots generated)

**Dashboard URL**: `http://localhost:8502`

### 🔧 Tomorrow's Technical Goals

1. **Populate AI Models Tab**:
   - Extract Q-learning model selection data
   - Show model performance rankings from multi-LLM system
   - Display model success rates and strategy generation stats

2. **Enhance AI Reasoning Tab**:
   - Extract MCTS simulation data and decision trees  
   - Show LLM reasoning patterns and confidence levels
   - Add decision factor analysis and market condition responses

3. **Integrate Diagnostic Plots**:
   - Embed generated plots directly in dashboard tabs
   - Add plot refresh functionality
   - Create interactive plot selection

4. **Add Time-Series Analytics**:
   - Price prediction confidence intervals
   - Market regime classification over time
   - Strategy performance correlation analysis

The learning journey visualization is now complete and shows the AI's progression beautifully. Tomorrow we'll focus on populating the remaining tabs with rich data from the Q-learning and reasoning systems!
- Monitor model performance in dashboard or `outputs/model_performance.json`

**Learning Issues**:
- Check `outputs/agent_memory.pkl` exists and is being updated
- Review experience buffer size in agent training stats
- Verify Q-learning updates in performance logs

**Dashboard Issues**:
- Ensure port 5000 is available
- Check dashboard logs for connection errors
- Verify Flask/Streamlit dependencies are installed

## HFT Success Metrics & Trading Benchmarks

### 🎯 Primary Success Criteria
**Daily Performance Targets**:
- **15% Daily Return**: Target 15% profit on allocated trading capital (10% of total balance)
- **8 Profitable Trades**: Average 8 successful trades per day with 2%+ take profit each
- **Risk Limit Compliance**: Never exceed 30% daily loss on allocated capital
- **Capital Efficiency**: Never allocate more than 10% of total balance to active trading

**System Performance Benchmarks**:
- **OCO Success Rate**: >70% of OCO orders should complete at take profit (not stop loss)
- **Reward Learning**: RL agent rewards should trend positive over 50+ trades
- **Model Confidence**: AI decision confidence should average >60% on profitable trades
- **Risk Management**: Zero instances of exceeding 10% capital allocation or 30% daily loss

### 🚦 Live Trading Readiness Criteria
**Minimum Requirements for Live Trading Consideration**:
1. **Experience Threshold**: 100+ trading experiences with consistent OCO execution
2. **Profitability**: 5+ consecutive days achieving 10%+ daily returns without stop loss triggers
3. **Risk Compliance**: Perfect adherence to 10% capital allocation and 30% daily stop loss limits
4. **System Stability**: Zero critical errors in OCO order placement/monitoring over 200+ trades
5. **AI Confidence**: Average decision confidence >70% with consistent model consensus

### 📊 Success Measurement Framework
**Daily Success Scorecard (100 points total)**:
- **Profitability** (40 pts): Daily return % on allocated capital (15% = full points)
- **Risk Management** (30 pts): Compliance with capital allocation and stop loss limits
- **Trading Efficiency** (20 pts): Number of profitable OCO completions (8 trades = full points)
- **System Performance** (10 pts): Zero errors in OCO execution and monitoring

**Weekly/Monthly Goals**:
- **Week 1**: Achieve 3+ days with 10%+ returns, perfect risk compliance
- **Week 2**: Achieve 4+ days with 12%+ returns, optimize OCO take profit timing
- **Week 3**: Achieve 5+ days with 15%+ returns, maintain perfect risk record
- **Month 1**: Average 12%+ daily returns with <2 stop loss triggers per week

### 🏆 Definition of "Successful Trading System"
A successful HFT trading system must demonstrate:
1. **Consistent Profitability**: 15%+ daily returns on allocated capital over 30+ trading days
2. **Risk Management Excellence**: Never exceed risk limits, average <1 stop loss trigger per week
3. **Operational Reliability**: 99%+ OCO order execution success rate with zero system failures
4. **AI Learning Progress**: Continuous improvement in decision quality and reward accumulation
5. **Capital Efficiency**: Achieve target returns using only 10% of total balance allocation

## Next Development Priorities

1. **OCO Order Optimization**: Fine-tune take profit levels based on market volatility
2. **Multi-Asset HFT**: Expand OCO strategy beyond BTCUSDT to multiple cryptocurrency pairs
3. **Advanced Position Sizing**: Dynamic allocation within 10% limit based on confidence scores
4. **Alert System**: Real-time notifications for OCO completions and risk limit approaches
5. **Live Trading Preparation**: Final safety checks and risk controls for production deployment

## Tomorrow's HFT Trading Plan (Aug 27, 2025)

### Objective: Test New HFT System with Real Performance Data

**Primary Goals**:
1. Generate meaningful trading data with new OCO order system
2. Validate 15% daily return targeting with proper risk management
3. Test 3 Groq API key rotation under continuous learning load
4. Populate dashboard with real HFT metrics and performance data

**Trading Session Commands**:
```bash
# Extended HFT session with dashboard monitoring
python main_simplified.py --mode demo --episodes 15 --mcts_sims 12 --dashboard

# High-intensity learning with all 3 API keys
python main_simplified.py --mode demo --episodes 25 --max_steps 60

# Fast OCO system validation
python main_simplified.py --mode demo --episodes 5 --fast --dashboard
```

**Success Criteria for Tomorrow**:
- **Generate 50+ new trading experiences** with OCO order data
- **Achieve 3+ profitable trading episodes** demonstrating 8%+ returns
- **Zero system failures** in OCO order placement/monitoring
- **Test API key rotation** under heavy token usage (200k+ tokens)
- **Dashboard populated** with meaningful HFT performance metrics

**Key Monitoring Points**:
1. **OCO Order Execution**: Every BUY should automatically create stop loss + take profit
2. **Capital Allocation**: Never exceed 10% of total balance in active positions
3. **Daily P&L Tracking**: Accumulation toward 15% daily target
4. **Reward System**: Proper 1 point per 1% return scaling (±15%/30% caps)
5. **API Key Rotation**: Seamless switching when rate limits hit

**Expected Outcomes**:
- Dashboard showing real profit/loss data instead of zeros
- Agent memory expanded from 33 to 80+ meaningful experiences
- Q-learning system with actual performance data for model selection
- Proof of concept for 15% daily return capability with 30% risk limits

This trading session will validate the complete HFT system and provide the foundation for live trading readiness assessment.