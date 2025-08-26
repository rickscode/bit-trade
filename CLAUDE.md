
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Overview

**Bit-Trade v3.0 Simplified** - An autonomous crypto trading agent with:
- **5 Groq Models**: Simplified high-performance AI ensemble
- **AlphaZero-Style RL**: Monte Carlo Tree Search with LLM policy/value networks
- **Real Demo Trading**: Binance Testnet integration for live market simulation
- **Streamlined Architecture**: Focus on proven Groq models for reliability

## Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Test system components
python main_simplified.py --mode test

# Run demo trading with AlphaZero agent
python main_simplified.py --mode demo --episodes 3 --balance 10000

# Run training mode for extended learning
python main_simplified.py --mode train --episodes 10 --mcts_sims 50
```

## Environment Setup

Required environment variables in `.env`:
- `GROQ_API_KEY`: For 5 Groq LLM models (REQUIRED)
- `BINANCE_API_KEY` and `BINANCE_API_SECRET`: For Binance Testnet demo trading (REQUIRED)
- `SUPABASE_URL` and `SUPABASE_KEY`: For result storage (optional)

## Repository Structure

```
bit-trade/
├── main.py                     # Main entry point for enhanced system
├── core/                       # Enhanced core modules
│   ├── multi_llm_manager.py           # 23+ model AI orchestration with Q-learning
│   ├── cloudflare_ai_client.py        # Cloudflare Workers AI integration  
│   ├── openrouter_ai_client.py        # OpenRouter AI integration (15+ free models)
│   ├── q_learning_agent.py            # Q-learning model selector (expanded)
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

### 2. AI Quant Fund Team (`core/multi_llm_manager.py`) - Renaissance Technologies Model
- **23+ AI "Staff Members"**: 5 Core Team (Groq) + 15+ Specialists (OpenRouter) + 3 Senior Advisors (Cloudflare)
- **Team Collaboration**: Ensemble consensus with multi-model collaboration (not just fallback)
- **Hierarchical Structure**: Core Team → Specialists → Advisory Board (Groq → OpenRouter → Cloudflare priority)
- **Daily Operations**: Morning briefings, strategy sessions, performance reviews, consensus decisions
- **Specialized Roles**: Each AI model has specific job function (Portfolio Manager, Quant, Creative Strategist, etc.)
- **Q-Learning Performance Reviews**: System promotes/demotes models based on performance like real staff

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
python test_openrouter_integration.py  # New: Test 23-model system

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

## AI Quant Fund Team Structure (Renaissance Technologies Model)

### 🎯 Core Team (Groq Division - Priority 1) - Fast & Reliable Operations
- **👤 Versatile (Portfolio Manager)**: `llama-3.3-70b-versatile` - General strategy oversight, balanced approach
- **📊 Analytical (Senior Quant)**: `deepseek-r1-distill-llama-70b` - Mathematical analysis, risk-adjusted returns  
- **🎲 Maverick (Creative Strategist)**: `meta-llama/llama-4-maverick-17b` - Unconventional approaches, high-risk/reward
- **🔍 Scout (Market Explorer)**: `llama/llama-4-scout-17b` - Pattern exploration & discovery
- **🌟 Diverse (Perspectives Analyst)**: `qwen/qwen3-32b` - Alternative perspectives & diverse thinking

### 🚀 Specialist Department (OpenRouter Division - Priority 2) - Maximum Diversity
**Leadership Team:**
- **🎯 Horizon (Chief Analyst)**: 256K context, comprehensive market analysis
- **🤖 GLM Agent (AI Operations)**: Agent-optimized, efficient decision-making
- **🚀 Kimi K2 (Innovation Director)**: 1T parameters (32B active), complex reasoning
- **🧮 DeepSeek R1 (Senior Math)**: Advanced reasoning, risk analysis

**Technical Team:**
- **💻 Qwen Coder (Tech Lead)**: Code generation, algorithm implementation
- **🎭 Chimera (Creative Director)**: Novel strategy generation, hybrid approaches
- **🌍 Sarvam (Global Markets)**: Multilingual analysis, international perspective
- **🎨 Venice (Contrarian)**: Uncensored analysis, unconventional strategies

**Efficiency Specialists:**
- **⚡ Gemma 3n Team**: Ultra-efficient processing, fast decisions
- **🔧 Mistral Optimizers**: Instruction-following, strategy execution
- **🧠 Hunyuan Reasoner**: Chain-of-thought, logical analysis

### 🎓 Senior Advisory Board (Cloudflare Division - Priority 3) - Specialized Validation
- **🧠 Dr. Reasoning**: `@cf/deepseek-ai/deepseek-r1-distill-qwen-32b` - Risk Director, strategic validation
- **💻 Sr. Developer**: `@cf/qwen/qwen2.5-coder-32b-instruct` - Code Architect, technical oversight
- **❓ Chief Questioner**: `@cf/qwen/qwq-32b` - Strategy Validator, quality assurance

### Team Collaboration Process (Renaissance Technologies Workflow)
1. **Morning Briefing**: All 23+ AI staff members analyze overnight market movements
2. **Research Phase**: Data collection feeds market intelligence to all departments
3. **Strategy Sessions**: Cross-departmental brainstorming using ensemble consensus
4. **Collaborative Analysis**: Core Team, Specialists, and Advisory Board work together
5. **Quality Assurance**: Senior Advisory validates all strategies before deployment  
6. **Performance Reviews**: Q-learning system promotes/demotes staff based on results
7. **Consensus Decision**: Final strategy combines perspectives from multiple departments

### Enhanced Q-Learning with Team Management
1. **Market Regime Analysis**: Calculate volatility, trend, and performance metrics
2. **Team Assignment**: Match market conditions to optimal department mix
3. **Ensemble Selection**: Choose 5-7 AI staff members for collaborative session
4. **Multi-Model Generation**: Each team member contributes strategy perspective
5. **Consensus Building**: Combine insights for final strategy recommendation
6. **Performance Tracking**: Update individual and team performance metrics
7. **Staff Optimization**: Promote high-performers, retrain underperformers

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

## Performance Expectations (Renaissance Technologies Model)

- **Daily Operations**: 100+ collaborative strategies per day vs. 1-2 with legacy system
- **Team Diversity**: 23+ AI "staff members" across 3 departments providing maximum perspective variety
- **Collaborative Intelligence**: Ensemble consensus combining 5-7 models per strategy decision
- **Quality Assurance**: Multi-departmental validation before deployment (Core → Specialists → Advisory)
- **Adaptive Management**: Q-learning "HR system" promotes/demotes staff based on performance
- **Operational Reliability**: Department hierarchy (Groq → OpenRouter → Cloudflare) ensures 24/7 operation
- **Specialized Excellence**: Each AI staff member optimized for specific role (Portfolio Manager, Quant, etc.)
- **Continuous Learning**: Daily performance reviews and team optimization like real quant fund

## Development Notes

- **Quant Fund Operations**: System operates like Renaissance Technologies with 23+ AI "employees"
- **Departmental Structure**: Core Team (Groq) → Specialists (OpenRouter) → Advisory (Cloudflare)
- **Collaborative Intelligence**: Multi-model ensemble consensus, not just sequential fallback
- **Performance Management**: Q-learning acts as "HR system" promoting/demoting based on results
- **Daily Operations**: Morning briefings, strategy sessions, performance reviews, consensus decisions
- **Staff Specialization**: Each AI model has specific role (Portfolio Manager, Quant, Creative Director, etc.)
- **Quality Assurance**: Multi-departmental validation process before strategy deployment
- **Continuous Learning**: Team optimization and adaptive role assignment based on market conditions
- **Autonomous Management**: Self-managing organization with minimal human oversight required

## Recent Major Updates (v2.3) - Renaissance Technologies Model

1. **OpenRouter AI Integration** - Added 15+ free AI models for maximum diversity
2. **Renaissance Tech Team Structure** - 23+ AI "staff members" with specific roles
3. **Ensemble Collaboration System** - Multi-model consensus, not just fallback
4. **Departmental Hierarchy** - Core Team → Specialists → Advisory (Groq → OpenRouter → Cloudflare)
5. **AI Performance Management** - Q-learning "HR system" with promotions/demotions
6. **Collaborative Decision Making** - Daily strategy sessions with cross-departmental teams
7. **Staff Specialization** - Portfolio Managers, Quants, Creative Directors, Risk Directors, etc.
8. **Comprehensive Testing** - Full 23-model integration validation suite

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