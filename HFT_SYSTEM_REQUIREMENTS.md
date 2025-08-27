# HFT Trading System Requirements - Clean Build

## Overview
Build a High-Frequency Trading (HFT) system with AI-driven decision making, OCO order management, and comprehensive risk controls. This system should be clean, reliable, and ready for live trading.

## Core System Architecture

### 1. Trading Environment (`trading_env.py`)
**Purpose**: Real Binance API integration with HFT capabilities

**Key Features**:
- **10% Capital Allocation**: Only 10% of total balance used for active trading
- **OCO Order System**: Every BUY automatically creates One-Cancels-Other orders
  - Take Profit: +2% minimum 
  - Stop Loss: -30% maximum
- **Daily Risk Management**: 
  - Target: 15% daily return on allocated capital
  - Stop Loss: 30% daily loss triggers trading halt
- **Real API Integration**: Binance Testnet for paper trading
- **WebSocket Price Feeds**: Real-time market data

**Methods Needed**:
```python
def execute_buy_with_oco(symbol, quantity, current_price):
    """Execute buy and immediately place OCO order"""
    
def check_oco_completions():
    """Monitor OCO orders and calculate P&L rewards"""
    
def calculate_reward(entry_price, exit_price, quantity):
    """New reward system: 1 point per 1% return, capped ±15%/30%"""
    
def get_trading_capital():
    """Return 10% of total balance available for trading"""
    
def check_daily_limits():
    """Enforce 15% target and 30% stop loss limits"""
```

### 2. Multi-LLM Manager (`llm_manager.py`) 
**Purpose**: 3 Groq API keys with automatic rotation (300k tokens/day total)

**Models**:
- **versatile**: `llama-3.3-70b-versatile` - General analysis
- **analytical**: `deepseek-r1-distill-llama-70b` - Mathematical analysis  
- **diverse**: `qwen/qwen3-32b` - Alternative perspectives

**Key Features**:
- **Automatic Key Rotation**: Switch keys when rate limits hit
- **Ensemble Decision Making**: Combine multiple model outputs
- **Performance Tracking**: Track which models perform best

**Methods Needed**:
```python
def get_ensemble_decision(market_data, portfolio_state):
    """Get trading decision from multiple models"""
    
def rotate_api_key(model_name):
    """Switch to next API key when rate limited"""
    
def track_model_performance(model, decision, outcome):
    """Update model performance metrics"""
```

### 3. AI Trading Agent (`ai_agent.py`)
**Purpose**: MCTS-based decision making with experience learning

**Key Features**:
- **MCTS Decision Making**: Monte Carlo Tree Search for action selection
- **Experience Buffer**: Store and learn from trading outcomes  
- **Q-Learning Integration**: Adaptive model selection
- **Real Reward System**: Learn from actual P&L results

**Methods Needed**:
```python
def mcts_search(state, simulations=10):
    """MCTS tree search for optimal action"""
    
def store_experience(state, action, reward, next_state):
    """Add trading experience to buffer"""
    
def update_q_values(model_performance):
    """Update Q-learning model selection"""
```

### 4. Dashboard (`dashboard.py`)
**Purpose**: Real-time web interface for monitoring

**Key Features**:
- **Portfolio Tracking**: Live balance, P&L, positions
- **OCO Monitor**: Active OCO orders with take profit/stop loss levels
- **Capital Allocation**: Visual display of 10% trading capital usage
- **Daily Progress**: Progress toward 15% daily target
- **AI Decisions**: MCTS reasoning and model consensus
- **Risk Alerts**: Warning when approaching daily limits

### 5. Risk Management (`risk_manager.py`)
**Purpose**: Comprehensive risk controls and monitoring

**Key Features**:
- **Position Sizing**: Never exceed 10% of total balance
- **Daily Limits**: 15% profit target, 30% loss stop
- **OCO Validation**: Ensure all positions have stop loss protection
- **Emergency Controls**: Automatic trading halt on limit breach

## Environment Variables Required

```bash
# Binance API (Testnet)
BINANCE_API_KEY=your_binance_testnet_api_key
BINANCE_API_SECRET=your_binance_testnet_secret

# Groq API Keys (3 keys for 300k tokens/day)
GROQ_API_KEY=your_primary_groq_key
GROQ_API_KEY_2=your_secondary_groq_key  
GROQ_API_KEY_3=your_tertiary_groq_key
```

## Success Metrics

### Daily Performance Targets
- **15% Daily Return**: Target profit on 10% allocated capital
- **8 Profitable Trades**: Average trades per day with 2%+ profit each
- **Risk Compliance**: Never exceed 10% allocation or 30% daily loss
- **OCO Success Rate**: >70% of orders complete at take profit (not stop loss)

### Live Trading Readiness Criteria
1. **100+ Trading Experiences**: Consistent OCO execution without errors
2. **5+ Consecutive Profitable Days**: 10%+ daily returns without stop loss triggers
3. **Perfect Risk Compliance**: No violations of allocation or daily limits
4. **System Stability**: 99%+ uptime with zero critical errors
5. **AI Confidence**: Average decision confidence >70%

## File Structure (Clean Build)

```
hft-trading-system/
├── README.md
├── requirements.txt
├── .env.example
├── main.py                 # Entry point
├── config/
│   ├── __init__.py
│   └── settings.py         # Configuration management
├── core/
│   ├── __init__.py
│   ├── trading_env.py      # Binance integration + OCO system
│   ├── llm_manager.py      # Multi-LLM with 3 API keys
│   ├── ai_agent.py         # MCTS + Q-learning agent
│   └── risk_manager.py     # Risk controls and limits
├── dashboard/
│   ├── __init__.py
│   └── app.py             # Streamlit dashboard
├── utils/
│   ├── __init__.py
│   ├── logger.py          # Logging system
│   └── data_manager.py    # Data persistence
├── tests/
│   ├── test_trading_env.py
│   ├── test_llm_manager.py
│   └── test_ai_agent.py
└── data/
    ├── agent_memory.json   # Trading experiences
    ├── model_performance.json
    └── trading_sessions/
```

## Implementation Priority

### Phase 1: Core Foundation
1. **Trading Environment**: Binance API + basic buy/sell
2. **LLM Manager**: Single model integration with Groq
3. **Simple Agent**: Basic decision making without MCTS
4. **Risk Manager**: 10% allocation and daily limits

### Phase 2: HFT Features  
1. **OCO Order System**: Automatic stop loss + take profit
2. **Multi-LLM**: 3 API keys with rotation
3. **MCTS Agent**: Advanced decision making
4. **Dashboard**: Real-time monitoring

### Phase 3: Production Ready
1. **Comprehensive Testing**: Unit tests and integration tests
2. **Error Handling**: Robust error recovery and logging
3. **Performance Optimization**: Speed and reliability improvements
4. **Live Trading Preparation**: Final safety checks

## Key Differences from Old System

1. **Clean Architecture**: Separated concerns, clear interfaces
2. **Proper OCO Integration**: Native Binance OCO API usage
3. **Real Risk Management**: Hard limits enforced at system level
4. **Simplified Codebase**: Remove unnecessary complexity
5. **Better Error Handling**: Graceful degradation and recovery
6. **Comprehensive Testing**: Ensure reliability before live trading

## Development Approach

1. **Start Simple**: Basic buy/sell with manual orders first
2. **Add Features Incrementally**: One feature at a time with testing
3. **Focus on Reliability**: Prefer simple, working code over complexity
4. **Test Everything**: Unit tests and integration tests for each component
5. **Document as We Go**: Clear documentation and examples

This approach will give us a clean, reliable HFT system that's properly architected and ready for live trading, instead of trying to fix the existing broken codebase.