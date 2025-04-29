# Bit-Trade: Autonomous Crypto Trading Agent

> MVP a fully automated trading research system powered by deep learning and fine-tuned LLMs. Designed to **think, act, evaluate, and adapt** — without human intervention.

---

## Project Goal

To build an agentic system that:
- **Thinks**: Uses deep learning to generate math trading strategies
- **Acts**: Applies and backtests the strategy using historical market data
- **Evaluates**: Judges performance and determines viability
- **Adapts**: Learns, saves, or regenerates strategies based on results

Continuous loop. No human intervention on trades (No emotion). Humnan intervention only needed for architecture and project updates.

Inspired by the vision of building a "Renaissance Technologies" like system for crypto.

---

## Core Components

| Module | Description |
|--------|-------------|
| `data_fetch.py` | Fetches 1-year historical data from Binance |
| `format_csv_to_json.py` | Converts CSV to LLM-ready JSON format |
| `llm_strategy_generator.py` | Uses Groq LLM to create trading strategies |
| `backtest_strategy.py` | Tests generated strategy using VectorBT |
| `evaluate_strategy.py` | Uses LLM to analyze backtest results |
| `save_to_supabase.py` | Uploads approved strategies to Supabase |
| `feedback_loop.py` | Orchestrates the full autonomous loop |
| `create_database.py` | Creates local SQLite DB (deprecated) |

---

## Agent Workflow (MVP)

1. **LLM Strategy Generator** → Creates a new trading strategy.
2. **Backtester** → Simulates how well that strategy performs.
3. **LLM Evaluator** → Reviews backtest results.
4. **Decision Logic**:
    - ✅ Viable → Save to Supabase
    - ❌ Poor → Generate new strategy 
5. Repeat loop continuously.

---

## Supabase Schema

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

## Folder Structure

```
bit-trade/
├── data/                  # Raw CSVs
├── formatted_data/        # JSONs for LLM
├── strategies/            # LLM-generated strategy text
├── utils/                 # Utility scripts
├── feedback_loop.py       # Automation entry point
├── save_to_supabase.py    # Uploads data to Supabase
├── requirements.txt       # Dependencies
└── venv/                  # Virtual env
```

---

## Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Make sure to create a `.env` file with your keys:

```env
GROQ_API_KEY=your_key_here
SUPABASE_URL=https://xyzcompany.supabase.co
SUPABASE_KEY=your_secret_service_role_key
BINANCE_API_KEY= 
BINANCE_API_SECRET=
```

---

## Usage

```bash
# Fetch data
python data_fetch.py

# Format to JSON
python format_csv_to_json.py

# Generate strategy
python llm_strategy_generator.py

# Backtest
python backtest_strategy.py

# Evaluate viability
python evaluate_strategy.py

# Save to Supabase (if viable)
python save_to_supabase.py

# OR run full autonomous cycle
python feedback_loop.py
```

---

## Future Agents (Post-MVP)

- Strategy Generator (LLM)
- Evaluator Agent (LLM)
- Risk Manager Agent
- Sentiment Agent (news, Twitter, etc.)
- Execution Agent (place trades live)
- Memory Agent (archive learnings)

---

## Notes

- Strategies are backtested using [VectorBT](https://vectorbt.dev/)
- LLM is accessed via Groq Cloud
- Database is hosted on Supabase (PostgreSQL)


