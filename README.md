# Market Making Game

An interactive market-making simulator that lets you experience the challenges of providing liquidity in financial markets. Quote bid/ask spreads, manage inventory risk, and learn about adverse selection through hands-on gameplay.

Built on the **Glosten-Milgrom (1985)** model of market microstructure.

## What Is Market Making?

A **market maker** provides liquidity by continuously quoting two prices:
- **Bid**: the price at which they are willing to buy
- **Ask**: the price at which they are willing to sell

The difference between ask and bid is the **spread**. Market makers earn the spread as compensation for bearing two key risks:

1. **Adverse selection** -- trading against informed traders who know the true asset value
2. **Inventory risk** -- holding a position that could move against them

## The Glosten-Milgrom Model

The Glosten-Milgrom (1985) model explains why bid-ask spreads exist even in a competitive market with zero operating costs. The key insight is that spreads compensate market makers for losses to informed traders.

### Setup

- An asset has an unknown true value **V**, which can be **V_H** (high) or **V_L** (low)
- Prior probability that V = V_H is **mu** (typically 0.5)
- A fraction **alpha** of traders are **informed** (they know V)
- The remaining **(1 - alpha)** are **noise traders** (trade randomly)

### Equilibrium

Using Bayesian updating, the competitive market maker sets:

```
Ask = E[V | buy order]   -- buy orders signal higher value
Bid = E[V | sell order]  -- sell orders signal lower value
```

The equilibrium spread (with mu = 0.5) simplifies to:

```
Spread = (V_H - V_L) * [P(V_H | buy) - P(V_H | sell)]
```

Where:
```
P(V_H | buy)  = [alpha + (1-alpha)/2] / 1  (Bayes' rule with mu=0.5)
P(V_H | sell) = [(1-alpha)/2] / 1
```

At equilibrium, the market maker earns **zero expected profit** -- the spread earned on noise traders exactly offsets losses to informed traders.

### Adverse Selection

When you trade against an informed trader:
- If they **buy** from you (at your ask), the true value is likely **above** your ask -- you sold too cheaply
- If they **sell** to you (at your bid), the true value is likely **below** your bid -- you bought too expensively

This systematic loss is **adverse selection** -- the more informed traders in the market (higher alpha), the wider you must quote to survive.

## How the Game Works

### Market Simulation

The hidden true value follows a stochastic process:

```
V(t) = V(t-1) + sigma * epsilon + jump_size * Poisson(lambda)
```

- **Brownian motion**: continuous random drift (`sigma * N(0,1)`)
- **Poisson jumps**: occasional large moves (news events, regime changes)

You never see the true value during the game -- it is revealed only after the game ends.

### Order Flow

Each round, an order arrives from either:
- **Informed trader** (probability alpha): knows V, buys if V > ask, sells if V < bid
- **Noise trader** (probability 1 - alpha): random buy or sell

### Your Job

Each round you:
1. Set your **bid** and **ask** prices
2. An order arrives and executes against your quote
3. Your position and PnL update

### PnL Decomposition

Your profit/loss breaks down into three components:

```
Total PnL = Spread Capture - Adverse Selection + Inventory PnL
```

| Component | Description |
|-----------|-------------|
| **Spread Capture** | Half-spread earned on each trade (always positive) |
| **Adverse Selection** | Losses from trading against informed traders |
| **Inventory PnL** | Mark-to-market gains/losses from holding inventory |

## Difficulty Levels

| Level | Alpha | Sigma | Jumps | Rounds | Description |
|-------|-------|-------|-------|--------|-------------|
| Tutorial | 0.10 | 0.1 | None | 20 | Learn the basics |
| Easy | 0.20 | 0.3 | Rare | 30 | Practice spread management |
| Medium | 0.30 | 0.5 | Moderate | 40 | Active inventory management required |
| Hard | 0.40 | 0.8 | Frequent | 50 | Precise quoting under pressure |
| Expert | 0.50 | 1.2 | Heavy | 60 | Extremely toxic order flow |

## Installation

```bash
# Clone the repository
git clone https://github.com/abhi-wadhwa/market-making-game.git
cd market-making-game

# Install dependencies
pip install -e ".[dev]"
```

## Usage

### Streamlit UI (recommended)

```bash
streamlit run src/viz/app.py
```

### Command-Line Interface

```bash
python -m src.cli --difficulty medium --seed 42
```

### Run the Demo

```bash
python examples/demo.py
```

### Docker

```bash
docker build -t market-making-game .
docker run -p 8501:8501 market-making-game
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linter
ruff check src/ tests/

# Run type checker
mypy src/
```

## Project Structure

```
market-making-game/
├── src/
│   ├── core/
│   │   ├── market.py          # Hidden true value simulation (Brownian + jumps)
│   │   ├── order_flow.py      # Informed + noise trader order generation
│   │   ├── market_maker.py    # Quote management and order execution
│   │   ├── inventory.py       # Position tracking and risk metrics
│   │   ├── glosten_milgrom.py # Theoretical equilibrium spread model
│   │   ├── analytics.py       # PnL decomposition and game statistics
│   │   └── difficulty.py      # Difficulty level configurations
│   ├── viz/
│   │   └── app.py             # Streamlit interactive UI
│   └── cli.py                 # Terminal-based game interface
├── tests/
│   ├── test_market.py         # Market simulation tests
│   ├── test_order_flow.py     # Order flow generation tests
│   ├── test_analytics.py      # PnL accounting balance tests
│   └── test_glosten_milgrom.py # Theoretical model verification
├── examples/
│   └── demo.py                # Automated demo with simple strategy
├── pyproject.toml
├── Dockerfile
├── Makefile
└── LICENSE
```

## Key References

- **Glosten, L. R., & Milgrom, P. R.** (1985). Bid, ask and transaction prices in a specialist market with heterogeneously informed traders. *Journal of Financial Economics*, 14(1), 71-100.
- **Kyle, A. S.** (1985). Continuous auctions and insider trading. *Econometrica*, 53(6), 1315-1335.
- **O'Hara, M.** (1995). *Market Microstructure Theory*. Blackwell Publishers.

## License

MIT
