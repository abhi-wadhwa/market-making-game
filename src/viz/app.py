"""Streamlit UI for the interactive market-making game.

Run with:
    streamlit run src/viz/app.py
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.core.market import MarketSimulator
from src.core.order_flow import OrderFlowGenerator, TraderType, OrderSide
from src.core.market_maker import MarketMaker
from src.core.inventory import InventoryManager
from src.core.glosten_milgrom import GlostenMilgromModel
from src.core.analytics import AnalyticsEngine
from src.core.difficulty import (
    DifficultyLevel,
    DifficultyConfig,
    DIFFICULTY_PRESETS,
    get_difficulty_config,
)

# ---- Page Configuration ----
st.set_page_config(
    page_title="Market Making Game",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

LEADERBOARD_FILE = Path(__file__).resolve().parent.parent.parent / "leaderboard.json"


def load_leaderboard() -> list[dict]:
    """Load the leaderboard from disk."""
    if LEADERBOARD_FILE.exists():
        with open(LEADERBOARD_FILE, "r") as f:
            return json.load(f)
    return []


def save_leaderboard(entries: list[dict]) -> None:
    """Save the leaderboard to disk."""
    with open(LEADERBOARD_FILE, "w") as f:
        json.dump(entries, f, indent=2)


def init_game_state(config: DifficultyConfig, seed: int | None = None) -> dict:
    """Initialize all game components from a difficulty config."""
    market = MarketSimulator(
        initial_value=config.initial_value,
        sigma=config.sigma,
        jump_intensity=config.jump_intensity,
        jump_size=config.jump_size,
        seed=seed,
    )
    order_gen = OrderFlowGenerator(
        alpha=config.alpha,
        arrival_rate=config.arrival_rate,
        seed=seed + 1 if seed is not None else None,
    )
    mm = MarketMaker(initial_cash=config.initial_cash)
    inv_mgr = InventoryManager(
        max_position=config.max_position,
        initial_cash=config.initial_cash,
        inventory_penalty_coeff=config.inventory_penalty,
    )
    gm_model = GlostenMilgromModel(
        v_high=config.initial_value + 3 * config.sigma * np.sqrt(config.n_rounds),
        v_low=config.initial_value - 3 * config.sigma * np.sqrt(config.n_rounds),
        alpha=config.alpha,
    )
    analytics = AnalyticsEngine(mm, inv_mgr)

    return {
        "market": market,
        "order_gen": order_gen,
        "mm": mm,
        "inv_mgr": inv_mgr,
        "gm_model": gm_model,
        "analytics": analytics,
        "config": config,
        "current_round": 0,
        "game_over": False,
        "order_log": [],
        "mid_prices": [config.initial_value],
        "bids": [],
        "asks": [],
        "trade_prices": [],
        "trade_rounds": [],
        "trade_sides": [],
    }


def render_sidebar():
    """Render the sidebar with game controls."""
    st.sidebar.title("Market Making Game")
    st.sidebar.markdown("---")

    # Difficulty selector
    st.sidebar.subheader("Difficulty")
    difficulty_names = {level.value: config.name for level, config in DIFFICULTY_PRESETS.items()}
    selected = st.sidebar.selectbox(
        "Select difficulty",
        options=list(difficulty_names.keys()),
        format_func=lambda x: difficulty_names[x],
        index=2,  # Default to Medium
        key="difficulty_select",
    )
    config = get_difficulty_config(DifficultyLevel(selected))

    # Show difficulty parameters
    with st.sidebar.expander("Level Parameters"):
        st.write(f"**Rounds:** {config.n_rounds}")
        st.write(f"**Volatility (sigma):** {config.sigma}")
        st.write(f"**Informed trader prob (alpha):** {config.alpha}")
        st.write(f"**Jump intensity:** {config.jump_intensity}")
        st.write(f"**Jump size:** {config.jump_size}")
        st.write(f"**Max position:** {config.max_position}")
        st.markdown(f"*{config.description}*")

    st.sidebar.markdown("---")

    # Random seed
    use_seed = st.sidebar.checkbox("Set random seed", value=False)
    seed = None
    if use_seed:
        seed = st.sidebar.number_input("Seed", value=42, step=1)

    # New game button
    if st.sidebar.button("New Game", type="primary", use_container_width=True):
        st.session_state["game"] = init_game_state(config, seed)
        st.session_state["player_name"] = st.session_state.get("player_name", "Player")
        st.rerun()

    st.sidebar.markdown("---")

    # Player name for leaderboard
    st.session_state["player_name"] = st.sidebar.text_input(
        "Your name (for leaderboard)", value=st.session_state.get("player_name", "Player")
    )

    return config, seed


def render_game_info(game: dict):
    """Render the game status bar."""
    config = game["config"]
    mm = game["mm"]
    current_round = game["current_round"]

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Round", f"{current_round} / {config.n_rounds}")
    with col2:
        st.metric("Position", f"{mm.position:+d}")
    with col3:
        mid = game["mid_prices"][-1] if game["mid_prices"] else config.initial_value
        total_pnl = mm.get_total_pnl(mid)
        st.metric("Total PnL", f"${total_pnl:,.2f}")
    with col4:
        st.metric("Cash", f"${mm.cash:,.2f}")
    with col5:
        n_trades = len(mm.execution_history)
        st.metric("Trades", str(n_trades))


def render_quote_input(game: dict) -> tuple[float, float] | None:
    """Render the quote input form and return bid/ask."""
    config = game["config"]
    mm = game["mm"]

    st.subheader("Set Your Quote")

    # Suggest starting values based on last mid or initial value
    if mm.quote_history:
        last_quote = mm.quote_history[-1]
        default_bid = last_quote.bid
        default_ask = last_quote.ask
    else:
        default_bid = config.initial_value - 1.0
        default_ask = config.initial_value + 1.0

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        bid = st.number_input(
            "Bid Price (you buy at)",
            value=float(round(default_bid, 2)),
            step=0.1,
            format="%.2f",
            key=f"bid_input_{game['current_round']}",
        )

    with col2:
        ask = st.number_input(
            "Ask Price (you sell at)",
            value=float(round(default_ask, 2)),
            step=0.1,
            format="%.2f",
            key=f"ask_input_{game['current_round']}",
        )

    with col3:
        st.write("")  # spacing
        st.write("")
        spread = ask - bid
        if spread > 0:
            st.info(f"Spread: {spread:.2f}")
        else:
            st.error("Invalid: bid >= ask")

    # Inventory warning
    if abs(mm.position) > config.max_position * 0.7:
        direction = "LONG" if mm.position > 0 else "SHORT"
        st.warning(
            f"Inventory warning: position is {mm.position:+d} ({direction}). "
            f"Max allowed: +/-{config.max_position}. Consider skewing your quotes."
        )

    submitted = st.button(
        "Submit Quote & Advance Round",
        type="primary",
        use_container_width=True,
        disabled=game["game_over"] or bid >= ask,
    )

    if submitted and bid < ask:
        return bid, ask
    return None


def process_round(game: dict, bid: float, ask: float) -> str:
    """Process a single round of the game.

    Returns a log message describing what happened.
    """
    market = game["market"]
    order_gen = game["order_gen"]
    mm = game["mm"]
    inv_mgr = game["inv_mgr"]

    # Advance true value
    true_value = market.step()
    current_round = market.current_round

    # Set the market maker's quote
    mm.set_quote(bid, ask, current_round)

    # Generate an incoming order
    order = order_gen.generate_order(
        round_number=current_round,
        true_value=true_value,
        bid=bid,
        ask=ask,
    )

    log_msg = f"**Round {current_round}:** "

    if order is not None:
        # Execute the order
        execution = mm.process_order(order)

        trader_label = "Informed" if order.is_informed else "Noise"
        side_label = "BUY" if order.is_buy else "SELL"

        if execution:
            log_msg += (
                f"{trader_label} trader {side_label}s at ${execution.price:.2f}. "
                f"Your position: {mm.position:+d}. "
            )
            game["trade_prices"].append(execution.price)
            game["trade_rounds"].append(current_round)
            game["trade_sides"].append("buy" if order.is_buy else "sell")
        else:
            log_msg += "Order did not execute. "
    else:
        log_msg += "No order this round. "

    # Record inventory snapshot
    inv_mgr.record_snapshot(current_round, mm.position, mm.cash, true_value)

    # Update game state
    game["current_round"] = current_round
    game["bids"].append(bid)
    game["asks"].append(ask)
    game["mid_prices"].append((bid + ask) / 2.0)

    if order is not None:
        game["order_log"].append(
            {
                "round": current_round,
                "trader": order.trader_type.value,
                "side": order.side.value,
                "bid": bid,
                "ask": ask,
                "executed": order.executed,
                "exec_price": order.execution_price,
                "true_value": true_value,
                "position": mm.position,
                "pnl": mm.get_total_pnl(true_value),
            }
        )

    # Check if game is over
    if current_round >= game["config"].n_rounds:
        game["game_over"] = True
        log_msg += "**GAME OVER!**"

    return log_msg


def render_price_chart(game: dict):
    """Render the price chart."""
    market = game["market"]
    config = game["config"]
    is_game_over = game["game_over"]

    fig = go.Figure()

    rounds = list(range(len(game["mid_prices"])))

    # Player's mid prices
    fig.add_trace(go.Scatter(
        x=rounds,
        y=game["mid_prices"],
        mode="lines",
        name="Your Mid Price",
        line=dict(color="blue", width=2),
    ))

    # Bid/Ask bands
    if game["bids"]:
        bid_rounds = list(range(1, len(game["bids"]) + 1))
        fig.add_trace(go.Scatter(
            x=bid_rounds,
            y=game["asks"],
            mode="lines",
            name="Ask",
            line=dict(color="rgba(255,0,0,0.3)", dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=bid_rounds,
            y=game["bids"],
            mode="lines",
            name="Bid",
            line=dict(color="rgba(0,128,0,0.3)", dash="dot"),
            fill="tonexty",
            fillcolor="rgba(200,200,200,0.1)",
        ))

    # Trade markers
    if game["trade_rounds"]:
        buy_rounds = [r for r, s in zip(game["trade_rounds"], game["trade_sides"]) if s == "buy"]
        buy_prices = [p for p, s in zip(game["trade_prices"], game["trade_sides"]) if s == "buy"]
        sell_rounds = [r for r, s in zip(game["trade_rounds"], game["trade_sides"]) if s == "sell"]
        sell_prices = [p for p, s in zip(game["trade_prices"], game["trade_sides"]) if s == "sell"]

        if buy_rounds:
            fig.add_trace(go.Scatter(
                x=buy_rounds, y=buy_prices,
                mode="markers", name="Incoming Buy (you sell)",
                marker=dict(color="red", size=10, symbol="triangle-up"),
            ))
        if sell_rounds:
            fig.add_trace(go.Scatter(
                x=sell_rounds, y=sell_prices,
                mode="markers", name="Incoming Sell (you buy)",
                marker=dict(color="green", size=10, symbol="triangle-down"),
            ))

    # Reveal true value after game over
    if is_game_over:
        true_vals = market.true_value_history
        fig.add_trace(go.Scatter(
            x=list(range(len(true_vals))),
            y=true_vals,
            mode="lines",
            name="True Value (revealed)",
            line=dict(color="orange", width=3, dash="dash"),
        ))

    fig.update_layout(
        title="Price Chart" + (" (True value revealed!)" if is_game_over else ""),
        xaxis_title="Round",
        yaxis_title="Price",
        height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_position_chart(game: dict):
    """Render position and PnL over time."""
    inv_mgr = game["inv_mgr"]
    snapshots = inv_mgr.snapshots

    if not snapshots:
        return

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("Position Over Time", "Cumulative PnL"),
        vertical_spacing=0.12,
    )

    rounds = [s.round_number for s in snapshots]
    positions = [s.position for s in snapshots]
    pnls = [s.total_pnl for s in snapshots]

    # Position
    colors = ["green" if p >= 0 else "red" for p in positions]
    fig.add_trace(
        go.Bar(x=rounds, y=positions, marker_color=colors, name="Position", showlegend=False),
        row=1, col=1,
    )

    # Max position lines
    config = game["config"]
    fig.add_hline(y=config.max_position, line_dash="dash", line_color="red",
                  annotation_text="Max Long", row=1, col=1)
    fig.add_hline(y=-config.max_position, line_dash="dash", line_color="red",
                  annotation_text="Max Short", row=1, col=1)

    # PnL
    pnl_colors = ["green" if p >= 0 else "red" for p in pnls]
    fig.add_trace(
        go.Scatter(x=rounds, y=pnls, mode="lines+markers", name="PnL",
                   line=dict(color="blue", width=2), showlegend=False),
        row=2, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    fig.update_layout(height=500)
    fig.update_yaxes(title_text="Units", row=1, col=1)
    fig.update_yaxes(title_text="PnL ($)", row=2, col=1)
    fig.update_xaxes(title_text="Round", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


def render_pnl_decomposition(game: dict):
    """Render PnL decomposition dashboard."""
    mm = game["mm"]
    analytics = game["analytics"]
    market = game["market"]

    if not mm.execution_history:
        st.info("No trades yet. Submit quotes to start trading!")
        return

    current_value = market.true_value
    decomp = analytics.compute_pnl_decomposition(current_value)

    st.subheader("PnL Decomposition")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Spread Capture", f"${decomp.spread_capture:,.2f}")
    with col2:
        color = "inverse" if decomp.adverse_selection < 0 else "normal"
        st.metric("Adverse Selection", f"${decomp.adverse_selection:,.2f}")
    with col3:
        st.metric("Inventory PnL", f"${decomp.inventory_pnl:,.2f}")
    with col4:
        st.metric("Total PnL", f"${decomp.total_pnl:,.2f}")

    # PnL waterfall chart
    fig = go.Figure(go.Waterfall(
        name="PnL",
        orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=["Spread Capture", "Adverse Selection", "Inventory PnL", "Total PnL"],
        y=[decomp.spread_capture, decomp.adverse_selection, decomp.inventory_pnl, 0],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "green"}},
        decreasing={"marker": {"color": "red"}},
        totals={"marker": {"color": "blue"}},
    ))

    fig.update_layout(
        title="PnL Waterfall",
        height=350,
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_game_over(game: dict):
    """Render the game over summary."""
    analytics = game["analytics"]
    market = game["market"]
    config = game["config"]

    true_values = market.true_value_history
    summary = analytics.compute_summary(
        total_rounds=config.n_rounds,
        current_value=market.true_value,
        true_values=true_values,
    )

    st.markdown("---")
    st.header("Game Over - Final Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Final PnL", f"${summary.final_pnl:,.2f}")
    with col2:
        st.metric("Sharpe Ratio", f"{summary.sharpe_ratio:.3f}")
    with col3:
        st.metric("Fill Rate", f"{summary.fill_rate:.1%}")
    with col4:
        st.metric("Max Drawdown", f"${summary.max_drawdown:,.2f}")

    col5, col6, col7, col8 = st.columns(4)

    with col5:
        st.metric("Total Trades", str(summary.total_trades))
    with col6:
        st.metric("Avg Spread", f"${summary.avg_spread:.2f}")
    with col7:
        st.metric("Avg Position", f"{summary.avg_position:.1f}")
    with col8:
        decomp = summary.pnl_decomposition
        st.metric("Buys / Sells", f"{decomp.n_buys} / {decomp.n_sells}")

    # Theoretical comparison
    gm = game["gm_model"]
    eq = gm.compute_equilibrium()
    st.markdown("---")
    st.subheader("Glosten-Milgrom Theoretical Benchmark")
    tcol1, tcol2, tcol3 = st.columns(3)
    with tcol1:
        st.metric("Theoretical Spread", f"${eq.spread:.2f}")
    with tcol2:
        st.metric("Your Avg Spread", f"${summary.avg_spread:.2f}")
    with tcol3:
        diff = summary.avg_spread - eq.spread
        st.metric("Spread Difference", f"${diff:+.2f}")

    # Save to leaderboard
    player_name = st.session_state.get("player_name", "Player")
    leaderboard = load_leaderboard()

    entry = {
        "name": player_name,
        "difficulty": config.name,
        "pnl": round(summary.final_pnl, 2),
        "sharpe": round(summary.sharpe_ratio, 3),
        "trades": summary.total_trades,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    # Check if this game was already saved
    if not any(
        e["name"] == entry["name"]
        and e["date"] == entry["date"]
        and e["pnl"] == entry["pnl"]
        for e in leaderboard
    ):
        leaderboard.append(entry)
        leaderboard.sort(key=lambda x: x["pnl"], reverse=True)
        leaderboard = leaderboard[:50]  # keep top 50
        save_leaderboard(leaderboard)


def render_leaderboard():
    """Render the leaderboard."""
    st.subheader("Leaderboard")
    leaderboard = load_leaderboard()

    if not leaderboard:
        st.info("No scores yet. Be the first to play!")
        return

    df = pd.DataFrame(leaderboard)
    df.index = range(1, len(df) + 1)
    df.columns = ["Player", "Difficulty", "PnL ($)", "Sharpe", "Trades", "Date"]

    st.dataframe(df, use_container_width=True)


def render_order_log(game: dict):
    """Render the order log table."""
    if not game["order_log"]:
        return

    st.subheader("Trade Log")
    df = pd.DataFrame(game["order_log"])
    df.columns = [
        "Round", "Trader", "Side", "Bid", "Ask",
        "Executed", "Exec Price", "True Value", "Position", "PnL"
    ]

    # Format numeric columns
    for col in ["Bid", "Ask", "Exec Price", "True Value", "PnL"]:
        df[col] = df[col].apply(lambda x: f"${x:.2f}" if x is not None else "-")

    st.dataframe(df, use_container_width=True, height=300)


def render_theory_tab():
    """Render the theory/education tab."""
    st.header("Market Microstructure Theory")

    st.markdown("""
    ### The Market Maker's Role

    A **market maker** provides liquidity by continuously quoting **bid** (buy) and **ask** (sell)
    prices. The difference between ask and bid is the **spread**. Market makers earn the spread
    as compensation for:

    1. **Inventory risk**: holding a position that could move against them
    2. **Adverse selection**: trading against informed traders who know the true value
    3. **Operating costs**: technology, capital, and regulatory costs

    ### The Glosten-Milgrom Model (1985)

    The GM model explains why bid-ask spreads exist even with zero operating costs.

    **Setup:**
    - An asset has an unknown true value V, which is either V_H (high) or V_L (low)
    - Prior probability of V = V_H is mu (typically 0.5)
    - A fraction alpha of traders are **informed** (they know V)
    - The remaining (1 - alpha) are **noise traders** (trade randomly)

    **Key insight:** When a buy order arrives, the market maker updates their belief upward
    (the order might be from an informed trader who knows V = V_H). Similarly, sell orders
    cause downward belief updates.

    **Equilibrium prices** (using Bayes' rule):
    - **Ask** = E[V | buy order arrives]
    - **Bid** = E[V | sell order arrives]

    The spread arises because buy orders are more likely when V = V_H, and sell orders
    are more likely when V = V_L.

    **Simplified spread formula** (when mu = 0.5):

    spread = 2 * alpha * (V_H - V_L) / (1 + alpha)

    ### Adverse Selection

    When you trade against an **informed** trader, you systematically lose money:
    - If an informed trader buys from you (at your ask), the true value is likely above your ask
    - If an informed trader sells to you (at your bid), the true value is likely below your bid

    This is **adverse selection** -- the informed trader's gain is your loss.

    ### Inventory Management

    As a market maker, you accumulate inventory (positive or negative position) from trades.
    Large positions create risk because the asset value may move against you.

    **Strategies:**
    - **Skew your quotes**: if you are long, lower both bid and ask to attract sellers
    - **Widen the spread**: reduces trading volume but protects against adverse moves
    - **Manage position limits**: avoid excessive exposure

    ### PnL Decomposition

    Your profit/loss decomposes into:
    1. **Spread capture**: half-spread earned on each trade
    2. **Adverse selection cost**: losses from trading against informed traders
    3. **Inventory PnL**: mark-to-market gains/losses from holding inventory

    **Total PnL = Spread Capture - Adverse Selection + Inventory PnL**
    """)


def main():
    """Main Streamlit application."""
    config, seed = render_sidebar()

    # Initialize game state if not exists
    if "game" not in st.session_state:
        st.session_state["game"] = init_game_state(
            get_difficulty_config(DifficultyLevel.MEDIUM), seed=42
        )

    game = st.session_state["game"]

    # Main tabs
    tab_game, tab_analytics, tab_leaderboard, tab_theory = st.tabs(
        ["Game", "Analytics", "Leaderboard", "Theory"]
    )

    with tab_game:
        st.header("Market Making Simulator")

        # Game info bar
        render_game_info(game)
        st.markdown("---")

        if not game["game_over"]:
            # Quote input
            result = render_quote_input(game)

            if result is not None:
                bid, ask = result
                log_msg = process_round(game, bid, ask)
                st.success(log_msg)
                st.rerun()

            # Show recent activity
            if game["order_log"]:
                st.markdown("---")
                recent = game["order_log"][-5:]
                for entry in reversed(recent):
                    trader = entry["trader"]
                    side = entry["side"]
                    executed = entry["executed"]
                    price = entry["exec_price"]
                    rnd = entry["round"]
                    if executed and price is not None:
                        st.text(
                            f"Round {rnd}: {trader} trader {side}s at ${price:.2f} | "
                            f"Position: {entry['position']:+d} | PnL: ${entry['pnl']:.2f}"
                        )
        else:
            render_game_over(game)

        # Price chart (always visible)
        st.markdown("---")
        render_price_chart(game)

    with tab_analytics:
        render_pnl_decomposition(game)
        render_position_chart(game)
        render_order_log(game)

    with tab_leaderboard:
        render_leaderboard()

    with tab_theory:
        render_theory_tab()


if __name__ == "__main__":
    main()
