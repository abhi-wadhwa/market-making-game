"""Demo script showing the market-making game engine in action.

This script runs an automated game where a simple strategy
quotes around the current mid price with a fixed spread.
It demonstrates all core components working together.

Usage:
    python examples/demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.core.market import MarketSimulator
from src.core.order_flow import OrderFlowGenerator
from src.core.market_maker import MarketMaker
from src.core.inventory import InventoryManager
from src.core.glosten_milgrom import GlostenMilgromModel
from src.core.analytics import AnalyticsEngine
from src.core.difficulty import DifficultyLevel, get_difficulty_config


def run_demo():
    """Run a demo game with an automated quoting strategy."""
    print("=" * 60)
    print("  Market Making Game - Automated Demo")
    print("=" * 60)

    # Use medium difficulty
    config = get_difficulty_config(DifficultyLevel.MEDIUM)
    seed = 42

    # Initialize components
    market = MarketSimulator(
        initial_value=config.initial_value,
        sigma=config.sigma,
        jump_intensity=config.jump_intensity,
        jump_size=config.jump_size,
        seed=seed,
    )
    order_gen = OrderFlowGenerator(alpha=config.alpha, seed=seed + 1)
    mm = MarketMaker(initial_cash=config.initial_cash)
    inv_mgr = InventoryManager(
        max_position=config.max_position,
        initial_cash=config.initial_cash,
        inventory_penalty_coeff=config.inventory_penalty,
    )
    gm = GlostenMilgromModel(
        v_high=config.initial_value + 3 * config.sigma * np.sqrt(config.n_rounds),
        v_low=config.initial_value - 3 * config.sigma * np.sqrt(config.n_rounds),
        alpha=config.alpha,
    )
    analytics = AnalyticsEngine(mm, inv_mgr)

    print(f"\nDifficulty: {config.name}")
    print(f"Rounds: {config.n_rounds}")
    print(f"Alpha (informed prob): {config.alpha}")
    print(f"Sigma (volatility): {config.sigma}")
    print()

    # Simple strategy: quote with a fixed spread, skewed by inventory
    base_spread = 2.0
    mid_estimate = config.initial_value

    for round_num in range(1, config.n_rounds + 1):
        # Skew quotes based on inventory
        skew = inv_mgr.suggested_skew(mm.position, base_spread)
        bid = mid_estimate + skew - base_spread / 2
        ask = mid_estimate + skew + base_spread / 2

        # Advance the true value
        true_value = market.step()

        # Set our quote
        mm.set_quote(bid, ask, round_num)

        # Generate and process order
        order = order_gen.generate_order(round_num, true_value, bid, ask)
        if order is not None:
            execution = mm.process_order(order)
            if execution:
                trader = "INF" if order.is_informed else "NSE"
                side = "BUY" if order.is_buy else "SEL"
                print(
                    f"  R{round_num:3d}: {trader} {side} @ ${execution.price:.2f} | "
                    f"TV=${true_value:.2f} | Pos={mm.position:+3d} | "
                    f"PnL=${mm.get_total_pnl(true_value):+8.2f}"
                )

        # Record snapshot
        inv_mgr.record_snapshot(round_num, mm.position, mm.cash, true_value)

        # Update mid estimate (in real game, player does not know true value)
        # Strategy: use a simple moving average of trade prices
        if mm.execution_history:
            recent_trades = mm.execution_history[-5:]
            mid_estimate = np.mean([e.price for e in recent_trades])
        else:
            mid_estimate = config.initial_value

    # Print results
    print("\n" + "=" * 60)
    print("  Results")
    print("=" * 60)

    summary = analytics.compute_summary(
        total_rounds=config.n_rounds,
        current_value=market.true_value,
        true_values=market.true_value_history,
    )
    decomp = summary.pnl_decomposition

    print(f"\n  Final PnL:          ${summary.final_pnl:+,.2f}")
    print(f"  Sharpe Ratio:       {summary.sharpe_ratio:.3f}")
    print(f"  Fill Rate:          {summary.fill_rate:.1%}")
    print(f"  Max Drawdown:       ${summary.max_drawdown:,.2f}")
    print(f"  Total Trades:       {summary.total_trades}")
    print(f"  Avg Spread:         ${summary.avg_spread:.2f}")
    print(f"  Avg |Position|:     {summary.avg_position:.1f}")

    print(f"\n  PnL Breakdown:")
    print(f"    Spread Capture:     ${decomp.spread_capture:+,.2f}")
    print(f"    Adverse Selection:  ${decomp.adverse_selection:+,.2f}")
    print(f"    Inventory PnL:      ${decomp.inventory_pnl:+,.2f}")

    eq = gm.compute_equilibrium()
    print(f"\n  GM Theoretical Spread: ${eq.spread:.2f}")
    print(f"  Your Average Spread:   ${summary.avg_spread:.2f}")

    print(f"\n  True Value: ${market.true_value_history[0]:.2f} -> ${market.true_value:.2f}")
    print()


if __name__ == "__main__":
    run_demo()
