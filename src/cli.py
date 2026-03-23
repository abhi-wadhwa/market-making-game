"""Command-line interface for the market-making game.

Play the game in the terminal without Streamlit.

Usage:
    python -m src.cli [--difficulty LEVEL] [--rounds N] [--seed S]
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from src.core.market import MarketSimulator
from src.core.order_flow import OrderFlowGenerator
from src.core.market_maker import MarketMaker
from src.core.inventory import InventoryManager
from src.core.glosten_milgrom import GlostenMilgromModel
from src.core.analytics import AnalyticsEngine
from src.core.difficulty import (
    DifficultyLevel,
    DifficultyConfig,
    DIFFICULTY_PRESETS,
    get_difficulty_by_name,
)

import numpy as np


def print_banner():
    """Print the game banner."""
    print("=" * 60)
    print("       MARKET MAKING GAME - CLI Edition")
    print("=" * 60)
    print()


def print_round_header(round_num: int, total: int, position: int, cash: float, pnl: float):
    """Print the round information header."""
    print(f"\n--- Round {round_num}/{total} ---")
    print(f"  Position: {position:+d}  |  Cash: ${cash:,.2f}  |  PnL: ${pnl:,.2f}")


def get_quote(default_bid: float, default_ask: float) -> tuple[float, float]:
    """Get bid and ask from the player."""
    while True:
        try:
            bid_str = input(f"  Enter bid price [{default_bid:.2f}]: ").strip()
            bid = float(bid_str) if bid_str else default_bid

            ask_str = input(f"  Enter ask price [{default_ask:.2f}]: ").strip()
            ask = float(ask_str) if ask_str else default_ask

            if bid >= ask:
                print("  Error: bid must be less than ask. Try again.")
                continue

            print(f"  Your quote: {bid:.2f} / {ask:.2f}  (spread: {ask - bid:.2f})")
            return bid, ask
        except ValueError:
            print("  Invalid input. Enter a number.")
        except (EOFError, KeyboardInterrupt):
            print("\n  Game aborted.")
            sys.exit(0)


def run_game(config: DifficultyConfig, seed: Optional[int] = None):
    """Run the game loop."""
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

    print(f"Difficulty: {config.name}")
    print(f"  {config.description}")
    print(f"  Rounds: {config.n_rounds} | Alpha: {config.alpha} | Sigma: {config.sigma}")
    print(f"  Max position: +/-{config.max_position}")
    print()

    default_bid = config.initial_value - 1.0
    default_ask = config.initial_value + 1.0

    for round_num in range(1, config.n_rounds + 1):
        mid = (default_bid + default_ask) / 2.0
        current_pnl = mm.get_total_pnl(mid)
        print_round_header(round_num, config.n_rounds, mm.position, mm.cash, current_pnl)

        # Position warning
        if abs(mm.position) > config.max_position * 0.7:
            direction = "LONG" if mm.position > 0 else "SHORT"
            print(f"  ** WARNING: Large {direction} position! Consider skewing quotes. **")

        # Get player's quote
        bid, ask = get_quote(default_bid, default_ask)

        # Advance the market
        true_value = market.step()

        # Set quote
        mm.set_quote(bid, ask, round_num)

        # Generate order
        order = order_gen.generate_order(round_num, true_value, bid, ask)

        if order is not None:
            execution = mm.process_order(order)
            trader_label = "INFORMED" if order.is_informed else "NOISE"
            side_label = "BUYS" if order.is_buy else "SELLS"

            if execution:
                print(
                    f"  >> {trader_label} trader {side_label} at ${execution.price:.2f}"
                    f"  (your position now: {mm.position:+d})"
                )
            else:
                print("  >> Order did not execute.")
        else:
            print("  >> No order this round.")

        # Record snapshot
        inv_mgr.record_snapshot(round_num, mm.position, mm.cash, true_value)

        # Update defaults for next round
        default_bid = bid
        default_ask = ask

    # Game over
    print("\n" + "=" * 60)
    print("                    GAME OVER")
    print("=" * 60)

    true_values = market.true_value_history
    summary = analytics.compute_summary(
        total_rounds=config.n_rounds,
        current_value=market.true_value,
        true_values=true_values,
    )

    decomp = summary.pnl_decomposition

    print(f"\n  Final PnL:        ${summary.final_pnl:,.2f}")
    print(f"  Sharpe Ratio:     {summary.sharpe_ratio:.3f}")
    print(f"  Fill Rate:        {summary.fill_rate:.1%}")
    print(f"  Max Drawdown:     ${summary.max_drawdown:,.2f}")
    print(f"  Total Trades:     {summary.total_trades}")
    print(f"  Avg Spread:       ${summary.avg_spread:.2f}")
    print(f"  Avg |Position|:   {summary.avg_position:.1f}")

    print(f"\n  PnL Breakdown:")
    print(f"    Spread Capture:     ${decomp.spread_capture:,.2f}")
    print(f"    Adverse Selection:  ${decomp.adverse_selection:,.2f}")
    print(f"    Inventory PnL:      ${decomp.inventory_pnl:,.2f}")

    # Theoretical comparison
    eq = gm_model.compute_equilibrium()
    print(f"\n  Glosten-Milgrom Benchmark:")
    print(f"    Theoretical Spread: ${eq.spread:.2f}")
    print(f"    Your Avg Spread:    ${summary.avg_spread:.2f}")

    print(f"\n  True value path (first visible now):")
    print(f"    Start: ${true_values[0]:.2f}  ->  End: ${true_values[-1]:.2f}")
    print(f"    Range: ${min(true_values):.2f} - ${max(true_values):.2f}")
    print()


def main():
    """Entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Market Making Game - CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="medium",
        choices=[l.value for l in DifficultyLevel],
        help="Difficulty level (default: medium)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    print_banner()
    config = get_difficulty_by_name(args.difficulty)
    run_game(config, seed=args.seed)


if __name__ == "__main__":
    main()
