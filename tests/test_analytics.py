"""Tests for the analytics and PnL decomposition module."""

import numpy as np
import pytest

from src.core.market import MarketSimulator
from src.core.order_flow import OrderFlowGenerator
from src.core.market_maker import MarketMaker, Quote
from src.core.inventory import InventoryManager
from src.core.analytics import AnalyticsEngine, PnLDecomposition


class TestPnLAccounting:
    """Test that PnL accounting balances correctly."""

    def _run_game(self, n_rounds: int = 30, seed: int = 42) -> tuple:
        """Run a simulated game and return all components."""
        market = MarketSimulator(initial_value=100.0, sigma=0.5, seed=seed)
        order_gen = OrderFlowGenerator(alpha=0.3, seed=seed + 1)
        mm = MarketMaker(initial_cash=10000.0)
        inv_mgr = InventoryManager(max_position=20, initial_cash=10000.0)
        analytics = AnalyticsEngine(mm, inv_mgr)

        bid, ask = 99.0, 101.0

        for round_num in range(1, n_rounds + 1):
            true_value = market.step()
            mm.set_quote(bid, ask, round_num)
            order = order_gen.generate_order(round_num, true_value, bid, ask)
            if order is not None:
                mm.process_order(order)
            inv_mgr.record_snapshot(round_num, mm.position, mm.cash, true_value)

            # Simple adaptive quoting
            mid = true_value  # cheat for testing purposes
            bid = mid - 1.0
            ask = mid + 1.0

        return market, mm, inv_mgr, analytics

    def test_cash_flow_accounting(self):
        """Cash changes must match sum of execution prices."""
        _, mm, _, _ = self._run_game()

        # Total cash spent on buys
        cash_spent = sum(
            e.price for e in mm.execution_history if e.mm_side == "buy"
        )
        # Total cash received from sells
        cash_received = sum(
            e.price for e in mm.execution_history if e.mm_side == "sell"
        )

        expected_cash = mm.initial_cash - cash_spent + cash_received
        assert abs(mm.cash - expected_cash) < 1e-10, (
            f"Cash mismatch: actual={mm.cash}, expected={expected_cash}"
        )

    def test_position_accounting(self):
        """Position must equal buys minus sells."""
        _, mm, _, _ = self._run_game()

        n_buys = sum(1 for e in mm.execution_history if e.mm_side == "buy")
        n_sells = sum(1 for e in mm.execution_history if e.mm_side == "sell")

        expected_position = n_buys - n_sells
        assert mm.position == expected_position

    def test_total_pnl_consistency(self):
        """Total PnL must equal cash change + mark-to-market."""
        market, mm, _, analytics = self._run_game()
        current_value = market.true_value

        total_pnl = mm.get_total_pnl(current_value)
        cash_pnl = mm.cash - mm.initial_cash
        mtm = mm.position * current_value

        assert abs(total_pnl - (cash_pnl + mtm)) < 1e-10

    def test_pnl_decomposition_components(self):
        """PnL decomposition components should be computed."""
        market, mm, _, analytics = self._run_game()

        decomp = analytics.compute_pnl_decomposition(market.true_value)

        # Spread capture should be non-negative (always earn half-spread)
        assert decomp.spread_capture >= 0

        # Should have some trades
        assert decomp.n_trades > 0
        assert decomp.n_trades == decomp.n_buys + decomp.n_sells

    def test_spread_capture_matches_trades(self):
        """Spread capture should equal sum of half-spreads."""
        _, mm, _, analytics = self._run_game()

        expected_spread = sum(e.spread_earned for e in mm.execution_history)
        decomp = analytics.compute_pnl_decomposition(100.0)

        assert abs(decomp.spread_capture - expected_spread) < 1e-10

    def test_fill_rate_bounds(self):
        """Fill rate should be between 0 and 1."""
        _, _, _, analytics = self._run_game(n_rounds=30)
        fill_rate = analytics.compute_fill_rate(30)
        assert 0.0 <= fill_rate <= 1.0

    def test_sharpe_ratio_defined(self):
        """Sharpe ratio should be a finite number after enough rounds."""
        market, _, _, analytics = self._run_game(n_rounds=30)
        sharpe = analytics.compute_sharpe_ratio(market.true_value_history)
        assert np.isfinite(sharpe)

    def test_summary_complete(self):
        """Summary should contain all fields."""
        market, _, _, analytics = self._run_game(n_rounds=30)
        summary = analytics.compute_summary(
            total_rounds=30,
            current_value=market.true_value,
            true_values=market.true_value_history,
        )

        assert summary.total_rounds == 30
        assert summary.total_trades >= 0
        assert 0.0 <= summary.fill_rate <= 1.0
        assert np.isfinite(summary.sharpe_ratio)
        assert summary.max_drawdown >= 0
        assert summary.avg_spread >= 0
        assert summary.avg_position >= 0

    def test_empty_game(self):
        """Analytics should handle empty game (no trades)."""
        mm = MarketMaker(initial_cash=10000.0)
        inv_mgr = InventoryManager(initial_cash=10000.0)
        analytics = AnalyticsEngine(mm, inv_mgr)

        decomp = analytics.compute_pnl_decomposition(100.0)
        assert decomp.n_trades == 0
        assert decomp.spread_capture == 0.0


class TestInventoryManager:
    """Tests for inventory tracking."""

    def test_snapshot_recording(self):
        """Snapshots record correctly."""
        inv = InventoryManager(initial_cash=10000.0)
        snap = inv.record_snapshot(1, position=5, cash=9500.0, true_value=100.0)
        assert snap.position == 5
        assert snap.mark_to_market == 500.0
        assert snap.total_pnl == (9500.0 - 10000.0) + 500.0

    def test_position_breach(self):
        """Position breach detection works."""
        inv = InventoryManager(max_position=10)
        assert not inv.is_position_breached(10)
        assert inv.is_position_breached(11)
        assert inv.is_position_breached(-11)
        assert not inv.is_position_breached(-10)

    def test_inventory_penalty_quadratic(self):
        """Penalty is quadratic in position."""
        inv = InventoryManager(inventory_penalty_coeff=0.01)
        assert inv.inventory_penalty(0) == 0.0
        assert abs(inv.inventory_penalty(5) - 0.25) < 1e-10
        assert abs(inv.inventory_penalty(-5) - 0.25) < 1e-10
        assert abs(inv.inventory_penalty(10) - 1.0) < 1e-10

    def test_suggested_skew_direction(self):
        """Skew direction is correct."""
        inv = InventoryManager()
        # Long position -> negative skew (lower prices to attract sells)
        assert inv.suggested_skew(5, 2.0) < 0
        # Short position -> positive skew (raise prices to attract buys)
        assert inv.suggested_skew(-5, 2.0) > 0
        # Flat position -> no skew
        assert inv.suggested_skew(0, 2.0) == 0.0

    def test_drawdown_calculation(self):
        """Max drawdown is calculated correctly."""
        inv = InventoryManager(initial_cash=10000.0)
        # Simulate: PnL goes 0, 10, 20, 5, 15, 2
        pnl_sequence = [0, 10, 20, 5, 15, 2]
        for i, pnl in enumerate(pnl_sequence):
            cash = 10000.0 + pnl  # simplified
            inv.record_snapshot(i + 1, position=0, cash=cash, true_value=100.0)

        max_dd = inv.max_position_pnl_drawdown()
        # Max drawdown should be 20 - 2 = 18? No, 20 - 5 = 15 then 15 - 2 = 13
        # Peak was 20 at index 2, then drops to 5 (dd=15), recovers to 15, drops to 2 (dd=18)
        assert abs(max_dd - 18.0) < 1e-10
