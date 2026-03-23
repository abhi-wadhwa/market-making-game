"""Tests for the order flow generation module."""

import numpy as np
import pytest

from src.core.order_flow import OrderFlowGenerator, Order, TraderType, OrderSide


class TestOrderFlowGenerator:
    """Tests for OrderFlowGenerator."""

    def test_alpha_bounds(self):
        """Alpha must be in [0, 1]."""
        with pytest.raises(ValueError):
            OrderFlowGenerator(alpha=-0.1)
        with pytest.raises(ValueError):
            OrderFlowGenerator(alpha=1.5)

    def test_arrival_rate_positive(self):
        """Arrival rate must be positive."""
        with pytest.raises(ValueError):
            OrderFlowGenerator(alpha=0.3, arrival_rate=0.0)
        with pytest.raises(ValueError):
            OrderFlowGenerator(alpha=0.3, arrival_rate=-1.0)

    def test_generates_order(self):
        """generate_order returns an Order object."""
        gen = OrderFlowGenerator(alpha=0.3, seed=42)
        order = gen.generate_order(round_number=1, true_value=100.0, bid=99.0, ask=101.0)
        assert isinstance(order, Order)
        assert order.round_number == 1
        assert order.true_value == 100.0

    def test_informed_vs_noise_ratio(self):
        """Fraction of informed traders should approximate alpha."""
        alpha = 0.4
        gen = OrderFlowGenerator(alpha=alpha, seed=42)
        n_trials = 5000
        informed_count = 0
        for i in range(n_trials):
            order = gen.generate_order(i, 100.0, 99.0, 101.0)
            if order is not None and order.is_informed:
                informed_count += 1

        observed_ratio = informed_count / n_trials
        # Should be close to alpha (within statistical tolerance)
        assert abs(observed_ratio - alpha) < 0.05

    def test_informed_trader_buys_when_underpriced(self):
        """Informed trader buys when true value > ask."""
        gen = OrderFlowGenerator(alpha=1.0, seed=42)  # 100% informed
        buy_count = 0
        n_trials = 100
        for i in range(n_trials):
            order = gen.generate_order(i, true_value=110.0, bid=99.0, ask=101.0)
            if order is not None and order.is_buy:
                buy_count += 1
        # All informed traders should buy when value >> ask
        assert buy_count == n_trials

    def test_informed_trader_sells_when_overpriced(self):
        """Informed trader sells when true value < bid."""
        gen = OrderFlowGenerator(alpha=1.0, seed=42)
        sell_count = 0
        n_trials = 100
        for i in range(n_trials):
            order = gen.generate_order(i, true_value=90.0, bid=99.0, ask=101.0)
            if order is not None and not order.is_buy:
                sell_count += 1
        assert sell_count == n_trials

    def test_noise_trader_random_direction(self):
        """Noise traders should have approximately 50/50 buy/sell."""
        gen = OrderFlowGenerator(alpha=0.0, seed=42)  # 100% noise
        buy_count = 0
        n_trials = 5000
        for i in range(n_trials):
            order = gen.generate_order(i, 100.0, 99.0, 101.0)
            if order is not None and order.is_buy:
                buy_count += 1
        ratio = buy_count / n_trials
        assert abs(ratio - 0.5) < 0.05

    def test_order_not_executed_initially(self):
        """Orders start as not executed."""
        gen = OrderFlowGenerator(alpha=0.3, seed=42)
        order = gen.generate_order(1, 100.0, 99.0, 101.0)
        assert order is not None
        assert not order.executed
        assert order.execution_price is None

    def test_low_arrival_rate_skips(self):
        """Low arrival rate sometimes produces no order."""
        gen = OrderFlowGenerator(alpha=0.3, arrival_rate=0.3, seed=42)
        results = [gen.generate_order(i, 100.0, 99.0, 101.0) for i in range(100)]
        none_count = sum(1 for r in results if r is None)
        # With arrival_rate=0.3, roughly 70% should be None
        assert none_count > 40

    def test_batch_generation(self):
        """Batch generation respects arrival rate."""
        gen = OrderFlowGenerator(alpha=0.3, arrival_rate=2.0, seed=42)
        all_orders = []
        for i in range(100):
            batch = gen.generate_orders_batch(i, 100.0, 99.0, 101.0)
            all_orders.extend(batch)
        # With arrival_rate=2.0, average batch size should be ~2
        avg_batch = len(all_orders) / 100
        assert 1.0 < avg_batch < 3.0

    def test_order_properties(self):
        """Order dataclass properties work correctly."""
        order = Order(
            round_number=5,
            trader_type=TraderType.INFORMED,
            side=OrderSide.BUY,
            true_value=105.0,
        )
        assert order.is_buy
        assert order.is_informed
        assert order.round_number == 5
