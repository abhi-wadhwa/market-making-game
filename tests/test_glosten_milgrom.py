"""Tests for the Glosten-Milgrom model."""

import numpy as np
import pytest

from src.core.glosten_milgrom import GlostenMilgromModel, GMEquilibrium


class TestGlostenMilgromModel:
    """Tests for the Glosten-Milgrom equilibrium computation."""

    def test_zero_alpha_zero_spread(self):
        """With no informed traders, the spread should be zero."""
        gm = GlostenMilgromModel(v_high=110.0, v_low=90.0, alpha=0.0, mu=0.5)
        eq = gm.compute_equilibrium()
        assert abs(eq.spread) < 1e-10

    def test_positive_alpha_positive_spread(self):
        """With informed traders, the spread should be positive."""
        gm = GlostenMilgromModel(v_high=110.0, v_low=90.0, alpha=0.3, mu=0.5)
        eq = gm.compute_equilibrium()
        assert eq.spread > 0
        assert eq.ask > eq.bid

    def test_higher_alpha_wider_spread(self):
        """Higher adverse selection intensity should widen the spread."""
        gm_low = GlostenMilgromModel(v_high=110.0, v_low=90.0, alpha=0.1, mu=0.5)
        gm_high = GlostenMilgromModel(v_high=110.0, v_low=90.0, alpha=0.5, mu=0.5)

        spread_low = gm_low.compute_equilibrium().spread
        spread_high = gm_high.compute_equilibrium().spread

        assert spread_high > spread_low

    def test_symmetric_with_equal_prior(self):
        """With mu=0.5, the mid price should be the unconditional mean."""
        gm = GlostenMilgromModel(v_high=110.0, v_low=90.0, alpha=0.3, mu=0.5)
        eq = gm.compute_equilibrium()

        unconditional_mean = 0.5 * 110.0 + 0.5 * 90.0  # = 100.0
        assert abs(eq.mid - unconditional_mean) < 1e-10

    def test_ask_above_unconditional_mean(self):
        """Ask should be above the unconditional mean (buy is bullish signal)."""
        gm = GlostenMilgromModel(v_high=110.0, v_low=90.0, alpha=0.3, mu=0.5)
        eq = gm.compute_equilibrium()
        unconditional_mean = 100.0
        assert eq.ask > unconditional_mean

    def test_bid_below_unconditional_mean(self):
        """Bid should be below the unconditional mean (sell is bearish signal)."""
        gm = GlostenMilgromModel(v_high=110.0, v_low=90.0, alpha=0.3, mu=0.5)
        eq = gm.compute_equilibrium()
        unconditional_mean = 100.0
        assert eq.bid < unconditional_mean

    def test_full_alpha_equals_value_range(self):
        """With alpha=1 (all informed), ask=V_H and bid=V_L."""
        gm = GlostenMilgromModel(v_high=110.0, v_low=90.0, alpha=1.0, mu=0.5)
        eq = gm.compute_equilibrium()
        # With all informed traders:
        # Buy always means V=V_H, Sell always means V=V_L
        assert abs(eq.ask - 110.0) < 1e-10
        assert abs(eq.bid - 90.0) < 1e-10

    def test_zero_profit_at_equilibrium(self):
        """Expected profit should be approximately zero at equilibrium."""
        gm = GlostenMilgromModel(v_high=110.0, v_low=90.0, alpha=0.3, mu=0.5)
        profit = gm.expected_profit_per_trade()
        assert abs(profit) < 1e-10

    def test_posterior_update_buy(self):
        """After a buy order, P(V_H) should increase."""
        gm = GlostenMilgromModel(v_high=110.0, v_low=90.0, alpha=0.3, mu=0.5)
        posterior = gm.update_beliefs(order_is_buy=True, mu=0.5)
        assert posterior > 0.5

    def test_posterior_update_sell(self):
        """After a sell order, P(V_H) should decrease."""
        gm = GlostenMilgromModel(v_high=110.0, v_low=90.0, alpha=0.3, mu=0.5)
        posterior = gm.update_beliefs(order_is_buy=False, mu=0.5)
        assert posterior < 0.5

    def test_spread_formula_consistency(self):
        """Verify spread matches the Bayesian derivation.

        With mu = 0.5:
        Ask = V_L + (V_H - V_L) * P(V_H | buy)
        Bid = V_L + (V_H - V_L) * P(V_H | sell)
        Spread = (V_H - V_L) * (P(V_H | buy) - P(V_H | sell))
        """
        v_h, v_l = 110.0, 90.0
        alpha = 0.3
        gm = GlostenMilgromModel(v_high=v_h, v_low=v_l, alpha=alpha, mu=0.5)
        eq = gm.compute_equilibrium()

        # Manual computation
        p_buy_h = alpha + (1 - alpha) * 0.5  # P(buy | V_H)
        p_buy_l = (1 - alpha) * 0.5  # P(buy | V_L)
        p_h_buy = (p_buy_h * 0.5) / (p_buy_h * 0.5 + p_buy_l * 0.5)

        p_sell_h = (1 - alpha) * 0.5
        p_sell_l = alpha + (1 - alpha) * 0.5
        p_h_sell = (p_sell_h * 0.5) / (p_sell_h * 0.5 + p_sell_l * 0.5)

        expected_ask = p_h_buy * v_h + (1 - p_h_buy) * v_l
        expected_bid = p_h_sell * v_h + (1 - p_h_sell) * v_l

        assert abs(eq.ask - expected_ask) < 1e-10
        assert abs(eq.bid - expected_bid) < 1e-10
        assert abs(eq.spread - (expected_ask - expected_bid)) < 1e-10

    def test_theoretical_spread_for_alpha(self):
        """theoretical_spread_for_alpha should be consistent."""
        gm = GlostenMilgromModel(v_high=110.0, v_low=90.0, alpha=0.3, mu=0.5)

        spread_03 = gm.theoretical_spread_for_alpha(0.3)
        spread_05 = gm.theoretical_spread_for_alpha(0.5)

        assert spread_05 > spread_03
        assert abs(spread_03 - gm.compute_equilibrium().spread) < 1e-10

    def test_invalid_alpha_raises(self):
        """Invalid alpha should raise ValueError."""
        with pytest.raises(ValueError):
            GlostenMilgromModel(v_high=110.0, v_low=90.0, alpha=-0.1)
        with pytest.raises(ValueError):
            GlostenMilgromModel(v_high=110.0, v_low=90.0, alpha=1.5)

    def test_invalid_values_raises(self):
        """v_high <= v_low should raise ValueError."""
        with pytest.raises(ValueError):
            GlostenMilgromModel(v_high=90.0, v_low=110.0)
        with pytest.raises(ValueError):
            GlostenMilgromModel(v_high=100.0, v_low=100.0)

    def test_relative_spread(self):
        """Relative spread should be spread / mid."""
        gm = GlostenMilgromModel(v_high=110.0, v_low=90.0, alpha=0.3, mu=0.5)
        eq = gm.compute_equilibrium()
        expected = eq.spread / eq.mid
        assert abs(eq.relative_spread - expected) < 1e-10

    def test_equilibrium_dataclass(self):
        """GMEquilibrium has all expected fields."""
        gm = GlostenMilgromModel(v_high=110.0, v_low=90.0, alpha=0.3, mu=0.5)
        eq = gm.compute_equilibrium()
        assert hasattr(eq, "bid")
        assert hasattr(eq, "ask")
        assert hasattr(eq, "spread")
        assert hasattr(eq, "relative_spread")
        assert hasattr(eq, "mid")
        assert hasattr(eq, "prob_high_given_buy")
        assert hasattr(eq, "prob_high_given_sell")
