"""Tests for the market simulation module."""

import numpy as np
import pytest

from src.core.market import MarketSimulator


class TestMarketSimulator:
    """Tests for MarketSimulator."""

    def test_initial_value(self):
        """True value starts at initial_value."""
        sim = MarketSimulator(initial_value=100.0, seed=42)
        assert sim.true_value == 100.0
        assert sim.current_round == 0

    def test_step_advances_round(self):
        """Each step increments the round counter."""
        sim = MarketSimulator(seed=42)
        sim.step()
        assert sim.current_round == 1
        sim.step()
        assert sim.current_round == 2

    def test_step_changes_value(self):
        """Steps produce different values (with non-zero sigma)."""
        sim = MarketSimulator(sigma=1.0, seed=42)
        values = [sim.step() for _ in range(10)]
        # With sigma=1.0, values should not all be equal
        assert len(set(values)) > 1

    def test_zero_volatility(self):
        """With zero sigma and no jumps, value stays constant."""
        sim = MarketSimulator(
            initial_value=50.0, sigma=0.0,
            jump_intensity=0.0, jump_size=0.0, seed=42
        )
        for _ in range(20):
            sim.step()
        assert sim.true_value == 50.0

    def test_history_length(self):
        """History has correct length after steps."""
        sim = MarketSimulator(seed=42)
        n = 15
        sim.generate_path(n)
        # History includes initial value + n steps
        assert len(sim.true_value_history) == n + 1

    def test_generate_path_returns_all(self):
        """generate_path returns the complete path."""
        sim = MarketSimulator(seed=42)
        path = sim.generate_path(10)
        assert len(path) == 11
        assert path[0] == sim.initial_value

    def test_reset(self):
        """Reset restores initial state."""
        sim = MarketSimulator(initial_value=100.0, seed=42)
        sim.generate_path(10)
        sim.reset(seed=42)
        assert sim.true_value == 100.0
        assert sim.current_round == 0
        assert len(sim.true_value_history) == 1

    def test_reproducibility(self):
        """Same seed produces same path."""
        sim1 = MarketSimulator(seed=123)
        path1 = sim1.generate_path(20)

        sim2 = MarketSimulator(seed=123)
        path2 = sim2.generate_path(20)

        np.testing.assert_array_almost_equal(path1, path2)

    def test_different_seeds_differ(self):
        """Different seeds produce different paths."""
        sim1 = MarketSimulator(seed=1)
        path1 = sim1.generate_path(20)

        sim2 = MarketSimulator(seed=2)
        path2 = sim2.generate_path(20)

        # Very unlikely to be exactly equal with different seeds
        assert not np.allclose(path1, path2)

    def test_jumps_occur(self):
        """With high jump intensity, jumps should be evident."""
        sim = MarketSimulator(
            initial_value=100.0,
            sigma=0.0,  # No diffusion, only jumps
            jump_intensity=5.0,  # Very high jump rate
            jump_size=10.0,
            seed=42,
        )
        path = sim.generate_path(50)
        # With high jump intensity and zero sigma,
        # the path should show discrete jumps
        diffs = np.diff(path)
        # Some diffs should be approximately multiples of jump_size
        large_moves = [d for d in diffs if abs(d) > 5.0]
        assert len(large_moves) > 0

    def test_get_value_range(self):
        """Value range returns correct min/max."""
        sim = MarketSimulator(sigma=1.0, seed=42)
        sim.generate_path(20)
        vmin, vmax = sim.get_value_range(lookback=5)
        recent = sim.true_value_history[-5:]
        assert vmin == min(recent)
        assert vmax == max(recent)
