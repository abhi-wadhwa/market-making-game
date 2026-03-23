"""Market simulation with hidden true value.

The true asset value follows a Brownian motion with Poisson jumps:
    V(t) = V(t-1) + sigma * epsilon + jump_size * Poisson(lambda)

The true value is hidden from the market maker during gameplay and
revealed only after the game ends.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MarketState:
    """Snapshot of the market at a given round."""

    round_number: int
    true_value: float
    mid_price: float  # player's implied mid from their quotes
    bid: Optional[float] = None
    ask: Optional[float] = None
    last_trade_price: Optional[float] = None
    last_trade_side: Optional[str] = None  # "buy" or "sell"


class MarketSimulator:
    """Simulates the hidden true value of an asset over time.

    The true value evolves as:
        V(t) = V(t-1) + sigma * N(0,1) + jump_size * Poisson(lambda)

    Parameters
    ----------
    initial_value : float
        Starting true value of the asset.
    sigma : float
        Per-round volatility (standard deviation of Gaussian noise).
    jump_intensity : float
        Poisson intensity parameter (lambda) for jump arrivals.
    jump_size : float
        Mean absolute size of jumps (actual jump is +/- this value).
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        initial_value: float = 100.0,
        sigma: float = 0.5,
        jump_intensity: float = 0.05,
        jump_size: float = 3.0,
        seed: Optional[int] = None,
    ):
        self.initial_value = initial_value
        self.sigma = sigma
        self.jump_intensity = jump_intensity
        self.jump_size = jump_size
        self.rng = np.random.default_rng(seed)

        self._true_values: List[float] = [initial_value]
        self._current_round: int = 0

    @property
    def current_round(self) -> int:
        return self._current_round

    @property
    def true_value(self) -> float:
        """Current true value (hidden from the player during game)."""
        return self._true_values[-1]

    @property
    def true_value_history(self) -> List[float]:
        """Full history of true values."""
        return list(self._true_values)

    def step(self) -> float:
        """Advance the market by one round and return the new true value.

        Returns
        -------
        float
            The new true value after this step.
        """
        self._current_round += 1

        # Brownian motion component
        diffusion = self.sigma * self.rng.standard_normal()

        # Poisson jump component
        n_jumps = self.rng.poisson(self.jump_intensity)
        jump = 0.0
        for _ in range(n_jumps):
            # Jump direction is random (+/-)
            direction = 1 if self.rng.random() > 0.5 else -1
            jump += direction * self.jump_size

        new_value = self._true_values[-1] + diffusion + jump
        self._true_values.append(new_value)

        return new_value

    def generate_path(self, n_rounds: int) -> List[float]:
        """Pre-generate a full path of true values.

        Parameters
        ----------
        n_rounds : int
            Number of rounds to simulate.

        Returns
        -------
        list of float
            True values from round 0 to n_rounds (inclusive).
        """
        for _ in range(n_rounds):
            self.step()
        return list(self._true_values)

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the simulator to initial state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._true_values = [self.initial_value]
        self._current_round = 0

    def get_value_range(self, lookback: int = 10) -> tuple[float, float]:
        """Get the recent range of true values for reference.

        Parameters
        ----------
        lookback : int
            Number of past rounds to consider.

        Returns
        -------
        tuple of (float, float)
            (min_value, max_value) over the lookback window.
        """
        recent = self._true_values[-lookback:]
        return min(recent), max(recent)
