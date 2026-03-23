"""Inventory and position tracking for the market maker.

Tracks position over time, calculates inventory risk metrics,
and provides position snapshots for analytics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class PositionSnapshot:
    """Snapshot of inventory state at a given round.

    Attributes
    ----------
    round_number : int
        The round number for this snapshot.
    position : int
        Net position (positive = long, negative = short).
    cash : float
        Current cash balance.
    true_value : float
        True asset value at this point.
    mark_to_market : float
        Position * true_value.
    total_pnl : float
        (cash - initial_cash) + mark_to_market.
    """

    round_number: int
    position: int
    cash: float
    true_value: float
    mark_to_market: float
    total_pnl: float


class InventoryManager:
    """Tracks inventory, position limits, and risk metrics.

    Parameters
    ----------
    max_position : int
        Maximum absolute position allowed. If exceeded, quotes are
        adjusted or the game may penalize the player.
    initial_cash : float
        Starting cash balance for PnL reference.
    inventory_penalty_coeff : float
        Coefficient for quadratic inventory penalty in PnL calculation.
        PnL_adjusted = PnL - penalty_coeff * position^2.
    """

    def __init__(
        self,
        max_position: int = 20,
        initial_cash: float = 10000.0,
        inventory_penalty_coeff: float = 0.01,
    ):
        self.max_position = max_position
        self.initial_cash = initial_cash
        self.inventory_penalty_coeff = inventory_penalty_coeff

        self._snapshots: List[PositionSnapshot] = []

    @property
    def snapshots(self) -> List[PositionSnapshot]:
        return list(self._snapshots)

    @property
    def position_history(self) -> List[int]:
        return [s.position for s in self._snapshots]

    @property
    def pnl_history(self) -> List[float]:
        return [s.total_pnl for s in self._snapshots]

    def record_snapshot(
        self,
        round_number: int,
        position: int,
        cash: float,
        true_value: float,
    ) -> PositionSnapshot:
        """Record a position snapshot for the current round.

        Parameters
        ----------
        round_number : int
            Current round.
        position : int
            Current net position.
        cash : float
            Current cash balance.
        true_value : float
            Current true asset value.

        Returns
        -------
        PositionSnapshot
            The recorded snapshot.
        """
        mark_to_market = position * true_value
        total_pnl = (cash - self.initial_cash) + mark_to_market

        snapshot = PositionSnapshot(
            round_number=round_number,
            position=position,
            cash=cash,
            true_value=true_value,
            mark_to_market=mark_to_market,
            total_pnl=total_pnl,
        )
        self._snapshots.append(snapshot)
        return snapshot

    def is_position_breached(self, position: int) -> bool:
        """Check if position exceeds the maximum allowed.

        Parameters
        ----------
        position : int
            Current position.

        Returns
        -------
        bool
            True if |position| > max_position.
        """
        return abs(position) > self.max_position

    def inventory_penalty(self, position: int) -> float:
        """Calculate quadratic inventory penalty.

        Parameters
        ----------
        position : int
            Current position.

        Returns
        -------
        float
            Penalty amount (always non-negative).
        """
        return self.inventory_penalty_coeff * position * position

    def suggested_skew(self, position: int, base_spread: float) -> float:
        """Suggest a price skew to manage inventory.

        When the MM is long, skew quotes down to attract sells.
        When short, skew up to attract buys.

        Parameters
        ----------
        position : int
            Current position.
        base_spread : float
            The base bid-ask spread.

        Returns
        -------
        float
            Suggested skew to apply to mid price (negative = skew down).
        """
        # Skew proportional to position, scaled by spread
        skew_per_unit = base_spread * 0.1  # 10% of spread per unit
        return -position * skew_per_unit

    def max_position_pnl_drawdown(self) -> float:
        """Calculate the maximum PnL drawdown from peak.

        Returns
        -------
        float
            Maximum drawdown (non-negative value representing the loss).
        """
        if not self._snapshots:
            return 0.0

        pnl_series = [s.total_pnl for s in self._snapshots]
        peak = pnl_series[0]
        max_dd = 0.0

        for pnl in pnl_series:
            if pnl > peak:
                peak = pnl
            drawdown = peak - pnl
            if drawdown > max_dd:
                max_dd = drawdown

        return max_dd

    def average_absolute_position(self) -> float:
        """Calculate average absolute position over the game.

        Returns
        -------
        float
            Mean of |position| across all snapshots.
        """
        if not self._snapshots:
            return 0.0
        return float(np.mean([abs(s.position) for s in self._snapshots]))

    def reset(self) -> None:
        """Clear all recorded snapshots."""
        self._snapshots.clear()
