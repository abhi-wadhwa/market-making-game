"""PnL decomposition and analytics engine.

Decomposes market maker PnL into:
1. Spread capture: revenue from the bid-ask spread
2. Adverse selection cost: losses from trading against informed traders
3. Inventory risk: mark-to-market gains/losses from holding inventory

Total PnL = Spread Capture - Adverse Selection + Inventory Mark-to-Market
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from .market_maker import MarketMaker, Execution
from .inventory import InventoryManager, PositionSnapshot


@dataclass
class PnLDecomposition:
    """Decomposition of PnL into its components.

    Attributes
    ----------
    total_pnl : float
        Total PnL including all components.
    spread_capture : float
        Total revenue from half-spreads on all trades.
    adverse_selection : float
        Total adverse selection cost (negative means loss).
    inventory_pnl : float
        Mark-to-market PnL from inventory changes.
    inventory_penalty : float
        Penalty for carrying large inventory.
    n_trades : int
        Number of trades executed.
    n_buys : int
        Number of times MM bought (incoming sell orders).
    n_sells : int
        Number of times MM sold (incoming buy orders).
    """

    total_pnl: float
    spread_capture: float
    adverse_selection: float
    inventory_pnl: float
    inventory_penalty: float
    n_trades: int
    n_buys: int
    n_sells: int


@dataclass
class RoundSummary:
    """Summary statistics for the completed game.

    Attributes
    ----------
    total_rounds : int
        Number of rounds played.
    total_trades : int
        Number of trades executed.
    fill_rate : float
        Fraction of rounds with a trade.
    sharpe_ratio : float
        Risk-adjusted return (annualized approximation).
    max_drawdown : float
        Largest peak-to-trough PnL decline.
    avg_spread : float
        Average bid-ask spread quoted.
    avg_position : float
        Average absolute position held.
    final_pnl : float
        PnL at the end of the game.
    pnl_decomposition : PnLDecomposition
        Full PnL breakdown.
    """

    total_rounds: int
    total_trades: int
    fill_rate: float
    sharpe_ratio: float
    max_drawdown: float
    avg_spread: float
    avg_position: float
    final_pnl: float
    pnl_decomposition: PnLDecomposition


class AnalyticsEngine:
    """Computes analytics and PnL decomposition for the market maker.

    Parameters
    ----------
    market_maker : MarketMaker
        The market maker instance to analyze.
    inventory_manager : InventoryManager
        The inventory manager for position tracking.
    """

    def __init__(
        self,
        market_maker: MarketMaker,
        inventory_manager: InventoryManager,
    ):
        self.mm = market_maker
        self.inv = inventory_manager

    def compute_pnl_decomposition(
        self, current_value: float
    ) -> PnLDecomposition:
        """Compute the full PnL decomposition.

        Parameters
        ----------
        current_value : float
            Current true value for mark-to-market.

        Returns
        -------
        PnLDecomposition
            The decomposed PnL.
        """
        executions = self.mm.execution_history

        spread_capture = sum(e.spread_earned for e in executions)
        adverse_selection = sum(e.adverse_selection for e in executions)

        n_buys = sum(1 for e in executions if e.mm_side == "buy")
        n_sells = sum(1 for e in executions if e.mm_side == "sell")

        # Inventory PnL: mark-to-market of current position
        inventory_pnl = self.mm.position * current_value

        # Total PnL from cash flow perspective
        total_pnl = self.mm.get_total_pnl(current_value)

        # Inventory penalty
        inventory_penalty = self.inv.inventory_penalty(self.mm.position)

        return PnLDecomposition(
            total_pnl=total_pnl,
            spread_capture=spread_capture,
            adverse_selection=adverse_selection,
            inventory_pnl=inventory_pnl,
            inventory_penalty=inventory_penalty,
            n_trades=len(executions),
            n_buys=n_buys,
            n_sells=n_sells,
        )

    def compute_round_pnl(self, round_number: int, current_value: float) -> float:
        """Compute PnL for a specific round.

        Parameters
        ----------
        round_number : int
            The round to compute PnL for.
        current_value : float
            True value at that round.

        Returns
        -------
        float
            PnL contribution from that round's trades.
        """
        round_executions = [
            e for e in self.mm.execution_history
            if e.round_number == round_number
        ]

        pnl = 0.0
        for e in round_executions:
            if e.mm_side == "buy":
                # MM bought at price, gained inventory worth current_value
                pnl += current_value - e.price
            else:
                # MM sold at price, lost inventory worth current_value
                pnl += e.price - current_value

        return pnl

    def compute_per_round_pnl(self, true_values: List[float]) -> List[float]:
        """Compute PnL attribution for each round.

        Parameters
        ----------
        true_values : list of float
            True values for each round (index 0 = round 0).

        Returns
        -------
        list of float
            PnL for each round.
        """
        if not self.inv.snapshots:
            return []

        pnl_per_round = []
        prev_pnl = 0.0

        for snapshot in self.inv.snapshots:
            current_pnl = snapshot.total_pnl
            round_pnl = current_pnl - prev_pnl
            pnl_per_round.append(round_pnl)
            prev_pnl = current_pnl

        return pnl_per_round

    def compute_sharpe_ratio(self, true_values: List[float]) -> float:
        """Compute Sharpe ratio of per-round PnL.

        Parameters
        ----------
        true_values : list of float
            True values for each round.

        Returns
        -------
        float
            Sharpe ratio (mean / std of per-round PnL).
            Returns 0.0 if insufficient data.
        """
        pnl_per_round = self.compute_per_round_pnl(true_values)

        if len(pnl_per_round) < 2:
            return 0.0

        mean_pnl = np.mean(pnl_per_round)
        std_pnl = np.std(pnl_per_round, ddof=1)

        if std_pnl < 1e-10:
            return 0.0 if abs(mean_pnl) < 1e-10 else float("inf") * np.sign(mean_pnl)

        return float(mean_pnl / std_pnl)

    def compute_fill_rate(self, total_rounds: int) -> float:
        """Compute the fill rate (fraction of rounds with a trade).

        Parameters
        ----------
        total_rounds : int
            Total number of rounds played.

        Returns
        -------
        float
            Fill rate in [0, 1].
        """
        if total_rounds == 0:
            return 0.0
        return len(self.mm.execution_history) / total_rounds

    def compute_summary(
        self,
        total_rounds: int,
        current_value: float,
        true_values: List[float],
    ) -> RoundSummary:
        """Compute a full game summary.

        Parameters
        ----------
        total_rounds : int
            Number of rounds played.
        current_value : float
            Final true value.
        true_values : list of float
            Full history of true values.

        Returns
        -------
        RoundSummary
            Complete game analytics.
        """
        pnl_decomp = self.compute_pnl_decomposition(current_value)
        sharpe = self.compute_sharpe_ratio(true_values)
        fill_rate = self.compute_fill_rate(total_rounds)
        max_dd = self.inv.max_position_pnl_drawdown()
        avg_pos = self.inv.average_absolute_position()

        avg_spread = 0.0
        if self.mm.quote_history:
            avg_spread = float(
                np.mean([q.spread for q in self.mm.quote_history])
            )

        return RoundSummary(
            total_rounds=total_rounds,
            total_trades=pnl_decomp.n_trades,
            fill_rate=fill_rate,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            avg_spread=avg_spread,
            avg_position=avg_pos,
            final_pnl=pnl_decomp.total_pnl,
            pnl_decomposition=pnl_decomp,
        )
