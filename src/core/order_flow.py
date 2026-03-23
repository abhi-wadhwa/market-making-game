"""Order flow generation with informed and noise traders.

Informed traders know the true value and trade directionally.
Noise traders submit random buy/sell orders.

The probability of an incoming order being from an informed trader
is controlled by the parameter alpha (adverse selection intensity).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TraderType(Enum):
    """Type of trader submitting an order."""

    INFORMED = "informed"
    NOISE = "noise"


class OrderSide(Enum):
    """Side of the order."""

    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """An incoming order from a trader.

    Attributes
    ----------
    round_number : int
        The round in which this order was generated.
    trader_type : TraderType
        Whether the trader is informed or noise.
    side : OrderSide
        Buy or sell.
    true_value : float
        The true value at the time of the order (for analytics).
    executed : bool
        Whether the order was filled against the market maker's quote.
    execution_price : float or None
        Price at which the order was filled (None if not executed).
    """

    round_number: int
    trader_type: TraderType
    side: OrderSide
    true_value: float
    executed: bool = False
    execution_price: Optional[float] = None

    @property
    def is_buy(self) -> bool:
        return self.side == OrderSide.BUY

    @property
    def is_informed(self) -> bool:
        return self.trader_type == TraderType.INFORMED


class OrderFlowGenerator:
    """Generates stochastic order flow with informed and noise traders.

    Parameters
    ----------
    alpha : float
        Probability that an incoming order is from an informed trader.
        Higher alpha means more adverse selection.
    arrival_rate : float
        Average number of orders per round (Poisson parameter).
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        arrival_rate: float = 1.0,
        seed: Optional[int] = None,
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if arrival_rate <= 0:
            raise ValueError(f"arrival_rate must be positive, got {arrival_rate}")

        self.alpha = alpha
        self.arrival_rate = arrival_rate
        self.rng = np.random.default_rng(seed)

    def generate_order(
        self,
        round_number: int,
        true_value: float,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
    ) -> Optional[Order]:
        """Generate an order for a given round.

        For simplicity, we generate exactly one order per round in the
        game loop. The arrival_rate parameter can be used to sometimes
        generate no order (if < 1) or multiple orders (if > 1), but
        the primary game loop uses single orders.

        Parameters
        ----------
        round_number : int
            Current round number.
        true_value : float
            Current true value of the asset.
        bid : float or None
            Market maker's current bid price.
        ask : float or None
            Market maker's current ask price.

        Returns
        -------
        Order or None
            The generated order, or None if no order arrives.
        """
        # Check if an order arrives this round
        if self.arrival_rate < 1.0:
            if self.rng.random() > self.arrival_rate:
                return None

        # Determine trader type
        is_informed = self.rng.random() < self.alpha
        trader_type = TraderType.INFORMED if is_informed else TraderType.NOISE

        # Determine order side
        if is_informed:
            # Informed trader knows the true value
            if bid is not None and ask is not None:
                # Buy if true value > ask (asset is underpriced at ask)
                # Sell if true value < bid (asset is overpriced at bid)
                if true_value > ask:
                    side = OrderSide.BUY
                elif true_value < bid:
                    side = OrderSide.SELL
                else:
                    # True value is within the spread; informed trader
                    # still trades in the direction of value vs mid
                    mid = (bid + ask) / 2.0
                    side = OrderSide.BUY if true_value >= mid else OrderSide.SELL
            else:
                # No quotes available yet; informed trader trades toward value
                side = OrderSide.BUY if self.rng.random() > 0.5 else OrderSide.SELL
        else:
            # Noise trader: random buy/sell
            side = OrderSide.BUY if self.rng.random() > 0.5 else OrderSide.SELL

        return Order(
            round_number=round_number,
            trader_type=trader_type,
            side=side,
            true_value=true_value,
        )

    def generate_orders_batch(
        self,
        round_number: int,
        true_value: float,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
    ) -> list[Order]:
        """Generate a batch of orders for a round based on arrival_rate.

        Parameters
        ----------
        round_number : int
            Current round number.
        true_value : float
            Current true value.
        bid, ask : float or None
            Market maker's current quotes.

        Returns
        -------
        list of Order
            Orders that arrived this round (could be empty).
        """
        n_orders = self.rng.poisson(self.arrival_rate)
        orders = []
        for _ in range(n_orders):
            order = self.generate_order(round_number, true_value, bid, ask)
            if order is not None:
                orders.append(order)
        return orders

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the generator."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
