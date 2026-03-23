"""Glosten-Milgrom (1985) model for theoretical spread computation.

The Glosten-Milgrom model derives the equilibrium bid-ask spread
that a competitive market maker should set, given the probability
of informed trading and the value uncertainty.

Key insight: the spread compensates the market maker for losses
to informed traders (adverse selection).

Model Setup:
- Asset has true value V, which can be V_H (high) or V_L (low)
- Prior probability of V = V_H is mu
- Fraction alpha of traders are informed (know V)
- Fraction (1 - alpha) are noise traders (random buy/sell)

Equilibrium:
- Ask = E[V | buy order]  = V_H * P(V_H | buy) + V_L * P(V_L | buy)
- Bid = E[V | sell order] = V_H * P(V_H | sell) + V_L * P(V_L | sell)

Where Bayesian updating gives:
- P(V_H | buy)  = (alpha + (1-alpha)/2) * mu / ((alpha + (1-alpha)/2)*mu + (1-alpha)/2*(1-mu))
- P(V_H | sell) = ((1-alpha)/2 * mu) / ((1-alpha)/2*mu + (alpha + (1-alpha)/2)*(1-mu))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class GMEquilibrium:
    """Glosten-Milgrom equilibrium values.

    Attributes
    ----------
    bid : float
        Equilibrium bid price.
    ask : float
        Equilibrium ask price.
    spread : float
        Equilibrium spread (ask - bid).
    relative_spread : float
        Spread as a fraction of mid price.
    mid : float
        Mid price = (ask + bid) / 2.
    prob_high_given_buy : float
        Posterior P(V_H | buy order).
    prob_high_given_sell : float
        Posterior P(V_H | sell order).
    """

    bid: float
    ask: float
    spread: float
    relative_spread: float
    mid: float
    prob_high_given_buy: float
    prob_high_given_sell: float


class GlostenMilgromModel:
    """Implementation of the Glosten-Milgrom adverse selection model.

    Parameters
    ----------
    v_high : float
        High value of the asset.
    v_low : float
        Low value of the asset.
    alpha : float
        Probability of informed trader (adverse selection intensity).
    mu : float
        Prior probability that V = V_H.
    """

    def __init__(
        self,
        v_high: float = 105.0,
        v_low: float = 95.0,
        alpha: float = 0.3,
        mu: float = 0.5,
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if not 0.0 <= mu <= 1.0:
            raise ValueError(f"mu must be in [0, 1], got {mu}")
        if v_high <= v_low:
            raise ValueError(f"v_high ({v_high}) must be > v_low ({v_low})")

        self.v_high = v_high
        self.v_low = v_low
        self.alpha = alpha
        self.mu = mu

    def compute_equilibrium(self, mu: Optional[float] = None) -> GMEquilibrium:
        """Compute the Glosten-Milgrom equilibrium bid and ask.

        Parameters
        ----------
        mu : float or None
            Override prior probability. If None, use self.mu.

        Returns
        -------
        GMEquilibrium
            The equilibrium values.
        """
        mu = mu if mu is not None else self.mu
        alpha = self.alpha

        # Probability of a buy order given V = V_H:
        # Informed buys (prob alpha), noise buys with prob 0.5 (prob (1-alpha)*0.5)
        prob_buy_given_high = alpha + (1 - alpha) * 0.5

        # Probability of a buy order given V = V_L:
        # Informed does NOT buy (sells), noise buys with prob 0.5
        prob_buy_given_low = (1 - alpha) * 0.5

        # Probability of a sell order given V = V_H:
        # Informed does NOT sell (buys), noise sells with prob 0.5
        prob_sell_given_high = (1 - alpha) * 0.5

        # Probability of a sell order given V = V_L:
        # Informed sells (prob alpha), noise sells with prob 0.5
        prob_sell_given_low = alpha + (1 - alpha) * 0.5

        # Bayesian updating: P(V_H | buy)
        numerator_buy = prob_buy_given_high * mu
        denominator_buy = prob_buy_given_high * mu + prob_buy_given_low * (1 - mu)
        prob_high_given_buy = numerator_buy / denominator_buy if denominator_buy > 0 else mu

        # Bayesian updating: P(V_H | sell)
        numerator_sell = prob_sell_given_high * mu
        denominator_sell = prob_sell_given_high * mu + prob_sell_given_low * (1 - mu)
        prob_high_given_sell = numerator_sell / denominator_sell if denominator_sell > 0 else mu

        # Equilibrium prices
        ask = prob_high_given_buy * self.v_high + (1 - prob_high_given_buy) * self.v_low
        bid = prob_high_given_sell * self.v_high + (1 - prob_high_given_sell) * self.v_low

        spread = ask - bid
        mid = (ask + bid) / 2.0
        relative_spread = spread / mid if mid > 0 else 0.0

        return GMEquilibrium(
            bid=bid,
            ask=ask,
            spread=spread,
            relative_spread=relative_spread,
            mid=mid,
            prob_high_given_buy=prob_high_given_buy,
            prob_high_given_sell=prob_high_given_sell,
        )

    def compute_spread_simple(self) -> float:
        """Compute the simplified Glosten-Milgrom spread.

        The simplified formula (with mu = 0.5) gives:
            spread = 2 * alpha * (V_H - V_L) / (1 + alpha)

        This is derived from the full Bayesian formula and represents
        the spread needed to break even against informed traders.

        Returns
        -------
        float
            The theoretical equilibrium spread.
        """
        value_range = self.v_high - self.v_low
        # With mu = 0.5:
        # ask = V_L + (V_H - V_L) * (alpha + (1-alpha)/2) / 1
        #     = V_L + (V_H - V_L) * (1 + alpha) / 2
        # Simplified: spread = 2 * alpha * (V_H - V_L) / (1 + alpha)
        # But this is an approximation. Use the exact formula:
        eq = self.compute_equilibrium(mu=0.5)
        return eq.spread

    def update_beliefs(self, order_is_buy: bool, mu: Optional[float] = None) -> float:
        """Update beliefs about the true value after observing an order.

        Parameters
        ----------
        order_is_buy : bool
            True if the observed order is a buy.
        mu : float or None
            Current prior. If None, use self.mu.

        Returns
        -------
        float
            Updated posterior probability that V = V_H.
        """
        mu = mu if mu is not None else self.mu
        eq = self.compute_equilibrium(mu)

        if order_is_buy:
            return eq.prob_high_given_buy
        else:
            return eq.prob_high_given_sell

    def expected_profit_per_trade(self) -> float:
        """Calculate the expected profit per trade for the market maker.

        The MM earns the half-spread on noise trades but loses to
        informed traders. With correct pricing, expected profit = 0
        (zero-profit condition).

        Returns
        -------
        float
            Expected profit per trade (should be ~0 at equilibrium).
        """
        eq = self.compute_equilibrium()

        # Expected profit from a buy order arriving:
        # MM sells at ask. E[profit | buy] = ask - E[V | buy] = ask - ask = 0
        profit_from_buy = eq.ask - (eq.prob_high_given_buy * self.v_high +
                                     (1 - eq.prob_high_given_buy) * self.v_low)

        # Expected profit from a sell order arriving:
        # MM buys at bid. E[profit | sell] = E[V | sell] - bid = bid - bid = 0
        profit_from_sell = (eq.prob_high_given_sell * self.v_high +
                            (1 - eq.prob_high_given_sell) * self.v_low) - eq.bid

        # Average (equal probability of buy/sell)
        return (profit_from_buy + profit_from_sell) / 2.0

    def theoretical_spread_for_alpha(self, alpha: float) -> float:
        """Compute spread for a different alpha, keeping other params fixed.

        Parameters
        ----------
        alpha : float
            Informed trading probability.

        Returns
        -------
        float
            Equilibrium spread.
        """
        model = GlostenMilgromModel(
            v_high=self.v_high,
            v_low=self.v_low,
            alpha=alpha,
            mu=self.mu,
        )
        return model.compute_equilibrium().spread
