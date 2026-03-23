"""Market maker logic for quoting and execution.

The player acts as a market maker, setting bid and ask prices each round.
Incoming orders execute against these quotes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .order_flow import Order, OrderSide


@dataclass
class Quote:
    """A two-sided quote from the market maker.

    Attributes
    ----------
    bid : float
        Price at which the MM is willing to buy.
    ask : float
        Price at which the MM is willing to sell.
    round_number : int
        The round this quote is for.
    """

    bid: float
    ask: float
    round_number: int = 0

    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.ask - self.bid

    @property
    def mid(self) -> float:
        """Mid price."""
        return (self.bid + self.ask) / 2.0

    def is_valid(self) -> bool:
        """Check if the quote is valid (bid < ask)."""
        return self.bid < self.ask


@dataclass
class Execution:
    """Record of a trade execution.

    Attributes
    ----------
    round_number : int
        Round in which the trade occurred.
    price : float
        Execution price.
    side : str
        "buy" or "sell" from the incoming order's perspective.
        If the order buys, the MM sells at the ask.
        If the order sells, the MM buys at the bid.
    mm_side : str
        The market maker's side of the trade ("buy" or "sell").
    true_value : float
        True value at time of execution.
    spread_earned : float
        Half-spread earned by the MM on this trade.
    adverse_selection : float
        Adverse selection cost: difference between true value and execution price.
    """

    round_number: int
    price: float
    side: str
    mm_side: str
    true_value: float
    spread_earned: float
    adverse_selection: float


class MarketMaker:
    """Market maker that processes quotes and executes orders.

    The market maker:
    1. Sets bid and ask prices each round.
    2. Receives incoming orders.
    3. Executes orders that cross the quotes.
    4. Tracks all executions for analytics.

    Parameters
    ----------
    initial_cash : float
        Starting cash balance.
    """

    def __init__(self, initial_cash: float = 10000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position: int = 0  # number of units held (+ long, - short)
        self.quote_history: List[Quote] = []
        self.execution_history: List[Execution] = []
        self._current_quote: Optional[Quote] = None

    @property
    def current_quote(self) -> Optional[Quote]:
        return self._current_quote

    def set_quote(self, bid: float, ask: float, round_number: int = 0) -> Quote:
        """Set a new two-sided quote.

        Parameters
        ----------
        bid : float
            Bid price.
        ask : float
            Ask price.
        round_number : int
            Current round number.

        Returns
        -------
        Quote
            The new quote object.

        Raises
        ------
        ValueError
            If bid >= ask.
        """
        quote = Quote(bid=bid, ask=ask, round_number=round_number)
        if not quote.is_valid():
            raise ValueError(
                f"Invalid quote: bid ({bid:.2f}) must be less than ask ({ask:.2f})"
            )
        self._current_quote = quote
        self.quote_history.append(quote)
        return quote

    def process_order(self, order: Order) -> Optional[Execution]:
        """Process an incoming order against the current quote.

        Parameters
        ----------
        order : Order
            The incoming order.

        Returns
        -------
        Execution or None
            The execution record if the order was filled, None otherwise.
        """
        if self._current_quote is None:
            return None

        quote = self._current_quote

        if order.is_buy:
            # Incoming buy order: executes at the MM's ask price
            # MM sells to the buyer
            execution_price = quote.ask
            mm_side = "sell"
            self.position -= 1
            self.cash += execution_price
        else:
            # Incoming sell order: executes at the MM's bid price
            # MM buys from the seller
            execution_price = quote.bid
            mm_side = "buy"
            self.position += 1
            self.cash -= execution_price

        # Mark order as executed
        order.executed = True
        order.execution_price = execution_price

        # Calculate spread earned: half the spread
        half_spread = quote.spread / 2.0

        # Adverse selection: loss from trading against informed direction
        # If MM buys (order sells), adverse selection = true_value - bid
        #   (if true_value < bid, MM overpaid -> loss)
        # If MM sells (order buys), adverse selection = ask - true_value
        #   (if true_value > ask, MM undersold -> loss)
        if mm_side == "buy":
            adverse_selection = order.true_value - execution_price
        else:
            adverse_selection = execution_price - order.true_value

        execution = Execution(
            round_number=order.round_number,
            price=execution_price,
            side=order.side.value,
            mm_side=mm_side,
            true_value=order.true_value,
            spread_earned=half_spread,
            adverse_selection=adverse_selection,
        )

        self.execution_history.append(execution)
        return execution

    def get_unrealized_pnl(self, current_value: float) -> float:
        """Calculate unrealized PnL from current inventory.

        Parameters
        ----------
        current_value : float
            Current mark-to-market value per unit.

        Returns
        -------
        float
            Unrealized PnL.
        """
        return self.position * current_value

    def get_total_pnl(self, current_value: float) -> float:
        """Calculate total PnL (realized + unrealized).

        Parameters
        ----------
        current_value : float
            Current value for marking inventory.

        Returns
        -------
        float
            Total PnL = (cash - initial_cash) + position * current_value.
        """
        return (self.cash - self.initial_cash) + self.position * current_value

    def reset(self) -> None:
        """Reset the market maker to initial state."""
        self.cash = self.initial_cash
        self.position = 0
        self.quote_history.clear()
        self.execution_history.clear()
        self._current_quote = None
