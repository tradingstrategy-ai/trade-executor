"""Price impact estimation."""
import datetime
import enum
from dataclasses import dataclass
from typing import Optional, List

from dataclasses_json import dataclass_json

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.types import Percent, BlockNumber, TokenAmount, USDollarAmount, USDollarPrice


class SizingType(enum.Enum):

    #: Individual trade, buy
    buy = "buy"

    #: Individual trade, sell
    sell = "sell"

    #: Position size
    hold = "hold"


@dataclass_json
@dataclass(slots=True)
class SizeRisk:
    """Result of a price impact estimation during decide_trades().

    - Used by py:class:`~tradeexecutor.strategy.trade_sizer.TradeSizer`
      to return the estimations of the safe trade sizes (not too much price impact, not too much liquidity risk)

    - Capture and save data about the price impact

    - Allows us to use this to cap the max position size
      when we enter to a position.

    .. note ::

      A position size is different from a trade size,
      as we may grow a larger position using a multiple trades to enter

    See also :py:mod:`tradeexecutor.strategy.trade_pricing`.
    """

    #: For which timepoint this price impact estimation was made
    timestamp: datetime.datetime

    pair: TradingPairIdentifier

    #: Buy or sell
    type: SizingType

    #: Path of the trade
    #: One trade can have multiple swaps if there is an intermediary pair.
    path: list[TradingPairIdentifier]

    #: Was this trade hitting the maximum
    capped: bool

    #: Block number if we use raw price data
    block_number: BlockNumber | None = None

    #: Venue mid price per token
    mid_price: USDollarPrice | None = None

    #: Avg price per token when accepted size is filled
    avg_price: USDollarPrice | None = None

    #: How much we want to get
    asked_quantity: TokenAmount | None = None

    #: How much we want to get
    asked_size: USDollarAmount | None = None

    #: What was the capped size by the
    accepted_quantity: TokenAmount | None = None

    #: What was the capped size
    accepted_size: USDollarAmount | None = None

    @property
    def cost(self) -> USDollarAmount:
        """How much this trade costs us.

        :return:
            The cost of price impact in US dollar terms
        """
        return abs(self.accepted_size * self.mid_price - self.accepted_size * self.avg_price)

    @property
    def impact(self) -> Percent:
        """What is the price impact of this trade.

        :return:
            Price impact as % of the trade.

            0% = no impact.

            1% =
        """
        return abs(self.avg_price - self.mid_price) / self.mid_price