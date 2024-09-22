"""Trade size risk estimation.

- See :py:class:`SizeRisk`

"""
import datetime
import enum
from dataclasses import dataclass
from decimal import Decimal
from types import NoneType

from dataclasses_json import dataclass_json

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.types import BlockNumber, TokenAmount, USDollarAmount, USDollarPrice


class SizingType(enum.Enum):
    """What kind of size risk the result is for.

    - See :py:class:`SizeRisk`
    """

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

    - Capture and save data about the price impact, so we can diagnose this later.
      Not just the result, but the variables that lead to the result.

    - Allows us to use this to cap the max position size
      when we enter to a position.

    .. note ::

      A position size is different from a trade size,
      as we may grow a larger position using a multiple trades to enter

    See also :py:mod:`tradeexecutor.strategy.trade_pricing`.
    """

    #: For which timepoint this price impact estimation was made
    #:
    #: Can be set to None if not relevant.
    #:
    timestamp: datetime.datetime | None

    pair: TradingPairIdentifier

    #: Buy or sell
    sizing_type: SizingType

    #: Path of the trade
    #: One trade can have multiple swaps if there is an intermediary pair.
    path: list[TradingPairIdentifier]

    #: Was this trade hitting the maximum cap.
    #:
    #: This means the trade size was reduced due to risk.
    #:
    capped: bool

    #: Block number we used for onchain estimation
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

    #: Store various diagnostics data items ehre
    #:
    #: Each implementation can store e.g. percents
    #:
    diagnostics_data: dict | None = None

    def __post_init__(self):
        assert isinstance(self.timestamp, (datetime.datetime, NoneType)), f"Timestamp was {self.timestamp}"
        if self.asked_quantity is not None:
            assert isinstance(self.asked_quantity, Decimal)
        if self.accepted_quantity is not None:
            assert isinstance(self.accepted_quantity, Decimal)
        if self.diagnostics_data is not None:
            assert isinstance(self.diagnostics_data, dict)
