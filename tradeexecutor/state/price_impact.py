"""Price impact estimation."""
import datetime
import enum
from dataclasses import dataclass

from dataclasses_json import dataclass_json

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.types import Percent, BlockNumber, TokenAmount, USDollarAmount


class PriceImpactSide(enum.Enum):
    buy = "buy"
    sell = "sell"


@dataclass_json
@dataclass(slots=True, frozen=True)
class PriceImpactEstimation:
    """Result of a price impact estimation during decide_trades().

    - Capture and save data about the price impact

    - Allows us to use this to cap the max position size
      when we enter to a position
    """

    #: For which timepoint this price impact estimation was made
    timestamp: datetime.datetime

    #: Trading pair for which the estimation is
    pair: TradingPairIdentifier

    #: Block number if we use raw price data
    block_number: BlockNumber | None

    #: Buy or sell
    side: PriceImpactSide

    #: How much we want to get
    asked_quantity: TokenAmount

    #: How much we want to get
    asked_size: USDollarAmount

    #: What was the capped size
    accepted_quantity: TokenAmount

    #: What was the capped size
    accepted_size: USDollarAmount

    #: How much we estimated price impact as %
    estimated_price_impact: Percent

    #: What was the cap for the price impact for this estimation
    acceptable_price_impact: Percent

    def is_buy(self):
        return self.side is True

    def is_sell(self):
        return not self.is_buy()

    def is_maxed_out(self):
        return self.estimated_price_impact >= self.acceptable_price_impact

    @property
    def cost(self) -> USDollarAmount:
        """How much this trade costs us.

        -
        """