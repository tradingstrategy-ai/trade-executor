"""Alpha model and portfolio construction model related logic."""
from dataclasses import dataclass
from typing import Optional, TypeAlias, Dict

from dataclasses_json import dataclass_json

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.types import PairInternalId, USDollarAmount


@dataclass_json
@dataclass(slots=True)
class TradingPairWeight:
    """Present one asset in alpha model weighting.

    - Required variables are needed as an input from `decide_trades()` function in a strategy

    - Optional variables are calculated in the various phases of alpha model processing
    """

    #: For which pair is this alpha weight
    #:
    #:
    pair: TradingPairIdentifier

    #: Raw weight
    weight: float

    #: Stop loss for this position
    #:
    #: Used for the risk management.
    #:
    #: 0.98 means 2% stop loss.
    #:
    #: Set to 0 to disable stop loss.
    stop_loss: float

    #: Weight 0...1 so that all portfolio weights sum to 1
    normalised_weight: Optional[float] = None

    #: Old weight of this value from the previous cycle.
    #:
    #: If this asset was part of the portfolio at :term:`Strategy cycle`
    #: When
    old_weight: Optional[float] = None

    #: How many dolars we plan to invest on this one.
    #:
    #: Calculated by portfolio total investment equity * normalised weight * price.
    investment_amount: Optional[USDollarAmount] = None



#: Map of different weights of trading pairs for alpha model.
#:
#: If there is no entry present assume its weight is zero.
#:
AlphaWeights: TypeAlias = Dict[PairInternalId, TradingPairWeight]