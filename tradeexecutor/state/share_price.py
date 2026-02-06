"""Running state for internal share price tracking.

This module provides data structures for tracking share price state
incrementally on each trade, rather than recalculating from scratch.
"""
import datetime
from dataclasses import dataclass

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass(slots=True)
class SharePriceState:
    """Running state for internal share price tracking.

    Updated incrementally on each trade execution rather than
    recalculated from scratch. Follows the pattern of TradingPosition.loan.

    See :py:func:`tradeexecutor.strategy.share_price.create_share_price_state`
    and :py:func:`tradeexecutor.strategy.share_price.update_share_price_state`.
    """

    #: Current share price (total_assets / total_supply)
    current_share_price: float

    #: Total internal shares outstanding
    total_supply: float

    #: Cumulative quantity held (for proportion calculations on sells)
    cumulative_quantity: float

    #: Total amount invested (for profit calculations)
    total_invested: float

    #: Peak total supply seen (for closed position profit calc)
    peak_total_supply: float

    #: Initial share price (typically 1.0)
    initial_share_price: float = 1.0

    #: When this state was last updated
    last_updated_at: datetime.datetime | None = None
