"""Position valuation state management."""

import datetime
from decimal import Decimal

from tradeexecutor.state.types import USDollarAmount


class RevaluationFailed(Exception):
    """Should not happen.

    Something failed within the revaluation - like trading pair disappearing.
    """


class RevalueEvent:
    """Describe how asset was revalued"""
    position_id: str
    revalued_at: datetime.datetime
    quantity: Decimal
    old_price: USDollarAmount
    new_price: USDollarAmount