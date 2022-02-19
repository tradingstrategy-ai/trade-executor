import datetime
from decimal import Decimal
from typing import Callable

from tradeexecutor.state.state import TradingPosition


class RevaluationFailed(Exception):
    """Should not happen.

    Something failed within the revaluation - like trading pair disappearing.
    """


#: Callable for revaluating existing trading positions
RevaluationMethod = Callable[[datetime.datetime, TradingPosition], Decimal]

