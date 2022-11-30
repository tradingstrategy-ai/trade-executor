"""Strategy status summary."""
import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from dataclasses_json import dataclass_json

from tradeexecutor.state.types import USDollarAmount


@dataclass_json
@dataclass(frozen=True)
class StrategySummaryStatistics:
    """Performance statistics displayed on the tile cards."""

    #: When these stats where calculated
    #:
    calculated_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)

    #: When this strategy truly started.
    #:
    #: We mark the time of the first trade when the strategy
    #: started to perform.
    first_trade_at: Optional[datetime.datetime] = None

    #: When was the last time this strategy made a trade
    #:
    last_trade_at: Optional[datetime.datetime] = None

    #: Has the strategy been running 90 days so that the annualised profitability
    #: can be correctly calcualted.
    #:
    enough_data: Optional[bool] = None

    #: Total equity of this strategy.
    #:
    #: Also known as Total Value locked (TVL) in DeFi.
    #: It's cash + open hold positions
    current_value: Optional[USDollarAmount] = None

    #: Profitability of last 90 days
    #:
    #:
    #: If :py:attr:`enough_data` is set we can display this annualised,
    #: otherwise we can say so sar.
    profitability_90_days: Optional[float] = None

    #: Data for the performance chart used in the summary card.
    #:
    #: Contains (UNIX time, performance %) tuples.
    #:
    #: Relative performance -1 ... 1 (100%) up and
    #: 0 is no gains/no losses.
    #:
    #: One point per day.
    #: Note that we might have 90 or 91 points because date ranges
    #: are inclusive.
    performance_chart_90_days: Optional[List[Tuple[datetime.datetime, float]]] = None


@dataclass_json
@dataclass(frozen=True)
class StrategySummary:
    """Strategy summary.

    - Helper class to render strategy tiles data

    - Contains mixture of static metadata, trade executor crash status,
      latest strategy performance stats and visualisation

    - Is not stored as the part of the strategy state.
      In the case of a restart, summary statistics are calculated again.

    - See /summary API endpoint where it is constructed before returning to the client
    """

    #: Strategy name
    name: str

    #: 1 sentence
    short_description: Optional[str]

    #: Multiple paragraphs.
    long_description: Optional[str]

    #: For <img src>
    icon_url: Optional[str]

    #: When the instance was started last time
    #:
    #: Unix timestamp, as UTC
    started_at: float

    #: Is the executor main loop running or crashed.
    #:
    #: Use /status endpoint to get the full exception info.
    #:
    #: Not really a part of metadata, but added here to make frontend
    #: queries faster. See also :py:class:`tradeexecutor.state.executor_state.ExecutorState`.
    executor_running: bool

    #: Strategy statistics for summary tiles
    #:
    #: Helps rendering the web tiles.
    summary_statistics: StrategySummaryStatistics = field(default_factory=StrategySummaryStatistics)


