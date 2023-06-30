"""Strategy status summary."""
import datetime
import enum
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any

from dataclasses_json import dataclass_json

from tradeexecutor.state.metadata import OnChainData
from tradeexecutor.state.types import USDollarAmount



class KeyMetricKind(enum.Enum):
    """What key metrics we have available on a strategy summary card.

    All othe metrics will be available as well, but we do not
    cache them for the quick frontend rendering.
    """

    #: Sharpe ratio for the execution
    sharpe = "sharpe"

    #: Sortino ratio for the execution
    sortino = "sortino"

    #: Negative value 0...-1
    max_drawdown = "max_drawdown"

    #: UNIX timestamp when the first trade was executd
    started_at = "started_at"

    #: Annualised profitability
    profitability = "profitability"

    #: Total equity
    total_equity = "total_equity"


class KeyMetricSource(enum.Enum):
    """Did we calcualte a key metric based on backtesting data or live trading data."""
    backtesting = "backtesting"
    live_trading = "live_trading"
    missing = "missing"


@dataclass_json
@dataclass(slots=True, frozen=True)
class KeyMetric:
    """One of available key metrics on the summary card."""

    #: What is this metric
    kind: KeyMetricKind

    #: Did we calculate this metric based on live trading or backtesting data
    source: KeyMetricSource

    #: What's the time period for which this metric was calculated.
    #:
    #: Different Python value types supported,
    #: but everything is serialised to JavaScript Number type
    #: in for JSON.
    #:
    #: Set to `None` when unavailable. In this case this should be
    #: presented as "N/A" in the frontend.
    #:
    value: float | datetime.datetime | datetime.timedelta | None

    #: What's the time period for which this metric was calculated
    calculation_window_start_at: datetime.datetime | None = None

    #: What's the time period for which this metric was calculated
    calculation_window_end_at: datetime.timedelta | None = None

    #: Unavaiability reason.
    #:
    #: Human readable error message why this metric is not available.
    #: Useful for tooltips.
    #:
    unavailability_reason: str | None = None

    #: Help link
    #:
    #: Read more link.
    #:
    #: Does not need to be part of the state,
    #: but we make the frontend dev life easy.
    #:
    help_link: str | None = None

    @staticmethod
    def create_na(kind: KeyMetricKind, reason: str) -> "KeyMetric":
        """Create missing value placeholder."""
        return KeyMetric(
            kind,
            KeyMetricSource.missing,
            None,
            unavailability_reason=reason,
            help_link=_KEY_METRIC_HELP.get(kind),
        )

    @staticmethod
    def create_metric(
            kind: KeyMetricKind,
            source: KeyMetricSource,
            value: Any,
            calculation_window_start_at: datetime.datetime,
            calculation_window_end_at: datetime.datetime) -> "KeyMetric":
        """Create a metric value.

        Automatically fill in the help text link from our hardcoded mapping.
        """
        return KeyMetric(
            kind,
            source,
            value,
            calculation_window_start_at=calculation_window_start_at,
            calculation_window_end_at=calculation_window_end_at,
            help_link=_KEY_METRIC_HELP.get(kind),
        )

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

    #: Strategy performance metrics to be displayed on the summary card
    #:
    #: We use :py:class:`KeyMetricKind` value as the key.
    #:
    key_metrics: Dict[str, KeyMetric] = field(default_factory=dict)


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

    #: List of smart contracts and related web3 interaction information for this strategy.
    #:
    on_chain_data: OnChainData 

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

    #: Number of frozen positions this strategy has and need to manual intervention
    frozen_positions: int

    #: Strategy statistics for summary tiles
    #:
    #: Helps rendering the web tiles.
    summary_statistics: StrategySummaryStatistics = field(default_factory=StrategySummaryStatistics)

    #: Exception message from the run-time loop
    #:
    error_message: str | None = None


#: Help links for different metrics
_KEY_METRIC_HELP = {
   KeyMetricKind.sharpe: "https://tradingstrategy.ai/glossary/sharpe",
   KeyMetricKind.sortino: "https://tradingstrategy.ai/glossary/sortino",
   KeyMetricKind.max_drawdown: "https://tradingstrategy.ai/glossary/maximum-drawdown",
   KeyMetricKind.profitability: "https://tradingstrategy.ai/glossary/profitability",
   KeyMetricKind.total_equity: "https://tradingstrategy.ai/glossary/total-equity",
}
