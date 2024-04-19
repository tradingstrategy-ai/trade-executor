"""Strategy status summary."""
import datetime
import enum
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Set

from dataclasses_json import dataclass_json

from tradeexecutor.state.metadata import OnChainData
from tradeexecutor.state.types import USDollarAmount, UnixTimestamp, Percent
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.tag import StrategyTag


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

    #: CAGR
    #:
    #: See :term:`CAGR`
    cagr = "cagr"

    #: All-time profitability
    profitability = "profitability"

    #: Total equity
    total_equity = "total_equity"

    #: Last trade
    last_trade = "last_trade"

    #: Trades last week
    trades_last_week = "trades_last_week"

    #: Trades per month estimate
    trades_per_month = "trades_per_month"

    #: Duration of the trading period
    trading_period_length = "trading_period_length"

    #: Percentage return over the trading period
    return_percent = "return_percent"

    #: Annualized percentage return, adjusted for the length of the trading period
    annualised_return_percent = "annualised_return_percent"

    #: Initial cash amount at the start of the trading period
    cash_at_start = "cash_at_start"

    #: Portfolio value at the end of the trading period
    value_at_end = "value_at_end"

    #: Total value of all trades conducted
    trade_volume = "trade_volume"

    #: Percentage of positions that were profitable
    position_win_percent = "position_win_percent"

    #: Total number of trading positions taken
    total_positions = "total_positions"

    #: Number of positions that resulted in a profit
    won_positions = "won_positions"

    #: Number of positions that resulted in a loss
    lost_positions = "lost_positions"

    #: Number of times stop losses were triggered
    stop_losses_triggered = "stop_losses_triggered"

    #: Percentage of all positions where stop losses were triggered
    stop_loss_percent_of_all = "stop_loss_percent_of_all"

    #: Percentage of losing positions where stop losses were triggered
    stop_loss_percent_of_lost = "stop_loss_percent_of_lost"

    #: Number of winning positions where stop losses were set
    winning_stop_losses = "winning_stop_losses"

    #: Percentage of winning positions where stop losses were set
    winning_stop_losses_percent = "winning_stop_losses_percent"

    #: Number of losing positions where stop losses were triggered
    losing_stop_losses = "losing_stop_losses"

    #: Percentage of losing positions where stop losses were triggered
    losing_stop_losses_percent = "losing_stop_losses_percent"

    #: Number of times take profits were triggered
    take_profits_triggered = "take_profits_triggered"

    #: Percentage of all positions where take profits were triggered
    take_profit_percent_of_all = "take_profit_percent_of_all"

    #: Percentage of winning positions where take profits were triggered
    take_profit_percent_of_won = "take_profit_percent_of_won"

    #: Number of positions closed with zero profit or loss
    zero_profit_positions = "zero_profit_positions"

    #: Number of positions still open at the end of the trading period
    positions_open_at_the_end = "positions_open_at_the_end"

    #: Total realized profit and loss from all closed positions
    realised_profit_and_loss = "realised_profit_and_loss"

    #: Unrealized profit and loss from positions still open
    unrealised_profit_and_loss = "unrealised_profit_and_loss"

    #: Unrealized value of the portfolio
    portfolio_unrealised_value = "portfolio_unrealised_value"

    #: Extra returns earned from lending pool interest
    extra_returns_on_lending_pool_interest = "extra_returns_on_lending_pool_interest"

    #: Cash amount remaining at the end of the trading period
    cash_left_at_the_end = "cash_left_at_the_end"

    #: Average profit percentage for winning positions
    average_winning_position_profit_percent = "average_winning_position_profit_percent"

    #: Average loss percentage for losing positions
    average_losing_position_loss_percent = "average_losing_position_loss_percent"

    #: Largest percentage profit for a single position
    biggest_winning_position_percent = "biggest_winning_position_percent"

    #: Largest percentage loss for a single position
    biggest_losing_position_percent = "biggest_losing_position_percent"

    #: Average duration for positions that ended in profit
    average_duration_of_winning_positions = "average_duration_of_winning_positions"

    #: Average duration for positions that ended in loss
    average_duration_of_losing_positions = "average_duration_of_losing_positions"

    #: Average number of price bars for winning positions
    average_bars_of_winning_positions = "average_bars_of_winning_positions"

    #: Average number of price bars for losing positions
    average_bars_of_losing_positions = "average_bars_of_losing_positions"

    #: Total liquidity provider fees paid
    lp_fees_paid = "lp_fees_paid"

    #: Percentage of trade volume spent on liquidity provider fees
    lp_fees_paid_percent_of_volume = "lp_fees_paid_percent_of_volume"

    #: Average profit or loss for all positions
    average_position = "average_position"

    #: Median profit or loss for all positions
    median_position = "median_position"

    #: Highest number of consecutive winning trades
    most_consecutive_wins = "most_consecutive_wins"

    #: Highest number of consecutive losing trades
    most_consecutive_losses = "most_consecutive_losses"

    #: Largest risk realized in a single trade
    biggest_realised_risk = "biggest_realised_risk"

    #: Average risk realized across all trades
    avg_realised_risk = "avg_realised_risk"

    #: Maximum percentage pullback from peak capital
    max_pullback_of_total_capital = "max_pullback_of_total_capital"

    #: Maximum loss risked at the opening of a position
    max_loss_risk_at_opening_of_position = "max_loss_risk_at_opening_of_position"

    average_interest_paid_usd = "average_interest_paid_usd"

    median_interest_paid_usd = "median_interest_paid_usd"

    max_interest_paid_usd = "max_interest_paid_usd"

    min_interest_paid_usd = "min_interest_paid_usd"
    
    total_interest_paid_usd = "total_interest_paid_usd"

    average_duration_between_position_openings = "average_duration_between_position_openings"

    average_position_frequency = "average_position_frequency"

    decision_cycle_duration = "decision_cycle_duration"

    def get_help_link(self) -> Optional[str]:
        return _KEY_METRIC_HELP[self]


class KeyMetricSource(enum.Enum):
    """Did we calcualte a key metric based on backtesting data or live trading data."""
    backtesting = "backtesting"
    live_trading = "live_trading"
    missing = "missing"


class KeyMetricCalculationMethod(enum.Enum):
    """How this key metric is calculated.

    Will have effect on the frontend displaying of the value.
    """

    #: We just take the latest value e.g. for total assets
    latest_value = "latest_value"

    #: We calculae over the period of time
    historical_data = "historical_data"



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

    #: How this key metric is calculated
    #:
    #: Hint for the frontend
    calculation_method: KeyMetricCalculationMethod | None = None

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

    #: Name of the metric
    #: 
    #: Should be in human readable format
    name: str | None = None

    def __post_init__(self):
        assert isinstance(self.source, KeyMetricSource)
        assert isinstance(self.kind, KeyMetricKind)

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
            calculation_window_end_at: datetime.datetime,
            method: KeyMetricCalculationMethod,
    ) -> "KeyMetric":
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
            calculation_method=method,
        )

@dataclass_json
@dataclass(frozen=True)
class StrategySummaryStatistics:
    """Performance statistics displayed on the tile cards."""

    #: When these stats where calculated
    #:
    calculated_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)

    #: When this trade executor was launched first time.
    #:
    #: If the trade-executor needs reset, this value is reset as well.
    launched_at: Optional[datetime.datetime] = None

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
    #:
    #: Based on :ref:`compounding realised positions profit`.
    profitability_90_days: Optional[Percent] = None

    #: All time returns, %
    #:
    #: Based on :ref:`compounding realised positions profit`.
    return_all_time: Optional[Percent] = None

    #: Annualised returns, %
    #:
    #: Based on :ref:`compounding realised positions profit`.
    return_annualised: Optional[Percent] = None

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
    #:
    #: Based on :ref:`compounding realised positions profit`.
    performance_chart_90_days: Optional[List[Tuple[UnixTimestamp, USDollarAmount]]] = None

    #: Strategy performance metrics to be displayed on the summary card
    #:
    #: We use :py:class:`KeyMetricKind` value as the key.
    #:
    key_metrics: Dict[str, KeyMetric] = field(default_factory=dict)

    #: After which period the default metrics will switch from backtested data to live data.
    #:
    #: This mostly affects strategy summary tiles.
    #:
    backtest_metrics_cut_off_period: Optional[datetime.timedelta] = None

    #: Duration of each trade cycle
    cycle_duration: Optional[CycleDuration] = None


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

    #: Can the server server backtest files
    #:
    #:
    backtest_available: bool = False

    #: When the executor bailed out with an exception
    #:
    crashed_at: datetime.datetime | None = None

    #: List of strategy tile badges
    #:
    #: See `Metadata.badges` for description.
    #:
    badges: List[str] = field(default_factory=list)

    #: List of strategy tile badges
    #:
    #: See `Metadata.tags` for description.
    #:
    tags: Set[StrategyTag] = field(default_factory=set)


#: Help links for different metrics
_KEY_METRIC_HELP = {
   KeyMetricKind.cagr: "https://tradingstrategy.ai/glossary/compound-annual-growth-rate-cagr",
   KeyMetricKind.sharpe: "https://tradingstrategy.ai/glossary/sharpe",
   KeyMetricKind.sortino: "https://tradingstrategy.ai/glossary/sortino",
   KeyMetricKind.max_drawdown: "https://tradingstrategy.ai/glossary/maximum-drawdown",
   KeyMetricKind.profitability: "https://tradingstrategy.ai/glossary/profitability",
   KeyMetricKind.total_equity: "https://tradingstrategy.ai/glossary/total-equity",
   KeyMetricKind.started_at: "https://tradingstrategy.ai/glossary/strategy-age",
   KeyMetricKind.last_trade: "https://tradingstrategy.ai/glossary/last-trade",
   KeyMetricKind.trades_last_week: "https://tradingstrategy.ai/glossary/trades-last-week",
   KeyMetricKind.trades_per_month: "https://tradingstrategy.ai/glossary/trade-frequency",
   KeyMetricKind.decision_cycle_duration: "https://tradingstrategy.ai/glossary/cycle-duration",
}
