"""Key metric calculations.

Calculate key metrics used in the web frontend summary cards.
"""
import datetime
from typing import List, Iterable

from tradeexecutor.state.state import State
from tradeexecutor.strategy.summary import KeyMetric, KeyMetricKind, KeyMetricSource


def calculate_key_metrics(
        live_state: State,
        backtested_state: State | None = None,
        required_history = datetime.timedelta(days=90),
        now_: datetime.datetime | None = None,
) -> Iterable[KeyMetric]:
    """Calculate summary metrics to be displayed on the web frontend.

    - Metrics are calcualted either based live trading data or backtested data,
      whichever makes more sense

    - Live execution state is used if it has enough history

    :param live_state:
        The current live execution state

    :param backtested_state:
        The backtested state

    :param required_history:
        How long history we need before using live execution
        as the basis for the key metric calculations

    :param now_:
        Override the current timestamp for testing

    :return:
        Key metrics
    """

    assert isinstance(live_state, State)

    if not now_:
        now_ = datetime.datetime.utcnow()

    # Live history is calculated from the
    live_history = now_ - live_state.created_at
    if live_history >= required_history:
        source_state = live_state
        source = KeyMetricSource.live_trading
    else:
        if backtested_state:
            source_state = backtested_state
            source = KeyMetricSource.backtesting
        else:
            source_state = None

    if source_state and source_state.has_trading_history():
        # We have one state based on which we can calculate metrics
        first_trade, last_trade = source_state.portfolio.get_first_and_last_executed_trade()
        calculation_window_start_at = first_trade.executed_at
        calculation_window_end_at = last_trade.executed_at

    else:
        reason = "Not enough live trading or backtesting data available"
        yield KeyMetric.create_na(KeyMetricKind.sharpe, reason)
        yield KeyMetric.create_na(KeyMetricKind.sortino, reason)
        yield KeyMetric.create_na(KeyMetricKind.max_drawdown, reason)

    yield KeyMetric(
        KeyMetricKind.started_at,
        KeyMetricSource.live_trading,
        live_state.created_at,
    )