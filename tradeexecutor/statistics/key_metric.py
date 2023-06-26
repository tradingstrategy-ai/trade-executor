"""Key metric calculations.

Calculate key metrics used in the web frontend summary cards.
"""
import datetime
from typing import List, Iterable

import pandas as pd

from tradeexecutor.state.state import State
from tradeexecutor.strategy.summary import KeyMetric, KeyMetricKind, KeyMetricSource
from tradeexecutor.visual.equity_curve import calculate_deposit_adjusted_returns



def calculate_sharpe(returns: pd.Series, periods=365):
    """Calculate annualised sharpe ratio.

    Internally uses quantstats.

    See :term:`sharpe`.

    :param returns:
        Returns series

    :param periods:
        How many periods per year returns series has

    """
    # Lazy import to allow optional dependency
    from quantstats.stats import sharpe
    return sharpe(
        returns,
        periods=periods,
    )


def calculate_sortino(returns: pd.Series, periods=365):
    """Calculate annualised share ratio.

    Internally uses quantstats.

    See :term:`sortino`.

    :param returns:
        Returns series

    :param periods:
        How many periods per year returns series has

    """
    # Lazy import to allow optional dependency
    from quantstats.stats import sortino
    return sortino(
        returns,
        periods=periods,
    )


def calculate_key_metrics(
        live_state: State,
        backtested_state: State | None = None,
        required_history = datetime.timedelta(days=90),
) -> Iterable[KeyMetric]:
    """Calculate summary metrics to be displayed on the web frontend.

    - Metrics are calculated either based live trading data or backtested data,
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

    source_state = None
    source = None

    # Live history is calculated from the
    live_history = live_state.portfolio.get_trading_history_duration()
    if live_history >= required_history:
        source_state = live_state
        source = KeyMetricSource.live_trading
    else:
        if backtested_state:
            if backtested_state.portfolio.get_trading_history_duration():
                source_state = backtested_state
                source = KeyMetricSource.backtesting

    if source_state:
        # We have one state based on which we can calculate metrics
        first_trade, last_trade = source_state.portfolio.get_first_and_last_executed_trade()
        calculation_window_start_at = first_trade.executed_at
        calculation_window_end_at = last_trade.executed_at

        # Use deposit/redemption flow adjusted equity curve
        returns = calculate_deposit_adjusted_returns(source_state)

        sharpe = calculate_sharpe(returns, periods=365)
        yield KeyMetric.create_metric(KeyMetricKind.sharpe, source, sharpe, calculation_window_start_at, calculation_window_end_at)

        sortino = calculate_sortino(returns, periods=365)
        yield KeyMetric.create_metric(KeyMetricKind.sortino, source, sortino, calculation_window_start_at, calculation_window_end_at)

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