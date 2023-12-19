"""Key metric calculations.

Calculate key metrics used in the web frontend summary cards.
"""
import datetime
from typing import List, Iterable

import pandas as pd
import numpy as np

from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.state import State
from tradeexecutor.state.types import Percent
from tradeexecutor.strategy.summary import KeyMetric, KeyMetricKind, KeyMetricSource, KeyMetricCalculationMethod
from tradeexecutor.visual.equity_curve import calculate_size_relative_realised_trading_returns
from tradeexecutor.analysis.trade_analyser import build_trade_analysis, TradeSummary


def calculate_sharpe(returns: pd.Series, periods=365) -> float:
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


def calculate_sortino(returns: pd.Series, periods=365) -> float:
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


def calculate_profit_factor(returns: pd.Series) -> float:
    """Calculate profit factor.

    Internally uses quantstats.

    See :term:`profit factor`.

    :param returns:
        Returns series

    """
    # Lazy import to allow optional dependency
    from quantstats.stats import profit_factor
    return profit_factor(returns)


def calculate_max_drawdown(returns: pd.Series) -> Percent:
    """Calculate maximum drawdown.

    Internally uses quantstats.

    See :term:`maximum drawdown`.

    :param returns:
        Returns series

    :return:
        Negative value 0...-1

    """
    # Lazy import to allow optional dependency
    from quantstats.stats import to_drawdown_series
    dd = to_drawdown_series(returns)
    return dd.min()


def calculate_max_runup(returns: pd.Series) -> Percent:
    """Calculate maximum runup. Somewhat manual implementation since quantstats doesn't have this.

    :param returns:
        Returns series (can use original returns, doesn't have to be daily returns since not annualised)

    :return:
        Positive value
    
    """

    from quantstats import utils

    # convert returns to runup series
    prices = utils._prepare_prices(returns)
    ru = prices / np.minimum.accumulate(prices) - 1.
    runup_series = ru.replace([np.inf, -np.inf, -0], 0)
    
    return runup_series.max()


def calculate_profitability(returns: pd.Series) -> Percent:
    """Calculate annualised profitability.

    Internally uses quantstats.

    See :term:`profitability`.

    :param returns:
        Returns series

    :return:
        Value -1...inf

    """
    compounded = returns.add(1).cumprod().sub(1)
    return compounded[-1]


def calculate_trades_last_week(portfolio: Portfolio, cut_off_date=None) -> int:
    """How many trades were executed last week.

    See :term:`trades last week`.
    """
    if cut_off_date is None:
        cut_off_date = datetime.datetime.utcnow() - datetime.timedelta(days=7)

    end_date = datetime.datetime.utcnow()

    trades = portfolio.get_all_trades()
    trades_last_week = [t for t in trades if t.is_success() and t.executed_at >= cut_off_date and t.executed_at <= end_date]
    return len(trades_last_week)


def calculate_key_metrics(
        live_state: State,
        backtested_state: State | None = None,
        required_history = datetime.timedelta(days=90),
        freq_base: pd.DateOffset = pd.offsets.Day(),
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

    :param freq_base:
        The frequency for which we resample data when resamping is needed for calculations.

    :param now_:
        Override the current timestamp for testing

    :return:
        Key metrics.

        Currently sharpe, sortino, max drawdown and age.
    """

    assert isinstance(live_state, State)

    source_state = None
    source = None
    calculation_window_start_at = None
    calculation_window_end_at = None

    # Live history is calculated from the
    live_history = live_state.portfolio.get_trading_history_duration()
    if live_history is not None and live_history >= required_history:
        source_state = live_state
        source = KeyMetricSource.live_trading
        calculation_window_start_at = source_state.created_at
        calculation_window_end_at = datetime.datetime.utcnow()
    else:
        if backtested_state:
            if backtested_state.portfolio.get_trading_history_duration():
                source_state = backtested_state
                source = KeyMetricSource.backtesting
                first_trade, last_trade = source_state.portfolio.get_first_and_last_executed_trade()
                calculation_window_start_at = first_trade.executed_at
                calculation_window_end_at = last_trade.executed_at

    if source_state:

        # Use trading profitability instead of the fund performance
        # as the base for calculations to ensure
        # sharpe/sortino/etc. stays compatible regardless of deposit flow
        returns = calculate_size_relative_realised_trading_returns(source_state)
        returns = returns.resample(freq_base).max().fillna(0)

        periods = pd.Timedelta(days=365) / freq_base

        sharpe = calculate_sharpe(returns, periods=periods)
        yield KeyMetric.create_metric(KeyMetricKind.sharpe, source, sharpe, calculation_window_start_at, calculation_window_end_at, KeyMetricCalculationMethod.historical_data)

        sortino = calculate_sortino(returns, periods=periods)
        yield KeyMetric.create_metric(KeyMetricKind.sortino, source, sortino, calculation_window_start_at, calculation_window_end_at, KeyMetricCalculationMethod.historical_data)

        # Flip the sign of the max drawdown
        max_drawdown = -calculate_max_drawdown(returns)
        yield KeyMetric.create_metric(KeyMetricKind.max_drawdown, source, max_drawdown, calculation_window_start_at, calculation_window_end_at, KeyMetricCalculationMethod.historical_data)

        profitability = calculate_profitability(returns)
        yield KeyMetric.create_metric(KeyMetricKind.profitability, source, profitability, calculation_window_start_at, calculation_window_end_at, KeyMetricCalculationMethod.historical_data)

        if live_state:
            total_equity = live_state.portfolio.get_total_equity()

            # The total equity is made available always
            yield KeyMetric(
                KeyMetricKind.total_equity,
                KeyMetricSource.live_trading,
                total_equity,
                calculation_window_start_at=calculation_window_start_at,
                calculation_window_end_at=calculation_window_end_at,
                calculation_method=KeyMetricCalculationMethod.latest_value,
                help_link=KeyMetricKind.total_equity.get_help_link(),
            )

    else:
        # No live or backtesting data available,
        # mark all metrics N/A
        reason = "Not enough live trading or backtesting data available"
        calculation_window_start_at = None
        calculation_window_end_at = None

        yield KeyMetric.create_na(KeyMetricKind.sharpe, reason)
        yield KeyMetric.create_na(KeyMetricKind.sortino, reason)
        yield KeyMetric.create_na(KeyMetricKind.max_drawdown, reason)
        yield KeyMetric.create_na(KeyMetricKind.profitability, reason)
        yield KeyMetric.create_na(KeyMetricKind.total_equity, reason)

    # The age of the trading history is made available always
    yield KeyMetric(
        KeyMetricKind.started_at,
        KeyMetricSource.live_trading,
        live_state.created_at,
        calculation_window_start_at=calculation_window_start_at,
        calculation_window_end_at=calculation_window_end_at,
        calculation_method=KeyMetricCalculationMethod.latest_value,
        help_link=KeyMetricKind.started_at.get_help_link(),
    )

    # The age of the trading history is made available always
    #
    # Always live
    _, last_trade = live_state.portfolio.get_first_and_last_executed_trade()
    if last_trade is not None:
        yield KeyMetric(
            KeyMetricKind.last_trade,
            KeyMetricSource.live_trading,
            last_trade.executed_at,
            calculation_window_start_at=calculation_window_start_at,
            calculation_window_end_at=calculation_window_end_at,
            calculation_method=KeyMetricCalculationMethod.latest_value,
            help_link=KeyMetricKind.last_trade.get_help_link(),
        )

    # The age of the trading history is made available always
    #
    # Always live
    trades_last_week = calculate_trades_last_week(live_state.portfolio)
    yield KeyMetric(
        KeyMetricKind.trades_last_week,
        KeyMetricSource.live_trading,
        trades_last_week,
        calculation_window_start_at=calculation_window_start_at,
        calculation_window_end_at=calculation_window_end_at,
        calculation_method=KeyMetricCalculationMethod.latest_value,
        help_link=KeyMetricKind.trades_last_week.get_help_link(),
    )

    long_short_table = serialise_summary_statistics_as_json_table(source_state, source)
    yield from (row for row in long_short_table.rows.values())


class StatisticsTable:
      

    def __init__(
        self,
        columns: list[str],
        rows: dict[KeyMetricKind, KeyMetric], 
        created_at: datetime.datetime, 
        source: KeyMetricSource, 
        calculation_window_start_at: datetime.datetime | None = None, 
        calculation_window_end_at: datetime.timedelta | None = None
    ):
        """Create a new statistics table.
        
        :param columns: The columns of the table.
        :param rows: The rows of the table.
        :param created_at: The time at which the table was created.
        :param source: The source of the table. Can either be live trading or backtesting.
        :param calculation_window_start_at: The start of the calculation window.
        :param calculation_window_end_at: The end of the calculation window.
        """
        self.columns = columns
        self.rows = rows
        self.created_at = created_at
        self.source = source
        self.calculation_window_start_at = calculation_window_start_at
        self.calculation_window_end_at = calculation_window_end_at        


def serialise_summary_statistics_as_json_table(
    source_state: State,
    source: KeyMetricSource,
) -> StatisticsTable:
    """Calculate long/short statistics for the summary tile.

    :param state: Strategy state from which we calculate the summary
    :param execution_mode: If we need to skip calculations during backtesting.
    :param time_window: How long we look back for the summary statistics
    :param now_: Override current time for unit testing.
    :param legacy_workarounds: Skip some calculations on old data, because data is missing.
    :return: Summary statistics for all, long, and short positions
    """

    analysis = build_trade_analysis(source_state.portfolio)
    summary = analysis.calculate_all_summary_stats_by_side(state=source_state, urls=True)  # TODO timebucket

    key_metrics_map = {
        KeyMetricKind.trading_period_length: 'Trading period length',
        KeyMetricKind.return_percent: 'Return %',
        KeyMetricKind.annualised_return_percent: 'Annualised return %',
        KeyMetricKind.cash_at_start: 'Cash at start',
        KeyMetricKind.value_at_end: 'Value at end',
        KeyMetricKind.trade_volume: 'Trade volume',
        KeyMetricKind.position_win_percent: 'Position win percent',
        KeyMetricKind.total_positions: 'Total positions',
        KeyMetricKind.won_positions: 'Won positions',
        KeyMetricKind.lost_positions: 'Lost positions',
        KeyMetricKind.stop_losses_triggered: 'Stop losses triggered',
        KeyMetricKind.stop_loss_percent_of_all: 'Stop loss % of all',
        KeyMetricKind.stop_loss_percent_of_lost: 'Stop loss % of lost',
        KeyMetricKind.winning_stop_losses: 'Winning stop losses',
        KeyMetricKind.winning_stop_losses_percent: 'Winning stop losses percent',
        KeyMetricKind.losing_stop_losses: 'Losing stop losses',
        KeyMetricKind.losing_stop_losses_percent: 'Losing stop losses percent',
        KeyMetricKind.take_profits_triggered: 'Take profits triggered',
        KeyMetricKind.take_profit_percent_of_all: 'Take profit % of all',
        KeyMetricKind.take_profit_percent_of_won: 'Take profit % of won',
        KeyMetricKind.zero_profit_positions: 'Zero profit positions',
        KeyMetricKind.positions_open_at_the_end: 'Positions open at the end',
        KeyMetricKind.realised_profit_and_loss: 'Realised profit and loss',
        KeyMetricKind.unrealised_profit_and_loss: 'Unrealised profit and loss',
        KeyMetricKind.portfolio_unrealised_value: 'Portfolio unrealised value',
        KeyMetricKind.extra_returns_on_lending_pool_interest: 'Extra returns on lending pool interest',
        KeyMetricKind.cash_left_at_the_end: 'Cash left at the end',
        KeyMetricKind.average_winning_position_profit_percent: 'Average winning position profit %',
        KeyMetricKind.average_losing_position_loss_percent: 'Average losing position loss %',
        KeyMetricKind.biggest_winning_position_percent: 'Biggest winning position %',
        KeyMetricKind.biggest_losing_position_percent: 'Biggest losing position %',
        KeyMetricKind.average_duration_of_winning_positions: 'Average duration of winning positions',
        KeyMetricKind.average_duration_of_losing_positions: 'Average duration of losing positions',
        KeyMetricKind.lp_fees_paid: 'LP fees paid',
        KeyMetricKind.lp_fees_paid_percent_of_volume: 'LP fees paid % of volume',
        KeyMetricKind.average_position: 'Average position',
        KeyMetricKind.median_position: 'Median position',
        KeyMetricKind.most_consecutive_wins: 'Most consecutive wins',
        KeyMetricKind.most_consecutive_losses: 'Most consecutive losses',
        KeyMetricKind.biggest_realised_risk: 'Biggest realised risk',
        KeyMetricKind.avg_realised_risk: 'Avg realised risk',
        KeyMetricKind.max_pullback_of_total_capital: 'Max pullback of total capital',
        KeyMetricKind.max_loss_risk_at_opening_of_position: 'Max loss risk at opening of position',
        KeyMetricKind.max_drawdown: 'Max drawdown',
    }

    rows = {}
    for key_metric_kind, summary_index in key_metrics_map.items():
        if summary_index in summary.index:
            metric_data = summary.loc[summary_index]

            rows[key_metric_kind] = KeyMetric(
                kind=key_metric_kind,
                value={"All": metric_data[0], "Long": metric_data[1], "Short": metric_data[2]},
                help_link=metric_data[3],
                source=source_state,
                calculation_window_start_at = source_state.created_at,
                calculation_window_end_at = datetime.datetime.utcnow(),
            )

    if 'Average bars of winning positions' in summary.index:
        average_bars_of_winning_positions = summary.loc['Average bars of winning positions']
        average_bars_of_losing_positions = summary.loc['Average bars of losing positions']

        rows[KeyMetricKind.average_bars_of_winning_positions] = KeyMetric(
            kind=KeyMetricKind.average_bars_of_winning_positions,
            value={"All": average_bars_of_winning_positions[0], "Long": average_bars_of_winning_positions[1], "Short": average_bars_of_winning_positions[2]},
            help_link=average_bars_of_winning_positions[3],
        )
        rows[KeyMetricKind.average_bars_of_losing_positions] = KeyMetric(
            kind=KeyMetricKind.average_bars_of_losing_positions,
            value={"All": average_bars_of_losing_positions[0], "Long": average_bars_of_losing_positions[1], "Short": average_bars_of_losing_positions[2]},
            help_link=average_bars_of_losing_positions[3],
        )

    table = StatisticsTable(
        columns=["All", "Long", "Short"],
        created_at=source_state.created_at,
        source=source,
        rows=rows,
    )

    return table