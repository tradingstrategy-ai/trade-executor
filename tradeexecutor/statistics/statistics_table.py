"""Module for statistics table. 

This table can work with any table data that can be represented as a dictionary of key metrics. 
It is also used to help display statistics in the web frontend.
"""
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import datetime
from typing import Literal

from tradeexecutor.analysis.trade_analyser import build_trade_analysis
from tradeexecutor.strategy.summary import KeyMetricKind, KeyMetricSource, KeyMetric
from tradeexecutor.state.state import State


@dataclass_json
@dataclass
class StatisticsTable:
    """
    - A table of statistics. 
    - This table can work with any table data that can be represented as a dictionary of key metrics. 
    - It is also used to help display statistics in the web frontend.
    """

    #: The columns of the table.
    columns: list[str]

    #: The rows of the table.
    rows: dict[KeyMetricKind, KeyMetric]
    
    #: The time at which the table was created.
    created_at: datetime.datetime

    #: The source of the table. Can either be live trading or backtesting.
    source: KeyMetricSource

    #: The start of the calculation window.
    calculation_window_start_at: datetime.datetime | None = None
    
    #: The end of the calculation window.
    calculation_window_end_at: datetime.timedelta | None = None


def serialise_long_short_stats_as_json_table(
    live_state: State,
    backtested_state: State | None,
    required_history: datetime.timedelta,
) -> StatisticsTable:
    """Calculate long/short statistics for the summary tile.

    :param state: Strategy state from which we calculate the summary
    :param execution_mode: If we need to skip calculations during backtesting.
    :param time_window: How long we look back for the summary statistics
    :param now_: Override current time for unit testing.
    :param legacy_workarounds: Skip some calculations on old data, because data is missing.
    :return: Summary statistics for all, long, and short positions
    """

    source_state, source, calculation_window_start_at, calculation_window_end_at = get_data_source_and_calculation_window(live_state, backtested_state, required_history)

    if source_state is None:
        return StatisticsTable(
            columns=["All", "Long", "Short"],
            created_at=datetime.datetime.utcnow(),
            source=source,
            rows={},
        )

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
                calculation_window_start_at = calculation_window_start_at,
                calculation_window_end_at = calculation_window_end_at,
                name = metric_data.name,
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


def get_data_source_and_calculation_window(
    live_state: State, 
    backtested_state: State | None, 
    required_history: datetime.timedelta,
)-> tuple[State | None, Literal[KeyMetricSource.live_trading, KeyMetricSource.backtesting] | None, datetime.datetime | None, datetime.datetime | None]:
    """Get the data source and calculation window for the key metrics.
    
    :param live_state: The current live execution state
    :param backtested_state: The backtested state
    :param required_history: How long history we need before using live execution as the basis for the key metric calculations
    :return: The data source and calculation window for the key metrics.
    """
    source_state, source, calculation_window_start_at, calculation_window_end_at = None, None, None, None
    live_history = live_state.portfolio.get_trading_history_duration()

    if live_history is not None and live_history >= required_history:
        source_state = live_state
        source = KeyMetricSource.live_trading
        calculation_window_start_at = source_state.created_at
        calculation_window_end_at = datetime.datetime.utcnow()
    elif backtested_state and backtested_state.portfolio.get_trading_history_duration():
        source_state = backtested_state
        source = KeyMetricSource.backtesting
        first_trade, last_trade = source_state.portfolio.get_first_and_last_executed_trade()
        calculation_window_start_at = first_trade.executed_at
        calculation_window_end_at = last_trade.executed_at

    return source_state, source, calculation_window_start_at, calculation_window_end_at