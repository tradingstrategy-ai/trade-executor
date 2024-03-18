"""Module for statistics table. 

This table can work with any table data that can be represented as a dictionary of key metrics. 
It is also used to help display statistics in the web frontend.
"""
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import datetime
from typing import Literal

from tradeexecutor.analysis.trade_analyser import build_trade_analysis, calculate_annualised_return
from tradeexecutor.strategy.summary import KeyMetricKind, KeyMetricSource, KeyMetric
from tradeexecutor.state.state import State
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.statistics.key_metric import calculate_max_drawdown
from tradeexecutor.visual.equity_curve import calculate_compounding_realised_trading_profitability, calculate_non_cumulative_daily_returns
from tradeexecutor.utils.summarydataframe import as_percent, format_value


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
    rows: dict[str, KeyMetric]
    
    #: The time at which the table was created.
    created_at: datetime.datetime

    #: The source of the table. Can either be live trading or backtesting.
    source: KeyMetricSource | None = None

    #: The start of the calculation window.
    calculation_window_start_at: datetime.datetime | None = None
    
    #: The end of the calculation window.
    calculation_window_end_at: datetime.timedelta | None = None

    def __post_init__(self):
        if self.source is not None:
            assert isinstance(self.source, KeyMetricSource), f"Got {self.source}"
        assert type(self.columns) == list


def serialise_long_short_stats_as_json_table(
    live_state: State | None = None,
    backtested_state: State | None = None,
) -> dict[StatisticsTable]:
    """Calculate long/short statistics for the summary tile.

    :param live_state: Live trading strategy state
    :param backtested_state: Backtested strategy state
    :return: Dict with keys "live_stats" and "backtested_stats" containing the statistics tables for live and backtested trading.
    """
    
    live_stats = serialise_live_long_short_stats(live_state)
    backtested_stats = serialise_backtested_long_short_stats(backtested_state)
    
    return dict(
        live_stats=live_stats, 
        backtested_stats=backtested_stats
    )


def serialise_live_long_short_stats(live_state: State) -> StatisticsTable:
    """Serialise live long/short statistics as a JSON table.
    
    :param live_state: The current live execution state
    :return: The live long/short statistics as a JSON table.
    """
    live_start_at, live_end_at = None, None
    if live_state:
        live_start_at = live_state.created_at
        live_end_at = datetime.datetime.utcnow()
        
    live_stats = _serialise_long_short_stats_as_json_table(
        source_state=live_state,
        source=KeyMetricSource.live_trading,
        calculation_window_start_at=live_start_at,
        calculation_window_end_at=live_end_at,
    )
    return live_stats


def serialise_backtested_long_short_stats(backtested_state: State) -> StatisticsTable:
    """Serialise backtested long/short statistics as a JSON table.
    
    :param backtested_state: The backtested state
    :return: The backtested long/short statistics as a JSON table.
    """
    backtested_start_at, backtested_end_at = None, None
    if backtested_state:
        first_trade, last_trade = backtested_state.portfolio.get_first_and_last_executed_trade()
        backtested_start_at = first_trade.executed_at
        backtested_end_at = last_trade.executed_at
        
    backtested_stats = _serialise_long_short_stats_as_json_table(
        source_state=backtested_state,
        source=KeyMetricSource.backtesting,
        calculation_window_start_at=backtested_start_at,
        calculation_window_end_at=backtested_end_at,
    )
    return backtested_stats
    

def _serialise_long_short_stats_as_json_table(
    source_state: State,
    source,
    calculation_window_start_at: datetime.datetime,
    calculation_window_end_at: datetime.datetime,
) -> dict[StatisticsTable]:
    """Calculate long/short statistics for the summary tile."""
    
    if not source_state:
        return StatisticsTable(
            columns=["All", "Long", "Short"],
            created_at=datetime.datetime.utcnow(),
            source=source,
            rows={},
        )
        
    assert isinstance(calculation_window_start_at, datetime.datetime), "calculation_window_start_at is not a datetime"
    assert isinstance(calculation_window_end_at, datetime.datetime), "calculation_window_end_at is not a datetime"
    
    analysis = build_trade_analysis(source_state.portfolio)
    summary = analysis.calculate_all_summary_stats_by_side(state=source_state, urls=True)  # TODO timebucket

    # correct erroneous values if live
    compounding_returns = None
    if source == KeyMetricSource.live_trading and source_state:
        compounding_returns = calculate_compounding_realised_trading_profitability(source_state)
        summary.loc['Trading period length']['All'] = source_state.get_formatted_strategy_duration()
    
    if compounding_returns is not None and len(compounding_returns) > 0:
        daily_returns = calculate_non_cumulative_daily_returns(source_state)
        portfolio_return = compounding_returns.iloc[-1]
        annualised_return_percent = calculate_annualised_return(portfolio_return, calculation_window_end_at - calculation_window_start_at)
        summary.loc['Return %']['All'] = format_value(as_percent(portfolio_return))
        summary.loc['Annualised return %']['All'] = format_value(as_percent(annualised_return_percent))

        max_drawdown = -calculate_max_drawdown(daily_returns)
        summary.loc['Max drawdown']['All'] = format_value(as_percent(max_drawdown))

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
        KeyMetricKind.average_interest_paid_usd: 'average_interest_paid_usd',
        KeyMetricKind.median_interest_paid_usd: 'median_interest_paid_usd',
        KeyMetricKind.max_interest_paid_usd: 'max_interest_paid_usd',
        KeyMetricKind.min_interest_paid_usd: 'min_interest_paid_usd',
        KeyMetricKind.total_interest_paid_usd: 'total_interest_paid_usd',
        KeyMetricKind.average_duration_between_position_openings: 'average_duration_between_position_openings',
        KeyMetricKind.average_position_frequency: 'average_position_frequency',
    }

    rows = {}
    for key_metric_kind, summary_index in key_metrics_map.items():
        if summary_index in summary.index:
            metric_data = summary.loc[summary_index]
            
            for i in metric_data:
                assert isinstance(i, str | None), f"Should be string. Got {i}"

            rows[key_metric_kind.value] = KeyMetric(
                kind=key_metric_kind,
                value={"All": metric_data[0], "Long": metric_data[1], "Short": metric_data[2]},
                help_link=metric_data[3],
                source=source,
                calculation_window_start_at = calculation_window_start_at,
                calculation_window_end_at = calculation_window_end_at,
                name = metric_data.name,
            )

    if 'Average bars of winning positions' in summary.index:
        average_bars_of_winning_positions = summary.loc['Average bars of winning positions']
        average_bars_of_losing_positions = summary.loc['Average bars of losing positions']

        rows[KeyMetricKind.average_bars_of_winning_positions.value] = KeyMetric(
            kind=KeyMetricKind.average_bars_of_winning_positions,
            value={"All": average_bars_of_winning_positions[0], "Long": average_bars_of_winning_positions[1], "Short": average_bars_of_winning_positions[2]},
            help_link=average_bars_of_winning_positions[3],
        )
        rows[KeyMetricKind.average_bars_of_losing_positions.value] = KeyMetric(
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
