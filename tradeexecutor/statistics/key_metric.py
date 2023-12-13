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
from tradeexecutor.analysis.trade_analyser import build_trade_analysis


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


def calculate_long_short_metrics(
    source_state: State
) -> StatisticsTable:
    """Calculate long/short statistics for the summary tile.

    :param state: Strategy state from which we calculate the summary
    :param execution_mode: If we need to skip calculations during backtesting.
    :param time_window: How long we look back for the summary statistics
    :param now_: Override current time for unit testing.
    :param legacy_workarounds: Skip some calculations on old data, because data is missing.

    """

    analysis = build_trade_analysis(source_state.portfolio)
    summary = analysis.calculate_all_summary_stats_by_side(state=source_state, format_headings=False)  # TODO timebucket

    trading_period_length = summary.loc['Trading period length']
    return_percent = summary.loc['Return %']
    annualised_return_percent = summary.loc['Annualised return %']
    cash_at_start = summary.loc['Cash at start']
    value_at_end = summary.loc['Value at end']
    trade_volume = summary.loc['Trade volume']
    position_win_percent = summary.loc['Position win percent']
    total_positions = summary.loc['Total positions']
    won_positions = summary.loc['Won positions']
    lost_positions = summary.loc['Lost positions']
    stop_losses_triggered = summary.loc['Stop losses triggered']
    stop_loss_percent_of_all = summary.loc['Stop loss % of all']
    stop_loss_percent_of_lost = summary.loc['Stop loss % of lost']
    winning_stop_losses = summary.loc['Winning stop losses']
    winning_stop_losses_percent = summary.loc['Winning stop losses percent']
    losing_stop_losses = summary.loc['Losing stop losses']
    losing_stop_losses_percent = summary.loc['Losing stop losses percent']
    take_profits_triggered = summary.loc['Take profits triggered']
    take_profit_percent_of_all = summary.loc['Take profit % of all']
    take_profit_percent_of_won = summary.loc['Take profit % of won']
    zero_profit_positions = summary.loc['Zero profit positions']
    positions_open_at_the_end = summary.loc['Positions open at the end']
    realised_profit_and_loss = summary.loc['Realised profit and loss']
    unrealised_profit_and_loss = summary.loc['Unrealised profit and loss']
    portfolio_unrealised_value = summary.loc['Portfolio unrealised value']
    extra_returns_on_lending_pool_interest = summary.loc['Extra returns on lending pool interest']
    cash_left_at_the_end = summary.loc['Cash left at the end']
    average_winning_position_profit_percent = summary.loc['Average winning position profit %']
    average_losing_position_loss_percent = summary.loc['Average losing position loss %']
    biggest_winning_position_percent = summary.loc['Biggest winning position %']
    biggest_losing_position_percent = summary.loc['Biggest losing position %']
    average_duration_of_winning_positions = summary.loc['Average duration of winning positions']
    average_duration_of_losing_positions = summary.loc['Average duration of losing positions']
    average_bars_of_winning_positions = summary.loc['Average bars of winning positions']
    average_bars_of_losing_positions = summary.loc['Average bars of losing positions']
    lp_fees_paid = summary.loc['LP fees paid']
    lp_fees_paid_percent_of_volume = summary.loc['LP fees paid % of volume']
    average_position = summary.loc['Average position']
    median_position = summary.loc['Median position']
    most_consecutive_wins = summary.loc['Most consecutive wins']
    most_consecutive_losses = summary.loc['Most consecutive losses']
    biggest_realised_risk = summary.loc['Biggest realised risk']
    avg_realised_risk = summary.loc['Avg realised risk']
    max_pullback_of_total_capital = summary.loc['Max pullback of total capital']
    max_loss_risk_at_opening_of_position = summary.loc['Max loss risk at opening of position']
    max_drawdown = summary.loc['Max drawdown']

    help_links = summary['help_links']

    table = StatisticsTable(
        columns=["All", "Long", "Short"],
        rows={
            KeyMetricKind.trading_period_length: KeyMetric(
                kind=KeyMetricKind.trading_period_length,
                value={"All": trading_period_length[0], "Long": trading_period_length[1], "Short": trading_period_length[2]},
                help_link=trading_period_length[3],
            ),
            KeyMetricKind.return_percent: KeyMetric(
                kind=KeyMetricKind.return_percent,
                value={"All": return_percent[0], "Long": return_percent[1], "Short": return_percent[2]},
                help_link=return_percent[3],
            ),
            KeyMetricKind.annualised_return_percent: KeyMetric(
                kind=KeyMetricKind.annualised_return_percent,
                value={"All": annualised_return_percent[0], "Long": annualised_return_percent[1], "Short": annualised_return_percent[2]},
                help_link=annualised_return_percent[3],
            ),
            KeyMetricKind.cash_at_start: KeyMetric(
                kind=KeyMetricKind.cash_at_start,
                value={"All": cash_at_start[0], "Long": cash_at_start[1], "Short": cash_at_start[2]},
                help_link=cash_at_start[3],
            ),
            KeyMetricKind.value_at_end: KeyMetric(
                kind=KeyMetricKind.value_at_end,
                value={"All": value_at_end[0], "Long": value_at_end[1], "Short": value_at_end[2]},
                help_link=value_at_end[3],
            ),
            KeyMetricKind.trade_volume: KeyMetric(
                kind=KeyMetricKind.trade_volume,
                value={"All": trade_volume[0], "Long": trade_volume[1], "Short": trade_volume[2]},
                help_link=trade_volume[3],
            ),
            KeyMetricKind.position_win_percent: KeyMetric(
                kind=KeyMetricKind.position_win_percent,
                value={"All": position_win_percent[0], "Long": position_win_percent[1], "Short": position_win_percent[2]},
                help_link=position_win_percent[3],
            ),
            KeyMetricKind.total_positions: KeyMetric(
                kind=KeyMetricKind.total_positions,
                value={"All": total_positions[0], "Long": total_positions[1], "Short": total_positions[2]},
                help_link=total_positions[3],
            ),
            KeyMetricKind.won_positions: KeyMetric(
                kind=KeyMetricKind.won_positions,
                value={"All": won_positions[0], "Long": won_positions[1], "Short": won_positions[2]},
                help_link=won_positions[3],
            ),
            KeyMetricKind.lost_positions: KeyMetric(
                kind=KeyMetricKind.lost_positions,
                value={"All": lost_positions[0], "Long": lost_positions[1], "Short": lost_positions[2]},
                help_link=lost_positions[3],
            ),
            KeyMetricKind.stop_losses_triggered: KeyMetric(
                kind=KeyMetricKind.stop_losses_triggered,
                value={"All": stop_losses_triggered[0], "Long": stop_losses_triggered[1], "Short": stop_losses_triggered[2]},
                help_link=stop_losses_triggered[3],
            ),
            KeyMetricKind.stop_loss_percent_of_all: KeyMetric(
                kind=KeyMetricKind.stop_loss_percent_of_all,
                value={"All": stop_loss_percent_of_all[0], "Long": stop_loss_percent_of_all[1], "Short": stop_loss_percent_of_all[2]},
                help_link=stop_loss_percent_of_all[3],
            ),
            KeyMetricKind.stop_loss_percent_of_lost: KeyMetric(
                kind=KeyMetricKind.stop_loss_percent_of_lost,
                value={"All": stop_loss_percent_of_lost[0], "Long": stop_loss_percent_of_lost[1], "Short": stop_loss_percent_of_lost[2]},
                help_link=stop_loss_percent_of_lost[3],
            ),
            KeyMetricKind.winning_stop_losses: KeyMetric(
                kind=KeyMetricKind.winning_stop_losses,
                value={"All": winning_stop_losses[0], "Long": winning_stop_losses[1], "Short": winning_stop_losses[2]},
                help_link=winning_stop_losses[3],
            ),
            KeyMetricKind.winning_stop_losses_percent: KeyMetric(
                kind=KeyMetricKind.winning_stop_losses_percent,
                value={"All": winning_stop_losses_percent[0], "Long": winning_stop_losses_percent[1], "Short": winning_stop_losses_percent[2]},
                help_link=winning_stop_losses_percent[3],
            ),
            KeyMetricKind.losing_stop_losses: KeyMetric(
                kind=KeyMetricKind.losing_stop_losses,
                value={"All": losing_stop_losses[0], "Long": losing_stop_losses[1], "Short": losing_stop_losses[2]},
                help_link=losing_stop_losses[3],
            ),
            KeyMetricKind.losing_stop_losses_percent: KeyMetric(
                kind=KeyMetricKind.losing_stop_losses_percent,
                value={"All": losing_stop_losses_percent[0], "Long": losing_stop_losses_percent[1], "Short": losing_stop_losses_percent[2]},
                help_link=losing_stop_losses_percent[3],
            ),
            KeyMetricKind.take_profits_triggered: KeyMetric(
                kind=KeyMetricKind.take_profits_triggered,
                value={"All": take_profits_triggered[0], "Long": take_profits_triggered[1], "Short": take_profits_triggered[2]},
                help_link=take_profits_triggered[3],
            ),
            KeyMetricKind.take_profit_percent_of_all: KeyMetric(
                kind=KeyMetricKind.take_profit_percent_of_all,
                value={"All": take_profit_percent_of_all[0], "Long": take_profit_percent_of_all[1], "Short": take_profit_percent_of_all[2]},
                help_link=take_profit_percent_of_all[3],
            ),
            KeyMetricKind.take_profit_percent_of_won: KeyMetric(
                kind=KeyMetricKind.take_profit_percent_of_won,
                value={"All": take_profit_percent_of_won[0], "Long": take_profit_percent_of_won[1], "Short": take_profit_percent_of_won[2]},
                help_link=take_profit_percent_of_won[3],
            ),
            KeyMetricKind.zero_profit_positions: KeyMetric(
                kind=KeyMetricKind.zero_profit_positions,
                value={"All": zero_profit_positions[0], "Long": zero_profit_positions[1], "Short": zero_profit_positions[2]},
                help_link=zero_profit_positions[3],
            ),
            KeyMetricKind.positions_open_at_the_end: KeyMetric(
                kind=KeyMetricKind.positions_open_at_the_end,
                value={"All": positions_open_at_the_end[0], "Long": positions_open_at_the_end[1], "Short": positions_open_at_the_end[2]},
                help_link=positions_open_at_the_end[3],
            ),
            KeyMetricKind.realised_profit_and_loss: KeyMetric(
                kind=KeyMetricKind.realised_profit_and_loss,
                value={"All": realised_profit_and_loss[0], "Long": realised_profit_and_loss[1], "Short": realised_profit_and_loss[2]},
                help_link=realised_profit_and_loss[3],
            ),
            KeyMetricKind.unrealised_profit_and_loss: KeyMetric(
                kind=KeyMetricKind.unrealised_profit_and_loss,
                value={"All": unrealised_profit_and_loss[0], "Long": unrealised_profit_and_loss[1], "Short": unrealised_profit_and_loss[2]},
                help_link=unrealised_profit_and_loss[3],
            ),
            KeyMetricKind.portfolio_unrealised_value: KeyMetric(
                kind=KeyMetricKind.portfolio_unrealised_value,
                value={"All": portfolio_unrealised_value[0], "Long": portfolio_unrealised_value[1], "Short": portfolio_unrealised_value[2]},
                help_link=portfolio_unrealised_value[3],
            ),
            KeyMetricKind.extra_returns_on_lending_pool_interest: KeyMetric(
                kind=KeyMetricKind.extra_returns_on_lending_pool_interest,
                value={"All": extra_returns_on_lending_pool_interest[0], "Long": extra_returns_on_lending_pool_interest[1], "Short": extra_returns_on_lending_pool_interest[2]},
                help_link=extra_returns_on_lending_pool_interest[3],
            ),
            KeyMetricKind.cash_left_at_the_end: KeyMetric(
                kind=KeyMetricKind.cash_left_at_the_end,
                value={"All": cash_left_at_the_end[0], "Long": cash_left_at_the_end[1], "Short": cash_left_at_the_end[2]},
                help_link=cash_left_at_the_end[3],
            ),
            KeyMetricKind.average_winning_position_profit_percent: KeyMetric(
                kind=KeyMetricKind.average_winning_position_profit_percent,
                value={"All": average_winning_position_profit_percent[0], "Long": average_winning_position_profit_percent[1], "Short": average_winning_position_profit_percent[2]},
                help_link=average_winning_position_profit_percent[3],
            ),
            KeyMetricKind.average_losing_position_loss_percent: KeyMetric(
                kind=KeyMetricKind.average_losing_position_loss_percent,
                value={"All": average_losing_position_loss_percent[0], "Long": average_losing_position_loss_percent[1], "Short": average_losing_position_loss_percent[2]},
                help_link=average_losing_position_loss_percent[3],
            ),
            KeyMetricKind.biggest_winning_position_percent: KeyMetric(
                kind=KeyMetricKind.biggest_winning_position_percent,
                value={"All": biggest_winning_position_percent[0], "Long": biggest_winning_position_percent[1], "Short": biggest_winning_position_percent[2]},
                help_link=biggest_winning_position_percent[3],
            ),
            KeyMetricKind.biggest_losing_position_percent: KeyMetric(
                kind=KeyMetricKind.biggest_losing_position_percent,
                value={"All": biggest_losing_position_percent[0], "Long": biggest_losing_position_percent[1], "Short": biggest_losing_position_percent[2]},
                help_link=biggest_losing_position_percent[3],
            ),
            KeyMetricKind.average_duration_of_winning_positions: KeyMetric(
                kind=KeyMetricKind.average_duration_of_winning_positions,
                value={"All": average_duration_of_winning_positions[0], "Long": average_duration_of_winning_positions[1], "Short": average_duration_of_winning_positions[2]},
                help_link=average_duration_of_winning_positions[3],
            ),
            KeyMetricKind.average_duration_of_losing_positions: KeyMetric(
                kind=KeyMetricKind.average_duration_of_losing_positions,
                value={"All": average_duration_of_losing_positions[0], "Long": average_duration_of_losing_positions[1], "Short": average_duration_of_losing_positions[2]},
                help_link=average_duration_of_losing_positions[3],
            ),
            KeyMetricKind.average_bars_of_winning_positions: KeyMetric(
                kind=KeyMetricKind.average_bars_of_winning_positions,
                value={"All": average_bars_of_winning_positions[0], "Long": average_bars_of_winning_positions[1], "Short": average_bars_of_winning_positions[2]},
                help_link=average_bars_of_winning_positions[3],
            ),
            KeyMetricKind.average_bars_of_losing_positions: KeyMetric(
                kind=KeyMetricKind.average_bars_of_losing_positions,
                value={"All": average_bars_of_losing_positions[0], "Long": average_bars_of_losing_positions[1], "Short": average_bars_of_losing_positions[2]},
                help_link=average_bars_of_losing_positions[3],
            ),
            KeyMetricKind.lp_fees_paid: KeyMetric(
                kind=KeyMetricKind.lp_fees_paid,
                value={"All": lp_fees_paid[0], "Long": lp_fees_paid[1], "Short": lp_fees_paid[2]},
                help_link=lp_fees_paid[3],
            ),
            KeyMetricKind.lp_fees_paid_percent_of_volume: KeyMetric(
                kind=KeyMetricKind.lp_fees_paid_percent_of_volume,
                value={"All": lp_fees_paid_percent_of_volume[0], "Long": lp_fees_paid_percent_of_volume[1], "Short": lp_fees_paid_percent_of_volume[2]},
                help_link=lp_fees_paid_percent_of_volume[3],
            ),
            KeyMetricKind.average_position: KeyMetric(
                kind=KeyMetricKind.average_position,
                value={"All": average_position[0], "Long": average_position[1], "Short": average_position[2]},
                help_link=average_position[3],
            ),
            KeyMetricKind.median_position: KeyMetric(
                kind=KeyMetricKind.median_position,
                value={"All": median_position[0], "Long": median_position[1], "Short": median_position[2]},
                help_link=median_position[3],
            ),
            KeyMetricKind.most_consecutive_wins: KeyMetric(
                kind=KeyMetricKind.most_consecutive_wins,
                value={"All": most_consecutive_wins[0], "Long": most_consecutive_wins[1], "Short": most_consecutive_wins[2]},
                help_link=most_consecutive_wins[3],
            ),
            KeyMetricKind.most_consecutive_losses: KeyMetric(
                kind=KeyMetricKind.most_consecutive_losses,
                value={"All": most_consecutive_losses[0], "Long": most_consecutive_losses[1], "Short": most_consecutive_losses[2]},
                help_link=most_consecutive_losses[3],
            ),
            KeyMetricKind.biggest_realised_risk: KeyMetric(
                kind=KeyMetricKind.biggest_realised_risk,
                value={"All": biggest_realised_risk[0], "Long": biggest_realised_risk[1], "Short": biggest_realised_risk[2]},
                help_link=biggest_realised_risk[3],
            ),
            KeyMetricKind.avg_realised_risk: KeyMetric(
                kind=KeyMetricKind.avg_realised_risk,
                value={"All": avg_realised_risk[0], "Long": avg_realised_risk[1], "Short": avg_realised_risk[2]},
                help_link=avg_realised_risk[3],
            ),
            KeyMetricKind.max_pullback_of_total_capital: KeyMetric(
                kind=KeyMetricKind.max_pullback_of_total_capital,
                value={"All": max_pullback_of_total_capital[0], "Long": max_pullback_of_total_capital[1], "Short": max_pullback_of_total_capital[2]},
                help_link=max_pullback_of_total_capital[3],
            ),
            KeyMetricKind.max_loss_risk_at_opening_of_position: KeyMetric(
                kind=KeyMetricKind.max_loss_risk_at_opening_of_position,
                value={"All": max_loss_risk_at_opening_of_position[0], "Long": max_loss_risk_at_opening_of_position[1], "Short": max_loss_risk_at_opening_of_position[2]},
                help_link=max_loss_risk_at_opening_of_position[3],
            ),
            KeyMetricKind.max_drawdown: KeyMetric(
                kind=KeyMetricKind.max_drawdown,
                value={"All": max_drawdown[0], "Long": max_drawdown[1], "Short": max_drawdown[2]},
                help_link=max_drawdown[3],
            ),
        }
    )

    # add state to each row
    for row in table.rows:
        row.source = source_state
        row.calculation_window_start_at = source_state.created_at
        row.calculation_window_end_at = datetime.datetime.utcnow()