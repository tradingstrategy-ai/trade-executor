"""Summary statistics are displayed on the summary tiles of the strategies."""
import datetime
from time import perf_counter
from typing import Optional
import logging

import pandas as pd

from tradeexecutor.analysis.trade_analyser import calculate_annualised_return
from tradeexecutor.state.state import State
from tradeexecutor.statistics.key_metric import calculate_key_metrics
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.execution_context import ExecutionMode
from tradeexecutor.strategy.summary import StrategySummaryStatistics
from tradeexecutor.visual.equity_curve import calculate_compounding_realised_trading_profitability, calculate_cumulative_daily_returns, \
    calculate_compounding_unrealised_trading_profitability, extract_compounding_unrealised_trading_profitability_portfolio_statistics, calculate_share_price
from tradeexecutor.visual.web_chart import export_time_series

logger = logging.getLogger(__name__)



def prepare_share_price_summary_statistics(
    share_price_df: pd.DataFrame,
    start_at: pd.Timestamp,
    age: datetime.timedelta,
):
    """Profitability statistics for the share price-based returns.

    - Used for Lagoon vaults

    To run from the console:

    .. code-block:: python

        from tradeexecutor.visual.equity_curve import calculate_share_price
        from tradeexecutor.statistics.summary import prepare_share_price_summary_statistics

        share_price_df = calculate_share_price(state)
        returns_annualised, nav_90_days, performance_90_days = prepare_share_price_summary_statistics(
            share_price_df,
            start_at=pd.Timestamp("2025-05-01"),
            age=datetime.timedelta(days=30),
        )

    """

    assert isinstance(share_price_df, pd.DataFrame), "share_price_returns must be a pandas DataFrame"
    assert isinstance(start_at, pd.Timestamp), "start_at must be a pandas Timestamp"
    assert isinstance(age, datetime.timedelta), "age must be a datetime.timedelta"

    if len(share_price_df) > 0:
        returns_all_time = share_price_df["returns"].iloc[-1] - 1.0
        returns_annualised = calculate_annualised_return(returns_all_time, age)
    else:
        returns_annualised = 0
        nav_90_days = export_time_series(pd.Series())
        performance_90_days = export_time_series(pd.Series())
        return returns_annualised, nav_90_days, performance_90_days

    assert "returns" in share_price_df.columns, f"share_price_df must contain 'returns' column, got {share_price_df.columns}"

    logger.info("Returns %d, annualised %s", returns_all_time, returns_annualised)

    daily_resample = share_price_df.resample("1d").last()

    performance_90_days = daily_resample.loc[start_at:]["returns"]
    performance_90_days = performance_90_days.dropna()

    nav_90_days = daily_resample.loc[start_at:]["nav"]

    if len(performance_90_days) > 0:
        performance_90_days = performance_90_days
        first = performance_90_days.iloc[0]
        last = performance_90_days.iloc[-1]
    else:
        performance_90_days = pd.Series(dtype=float)
        first = None
        last = None

    performance_90_days = export_time_series(performance_90_days.dropna())
    nav_90_days = export_time_series(nav_90_days.dropna())

    logger.info("Profitability time windowed: %d entries, %s - %s", len(performance_90_days), first, last)

    return returns_annualised, nav_90_days, performance_90_days


def calculate_summary_statistics(
    state: State,
    execution_mode: ExecutionMode = ExecutionMode.one_off,
    time_window = pd.Timedelta(days=90),
    now_: Optional[pd.Timestamp | datetime.datetime] = None,
    legacy_workarounds=False,
    backtested_state: State | None = None,
    key_metrics_backtest_cut_off = datetime.timedelta(days=90),
    cycle_duration: Optional[CycleDuration] = None,
    share_price=False,
) -> StrategySummaryStatistics:
    """Preprocess the strategy statistics for the summary card in the web frontend.

    TODO: Rename to `calculate_strategy_preview_statistics()`.

    To test out in the :ref:`console`:

    .. code-block:: python

        from tradeexecutor.statistics.summary import calculate_summary_statistics
        from tradeexecutor.strategy.execution_context import ExecutionMode

        calculate_summary_statistics(state, ExecutionMode.preflight_check)

    :param state:
        Strategy state from which we calculate the summary

    :param execution_mode:
        If we need to skip calculations during backtesting.

    :param time_window:
        How long we look back for the summary statistics

    :param now_:
        Override current time for unit testing.

        Set this to the date of the last trade.

    :param legacy_workarounds:
        Skip some calculations on old data, because data is missing.

    :param backtested_state:

        The result of the earlier backtest run.
        The live web server needs to show backtested metrics on the side of
        live trading metrics. This state is used to calculate them.

    :param key_metrics_backtest_cut_off:
        How many days live data is collected until key metrics are switched from backtest to live trading based,

    :param cycle_duration:
        The duration of each trade decision cycle.

    :param share_price:
        Use share price-based profit calculations instead of equity curve.

    :return:
        Summary calculations for the summary tile,
        or empty `StrategySummaryStatistics` if cannot be calculated.
    """

    logger.info("calculate_summary_statistics() for %s", state.name)
    func_started_at = perf_counter()

    portfolio = state.portfolio

    # We can alway get the current value even if there are no trades
    current_value = portfolio.calculate_total_equity()

    first_trade, last_trade = portfolio.get_first_and_last_executed_trade()

    first_trade_at = first_trade.executed_at if first_trade else None
    last_trade_at = last_trade.executed_at if last_trade else None

    if not now_:
        now_ = pd.Timestamp.utcnow().tz_localize(None)

    start_at = now_ - time_window
    age = state.get_strategy_duration()

    stats = state.stats

    profitability_90_days = None
    enough_data = False
    performance_chart_90_days = None
    returns_all_time = returns_annualised = None
    compounding_unrealised_trading_profitability = None

    if len(stats.portfolio) > 0 and not legacy_workarounds:
        profitability = calculate_compounding_unrealised_trading_profitability(state, freq="D")

        unrealised_profit_data = extract_compounding_unrealised_trading_profitability_portfolio_statistics(state)
        compounding_unrealised_trading_profitability = export_time_series(unrealised_profit_data)

        enough_data = len(profitability.index) > 1 and profitability.index[0] <= start_at
        if len(profitability) >= 2:  # TypeError: cannot do slice indexing on RangeIndex with these indexers [2023-09-08 13:42:01.749186] of type Timestamp
            profitability_time_windowed = profitability[start_at:]
            if len(profitability_time_windowed) > 0:
                profitability_daily = profitability
                profitability_daily = profitability_daily[start_at:]
                # We do not generate entry for dates without trades so forward fill from the previous day
                profitability_daily = profitability_daily.ffill()
                profitability_90_days = profitability_daily.iloc[-1]
                performance_chart_90_days = export_time_series(profitability_daily)
                returns_all_time = profitability.iloc[-1]
            else:
                profitability_90_days = None
                performance_chart_90_days = None

    share_price_returns_90_days = nav_90_days =None
    if share_price:
        logger.info("Using share calculations for summary statistics, age %s, start_at %s", age, start_at)

        share_price_df = calculate_share_price(state)

        if share_price_df is not None:
            returns_annualised, nav_90_days, performance_chart_90_days = prepare_share_price_summary_statistics(
                share_price_df,
                age=age,
                start_at=start_at,
            )
            if len(performance_chart_90_days) > 0:
                profitability_90_days = performance_chart_90_days[-1][1]
                assert type(profitability_90_days) == float, f"We got {profitability_90_days}"
                share_price_returns_90_days = performance_chart_90_days

    else:
        logger.info("Using legacy profitability calculations for summary statistics")
        if age and returns_all_time:
            returns_annualised = calculate_annualised_return(returns_all_time, age)

    metrics_iter = calculate_key_metrics(
        state,
        backtested_state,
        required_history=key_metrics_backtest_cut_off,
        cycle_duration=cycle_duration,
        share_price_based=share_price,
    )

    key_metrics = {m.kind.value: m for m in metrics_iter}

    logger.info("calculate_summary_statistics() finished, took %s seconds", perf_counter() - func_started_at)

    return StrategySummaryStatistics(
        first_trade_at=first_trade_at,
        last_trade_at=last_trade_at,
        enough_data=enough_data,
        current_value=current_value,
        profitability_90_days=profitability_90_days,
        performance_chart_90_days=performance_chart_90_days,
        compounding_unrealised_trading_profitability=compounding_unrealised_trading_profitability,
        key_metrics=key_metrics,
        launched_at=state.created_at,
        backtest_metrics_cut_off_period=key_metrics_backtest_cut_off,
        return_all_time=returns_all_time,
        return_annualised=returns_annualised,
        share_price_returns_90_days=share_price_returns_90_days,
        nav_90_days=nav_90_days,
    )
