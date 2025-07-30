"""Equity curve based statistics and visualisations.

For more information see the narrative documentation on :ref:`profitability`.
"""
import datetime
import warnings
from typing import List

import pandas as pd
import numpy as np
from matplotlib.figure import Figure

from tradeexecutor.analysis.curve import CurveType, DEFAULT_BENCHMARK_COLOURS
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.state import State
from tradeexecutor.state.statistics import Statistics, PortfolioStatistics
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.visual.qs_wrapper import import_quantstats_wrapped


def calculate_equity_curve(
    state: State,
    attribute_name="total_equity",
    fill_time_gaps=False,
) -> pd.Series:
    """Calculate equity curve for the portfolio.

    Translate the portfolio internal :py:attr:`Statistics.portfolio`
    to :py:class:`pd.Series` that allows easy equity curve calculations.

    This reads :py:class:`tradeexecutor.state.stats.PortfolioStatistics`

    Example:

    .. code-block:: python

        equity_curve = calculate_equity_curve(state)

    :param attribute_name:
        Calculate equity curve based on this attribute of :py:class:`

    :param fill_time_gaps:
        Insert a faux book keeping entries at start and end.

        If not set, only renders the chart when there was some activate
        deposits and ignores non-activity gaps at start and end.

        See :py:meth:`tradeexecutor.state.state.State.get_strategy_time_range`.

    :return:
        Pandas series (timestamp, equity value).

        Index is DatetimeIndex.

        Empty series is returned if there is no data.

        We ensure only one entry per timestamp through
        filtering out duplicate indices.

    """

    stats: Statistics = state.stats
    portfolio_stats: List[PortfolioStatistics] = stats.portfolio
    data = [(s.calculated_at, getattr(s, attribute_name)) for s in portfolio_stats]

    if len(data) == 0:
        return pd.Series([], index=pd.to_datetime([]), dtype='float64')

    if fill_time_gaps:
        start, end = state.get_strategy_time_range()
        end_val = data[-1][1]

        if data[0][0] != start:
            data = [(start, 0)] + data

        if end != end_val:
            data.append((end, end_val))

        # TODO: fill_time_gaps missing forward fill
        # for slowly starting strategies

    # https://stackoverflow.com/a/66772284/315168
    df = pd.DataFrame(data).set_index(0)[1]

    # Remove duplicates
    #
    # Happens in unit tests as we get a calculate event from deposit
    # and recalculatin at the same timestam

    # 0
    # 2021-06-01 00:00:00.000000        0.000000
    # 2023-10-18 10:04:05.834542    10000.000000
    # 2021-06-01 00:00:00.000000    10000.000000

    # https://stackoverflow.com/a/34297689/315168
    series = df[~df.index.duplicated(keep='last')]
    # See curve.py
    series.attrs["name"] = state.name
    series.attrs["curve"] = CurveType.equity
    series.attrs["colour"] = DEFAULT_BENCHMARK_COLOURS["Strategy"]
    return series


def calculate_returns(equity_curve: pd.Series) -> pd.Series:
    """Calculate returns of an equity curve.

    Calculate % change for each time point we have on the equity curve.

    See also https://quantdare.com/calculate-monthly-returns-with-pandas/

    :param equity_curve:
        The equity curve.

        Created with :py:func:`calculate_equity_curve`

    :return:
        Series of returns.

        This is % change over the previous entry in the time series.


    """

    if len(equity_curve) == 0:
        series = pd.Series([], index=pd.to_datetime([]), dtype='float64')
    else:
        series = equity_curve.pct_change().fillna(0.0)

    series.attrs = equity_curve.attrs.copy()
    series.attrs["curve"] = CurveType.returns

    return series


def generate_buy_and_hold_returns(
    buy_and_hold_price_series: pd.Series,
):
    """Create a benchmark series based on price action.

    - Create a returns series that can be used as a benchmark in :py:func:`tradeexecutor.analysis.advanced_metrics.visualise_advanced_metrics`

    Example:

    .. code-block:: python

        eth_index = strategy_universe.data_universe.candles.get_candles_by_pair(eth_pair)["close"]
        benchmark_returns = generate_buy_and_hold_returns(eth_index)

    """
    assert isinstance(buy_and_hold_price_series, pd.Series)
    returns = calculate_returns(buy_and_hold_price_series)

    returns.attrs["curve"] = CurveType.returns
    return returns


def calculate_cumulative_return(returns: pd.Series) -> pd.Series:
    """Calculate cumulative returns for the equity curve.

    See :term:`Cumulative return`.

    :param returns:
        The returns curve of a portfolio.

        See :py:func:`calculate_returns`.

    :return:
        Pandas DataFrame representing cumulative returns.
    """

    raise NotImplementedError()

    # Taken from qstrader.statistics.performance
    #return np.exp(np.log(1 + returns).cumsum())[-1] - 1


def calculate_aggregate_returns(equity_curve: pd.Series, freq: str | pd.DateOffset = "BM") -> pd.Series:
    """Calculate strategy aggregatd results over different timespans.

    Good to calculate

    - Monthly returns

    - Quaterly returns

    See :term:`Aggregate return` for more information what this metric represents.

    .. note ::

        There are multiple ways to calculate aggregated returns and they give a bit different results.
        See this article for details: https://quantdare.com/calculate-monthly-returns-with-pandas/

    .. note ::

        The current simplicist method does not calculate returns for the first and last period.

    :param equity_curve:
        The equity curve of the portfolio.

        See :py:func:`calculate_equity_curve`

    :param freq:

        Pandas frequency string.

        The default value is "month-end frequency".

        For valid values see https://stackoverflow.com/a/35339226/315168

    :return:
        Monthly returns for each month.

        The array is keyed by the end date of the period e.g. July is `2021-07-31`.

        The first month can be incomplete, so its value is `NaN`.
    """
    assert isinstance(equity_curve.index, pd.DatetimeIndex), f"Got {equity_curve.index}"

    if len(equity_curve) == 0:
        return pd.Series([], index=pd.to_datetime([]), dtype='float64')

    equity_curve.sort_index(inplace=True)

    # Each equity curve sample is the last day of the period
    # https://stackoverflow.com/a/14039589/315168
    sampled = equity_curve.asfreq(freq, method='ffill')
    return sampled.pct_change()


def calculate_daily_returns(
    state: State,
    freq: pd.DateOffset | str= "D",
) -> (pd.Series | None):
    """Calculate daily returns of a backtested results.

    Used for advanced statistics.

    .. warning::

        Uses equity curve to calculate profits. This does not correctly
        handle deposit/redemptions. Use in backtesting only.

    :param freq:
        Frequency of the binned data.

    :returns:
        If valid state provided, returns are returned as calendar day (D) frequency, else None"""

    if state.backtest_data:
        equity_curve = calculate_equity_curve(state, fill_time_gaps=True)
    else:
        # Legacy
        equity_curve = calculate_equity_curve(state, fill_time_gaps=False)
    returns = calculate_aggregate_returns(equity_curve, freq=freq)
    return returns


def visualise_equity_curve(
    returns: pd.Series,
    title="Equity curve",
    line_width=1.5,
) -> Figure:
    """Draw equity curve, drawdown and daily returns using Quantstats.

    `See Quantstats README for more details <https://github.com/ranaroussi/quantstats>`__.

    See also

    - :py:func:`tradeexecutor.visual.benchmark.visualise_equity_curve_benchmark`

    - :py:func:`tradeexecutor.visual.grid_search.visualise_single_grid_search_result_benchmark`

    Example:

    .. code-block:: python

        from tradeexecutor.visual.equity_curve import calculate_equity_curve, calculate_returns
        from tradeexecutor.visual.equity_curve import visualise_equity_curve

        curve = calculate_equity_curve(state)
        returns = calculate_returns(curve)
        fig = visualise_equity_curve(returns)
        display(fig)

    :return:
        Matplotlit figure

    """
    qs = import_quantstats_wrapped()
    fig = qs.plots.snapshot(
        returns,
        title=title,
        lw=line_width,
        show=False,
        subtitle=False,
        )
    return fig


def visualise_returns_over_time(
        returns: pd.Series,
) -> Figure:
    """Draw a grid of returns over time.

    - Currently only monthly breakdown supported

    - `See Quantstats README for more details <https://github.com/ranaroussi/quantstats>`__.

    Example:

    .. code-block:: python

        from tradeexecutor.visual.equity_curve import calculate_equity_curve, calculate_returns
        from tradeexecutor.visual.equity_curve import visualise_returns_over_time

        curve = calculate_equity_curve(state)
        returns = calculate_returns(curve)
        fig = visualise_equity_performance(returns)
        display(fig)

    :return:
        Matplotlit figure

    """

    # /Users/moo/Library/Caches/pypoetry/virtualenvs/tradingview-defi-strategy-XB2Vkmi1-py3.10/lib/python3.10/site-packages/quantstats/stats.py:968: FutureWarning:
    #
    # In a future version of pandas all arguments of DataFrame.pivot will be keyword-only.

    qs = import_quantstats_wrapped()

    with warnings.catch_warnings():  #  DeprecationWarning: Importing display from IPython.core.display is deprecated since IPython 7.14, please import from IPython display
        warnings.simplefilter(action='ignore', category=FutureWarning)
        # /usr/local/lib/python3.10/site-packages/quantstats/stats.py:968: FutureWarning: In a future version of pandas all arguments of DataFrame.pivot will be keyword-only.
        #    returns = returns.pivot('Year', 'Month', 'Returns').fillna(0)
        fig = qs.plots.monthly_returns(
            returns,
            show=False)
    
    return fig


def visualise_returns_distribution(
        returns: pd.Series,
) -> Figure:
    """Breakdown the best day/month/yearly returns

    - `See Quantstats README for more details <https://github.com/ranaroussi/quantstats>`__.

    Example:

    .. code-block:: python

        from tradeexecutor.visual.equity_curve import calculate_equity_curve, calculate_returns
        from tradeexecutor.visual.equity_curve import visualise_returns_distribution

        curve = calculate_equity_curve(state)
        returns = calculate_returns(curve)
        fig = visualise_returns_distribution(returns)
        display(fig)

    :return:
        Matplotlit figure

    """

    qs = import_quantstats_wrapped()
    with warnings.catch_warnings():  #  DeprecationWarning: Importing display from IPython.core.display is deprecated since IPython 7.14, please import from IPython display
        warnings.simplefilter(action='ignore', category=FutureWarning)
        fig = qs.plots.distribution(
            returns,
            show=False)
    return fig


def calculate_investment_flow(
    state: State,
) -> pd.Series:
    """Calculate deposit/redemption i nflows/outflows of a strategy.

    See :ref:`profitability` for more information.

    :return:
        Pandas series (DatetimeIndex by the timestamp when the strategy treasury included the flow event, usd deposits/redemption amount)
    """

    treasury = state.sync.treasury
    balance_updates = treasury.balance_update_refs
    index = [e.strategy_cycle_included_at for e in balance_updates]
    values = [e.usd_value for e in balance_updates]

    if len(index) == 0:
        return pd.Series([], index=pd.to_datetime([]), dtype='float64')

    return pd.Series(values, index)


def calculate_realised_profitability(
    state: State,
) -> pd.Series:
    """Calculate realised profitability of closed trading positions.

    This function returns the :term:`profitability` of individually
    closed trading positions.

    See :ref:`profitability` for more information.

    :return:
        Pandas series (DatetimeIndex, % profit).

        Empty series if there are no trades.
    """
    data = [(p.closed_at, p.get_realised_profit_percent()) for p in state.portfolio.closed_positions.values()]

    if len(data) == 0:
        return pd.Series(dtype='float64')

    # https://stackoverflow.com/a/66772284/315168
    return pd.DataFrame(data).set_index(0)[1]


def calculate_size_relative_realised_trading_returns(
    state: State,
) -> pd.Series:
    """Calculate realised profitability of closed trading positions relative to the portfolio size..

    This function returns the :term:`profitability` of individually
    closed trading positions.

    See :ref:`profitability` for more information.

    :return:
        Pandas series (DatetimeIndex, % profit).

        Empty series if there are no trades.
    """
    positions = [p for p in state.portfolio.closed_positions.values() if p.is_closed()]
    return _calculate_size_relative_trading_returns(positions)


def _calculate_size_relative_trading_returns(positions: list[TradingPosition]):

    # Legacy path
    data = [(p.closed_at, p.get_size_relative_realised_profit_percent()) for p in positions]

    if len(data) == 0:
        return pd.Series(dtype='float64', index=pd.to_datetime([]))

    # https://stackoverflow.com/a/66772284/315168
    return pd.DataFrame(data).set_index(0)[1]


def calculate_compounding_realised_trading_profitability(
    state: State,
    fill_time_gaps=True,
) -> pd.Series:
    """Calculate realised profitability of closed trading positions, with the compounding effect.

    Assume the profits from the previous PnL are used in the next one.

    This function returns the :term:`profitability` of individually
    closed trading positions, relative to the portfolio total equity.

    - See :py:func:`calculate_realised_profitability` for more information

    - See :ref:`profitability` for more information.

    :param fill_time_gaps:
        Insert a faux book keeping entries at star tand end.

        There chart ends at the last profitable trade.
        However, we want to render the chart all the way up to the current date.
        If `True` then insert a booking keeping entry to the last strategy timestamp
        (`State.last_updated_at),
        working around various issues when dealing with this data at the frontend.

        Any fixes are not applied to empty arrays.

    :return:
        Pandas series (DatetimeIndex, cumulative % profit).

        Cumulative profit is 0% if there is no market action.
        Cumulative profit is 1 (100%) if the equity has doubled during the period.

        The last value of the series is the total trading profitability
        of the strategy over its lifetime.
    """
    started_at, last_ts = _get_strategy_time_range(state, fill_time_gaps)
    positions = [p for p in state.portfolio.closed_positions.values() if p.is_closed()]
    return _calculate_compounding_trading_profitability(positions, fill_time_gaps, started_at, last_ts)


def calculate_compounding_unrealised_trading_profitability(
    state_or_portfolio: State | Portfolio,
    freq: str | None="D",
) -> pd.Series:
    """Calculate the current profitability of open and closed trading positions, with the compounding effect.

    :param freq:
        Bin results to this Pandas frequency.

        Default to daily.

    :return:
        Sparse Pandas series of compounded returns.

        Timeline of (timestamp, position profit) for each timestamp when the position was last updated or closed.
    """

    if isinstance(state_or_portfolio, State):
        # State not needed, but wanted to maintain the similar signature check
        portfolio = state_or_portfolio.portfolio
    else:
        # Path from calculate_statistics()
        portfolio = state_or_portfolio

    # Calculate unrealised/realised profit pct for each position
    profit_data = [(p.get_profit_timeline_timestamp(), p.get_size_relative_unrealised_or_realised_profit_percent()) for p in portfolio.get_all_positions()]

    if len(profit_data) == 0:
        return pd.Series([], index=pd.to_datetime([]), dtype='float64')

    profit_data.sort(key=lambda t: t[0])

    index, profit = list(zip(*profit_data))
    returns = pd.Series(data=profit, index=pd.DatetimeIndex(index), dtype="float64")

    if freq:

        # If we haved closed two positions on the same day, asfreq() will fail unless we merge profit values
        def custom_cumprod_resampler(intraday_series):
            match len(intraday_series):
                case 0:
                    return 0
                case 1:
                    return intraday_series.iloc[0]
                case _:
                    # Multiple closed positions within the same day, calculate the daily overall return
                    daily_compounded = intraday_series.add(1).cumprod().sub(1)
                    return daily_compounded.iloc[-1]

        try:
            resampled_returns = returns.resample(freq).agg(custom_cumprod_resampler)
            resampled_compounded = resampled_returns.add(1).cumprod().sub(1)
            resampled_compounded = resampled_compounded.ffill()
        except Exception as e:
            raise RuntimeError(f"Daily binning failed for: {profit_data}") from e
    else:
        resampled_compounded = returns.add(1).cumprod().sub(1)

    return resampled_compounded


def extract_compounding_unrealised_trading_profitability_portfolio_statistics(state: State) -> pd.Series:
    """Get the statistics out from the summary profit data.

    :return:
        Pandas series with profitability % as value and DatetimeIndex as the sampled time
    """
    data = [(stat_entry.calculated_at, stat_entry.unrealised_profitability) for stat_entry in state.stats.portfolio if stat_entry.unrealised_profitability is not None]

    if len(data) == 0:
        return pd.Series([], index=pd.to_datetime([]), dtype='float64')

    # https://stackoverflow.com/a/78156103/315168
    df = pd.DataFrame(data, columns=["timestamp", "profitability"])
    df = df.set_index("timestamp")
    series = df["profitability"]
    series = series.dropna()
    return series


def _calculate_compounding_trading_profitability(
    positions: list[TradingPosition],
    fill_time_gaps: bool,
    started_at: pd.Timestamp = None,
    last_ts: pd.Timestamp = None,
    realised_only=True,
):

    realised_profitability = _calculate_size_relative_trading_returns(positions)

    # https://stackoverflow.com/a/42672553/315168
    compounded = realised_profitability.add(1).cumprod().sub(1)

    if fill_time_gaps and len(compounded) > 0:

        assert started_at
        assert last_ts
        last_value = compounded.iloc[-1]

        # Strategy always starts at zero
        compounded[started_at] = 0

        # Fill from he last sample to current
        if last_ts and len(compounded) > 0 and last_ts > compounded.index[-1]:
            compounded[last_ts] = last_value

        # Because we insert new entries, we need to resort the array
        compounded = compounded.sort_index()

    return compounded


    started_at, last_ts = _get_strategy_time_range(state, fill_time_gaps)
    positions = state.portfolio.get_all_positions()  # TODO: Frozen positions may cause issues
    return _calculate_compounding_trading_profitability(positions, fill_time_gaps, started_at, last_ts)


def _calculate_compounding_realised_trading_profitability(
    positions: list[TradingPosition],
    fill_time_gaps: bool,
    started_at: pd.Timestamp = None,
    last_ts: pd.Timestamp = None,
):
    realised_profitability = _calculate_size_relative_realised_trading_returns(positions)
    # https://stackoverflow.com/a/42672553/315168
    compounded = realised_profitability.add(1).cumprod().sub(1)

    if fill_time_gaps and len(compounded) > 0:

        assert started_at
        assert last_ts
        last_value = compounded.iloc[-1]

        # Strategy always starts at zero
        compounded[started_at] = 0

        # Fill fromt he last sample to current
        if last_ts and len(compounded) > 0 and last_ts > compounded.index[-1]:
            compounded[last_ts] = last_value

        # Because we insert new entries, we need to resort the array
        compounded = compounded.sort_index()

    return compounded


def calculate_long_compounding_realised_trading_profitability(state, fill_time_gaps=True):
    started_at, last_ts = _get_strategy_time_range(state, fill_time_gaps)
    positions = [p for p in state.portfolio.closed_positions.values() if (p.is_closed() and p.is_long())]
    return _calculate_compounding_trading_profitability(positions, fill_time_gaps, started_at, last_ts)


def calculate_short_compounding_realised_trading_profitability(state, fill_time_gaps=True):
    started_at, last_ts = _get_strategy_time_range(state, fill_time_gaps)
    positions = [p for p in state.portfolio.closed_positions.values() if (p.is_closed() and p.is_short())]
    return _calculate_compounding_trading_profitability(positions, fill_time_gaps, started_at, last_ts)
  

def _get_strategy_time_range(state, fill_time_gaps):
    started_at, last_ts = None, None
    if fill_time_gaps:
        started_at, last_ts = state.get_strategy_time_range()
    if not last_ts:
        last_ts = datetime.datetime.utcnow()
    return started_at,last_ts
  

def calculate_deposit_adjusted_returns(
    state: State,
    freq: pd.DateOffset = pd.offsets.Day(),
) -> pd.Series:
    """Calculate daily/monthly/returns on capital.
z
    See :ref:`profitability` for more information

    - This is `Total equity - net deposits`

    - This result is compounding

    - The result is resampled to a timeframe

    :param freq:
        Which sampling frequency we use for the resulting series.

        By default resample the results for daily timeframe.

    :return:
        Pandas series (DatetimeIndex by the the start timestamp fo the frequency, USD amount)
    """
    equity = calculate_equity_curve(state)
    flow = calculate_investment_flow(state)

    # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#resampling
    equity_resampled = equity.resample(freq)
    equity_delta = equity_resampled.last() - equity_resampled.first()

    flow_delta = flow.resample(freq).sum()

    return equity_delta - flow_delta


def calculate_non_cumulative_daily_returns(state: State, freq_base: pd.offsets.DateOffset | None = pd.offsets.Day()) -> pd.Series:
    """Calculates the the non cumulative daily returns for the strategy over time. 

    - Accounts for multiple positions in the same day
    - If no positions/trades are made on a day, it will be filled with 0

    .. note:: 
        Forward fill cannot be used with this method since the stat for each day represents the realised profit for that day only.
        So we fill na values with 0 

    :param state: Strategy state
    :param freq_base: Time frequency to resample to
    :return: Pandas series
    """
    returns = calculate_size_relative_realised_trading_returns(state)
    non_cumulative_daily_returns = returns.add(1).resample(freq_base).prod(min_count=1).sub(1).fillna(0)
    return non_cumulative_daily_returns


def calculate_cumulative_daily_returns(state: State, freq_base: pd.offsets.DateOffset | None = pd.offsets.Day()) -> pd.Series:
    """Calculates the cumulative daily returns for the strategy over time

    - Accounts for multiple positions in the same day
    - If no positions/trades are made on a day, that day will use latest non-zero profit value (forward fill)

    :param state: Strategy state
    :param freq_base: Time frequency to resample to
    :return: Pandas series
    """
    returns = calculate_compounding_realised_trading_profitability(state)
    _returns = returns.copy()
    cumulative_daily_returns = _returns.resample(freq_base).last().ffill()
    return cumulative_daily_returns


def resample_returns(returns: pd.Series, freq: pd.DateOffset | str) -> pd.Series:
    """Resample returns series to a longer time frame.

    - Transform daily returns series to monthly and so on

    - The returns of each period is the cumulative product of the sub-returns

    - Does this with a cumulative product transformation

    Example:

    We have `returns`:

    .. code-block:: text

        2021-06-01 00:00:00    0.000000
        2021-06-01 08:00:00    0.000000
        2021-06-01 16:00:00    0.000000
        2021-06-02 00:00:00    0.000000
        2021-06-02 08:00:00    0.000000
                                 ...
        2024-03-08 08:00:00   -0.002334
        2024-03-08 16:00:00   -0.012170
        2024-03-09 00:00:00   -0.003148
        2024-03-09 08:00:00    0.010400
        2024-03-09 16:00:00   -0.000277

    Make it quarterly:

    .. code-block:: python

        # Transform daily returns to monthly for easier comparison
        freq = QuarterBegin()
        resampled_returns = resample_returns(returns, freq)

    Now it is:

    .. code-block:: text

        2021-06-01    0.483777
        2021-09-01    0.287191
        2021-12-01    0.265714
        2022-03-01   -0.035728
        2022-06-01    0.194215
        2022-09-01    0.059003
        2022-12-01    0.062195
        2023-03-01   -0.091300

    :param returns:
        Hourly, 8h, etc. returns

    :param freq:
        Pandas resample frequency.

        Use "D" for daily.

    :return:
        Returns series where the returns are binned by a new timeframe.
    """
    # https://stackoverflow.com/a/46216956/315168
    assert isinstance(returns, pd.Series)

    # Handle daily specially so that time-to-market etc. analysis
    # will work for weekly data
    if freq == "D":
        period = None
        if len(returns) > 2:
            period = (returns.index[1] - returns.index[0])

        if True or period is None or period <= pd.Timedelta(hours=24):
            # From more frequent to daily
            return (1 + returns).resample(freq).prod() - 1

        match period:
            case pd.Timedelta(days=7):
                # Distribute returns to each week day equally so that our
                # time to market stats looks sane
                forward_filled = returns.resample("D").ffill() / 7
                resampled = (1 + forward_filled).resample(freq).prod() - 1
                return resampled
            case _:
                raise NotImplementedError(f"Unsupported period {period}")
    else:
        # All other cases - hit or miss
        return (1 + returns).resample(freq).prod() - 1

def calculate_rolling_sharpe(
    returns: pd.Series,
    freq: pd.DateOffset | str | None = "D",
    periods=90,  # 90 Days
) -> pd.Series:
    """Calculate rolling Sharpe ration.

    - Declining rolling :term:`sharpe` means that the alpha of the :term:`strategy is decaying <strategy decay>`.

    - `See this QuantStrat post for more information <https://www.quantstart.com/articles/annualised-rolling-sharpe-ratio-in-qstrader/>`__

    - `Alternative example implementation <https://github.com/pranaysjha/rolling-sharpe/blob/main/rolling_sharpe.py>`__

    Explanation how to interpret rolling sharpe from QuantStrat:

        It can be seen that the strategy had a significant upward period in 2013 which gives rise to a high trailing annualised Sharpe of 2.5, exceeding 3.5 by the start of 2014. However the strategy performance remained flat through 2014, which caused a gradual reduction in the annualised rolling Sharpe since the volatility of returns was largely similar. By the start of 2015 the Sharpe was between 0.5 and 1.0, meaning more risk was being taken per unit of return at this stage. By the end of 2015 the Sharpe had risen slightly to around 1.5, largely due to some consistent upward gains in the latter half of 2015.

    Example:

    .. code-block:: python

        import plotly.express as px

        from tradeexecutor.visual.equity_curve import calculate_rolling_sharpe

        rolling_sharpe = calculate_rolling_sharpe(
            returns,
            freq="D",
            periods=180,
        )

        fig = px.line(rolling_sharpe, title='Strategy rolling Sharpe (6 months)')
        fig.update_layout(showlegend=False)
        fig.update_yaxes(title="Sharpe")
        fig.update_xaxes(title="Time")
        fig.show()

    :param returns:
        Returns series with rolling sharpe

    :param freq:
        Returns binning frequency for Sharpe calculations

    :param periods:
        How many periods of data we sample for rolling sharpe.
    """

    if freq is not None:
        resampled_returns = resample_returns(returns, freq)
    else:
        resampled_returns = returns

    rolling = resampled_returns.rolling(window=periods)
    rolling_sharpe = np.sqrt(periods) * (
        rolling.mean() / rolling.std()
    )

    # Remove NA entries at the beginning of the series
    return rolling_sharpe.dropna()


def calculate_share_price(state: State, initial_share_price=1.0) -> pd.DataFrame:
    """Calculate share price of the strategy.

    Share price is the value of a single share in the strategy.

    Example:

        from tradeexecutor.visual.equity_curve import calculate_share_price

        share_price_df = calculate_share_price(state)
        print(share_price_df)


    :return:
        DataFrame with returns, share price USD, NAV.

        For the life time of the strategy.

        - share_price_usd: price of a single share in USD
        - returns: cumulative returns
    """

    profit = [
        {"calculated_at": s.calculated_at, "share_price_usd": s.share_price_usd, "nav": s.net_asset_value} for s in state.stats.portfolio
    ]
    df = pd.DataFrame(profit)
    if len(df) == 0:
        return pd.DataFrame([], index=pd.to_datetime([]))

    df = df.set_index("calculated_at").sort_index()  # Make sure index comes through in the correct order
    df = df.dropna()
    df["returns"] = df["share_price_usd"] - initial_share_price
    return df
