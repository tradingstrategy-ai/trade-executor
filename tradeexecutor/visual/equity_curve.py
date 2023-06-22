"""Equity curve based statistics and visualisations.

For more information see the narrative documentation on :ref:`profitability`.
"""
import warnings
from typing import List

import pandas as pd
from matplotlib.figure import Figure

from tradeexecutor.state.state import State
from tradeexecutor.state.statistics import Statistics, PortfolioStatistics



def calculate_equity_curve(
        state: State,
        attribute_name="total_equity",
) -> pd.Series:
    """Calculate equity curve for the portfolio.

    Translate the portfolio internal :py:attr:`Statistics.portfolio`
    to :py:class:`pd.Series` that allows easy equity curve calculations.

    This reads :py:class:`tradeexecutor.state.stats.PortfolioStatistics`

    :param attribute_name:
        Calculate equity curve based on this attribute of :py:class:`

    :return:
        Pandas series (timestamp, equity value).

        Indxe is DatetimeIndex.

    """

    stats: Statistics = state.stats

    portfolio_stats: List[PortfolioStatistics] = stats.portfolio

    data = [(s.calculated_at, getattr(s, attribute_name)) for s in portfolio_stats]

    if len(data) == 0:
        return pd.Series([], index=pd.to_datetime([]))

    # https://stackoverflow.com/a/66772284/315168
    return pd.DataFrame(data).set_index(0)[1]


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
    return equity_curve.pct_change().fillna(0.0)


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


def calculate_aggregate_returns(equity_curve: pd.Series, freq: str = "BM") -> pd.Series:
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

    # Each equity curve sample is the last day of the period
    # https://stackoverflow.com/a/14039589/315168
    sampled = equity_curve.asfreq(freq, method='ffill')
    return sampled.pct_change()


def get_daily_returns(state: State) -> (pd.Series | None):
    """Used for advanced statistics

    :returns:
        If valid state provided, returns are returned as calendar day (D) frequency, else None"""
    
    equity_curve = calculate_equity_curve(state)
    return calculate_aggregate_returns(equity_curve, freq="D")


def visualise_equity_curve(
        returns: pd.Series,
        title="Equity curve",
        line_width=1.5,
) -> Figure:
    """Draw equity curve, drawdown and daily returns using quantstats.

    `See Quantstats README for more details <https://github.com/ranaroussi/quantstats>`__.

    Example:

    .. code-block:: python

        from tradeexecutor.visual.equity_curve import calculate_equity_curve, calculate_returns
        from tradeexecutor.visual.equity_curve import visualise_equity_performance

        curve = calculate_equity_curve(state)
        returns = calculate_returns(curve)
        fig = visualise_equity_performance(returns)
        display(fig)

    :return:
        Matplotlit figure

    """
    import quantstats as qs  # Optional dependency
    fig = qs.plots.snapshot(
        returns,
        title=title,
        lw=line_width,
        show=False)
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

    with warnings.catch_warnings():
        import quantstats as qs  # Optional dependency
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
    import quantstats as qs  # Optional dependency
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
        return pd.Series([], index=pd.to_datetime([]))

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
    data = [(p.closed_at, p.get_realised_profit_percent()) for p in state.portfolio.closed_positions.values() if p.is_closed()]

    if len(data) == 0:
        return pd.Series()

    # https://stackoverflow.com/a/66772284/315168
    return pd.DataFrame(data).set_index(0)[1]


def calculate_compounding_realised_profitability(
    state: State,
) -> pd.Series:
    """Calculate realised profitability of closed trading positions, with the compounding effect.

    Assume the profits from the previous PnL are used in the next one.

    This function returns the :term:`profitability` of individually
    closed trading positions.

    - See :py:func:`calculate_realised_profitability` for more information

    - See :ref:`profitability` for more information.

    :return:
        Pandas series (DatetimeIndex, cumulative % profit).

        Cumulative profit is 0% if there is no market action.
        Cumulative profit is 1 (100%) if the equity has doubled during the period.

        The last value of the series is the total trading profitability
        of the strategy over its lifetime.
    """
    realised_profitability = calculate_realised_profitability(state)
    # https://stackoverflow.com/a/42672553/315168
    compounded = realised_profitability.add(1).cumprod().sub(1)
    return compounded


def calculate_deposit_adjusted_returns(
    state: State,
    freq: pd.DateOffset = pd.offsets.MonthBegin(),
) -> pd.Series:
    """Calculate daily/monthly/returns on capital.
z
    See :ref:`profitability` for more information

    - This is `Total equity - net deposits`

    - This result is compounding

    - The result is resampled to a timeframe

    :param freq:
        Which sampling frequency we use for the resulting series.

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
