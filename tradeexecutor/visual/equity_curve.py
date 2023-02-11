"""Equity curve based statistics and visualisations."""
from typing import Iterable, List

import pandas as pd

from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.statistics import Statistics, PortfolioStatistics


def calculate_aggregate_returns(portfolio: Portfolio, period: pd.Timedelta):
    """Calculate strategy aggregatd results over different timespans.

    Good to calcualte

    - Monthly returns

    - Quaterly returns
    """


def calculate_equity_curve(
        portfolio: Portfolio,
        attribute_name="total_equity",
) -> pd.Series:
    """Calculate equity curve for the portfolio.

    Translate the portfolio internal :py:attr:`Statistics.portfolio`
    to :py:class:`pd.Series` that allows easy equity curve calculations.

    :param attribute_name:
        Calculate equity curve based on this attribute of :py:class:`

    :return:
        Pandas series (timestamp, equity value)
    """

    stats: Statistics = portfolio.stats

    portfolio_stats: List[PortfolioStatistics] = stats.portfolio
    index = [stat.calculated_at for stat in portfolio_stats]

    assert len(index) > 0, "Cannot calculate equity curve because there are no portfolio.stats.portfolio entries"

    values = [getattr(stat, attribute_name) for stat in portfolio_stats]
    return pd.Series(values, index)




def foobar():
    equity_df["returns"] = equity_df["Equity"].pct_change().fillna(0.0)

    # Cummulative Returns
    equity_df["cum_returns"] = np.exp(np.log(1 + equity_df["returns"]).cumsum())


