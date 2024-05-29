"""Various statistics calculated across portfolios and positions.

Statistics are calculated/refreshed on the server-side and exported as a part of the state.
This way the clients (JavaScript) do not need to reconstruct this information.

Any statistics are optional: they are not needed to make any state transitions, they are
purely there for profit and loss calculations.
"""
import datetime
from collections import defaultdict
from dataclasses import field, dataclass
from math import isnan
from typing import Dict, List, Optional, Tuple

import pandas as pd
from dataclasses_json import dataclass_json
from pandas import DatetimeIndex

from tradeexecutor.state.types import Percent
from tradingstrategy.types import USDollarAmount
from tradeexecutor.analysis.trade_analyser import TradeSummary


@dataclass_json
@dataclass
class PositionStatistics:
    """Time-series of statistics calculated for each open position.

    Position statistics are recalculated at the same time positions are revalued.
    The time-series of these statistics are stored as a part of the state,
    allowing one to plot the position performance over time.
    """

    #: Real-time clock when these stats were calculated
    #:
    calculated_at: datetime.datetime

    #: When this position was revalued last time.
    #: Should not be far off from `calculated_at`
    #: because we should revalue positions always before calculating their stats.
    last_valuation_at: datetime.datetime

    #: Profitability %
    #:
    #: Unrealised profitability
    #:
    profitability: float

    #: How much profit we made so far
    profit_usd: USDollarAmount

    #: The current number of owned units
    quantity: float

    #: The current position size dollars
    #:
    value: USDollarAmount

    def __post_init__(self):
        assert isinstance(self.calculated_at, datetime.datetime)
        assert isinstance(self.last_valuation_at, datetime.datetime)
        assert not isnan(self.profitability)


@dataclass_json
@dataclass
class FinalPositionStatistics:
    """When position is closed, its final statistics are calculated.

    These statistics contain fields that is not present in open positions.
    """

    #: Real-time clock when these stats were calculated
    calculated_at: datetime.datetime

    #: How many trades we have made
    trade_count: int

    #: How much was the first purchase
    value_at_open: USDollarAmount

    #: How much was we held at the maximum point of time
    value_at_max: USDollarAmount


@dataclass_json
@dataclass
class PortfolioStatistics:
    """Portfolio statistics for each timepoint.

    Updated with regular ticks for a live strategy.

    If backtesting, only calculated_at and total_equity are necessary for later visualisations and metrics
    If livetrading, then all attributes should be specified so that for displaying updated metrics after each trade

    See :py:attr:`Statistics.portfolio` for reading.


    """

    #: Real-time clock when these stats were calculated
    calculated_at: datetime.datetime

    # Deprecated: Use net_asset_value
    total_equity: USDollarAmount

    #: How much was TVL equivalent
    net_asset_value: Optional[USDollarAmount] = None

    #: The unrealised all-time profitability of this strategy at this point of time
    #:
    #: See :py:func:`tradeexecutor.visualisation.equity_curve.calculate_compounding_unrealised_trading_profitability`
    #:
    #: Set to 0 if cannot be calculated yet.
    #:
    unrealised_profitability: Optional[Percent] = None
    
    free_cash: Optional[USDollarAmount] = None
    open_position_count: Optional[int] = None
    open_position_equity: Optional[USDollarAmount] = None
    frozen_position_count: Optional[int] = None
    frozen_position_equity: Optional[USDollarAmount] = None
    closed_position_count: Optional[int] = None
    unrealised_profit_usd: Optional[USDollarAmount] = None

    first_trade_at: Optional[datetime.datetime] = None
    last_trade_at: Optional[datetime.datetime] = None

    realised_profit_usd: Optional[USDollarAmount] = 0
    summary: Optional[TradeSummary] = None

    def get_value(self) -> USDollarAmount:
        if self.net_asset_value is not None:
            return self.net_asset_value

        # Legacy
        return self.total_equity

    def __post_init__(self):
        pass
        # TODO: Cannot do this yet because of legacy data
        # assert (self.total_equity or self.net_asset_value), "PortfolioStatistics: could not calculate value for the portfolio"

        # Safety checks for the bad data
        if self.unrealised_profitability is not None:
            assert not pd.isna(self.unrealised_profitability)


@dataclass_json
@dataclass
class Statistics:
    """Statistics for a trade execution state.

    We calculate various statistics on the server-side and make them part of the state,
    so that JS clients can easily display this information.

    Statistics are collected over time and more often than trading ticks.
    We store historical statistics for each position as the part of the state.
    """

    #: Per portfolio statistics.
    #:
    #: Contains list of statistics for the portfolio over time.
    #: The first timestamp is the first entry in the list.
    #: Note that now we have only one portfolio per state.
    #:
    #: This is calculated in :py:func:`tradeexecutor.statistics.core.calculate_statistics`.
    #:
    portfolio: List[PortfolioStatistics] = field(default_factory=list)

    #: Per position statistics.
    #: We look them up by position id.
    #: Each position contains list of statistics for the position over time.
    #: The first timestamp is the first entry in the list.
    positions: Dict[int, List[PositionStatistics]] = field(default_factory=lambda: defaultdict(list))

    #: Per position statistics for closed positions.
    closed_positions: Dict[int, FinalPositionStatistics] = field(default_factory=dict)
    
    #: Latest long short metrics
    long_short_metrics_latest: Optional[str] = None

    def get_latest_portfolio_stats(self) -> PortfolioStatistics:
        return self.portfolio[-1]
    
    def get_earliest_portfolio_stats(self) -> PortfolioStatistics:
        return self.portfolio[0]
    
    def get_equity_series(self) -> pd.Series:
        """Get the time series of portfolio equity.

        :return: Pandas Series with timestamps as index and equity as values.
        """
        return pd.Series(
            [ps.total_equity for ps in self.portfolio],
            index=[ps.calculated_at for ps in self.portfolio],
            name="equity"
        )

    def get_latest_position_stats(self, position_id: int) -> PositionStatistics:
        return self.positions[position_id][-1]

    def add_positions_stats(self, position_id: int, p_stats: PositionStatistics):
        """Add a new sample to position stats.

        We cannot use defaultdict() here because we lose defaultdict instance on state serialization.
        """
        assert isinstance(position_id, int)
        assert isinstance(p_stats, PositionStatistics)
        stat_list = self.positions.get(position_id, [])
        stat_list.append(p_stats)
        self.positions[position_id] = stat_list

    def get_portfolio_statistics_dataframe(
            self,
            attr_name: str,
            resampling_time: str="D",
            resampling_method: str="max") -> pd.Series:
        """Get any of position statistcs value as a columnar data.

        Get the daily performance of the portfolio.

        Example:

        .. code-block:: python

            # Create time series of portfolio "total_equity" over its lifetime
            s = stats.get_portfolio_statistics_dataframe("total_equity")

        :param attr_name:
            Which variable we are interested in.
            E.g. `total_equity`.

        :param resampling_time:
            See http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#resampling

        :param resamping_method:
            See http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#resampling

        :return:
            DataFrame for the value with time as index.
        """

        assert len(self.portfolio) > 0, f"Statistics did not have any calculations for positions: {self.portfolio}"

        # https://stackoverflow.com/questions/40815238/convert-dataframe-index-to-datetime
        s = pd.Series(
            [getattr(ps, attr_name) for ps in self.portfolio],
            index=DatetimeIndex([ps.calculated_at for ps in self.portfolio]),
        )

        # Convert data to daily if we have to
        assert resampling_method == "max", f"Unsupported resamping method {resampling_method}"
        return s.resample(resampling_time).max()
    
    def get_naive_rolling_pnl_pct(self) -> float:
        """Get the naive rolling PnL percentage.

        Used to display the PnL on the backtest progress bar.

        :return:
            Profitability -1...inf
        """

        if len(self.portfolio) == 0:
            return 0.0

        return (self.portfolio[-1].get_value() - self.portfolio[0].get_value()) / self.portfolio[0].get_value()


def calculate_naive_profitability(
        total_equity_series: pd.Series,
        look_back: Optional[pd.Timedelta] = None,
        start_at: Optional[pd.Timestamp] = None,
        end_at: Optional[pd.Timestamp] = None) -> Tuple[Optional[float], Optional[pd.Timedelta]]:
    """Calculate the profitability as value at end - value at start.

    .. warning::

        This method assumes there are no deposits or redemptions.
        See :py:mod:`tradeexecutor.visual.equity_curve` for more advanced
        profit calculations.

    :param total_equity:
        As received from get_portfolio_statistics_dataframe()

    :return:
        Tuple (Profitability as %, duration of the sample period).
        (None, None) if we cannot calculate anything yet.
    """

    if len(total_equity_series) < 2:
        return None, None

    if look_back:
        assert not(start_at or end_at), "Give either look_back or range"

        end_at = total_equity_series.index[-1]
        start_at = end_at - look_back

        # We cannot look back data we do not have
        start_at = max(total_equity_series.index[0], start_at)
    else:
        assert start_at and end_at, "Give either look_back or range"


    # https://stackoverflow.com/a/42266376/315168
    start_val_idx = total_equity_series.index.get_indexer([start_at], method="nearest")
    end_val_idx = total_equity_series.index.get_indexer([end_at], method="nearest")

    start_val = float(total_equity_series.iloc[start_val_idx])
    end_val = float(total_equity_series.iloc[end_val_idx])

    return (end_val - start_val) / (start_val), end_at - start_at
