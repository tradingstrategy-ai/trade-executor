"""Analyze the trade performance of algorithm.

Calculate success/fail rate of trades and plot success distribution.

Example analysis include:

- Table: Summary of all trades

- Graph: Trade won/lost distribution

- Timeline: Analysis of each individual trades made

.. note ::

    A lot of this code has been lifted off from trading-strategy package
    where it had to deal with different trading frameworks.
    It could be simplified greatly now.

"""

import datetime
import warnings
import enum
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Iterable, Optional, Tuple, Callable, Set

import numpy as np
import pandas as pd
from IPython.core.display_functions import display
from IPython.display import HTML
from dataclasses_json import dataclass_json, config
from statistics import median

from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.trade import TradeExecution, TradeType
from tradeexecutor.state.types import USDollarPrice, Percent
from tradeexecutor.utils.format import calculate_percentage
from tradeexecutor.utils.timestamp import json_encode_timedelta, json_decode_timedelta
from tradeexecutor.utils.summarydataframe import format_value as format_value_for_summary_table

from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.exchange import Exchange
from tradingstrategy.pair import PandasPairUniverse
from tradingstrategy.types import PrimaryKey, USDollarAmount
from tradingstrategy.utils.format import format_value, format_price, format_duration_days_hours_mins, \
    format_percent_2_decimals
from tradingstrategy.utils.jupyter import make_clickable
from tradeexecutor.utils.summarydataframe import as_dollar, as_integer, create_summary_table, as_percent, as_duration, as_bars, as_decimal

try:

    with warnings.catch_warnings():  #  DeprecationWarning: Importing display from IPython.core.display is deprecated since IPython 7.14, please import from IPython display
        warnings.simplefilter(action='ignore', category=FutureWarning)  # yfinance: The default dtype for empty Series will be 'object' instead of 'float64' in a future version. Specify a dtype explicitly to silence this warning.
        import quantstats as qs

    HAS_QUANTSTATS = True
except Exception:
    HAS_QUANTSTATS = False

logger = logging.getLogger(__name__)


@dataclass_json
@dataclass(slots=True)
class TradeSummary:
    """Some generic statistics over all the trades

    .. warning:: 
        Be very careful when editing or removing existing fields here as this can intefere with legacy data testing

    TDOO: Cleam this up
    """
    won: int
    lost: int
    zero_loss: int
    stop_losses: int
    undecided: int
    realised_profit: USDollarAmount

    #: Value at the open positinos at the end
    open_value: USDollarAmount

    #: Cash at hand at the end
    uninvested_cash: USDollarAmount

    initial_cash: USDollarAmount
    extra_return: USDollarAmount
    duration: Optional[datetime.timedelta] = field(metadata=config(
        encoder=json_encode_timedelta,
        decoder=json_decode_timedelta,
    ))

    average_winning_trade_profit_pc: Optional[float]  # position
    average_losing_trade_loss_pc: Optional[float]

    biggest_winning_trade_pc: Optional[float]
    biggest_losing_trade_pc: Optional[float]

    average_duration_of_winning_trades: datetime.timedelta = field(metadata=config(
        encoder=json_encode_timedelta,
        decoder=json_decode_timedelta,
    ))  # position
    average_duration_of_losing_trades: datetime.timedelta = field(metadata=config(
        encoder=json_encode_timedelta,
        decoder=json_decode_timedelta,
    ))  # position
    time_bucket: Optional[TimeBucket] = None

    # these stats calculate in post-init, so init=False
    total_positions: int = field(init=False)
    win_percent: float = field(init=False)
    lost_percent: float = field(init=False)
    delta_neutral_percent: float = field(init=False)
    return_percent: float = field(init=False)
    annualised_return_percent: float = field(init=False)
    all_stop_loss_percent: float = field(init=False)
    # (total stop losses)/(lost positions)
    # can be > 1 if there are more stop losses than lost positions
    # TODO this is a confusing metric, more intuitive is (losing stop losses)/(total_stop_losses)
    # and (winning stop losses)/(total_stop_losses)
    lost_stop_loss_percent: float = field(init=False)

    all_take_profit_percent: float = field(init=False)
    # (take profits)/(won positions)
    # can be > 1 if there are more take profits than won positions
    # same confusion as lost_stop_loss_percent
    # TODO add (winning take profits)/(total_take_profits)
    # and (losing take profits)/(total_take_profits)
    won_take_profit_percent: float = field(init=False)

    average_net_profit: USDollarAmount = field(init=False)
    end_value: USDollarAmount = field(init=False)

    total_trades: Optional[int] = None
    average_trade: Optional[float] = None  # position
    median_trade: Optional[float] = None  # position
    max_pos_cons: Optional[int] = None
    max_neg_cons: Optional[int] = None
    max_delta_neutral_cons: Optional[int] = None
    max_pullback: Optional[float] = None
    max_loss_risk: Optional[float] = None
    max_realised_loss: Optional[float] = None
    avg_realised_risk: Optional[Percent] = None

    average_delta_neutral_profit_pc: Optional[float] = None
    biggest_delta_neutral_pc: Optional[float] = None
    delta_neutral: int = field(default=0)

    take_profits: int = field(default=0)

    trade_volume: USDollarAmount = field(default=0.0)

    lp_fees_paid: Optional[USDollarPrice] = 0
    lp_fees_average_pc: Optional[USDollarPrice] = 0

    #: advanced users can use this property instead of the
    #: provided quantstats helper methods
    daily_returns: Optional[pd.Series] = None
    compounding_returns: Optional[pd.Series] = None

    winning_stop_losses: Optional[int] = 0
    losing_stop_losses: Optional[int] = 0

    winning_take_profits: Optional[int] = 0
    losing_take_profits: Optional[int] = 0

    winning_stop_losses_percent: Optional[float] = field(init=False)
    losing_stop_losses_percent: Optional[float] = field(init=False)

    winning_take_profits_percent: Optional[float] = field(init=False)
    losing_take_profits_percent: Optional[float] = field(init=False)

    median_win: Optional[float] = None
    median_loss: Optional[float] = None
    median_delta_neutral: Optional[float] = None

    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    profit_factor: Optional[float] = None
    max_drawdown: Optional[float] = None
    max_runup: Optional[float] = None

    average_duration_of_delta_neutral_positions: Optional[datetime.timedelta] = None
    average_duration_of_zero_loss_trades: Optional[datetime.timedelta] = None
    average_duration_of_all_trades: Optional[datetime.timedelta] = None

    #: Profit in open positions at the end
    unrealised_profit: Optional[USDollarAmount] = None

    #: Interest stats
    average_interest_paid_usd: Optional[USDollarPrice] = None
    total_interest_paid_usd: Optional[USDollarPrice] = None
    median_interest_paid_usd: Optional[USDollarPrice] = None
    max_interest_paid_usd: Optional[USDollarPrice] = None
    min_interest_paid_usd: Optional[USDollarPrice] = None
    total_claimed_interest: Optional[USDollarPrice] = None
    # average_claimed_interest: Optional[USDollarPrice] = None

    average_duration_between_position_openings: Optional[datetime.timedelta] = None
    average_position_frequency: Optional[datetime.timedelta] = None

    # Backwards compatiblity only.
    #
    # TODO: Remove these in ~3 months.
    #
    average_duration_between_positions: int = 0
    average_duration_between_postions: int = 0

    # Time in market
    # Doesn't include any open positions
    # Includes credit supply (delta neutral) positions
    time_in_market: Optional[datetime.timedelta] = None

    # Doens't include any open positions
    # Doesn't include credit supply (delta neutral) positions
    time_in_market_volatile: Optional[datetime.timedelta] = None

    def __post_init__(self):
        self.total_positions = self.won + self.lost + self.zero_loss + self.delta_neutral
        self.win_percent = calculate_percentage(self.won, self.total_positions)
        self.lost_percent = calculate_percentage(self.lost, self.total_positions)
        self.delta_neutral_percent = calculate_percentage(self.delta_neutral, self.total_positions)
        self.all_stop_loss_percent = calculate_percentage(self.stop_losses, self.total_positions)
        self.all_take_profit_percent = calculate_percentage(self.take_profits, self.total_positions)
        self.lost_stop_loss_percent = calculate_percentage(self.stop_losses, self.lost)
        self.won_take_profit_percent = calculate_percentage(self.take_profits, self.won)
        self.average_net_profit = calculate_percentage(self.realised_profit, self.total_positions)
        self.end_value = self.open_value + self.uninvested_cash
        initial_cash = self.initial_cash or 0
        self.return_percent = calculate_percentage(self.end_value - initial_cash, initial_cash)
        self.winning_stop_losses_percent = calculate_percentage(self.winning_stop_losses, self.stop_losses)
        self.losing_stop_losses_percent = calculate_percentage(self.losing_stop_losses, self.stop_losses)

        self.winning_take_profits_percent = calculate_percentage(self.winning_take_profits, self.take_profits)
        self.losing_take_profits_percent = calculate_percentage(self.losing_take_profits, self.take_profits)

        if self.duration is not None:
            days_as_decimal = self.duration.total_seconds() / (60 * 60 * 24)
            self.average_position_frequency = self.total_positions / days_as_decimal if days_as_decimal else 0
            self.annualised_return_percent = calculate_annualised_return(self.return_percent, self.duration)
        else:
            self.average_position_frequency = 0
            self.annualised_return_percent = 0

    def get_trading_core_metrics(self) -> Dict[str, str]:
        """Get metrics displayed on a equity curve benchmark tooltip.

        See :py:func:``tradeexecutor.analysis.grid_search._get_hover_template`.

        :return:
            Preformatted dictionary
        """
        return {
            "Positions taken": str(self.total_positions),
            "Trades taken": str(self.total_trades),
            "Positions won": str(as_percent(self.win_percent)),
            "Return all time": str(as_percent(self.return_percent)),
            "Return annualised": str(as_percent(self.annualised_return_percent)),
            "Median win": str(as_percent(self.median_win)),
            "Most consequence wins": str(self.max_pos_cons),
            "Most consequence losses": str(self.max_neg_cons),
        }

    def to_dataframe(self, format_headings=True) -> pd.DataFrame:
        """Creates a human-readable Pandas dataframe summary table from the object."""

        human_data = {
            "Trading period length": as_duration(self.duration),
            "Return %": as_percent(self.return_percent),
            "Annualised return %": as_percent(self.annualised_return_percent),
            "Cash at start": as_dollar(self.initial_cash),
            "Value at end": as_dollar(self.end_value),
            "Time in market": as_percent(self.time_in_market),
            "Time in market volatile": as_percent(self.time_in_market_volatile),
            "Trade volume": as_dollar(self.trade_volume),
            "Position win percent": as_percent(self.win_percent),
            "Total positions": as_integer(self.total_positions),
            "Won positions": as_integer(self.won),
            "Lost positions": as_integer(self.lost),
            "Stop losses triggered": as_integer(self.stop_losses),
            "Stop loss % of all": as_percent(self.all_stop_loss_percent),
            # "Stop loss % of lost": as_percent(self.lost_stop_loss_percent),  # confusing metric
            "Winning stop losses": as_integer(self.winning_stop_losses),
            "Winning stop losses percent": as_percent(self.winning_stop_losses_percent),
            "Losing stop losses": as_integer(self.losing_stop_losses),
            "Losing stop losses percent": as_percent(self.losing_stop_losses_percent),
            "Take profits triggered": as_integer(self.take_profits),
            "Take profit % of all": as_percent(self.all_take_profit_percent),
            "Take profit % of won": as_percent(self.won_take_profit_percent),
            "Zero profit positions": as_integer(self.zero_loss),
            "Positions open at the end": as_integer(self.undecided),
            "Realised profit and loss": as_dollar(self.realised_profit),
            "Unrealised profit and loss": as_dollar(self.unrealised_profit),
            "Portfolio unrealised value": as_dollar(self.open_value),
            "Extra returns on lending pool interest": as_dollar(self.extra_return),
            "Cash left at the end": as_dollar(self.uninvested_cash),
            "Average winning position profit %": as_percent(self.average_winning_trade_profit_pc),
            "Average losing position loss %": as_percent(self.average_losing_trade_loss_pc),
            "Biggest winning position %": as_percent(self.biggest_winning_trade_pc),
            "Biggest losing position %": as_percent(self.biggest_losing_trade_pc),
            "Average duration of winning positions": self.format_duration(self.average_duration_of_winning_trades),
            "Average duration of losing positions": self.format_duration(self.average_duration_of_losing_trades),
        }

        if self.time_bucket:
            human_data.update({
                "Average bars of winning positions": self.format_bars(self.average_duration_of_winning_trades),
                "Average bars of losing positions": self.format_bars(self.average_duration_of_losing_trades),
            })

        human_data.update({
            "Average duration between position openings": self.format_duration(self.average_duration_between_position_openings),
            "Average positions per day": as_decimal(self.average_position_frequency),
            "Average interest paid": as_dollar(self.average_interest_paid_usd),
            "Median interest paid": as_dollar(self.median_interest_paid_usd),
            "Total interest paid": as_dollar(self.total_interest_paid_usd),
            "LP fees paid": as_dollar(self.lp_fees_paid),
            "LP fees paid % of volume": as_percent(self.lp_fees_average_pc),
        })

        def add_prop(value, key: str, formatter: Callable):
            human_data[key] = (
                formatter(value)
                if value is not None
                else formatter(0)
            )

        add_prop(self.average_trade, 'Average position', as_percent)
        add_prop(self.median_trade, 'Median position', as_percent)
        add_prop(self.max_pos_cons, 'Most consecutive wins', as_integer)
        add_prop(self.max_neg_cons, 'Most consecutive losses', as_integer)
        add_prop(self.max_realised_loss, 'Biggest realised risk', as_percent)
        add_prop(self.avg_realised_risk, 'Avg realised risk', as_percent)
        add_prop(self.max_pullback, 'Max pullback of total capital', as_percent)
        add_prop(self.max_loss_risk, 'Max loss risk at opening of position', as_percent)

        if self.daily_returns is not None:
            add_prop(self.max_drawdown, "Max drawdown", as_percent)

        df = create_summary_table(human_data)

        if format_headings:
            return self.format_summary_dataframe(df)

        return df

    @staticmethod
    def help_links() -> dict[str, str]:
        return {
            "Trading period length": None,
            "Return %": "https://tradingstrategy.ai/glossary/aggregate-return",
            "Annualised return %": None,
            "Cash at start": None,
            "Value at end": None,
            "Time in market": None,
            "Time in market volatile": None,
            "Trade volume": "https://tradingstrategy.ai/glossary/volume",
            "Position win percent": "https://tradingstrategy.ai/glossary/position",
            "Total positions": "https://tradingstrategy.ai/glossary/position",
            "Won positions": "https://tradingstrategy.ai/glossary/position",
            "Lost positions": "https://tradingstrategy.ai/glossary/position",
            "Stop losses triggered": "https://tradingstrategy.ai/glossary/stop-loss",
            "Stop loss % of all": "https://tradingstrategy.ai/glossary/stop-loss",
            "Stop loss % of lost": "https://tradingstrategy.ai/glossary/stop-loss",
            "Winning stop losses": "https://tradingstrategy.ai/glossary/stop-loss",
            "Winning stop losses percent": "https://tradingstrategy.ai/glossary/stop-loss",
            "Losing stop losses": "https://tradingstrategy.ai/glossary/stop-loss",
            "Losing stop losses percent": "https://tradingstrategy.ai/glossary/stop-loss",
            "Take profits triggered": "https://tradingstrategy.ai/glossary/take-profit",
            "Take profit % of all": "https://tradingstrategy.ai/glossary/take-profit",
            "Take profit % of won": "https://tradingstrategy.ai/glossary/take-profit",
            "Zero profit positions": "https://tradingstrategy.ai/glossary/position",
            "Positions open at the end": "https://tradingstrategy.ai/glossary/open-position",
            "Realised profit and loss": "https://tradingstrategy.ai/glossary/realised-profit-and-loss",
            "Unrealised profit and loss": "https://tradingstrategy.ai/glossary/unrealised-profit-and-loss",
            "Portfolio unrealised value": "https://tradingstrategy.ai/glossary/unrealised-profit-and-loss",
            "Extra returns on lending pool interest": "https://tradingstrategy.ai/glossary/lending-pool",
            "Cash left at the end": None,
            "Average winning position profit %": None,
            "Average losing position loss %": None,
            "Biggest winning position %": None,
            "Biggest losing position %": None,
            "Average duration of winning positions": None,
            "Average duration of losing positions": None,
            "Average duration between position openings": None,
            "Average positions per day": None,
            "Average interest paid": None,
            "Median interest paid": None,
            "Total interest paid": None, 
            "Average bars of winning positions": None,
            "Average bars of losing positions": None,
            "LP fees paid": "https://tradingstrategy.ai/glossary/liquidity-provider",
            "LP fees paid % of volume": "https://tradingstrategy.ai/glossary/liquidity-provider",
            "Average position": "https://tradingstrategy.ai/glossary/mean",
            "Median position": "https://tradingstrategy.ai/glossary/median",
            "Most consecutive wins": None,
            "Most consecutive losses": None,
            "Biggest realised risk": "https://tradingstrategy.ai/glossary/realised-risk",
            "Avg realised risk": "https://tradingstrategy.ai/glossary/realised-risk",
            "Max pullback of total capital": "https://tradingstrategy.ai/glossary/maximum-pullback",
            "Max loss risk at opening of position": None,
            "Max drawdown": "https://tradingstrategy.ai/glossary/maximum-drawdown",
        }

    @staticmethod
    def format_summary_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Format the summary dataframe for display in Jupyter notebook with clickable links.
        
        :param df:
            The dataframe to format.

        :return:
            Formatted dataframe with clickable links.
        """
        help_links = TradeSummary.help_links()
        links = []
        for h in df.index:
            links.append(help_links.get(h, None))

        df.index = [make_clickable(h, url) if url else h for h, url in zip(df.index, links)]

        return HTML(df.to_html(escape=False))

    def display(self):
        """Create human readable summary tables and display them in IPython notebook."""

        assert self.daily_returns is not None, "No daily returns data available. Remember to add state argument to calculate_summary_statistics() i.e. summary.calculate_summary_statistics(time_bucket, state). \n Otherwise you can display the old summary table with display(summary.to_dataframe())"

        assert self.time_bucket is not None, "Please provide time bucket argument to calculate_summary_statistics() i.e. summary.calculate_summary_statistics(time_bucket, state). \n Otherwise you can display the old summary table with display(summary.to_dataframe())"

        data1 = {
            "Annualised return %": as_percent(self.annualised_return_percent),
            "Lifetime return %": as_percent(self.return_percent),
            "Realised PnL": as_dollar(self.realised_profit),
            "Unrealised PnL": as_dollar(self.unrealised_profit) if self.unrealised_profit else as_dollar(0),
            "Trade period": as_duration(self.duration),
            "Time in market volatile": as_percent(self.time_in_market_volatile),
            "Total interest earned": as_dollar(self.total_claimed_interest),
            "Total funding cost": as_dollar(self.total_interest_paid_usd),
        }

        df1 = create_summary_table(data1, "", "Returns")

        data2 = {
            "Total assets": as_dollar(self.end_value),
            "Cash left": as_dollar(self.uninvested_cash),
            "Open position value": as_dollar(self.open_value),
            "Open positions": as_integer(self.undecided),
        }

        df2 = create_summary_table(data2, "", "Holdings")

        data3 = {
            'Number of positions': [
                as_integer(self.won),
                as_integer(self.lost),
                as_integer(self.delta_neutral),
                as_integer(self.total_positions)
            ],
            '% of total': [
                as_percent(self.win_percent),
                as_percent(self.lost_percent),
                as_percent(self.delta_neutral_percent),
                as_percent((self.won + self.lost + self.delta_neutral) / self.total_positions) if self.total_positions else as_percent(0)
            ],
            'Average PnL %': [
                as_percent(self.average_winning_trade_profit_pc),
                as_percent(self.average_losing_trade_loss_pc),
                as_percent(self.average_delta_neutral_profit_pc),
                as_percent(self.average_trade)
            ],
            'Median PnL %': [
                as_percent(self.median_win),
                as_percent(self.median_loss),
                as_percent(self.median_delta_neutral),
                as_percent(self.median_trade)],
            'Biggest PnL %': [
                as_percent(self.biggest_winning_trade_pc),
                as_percent(self.biggest_losing_trade_pc),
                as_percent(self.biggest_delta_neutral_pc),
                as_percent(None)
            ],
            'Average duration': [
                self.format_duration(self.average_duration_of_winning_trades),
                self.format_duration(self.average_duration_of_losing_trades),
                self.format_duration(self.average_duration_of_delta_neutral_positions),
                self.format_duration(self.average_duration_of_all_trades)
            ],
            'Max consecutive streak': [
                as_integer(self.max_pos_cons),
                as_integer(self.max_neg_cons),
                as_integer(self.max_delta_neutral_cons),
                as_percent(None)
            ],
            'Max runup / drawdown': [
                as_percent(self.max_runup),
                as_percent(self.max_drawdown),
                as_percent(None),
                as_percent(None)
            ],
        }

        df3 = create_summary_table(data3, ["Winning", "Losing", "Delta Neutral", "Total"], "Closed Positions")

        data4 = {
            'Triggered exits': [
                as_integer(self.stop_losses),
                as_integer(self.take_profits)
            ],
            'Percent winning': [
                as_percent(self.winning_stop_losses_percent),
                as_percent(self.winning_take_profits_percent)
            ],
            'Percent losing': [
                as_percent(self.losing_stop_losses_percent),
                as_percent(self.losing_take_profits_percent)
            ],
            'Percent of total': [
                as_percent(self.all_stop_loss_percent),
                as_percent(self.all_take_profit_percent)
            ],
        }

        df4 = create_summary_table(data4, ["Stop losses", "Take profits"], "Position Exits")

        data5 = {
            'Biggest realised risk': as_percent(self.max_realised_loss),
            'Average realised risk': as_percent(self.avg_realised_risk),
            'Max pullback of capital': as_percent(self.max_pullback),
            'Sharpe Ratio': as_decimal(self.sharpe_ratio),
            'Sortino Ratio': as_decimal(self.sortino_ratio),
            'Profit Factor': as_decimal(self.profit_factor),
        }

        df5 = create_summary_table(data5, "", "Risk Analysis")

        # data6 = {
        #     'Average interest paid': as_dollar(self.average_interest_paid_usd),
        #     'Median interest paid': as_dollar(self.median_interest_paid_usd),
        #     'Max interest paid': as_dollar(self.max_interest_paid_usd),
        #     'Min interest paid': as_dollar(self.min_interest_paid_usd),
        #     'Total interest paid': as_dollar(self.total_interest_paid_usd),
        #     'Total interest claimed': as_dollar(self.total_claimed_interest),
        # }

        # df6 = create_summary_table(data6, "", "Interest Paid")

        display(self.single_column_dfs(df1, df2, df3, df4, df5))

    def format_duration(self, duration_timedelta):
        if not duration_timedelta:
            return as_duration(datetime.timedelta(0))

        return as_duration(duration_timedelta)

    def format_bars(self, duration_timedelta):
        if not duration_timedelta:
            return as_bars(0)

        if self.time_bucket is not None:
            return as_bars(duration_timedelta / self.time_bucket.to_timedelta())
        else:
            raise ValueError("Time bucket not specified")

    @staticmethod
    def single_column_dfs(*dfs):
        html = '<div style="display:flex; flex-direction:column">'
        for df in dfs:
            html += '<div style="margin-bottom: 2em">'
            html += df.to_html()
            html += '</div>'
        html += '</div>'
        return HTML(html)


@dataclass
class TradeAnalysis:
    """Analysis of trades in a portfolio."""

    #: The portfolio we analysed
    portfolio: Portfolio

    #: All taken positions sorted by the position id, or when they were opened
    filtered_sorted_positions: list[TradingPosition] = field(init=False)

    #: Decision cycle frequency is needed to calculate some performance metrics like sharpe, etc.
    #:
    #: If not given assume daily for the legacy compatibilty
    decision_cycle_frequency: pd.DateOffset = field(default_factory=pd.offsets.Day)

    def get_first_opened_at(self) -> Optional[pd.Timestamp]:
        """Get the opened_at timestamp of the first position in the portfolio."""
        return min(
            position.opened_at for id, position in self.get_all_positions()
        )

    def get_last_closed_at(self) -> Optional[pd.Timestamp]:
        """Get the closed_at timestamp of the last position in the portfolio."""

        return max(
            position.closed_at for id, position in self.get_all_positions()
        )

    def get_all_positions(self) -> Iterable[Tuple[PrimaryKey, TradingPosition]]:
        """Return open and closed positions over all traded assets.
        
        Positions are sorted by position_id."""

        for position in self.portfolio.get_all_positions():
            # pair_id, position
            yield position.pair.internal_id, position

    def get_open_positions(self) -> Iterable[Tuple[PrimaryKey, TradingPosition]]:
        """Return open positions over all traded assets.
        
        Positions are sorted by position_id."""

        for id, position in self.get_all_positions():
            if position.is_open():
                # pair_id, position
                yield position.pair.internal_id, position

    def get_short_positions(self) -> Iterable[Tuple[PrimaryKey, TradingPosition]]:
        """Return short positions over all traded assets.
        
        Positions are sorted by position_id."""

        for id, position in self.get_all_positions():
            if position.is_short():
                # pair_id, position
                yield position.pair.internal_id, position

    def get_long_positions(self) -> Iterable[Tuple[PrimaryKey, TradingPosition]]:
        """Return long positions over all traded assets.
        
        Positions are sorted by position_id."""

        for id, position in self.get_all_positions():
            if position.is_long():
                # pair_id, position
                yield position.pair.internal_id, position

    def calculate_summary_statistics(
            self,
            time_bucket: Optional[TimeBucket] = None,
            state=None,
    ) -> TradeSummary:
        """Calculate some statistics how our trades went. This is just for overall statistics. For an analysis by overall, long, and short trades, use :py:meth:`calculate_all_summary_stats_by_side`

        :param time_bucket:
            Optional, used to display average duration as 'number of bars' instead of 'number of days'.

        :param state:
            Optional, should be specified if user would like to see advanced statistics
        
        :return:
            TradeSummary instance
        """
        
        if state is not None:
            # for advanced statistics
            # import here to avoid circular import error
            from tradeexecutor.visual.equity_curve import calculate_daily_returns, calculate_returns, calculate_equity_curve, calculate_compounding_realised_trading_profitability
            from tradeexecutor.statistics.key_metric import calculate_sharpe, calculate_sortino, calculate_profit_factor, calculate_max_drawdown, calculate_max_runup

            daily_returns = calculate_daily_returns(state, freq="D")
            equity_curve = calculate_equity_curve(state)
            original_returns = calculate_returns(equity_curve)
            compounding_returns = calculate_compounding_realised_trading_profitability(state)

            sharpe_ratio = calculate_sharpe(daily_returns)
            sortino_ratio = calculate_sortino(daily_returns)

            # these stats not annualised, so better to calculate it on the original returns
            profit_factor = calculate_profit_factor(original_returns)
            max_drawdown = calculate_max_drawdown(original_returns)
            max_runup = calculate_max_runup(original_returns)
        else:
            # logger.warning("State not provided, some advanced statistics will not be calculated, and time_in_market may be inaccurate. Rather provide state and time_bucket arguments")

            daily_returns = None
            compounding_returns = None
            sharpe_ratio = None
            sortino_ratio = None
            profit_factor = None
            max_drawdown = None
            max_runup = None
        
        trade_summary = self.calculate_summary_statistics_for_positions(time_bucket, state, self.get_all_positions())
        
        trade_summary.daily_returns = daily_returns
        trade_summary.compounding_returns = compounding_returns
        trade_summary.sharpe_ratio = sharpe_ratio
        trade_summary.sortino_ratio = sortino_ratio
        trade_summary.profit_factor = profit_factor
        trade_summary.max_drawdown = max_drawdown
        trade_summary.max_runup = max_runup
        
        return trade_summary

    def calculate_short_summary_statistics(
            self,
            time_bucket: TimeBucket,
            state: 'State',
    ) -> TradeSummary:
        """Calculate some statistics how our short trades went.

        :param time_bucket:
            Optional, used to display average duration as 'number of bars' instead of 'number of days'.

        :param state:
            Optional, should be specified if user would like to see advanced statistics
        
        :return:
            TradeSummary instance
        """
        compounding_returns = None
        assert state is not None, "State must be provided to calculate short summary statistics"
        
        # import here to avoid circular import error
        from tradeexecutor.visual.equity_curve import calculate_short_compounding_realised_trading_profitability
        compounding_returns = calculate_short_compounding_realised_trading_profitability(state)
        
        short_summary = self.calculate_summary_statistics_for_positions(time_bucket, state, self.get_short_positions())
        short_summary.compounding_returns = compounding_returns

        short_summary.return_percent = compounding_returns.iloc[-1] if len(compounding_returns) > 0 else None
        
        return short_summary

    def calculate_long_summary_statistics(
            self,
            time_bucket: TimeBucket,
            state: 'State',
    ) -> TradeSummary:
        """Calculate some statistics how our long trades went.

        :param time_bucket:
            Optional, used to display average duration as 'number of bars' instead of 'number of days'.

        :param state:
            Optional, should be specified if user would like to see advanced statistics
        
        :return:
            TradeSummary instance
        """
        compounding_returns = None
        assert state is not None, "State must be provided to calculate long summary statistics"
        
        # import here to avoid circular import error
        from tradeexecutor.visual.equity_curve import calculate_long_compounding_realised_trading_profitability
        compounding_returns = calculate_long_compounding_realised_trading_profitability(state)

        long_summary =  self.calculate_summary_statistics_for_positions(time_bucket, state, self.get_long_positions())
        long_summary.compounding_returns = compounding_returns

        long_summary.return_percent = compounding_returns.iloc[-1] if len(compounding_returns) > 0 else None
        
        return long_summary

    def calculate_summary_statistics_for_positions(
            self,
            time_bucket: Optional[TimeBucket],
            state,
            positions: Iterable[Tuple[PrimaryKey, TradingPosition]]
    ) -> TradeSummary:
        """Calculate some statistics how our trades went.

            :param time_bucket:
                Optional, used to display average duration as 'number of bars' instead of 'number of days'.

            :param state:
                Optional, should be specified if user would like to see advanced statistics
            
            :return:
                TradeSummary instance
        """

        sorted_positions = sorted(positions, key=lambda position: position[1].opened_at)

        if time_bucket is not None:
            assert isinstance(time_bucket, TimeBucket), "Not a valid time bucket"

        def get_avg_profit_pct_check(trades: List | None):
            return float(np.mean(trades)) if trades else None

        def get_avg_trade_duration(duration_list: List | None):
            if duration_list:
                return pd.Timedelta(np.mean(duration_list))
            else:
                return pd.Timedelta(datetime.timedelta(0))

        def func_check(lst, func):
            return func(lst) if lst else None
        
        def _get_final_start_time(previous_position_closed_at, current_position_opened_at):
            """Used in `time_in_market` calculation"""
            if not previous_position_closed_at or current_position_opened_at > previous_position_closed_at:
                return current_position_opened_at
            else:
                return previous_position_closed_at  # overlapping
        
        times_in_market_all = []
        times_in_market_volatile = []  # excludes delta neutral positions
        _positions = []

        def _append_last_position(
            grouped_duration: pd.Timedelta,
            position: TradingPosition,
            previous_position_closed_at: pd.Timedelta | None,
            _position_duration: pd.Timedelta | None = None,
        ) -> None:
            """Last position has to be handled separately.
            
            :param grouped_duration:
                - Duration of the previous group of positions. If `position` is closed and is overlapping with previous position, **will** include the current position's duration.
                - Should never include credit supply positions (we calculate them separately)

            :param position:
                The last TradingPosition instance

            :param previous_position_closed_at:
                The closed_at timestamp of the previous position. Used to check for overlapping positions.
            
            :_position_duration:
                Override the position's duration. Needed when doing calculations on open positions

            :return:
                None
            """
            if _position_duration:
                assert position.is_open(), "Should be open position"
            else:
                assert position.is_closed(), "Should be closed position"
            
            if position.is_closed() and open_position_lock:
                # already included when dealing with open position
                return None

            position_duration = _position_duration or position.get_duration()
            
            assert position_duration is not None, "position_duration should not be None here"
            
            if not previous_position_closed_at:
                assert len(_positions) == (1 if position.is_closed() else 0), "Should be the only position"
                assert grouped_duration == (position_duration if position.is_closed() else datetime.timedelta(0)), "Should be the only position"

                times_in_market_all.append(position_duration)
                if not position.is_credit_supply():
                    times_in_market_volatile.append(position_duration)
            elif previous_position_closed_at > position.opened_at:
                # overlapping group
                assert not position.is_credit_supply(), "Delta neutral positions should not be here"
                
                if not position.closed_at:
                    grouped_duration += position_duration
                    times_in_market_all.append(grouped_duration)
                    times_in_market_volatile.append(grouped_duration)
                else:
                    pass
                    # already included if closed and overlapping
                    # grouped_duration += position.closed_at - previous_position_closed_at
            else:
                # new group
                if position.is_open():
                    times_in_market_all.append(grouped_duration)
                    times_in_market_volatile.append(grouped_duration)
                else:
                    pass
                    # grouped_duration = position_duration if closed and not overlapping
                times_in_market_all.append(position_duration)
                if not position.is_credit_supply():
                    times_in_market_volatile.append(position_duration)

            # for safety
            if last_closed_position_info:
                last_closed_position_info["grouped_duration"] = datetime.timedelta(0)
            
            return None

        def _get_new_grouped_duration_and_append(
            grouped_duration: pd.Timedelta, 
            position: TradingPosition, 
        ) -> pd.Timedelta | None:
            """Called when we have a new position or new group of overlapping positions. 
            This new group should not be overlapping with any previous position
            
            Append position duration to `times_in_market_all` and `times_in_market_volatile` lists.
            
            :param grouped_duration:
                - NB: does not include current position's duration. It is the duration of the previous group. (unlike in _append_last_position)
                - Should never include credit supply positions (added separately to lists)

            :param position:
                The TradingPosition instance

            :param _position_duration:
                Override the position's duration. Needed when doing calculations on open positions

            :param last_position:
                Whether the provided position is the last open or last closed position chronologically
                
            :return:
                The new grouped_duration that includes the current position's duration (unless it is credit supply). 
            """
            if len(_positions) > 1:
                assert _positions[-1].closed_at <= position.opened_at, "Overlapping positions. Should not happen here"
            
            position_duration = position.get_duration()
            assert position_duration is not None, "Position duration should be provided"
            assert grouped_duration, "Grouped duration should not be Falsy here"

            times_in_market_all.append(grouped_duration)
            times_in_market_volatile.append(grouped_duration)

            if position.is_credit_supply():
                times_in_market_all.append(position_duration)
                grouped_duration = datetime.timedelta(0) # don't include credit supply in new group
            else:
                grouped_duration = position_duration or datetime.timedelta(0)  # new group

            return grouped_duration

        initial_cash = self.portfolio.get_initial_cash()

        uninvested_cash = self.portfolio.get_cash()                          

        # EthLisbon hack
        extra_return = 0

        duration = datetime.timedelta(0)

        # Note: 'trades' actually refers to positions
        winning_trades = []
        losing_trades = []
        delta_neutral_positions = []
        winning_trades_duration = []
        losing_trades_duration = []
        delta_neutral_positions_duration = []
        zero_loss_trades_duration = []
        
        loss_risk_at_open_pc = []
        realised_losses_pct = []
        interest_paid_usd = []
        durations_between_positions = []

        biggest_winning_trade_pc = None
        biggest_losing_trade_pc = None

        average_duration_of_losing_trades = datetime.timedelta(0)
        average_duration_of_winning_trades = datetime.timedelta(0)
        average_duration_of_delta_neutral_positions = datetime.timedelta(0)
        average_duration_of_zero_loss_trades = None
        average_duration_of_all_trades = None

        if state is None:
            # legacy
            strategy_duration = self.portfolio.get_trading_history_duration()
        else:
            strategy_duration = state.get_strategy_duration()
        
        won = lost = zero_loss = stop_losses = take_profits = undecided = delta_neutral = 0
        open_value: USDollarAmount = 0
        profit: USDollarAmount = 0
        unrealised_profit_usd: USDollarAmount = 0
        trade_volume = 0
        lp_fees_paid = 0

        max_pos_cons = 0
        max_neg_cons = 0
        max_delta_neutral_cons = 0
        max_pullback_pct = 0
        pos_cons = 0
        neg_cons = 0
        delta_neutral_cons = 0
        pullback = 0
        total_trades = 0

        winning_stop_losses = 0
        losing_stop_losses = 0
        winning_take_profits = 0
        losing_take_profits = 0

        previous_position_opened_at = None
        previous_position_closed_at = None
        last_closed_position_info = None
        grouped_duration = datetime.timedelta(0)
        open_position_lock = False

        total_claimed_interest = 0

        for pair_id, position in sorted_positions:

            total_trades += len(position.trades)
            portfolio_value_at_open = position.portfolio_value_at_open

            capital_tied_at_open_pct = self.get_capital_tied_at_open(position)
            
            if position.stop_loss:
                maximum_risk = position.get_loss_risk_at_open()  # TODO use maximum_risk
                loss_risk_at_open_pc.append(position.get_loss_risk_at_open_pct())
            else:
                maximum_risk = None
                loss_risk_at_open_pc.append(capital_tied_at_open_pct)

            lp_fees_paid += position.get_total_lp_fees_paid() or 0
            interest_paid_usd.append(position.get_repaid_interest())

            for t in position.trades.values():
                trade_volume += t.get_value()
            
            if previous_position_opened_at is not None:
                durations_between_positions.append(position.opened_at - previous_position_opened_at) 

            if position.is_credit_supply():
                delta_neutral_cons += 1
            else:
                delta_neutral_cons = 0

            if delta_neutral_cons > max_delta_neutral_cons:
                max_delta_neutral_cons = delta_neutral_cons

            total_claimed_interest += position.get_claimed_interest()

            if position.is_open():
                open_value += position.get_value()
                unrealised_profit_usd += position.get_unrealised_profit_usd()
                undecided += 1
                
                # time in market for open positions
                # only count the first open position for `time in market`
                # otherwise double counting
                if open_position_lock == False and state:
                    strategy_start, strategy_end = state.get_strategy_time_range()

                    if strategy_end:
                        position_duration = strategy_end - position.opened_at
                        _append_last_position(grouped_duration, position, previous_position_closed_at, position_duration)
                        previous_position_closed_at = None
                    open_position_lock = True

                continue

            # time in market for closed positions
            if grouped_duration == datetime.timedelta(0):
                # start new group
                if len(_positions) > 0:
                    assert not position.is_credit_supply(), "Delta neutral positions should not be here"
                grouped_duration += position.get_duration()
            elif previous_position_closed_at:
                if position.opened_at < previous_position_closed_at:
                    # overlapping group
                    assert not position.is_credit_supply(), "Delta neutral positions should not overlap"
                    grouped_duration += (position.closed_at - previous_position_closed_at)  
                else:
                    # end new group
                    # credit supply should only be here
                    grouped_duration = _get_new_grouped_duration_and_append(grouped_duration, position)
            else:
                raise ValueError("previous_position_closed_at is None. This should not happen.")

            _positions.append(position)
            last_closed_position_info = {
                "grouped_duration": grouped_duration,
                "position": position,
                "previous_position_closed_at": previous_position_closed_at,
            }
            previous_position_opened_at = position.opened_at
            previous_position_closed_at = position.closed_at

            is_stop_loss = position.is_stop_loss()
            is_take_profit = position.is_take_profit()

            if is_stop_loss:
                stop_losses += 1

            if is_take_profit:
                take_profits += 1

            realised_profit_usd = position.get_realised_profit_usd()
            assert realised_profit_usd is not None, f"Realised profit calculation failed for: {position}"
            realised_profit_percent = position.get_realised_profit_percent()

            duration = position.get_duration()

            if position.is_credit_supply():
                delta_neutral += 1
                delta_neutral_positions.append(realised_profit_percent)
                delta_neutral_positions_duration.append(duration)

            elif position.is_profitable():
                won += 1
                winning_trades.append(realised_profit_percent)
                winning_trades_duration.append(duration)

                if is_stop_loss:
                    winning_stop_losses += 1

                if is_take_profit:
                    winning_take_profits += 1

            elif position.is_loss():
                lost += 1
                losing_trades.append(realised_profit_percent)
                losing_trades_duration.append(duration)

                if portfolio_value_at_open := position.portfolio_value_at_open:
                    realised_loss_pct = realised_profit_usd / portfolio_value_at_open
                else:
                    # Bad data
                    realised_loss_pct = 0
                realised_losses_pct.append(realised_loss_pct)

                if is_stop_loss:
                    losing_stop_losses += 1

                if is_take_profit:
                    losing_take_profits += 1

            else:
                # Any profit exactly balances out loss in slippage and commission
                zero_loss += 1
                zero_loss_trades_duration.append(duration)

            profit += realised_profit_usd

            # for getting max consecutive wins/losses and max pullback
            # don't do anything if profit = $0

            if not position.is_credit_supply():
                if (realised_profit_usd > 0):
                    neg_cons = 0
                    pullback = 0
                    pos_cons += 1
                elif (realised_profit_usd < 0):
                    pos_cons = 0
                    neg_cons += 1
                    pullback += realised_profit_usd

            if (neg_cons > max_neg_cons):
                max_neg_cons = neg_cons
            if (pos_cons > max_pos_cons):
                max_pos_cons = pos_cons

            if portfolio_value_at_open:
                pullback_pct = pullback / (portfolio_value_at_open + realised_profit_usd)
                if (pullback_pct < max_pullback_pct):
                    # pull back is in the negative direction
                    max_pullback_pct = pullback_pct
            else:
                # Bad input data / legacy data
                max_pullback_pct = 0

        if last_closed_position_info:
            # add time in market for very last closed position
             _append_last_position(**last_closed_position_info)


        all_trades = winning_trades + losing_trades + [0 for i in range(zero_loss)]
        average_trade = func_check(all_trades, avg)
        median_trade = func_check(all_trades, median)
        median_win = func_check(winning_trades, median)
        median_loss = func_check(losing_trades, median)
        median_delta_neutral = func_check(delta_neutral_positions, median)

        average_winning_trade_profit_pc = get_avg_profit_pct_check(winning_trades)
        average_losing_trade_loss_pc = get_avg_profit_pct_check(losing_trades)
        average_delta_neutral_profit_pc = get_avg_profit_pct_check(delta_neutral_positions)

        max_realised_loss = func_check(realised_losses_pct, min)
        avg_realised_risk = func_check(realised_losses_pct, avg)
        max_loss_risk_at_open_pc = func_check(loss_risk_at_open_pc, max)
        
        average_interest_paid_usd = func_check(interest_paid_usd, avg)
        median_interest_paid_usd = func_check(interest_paid_usd, median)
        max_interest_paid_usd = func_check(interest_paid_usd, max)
        min_interest_paid_usd = func_check(interest_paid_usd, min)
        average_duration_between_position_openings = pd.to_timedelta(durations_between_positions).mean() if len(durations_between_positions) > 0 else datetime.timedelta(0)

        biggest_winning_trade_pc = func_check(winning_trades, max)
        biggest_losing_trade_pc = func_check(losing_trades, min)
        biggest_delta_neutral_pc = func_check(delta_neutral_positions, max)
        
        time_in_market = pd.to_timedelta(times_in_market_all).sum()/strategy_duration if (len(times_in_market_all) > 0 and strategy_duration) else 0
        time_in_market_volatile = pd.to_timedelta(times_in_market_volatile).sum()/strategy_duration if (len(times_in_market_volatile) > 0 and strategy_duration) else 0

        all_durations = winning_trades_duration + losing_trades_duration + zero_loss_trades_duration
        average_duration_of_winning_trades = get_avg_trade_duration(winning_trades_duration)
        average_duration_of_losing_trades = get_avg_trade_duration(losing_trades_duration)
        if delta_neutral_positions_duration:
            average_duration_of_delta_neutral_positions = get_avg_trade_duration(delta_neutral_positions_duration)
        if zero_loss_trades_duration:
            average_duration_of_zero_loss_trades = get_avg_trade_duration(zero_loss_trades_duration)
        if all_durations:
            average_duration_of_all_trades = get_avg_trade_duration(all_durations)

        lp_fees_average_pc = lp_fees_paid / trade_volume if trade_volume else 0

        total_interest_paid_usd = sum(interest_paid_usd) if interest_paid_usd else 0

        return TradeSummary(
            won=won,
            lost=lost,
            zero_loss=zero_loss,
            stop_losses=stop_losses,
            take_profits=take_profits,
            total_trades=total_trades,
            undecided=undecided,
            realised_profit=profit + extra_return,
            unrealised_profit=unrealised_profit_usd,
            open_value=open_value,
            uninvested_cash=uninvested_cash,
            initial_cash=initial_cash or 0,  # Do not pass None for serialisation for live strategies
            extra_return=extra_return,
            duration=strategy_duration or datetime.timedelta(seconds=0),  # Do not pass None for serialisation for live strategies
            average_winning_trade_profit_pc=average_winning_trade_profit_pc or 0,  # Do not pass None for serialisation for live strategies
            average_losing_trade_loss_pc=average_losing_trade_loss_pc or 0,  # Do not pass None for serialisation for live strategies
            biggest_winning_trade_pc=biggest_winning_trade_pc,
            biggest_losing_trade_pc=biggest_losing_trade_pc,
            average_duration_of_winning_trades=average_duration_of_winning_trades,
            average_duration_of_losing_trades=average_duration_of_losing_trades,
            average_duration_of_zero_loss_trades=average_duration_of_zero_loss_trades,
            average_duration_of_all_trades=average_duration_of_all_trades,
            average_duration_between_position_openings=average_duration_between_position_openings,
            average_trade=average_trade,
            average_interest_paid_usd=average_interest_paid_usd,
            median_interest_paid_usd=median_interest_paid_usd,
            max_interest_paid_usd=max_interest_paid_usd,
            min_interest_paid_usd=min_interest_paid_usd,
            total_interest_paid_usd=total_interest_paid_usd,
            total_claimed_interest=total_claimed_interest,
            median_trade=median_trade,
            median_win=median_win,
            median_loss=median_loss,
            max_pos_cons=max_pos_cons,
            max_neg_cons=max_neg_cons,
            max_delta_neutral_cons=max_delta_neutral_cons,
            max_pullback=max_pullback_pct,
            #max_drawdown=max_drawdown,
            #max_runup=max_runup,
            max_loss_risk=max_loss_risk_at_open_pc,
            max_realised_loss=max_realised_loss,
            avg_realised_risk=avg_realised_risk,
            time_bucket=time_bucket,
            trade_volume=trade_volume,
            lp_fees_paid=lp_fees_paid,
            lp_fees_average_pc=lp_fees_average_pc,
            #daily_returns=daily_returns,
            winning_stop_losses=winning_stop_losses,
            losing_stop_losses=losing_stop_losses,
            winning_take_profits=winning_take_profits,
            losing_take_profits=losing_take_profits,
            #sharpe_ratio=sharpe_ratio,
            #sortino_ratio=sortino_ratio,
            #profit_factor=profit_factor,
            time_in_market = time_in_market,
            time_in_market_volatile = time_in_market_volatile,
            delta_neutral=delta_neutral,
            median_delta_neutral=median_delta_neutral,
            average_delta_neutral_profit_pc=average_delta_neutral_profit_pc,
            average_duration_of_delta_neutral_positions=average_duration_of_delta_neutral_positions,
            biggest_delta_neutral_pc=biggest_delta_neutral_pc,
        )

    @staticmethod
    def calculate_weighted_average_realised_profit(positions: Iterable[Tuple[PrimaryKey, TradingPosition]]):
        """Calculate profit % of all positions, weighted by position size.
        
        :param positions:
            Iterable of position ids

        :return:
            Profit % weighted by position size
        """
        total_profit = 0
        total_value = 0

        for pair_id, position in positions:
            total_profit += (position.get_realised_profit_usd() or 0) * position.get_value_at_open()
            total_value += position.get_price_at_open() * position.get_value_at_open()

        return total_profit / total_value if total_value else 0

    def calculate_all_summary_stats_by_side(
            self,
            time_bucket: Optional[TimeBucket] = None,
            state: Optional['State'] = None,
            urls=False,
    ) -> pd.DataFrame:
        """Calculate some statistics how our trades went. This returns a DataFrame with 3 separate columns for overall, long and short.
        
        For just a single column table for overall statistics, use :py:meth:calculate_summary_statistics() instead.

        :param time_bucket:
            Optional, used to display average duration as 'number of bars' in addition to 'number of days'.

        :param state:
            Optional, should be specified if user would like to see advanced statistics such as sharpe ratio, sortino ratio, etc.

        :param urls:
            Optional, if True, include an extra column for the urls for each row that link to the relevant glorssary documentation.

        :return:
            DataFrame with all the stats for overall, long and short.
        """

        all_stats_trade_summary = self.calculate_summary_statistics(time_bucket, state)
        long_stats_trade_summary = self.calculate_long_summary_statistics(time_bucket, state)
        short_stats_trade_summary = self.calculate_short_summary_statistics(time_bucket, state)

        all_stats = all_stats_trade_summary.to_dataframe(format_headings=False)
        long_stats = long_stats_trade_summary.to_dataframe(format_headings=False)
        short_stats = short_stats_trade_summary.to_dataframe(format_headings=False)

        all_stats['Long'] = long_stats[0]
        all_stats['Short'] = short_stats[0]

        # left blank in long and short
        blank_rows = ['Trading period length', 'Cash at start', 'Value at end', 'Cash left at the end', 'Max drawdown', 'Max pullback of total capital']

        for row in blank_rows:
            if row in all_stats.index:
                all_stats.loc[row, 'Long'] = '-'
                all_stats.loc[row, 'Short'] = '-'

        new_columns = all_stats.columns.to_list()
        new_columns[0] = 'All'
        all_stats.columns = new_columns

        # get return % stats for long and short
        duration = all_stats_trade_summary.duration

        # Calculate TradingView like profit split
        #
        # We have total return % e.g. 74%. or 10,000 USD
        # This is divided across long and short profits from all dollar profits.
        # If long made 8000 USd then they get 74% * (8000/10000) profit %. = 59%
        #
        all_profit_usd = all_stats_trade_summary.unrealised_profit + all_stats_trade_summary.realised_profit
        long_profit_usd = long_stats_trade_summary.realised_profit + long_stats_trade_summary.unrealised_profit
        short_profit_usd = short_stats_trade_summary.realised_profit + short_stats_trade_summary.unrealised_profit

        profit_long_pct, profit_short_pct = 0, 0

        if all_stats_trade_summary.return_percent:
            profit_long_pct = calculate_percentage(all_stats_trade_summary.return_percent * long_profit_usd, all_profit_usd)
            profit_short_pct = calculate_percentage(all_stats_trade_summary.return_percent * short_profit_usd, all_profit_usd)
        elif len(all_stats_trade_summary.compounding_returns) > 0:
            # TODO weight long_stats_trade_summary.compounding_returns and also for short
            profit_long_pct = calculate_percentage(all_stats_trade_summary.compounding_returns.iloc[-1] * long_profit_usd, all_profit_usd)
            profit_short_pct = calculate_percentage(all_stats_trade_summary.compounding_returns.iloc[-1] * short_profit_usd, all_profit_usd)

        # profit_long_pct = self.calculate_weighted_average_realised_profit(self.get_long_positions())
        # profit_short_pct = self.calculate_weighted_average_realised_profit(self.get_short_positions())

        all_stats.loc['Return %', 'Long'] = format_value_for_summary_table(as_percent(profit_long_pct))
        all_stats.loc['Return %', 'Short'] = format_value_for_summary_table(as_percent(profit_short_pct))

        all_stats.loc['Annualised return %', 'Long'] = format_value_for_summary_table(as_percent(calculate_annualised_return(profit_long_pct, duration)))
        all_stats.loc['Annualised return %', 'Short'] = format_value_for_summary_table(as_percent(calculate_annualised_return(profit_short_pct, duration)))

        if urls:
            all_stats['help_links'] = [TradeSummary.help_links()[key] for key in all_stats.index]

        return all_stats

    def render_summary_statistics_side_by_side(self, time_bucket: Optional[TimeBucket] = None, state=None) -> HTML:
        """Render summary statistics as a JSON table.

        :param time_bucket:
            Optional, used to display average duration as 'number of bars' in addition to 'number of days'.

        :param state:
            Optional, should be specified if user would like to see advanced statistics such as sharpe ratio, sortino ratio, etc.

        :return:
            Returns a similar table to `calcualate_all_summary_stats_by_side`, but make it makes row headings clickable, directing user to relevant glossary link and, in this case, returns ``HTML`` object instaed of a pandas DataFrame.
        """
        all_stats = self.calculate_all_summary_stats_by_side(time_bucket, state)

        return TradeSummary.format_summary_dataframe(all_stats)

    @staticmethod
    def get_capital_tied_at_open(position) -> Percent | None:
        """Calculate how much capital % was allocated to this position when it was opened."""
        if position.portfolio_value_at_open:
            return position.get_capital_tied_at_open_pct()
        else:
            return None

    def create_timeline(self) -> pd.DataFrame:
        """Create a timeline feed how we traded over a course of time.

        Note: We assume each position has only one enter and exit event, not position increases over the lifetime.

        :return: DataFrame with timestamp and timeline_event columns
        """

        def gen_events():
            for pair_id, position in self.get_all_positions():
                yield (position.position_id, position)

        df = pd.DataFrame(gen_events(), columns=["position_id", "position"])
        return df


def calculate_annualised_return(profit_pct: float, duration: datetime.timedelta) -> float | None:
    """Calculate annualised return %. 
    
    Uses seconds as the base unit for duration to accomodate for strategies that have been live for less than a day.
    
    :param profit_pct: Profit % over the duration
    :param duration: Duration of the trade as a datetime object
    :return: Annualised return % if profit_pct is not None
    """
    assert isinstance(profit_pct, float | int | None), "Profit % should be a float or int or None"
    assert isinstance(duration, datetime.timedelta), f"Duration should be a datetime.timedelta object, got {type(duration)}"
    return calculate_percentage(profit_pct * 365 * 24 * 60 * 60, duration.total_seconds()) if profit_pct else None


class TimelineRowStylingMode(enum.Enum):
    #: Style using Pandas background_gradient
    gradient = "gradient"

    #: Simple
    #: Profit = green, loss = red
    simple = "simple"


class TimelineStyler:
    """Style the expanded trades timeline table.

    Give HTML hints for DataFrame how it should be rendered
    in the notebook output.
    """

    def __init__(self,
                 row_styling: TimelineRowStylingMode,
                 hidden_columns: List[str],
                 vmin: float,
                 vmax: float,
                 ):
        self.row_styling = row_styling
        self.hidden_columns = hidden_columns
        self.vmin = vmin
        self.vmax = vmax

    def colour_timelime_row_simple(self, row: pd.Series) -> pd.Series:
        """Set colour for each timeline row based on its profit.

        - +/- 5% colouring

        - More information: https://stackoverflow.com/a/49745352/315168

        - CSS colours: https://htmlcolorcodes.com/color-names/
        """

        pnl_raw = row["PnL % raw"]

        if pnl_raw < -0.05:
            return pd.Series('background-color: Salmon; color: black', row.index)
        elif pnl_raw < 0:
            return pd.Series('background-color: LightSalmon; color: black', row.index)
        elif pnl_raw > 0.05:
            return pd.Series('background-color: LawnGreen; color: black', row.index)
        else:
            return pd.Series('background-color: PaleGreen; color: black', row.index)

    def __call__(self, df: pd.DataFrame):
        """Applies styles on a dataframe

        :param df:
            Dataframe as returned by :py:func`expand_timeline`.
        """
        # Create a Pandas Styler with multiple styling options applied
        try:
            styles = df.style \
                .hide(axis="index") \
                .hide(axis="columns", subset=self.hidden_columns)
        except KeyError:
            # The input df was empty (no trades)
            styles = df.style

        # Don't let the text inside a cell to wrap
        styles = styles.set_table_styles({
            "Opened at": [{'selector': 'td', 'props': [('white-space', 'nowrap')]}],
            "Exchange": [{'selector': 'td', 'props': [('white-space', 'nowrap')]}],
        })

        if self.row_styling == TimelineRowStylingMode.gradient:
            # Dynamically color the background of trade outcome coluns # https://pandas.pydata.org/docs/reference/api/pandas.io.formats.style.Styler.background_gradient.html
            # TODO: This gradient styling is confusing
            # get rid of it long term
            styles = styles.background_gradient(
                axis=0,
                gmap=df['PnL % raw'],
                cmap='RdYlGn',
                vmin=self.vmin,  # We can only lose 100% of our money on position
                vmax=self.vmax)  # 50% profit is 21.5 position. Assume this is the max success color we can hit over
        else:
            styles = styles.apply(self.colour_timelime_row_simple, axis=1)

        return styles


def expand_timeline(
        exchanges: Set[Exchange],
        pair_universe: PandasPairUniverse,
        timeline: pd.DataFrame,
        vmin=-0.3,
        vmax=0.2,
        timestamp_format="%Y-%m-%d",
        hidden_columns=["Id", "PnL % raw"],
        row_styling_mode=TimelineRowStylingMode.simple,
) -> Tuple[pd.DataFrame, TimelineStyler]:
    """Expand trade history timeline to human readable table.

    This will the outputting much easier in Python Notebooks.

    Currently does not incrementing/decreasing positions gradually.

    Instaqd of applying styles or returning a styled dataframe, we return a callable that applies the styles.
    This is because of Pandas issue https://github.com/pandas-dev/pandas/issues/40675 - hidden indexes, columns,
    etc. are not exported.

    :param exchanges: Needed for exchange metadata

    :param pair_universe: Needed for trading pair metadata

    :param vmax: Trade success % to have the extreme green color.

    :param vmin: The % of lost capital on the trade to have the extreme red color.

    :param timestamp_format: How to format Opened at column, as passed to `strftime()`

    :param hidden_columns: Hide columns in the output table

    :return: DataFrame with human=readable position win/loss information, having DF indexed by timestamps and a styler function
    """

    exchange_map = {e.exchange_id: e for e in exchanges}

    # https://stackoverflow.com/a/52363890/315168
    def expander(row):
        position: TradingPosition = row["position"]
        # timestamp = row.name  # ???
        pair_id = position.pair.get_pricing_pair().internal_id
        pair_info = pair_universe.get_pair_by_id(pair_id)
        exchange = exchange_map.get(pair_info.exchange_id)
        if not exchange:
            raise RuntimeError(f"No exchange for id {pair_info.exchange_id}, pair {pair_info}")

        if position.is_stop_loss():
            remarks = "SL"
        elif position.is_take_profit():
            remarks = "TP"
        else:
            remarks = ""

        # Hack around to work with legacy data issue.
        # Not an issue for new strategies.
        if position.has_bad_data_issues():
            remarks += "BAD"

        duration = position.get_duration()

        r = {
            # "timestamp": timestamp,
            "Id": position.position_id,
            "Remarks": remarks,
            "Type": "Long" if position.is_long() else "Short",
            "Opened at": position.opened_at.strftime(timestamp_format),
            "Duration": format_duration_days_hours_mins(duration) if duration else np.nan,
            "Exchange": exchange.name,
            "Base asset": pair_info.base_token_symbol,
            "Quote asset": pair_info.quote_token_symbol,
            "Position max value": format_value(position.get_max_size()),
            "PnL USD": format_value(position.get_realised_profit_usd()) if position.is_closed() else np.nan,
            "PnL %": format_percent_2_decimals(position.get_realised_profit_percent()) if position.is_closed() else np.nan,
            "PnL % raw": position.get_realised_profit_percent() if position.is_closed() else 0,
            "Open mid price USD": format_price(position.get_opening_price()),
            "Close mid price USD": format_price(position.get_closing_price()) if position.is_closed() else np.nan,
            "Trade count": position.get_trade_count(),
            "LP fees": f"${position.get_total_lp_fees_paid():,.2f}",
            "Notes": f"{position.notes or ''}"
        }
        return r

    applied_df = timeline.apply(expander, axis='columns', result_type='expand')

    if len(applied_df) > 0:
        # https://stackoverflow.com/a/52720936/315168
        applied_df \
            .sort_values(by=['Id'], ascending=[True], inplace=True)

    # Get rid of NaN labels
    # https://stackoverflow.com/a/28390992/315168
    applied_df.fillna('', inplace=True)

    styling = TimelineStyler(
        row_styling=row_styling_mode,
        hidden_columns=hidden_columns,
        vmin=vmin,
        vmax=vmax,
    )

    return applied_df, styling


def expand_timeline_raw(
        timeline: pd.DataFrame,
        timestamp_format="%Y-%m-%d"
) -> pd.DataFrame:
    """A simplified version of expand_timeline that does not care about
    pair info, exchanges, or opening capital, and also provides raw figures.

    Unused in codebase, but can be useful for advanced users to use directly"""

    # https://stackoverflow.com/a/52363890/315168
    def expander(row):
        position: TradingPosition = row["position"]
        # timestamp = row.name  # ???
        pair_id = position.pair.internal_id

        if position.is_stop_loss():
            remarks = "SL"
        elif position.is_take_profit():
            remarks = "TP"
        else:
            remarks = ""

        pnl_usd = position.get_realised_profit_usd() if position.is_closed() else np.nan
        duration = position.get_duration()

        r = {
            # "timestamp": timestamp,
            "Id": position.position_id,
            "Remarks": remarks,
            "Opened at": position.opened_at.strftime(timestamp_format),
            "Duration": format_duration_days_hours_mins(duration) if duration else np.nan,
            "position_max_size": position.get_max_size(),
            "pnl_usd": pnl_usd,
            "pnl_pct_raw": position.get_realised_profit_percent() if position.is_closed() else 0,
            "open_price_usd": position.get_opening_price(),
            "close_price_usd": position.get_closing_price() if position.is_closed() else np.nan,
            "trade_count": position.get_trade_count(),
        }

        return r

    applied_df = timeline.apply(expander, axis='columns', result_type='expand')

    if len(applied_df) > 0:
        # https://stackoverflow.com/a/52720936/315168
        applied_df \
            .sort_values(by=['Id'], ascending=[True], inplace=True)

    # Get rid of NaN labels
    # https://stackoverflow.com/a/28390992/315168
    applied_df.fillna('', inplace=True)

    return applied_df


def build_trade_analysis(
        portfolio: Portfolio
) -> TradeAnalysis:
    """Build a trade analysis from list of positions.

    - Read positions from backtesting or live state

    - Create TradeAnalysis instance that can be used to display IPython notebook
      data on the performance

    """

    return TradeAnalysis(
        portfolio,
    )


def avg(lst: list[int]):
    return sum(lst) / len(lst)


def avg_timedelta(lst: list[pd.Timedelta]):
    return pd.Series(lst).mean()
