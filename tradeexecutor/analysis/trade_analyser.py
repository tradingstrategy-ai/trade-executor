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
from dataclasses_json import dataclass_json, config
from statistics import median

from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.trade import TradeExecution, TradeType
from tradeexecutor.state.types import USDollarPrice, Percent
from tradeexecutor.utils.format import calculate_percentage
from tradeexecutor.utils.timestamp import json_encode_timedelta, json_decode_timedelta
from tradingstrategy.timebucket import TimeBucket

from tradingstrategy.exchange import Exchange
from tradingstrategy.pair import PandasPairUniverse
from tradingstrategy.types import PrimaryKey, USDollarAmount
from tradingstrategy.utils.format import format_value, format_price, format_duration_days_hours_mins, \
    format_percent_2_decimals
from tradingstrategy.utils.summarydataframe import as_dollar, as_integer, create_summary_table, as_percent, as_duration, as_bars


try:
    import quantstats as qs
    HAS_QUANTSTATS = True
except Exception:
    HAS_QUANTSTATS = False


logger = logging.getLogger(__name__)


@dataclass_json
@dataclass(slots=True)
class TradeSummary:
    """Some generic statistics over all the trades"""
    won: int
    lost: int
    zero_loss: int
    stop_losses: int
    undecided: int
    realised_profit: USDollarAmount
    open_value: USDollarAmount
    uninvested_cash: USDollarAmount

    initial_cash: USDollarAmount
    extra_return: USDollarAmount
    duration: datetime.timedelta = field(metadata=config(
        encoder=json_encode_timedelta,
        decoder=json_decode_timedelta,
    ))

    average_winning_trade_profit_pc: float # position
    average_losing_trade_loss_pc: float # position
    biggest_winning_trade_pc: Optional[float] # position
    biggest_losing_trade_pc: Optional[float] # position

    average_duration_of_winning_trades: datetime.timedelta = field(metadata=config(
        encoder=json_encode_timedelta,
        decoder=json_decode_timedelta,
    )) # position
    average_duration_of_losing_trades: datetime.timedelta = field(metadata=config(
        encoder=json_encode_timedelta,
        decoder=json_decode_timedelta,
    )) # position
    time_bucket: Optional[TimeBucket] = None

    # these stats calculate in post-init, so init=False
    total_positions: int = field(init=False)
    win_percent: float = field(init=False)
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

    average_trade: Optional[float] = None # position
    median_trade: Optional[float] = None # position
    max_pos_cons: Optional[int] = None
    max_neg_cons: Optional[int] = None
    max_pullback: Optional[float] = None
    max_loss_risk: Optional[float] = None
    max_realised_loss: Optional[float] = None
    avg_realised_risk: Optional[Percent] = None

    take_profits: int = field(default=0)

    trade_volume: USDollarAmount = field(default=0.0)

    lp_fees_paid: Optional[USDollarPrice] = 0
    lp_fees_average_pc: Optional[USDollarPrice] = 0

    #: advanced users can use this property instead of the
    #: provided quantstats helper methods
    daily_returns: Optional[pd.Series] = None
    
    winning_stop_losses: Optional[int] = 0
    losing_stop_losses: Optional[int] = 0
    
    winning_stop_losses_percent: Optional[float] = field(init=False)
    losing_stop_losses_percent: Optional[float] = field(init=False)

    def __post_init__(self):

        self.total_positions = self.won + self.lost + self.zero_loss
        self.win_percent = calculate_percentage(self.won, self.total_positions)
        self.all_stop_loss_percent = calculate_percentage(self.stop_losses, self.total_positions)
        self.all_take_profit_percent = calculate_percentage(self.take_profits, self.total_positions)
        self.lost_stop_loss_percent = calculate_percentage(self.stop_losses, self.lost)
        self.won_take_profit_percent = calculate_percentage(self.take_profits, self.won)
        self.average_net_profit = calculate_percentage(self.realised_profit, self.total_positions)
        self.end_value = self.open_value + self.uninvested_cash
        initial_cash = self.initial_cash or 0
        self.return_percent = calculate_percentage(self.end_value - initial_cash, initial_cash)
        self.annualised_return_percent = calculate_percentage(self.return_percent * datetime.timedelta(days=365),
                                                              self.duration) if self.return_percent else None

        self.winning_stop_losses_percent = calculate_percentage(self.winning_stop_losses, self.stop_losses)
        self.losing_stop_losses_percent = calculate_percentage(self.losing_stop_losses, self.stop_losses)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the data to a human readable summary table.

        """
        if(self.time_bucket is not None):
            avg_duration_winning = as_bars(self.average_duration_of_winning_trades)
            avg_duration_losing = as_bars(self.average_duration_of_losing_trades)
        else:
            avg_duration_winning = as_duration(self.average_duration_of_winning_trades)
            avg_duration_losing = as_duration(self.average_duration_of_losing_trades)

        """Creates a human-readable Pandas dataframe table from the object."""

        human_data = {
            "Trading period length": as_duration(self.duration),
            "Return %": as_percent(self.return_percent),
            "Annualised return %": as_percent(self.annualised_return_percent),
            "Cash at start": as_dollar(self.initial_cash),
            "Value at end": as_dollar(self.end_value),
            "Trade volume": as_dollar(self.trade_volume),
            "Position win percent": as_percent(self.win_percent),
            "Total positions": as_integer(self.total_positions),
            "Won positions": as_integer(self.won),
            "Lost positions": as_integer(self.lost),
            "Stop losses triggered": as_integer(self.stop_losses),
            "Stop loss % of all": as_percent(self.all_stop_loss_percent),
            "Stop loss % of lost": as_percent(self.lost_stop_loss_percent),
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
            "Portfolio unrealised value": as_dollar(self.open_value),
            "Extra returns on lending pool interest": as_dollar(self.extra_return),
            "Cash left at the end": as_dollar(self.uninvested_cash),
            "Average winning position profit %": as_percent(self.average_winning_trade_profit_pc),
            "Average losing position loss %": as_percent(self.average_losing_trade_loss_pc),
            "Biggest winning position %": as_percent(self.biggest_winning_trade_pc),
            "Biggest losing position %": as_percent(self.biggest_losing_trade_pc),
            "Average duration of winning positions": avg_duration_winning,
            "Average duration of losing positions": avg_duration_losing,
            "LP fees paid": as_dollar(self.lp_fees_paid),
            "LP fees paid % of volume": as_percent(self.lp_fees_average_pc),
        }

        def add_prop(value, key: str, formatter: Callable):
            human_data[key] = (
                formatter(value)
                if value is not None
                else formatter(0)
            )

        add_prop(self.average_trade, 'Average position:', as_percent)
        add_prop(self.median_trade, 'Median position:', as_percent)
        add_prop(self.max_pos_cons, 'Most consecutive wins', as_integer)
        add_prop(self.max_neg_cons, 'Most consecutive losses', as_integer)
        add_prop(self.max_realised_loss, 'Biggest realized risk', as_percent)
        add_prop(self.avg_realised_risk, 'Avg realised risk', as_percent)
        add_prop(self.max_pullback, 'Max pullback of total capital', as_percent)
        add_prop(self.max_loss_risk, 'Max loss risk at opening of position', as_percent)

        df = create_summary_table(human_data)
        return df

    def show(self):
        """Render a summary table in IPython notebook."""
        self.show_custom(self.to_dataframe())
    
    @staticmethod
    def show_custom(df: pd.DataFrame):
        """Render a summary table in IPython notebook.
        
        TODO: truncate unnecassary decimals at the end of floats
        """
        with pd.option_context("display.max_row", None):
            display(df.style.set_table_styles([{'selector': 'thead', 'props': [('display', 'none')]}]))
    
    @staticmethod
    def check_quantstats(function):
        """Decorator function that checks that requirements are met for using quantstats library. Used as decorator for provided helper methods."""
        def wrapper(*args, **kwargs):
            if not HAS_QUANTSTATS:
                raise RuntimeError("Quantstats library not installed")

            if not hasattr(args[0], "daily_returns"):
                raise RuntimeError("Daily returns have not been calculated. Remember to provided state argument. E.g. summary = analysis.calculate_summary_statistics(state=state)")
            
            with warnings.catch_warnings():
                
                # Silences 2 warnings from quantstats library
                # 1.                
                # /usr/local/lib/python3.10/site-packages/quantstats/stats.py:968: FutureWarning: In a future version of pandas all arguments of DataFrame.pivot will be keyword-only.
                # returns = returns.pivot('Year', 'Month', 'Returns').fillna(0)
                
                # 2.
                # findfont: Font family 'Arial' not found.
                # Unfortunatly, this second warning is not silenced by the following lines. Users can silence it manually. See:
                # https://stackoverflow.com/questions/42097053/matplotlib-cannot-find-basic-fonts  

                warnings.simplefilter("ignore")
                result = function(*args, **kwargs)
                
                return result
        return wrapper
    
    @check_quantstats
    def show_full_report(self) -> None:
        """Show basic and advanced stats and plots
        
        - Should be used in IPython notebooks
        - Shows a bunch of statistics (basic and advanced) and some plots
        - This function cannot be used in normal python (.py) files since its only
        purpose to display
        """
        return qs.reports.full(self.daily_returns) 
    
    @check_quantstats
    def get_basic_stats(self) -> pd.DataFrame:
        """Gets basic stats only.
        
        returns: Pandas DataFrame object 
        """
        return qs.reports.metrics(self.daily_returns, display=False)

    @check_quantstats
    def get_full_stats(self) -> pd.DataFrame:
        """Gets basic and advanced stats.
        
        returns: Pandas DataFrame object"""
        return qs.reports.metrics(self.daily_returns, mode='full', display=False)

    @check_quantstats
    def show_basic_plots(self) -> None:
        """Show basic plots
        
        - Should be used in IPython notebooks
        - Shows some basic plots
        - This function cannot be used in normal python (.py) files since its only
        purpose to display"""
        return qs.reports.plots(self.daily_returns)

    @check_quantstats
    def show_full_plots(self) -> None:
        """Show basic and advanced plots
        
        - Should be used in IPython notebooks
        - Shows both basic and more advanced plots
        - This function cannot be used in normal python (.py) files since its only
        purpose to display"""
        return qs.reports.plots(self.daily_returns, mode='full')


@dataclass
class TradeAnalysis:
    """Analysis of trades in a portfolio."""

    portfolio: Portfolio

    filtered_sorted_positions: list[TradingPosition] = field(init=False)

    def __post_init__(self):
        
        _filtered_positions = self.portfolio.get_all_positions_filtered()
        
        self.filtered_sorted_positions = sorted(_filtered_positions, key=lambda x: x.position_id)
        
        #assert self.filtered_sorted_positions, "No positions found"
    
    def get_first_opened_at(self) -> Optional[pd.Timestamp]:
        """Get the opened_at timestamp of the first position in the portfolio."""
        
        return min(
            position.opened_at for position in self.filtered_sorted_positions
        )

    def get_last_closed_at(self) -> Optional[pd.Timestamp]:
        """Get the closed_at timestamp of the last position in the portfolio."""
        
        return max(
            position.closed_at for position in self.filtered_sorted_positions
        )

    def get_all_positions(self) -> Iterable[Tuple[PrimaryKey, TradingPosition]]:
        """Return open and closed positions over all traded assets.
        
        Positions are sorted by position_id."""
        
        for position in self.filtered_sorted_positions:
            # pair_id, position
            yield position.pair.internal_id, position

    def get_open_positions(self) -> Iterable[Tuple[PrimaryKey, TradingPosition]]:
        """Return open positions over all traded assets.
        
        Positions are sorted by position_id."""
        
        for position in self.filtered_sorted_positions:
            if position.is_open():
                # pair_id, position
                yield position.pair.internal_id, position

    def calculate_summary_statistics(
        self, 
        time_bucket: Optional[TimeBucket] = None,
        state = None
    ) -> TradeSummary:
        """Calculate some statistics how our trades went.

            :param time_bucket:
                Optional, used to display average duration as 'number of bars' instead of 'number of days'.

            :param state:
                Optional, should be specified if user would like to see advanced statistics
            
            :return:
                TradeSummary instance
        """

        if(time_bucket is not None):
            assert isinstance(time_bucket, TimeBucket), "Not a valid time bucket"

        # for advanced statistics
        # import here to avoid circular import error
        if state is not None and HAS_QUANTSTATS:
            from tradeexecutor.visual.equity_curve import get_daily_returns
            daily_returns = get_daily_returns(state)
        else:
            daily_returns = None


        def get_avg_profit_pct_check(trades: List | None):
            return float(np.mean(trades)) if trades else None

        def get_avg_trade_duration(duration_list: List | None, time_bucket: TimeBucket | None):
            if duration_list:
                if isinstance(time_bucket, TimeBucket):
                    return pd.Timedelta(np.mean(duration_list)/time_bucket.to_timedelta())
                else:
                    return pd.Timedelta(np.mean(duration_list))
            else:
                return pd.Timedelta(datetime.timedelta(0))

        def avg(lst: list[int]):
            return sum(lst) / len(lst)
        
        def func_check(lst, func):
            return func(lst) if lst else None
        
        initial_cash = self.portfolio.get_initial_deposit()

        uninvested_cash = self.portfolio.get_current_cash()

        # EthLisbon hack
        extra_return = 0

        duration = datetime.timedelta(0)

        winning_trades = []
        losing_trades = []
        winning_trades_duration = []
        losing_trades_duration = []
        loss_risk_at_open_pc = []
        realised_losses = []
        biggest_winning_trade_pc = None
        biggest_losing_trade_pc = None
        average_duration_of_losing_trades = datetime.timedelta(0)
        average_duration_of_winning_trades = datetime.timedelta(0)

        strategy_duration = self.portfolio.get_strategy_duration()

        won = lost = zero_loss = stop_losses = take_profits = undecided = 0
        open_value: USDollarAmount = 0
        profit: USDollarAmount = 0
        trade_volume = 0
        lp_fees_paid = 0
        
        max_pos_cons = 0
        max_neg_cons = 0
        max_pullback_pct = 0
        pos_cons = 0
        neg_cons = 0
        pullback = 0
        
        winning_stop_losses = 0
        losing_stop_losses = 0

        for pair_id, position in self.get_all_positions():
            
            portfolio_value_at_open = position.portfolio_value_at_open
            
            capital_tied_at_open_pct = self.get_capital_tied_at_open(position)
            
            if position.stop_loss:
                # TODO use maximum_risk
                maximum_risk = position.get_loss_risk_at_open()
                loss_risk_at_open_pc.append(position.get_loss_risk_at_open_pct())
            else:
                maximum_risk = None
                loss_risk_at_open_pc.append(capital_tied_at_open_pct)
            
            lp_fees_paid += position.get_total_lp_fees_paid() or 0
            
            for t in position.trades.values():
                trade_volume += abs(float(t.executed_quantity) * t.executed_price)

            if position.is_open():
                open_value += position.get_value()
                undecided += 1
                continue
            
            is_stop_loss = position.is_stop_loss()
            
            if is_stop_loss:
                stop_losses += 1

            if position.is_take_profit():
                take_profits += 1

            realised_profit_percent = position.get_realised_profit_percent()
            realised_profit_usd = position.get_realised_profit_usd()
            duration = position.get_duration()
            
            if position.is_profitable():
                won += 1
                winning_trades.append(realised_profit_percent)
                winning_trades_duration.append(duration)
                
                if is_stop_loss:
                    winning_stop_losses += 1

            elif position.is_loss():
                lost += 1
                losing_trades.append(realised_profit_percent)
                losing_trades_duration.append(duration)

                if portfolio_value_at_open := position.portfolio_value_at_open:
                    realised_loss = realised_profit_usd / portfolio_value_at_open
                else:
                    # Bad data
                    realised_loss = 0
                realised_losses.append(realised_loss)
                
                if is_stop_loss:
                    losing_stop_losses += 1

            else:
                # Any profit exactly balances out loss in slippage and commission
                zero_loss += 1

            profit += realised_profit_usd
            
            
            # for getting max consecutive wins/losses and max pullback
            # don't do anything if profit = $0
            
            if(realised_profit_usd > 0):
                    neg_cons = 0
                    pullback = 0
                    pos_cons += 1
            elif(realised_profit_usd < 0):
                    pos_cons = 0
                    neg_cons += 1
                    pullback += realised_profit_usd

            if(neg_cons > max_neg_cons):
                    max_neg_cons = neg_cons
            if(pos_cons > max_pos_cons):
                    max_pos_cons = pos_cons

            if portfolio_value_at_open:
                pullback_pct = pullback / (portfolio_value_at_open + realised_profit_usd)
                if(pullback_pct < max_pullback_pct):
                        # pull back is in the negative direction
                        max_pullback_pct = pullback_pct
            else:
                # Bad input data / legacy data
                max_pullback_pct = 0

        all_trades = winning_trades + losing_trades + [0 for i in range(zero_loss)]
        average_trade = func_check(all_trades, avg)
        median_trade = func_check(all_trades, median)

        average_winning_trade_profit_pc = get_avg_profit_pct_check(winning_trades)
        average_losing_trade_loss_pc = get_avg_profit_pct_check(losing_trades)

        max_realised_loss = func_check(realised_losses, min)
        avg_realised_risk = func_check(realised_losses, avg)

        max_loss_risk_at_open_pc = func_check(loss_risk_at_open_pc, max)

        biggest_winning_trade_pc = func_check(winning_trades, max)

        biggest_losing_trade_pc = func_check(losing_trades, min)

        average_duration_of_winning_trades = get_avg_trade_duration(winning_trades_duration, time_bucket)
        average_duration_of_losing_trades = get_avg_trade_duration(losing_trades_duration, time_bucket)

        lp_fees_average_pc = lp_fees_paid / trade_volume if trade_volume else 0

        return TradeSummary(
            won=won,
            lost=lost,
            zero_loss=zero_loss,
            stop_losses=stop_losses,
            take_profits=take_profits,
            undecided=undecided,
            realised_profit=profit + extra_return,
            open_value=open_value,
            uninvested_cash=uninvested_cash,
            initial_cash=initial_cash,
            extra_return=extra_return,
            duration=strategy_duration,
            average_winning_trade_profit_pc=average_winning_trade_profit_pc,
            average_losing_trade_loss_pc=average_losing_trade_loss_pc,
            biggest_winning_trade_pc=biggest_winning_trade_pc,
            biggest_losing_trade_pc=biggest_losing_trade_pc,
            average_duration_of_winning_trades=average_duration_of_winning_trades,
            average_duration_of_losing_trades=average_duration_of_losing_trades,
            average_trade=average_trade,
            median_trade=median_trade,
            max_pos_cons=max_pos_cons,
            max_neg_cons=max_neg_cons,
            max_pullback=max_pullback_pct,
            max_loss_risk=max_loss_risk_at_open_pc,
            max_realised_loss=max_realised_loss,
            avg_realised_risk=avg_realised_risk,
            time_bucket=time_bucket,
            trade_volume=trade_volume,
            lp_fees_paid=lp_fees_paid,
            lp_fees_average_pc=lp_fees_average_pc,
            daily_returns=daily_returns,
            winning_stop_losses=winning_stop_losses,
            losing_stop_losses=losing_stop_losses,
        )

    @staticmethod
    def get_capital_tied_at_open(position):
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
            return pd.Series('background-color: Salmon', row.index)
        elif pnl_raw < 0:
            return pd.Series('background-color: LightSalmon', row.index)
        elif pnl_raw > 0.05:
            return pd.Series('background-color: LawnGreen', row.index)
        else:
            return pd.Series('background-color: PaleGreen', row.index)

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
        pair_id = position.pair.internal_id
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
            "LP fees": f"${position.get_total_lp_fees_paid():,.2f}"
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