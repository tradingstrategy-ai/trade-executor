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
import enum
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Iterable, Optional, Tuple, Callable, Set

import numpy as np
import pandas as pd
from dataclasses_json import dataclass_json, Exclude, config

from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.trade import TradeExecution, TradeType
from tradeexecutor.utils.format import calculate_percentage
from tradeexecutor.utils.timestamp import json_encode_timedelta, json_decode_timedelta

from tradingstrategy.exchange import Exchange
from tradingstrategy.pair import PandasPairUniverse
from tradingstrategy.types import PrimaryKey, USDollarAmount
from tradingstrategy.utils.format import format_value, format_price, format_duration_days_hours_mins, \
    format_percent_2_decimals
from tradingstrategy.utils.summarydataframe import as_dollar, as_integer, create_summary_table, as_percent, as_duration

logger = logging.getLogger(__name__)


@dataclass
class SpotTrade:
    """Track spot trades to construct position performance.

    For sells, quantity is negative.
    """

    #: Internal running counter to uniquely label all trades in trade analysis
    trade_id: PrimaryKey

    #: Trading pair for this trade
    pair_id: PrimaryKey

    #: When this trade was made, the backtes simulation thick
    timestamp: pd.Timestamp

    #: Asset price at buy in
    price: USDollarAmount

    #: How much we bought the asset. Negative value for sells.
    quantity: float

    #: How much fees we paid to the exchange
    commission: USDollarAmount

    #: How much we lost against the midprice due to the slippage
    slippage: USDollarAmount

    #: Any hints applied for this trade why it was performed
    trade_type: Optional[TradeType] = None

    #: Internal state dump of the algorithm when this trade was made.
    #: This is mostly useful when doing the trade analysis try to understand
    #: why some trades were made.
    #: It also allows you to reconstruct the portfolio state over the time.
    state_details: Optional[Dict] = None

    def is_buy(self):
        return self.quantity > 0

    def is_sell(self):
        return self.quantity < 0

    @property
    def value(self) -> USDollarAmount:
        return abs(self.price * float(self.quantity))


@dataclass
class TradePosition:
    """How a particular asset traded.

    Each asset can have multiple entries (buys) and exits (sells)

    For a simple strategies there can be only one or two trades per position.

    * Enter (buy)

    * Exit (sell optionally)
    """

    #: List of all trades done for this position
    trades: List[SpotTrade] = field(default_factory=list)

    #: Closing the position could be deducted from the trades themselves,
    #: but we cache it by hand to speed up processing
    opened_at: Optional[pd.Timestamp] = None

    #: Closing the position could be deducted from the trades themselves,
    #: but we cache it by hand to speed up processing
    closed_at: Optional[pd.Timestamp] = None

    def __eq__(self, other: "TradePosition"):
        """Trade positions are unique by opening timestamp and pair id.]

        We assume there cannot be a position opened for the same asset at the same time twice.
        """
        return self.position_id == other.position_id

    def __hash__(self):
        """Allows easily create index (hash map) of all positions"""
        return hash((self.position_id))

    @property
    def position_id(self) -> PrimaryKey:
        """Position id is the same as the opening trade id."""
        return self.trades[0].trade_id

    @property
    def pair_id(self) -> PrimaryKey:
        """Position id is the same as the opening trade id."""
        return self.trades[0].pair_id

    @property
    def duration(self) -> Optional[datetime.timedelta]:
        """How long this position was held.

        :return: None if the position is still open
        """
        if not self.is_closed():
            return None
        return self.closed_at - self.opened_at

    def is_open(self):
        return self.closed_at is None

    def is_closed(self):
        return not self.is_open()

    @property
    def open_quantity(self) -> float:
        return sum([t.quantity for t in self.trades])

    @property
    def open_value(self) -> float:
        """The current value of this open position, with the price at the time of opening."""
        assert self.is_open()
        return sum([t.value for t in self.trades])

    @property
    def open_price(self) -> float:
        """At what price we opened this position.

        Supports only simple enter/exit positions.
        """
        return self.get_first_entry_price()

    def get_first_entry_price(self) -> float:
        """What was the price when the first entry buy for this position was made.
        """
        buys = list(self.buys)
        return buys[0].price

    def get_last_exit_price(self) -> float:
        """What was the time when the last sell for this position was executd.
        """
        sells = list(self.sells)
        return sells[-1].price
        assert len(sells) == 1

    @property
    def close_price(self) -> float:
        """At what price we exited this position.

        Supports only simple enter/exit positions.
        """
        return self.get_last_exit_price()

    @property
    def buys(self) -> Iterable[SpotTrade]:
        return [t for t in self.trades if t.is_buy()]

    @property
    def sells(self) -> Iterable[SpotTrade]:
        return [t for t in self.trades if t.is_sell()]

    @property
    def buy_value(self) -> USDollarAmount:
        return sum([t.value - t.commission for t in self.trades if t.is_buy()])

    @property
    def sell_value(self) -> USDollarAmount:
        return sum([t.value - t.commission for t in self.trades if t.is_sell()])

    @property
    def realised_profit(self) -> USDollarAmount:
        """Calculated life-time profit over this position."""
        assert not self.is_open()
        return -sum([float(t.quantity) * t.price - t.commission for t in self.trades])

    @property
    def realised_profit_percent(self) -> float:
        """Calculated life-time profit over this position."""
        assert not self.is_open()
        buy_value = self.buy_value
        sell_value = self.sell_value
        return sell_value / buy_value - 1

    def is_win(self):
        """Did we win this trade."""
        assert not self.is_open()
        return self.realised_profit > 0

    def is_lose(self):
        assert not self.is_open()
        return self.realised_profit < 0

    def is_stop_loss(self) -> bool:
        """Was stop loss triggered for this position"""
        for t in self.trades:
            if t.trade_type == TradeType.stop_loss:
                return True
        return False

    def is_take_profit(self) -> bool:
        """Was trake profit triggered for this position"""
        for t in self.trades:
            if t.trade_type == TradeType.take_profit:
                return True
        return False

    def add_trade(self, t: SpotTrade):
        if self.trades:
            last_trade = self.trades[-1]
            assert t.timestamp >= last_trade.timestamp, f"Tried to do trades in wrong order. Last: {last_trade}, got {t}"
        self.trades.append(t)

    def can_trade_close_position(self, t: SpotTrade):
        assert self.is_open()
        if not t.is_sell():
            return False
        open_quantity = self.open_quantity
        closing_quantity = -t.quantity
        assert closing_quantity <= open_quantity, "Cannot sell more than we have in balance sheet"
        return closing_quantity == open_quantity

    def get_max_size(self) -> USDollarAmount:
        """Get the largest size of this position over the time"""
        cur_size = 0
        max_size = 0

        if len(self.trades) > 2:
            logger.warning("Position has %d trades so this method might produce wrong result")

        for t in self.trades:
            cur_size = t.value
            max_size = max(cur_size, max_size)
        return max_size

    def get_trade_count(self) -> int:
        """How many individual trades was done to manage this position."""
        return len(self.trades)


@dataclass
class AssetTradeHistory:
    """How a particular asset traded.

    Each position can have increments or decrements.
    When position is decreased to zero, it is considered closed, and a new buy open a new position.
    """
    positions: List[TradePosition] = field(default_factory=list)

    def get_first_opened_at(self) -> Optional[pd.Timestamp]:
        if self.positions:
            return self.positions[0].opened_at
        return None

    def get_last_closed_at(self) -> Optional[pd.Timestamp]:
        for position in reversed(self.positions):
            if not position.is_open():
                return position.closed_at

        return None

    def add_trade(self, t: SpotTrade):
        """Adds a new trade to the asset history.

        If there is an open position the trade is added against this,
        otherwise a new position is opened for tracking.
        """
        current_position = None
        if self.positions:
            if self.positions[-1].is_open():
                current_position = self.positions[-1]

        if current_position:
            if current_position.can_trade_close_position(t):
                # Close the existing position
                current_position.closed_at = t.timestamp
                current_position.add_trade(t)
                assert current_position.open_quantity == 0
            else:
                # Add to the existing position
                current_position.add_trade(t)
        else:
            # Open new position
            new_position = TradePosition(opened_at=t.timestamp)
            new_position.add_trade(t)
            self.positions.append(new_position)


@dataclass_json
@dataclass
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

    average_winning_trade_profit_pc: float
    average_losing_trade_loss_pc: float
    biggest_winning_trade_pc: Optional[float]
    biggest_losing_trade_pc: Optional[float]

    average_duration_of_winning_trades: datetime.timedelta = field(metadata=config(
        encoder=json_encode_timedelta,
        decoder=json_decode_timedelta,
    ))
    average_duration_of_losing_trades: datetime.timedelta = field(metadata=config(
        encoder=json_encode_timedelta,
        decoder=json_decode_timedelta,
    ))

    total_trades: int = field(init=False)
    win_percent: float = field(init=False)
    return_percent: float = field(init=False)
    annualised_return_percent: float = field(init=False)
    all_stop_loss_percent: float = field(init=False)
    lost_stop_loss_percent: float = field(init=False)
    average_net_profit: USDollarAmount = field(init=False)
    end_value: USDollarAmount = field(init=False)

    def __post_init__(self):

        self.total_trades = self.won + self.lost + self.zero_loss
        self.win_percent = calculate_percentage(self.won, self.total_trades)
        self.return_percent = calculate_percentage(self.realised_profit, self.initial_cash)
        self.annualised_return_percent = calculate_percentage(self.return_percent * datetime.timedelta(days=365),
                                                              self.duration) if self.return_percent else None
        self.all_stop_loss_percent = calculate_percentage(self.stop_losses, self.total_trades)
        self.lost_stop_loss_percent = calculate_percentage(self.stop_losses, self.lost)
        self.average_net_profit = self.realised_profit / self.total_trades if self.total_trades else None
        self.end_value = self.open_value + self.uninvested_cash

    def to_dataframe(self) -> pd.DataFrame:
        """Creates a human-readable Pandas dataframe table from the object."""
        human_data = {
            "Trading period length": as_duration(self.duration),
            "Return %": as_percent(self.return_percent),
            "Annualised return %": as_percent(self.annualised_return_percent),
            "Cash at start": as_dollar(self.initial_cash),
            "Value at end": as_dollar(self.end_value),
            "Trade win percent": as_percent(self.win_percent),
            "Total trades done": as_integer(self.total_trades),
            "Won trades": as_integer(self.won),
            "Lost trades": as_integer(self.lost),
            "Stop losses triggered": as_integer(self.stop_losses),
            "Stop loss % of all": as_percent(self.all_stop_loss_percent),
            "Stop loss % of lost": as_percent(self.lost_stop_loss_percent),
            "Zero profit trades": as_integer(self.zero_loss),
            "Positions open at the end": as_integer(self.undecided),
            "Realised profit and loss": as_dollar(self.realised_profit),
            "Portfolio unrealised value": as_dollar(self.open_value),
            "Extra returns on lending pool interest": as_dollar(self.extra_return),
            "Cash left at the end": as_dollar(self.uninvested_cash),
            "Average winning trade profit %": as_percent(self.average_winning_trade_profit_pc),
            "Average losing trade loss %": as_percent(self.average_losing_trade_loss_pc),
            "Biggest winning trade %": as_percent(self.biggest_winning_trade_pc),
            "Biggest losing trade %": as_percent(self.biggest_losing_trade_pc),
            "Average duration of winning trades": as_duration(self.average_duration_of_winning_trades),
            "Average duration of losing trades": as_duration(self.average_duration_of_losing_trades)
        }
        return create_summary_table(human_data)


@dataclass
class TradeAnalysis:
    """Analysis of trades in a portfolio."""

    portfolio: Portfolio

    #: How a particular asset traded. Asset id -> Asset history mapping
    asset_histories: Dict[object, AssetTradeHistory] = field(default_factory=dict)

    def get_first_opened_at(self) -> Optional[pd.Timestamp]:
        def all_opens():
            for history in self.asset_histories.values():
                yield history.get_first_opened_at()

        return min(all_opens())

    def get_last_closed_at(self) -> Optional[pd.Timestamp]:
        def all_closes():
            for history in self.asset_histories.values():
                closed = history.get_last_closed_at()
                if closed:
                    yield closed

        return max(all_closes())

    def get_all_positions(self) -> Iterable[Tuple[PrimaryKey, TradePosition]]:
        """Return open and closed positions over all traded assets."""
        for pair_id, history in self.asset_histories.items():
            for position in history.positions:
                yield pair_id, position

    def get_open_positions(self) -> Iterable[Tuple[PrimaryKey, TradePosition]]:
        """Return open and closed positions over all traded assets."""
        for pair_id, history in self.asset_histories.items():
            for position in history.positions:
                if position.is_open():
                    yield pair_id, position

    def calculate_summary_statistics(self) -> TradeSummary:
        """Calculate some statistics how our trades went."""

        initial_cash = self.portfolio.get_initial_deposit()

        uninvested_cash = self.portfolio.get_current_cash()

        # EthLisbon hack
        extra_return = 0

        duration = datetime.timedelta(0)

        winning_trades = []
        losing_trades = []
        winning_trades_duration = []
        losing_trades_duration = []
        biggest_winning_trade_pc = None
        biggest_losing_trade_pc = None
        average_duration_of_losing_trades = datetime.timedelta(0)
        average_duration_of_winning_trades = datetime.timedelta(0)

        first_trade, last_trade = self.portfolio.get_first_and_last_executed_trade()
        if first_trade and first_trade != last_trade:
            duration = last_trade.executed_at - first_trade.executed_at

        won = lost = zero_loss = stop_losses = undecided = 0
        open_value: USDollarAmount = 0
        profit: USDollarAmount = 0
        for pair_id, position in self.get_all_positions():
            if position.is_open():
                open_value += position.open_value
                undecided += 1
                continue

            if position.is_stop_loss():
                stop_losses += 1

            if position.is_win():
                won += 1
                winning_trades.append(position.realised_profit_percent)
                winning_trades_duration.append(position.duration)


            elif position.is_lose():
                lost += 1
                losing_trades.append(position.realised_profit_percent)
                losing_trades_duration.append(position.duration)

            else:
                # Any profit exactly balances out loss in slippage and commission
                zero_loss += 1

            profit += position.realised_profit

        if len(winning_trades) > 0:
            average_winning_trade_profit_pc = float(np.mean(winning_trades))
        else:
            average_winning_trade_profit_pc = 0

        if len(losing_trades) > 0:
            average_losing_trade_loss_pc = float(np.mean(losing_trades))
        else:
            average_losing_trade_loss_pc = 0

        if winning_trades:
            biggest_winning_trade_pc = max(winning_trades)

        if losing_trades:
            biggest_losing_trade_pc = min(losing_trades)

        if winning_trades_duration:
            average_duration_of_winning_trades = np.mean(winning_trades_duration).to_pytimedelta()

        if losing_trades_duration:
            average_duration_of_losing_trades = np.mean(losing_trades_duration).to_pytimedelta()

        return TradeSummary(
            won=won,
            lost=lost,
            zero_loss=zero_loss,
            stop_losses=stop_losses,
            undecided=undecided,
            realised_profit=profit + extra_return,
            open_value=open_value,
            uninvested_cash=uninvested_cash,
            initial_cash=initial_cash,
            extra_return=extra_return,
            duration=duration,
            average_winning_trade_profit_pc=average_winning_trade_profit_pc,
            average_losing_trade_loss_pc=average_losing_trade_loss_pc,
            biggest_winning_trade_pc=biggest_winning_trade_pc,
            biggest_losing_trade_pc=biggest_losing_trade_pc,
            average_duration_of_winning_trades=average_duration_of_winning_trades,
            average_duration_of_losing_trades=average_duration_of_losing_trades,
        )

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
        position: TradePosition = row["position"]
        # timestamp = row.name  # ???
        pair_id = position.pair_id
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

        r = {
            # "timestamp": timestamp,
            "Id": position.position_id,
            "Remarks": remarks,
            "Opened at": position.opened_at.strftime(timestamp_format),
            "Duration": format_duration_days_hours_mins(position.duration) if position.duration else np.nan,
            "Exchange": exchange.name,
            "Base asset": pair_info.base_token_symbol,
            "Quote asset": pair_info.quote_token_symbol,
            "Position max size": format_value(position.get_max_size()),
            "PnL USD": format_value(position.realised_profit) if position.is_closed() else np.nan,
            "PnL %": format_percent_2_decimals(position.realised_profit_percent) if position.is_closed() else np.nan,
            "PnL % raw": position.realised_profit_percent if position.is_closed() else 0,
            "Open price USD": format_price(position.open_price),
            "Close price USD": format_price(position.close_price) if position.is_closed() else np.nan,
            "Trade count": position.get_trade_count(),
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


def build_trade_analysis(portfolio: Portfolio) -> TradeAnalysis:
    """Build a trade analysis from list of positions.

    - Read positions from backtesting or live state

    - Create TradeAnalysis instance that can be used to display Jupyter notebook
      data on the performance
    """

    histories = {}

    positions = list(portfolio.get_all_positions())

    # Sort positions based on their id
    # because open, closed and frozen positions might be in a mixed order
    positions = sorted(positions, key=lambda p: p.position_id)

    # Each Backtrader Trade instance presents a position
    # Trade instances contain TradeHistory entries that present change to this position
    # with Order instances attached
    for position in positions:

        pair = position.pair
        pair_id = pair.internal_id
        assert type(pair_id) == int

        trade: TradeExecution

        trades = list(position.trades.values())

        for trade in trades:

            history = histories.get(pair_id)
            if not history:
                history = histories[pair_id] = AssetTradeHistory()

            # filter out failed trade
            if trade.executed_at is None:
                continue

            # Internally negative quantities are for sells
            quantity = trade.executed_quantity
            timestamp = pd.Timestamp(trade.executed_at)
            price = trade.executed_price

            # print("Got event", event, status)
            assert quantity != 0, f"Got bad quantity for {trade}"
            # import ipdb ; ipdb.set_trace()
            assert price > 0, f"Got invalid trade {trade}"

            spot_trade = SpotTrade(
                pair_id=pair_id,
                trade_id=trade.trade_id,
                timestamp=timestamp,
                price=price,
                quantity=quantity,
                commission=0,
                slippage=0,  # TODO
                trade_type=trade.trade_type,
            )
            history.add_trade(spot_trade)

    return TradeAnalysis(portfolio, asset_histories=histories)
