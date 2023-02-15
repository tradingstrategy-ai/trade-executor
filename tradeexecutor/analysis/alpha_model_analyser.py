"""Analyse alpha model based strategies."""
import datetime
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, Dict, List

import pandas as pd
import plotly.graph_objects as go

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.types import USDollarPrice
from tradeexecutor.strategy.alpha_model import AlphaModel, TradingPairSignal
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, translate_trading_pair
from tradeexecutor.visual.equity_curve import calculate_equity_curve


def construct_event_timeline(state: State) -> pd.DatetimeIndex:
    """Create timestamps of the events needed for alpha model analysis.

    Event can be caused by

    - Rebalances

    - Stop loss triggers (happen outside the normal rebalance time period)
    """

    # UNIX Timestamps of rebalance events
    rebalance_timestamps = pd.to_datetime(
        list(state.visualisation.calculations.keys()),
        unit="s",
    )

    # Datetimes of stop loss trade opens
    triggered_timestamps = pd.to_datetime(
        [t.opened_at for t in state.portfolio.get_all_trades() if t.is_triggered()]
    )

    # https://stackoverflow.com/a/55696161/315168
    return rebalance_timestamps.union(triggered_timestamps)



def render_alpha_model_plotly_table(df: pd.DataFrame) -> Tuple[go.Figure, go.Table]:
    """Render alpha model analysis as a plotly figure."""

    # https://plotly.com/python/table/

    table = go.Table(
        header=dict(values=list(df.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=df.transpose().values.tolist(),
                   fill_color='lavender',
                   font={"family": "monospace"},
                   align='left'),
    )

    fig = go.Figure(data=table)

    return fig, table


def create_alpha_model_timeline_all_assets(
        state: State,
        universe: TradingStrategyUniverse,
        new_line="\n",
) -> pd.DataFrame:
    """Render a timeline for an alpha model for different positions it took over the time.

    - Show the alpha model state for each strategy cycle

    - Show asset column for each asset in the trading universe, making
      this very wide table

    :param new_line:
        The new line marker used in the table cells.

        Defaults to Plotly table compatible newlines.

    :return:
        DataFrame where each row is a strategy cycle
        and contains textual information about asset rebalances during the cycle.

    """

    timeline = construct_event_timeline(state)

    # https://plotly.com/python/table/
    headers = [
        "Cycle",
        "Equity",
    ]

    for pair in universe.universe.pairs.iterate_pairs():
        headers.append(f"{pair.base_token_symbol}")

    rows = []

    pair_universe = universe.universe.pairs

    equity_curve = calculate_equity_curve(state)

    portfolio = state.portfolio

    previous_cycle: Dict[TradingPairIdentifier, TradingPairSignal] = {}
    previous_positions_by_pair: Dict[TradingPairIdentifier, TradingPosition] = {}
    previous_prices_by_pair: Dict[TradingPairIdentifier, USDollarPrice] = {}

    # Build a map of stop loss trades triggered by a timestamp by a trading pair
    trigger_trade_map: Dict[datetime.datetime, Dict[TradingPairIdentifier, TradeExecution]] = defaultdict(dict)
    for t in state.portfolio.get_all_trades():
        if t.is_triggered():
            trigger_trade_map[t.opened_at][t.pair] = t

    event_ts: pd.Timestamp
    for event_ts in timeline:

        # We do not have intraminute action,
        # so if the seconds are set then something wrong with the data
        assert event_ts.second == 0

        # Check if got some stop loss trades for this timestamp
        # If both stop loss and rebalance happen at the same timestamp,
        # write out stop loss row first
        trigger_trades = trigger_trade_map[event_ts.to_pydatetime()]
        if len(trigger_trades) > 0:
            row = [
                event_ts.strftime("%Y-%m-%d %H:%M"),
                "",
            ]
            for idx, pair in enumerate(pair_universe.iterate_pairs()):
                pair = translate_trading_pair(pair)
                trade = trigger_trades.get(pair)
                text = ""
                if trade:
                    position = state.portfolio.get_position_by_id(trade.position_id)
                    profit = position.get_total_profit_usd()
                    profit_pct = position.get_total_profit_percent()
                    if t.is_stop_loss():
                        text += f"🛑 Stop loss{new_line}"
                        text += f"{new_line}"
                        text += f"Price: ${trade.planned_mid_price:,.4f}{new_line}"
                        text += f"{new_line}"
                        text += f"Loss: ${profit:,.0f} ({profit_pct * 100:.2f}%)"
                    else:
                        text += f"⭐️ Take Profit{new_line}"
                        text += f"{new_line}"
                        text += f"Price: ${trade.planned_mid_price:,.4f}{new_line}"
                        text += f"{new_line}"
                        profit = position.get_total_profit_usd()
                        text += f"Profit: ${profit:,.0f} ({profit_pct * 100:.2f}%)"

                row.append(text)
            rows.append(row)

        # Generate rebalance trades row for this timestamp
        snapshot_data = state.visualisation.calculations.get(int(event_ts.timestamp()))
        if snapshot_data:
            snapshot: AlphaModel = AlphaModel.from_dict(snapshot_data)
            ts = pd.Timestamp(snapshot.timestamp)

            # Get the total value of the portfolio from the equity curve
            total_equity = equity_curve[ts]

            row = [
                event_ts.strftime("%Y-%m-%d"),
                f"${total_equity:,.0f}"
            ]

            # Update the cell for each asset during this strategy cycle
            for idx, pair in enumerate(pair_universe.iterate_pairs()):

                pair = translate_trading_pair(pair)

                text = ""

                signal = snapshot.get_signal_by_pair_id(pair.internal_id)
                if signal is None:
                    # This pair did not see any trades action during this strategy cycle
                    row.append(text)

                    # Clear row-over-row book keeping
                    if pair in previous_cycle:
                        del previous_cycle[pair]
                    if pair in previous_prices_by_pair:
                        del previous_prices_by_pair[pair]
                    if pair in previous_positions_by_pair:
                        del previous_positions_by_pair[pair]

                    continue

                if not signal.position_adjust_ignored:
                    assert signal.position_id, f"No position info: {signal}, cycle {ts}"
                    position = portfolio.get_position_by_id(signal.position_id)
                    trades = list(position.get_trades_by_strategy_cycle(snapshot.timestamp))
                    # We may get more than 1 trade if take profit/stopp loss was triggered on the same cycle
                    assert len(trades) >= 1, f"No trades for signal found when there should have been"
                    trade = trades[0]
                    quantity = trade.executed_quantity
                    price = trade.executed_price
                    value = trade.get_value()
                    if quantity < 0:
                        value = -value
                else:
                    # Signal was given, but no trades where executed
                    # because any rebalance value was amount the min threshold
                    trade = None
                    quantity = 0
                    price = 0.0
                    value = 0

                weight = signal.normalised_weight

                previous_price = previous_prices_by_pair.get(pair)
                price_change = None

                if price:
                    if previous_price:
                        price_change = (price - previous_price) / previous_price

                    if price_change is not None:
                        if price_change > 0:
                            text += f"🌲 {price_change * 100:.2f}% {new_line}"
                        else:
                            text += f"🔻 {price_change * 100:.2f}% {new_line}"

                    text += f"Price: ${price:,.4f}{new_line}"\

                text += f"Weight: {weight * 100:.0f}% {new_line}" \
                        f"Signal: {signal.signal * 100:.0f}% {new_line}" \
                        f"{new_line}"

                # Get only what we hold
                if signal.has_trades():
                    if previous_cycle.get(pair) is None:
                        # Open position
                        text += f"Open: ${value:,.0f}"
                    else:
                        # Either adjust position or close
                        if trade:
                            if trade == position.get_last_trade():
                                # Close
                                text += f"Close: ${value:,.2f}{new_line}"
                            else:
                                # Adjust - buy or sell
                                text += f"Adjust: ${value:+,.0f}{new_line}"
                else:
                    #  Hold - no trades for this position
                    text += f"Hold{new_line}"

                profit = signal.profit_before_trades
                if profit:
                    text += f"Profit: ${profit:,.0f}"

                row.append(text)

                # Record previous price and signal so we can compare the change
                previous_cycle[pair] = signal
                if price is not None:
                    previous_prices_by_pair[pair] = price
                previous_positions_by_pair[pair] = position

            rows.append(row)

    df = pd.DataFrame(rows, columns=headers)
    return df