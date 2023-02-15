"""Analyse alpha model based strategies."""
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, Dict

import pandas as pd
import plotly.graph_objects as go

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.types import USDollarPrice
from tradeexecutor.strategy.alpha_model import AlphaModel, TradingPairSignal
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, translate_trading_pair
from tradeexecutor.visual.equity_curve import calculate_equity_curve


@dataclass
class AlphaModelStatistics:
    """Summary statistics that are specific to alpha model strategies."""
    pass


def render_alpha_model_timeline(state: State, max_assets_per_portfolio: int) -> Tuple[go.Table, pd.DataFrame]:
    """Render a timeline for an alpha model for different positions it took over the time.

    - Show the alpha model state for each strategy cycle

    - Show asset columns only for the assets currently hold in the portfolio,
      to keep column number low
    """

    headers = [
        "Cycle",
        "Total equity USD",
        "Buys USD",
        "Sells USD",
    ]

    for idx in range(max_assets_per_portfolio):
        headers.append(f"Asset {idx+1}")

    rows = []

    equity_curve = calculate_equity_curve(state)

    last_equity_value = 0

    # Create a map of trades for each timestamp
    trades_by_ts = defaultdict(list)
    for t in state.portfolio.get_all_trades():
        trades_by_ts[t.strategy_cycle_at].append(t)

    for snapshot_dict in state.visualisation.calculations.values():

        snapshot: AlphaModel = AlphaModel.from_dict(snapshot_dict)
        ts = pd.Timestamp(snapshot.timestamp)
        total_equity = equity_curve[ts]

        buys = sum([t.get_value() for t in trades_by_ts[snapshot.timestamp] if t.is_buy()]) or 0.0
        sells = sum([t.get_value() for t in trades_by_ts[snapshot.timestamp] if t.is_sell()]) or 0.0

        row = [
            snapshot.timestamp.strftime("%Y-%m-%d"),
            f"{total_equity:,.2f}",
            f"{buys:,.2}",
            f"{sells:,.2f}",
        ]

        # What assets we had going into the portfolio on this cycle
        assets = snapshot.get_signals_sorted_by_weight()
        assets_hold = 0
        for idx, signal in enumerate(assets):
            # Get only what we hold
            if signal.normalised_weight > 0:
                base_token = signal.pair.base.token_symbol
                weight = signal.normalised_weight
                target = signal.position_target
                row.append(f"{base_token}\n"
                           f"{weight * 100:.4f}%\n"
                           f"Target:{target:,.2f} USD\n"
                           )
                assets_hold += 1

        for i in range(assets_hold, max_assets_per_portfolio):
            # Pad empty cells if less than max assets hold
            row.append("")

        rows.append(row)
        last_equity_value = total_equity

    df = pd.DataFrame(rows, columns=headers)

    table = go.Table(
        header=dict(values=list(df.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=df.transpose().values.tolist(),
                   fill_color='lavender',
                   align='left'))

    return table, df



def create_alpha_model_timeline_all_assets(
        state: State,
        universe: TradingStrategyUniverse,
        new_line="\n",
) -> pd.DataFrame:
    """Render a timeline for an alpha model for different positions it took over the time.

    - Show the alpha model state for each strategy cycle

    - Show asset column for each asset in the trading universe, making
      this very wide table

    :return:
        DataFrame where each row is a strategy cycle
        and contains textual information about asset rebalances during the cycle.

    """

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
    previous_prices_by_pair: Dict[TradingPairIdentifier, USDollarPrice] = {}

    for snapshot_dict in state.visualisation.calculations.values():

        snapshot: AlphaModel = AlphaModel.from_dict(snapshot_dict)
        ts = pd.Timestamp(snapshot.timestamp)

        # Get the total value of the portfolio from the equity curve
        total_equity = equity_curve[ts]

        row = [
            snapshot.timestamp.strftime("%Y-%m-%d"),
            f"${total_equity:,.0f}"
        ]

        # Update the cell for each asset during this strategy cycle
        for idx, pair in enumerate(pair_universe.iterate_pairs()):

            pair = translate_trading_pair(pair)

            signal = snapshot.get_signal_by_pair_id(pair.internal_id)
            if signal is None:
                # This pair did not see any trades action during this strategy cycle
                row.append("-")
                # Clear row-over-row book keeping
                if pair in previous_cycle:
                    del previous_cycle[pair]
                if pair in previous_prices_by_pair:
                    del previous_prices_by_pair[pair]
                continue

            if not signal.position_adjust_ignored:
                assert signal.position_id, f"No position info: {signal}, cycle {ts}"
                position = portfolio.get_position_by_id(signal.position_id)
                trades = list(position.get_trades_by_strategy_cycle(snapshot.timestamp))
                assert len(trades) == 1, f"Unexpected trades: {trades}"
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
            text = ""
            if price:
                if previous_price:
                    price_change = (price - previous_price) / previous_price

                if price_change is not None:
                    if price_change > 0:
                        text += f"🌲 {price_change * 100:.2f}% {new_line}"
                    else:
                        text += f"🔻 {price_change * 100:.2f}% {new_line}"

                text += f"P:{price:,.2f}{new_line}"\
                       f"W:{weight * 100:.0f}% {new_line}" \
                       f"{new_line}"
            else:
                text = f"W: {weight * 100:.0f}% {new_line}" \
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

        rows.append(row)

    df = pd.DataFrame(rows, columns=headers)
    return df


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