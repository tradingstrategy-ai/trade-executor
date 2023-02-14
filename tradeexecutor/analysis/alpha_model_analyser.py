"""Analyse alpha model based strategies."""
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple

import pandas as pd
import plotly.graph_objects as go

from tradeexecutor.state.state import State
from tradeexecutor.strategy.alpha_model import AlphaModel
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
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

    # https://plotly.com/python/table/
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

        buys = sum([t.get_value() for t in trades_by_ts[snapshot.timestamp] if t.is_buy()])
        sells = sum([t.get_value() for t in trades_by_ts[snapshot.timestamp] if t.is_sell()])

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



def render_alpha_model_timeline_all_assets(state: State, universe: TradingStrategyUniverse) -> Tuple[go.Table, pd.DataFrame]:
    """Render a timeline for an alpha model for different positions it took over the time.

    - Show the alpha model state for each strategy cycle

    - Show asset column for each asset in the trading universe, making
      this very wide table
    """

    # https://plotly.com/python/table/
    headers = [
        "Cycle",
        "Total equity USD",
    ]

    for pair in universe.universe.pairs.iterate_pairs():
        headers.append(f"{pair.base_token_symbol}")

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

        buys = sum([t.get_value() for t in trades_by_ts[snapshot.timestamp] if t.is_buy()])
        sells = sum([t.get_value() for t in trades_by_ts[snapshot.timestamp] if t.is_sell()])

        row = [
            snapshot.timestamp.strftime("%Y-%m-%d"),
            f"{total_equity:,.2f}",
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




