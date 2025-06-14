"""Single position visualisation."""

import logging
import datetime

import numpy as np
import pandas as pd
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots

from tradeexecutor.state.position import TradingPosition
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradingstrategy.timebucket import TimeBucket

logger = logging.getLogger(__name__)


import pandas as pd
from collections import deque




def calculate_pnl(df):
    fifo = deque()  # FIFO: stores (quantity, entry_price)
    total_realised = 0.0
    realised_cost_basis = 0.0  # Total cost of closed positions

    realised_pnl_list = []
    unrealised_pnl_list = []
    realised_pct_list = []
    unrealised_pct_list = []
    total_pct_list = []

    for _, row in df.iterrows():
        delta = row['delta']
        price = row['executed_price']
        mark_price = row['mark_price']

        if delta > 0:
            # Buy: add to FIFO
            fifo.append((delta, price))

        elif delta < 0:
            # Sell: realise PnL from oldest lots
            remaining = -delta
            realised_pnl = 0.0
            cost_basis = 0.0

            while remaining > 0 and fifo:
                lot_qty, lot_price = fifo[0]
                matched_qty = min(remaining, lot_qty)

                cost_basis += matched_qty * lot_price
                realised_pnl += matched_qty * (price - lot_price)

                remaining -= matched_qty
                lot_qty -= matched_qty

                if lot_qty == 0:
                    fifo.popleft()
                else:
                    fifo[0] = (lot_qty, lot_price)

            total_realised += realised_pnl
            realised_cost_basis += cost_basis

        # Unrealised PnL and cost
        unrealised_pnl = sum(qty * (mark_price - entry_price) for qty, entry_price in fifo)
        unrealised_cost = sum(qty * entry_price for qty, entry_price in fifo)

        # Append all values
        realised_pnl_list.append(total_realised)
        unrealised_pnl_list.append(unrealised_pnl)

        realised_pct_list.append(
            total_realised / realised_cost_basis if realised_cost_basis != 0 else 0
        )
        unrealised_pct_list.append(
            unrealised_pnl / unrealised_cost if unrealised_cost != 0 else 0
        )

        total_cost_basis = realised_cost_basis + unrealised_cost
        total_pct = (total_realised + unrealised_pnl) / total_cost_basis if total_cost_basis != 0 else 0
        total_pct_list.append(total_pct)

    df['realised_pnl'] = realised_pnl_list
    df['unrealised_pnl'] = unrealised_pnl_list
    df['realised_pnl_pct'] = realised_pct_list
    df['unrealised_pnl_pct'] = unrealised_pct_list
    df['total_pnl_pct'] = total_pct_list

    return df


def calculate_position_timeline(
    strategy_universe: TradingStrategyUniverse,
    position: TradingPosition,
    end_at: datetime.datetime | None = None,
) -> pd.DataFrame:
    """Calculatea visualisation dato for a single position.

    - Price
    - Position size
    - PnL

    TODO: Missing vault perf and managemetn fees.

    Example data:

    .. code-block:: text

                                 quantity      delta  executed_price  fee  cumulative_cost  avg_price  mark_price         value  realised_delta  realised_pnl  unrealised_pnl        pnl
        timestamp
        2025-03-22 00:00:00     51.894608  51.894608        1.002456  0.0        52.022051   0.000000    1.002456     52.022051             0.0      0.000000         0.00000   0.000000
        2025-03-22 01:00:00     51.894608   0.000000        1.002456  0.0        52.022051   0.000000    1.002456     52.022051             0.0      0.000000         0.00000   0.000000
        2025-03-22 02:00:00     51.894608   0.000000        1.002456  0.0        52.022051   0.000000    1.002456     52.022051             0.0      0.000000         0.00000   0.000000
        2025-03-22 03:00:00     51.894608   0.000000        1.002456  0.0        52.022051   0.000000    1.002456     52.022051             0.0      0.000000         0.00000   0.000000

    :param end_at:
        End at timestamp for positions that are still open/were open at the end of the backtest.

    :return:
        DataFrame with columns as above,
    """

    assert isinstance(strategy_universe, TradingStrategyUniverse), f"Expected TradingStrategyUniverse, got {type(strategy_universe)}"
    assert isinstance(position, TradingPosition), f"Expected TradingPosition, got {type(position)}"

    assert position.is_spot() or position.is_vault(), "Interest calculations missing"
    assert not position.is_short(), "This function has not been tested for short positions yet"

    start = position.opened_at
    end = position.closed_at or end_at
    time_bucket = strategy_universe.data_universe.time_bucket

    assert time_bucket, "Universe missing the candle time bucket"
    assert end is not None, f"Position must have an end date or be closed at the end of the backtest: {position}"

    start = time_bucket.floor(pd.Timestamp(start))
    end = time_bucket.ceil(pd.Timestamp(end))
    pair = position.pair

    candles = strategy_universe.data_universe.candles.get_samples_by_pair(pair.internal_id)
    # TODO: index type may vary here,because get_samples_by_pair() gives (pair_id, timestamp) as index
    # use slow filtering method

    candles = candles.loc[(candles["timestamp"] >= start) & (candles["timestamp"] <= end)]

    assert len(candles) > 0, f"No candles found for {pair.internal_id} between {start} and {end} when plotting position {position}"

    entries = []

    cumulative_quantity = cumulative_cost = cumulative_value = avg_price = 0

    for trade in position.trades.values():
        delta = float(trade.executed_quantity)

        if delta > 0:
            # Buy: increase cost basis
            value = trade.get_value()
            cumulative_cost += value
            cumulative_value += value
            cumulative_quantity += delta
            avg_price = cumulative_value / float(cumulative_quantity)
        elif delta < 0:
            # Sell: reduce cost basis proportionally
            if cumulative_quantity > 0:
                cost_reduction = avg_price * abs(delta)
                cumulative_cost -= cost_reduction
                cumulative_quantity += delta  # delta is negative here
        else:
            raise NotImplementedError(f"Got a trade withe executed quantity zero: {trade}")

        entry = {
            "timestamp": trade.opened_at,  # When we made a trade
            "quantity": cumulative_quantity,  # Currently hold quantity if assetes
            "delta": delta,  # Positive if we bought, negative if we sold
            "executed_price": trade.executed_price,  # Price we got, fees included
            "fee": trade.lp_fees_paid,  # How much fees we paid
            "cumulative_cost": cumulative_cost,  # The cost of maintaining this position
            "cumulative_value": cumulative_value,  # How much this position is worth now
            "avg_price": avg_price,  # Volume-weighted average price of the position
        }
        entries.append(entry)

    if not position.closed_at:
        # Mark the unrealised position at the end of the backtest.
        # Generate timestamp entry at the end of the backtest,
        # so we can calculate unrealised PnL for this row
        entry = {
            "timestamp": end_at,
            "quantity": cumulative_quantity,
            "delta": 0,
            "executed_price": np.nan,
            "fee": 0,
            "avg_price": avg_price
        }
        entries.append(entry)

    df = pd.DataFrame(entries)
    df = df.set_index("timestamp")

    # Get price data for the position.
    # We create column mark_price which we use to calculate realised and unrealised pnl
    price_df = pd.DataFrame({
        "mark_price": candles["close"],  # Mark price
        "timestamp": candles["timestamp"],
    })
    price_df = price_df.set_index("timestamp")

    joined_df = pd.merge(df, price_df, left_index=True, right_index=True, how='outer')
    joined_df["delta"] = joined_df["delta"].fillna(0)
    joined_df["fee"] = joined_df["fee"].fillna(0)
    joined_df = joined_df.ffill()
    joined_df["value"] = joined_df["mark_price"] * joined_df["quantity"] * (1 - pair.fee)

    joined_df["mark_price"] = joined_df["mark_price"].astype(float)
    joined_df["delta"] = joined_df["delta"].astype(float)
    joined_df["quantity"] = joined_df["quantity"].astype(float)

    def _realised_pnl(row):
        if row["delta"] < 0:
            return (row["mark_price"] - row["avg_price"]) * abs(row["delta"])
        return 0

    # Calculate realised and unrealised PnL,
    # then how much PnL would be annually so we get APY
    joined_df["realised_delta"] = joined_df.apply(_realised_pnl, axis=1)

    # joined_df["realised_pnl"] = joined_df["realised_delta"].cumsum()
    # joined_df["unrealised_pnl"] = joined_df["value"] - joined_df["cumulative_cost"]

    joined_df = track_lot_profitability_detailed(joined_df)
    joined_df["pnl_pct"] = joined_df["total_pnl_pct"]
    joined_df["pnl"] = joined_df["unrealised_pnl"] + joined_df["realised_pnl"]

    #joined_df["pnl_pct"] = joined_df["pnl"] / joined_df["cumulative_value"]
    joined_df["duration"] = joined_df.index - joined_df.index[0]
    joined_df["annual_periods"] = pd.Timedelta(days=365) / joined_df["duration"]
    joined_df["pnl_annualised"] = (1 + joined_df["pnl_pct"]) ** (joined_df["annual_periods"]) - 1
    return joined_df


def visualise_position(
    position: TradingPosition,
    df: pd.DataFrame,
    height=800,
    width=1200,
    autosize=False,
    extended=False,
) -> Figure:
    """Visualise position as a Plotly figure with subplots.

    - Price chart with trade markers on top subplot
    - Position value in middle subplot
    - PnL charts on bottom subplot

    Example:

    .. code-block:: python

        from tradeexecutor.visual.position import visualise_position, calculate_position_timeline

        # Visualise the last position in the table above
        position_id = vault_df.index[-2]
        position = state.portfolio.get_position_by_id(position_id)
        position_df = calculate_position_timeline(
            strategy_universe,
            position,
            end_at=state.backtest_data.end_at,
        )
        fig = visualise_position(
            position,
            position_df,
            extended=True,
            height=1600,
            width=1200,
            autosize=False,
        )

        fig.update_layout(
            margin=dict(b=100),  # add bottom margin
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.05,
                xanchor="center",
                x=0.5
            )
        )

        fig.show(renderer="notebook")

    :param extended:
        Plots a tons of stuff.
    """
    assert isinstance(df, pd.DataFrame), f"Expected DataFrame, got {type(df)}"

    if df.empty:
        raise ValueError("DataFrame is empty, cannot visualise position")

    rows = 10 if extended else 4
    layout = [0.1] * 10 if extended else [0.4, 0.2, 0.2, 0.2]

    if extended:
        subplot_titles = (
            'Price and trades',
            'Position Value',
            'PnL %',
            'PnL USD',
            "Cumulative cost USD",
            "Cumulative value USD",
            "Avg price USD",
            "Realised PnL USD",
            "Unrealised PnL USD",
            "Realised delta USD",
        )
    else:
        subplot_titles = ('Price and trades', 'Position Value', 'PnL %', 'PnL USD')

    # Create subplots with 3 rows
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.01,
        subplot_titles=subplot_titles,
        row_heights=layout,
        x_title="Time",
    )

    # Convert DateTimeIndex to list of datetime objects.
    # Plotly 6 bug that you need to explicitly convert?
    index = df.index.to_pydatetime().tolist()

    # Price chart (top subplot)
    fig.add_trace({
        "x": index,
        "y": df["mark_price"],
        "type": "scatter",
        "mode": "lines",
        "name": "Price",
        "line": {"width": 2},
        "showlegend": False
    }, row=1, col=1)

    # Position value chart (middle subplot)
    fig.add_trace({
        "x": index,
        "y": df["value"],
        "type": "scatter",
        "mode": "lines",
        "name": "Position value USD",
        "line": {"width": 2, "color": "blue"},
        "showlegend": False
    }, row=2, col=1)

    # PnL percentage chart (bottom subplot)
    fig.add_trace({
        "x": index,
        "y": df["pnl_annualised"] * 100,
        "type": "scatter",
        "mode": "lines",
        "name": "PnL annualised %",
        "line": {"width": 2, "color": "#FF5733"},
        "showlegend": False
    }, row=3, col=1)

    # PnL USD chart (bottom subplot, secondary y-axis)
    fig.add_trace({
        "x": index,
        "y": df["pnl"],
        "type": "scatter",
        "mode": "lines",
        "name": "PnL USD",
        "line": {"width": 2, "color": "#FF8C00"},
        "yaxis": "y4",
        "showlegend": False
    }, row=4, col=1)

    if extended:
        fig.add_trace({
            "x": index,
            "y": df["cumulative_cost"],
            "type": "scatter",
            "mode": "lines",
            "name": "Cumulative cost USD",
            "line": {"width": 2, "color": "#FF8C00"},
            "showlegend": True
        }, row=5, col=1)

        fig.add_trace({
            "x": index,
            "y": df["cumulative_value"],
            "type": "scatter",
            "mode": "lines",
            "name": "Cumulative value USD",
            "line": {"width": 2, "color": "#FF8C00"},
            "showlegend": False
        }, row=6, col=1)

        fig.add_trace({
            "x": index,
            "y": df["avg_price"],
            "type": "scatter",
            "mode": "lines",
            "name": "Avg price USD",
            "line": {"width": 2, "color": "#FF8C00"},
            "showlegend": False
        }, row=7, col=1)

        fig.add_trace({
            "x": index,
            "y": df["realised_pnl"],
            "type": "scatter",
            "mode": "lines",
            "name": "Avg price USD",
            "line": {"width": 2, "color": "#FF8C00"},
            "showlegend": False
        }, row=8, col=1)

        fig.add_trace({
            "x": index,
            "y": df["unrealised_pnl"],
            "type": "scatter",
            "mode": "lines",
            "name": "Avg price USD",
            "line": {"width": 2, "color": "#FF8C00"},
            "showlegend": False
        }, row=9, col=1)

        fig.add_trace({
            "x": index,
            "y": df["realised_delta"],
            "name": "Realised delta USD",
            "showlegend": False,
            "type": "bar",
        }, row=10, col=1)

    # Add trades as markers on the price chart
    trade_count = 0
    for timestamp, trade in df.iterrows():

        if trade["delta"] == 0:
            # Mark to market row, no trade executed
            continue

        trade_count += 1
        color = "#00FF00" if trade['delta'] > 0 else "#FF0000"
        y_adjust = -10 if trade['delta'] > 0 else 10

        # Create annotation with pixel-based offset
        fig.add_annotation(
            x=timestamp,
            y=trade["mark_price"],
            text="▲" if trade['delta'] > 0 else "▼",
            showarrow=False,
            font=dict(
                family="Arial",
                size=18,
                color=color
            ),
            yshift=y_adjust,
            hovertext=f"Trade {trade_count}: {trade['delta']:+.2f}",
        )

    # # Get position tag
    if position.pair.is_vault():
        tag = position.pair.get_vault_name()
    else:
        tag = position.pair.base.token_symbol

    # Update layout
    fig.update_layout(
        title={
            'text': f"Position #{position.position_id} {tag} Timeline",
            'x': 0.5,
            'xanchor': 'center'
        },
        autosize=autosize,
        width=width,
        height=height,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Update y-axis labels
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Value (USD)", row=2, col=1)
    fig.update_yaxes(title_text="PnL annualised (%)", row=3, col=1)
    fig.update_yaxes(title_text="PnL (USD)", row=4, col=1)

    if extended:
        fig.update_yaxes(title_text="Cum. cost (USD)", row=5, col=1)
        fig.update_yaxes(title_text="Cum. value (USD)", row=6, col=1)
        fig.update_yaxes(title_text="Avg. price (USD)", row=7, col=1)
        fig.update_yaxes(title_text="Realised PnL (USD)", row=8, col=1)
        fig.update_yaxes(title_text="Unrealised PnL (USD)", row=9, col=1)
        fig.update_yaxes(title_text="Realised delta (USD)", row=10, col=1)

    return fig



def calculate_macb_pnl(df):
    position_size = 0.0
    average_cost = 0.0
    total_realised = 0.0

    realised_pnl_list = []
    unrealised_pnl_list = []
    realised_pct_list = []
    unrealised_pct_list = []
    total_pct_list = []

    for _, row in df.iterrows():
        delta = row['delta']
        trade_price = row['executed_price']
        mark_price = row['mark_price']

        # If a trade occurred
        if delta > 0:
            # Buying: update average cost
            total_cost = average_cost * position_size + trade_price * delta
            position_size += delta
            average_cost = total_cost / position_size if position_size != 0 else 0.0

        elif delta < 0 and position_size > 0:
            # Selling: realise PnL
            sell_qty = -delta
            if sell_qty > position_size:
                raise ValueError("Sell quantity exceeds current position size")
            realised_pnl = (trade_price - average_cost) * sell_qty
            total_realised += realised_pnl
            position_size -= sell_qty
            if position_size == 0:
                average_cost = 0.0  # Reset cost when fully closed

        # Calculate unrealised PnL
        unrealised_pnl = (mark_price - average_cost) * position_size
        invested_capital = average_cost * position_size

        # Append all stats
        realised_pnl_list.append(total_realised)
        unrealised_pnl_list.append(unrealised_pnl)
        realised_pct_list.append(
            total_realised / (total_realised + invested_capital + unrealised_pnl)
            if (total_realised + invested_capital + unrealised_pnl) != 0 else 0
        )
        unrealised_pct_list.append(
            unrealised_pnl / invested_capital if invested_capital != 0 else 0
        )
        total_value = total_realised + unrealised_pnl
        total_cost_basis = total_realised + invested_capital
        total_pct_list.append(
            total_value / total_cost_basis if total_cost_basis != 0 else 0
        )

    # Attach results to DataFrame
    df['realised_pnl'] = realised_pnl_list
    df['unrealised_pnl'] = unrealised_pnl_list
    df['realised_pnl_pct'] = realised_pct_list
    df['unrealised_pnl_pct'] = unrealised_pct_list
    df['total_pnl_pct'] = total_pct_list

    return df



def calculate_lot_level_pnl_fifo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate lot-level realised and unrealised profit using FIFO method.

    Parameters:
    df (pd.DataFrame): A DataFrame containing:
        - 'quantity': Positive for buy, negative for sell
        - 'executed_price': Price at which trade executed
        - 'mark_price': Market price at that timestamp

    Returns:
    pd.DataFrame: With additional columns for:
        - 'realised_profit'
        - 'unrealised_profit'
        - 'total_profit'
        - 'total_return_pct'
    """
    lots = []  # Open positions: [{'quantity': int, 'price': float}]
    realised_profits = []
    unrealised_profits = []
    total_profits = []
    total_return_pcts = []

    cumulative_realised = 0

    for _, row in df.iterrows():
        qty = row['quantity']
        executed_price = row['executed_price']
        mark_price = row['mark_price']
        realised_profit = 0

        if qty > 0:
            # Buy: add to open lots
            lots.append({'quantity': qty, 'price': executed_price})
        else:
            # Sell: match to open lots using FIFO
            sell_qty = -qty
            while sell_qty > 0 and lots:
                open_lot = lots[0]
                match_qty = min(open_lot['quantity'], sell_qty)
                realised_profit += match_qty * (executed_price - open_lot['price'])
                open_lot['quantity'] -= match_qty
                sell_qty -= match_qty
                if open_lot['quantity'] == 0:
                    lots.pop(0)

        cumulative_realised += realised_profit
        realised_profits.append(realised_profit)

        # Unrealised profit from open lots
        unrealised = sum(lot['quantity'] * (mark_price - lot['price']) for lot in lots)
        unrealised_profits.append(unrealised)

        total_profit = cumulative_realised + unrealised
        total_profits.append(total_profit)

        invested_capital = sum(lot['quantity'] * lot['price'] for lot in lots)
        total_base = invested_capital + cumulative_realised

        total_return_pct = (total_profit / total_base) * 100 if total_base != 0 else 0
        total_return_pcts.append(total_return_pct)

    df = df.copy()
    df['realised_pnl'] = realised_profits
    df['unrealised_pnl'] = unrealised_profits
    df['total_pnl'] = total_profits
    df['total_pnl_pct'] = total_return_pcts

    return df



def track_lot_profitability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tracks per-lot realised and unrealised profit and calculates weighted average return (%).
    Assumes FIFO matching for sells.

    Parameters:
        df (pd.DataFrame): Must contain columns:
            - 'quantity': positive for buy, negative for sell
            - 'executed_price': trade execution price
            - 'mark_price': current market price for unrealised PnL

    Returns:
        pd.DataFrame: Original dataframe + total_pnl and weighted_return_pct columns
    """
    open_lots = []
    closed_lots = []
    cumulative_realised = 0
    lot_id = 0

    weighted_returns = []
    total_pnls = []

    for _, row in df.iterrows():
        qty = row['quantity']
        exec_price = row['executed_price']
        mark_price = row['mark_price']
        realised_pnl = 0

        # SELL
        if qty < 0:
            remaining = -qty
            while remaining > 0 and open_lots:
                lot = open_lots[0]
                matched_qty = min(remaining, lot['quantity'])
                realised = matched_qty * (exec_price - lot['price'])

                closed_lots.append({
                    'lot_id': lot['id'],
                    'entry_price': lot['price'],
                    'exit_price': exec_price,
                    'quantity': matched_qty,
                    'capital': matched_qty * lot['price'],
                    'realised_pnl': realised,
                    'return_pct': (exec_price - lot['price']) / lot['price'] * 100
                })

                lot['quantity'] -= matched_qty
                if lot['quantity'] == 0:
                    open_lots.pop(0)

                cumulative_realised += realised
                remaining -= matched_qty

        # BUY
        elif qty > 0:
            open_lots.append({
                'id': lot_id,
                'quantity': qty,
                'price': exec_price
            })
            lot_id += 1

        # Unrealised from open lots
        unrealised = sum(l['quantity'] * (mark_price - l['price']) for l in open_lots)

        unrealised_lots = [
            {
                'lot_id': l['id'],
                'entry_price': l['price'],
                'exit_price': mark_price,
                'quantity': l['quantity'],
                'capital': l['quantity'] * l['price'],
                'realised_pnl': 0,
                'return_pct': (mark_price - l['price']) / l['price'] * 100
            }
            for l in open_lots
        ]

        # Combine all active lots
        all_lots = closed_lots + unrealised_lots
        total_capital = sum(l['capital'] for l in all_lots)
        total_pnl = sum(l['realised_pnl'] for l in all_lots) + unrealised

        # Weighted average return
        weighted_return = (
            sum(l['return_pct'] * l['capital'] for l in all_lots) / total_capital
            if total_capital else 0
        )

        weighted_returns.append(weighted_return)
        total_pnls.append(total_pnl)

    # Append results to DataFrame
    df = df.copy()
    df['total_pnl'] = total_pnls
    df['weighted_return_pct'] = weighted_returns
    return df



def track_lot_profitability_detailed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Track lot-level realised and unrealised PnL using FIFO method,
    and return position-level summary metrics at each step.

    Parameters:
        df (pd.DataFrame): Must contain columns:
            - 'quantity': +ve for buy, -ve for sell
            - 'executed_price': trade price
            - 'mark_price': current market price for unrealised PnL

    Returns:
        pd.DataFrame: with added columns:
            - 'total_pnl': realised + unrealised PnL (USD)
            - 'total_pnl_pct': profitability in percent of capital
            - 'unrealised_pnl': mark-to-market PnL of open lots
            - 'realised_pnl': cumulative PnL from closed lots
    """
    open_lots = []
    closed_lots = []
    lot_id = 0
    cumulative_realised = 0

    total_pnls = []
    total_pnl_pcts = []
    unrealised_pnls = []
    realised_pnls = []

    for _, row in df.iterrows():
        qty = row['quantity']
        exec_price = row['executed_price']
        mark_price = row['mark_price']
        realised_pnl = 0

        # SELL: Match against open lots using FIFO
        if qty < 0:
            remaining = -qty
            while remaining > 0 and open_lots:
                lot = open_lots[0]
                matched_qty = min(remaining, lot['quantity'])
                pnl = matched_qty * (exec_price - lot['price'])

                closed_lots.append({
                    'lot_id': lot['id'],
                    'entry_price': lot['price'],
                    'exit_price': exec_price,
                    'quantity': matched_qty,
                    'capital': matched_qty * lot['price'],
                    'realised_pnl': pnl,
                    'return_pct': (exec_price - lot['price']) / lot['price'] * 100
                })

                lot['quantity'] -= matched_qty
                if lot['quantity'] == 0:
                    open_lots.pop(0)

                realised_pnl += pnl
                cumulative_realised += pnl
                remaining -= matched_qty

        # BUY: Add to open lots
        elif qty > 0:
            open_lots.append({
                'id': lot_id,
                'quantity': qty,
                'price': exec_price
            })
            lot_id += 1

        # Unrealised PnL calculation
        unrealised_pnl = sum(
            l['quantity'] * (mark_price - l['price']) for l in open_lots
        )

        # Total invested capital across all lots (closed + open)
        unrealised_lots = [
            {
                'capital': l['quantity'] * l['price'],
                'return_pct': (mark_price - l['price']) / l['price'] * 100
            } for l in open_lots
        ]
        all_lots = closed_lots + unrealised_lots
        total_capital = sum(l['capital'] for l in all_lots)

        # Total PnL and PnL %
        total_pnl = cumulative_realised + unrealised_pnl
        total_pnl_pct = (total_pnl / total_capital * 100) if total_capital else 0

        # Store per-step results
        total_pnls.append(total_pnl)
        total_pnl_pcts.append(total_pnl_pct)
        unrealised_pnls.append(unrealised_pnl)
        realised_pnls.append(cumulative_realised)

    # Final DataFrame
    df = df.copy()
    df['total_pnl'] = total_pnls
    df['total_pnl_pct'] = total_pnl_pcts
    df['unrealised_pnl'] = unrealised_pnls
    df['realised_pnl'] = realised_pnls
    return df
