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

    cumulative_quantity = cumulative_cost = avg_price = 0

    for trade in position.trades.values():
        delta = float(trade.executed_quantity)

        if delta > 0:
            # Buy: increase cost basis
            cumulative_cost += trade.get_value()
            cumulative_quantity += delta
        elif delta < 0:
            # Sell: reduce cost basis proportionally
            if cumulative_quantity > 0:
                avg_price = cumulative_cost / float(cumulative_quantity)
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
    joined_df["realised_pnl"] = joined_df["realised_delta"].cumsum()
    joined_df["unrealised_pnl"] = joined_df["value"] - joined_df["cumulative_cost"]
    joined_df["pnl"] = joined_df["unrealised_pnl"] + joined_df["realised_pnl"]
    joined_df["pnl_pct"] = joined_df["pnl"] / joined_df["cumulative_cost"]
    joined_df["duration"] = joined_df.index - joined_df.index[0]
    joined_df["annual_periods"] = pd.Timedelta(days=365) / joined_df["duration"]
    joined_df["pnl_annualised"] = (1 + joined_df["pnl_pct"]) ** (joined_df["annual_periods"]) - 1
    return joined_df


def visualise_position(
        position: TradingPosition,
        df: pd.DataFrame,
        height=800,
) -> Figure:
    """Visualise position as a Plotly figure with subplots.

    - Price chart with trade markers on top subplot
    - Position value in middle subplot
    - PnL charts on bottom subplot
    """
    assert isinstance(df, pd.DataFrame), f"Expected DataFrame, got {type(df)}"

    if df.empty:
        raise ValueError("DataFrame is empty, cannot visualise position")

    # Create subplots with 3 rows
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=('Price and trades', 'Position Value', 'PnL %', 'PnL USD'),
        row_heights=[0.4, 0.2, 0.2, 0.2],
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
        "showlegend": True
    }, row=1, col=1)

    # Position value chart (middle subplot)
    fig.add_trace({
        "x": index,
        "y": df["value"],
        "type": "scatter",
        "mode": "lines",
        "name": "Position value USD",
        "line": {"width": 2, "color": "blue"},
        "showlegend": True
    }, row=2, col=1)

    # PnL percentage chart (bottom subplot)
    fig.add_trace({
        "x": index,
        "y": df["pnl_annualised"] * 100,
        "type": "scatter",
        "mode": "lines",
        "name": "PnL annualised %",
        "line": {"width": 2, "color": "#FF5733"},
        "showlegend": True
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
        "showlegend": True
    }, row=4, col=1)

    # Add trades as markers on the price chart
    trade_count = 0
    for timestamp, trade in df.iterrows():

        if trade["delta"] == 0:
            # Mark to market row, no trade executed
            continue

        trade_count += 1
        color = "#00FF00" if trade['delta'] > 0 else "#FF0000"
        symbol = "triangle-up" if trade['delta'] > 0 else "triangle-down"
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

    #     fig.add_trace({
    #         "x": [timestamp],
    #         "y": [trade["mark_price"]],
    #         "type": "scatter",
    #         "mode": "markers+text",
    #         "name": f"Trade {trade_count+1}: {trade['delta']:+.2f}",
    #         "marker": {
    #             "size": 15,
    #             "color": color,
    #             "symbol": symbol,
    #             "line": {"width": 2, "color": "black"}
    #         },
    #         # "text": f"{trade['delta']:+.2f}",
    #         # "textposition": 'top center' if trade['delta'] > 0 else 'bottom center',
    #         # "textfont": {"size": 10, "color": color},
    #         "showlegend": False,
    #         "legendgroup": "trades"
    #     }, row=1, col=1)

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

    return fig
