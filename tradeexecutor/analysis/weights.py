"""Weight allocations of assets.

- Visualise what the strategy portfolio consists of over time

- See :py:func:`visualise_weights` for usage

"""
import pandas as pd
from plotly.graph_objs import Figure
import plotly.express as px
import plotly.colors as colors

from tradeexecutor.state.state import State


def calculate_asset_weights(
    state: State,
) -> pd.Series:
    """Get timeline of asset weights for a backtest.

    - Designed for visualisation / human readable output

    - Might not handle complex cases correctly

    :return:
        Pandas Series of asset weights

        - (DateTime, asset symbol) MultiIndex
        - USD value of the asset in the portfolio in the given time
    """

    # Add cash rows
    reserver_asset, price = state.portfolio.get_default_reserve_asset()
    reserver_asset_symbol = reserver_asset.token_symbol
    reserve_rows = [{
            "timestamp": ps.calculated_at,
            "asset": reserver_asset_symbol,
            "value": ps.free_cash,
            } for ps in state.stats.portfolio]


    # Need to look up assets for every position
    position_asset_map = {p.position_id: p.pair.base.token_symbol for p in state.portfolio.get_all_positions()}

    # Add position values
    position_rows = [{
            "timestamp": ps.calculated_at,
            "asset": position_asset_map[position_id],
            "value": ps.value,
        } for position_id, position_stats in state.stats.positions.items() for ps in position_stats]

    df = pd.DataFrame(reserve_rows + position_rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.sort_values(by='timestamp')

    df = df.set_index(["timestamp", "asset"])
    series = df["value"]

    series_deduped = series[~series.index.duplicated(keep='last')]
    # For each timestamp, we may have multiple entries of the same asset
    # - in this case take the last one per asset.
    # These may cause e.g. by simulated deposit events.
    # 2021-06-01  USDC     1.000000e+06
    #             WBTC     9.840778e+05
    #             USDC     1.000000e+04
    return series_deduped


def visualise_weights(
    weights_series: pd.Series,
    normalised=True,
    color_palette = colors.qualitative.Light24,
) -> Figure:
    """Draw a chart of weights."""

    assert isinstance(weights_series, pd.Series)

    # Unstack to create DataFrame with asset symbols as columns
    df = weights_series.unstack(level=1)

    if normalised:
        df = df.div(df.sum(axis=1), axis=0) * 100

    fig = px.area(
        df,
        title='Asset weights (normalised)' if normalised else 'Asset weights (USD)',
        labels={
            'index': 'Time',
            'value': '% of portfolio' if normalised else 'US dollar size',
        },
        color_discrete_sequence=color_palette,
    )
    return fig
