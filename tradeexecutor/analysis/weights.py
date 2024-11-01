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
    reserve_asset, price = state.portfolio.get_default_reserve_asset()
    reserve_asset_symbol = reserve_asset.token_symbol
    reserve_rows = [{
            "timestamp": ps.calculated_at,
            "asset": reserve_asset_symbol,
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

    # For each timestamp, we may have multiple entries of the same asset
    # - in this case take the last one per asset.
    # These may cause e.g. by simulated deposit events.
    # 2021-06-01  USDC     1.000000e+06
    #             WBTC     9.840778e+05
    #             USDC     1.000000e+04
    series_deduped = series[~series.index.duplicated(keep='last')]

    # Pass to visualisation
    series_deduped.attrs["reserve_asset_symbol"] = reserve_asset_symbol
    return series_deduped


def visualise_weights(
    weights_series: pd.Series,
    normalised=True,
    color_palette = colors.qualitative.Light24,
    template="plotly_dark",
) -> Figure:
    """Draw a chart of weights."""

    assert isinstance(weights_series, pd.Series)

    reserve_asset_symbol = weights_series.attrs["reserve_asset_symbol"]

    def sort_key_reserve_first(col_name):
        if col_name == reserve_asset_symbol:
            return -1000, col_name
        return 0, col_name

    # Unstack to create DataFrame with asset symbols as columns
    df = weights_series.unstack(level=1)

    # Make sure reserve is always the lefmost column
    df = df[sorted(df.columns, key=sort_key_reserve_first)]

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
        template=template,
    )
    fig.update_traces(fillcolor='#aaa', selector=dict(name=reserve_asset_symbol))
    fig.update_traces(line_width=0)
    return fig


def calculate_weights_statistics(
    weights: pd.Series,
) -> pd.DataFrame:
    """Get statistics of weights during the portfolio construction.

    - Cash positions are excluded

    :param weights:
        US Dollar weights as series of MultiIndex(timestamp, pair)

    :return:
        Human-readable table of statistics
    """

    assert isinstance(weights, pd.Series)
    assert isinstance(weights.index, pd.MultiIndex)

    stats = []

    # Filter out reserve position
    reserve_asset_symbol = weights.attrs["reserve_asset_symbol"]
    weights = weights[weights.index.get_level_values(1) != reserve_asset_symbol]

    # Filter out zero values
    weights = weights[weights != 0]

    max_idx = weights.idxmax()
    at, pair = max_idx
    value = weights[max_idx]

    stats.append({
        "Name": "Max position",
        "At": at,
        "Pair": pair,
        "Value": value,
        "Unit": "USD",
    })

    min_idx = weights.idxmin()
    at, pair = min_idx
    value = weights[min_idx]

    stats.append({
        "Name": "Min position",
        "At": at,
        "Pair": pair,
        "Value": value,
        "Unit": "USD",
    })

    value = weights.mean()

    stats.append({
        "Name": "Mean position",
        "At": "",
        "Pair": "",
        "Value": value,
        "Unit": "USD",
    })

    # Normalised
    normalised = weights.groupby(level='timestamp').transform(lambda x: x / x.sum() * 100)

    max_idx = normalised.idxmax()
    at, pair = max_idx
    value = normalised[max_idx]

    stats.append({
        "Name": "Max position",
        "At": at,
        "Pair": pair,
        "Value": value,
        "Unit": "%",
    })

    min_idx = normalised.idxmin()
    at, pair = min_idx
    value = normalised[min_idx]

    stats.append({
        "Name": "Min position",
        "At": at,
        "Pair": pair,
        "Value": value,
        "Unit": "%",
    })

    value = normalised.mean()

    stats.append({
        "Name": "Mean position",
        "At": "",
        "Pair": "",
        "Value": value,
        "Unit": "%",
    })

    df = pd.DataFrame(stats)

    df = df.set_index("Name")
    df["Value"] = df["Value"].apply(lambda x: "{:,.2f}".format(x))
    return df


