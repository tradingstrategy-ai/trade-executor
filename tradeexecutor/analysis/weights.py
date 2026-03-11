"""Weight allocations of assets.

- Visualise what the strategy portfolio consists of over time

- See :py:func:`visualise_weights` for usage

"""
from __future__ import annotations

import enum
from collections import defaultdict

import pandas as pd
from plotly.graph_objs import Figure
import plotly.express as px
import plotly.colors as colors

from tradeexecutor.state.state import State
from tradeexecutor.state.types import USDollarAmount

class LegendMode(enum.Enum):
    side = "side"
    bottom = "bottom"

def calculate_asset_weights(
    state: State,
) -> pd.Series:
    """Get timeline of asset weights for a backtest.

    - Designed for visualisation / human readable output

    - Might not handle complex cases correctly

    - Uses portfolio positions as the input

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
        "kind": "reserve"
    } for ps in state.stats.portfolio]

    # Need to look up assets for every position
    position_asset_map = {p.position_id: p.pair.get_chart_label() for p in state.portfolio.get_all_positions()}
    position_kind_map = {p.position_id: p.pair.kind.value for p in state.portfolio.get_all_positions()}

    # Add position values
    position_rows = [{
        "timestamp": ps.calculated_at,
        "asset": position_asset_map[position_id],
        "value": ps.value,
        "kind": position_kind_map[position_id]
    } for position_id, position_stats in state.stats.positions.items() for ps in position_stats]

    df = pd.DataFrame(reserve_rows + position_rows)

    # For credit positions, we might have close and poen new position in the same
    # timestamp and need to handle this specially.
    mask = df["kind"] == "credit_supply"
    df_to_dedup = df[mask]  # rows that need deduplication
    df_keep = df[~mask]  # rows to keep as-is
    df_deduped = df_to_dedup.groupby(['timestamp', 'asset']).agg({'value': 'sum'}).reset_index()
    df = pd.concat([df_deduped, df_keep])

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.sort_values(by='timestamp')

    df = df.set_index(["timestamp", "asset"])
    series = df["value"]

    # For each timestamp, we may have multiple entries of the same asset
    # - in this case take the sum the assets
    # These may cause e.g. by simulated deposit events.
    # 2021-06-01  USDC     1.000000e+06
    #             WBTC     9.840778e+05
    #             USDC     1.000000e+04
    # import ipdb ; ipdb.set_trace()
    series_deduped = series[~series.index.duplicated(keep='last')]

    # Get out all Aave aUSDC positions
    credit_supply_symbols = [p.pair.base.token_symbol for p in state.portfolio.get_all_positions() if p.is_credit_supply()]

    # Get vaults
    vault_symbols = [p.pair.get_vault_name() for p in state.portfolio.get_all_positions() if p.is_vault()]

    # Pass to visualisation
    series_deduped.attrs["reserve_asset_symbol"] = reserve_asset_symbol
    series_deduped.attrs["credit_supply_symbols"] = credit_supply_symbols
    series_deduped.attrs["vault_symbols"] = vault_symbols

    return series_deduped


def visualise_weights(
    weights_series: pd.Series,
    normalised=True,
    color_palette=colors.qualitative.Light24,
    template="plotly_dark",
    include_reserves=True,
    legend_mode: LegendMode=LegendMode.side,
    aave_colour='#ccc',
    reserve_asset_colour='#aaa',
    clean=False,
) -> Figure:
    """Draw a chart of weights.

    :param normalised:
        Do 100% stacked chart over time

    :param include_reserves:
        Include reserve positions like USDC in the output.

        Setting False will remove cash, Aave, vaults.

    :param clean:
        Remove title texts.

        Good for screenshots.

    :return:
        Plotly chart
    """

    assert isinstance(weights_series, pd.Series)

    reserve_asset_symbol = weights_series.attrs["reserve_asset_symbol"]
    vault_symbols = weights_series.attrs["vault_symbols"]
    non_volatile_symbols = [weights_series.attrs["reserve_asset_symbol"]] + weights_series.attrs["credit_supply_symbols"]

    if not include_reserves:
        # Filter out reserve/credit position
        weights_series = weights_series[
            ~(weights_series.index.get_level_values(1).isin(non_volatile_symbols) | weights_series.index.get_level_values(1).isin(vault_symbols))
        ]

    def sort_key_reserve_first(col_name):
        if col_name == reserve_asset_symbol:
            return -1000, col_name
        elif col_name in non_volatile_symbols:
            return -500, col_name
        elif col_name in vault_symbols:
            return -200, col_name

        return 0, col_name

    # Unstack to create DataFrame with asset symbols as columns
    df = weights_series.unstack(level=1)

    # Make sure reserve is always the lefmost column
    df = df[sorted(df.columns, key=sort_key_reserve_first)]

    if normalised:
        df = df.div(df.sum(axis=1), axis=0) * 100

    if include_reserves:
        reserve_text = f"{reserve_asset_symbol} reserves included"
    else:
        reserve_text = f"{reserve_asset_symbol} reserves excluded"

    fig = px.area(
        df,
        title=f'Asset weights (%), {reserve_text}' if normalised else f'Asset weights (USD), {reserve_text}',
        labels={
            'index': 'Time',
            'value': '% of portfolio' if normalised else 'US dollar size',
        },
        color_discrete_sequence=color_palette,
        template=template,
    )

    for symbol in non_volatile_symbols:
        # Aave colour
        # https://aave.com/brand
        fig.update_traces(
            fillcolor=aave_colour,
            selector=dict(name=symbol),
            fillpattern=dict(
                shape="-",
                size=5,
                solidity=0.8
            ),
        )

    for symbol in vault_symbols:
        # Aave colour
        # https://aave.com/brand
        fig.update_traces(
            selector=dict(name=symbol),
            fillpattern=dict(
                shape="x",
                size=5,
                solidity=0.8
            ),
        )

    fig.update_traces(fillcolor=reserve_asset_colour, selector=dict(name=reserve_asset_symbol))
    fig.update_traces(line_width=0)

    match legend_mode:
        case LegendMode.bottom:
            # Adjust legend properties
            fig.update_layout(
                # Move legend to bottom
                legend=dict(
                    yanchor="top",
                    y=-0.1,  # Adjust this value to move legend up/down
                    xanchor="center",
                    x=0.5,
                    # Arrange items in 4 rows
                    orientation="h",
                    traceorder="normal",
                    # nrows=4
                    itemwidth=40,  # Adjust the multiplier as needed
                    title_text="",
                    font=dict(
                        size=20  # Adjust this value to make legend text bigger/smaller
                    ),
                )
            )

    if clean:
        fig.update_layout(
            title=None,
            xaxis=dict(
                title=None,
                # other x-axis properties...
                nticks=4,
                # Increase font size (default is usually 12)
                tickfont=dict(
                    size=22  # Adjust this value to make font bigger/smaller
                ),
                type='date',
            ),
            yaxis=dict(
                title=None,
                # other y-axis properties...
                nticks=5,
                # Optionally specify tick labels
                # ticktext=['0%', '50%', '100%'],
                tickfont=dict(
                    size=22,
                ),

            )
        )

    return fig


def calculate_weights_statistics(
    weights: pd.Series,
    state: State | None = None,
) -> pd.DataFrame:
    """Get statistics of weights during the portfolio construction.

    - Cash positions are excluded

    :param weights:
        US Dollar weights as series of MultiIndex(timestamp, pair)

    :param state:
        If provided, also calculate biggest winner/loser
        for individual positions and by vault (aggregated).

    :return:
        Human-readable table of statistics
    """

    assert isinstance(weights, pd.Series)
    assert isinstance(weights.index, pd.MultiIndex)

    stats = []

    source_weights = weights

    # Filter out reserve position
    reserve_asset_symbol = weights.attrs["reserve_asset_symbol"]
    weights = weights[weights.index.get_level_values(1) != reserve_asset_symbol]

    # Filter out zero values
    weights = weights[weights != 0]

    if len(weights.index) == 0:
        return pd.DataFrame([
            {"Error": "No trades or weights assigned"}
        ])

    max_idx = weights.idxmax()
    at, pair = max_idx
    value = weights[max_idx]

    stats.append({
        "Name": f"Max position (excluding {reserve_asset_symbol})",
        "At": at,
        "Pair": pair,
        "Value": value,
        "Unit": "USD",
    })

    min_idx = weights.idxmin()
    at, pair = min_idx
    value = weights[min_idx]

    stats.append({
        "Name": f"Min position (excluding {reserve_asset_symbol})",
        "At": at,
        "Pair": pair,
        "Value": value,
        "Unit": "USD",
    })

    value = weights.mean()

    stats.append({
        "Name": f"Mean position (excluding {reserve_asset_symbol})",
        "At": "",
        "Pair": "",
        "Value": value,
        "Unit": "USD",
    })

    #
    # Normalised weights
    #

    normalised = weights.groupby(level='timestamp').transform(lambda x: x / x.sum() * 100)

    max_idx = normalised.idxmax()
    at, pair = max_idx
    value = normalised[max_idx]

    stats.append({
        "Name": f"Max position (excluding {reserve_asset_symbol})",
        "At": at,
        "Pair": pair,
        "Value": value,
        "Unit": "%",
    })

    min_idx = normalised.idxmin()
    at, pair = min_idx
    value = normalised[min_idx]

    stats.append({
        "Name": f"Min position (excluding {reserve_asset_symbol})",
        "At": at,
        "Pair": pair,
        "Value": value,
        "Unit": "%",
    })

    value = normalised.mean()

    stats.append({
        "Name": f"Mean position (excluding {reserve_asset_symbol})",
        "At": "",
        "Pair": "",
        "Value": value,
        "Unit": "%",
    })

    #
    # Same, but USDC included in the mix
    #

    normalised = source_weights \
        [source_weights != 0] \
        .groupby(level='timestamp') \
        .transform(lambda x: x / x.sum() * 100)

    max_idx = normalised.idxmax()
    at, pair = max_idx
    value = normalised[max_idx]

    stats.append({
        "Name": f"Max position (including {reserve_asset_symbol})",
        "At": at,
        "Pair": pair,
        "Value": value,
        "Unit": "%",
    })

    min_idx = normalised.idxmin()
    at, pair = min_idx
    value = normalised[min_idx]

    stats.append({
        "Name": f"Min position (including {reserve_asset_symbol})",
        "At": at,
        "Pair": pair,
        "Value": value,
        "Unit": "%",
    })

    value = normalised.mean()

    stats.append({
        "Name": f"Mean position (including {reserve_asset_symbol})",
        "At": "",
        "Pair": "",
        "Value": value,
        "Unit": "%",
    })

    #
    # Biggest winner/loser by individual position and by vault
    #
    if state is not None:
        winner_loser_stats = _calculate_winner_loser_stats(state)
        stats.extend(winner_loser_stats)

    df = pd.DataFrame(stats)

    df = df.set_index("Name")
    df["Value"] = df["Value"].apply(lambda x: "{:,.2f}".format(x))
    return df


def _make_vault_link(address: str) -> str:
    """Create a clickable link to the vault on Trading Strategy."""
    return f"https://tradingstrategy.ai/trading-view/vaults/address/{address}"


def _calculate_winner_loser_stats(state: State) -> list[dict]:
    """Calculate biggest winner/loser for individual positions and by vault.

    :return:
        List of stat dicts with Name, At, Pair, Value, Unit, Address, Link keys.
    """
    stats = []
    positions = list(state.portfolio.get_all_positions())

    # Skip reserve/credit supply positions
    positions = [p for p in positions if not p.is_credit_supply()]

    if not positions:
        return stats

    # Individual position winner/loser (by USD profit)
    best_pos = max(positions, key=lambda p: p.get_total_profit_usd())
    worst_pos = min(positions, key=lambda p: p.get_total_profit_usd())

    stats.append({
        "Name": "Biggest winner (position)",
        "At": best_pos.opened_at,
        "Pair": best_pos.pair.get_chart_label(),
        "Value": best_pos.get_total_profit_usd(),
        "Unit": "USD",
        "Address": best_pos.pair.pool_address,
        "Link": _make_vault_link(best_pos.pair.pool_address),
    })

    stats.append({
        "Name": "Biggest loser (position)",
        "At": worst_pos.opened_at,
        "Pair": worst_pos.pair.get_chart_label(),
        "Value": worst_pos.get_total_profit_usd(),
        "Unit": "USD",
        "Address": worst_pos.pair.pool_address,
        "Link": _make_vault_link(worst_pos.pair.pool_address),
    })

    # Individual position winner/loser (by % profit)
    best_pct_pos = max(positions, key=lambda p: p.get_total_profit_percent())
    worst_pct_pos = min(positions, key=lambda p: p.get_total_profit_percent())

    stats.append({
        "Name": "Biggest winner (position)",
        "At": best_pct_pos.opened_at,
        "Pair": best_pct_pos.pair.get_chart_label(),
        "Value": best_pct_pos.get_total_profit_percent() * 100,
        "Unit": "%",
        "Address": best_pct_pos.pair.pool_address,
        "Link": _make_vault_link(best_pct_pos.pair.pool_address),
    })

    stats.append({
        "Name": "Biggest loser (position)",
        "At": worst_pct_pos.opened_at,
        "Pair": worst_pct_pos.pair.get_chart_label(),
        "Value": worst_pct_pos.get_total_profit_percent() * 100,
        "Unit": "%",
        "Address": worst_pct_pos.pair.pool_address,
        "Link": _make_vault_link(worst_pct_pos.pair.pool_address),
    })

    # Aggregate by vault (chart label)
    # Track address per vault label for links
    vault_profits: dict[str, USDollarAmount] = defaultdict(float)
    vault_bought: dict[str, USDollarAmount] = defaultdict(float)
    vault_address: dict[str, str] = {}
    for p in positions:
        label = p.pair.get_chart_label()
        vault_profits[label] += p.get_total_profit_usd()
        vault_bought[label] += p.get_total_bought_usd()
        vault_address[label] = p.pair.pool_address

    if vault_profits:
        best_vault = max(vault_profits, key=vault_profits.get)
        worst_vault = min(vault_profits, key=vault_profits.get)

        stats.append({
            "Name": "Biggest winner (vault)",
            "At": "",
            "Pair": best_vault,
            "Value": vault_profits[best_vault],
            "Unit": "USD",
            "Address": vault_address[best_vault],
            "Link": _make_vault_link(vault_address[best_vault]),
        })

        stats.append({
            "Name": "Biggest loser (vault)",
            "At": "",
            "Pair": worst_vault,
            "Value": vault_profits[worst_vault],
            "Unit": "USD",
            "Address": vault_address[worst_vault],
            "Link": _make_vault_link(vault_address[worst_vault]),
        })

        # Vault winner/loser by % (profit / total bought)
        vault_pct = {
            label: (vault_profits[label] / vault_bought[label] * 100) if vault_bought[label] else 0
            for label in vault_profits
        }
        best_pct_vault = max(vault_pct, key=vault_pct.get)
        worst_pct_vault = min(vault_pct, key=vault_pct.get)

        stats.append({
            "Name": "Biggest winner (vault)",
            "At": "",
            "Pair": best_pct_vault,
            "Value": vault_pct[best_pct_vault],
            "Unit": "%",
            "Address": vault_address[best_pct_vault],
            "Link": _make_vault_link(vault_address[best_pct_vault]),
        })

        stats.append({
            "Name": "Biggest loser (vault)",
            "At": "",
            "Pair": worst_pct_vault,
            "Value": vault_pct[worst_pct_vault],
            "Unit": "%",
            "Address": vault_address[worst_pct_vault],
            "Link": _make_vault_link(vault_address[worst_pct_vault]),
        })

    return stats


def render_weight_series_table(weights_series: pd.Series) -> pd.DataFrame:
    """Render the weight series in human readable format.

    - Each row is a timestamp
    - All assets are columns, representing USD allocation of the asset for the timestamp

    Example:

    .. code-block:: python

        from tradeexecutor.analysis.weights import calculate_asset_weights, visualise_weights, render_weight_series_table

        weights_series = calculate_asset_weights(state)

        with pd.option_context('display.max_rows', None):
            df = render_weight_series_table(weights_series)
            display(df)

    """

    assert isinstance(weights_series, pd.Series)
    weight_df = weights_series.unstack()
    filtered_df = weight_df.dropna(axis=1, how='all')

    # Get the index of first non-NA value for each column
    first_non_na = filtered_df.notna().idxmax()

    # Sort columns based on first non-NA index
    sorted_columns = first_non_na.sort_values().index

    # Reorder DataFrame
    reordered_df = filtered_df[sorted_columns]
    return reordered_df