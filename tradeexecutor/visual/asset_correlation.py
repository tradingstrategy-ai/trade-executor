import pandas as pd
import numpy as np

import plotly.graph_objects as go
from plotly.graph_objs import Figure

from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


def visualise_asset_correlation(
    strategy_universe: TradingStrategyUniverse,
    height=800,
    width=None,
) -> Figure:
    """Draw asset correlation heatmap.

    - Takes close price of all pairs in the trading universe

    """

    # Prepare correlation dataframe
    corr_data = {}

    for pair in strategy_universe.iterate_pairs():
        asset_symbol = pair.base.token_symbol
        candles = strategy_universe.data_universe.candles.get_candles_by_pair(pair.internal_id)
        if candles is None:
            print(f"Asset {asset_symbol} lacks OHLCV data")
            continue

        returns = candles["close"].pct_change()
        corr_data[asset_symbol] = returns

    corr_df = pd.DataFrame(corr_data)

    corr_matrix = corr_df.corr(
        method='pearson',
        min_periods=30
    )

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        zmin=-1,
        zmax=1,
        customdata=np.round(corr_matrix, 3),
        hoverongaps=False,
        hovertemplate=(
                '%{x} vs %{y}<br>' +
                'Correlation: %{customdata}<br>' +
                '<extra></extra>'
        ),
        colorscale='RdBu',
    ))

    # Update layout
    fig.update_layout(
        title={
            'text': 'Asset return correlation matrix',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        width=width,
        height=height,
        xaxis_title="Asset",
        yaxis_title="Asset",
        xaxis={'side': 'bottom'},
        yaxis={'autorange': 'reversed'},
    )
    return fig