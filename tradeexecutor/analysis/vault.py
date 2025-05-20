"""Vault analysis."""
import logging

import pandas as pd

from plotly.graph_objs import Figure
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio


from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


logger = logging.getLogger(__name__)


def plot_vault(
    pair: TradingPairIdentifier,
    price: pd.Series,
    tvl: pd.Series,
):
    assert isinstance(pair, TradingPairIdentifier)
    assert isinstance(price, pd.Series)
    assert isinstance(tvl, pd.Series)

    name = pair.get_vault_name()

    logger.info(f"Examining vault {name}: {id}, having {len(price):,} pirce rows")
    nav_series = tvl
    price_series = price
    daily_returns = price_series.pct_change()
    denomination = pair.quote.token_symbol

    # Calculate cumulative returns (what $1 would grow to)
    cumulative_returns = (1 + daily_returns).cumprod()

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add cumulative returns trace on a separate y-axis (share same axis as share price)
    fig.add_trace(
        go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns.values,
            name="Cumulative returns (cleaned)",
            line=dict(color='darkgreen', width=4),
            opacity=0.75
        ),
        secondary_y=False,
    )

    # Add share price trace on primary y-axis
    fig.add_trace(
        go.Scatter(
            x=price_series.index,
            y=price_series.values,
            name="Share Price",
            line=dict(color='green', width=4, dash='dash'),
            opacity=0.75

        ),
        secondary_y=False,
    )

    # Add NAV trace on secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=nav_series.index,
            y=nav_series.values,
            name="TVL",
            line=dict(color='blue', width=4),
            opacity=0.75

        ),
        secondary_y=True,
    )

    # Set titles and labels
    fig.update_layout(
        title_text=f"{name} - Returns TVL and share price",
        hovermode="x unified",
        template=pio.templates.default,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )

    # Set y-axes titles
    fig.update_yaxes(title_text=f"Share Price ({denomination})", secondary_y=False)
    fig.update_yaxes(title_text=f"TVL ({denomination})", secondary_y=True)
    return fig



def visualise_vaults(
    strategy_universe: TradingStrategyUniverse,
) -> list[Figure]:
    """Visualise vaults used in the strategy universe.

    - Plots cumulative returns and TVL of all vaults.
    - Each vault gets its own figure

    :return:
        Plotly figure for returns and TVL for all vaults
    """

    vault_pairs = [p for p in strategy_universe.iterate_pairs() if p.is_vault()]

    figures = []

    for pair in vault_pairs:
        candles = strategy_universe.data_universe.candles.get_candles_by_pair(pair.internal_id)
        price = candles["close"]
        liquidity_candles = strategy_universe.data_universe.liquidity.get_liquidity_samples_by_pair(pair.internal_id)
        tvl = liquidity_candles["close"]

        figures.append(
            plot_vault(
                pair,
                price,
                tvl
            )
        )
    return figures

