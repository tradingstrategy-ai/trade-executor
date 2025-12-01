"""Vault analysis."""
import logging
from typing import Callable, cast

import pandas as pd

from plotly.graph_objs import Figure
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio

from IPython.display import display


from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.strategy.execution_context import ExecutionMode
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

    assert isinstance(price.index, pd.DatetimeIndex), f"Price index is not a DatetimeIndex, got {type(price.index)}"
    assert isinstance(tvl.index, pd.DatetimeIndex), f"TVL index is not a DatetimeIndex, got {type(tvl.index)}"

    name = pair.get_vault_name()
    symbol = pair.base.token_symbol

    logger.info(f"Examining vault {name}: {id}, having {len(price):,} pirce rows")
    nav_series = tvl
    price_series = price

    daily_returns = price_series.pct_change()
    denomination = pair.quote.token_symbol

    # Calculate cumulative returns (what $1 would grow to)
    cumulative_returns = (1 + daily_returns).cumprod()

    df = pd.DataFrame({
        "cumulative_returns": cumulative_returns,
        "share_price": price_series,
        "tvl": nav_series
    })

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add cumulative returns trace on a separate y-axis (share same axis as share price)
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df.cumulative_returns,
            name="Cumulative returns (cleaned)",
            line=dict(color='darkgreen', width=4),
            opacity=0.75
        ),
        secondary_y=False,
    )

    # Add share price trace on primary y-axis
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df.share_price,
            name="Share Price",
            line=dict(color='green', width=4, dash='dash'),
            opacity=0.75

        ),
        secondary_y=False,
    )

    # Add NAV trace on secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df.tvl,
            name="TVL",
            line=dict(color='blue', width=4),
            opacity=0.75

        ),
        secondary_y=True,
    )

    # Set titles and labels
    fig.update_layout(
        title_text=f"{name} ({symbol}) - Returns, TVL and share price",
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
    printer: Callable=logger.warning,
) -> list[Figure]:
    """Visualise vaults used in the strategy universe.

    - Plots cumulative returns and TVL of all vaults.
    - Each vault gets its own figure

    :param printer:
        Logger to used for warnings.

        Use print() in notebooks.

    :return:
        Plotly figure for returns and TVL for all vaults
    """

    vault_pairs = [p for p in strategy_universe.iterate_pairs() if p.is_vault()]
    if not vault_pairs:
        raise ValueError("No vault pairs found in strategy universe")

    figures = []

    for pair in vault_pairs:
        candles = strategy_universe.data_universe.candles.get_candles_by_pair(pair.internal_id)

        if candles is None:
            printer(f"No candles found for pair {pair}")
            continue

        price = candles["close"]
        liquidity_candles = strategy_universe.data_universe.liquidity.get_liquidity_samples_by_pair(pair.internal_id)
        tvl = liquidity_candles["close"]

        # (pair_id, timestamp) -> (timestamp) conversion if needed
        if isinstance(tvl.index, pd.MultiIndex):
            tvl.index = tvl.index.get_level_values('timestamp')
            tvl.index = pd.to_datetime(tvl.index)
        elif isinstance(tvl.index, pd.DatetimeIndex):
            pass
        else:
            raise NotImplementedError()

        if tvl is None:
            printer(f"No liquidity data found for pair {pair}")
            continue

        # Because liquidity data is 1d we might need to resample it to price freq
        price_freq = pd.infer_freq(price.index)
        tvl = tvl.resample(price_freq).ffill()

        figures.append(
            plot_vault(
                pair,
                price,
                tvl
            )
        )
    return figures


def display_vaults(
    vaults: list[tuple[int, str]],
    strategy_universe: TradingStrategyUniverse,
    execution_mode: ExecutionMode,
    printer: Callable,
):
    """Dump vault diagnostics for the strategy universe in create_trading_universe()"""
    data = []

    from eth_defi.chain import get_chain_name

    for v in vaults:
        vault_error = strategy_universe.get_vault_error(v)
        vault_pair = strategy_universe.get_pair_by_smart_contract(
            v[1]
        )
        data.append({
            "Chain": get_chain_name(v[0]),
            "Vault": v[1],
            "Name": vault_pair.get_vault_name() if vault_pair else "-",
            "Protocol": vault_pair.get_vault_protocol() if vault_pair else "-",
            "Status": vault_error or "OK",
        })

    printer("Vault check list")
    df = pd.DataFrame(data)
    if execution_mode.is_live_trading():
        # In live trading we can display dataframes directly
        printer(df)
    else:
        # Backtesting uses HTML output
        display(df)
