"""Vault analysis."""
import logging
from typing import Callable, cast, Iterable

import pandas as pd
from pandas.io.formats.style import Styler

from plotly.graph_objs import Figure
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio

from IPython.display import display


from tradeexecutor.state.identifier import IGNORE_REASON_REQUIRES_WHITELIST, TradingPairIdentifier
from tradeexecutor.state.types import JSONHexAddress
from tradeexecutor.strategy.execution_context import ExecutionMode
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.vault import VaultDepositPermission, VaultUniverse, get_vault_page

logger = logging.getLogger(__name__)


#: Pair ``other_data`` flag marking a vault that needs depositor allow-list approval.
#:
#: This is retained independently of ``ignore_reason`` so the universe can
#: report the whitelist diagnostic even when another constraint, such as
#: incomplete fee data, has already made the pair non-tradeable.
WHITELISTED_VAULT_DIAGNOSTIC_FLAG = "requires_deposit_whitelist"

#: Columns of the table returned by :py:func:`build_whitelisted_vault_dataframe`.
WHITELISTED_VAULT_COLUMNS = ["Name", "Protocol", "Chain", "Address"]


def is_whitelisted_vault(pair: TradingPairIdentifier) -> bool:
    """Check whether a vault requires depositor allow-list approval.

    :param pair:
        Vault pair whose producer metadata has been loaded.
    :return:
        ``True`` when the vault requires whitelisting to deposit.
    """
    if not pair.is_vault():
        return False

    metadata = pair.get_vault_metadata()
    return getattr(metadata, "deposit_permission", None) == VaultDepositPermission.whitelisted


def mark_whitelisted_vaults_ignored(
    strategy_universe: TradingStrategyUniverse,
) -> list[TradingPairIdentifier]:
    """Retain whitelist-gated vaults for diagnostics but exclude them from allocation.

    - Call after constructing the strategy universe and loading vault metadata
    - The diagnostics flag is always set, including when another ignore reason
      already applies to the pair
    - The existing ignore reason is preserved so independent diagnostics, such
      as missing fee data, remain visible
    """
    flagged = []
    for pair in strategy_universe.iterate_pairs():
        if not is_whitelisted_vault(pair):
            continue

        #: Keep an explicit support flag for diagnostics and availability charts.
        pair.other_data[WHITELISTED_VAULT_DIAGNOSTIC_FLAG] = True
        if pair.get_ignore_reason() is None:
            pair.set_ignore_reason(IGNORE_REASON_REQUIRES_WHITELIST)
        flagged.append(pair)

    return flagged


def build_whitelisted_vault_dataframe(
    strategy_universe: TradingStrategyUniverse,
) -> pd.DataFrame:
    """Create a table of universe vaults the strategy cannot allocate to.

    Identifies vaults directly from their published permission metadata, so the
    widget can also inspect a universe before it is marked as data-only. The
    address is used to construct a Trading Strategy vault-page link.
    """
    rows = []
    for pair in strategy_universe.iterate_pairs():
        if not is_whitelisted_vault(pair):
            continue

        metadata = pair.get_vault_metadata()
        rows.append({
            "Name": pair.get_vault_name() or pair.base.token_symbol,
            "Protocol": pair.get_vault_protocol() or getattr(metadata, "protocol_slug", None) or "",
            "Chain": ChainId(pair.chain_id).get_name(),
            "Address": pair.pool_address,
        })

    df = pd.DataFrame(rows, columns=WHITELISTED_VAULT_COLUMNS)
    return df.sort_values(["Chain", "Name"]).reset_index(drop=True)


def style_whitelisted_vault_table(df: pd.DataFrame) -> Styler:
    """Render whitelist-gated vaults with Trading Strategy vault-page links."""

    def _linkify_address(address: str) -> str:
        if not address:
            return ""
        return f'<a href="{get_vault_page(address)}" target="_blank">{address}</a>'

    return df.style.format({"Address": _linkify_address}, escape="html").hide(axis="index")


def render_whitelisted_vaults(
    strategy_universe: TradingStrategyUniverse,
) -> Styler:
    """Render the whitelist-gated vault diagnostics widget for notebooks."""
    return style_whitelisted_vault_table(build_whitelisted_vault_dataframe(strategy_universe))


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
    max_count: int = 3,
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

    if max_count is not None:
        vault_pairs = vault_pairs[:max_count]

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
    vaults: list[tuple[int, str]] | VaultUniverse,
    strategy_universe: TradingStrategyUniverse,
    execution_mode: ExecutionMode,
    printer: Callable,
):
    """Dump vault diagnostics for the strategy universe in create_trading_universe()"""
    data = []

    from eth_defi.chain import get_chain_name

    if isinstance(vaults, VaultUniverse):
        vaults: Iterable[tuple[ChainId, JSONHexAddress]] = vaults.vaults.keys()

    for v in vaults:
        vault_error = strategy_universe.get_vault_error(v)
        vault_pair = strategy_universe.get_pair_by_smart_contract(
            v[1]
        )
        data.append({
            "Chain": get_chain_name(v[0]),
            "Vault": v[1],
            "Pair id": vault_pair.internal_id if vault_pair else "-",
            "Name": vault_pair.get_vault_name() if vault_pair else "-",
            "Protocol": vault_pair.get_vault_protocol() if vault_pair else "-",
            "Denomination": vault_pair.quote.token_symbol if vault_pair else "-",
            "Status": vault_error or "OK",
        })

    printer("Vault check list")
    df = pd.DataFrame(data)
    if execution_mode.is_live_trading():
        # Do not let pandas abbreviate pair details in Docker logs.
        printer(df.to_string(index=False, max_colwidth=None))
    else:
        # Backtesting uses HTML output
        display(df)
