"""Rich-based TUI for interactive test trade pair selection.

Displays the strategy's trading universe in a table with balances,
lets the user pick a pair, amount and trade mode, then returns
the selections for the command module to execute.
"""

import logging
from decimal import Decimal

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt
from rich.table import Table
from tradingstrategy.chain import ChainId
from tradingstrategy.liquidity import LiquidityDataUnavailable

from tradeexecutor.cli.commands.pair_mapping import construct_identifier_from_trading_pair
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse

logger = logging.getLogger(__name__)


def _get_tvl(strategy_universe: TradingStrategyUniverse, pair: TradingPairIdentifier, tvl_now) -> str:
    """Look up TVL for a pair, returning a formatted string."""
    if not strategy_universe.data_universe.liquidity:
        return "N/A"

    try:
        tvl, _ = strategy_universe.data_universe.liquidity.get_liquidity_with_tolerance(
            pair_id=pair.internal_id,
            when=tvl_now,
            tolerance=pd.Timedelta("90D"),
        )
        return f"${tvl:,.0f}"
    except LiquidityDataUnavailable:
        return "N/A"


def _get_position_info(state: State, pair: TradingPairIdentifier) -> str:
    """Get open position info for a pair, or empty string."""
    position = state.portfolio.get_position_by_trading_pair(pair)
    if position is None:
        return ""
    quantity = position.get_quantity()
    symbol = pair.base.token_symbol
    return f"{quantity} {symbol}"


def display_pair_selection_ui(
    pairs: list[TradingPairIdentifier],
    strategy_universe: TradingStrategyUniverse,
    reserve_balance: float,
    reserve_symbol: str,
    gas_balance: float,
    state: State,
    is_hyperliquid: bool,
) -> tuple[TradingPairIdentifier, Decimal, str]:
    """Display the interactive pair selection TUI.

    :return:
        Tuple of (selected_pair, amount, trade_mode) where trade_mode
        is ``"open_close"``, ``"open"`` or ``"close"``.
    """
    console = Console()

    # Header panel with balances
    header_text = (
        f"Reserve: [bold]{reserve_balance:.4f} {reserve_symbol}[/bold]    "
        f"Gas: [bold]{gas_balance:.6f}[/bold]"
    )
    console.print(Panel(header_text, title="Wallet balances"))

    # Build the pairs table
    tvl_now = None
    if strategy_universe.data_universe.liquidity:
        from eth_defi.compat import native_datetime_utc_now
        now_ = native_datetime_utc_now()
        if strategy_universe.data_universe.liquidity_time_bucket:
            tvl_now = strategy_universe.data_universe.liquidity_time_bucket.floor(pd.Timestamp(now_))
        else:
            tvl_now = now_

    table = Table(title="Trading pairs")
    table.add_column("#", justify="right", style="cyan")
    table.add_column("Symbol", style="bold")
    table.add_column("Exchange", style="dim")
    table.add_column("TVL", justify="right")
    table.add_column("Position", style="yellow")

    for idx, pair in enumerate(pairs, 1):
        if pair.is_vault():
            symbol = pair.get_vault_name() or f"{pair.base.token_symbol}/{pair.quote.token_symbol}"
        else:
            symbol = f"{pair.base.token_symbol}/{pair.quote.token_symbol}"

        exchange = pair.exchange_name or ""
        tvl = _get_tvl(strategy_universe, pair, tvl_now)
        position_info = _get_position_info(state, pair)

        table.add_row(str(idx), symbol, exchange, tvl, position_info)

    console.print(table)

    # Prompt: pair selection
    pair_idx = IntPrompt.ask(
        f"Select pair [1-{len(pairs)}]",
        default=1,
        console=console,
    )
    while pair_idx < 1 or pair_idx > len(pairs):
        console.print(f"[red]Please enter a number between 1 and {len(pairs)}[/red]")
        pair_idx = IntPrompt.ask(
            f"Select pair [1-{len(pairs)}]",
            default=1,
            console=console,
        )
    selected_pair = pairs[pair_idx - 1]

    # Prompt: amount
    amount_str = Prompt.ask(
        f"Amount in {reserve_symbol}",
        default="5",
        console=console,
    )
    amount = Decimal(amount_str)

    # Prompt: trade mode
    has_open_position = state.portfolio.get_position_by_trading_pair(selected_pair) is not None

    if has_open_position:
        default_mode = "3"
    elif is_hyperliquid:
        default_mode = "2"
    else:
        default_mode = "1"

    console.print("\nTrade mode:")
    console.print("  [1] open + close")
    console.print("  [2] open only")
    console.print("  [3] close only")

    mode_str = Prompt.ask(
        "Choose trade mode",
        choices=["1", "2", "3"],
        default=default_mode,
        console=console,
    )

    mode_map = {"1": "open_close", "2": "open", "3": "close"}
    trade_mode = mode_map[mode_str]

    # Output the equivalent CLI command
    pair_id_str = construct_identifier_from_trading_pair(selected_pair)
    buy_only_flag = " --buy-only" if trade_mode == "open" else ""
    console.print(
        f"\n[dim]Equivalent CLI command:[/dim]\n"
        f"  trade-executor perform-test-trade "
        f"--pair \"{pair_id_str}\" "
        f"--amount={amount}"
        f"{buy_only_flag}\n"
    )

    return selected_pair, amount, trade_mode
