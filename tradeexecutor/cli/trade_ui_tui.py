"""Textual full-screen TUI for interactive test trade pair selection.

Displays the strategy's trading universe in a navigatable DataTable
with a status bar showing wallet balances. The user picks a pair,
amount and trade mode, then the selections are returned for execution.

Controls:
- Arrow keys (up/down) navigate the table
- Enter confirms the selection
- Esc / q cancels and exits
"""

import logging
from decimal import Decimal

import pandas as pd
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import DataTable, Footer, Static, Input, RadioSet, RadioButton
from textual.containers import Vertical, Horizontal
from textual.screen import ModalScreen
from tradingstrategy.liquidity import LiquidityDataUnavailable

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse

logger = logging.getLogger(__name__)


def _get_tvl_value(strategy_universe: TradingStrategyUniverse, pair: TradingPairIdentifier, tvl_now) -> float | None:
    """Look up raw TVL value for a pair."""
    if not strategy_universe.data_universe.liquidity:
        return None

    try:
        tvl, _ = strategy_universe.data_universe.liquidity.get_liquidity_with_tolerance(
            pair_id=pair.internal_id,
            when=tvl_now,
            tolerance=pd.Timedelta("90D"),
        )
        return float(tvl)
    except LiquidityDataUnavailable:
        return None


def _format_tvl(tvl: float | None) -> Text:
    """Format a TVL value as a right-aligned Rich Text."""
    if tvl is None:
        return Text("N/A", style="dim", justify="right")
    return Text(f"${tvl:,.0f}", justify="right")


def _get_position_info(state: State, pair: TradingPairIdentifier) -> str:
    """Get open position info for a pair, or empty string."""
    position = state.portfolio.get_position_by_trading_pair(pair)
    if position is None:
        return ""
    quantity = position.get_quantity()
    symbol = pair.base.token_symbol
    return f"{quantity} {symbol}"


def _get_pair_symbol(pair: TradingPairIdentifier) -> str:
    """Get display symbol for a pair."""
    if pair.is_vault():
        return pair.get_vault_name() or f"{pair.base.token_symbol}/{pair.quote.token_symbol}"
    return f"{pair.base.token_symbol}/{pair.quote.token_symbol}"


class PairSelectionApp(App):
    """Full-screen Textual app for selecting a trading pair."""

    CSS = """
    #status-bar {
        dock: top;
        height: 1;
        background: $accent;
        color: $text;
        padding: 0 1;
    }
    #pair-table {
        height: 1fr;
    }
    """

    BINDINGS = [
        Binding("escape", "quit_app", "Cancel", show=True),
        Binding("q", "quit_app", "Quit", show=True),
        Binding("enter", "select_pair", "Select", show=True),
    ]

    def __init__(
        self,
        pairs: list[TradingPairIdentifier],
        strategy_universe: TradingStrategyUniverse,
        state: State,
        tvl_now,
        reserve_balance: float,
        reserve_symbol: str,
        gas_balance: float,
    ):
        super().__init__()
        self.pairs = pairs
        self.strategy_universe = strategy_universe
        self.state = state
        self.tvl_now = tvl_now
        self.reserve_balance = reserve_balance
        self.reserve_symbol = reserve_symbol
        self.gas_balance = gas_balance

        # Compute TVL values and sort by TVL descending
        self.tvl_values = {
            id(pair): _get_tvl_value(strategy_universe, pair, tvl_now)
            for pair in pairs
        }
        self.sorted_pairs = sorted(
            pairs,
            key=lambda p: self.tvl_values.get(id(p)) or 0,
            reverse=True,
        )
        self.selected_pair: TradingPairIdentifier | None = None

    def compose(self) -> ComposeResult:
        status_text = (
            f"Reserve: {self.reserve_balance:.4f} {self.reserve_symbol}"
            f"    Gas: {self.gas_balance:.6f}"
            f"    Pairs: {len(self.sorted_pairs)}"
        )
        yield Static(status_text, id="status-bar")
        yield DataTable(id="pair-table", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_column("#", key="idx", width=5)
        table.add_column("Symbol", key="symbol", width=40)
        table.add_column("Exchange", key="exchange", width=20)
        table.add_column("TVL", key="tvl", width=14)
        table.add_column("Position", key="position", width=20)

        for idx, pair in enumerate(self.sorted_pairs, 1):
            symbol = _get_pair_symbol(pair)
            exchange = pair.exchange_name or ""
            tvl = _format_tvl(self.tvl_values.get(id(pair)))
            position = _get_position_info(self.state, pair)

            table.add_row(
                str(idx),
                symbol,
                exchange,
                tvl,
                position,
                key=str(idx),
            )

        table.focus()

    def action_quit_app(self) -> None:
        self.selected_pair = None
        self.exit()

    def action_select_pair(self) -> None:
        table = self.query_one(DataTable)
        if table.cursor_row is not None and table.cursor_row < len(self.sorted_pairs):
            self.selected_pair = self.sorted_pairs[table.cursor_row]
            self.exit()

    @on(DataTable.RowSelected)
    def on_row_selected(self, event: DataTable.RowSelected) -> None:
        row_idx = event.cursor_row
        if row_idx < len(self.sorted_pairs):
            self.selected_pair = self.sorted_pairs[row_idx]
            self.exit()


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
    from rich.console import Console
    from rich.prompt import Prompt

    # Compute TVL timestamp
    tvl_now = None
    if strategy_universe.data_universe.liquidity:
        from eth_defi.compat import native_datetime_utc_now
        now_ = native_datetime_utc_now()
        if strategy_universe.data_universe.liquidity_time_bucket:
            tvl_now = strategy_universe.data_universe.liquidity_time_bucket.floor(pd.Timestamp(now_))
        else:
            tvl_now = now_

    # Full-screen Textual pair selection
    app = PairSelectionApp(
        pairs=pairs,
        strategy_universe=strategy_universe,
        state=state,
        tvl_now=tvl_now,
        reserve_balance=reserve_balance,
        reserve_symbol=reserve_symbol,
        gas_balance=gas_balance,
    )
    app.run()

    selected_pair = app.selected_pair
    if selected_pair is None:
        raise KeyboardInterrupt("Selection cancelled")

    console = Console()
    console.print(f"\nSelected: [bold]{_get_pair_symbol(selected_pair)}[/bold]")

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

    return selected_pair, amount, trade_mode
