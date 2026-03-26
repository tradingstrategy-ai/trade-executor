"""Textual full-screen TUI for interactive test trade pair selection.

Displays the strategy's trading universe in a navigatable DataTable
with a status bar showing wallet balances. On Enter, a dialog lets
the user choose trade mode (buy/sell, buy only, sell only) and amount.

Controls:
- Arrow keys (up/down) navigate the table
- Enter opens the trade dialog for the highlighted pair
- Esc / q cancels and exits
"""

import datetime
import logging
from decimal import Decimal, InvalidOperation

import pandas as pd
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Input,
    Label,
    RadioButton,
    RadioSet,
    Static,
)
from eth_defi.compat import native_datetime_utc_now
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


def _format_deposits_open(pair: TradingPairIdentifier) -> Text:
    """Format the deposit status for a vault pair.

    Non-vault pairs show a dash.  Vault pairs show ``Yes`` / ``No``
    based on :py:meth:`TradingPairIdentifier.can_deposit`.
    """
    if not pair.is_vault():
        return Text("-", style="dim", justify="center")
    if pair.can_deposit():
        return Text("Yes", style="green", justify="center")
    reason = pair.get_deposit_closed_reason() or "closed"
    return Text(f"No", style="red bold", justify="center")


def _get_price(pricing_model, pair: TradingPairIdentifier, ts: datetime.datetime | None = None) -> float | None:
    """Get the approximate mid-price for a pair via the pricing model.

    For Hypercore vaults this returns the share price from the data
    pipeline (not a live on-chain price). Returns ``None`` when the
    price is unavailable or is the uninformative 1.0 fallback.

    :param ts:
        Timestamp to query. Callers should pass a single timestamp
        when querying multiple pairs to avoid redundant clock reads.
    """
    if pricing_model is None:
        return None
    try:
        if ts is None:
            ts = native_datetime_utc_now()
        price = pricing_model.get_mid_price(ts, pair)
        # Filter out the 1.0 fallback — it means no real price data
        if price is not None and price == 1.0:
            return None
        return price
    except Exception as e:
        logger.debug("Could not get price for %s: %s", pair, e)
        return None


def _format_price(price: float | None) -> Text:
    """Format a price value as a right-aligned Rich Text."""
    if price is None:
        return Text("N/A", style="dim", justify="right")
    if price >= 1000:
        return Text(f"${price:,.0f}", justify="right")
    if price >= 1:
        return Text(f"${price:,.2f}", justify="right")
    return Text(f"${price:,.4f}", justify="right")


def _get_position_info(state: State, pair: TradingPairIdentifier) -> str:
    """Get open position info for a pair, or empty string."""
    position = state.portfolio.get_position_by_trading_pair(pair)
    if position is None:
        return ""
    quantity = position.get_quantity()
    symbol = pair.base.token_symbol
    return f"{quantity} {symbol}"


def _format_lockup(state: State, pair: TradingPairIdentifier) -> Text:
    """Format vault lockup time remaining from the stored expiry timestamp.

    Shows remaining hours until the lockup expires, or ``Unlocked``
    if the lockup has already passed. Non-vault pairs or positions
    without lockup data show a dash.
    """
    if not pair.is_vault():
        return Text("-", style="dim", justify="right")

    position = state.portfolio.get_position_by_trading_pair(pair)
    if position is None:
        return Text("-", style="dim", justify="right")

    expires_at_str = position.other_data.get("vault_lockup_expires_at")
    if expires_at_str is None:
        return Text("-", style="dim", justify="right")

    try:
        expires_at = datetime.datetime.fromisoformat(expires_at_str)
    except (ValueError, TypeError):
        return Text("-", style="dim", justify="right")

    now = native_datetime_utc_now()
    remaining = expires_at - now
    if remaining.total_seconds() <= 0:
        return Text("Unlocked", style="green", justify="right")

    hours = remaining.total_seconds() / 3600
    if hours >= 24:
        days = hours / 24
        return Text(f"{days:.1f}d", style="yellow", justify="right")
    return Text(f"{hours:.1f}h", style="yellow", justify="right")


def _get_pair_symbol(pair: TradingPairIdentifier) -> str:
    """Get display symbol for a pair."""
    if pair.is_vault():
        return pair.get_vault_name() or f"{pair.base.token_symbol}/{pair.quote.token_symbol}"
    return f"{pair.base.token_symbol}/{pair.quote.token_symbol}"


# ---------------------------------------------------------------------------
# Trade dialog (modal)
# ---------------------------------------------------------------------------

class TradeDialog(ModalScreen):
    """Modal dialog for choosing trade mode and amount."""

    CSS = """
    TradeDialog {
        align: center middle;
    }
    #trade-dialog {
        width: 60;
        height: auto;
        max-height: 20;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #trade-dialog Label {
        margin-bottom: 1;
    }
    #trade-dialog #amount-input {
        margin-bottom: 1;
    }
    #trade-dialog #error-label {
        color: $error;
        height: 1;
        margin-bottom: 1;
    }
    #button-row {
        height: 3;
        align: right middle;
    }
    #button-row Button {
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(
        self,
        pair_symbol: str,
        reserve_symbol: str,
        default_mode: str,
        min_amount: Decimal | None = None,
        position_value: Decimal | None = None,
    ):
        super().__init__()
        self.pair_symbol = pair_symbol
        self.reserve_symbol = reserve_symbol
        self.default_mode = default_mode
        self.min_amount = min_amount
        self.position_value = position_value
        self.result: tuple[str, Decimal] | None = None

    def compose(self) -> ComposeResult:
        mode_index = {"open_close": 0, "open": 1, "close": 2}.get(self.default_mode, 0)
        with Vertical(id="trade-dialog"):
            yield Label(f"Test trade: [bold]{self.pair_symbol}[/bold]")
            yield Label("Trade mode:")
            with RadioSet(id="mode-radio"):
                yield RadioButton("Buy and sell (open + close)", value=mode_index == 0)
                yield RadioButton("Buy only (open position)", value=mode_index == 1)
                yield RadioButton("Sell all (close full position)", value=mode_index == 2)
            yield Label(f"Amount ({self.reserve_symbol}):")
            if self.default_mode == "close" and self.position_value is not None:
                default_amount = str(self.position_value)
            elif self.min_amount:
                default_amount = str(self.min_amount)
            else:
                default_amount = "5"
            # Close mode always sells the full position — the amount field
            # is ignored, so disable it to avoid misleading the operator.
            amount_disabled = (self.default_mode == "close")
            yield Input(value=default_amount, id="amount-input", type="number", disabled=amount_disabled)
            yield Label("", id="error-label")
            with Horizontal(id="button-row"):
                yield Button("Cancel", variant="default", id="cancel-btn")
                yield Button("Execute", variant="primary", id="execute-btn")

    def on_mount(self) -> None:
        self.query_one("#amount-input", Input).focus()

    @on(RadioSet.Changed)
    def on_mode_changed(self, event: RadioSet.Changed) -> None:
        amount_input = self.query_one("#amount-input", Input)
        # Close mode always sells the full position — amount is ignored,
        # so disable the input to avoid misleading the operator.
        amount_input.disabled = (event.pressed_index == 2)

    @on(Button.Pressed, "#cancel-btn")
    def cancel_pressed(self) -> None:
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#execute-btn")
    def execute_pressed(self) -> None:
        self._submit()

    @on(Input.Submitted)
    def input_submitted(self) -> None:
        self._submit()

    def _submit(self) -> None:
        radio_set = self.query_one("#mode-radio", RadioSet)
        pressed_index = radio_set.pressed_index
        # Index 2 ("Sell all") always fully closes the position via
        # close_all — the amount field is disabled and ignored.
        mode_map = {0: "open_close", 1: "open", 2: "close_all"}
        trade_mode = mode_map.get(pressed_index, "open_close")

        if trade_mode == "close_all":
            # Sell all: amount is irrelevant — the backend always closes the
            # full position.  Use position_value as a placeholder so the
            # downstream code has a non-zero Decimal to carry around.
            amount = self.position_value or Decimal("0")
            self.dismiss((trade_mode, amount))
            return

        # For buy modes, validate the amount input.
        amount_str = self.query_one("#amount-input", Input).value.strip()
        try:
            amount = Decimal(amount_str)
            if amount <= 0:
                raise InvalidOperation()
        except (InvalidOperation, ValueError):
            self.query_one("#error-label", Label).update("Invalid amount — enter a positive number")
            return

        if self.min_amount is not None and amount < self.min_amount:
            self.query_one("#error-label", Label).update(
                f"Minimum amount is {self.min_amount} {self.reserve_symbol} "
                f"(includes activation cost on first deposit)"
            )
            return

        self.dismiss((trade_mode, amount))


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

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
    #hint-bar {
        dock: bottom;
        height: 1;
        background: $primary;
        color: $text;
        padding: 0 1;
        text-style: bold;
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
        is_hyperliquid: bool,
        pricing_model=None,
    ):
        super().__init__()
        self.pairs = pairs
        self.strategy_universe = strategy_universe
        self.state = state
        self.tvl_now = tvl_now
        self.reserve_balance = reserve_balance
        self.reserve_symbol = reserve_symbol
        self.gas_balance = gas_balance
        self.is_hyperliquid = is_hyperliquid
        self.pricing_model = pricing_model

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

        # Fetch prices — only for vault pairs (candle-based, no RPC calls).
        # Non-vault pairs would trigger on-chain price queries that fail
        # or retry endlessly on Anvil forks.
        self.prices = {}
        price_ts = native_datetime_utc_now()
        for pair in self.sorted_pairs:
            if pair.is_vault():
                self.prices[id(pair)] = _get_price(pricing_model, pair, ts=price_ts)
            else:
                self.prices[id(pair)] = None

        # Result set by the trade dialog
        self.selected_pair: TradingPairIdentifier | None = None
        self.trade_mode: str | None = None
        self.amount: Decimal | None = None

    def compose(self) -> ComposeResult:
        status_text = (
            f"Reserve: {self.reserve_balance:.4f} {self.reserve_symbol}"
            f"    Gas: {self.gas_balance:.6f}"
            f"    Pairs: {len(self.sorted_pairs)}"
        )
        yield Static(status_text, id="status-bar")
        yield DataTable(id="pair-table", cursor_type="row")
        yield Static("Choose pair with Enter to perform a test trade", id="hint-bar")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_column("#", key="idx", width=5)
        table.add_column("Symbol", key="symbol", width=40)
        table.add_column("Exchange", key="exchange", width=20)
        table.add_column("Price", key="price", width=14)
        table.add_column("TVL", key="tvl", width=14)
        table.add_column("Deposits", key="deposits", width=14)
        table.add_column("Position", key="position", width=20)
        table.add_column("Lockup", key="lockup", width=10)

        for idx, pair in enumerate(self.sorted_pairs, 1):
            symbol = _get_pair_symbol(pair)
            exchange = pair.exchange_name or ""
            price = _format_price(self.prices.get(id(pair)))
            tvl = _format_tvl(self.tvl_values.get(id(pair)))
            deposits = _format_deposits_open(pair)
            position = _get_position_info(self.state, pair)
            lockup = _format_lockup(self.state, pair)

            table.add_row(
                str(idx),
                symbol,
                exchange,
                price,
                tvl,
                deposits,
                position,
                lockup,
                key=str(idx),
            )

        table.focus()

    def action_quit_app(self) -> None:
        self.exit()

    def action_select_pair(self) -> None:
        table = self.query_one(DataTable)
        if table.cursor_row is not None and table.cursor_row < len(self.sorted_pairs):
            self._open_trade_dialog(table.cursor_row)

    @on(DataTable.RowSelected)
    def on_row_selected(self, event: DataTable.RowSelected) -> None:
        self._open_trade_dialog(event.cursor_row)

    def _open_trade_dialog(self, row_idx: int) -> None:
        pair = self.sorted_pairs[row_idx]
        position = self.state.portfolio.get_position_by_trading_pair(pair)
        has_open_position = position is not None

        if has_open_position:
            default_mode = "close"
            position_value = Decimal(str(position.get_value()))
        else:
            position_value = None
            if self.is_hyperliquid:
                default_mode = "open"
            else:
                default_mode = "open_close"

        # On Hyperliquid the first vault deposit requires a 2 USDC
        # activation cost on top of the 5 USDC minimum deposit.
        min_amount = Decimal("7") if self.is_hyperliquid else None

        dialog = TradeDialog(
            pair_symbol=_get_pair_symbol(pair),
            reserve_symbol=self.reserve_symbol,
            default_mode=default_mode,
            min_amount=min_amount,
            position_value=position_value,
        )

        def on_dialog_dismiss(result: tuple[str, Decimal] | None) -> None:
            if result is not None:
                self.selected_pair = pair
                self.trade_mode, self.amount = result
                self.exit()

        self.push_screen(dialog, on_dialog_dismiss)


def display_pair_selection_ui(
    pairs: list[TradingPairIdentifier],
    strategy_universe: TradingStrategyUniverse,
    reserve_balance: float,
    reserve_symbol: str,
    gas_balance: float,
    state: State,
    is_hyperliquid: bool,
    pricing_model=None,
) -> tuple[TradingPairIdentifier, Decimal, str]:
    """Display the interactive pair selection TUI.

    :return:
        Tuple of (selected_pair, amount, trade_mode) where trade_mode
        is ``"open_close"``, ``"open"``, ``"close"`` or ``"close_all"``.
        ``"close_all"`` is returned when the sell amount is >= 98%
        of the open position value.
    """
    # Compute TVL timestamp
    tvl_now = None
    if strategy_universe.data_universe.liquidity:
        now_ = native_datetime_utc_now()
        if strategy_universe.data_universe.liquidity_time_bucket:
            tvl_now = strategy_universe.data_universe.liquidity_time_bucket.floor(pd.Timestamp(now_))
        else:
            tvl_now = now_

    app = PairSelectionApp(
        pairs=pairs,
        strategy_universe=strategy_universe,
        state=state,
        tvl_now=tvl_now,
        reserve_balance=reserve_balance,
        reserve_symbol=reserve_symbol,
        gas_balance=gas_balance,
        is_hyperliquid=is_hyperliquid,
        pricing_model=pricing_model,
    )
    app.run()

    if app.selected_pair is None:
        raise KeyboardInterrupt("Selection cancelled")

    return app.selected_pair, app.amount, app.trade_mode
