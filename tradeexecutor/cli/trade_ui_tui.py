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

from tradingstrategy.chain import ChainId

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeStatus
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse

logger = logging.getLogger(__name__)

# Sentinel to distinguish "no live status was queried" (fall back to pair
# metadata) from an explicit ``None`` (live check failed → show unknown).
_NO_LIVE_STATUS = object()


def _get_deposit_status(pricing_model, pair: TradingPairIdentifier, ts: datetime.datetime | None = None) -> bool | None:
    """Get the current deposit status for a vault pair.

    Uses the live pricing model when available, falling back to the
    data-pipeline metadata snapshot stored on the pair.

    Returns ``True`` (open), ``False`` (closed), or ``None`` (unknown /
    live check failed).
    """
    if not pair.is_vault():
        return None

    if pricing_model is not None:
        try:
            return pricing_model.can_deposit(ts, pair)
        except Exception as e:
            logger.warning("Could not get live deposit status for %s, showing unknown: %s", pair, e)
            if pair.get_deposit_closed_reason() is not None:
                return False
            return None

    return pair.can_deposit()


def _get_redemption_status(pricing_model, pair: TradingPairIdentifier, ts: datetime.datetime | None = None) -> bool | None:
    """Get the current redemption status for a vault pair.

    Uses the live pricing model when available, falling back to the
    data-pipeline metadata snapshot stored on the pair.

    Returns ``True`` (open), ``False`` (closed), or ``None`` (unknown /
    live check failed).
    """
    if not pair.is_vault():
        return None

    if pricing_model is not None:
        try:
            return pricing_model.can_redeem(ts, pair)
        except Exception as e:
            logger.warning("Could not get live redemption status for %s, showing unknown: %s", pair, e)
            if pair.get_redemption_closed_reason() is not None:
                return False
            return None

    return pair.can_redeem()


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


def _format_deposits_open(pair: TradingPairIdentifier, deposit_status=_NO_LIVE_STATUS) -> Text:
    """Format the deposit status for a vault pair.

    Non-vault pairs show a dash.  Vault pairs show ``Yes`` / ``No``
    based on the live pricing model status or pair metadata.
    """
    return _format_vault_open_status(pair, deposit_status, pair.can_deposit)


def _format_redemptions_open(pair: TradingPairIdentifier, redemption_status=_NO_LIVE_STATUS) -> Text:
    """Format the redemption status for a vault pair.

    Non-vault pairs show a dash. Vault pairs show ``Yes`` / ``No``
    based on the live pricing model status or pair metadata.
    """
    return _format_vault_open_status(pair, redemption_status, pair.can_redeem)


def _format_vault_open_status(pair: TradingPairIdentifier, status, fallback_status) -> Text:
    """Format a vault open/closed status for the pair table."""
    if not pair.is_vault():
        return Text("-", style="dim", justify="center")
    if status is _NO_LIVE_STATUS:
        status = fallback_status()
    if status is None:
        return Text("?", style="yellow", justify="center")
    if status:
        return Text("Yes", style="green", justify="center")
    return Text("No", style="red bold", justify="center")


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


def _format_decimal_amount(amount: Decimal) -> str:
    """Format a Decimal for compact terminal display."""
    return format(amount.normalize(), "f")


def _get_position_info(state: State, pair: TradingPairIdentifier) -> str:
    """Get open position info for a pair, or empty string."""
    position = state.portfolio.get_position_by_trading_pair(pair, pending=True)
    if position is None:
        return ""
    quantity = position.get_quantity()
    pending_trade = _get_pending_settlement_trade(position)
    if quantity == 0 and pending_trade is not None:
        return f"{_format_decimal_amount(pending_trade.planned_reserve)} {pending_trade.reserve_currency.token_symbol}"
    symbol = pair.base.token_symbol
    return f"{_format_decimal_amount(quantity)} {symbol}"


def _format_remaining(remaining: datetime.timedelta, prefix: str = "") -> str:
    """Format a positive timedelta as a compact ``18.5h`` / ``2.1d`` string."""
    hours = remaining.total_seconds() / 3600
    if hours >= 24:
        return f"{prefix}{hours / 24:.1f}d"
    return f"{prefix}{hours:.1f}h"


def _format_lockup_days(days: float, prefix: str = "") -> str:
    """Format a metadata-only lockup duration as compact days or hours."""
    if days >= 1:
        return f"{prefix}{days:.1f}d"
    return f"{prefix}{days * 24:.1f}h"


def _format_vault_display_flags(pair: TradingPairIdentifier) -> Text | None:
    """Format generic vault warning flags for the Lockup column fallback."""
    display_flags = pair.other_data.get("vault_display_flags") if pair.other_data else None
    if not display_flags:
        return None

    red_count = sum(
        1 for flag in display_flags
        if isinstance(flag, dict) and flag.get("severity") == "red"
    )
    yellow_count = sum(
        1 for flag in display_flags
        if isinstance(flag, dict) and flag.get("severity") == "yellow"
    )

    parts = []
    if red_count:
        parts.append(f"red {red_count}")
    if yellow_count:
        parts.append(f"yellow {yellow_count}")

    if not parts:
        return None

    style = "red" if red_count else "yellow"
    return Text(" ".join(parts), style=style, justify="right")


def _format_empty_lockup(pair: TradingPairIdentifier) -> Text:
    """Format an empty Lockup column cell, falling back to vault warnings."""
    return _format_vault_display_flags(pair) or Text("-", style="dim", justify="right")


def _get_pending_settlement_trade(position: TradingPosition):
    """Return the first pending async-deposit trade for a position, or ``None``.

    A deposit that has been requested on-chain but not yet settled sits in
    :py:attr:`TradeStatus.vault_settlement_pending`.
    """
    for trade in position.trades.values():
        if trade.is_buy() and trade.get_status() == TradeStatus.vault_settlement_pending:
            return trade
    return None


def _format_lockup(state: State, pair: TradingPairIdentifier) -> Text:
    """Format the Lockup column for a vault pair.

    Priority:

    1. If an async deposit is pending settlement, show the estimated
       settlement eligibility time (``eligible 18.5h``), ``settling`` when the stored
       estimate has already passed, or ``pending`` when no on-chain ETA is
       available (operator-driven ERC-7540 vaults like Lagoon).
    2. Otherwise show the lockup time remaining from the stored expiry
       timestamp, or ``Unlocked`` once it has passed. Estimated timestamps are
       prefixed with ``~``.
    3. If no position timestamp exists, show a static estimated duration from
       pair metadata when available.
    4. If no lockup data exists, show generic vault warning flags when present.

    Non-vault pairs or positions without any of this data show a dash.
    """
    if not pair.is_vault():
        return Text("-", style="dim", justify="right")

    # Include pending positions — an unsettled first deposit may not yet be
    # an open position.
    position = state.portfolio.get_position_by_trading_pair(pair, pending=True)
    if position is None:
        return _format_empty_lockup(pair)

    now = native_datetime_utc_now()

    # 1. Pending async deposit settlement takes priority over lockup.
    pending_trade = _get_pending_settlement_trade(position)
    if pending_trade is not None:
        settles_at_str = pending_trade.other_data.get("vault_settlement_estimated_at")
        if settles_at_str is None:
            return Text("pending", style="yellow", justify="right")
        try:
            settles_at = datetime.datetime.fromisoformat(settles_at_str)
        except (ValueError, TypeError):
            return Text("pending", style="yellow", justify="right")
        remaining = settles_at - now
        if remaining.total_seconds() <= 0:
            # Estimate has passed — settlement is due / in progress.
            return Text("settling", style="yellow", justify="right")
        return Text(_format_remaining(remaining, prefix="eligible "), style="yellow", justify="right")

    # 2. Fall back to lockup expiry.
    expires_at_str = position.other_data.get("vault_lockup_expires_at")
    if expires_at_str is not None:
        try:
            expires_at = datetime.datetime.fromisoformat(expires_at_str)
        except (ValueError, TypeError):
            expires_at = None

        if expires_at is not None:
            remaining = expires_at - now
            if remaining.total_seconds() <= 0:
                return Text("Unlocked", style="green", justify="right")
            prefix = "~" if position.other_data.get("vault_lockup_estimated") is True else ""
            return Text(_format_remaining(remaining, prefix=prefix), style="yellow", justify="right")

    # 3. Metadata-only fallback. This is static and does not decay.
    lockup_days = pair.other_data.get("vault_lockup_days") if pair.other_data else None
    if lockup_days is not None:
        try:
            lockup_days = float(lockup_days)
        except (TypeError, ValueError):
            lockup_days = None
        if lockup_days is not None and lockup_days > 0:
            return Text(_format_lockup_days(lockup_days, prefix="~"), style="yellow", justify="right")

    return _format_empty_lockup(pair)


def _get_pair_symbol(pair: TradingPairIdentifier) -> str:
    """Get display symbol for a pair."""
    if pair.is_vault():
        return pair.get_vault_name() or f"{pair.base.token_symbol}/{pair.quote.token_symbol}"
    return f"{pair.base.token_symbol}/{pair.quote.token_symbol}"


# ---------------------------------------------------------------------------
# Trade dialog (modal)
# ---------------------------------------------------------------------------

class TradeModeRadioSet(RadioSet):
    """RadioSet where keyboard navigation immediately chooses the highlighted mode."""

    BINDINGS = [
        Binding("down,right", "choose_next", "Next option", show=False),
        Binding("enter,space", "toggle_button", "Toggle", show=False),
        Binding("up,left", "choose_previous", "Previous option", show=False),
    ]

    def action_choose_next(self) -> None:
        self._choose_relative(1)

    def action_choose_previous(self) -> None:
        self._choose_relative(-1)

    def _choose_relative(self, direction: int) -> None:
        buttons = list(self.query(RadioButton))
        current_index = self.pressed_index
        if direction > 0:
            target_range = range(current_index + direction, len(buttons), direction)
        else:
            target_range = range(current_index + direction, -1, direction)

        for target_index in target_range:
            button = buttons[target_index]
            if not button.disabled:
                self._selected = target_index
                button.toggle()
                return


class TradeDialog(ModalScreen):
    """Modal dialog for choosing trade mode and amount."""

    CSS = """
    TradeDialog {
        align: center middle;
    }
    #trade-dialog {
        width: 70;
        height: 24;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #trade-dialog Label {
        margin-bottom: 1;
    }
    #trade-dialog #amount-input {
        height: 3;
        margin-bottom: 1;
    }
    #trade-dialog #mode-radio {
        height: 5;
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

    def compose(self) -> ComposeResult:
        mode_index = {"open_close": 0, "open": 1, "close": 2}.get(self.default_mode, 0)
        with Vertical(id="trade-dialog"):
            yield Label(f"Test trade: [bold]{self.pair_symbol}[/bold]")
            yield Label("Trade mode:")
            with TradeModeRadioSet(id="mode-radio"):
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
                yield Button("OK", variant="primary", id="execute-btn")

    def on_mount(self) -> None:
        self.call_after_refresh(self._focus_mode_radio)

    def _focus_mode_radio(self) -> None:
        radio_set = self.query_one("#mode-radio", RadioSet)
        radio_set.focus()
        radio_set._selected = radio_set.pressed_index

    @on(RadioSet.Changed)
    def on_mode_changed(self, event: RadioSet.Changed) -> None:
        amount_input = self.query_one("#amount-input", Input)
        # Close mode always sells the full position — amount is ignored,
        # so disable the input to avoid misleading the operator.
        amount_input.disabled = (event.index == 2)

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
        # Filter out CCTP bridge pairs — they are handled automatically
        pairs = [p for p in pairs if not p.is_cctp_bridge()]
        self.pairs = pairs
        self.strategy_universe = strategy_universe
        self.state = state
        self.tvl_now = tvl_now
        self.reserve_balance = reserve_balance
        self.reserve_symbol = reserve_symbol
        self.gas_balance = gas_balance
        self.is_hyperliquid = is_hyperliquid
        self.pricing_model = pricing_model

        # Detect multichain universe for chain column display
        self.is_multichain = len(strategy_universe.data_universe.chains) > 1

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
        self.deposit_statuses = {}
        self.redemption_statuses = {}
        price_ts = native_datetime_utc_now()
        for pair in self.sorted_pairs:
            if pair.is_vault():
                self.prices[id(pair)] = _get_price(pricing_model, pair, ts=price_ts)
                self.deposit_statuses[id(pair)] = _get_deposit_status(pricing_model, pair, ts=price_ts)
                self.redemption_statuses[id(pair)] = _get_redemption_status(pricing_model, pair, ts=price_ts)
            else:
                self.prices[id(pair)] = None
                self.deposit_statuses[id(pair)] = None
                self.redemption_statuses[id(pair)] = None

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
        if self.is_multichain:
            table.add_column("Chain", key="chain", width=12)
        table.add_column("Symbol", key="symbol", width=40)
        table.add_column("Exchange", key="exchange", width=20)
        table.add_column("Price", key="price", width=14)
        table.add_column("TVL", key="tvl", width=14)
        table.add_column("Deposits", key="deposits", width=14)
        table.add_column("Redemptions", key="redemptions", width=14)
        table.add_column("Position", key="position", width=20)
        table.add_column("Lockup", key="lockup", width=14)

        for idx, pair in enumerate(self.sorted_pairs, 1):
            symbol = _get_pair_symbol(pair)
            exchange = pair.exchange_name or ""
            price = _format_price(self.prices.get(id(pair)))
            tvl = _format_tvl(self.tvl_values.get(id(pair)))
            deposits = _format_deposits_open(pair, self.deposit_statuses.get(id(pair)))
            redemptions = _format_redemptions_open(pair, self.redemption_statuses.get(id(pair)))
            position = _get_position_info(self.state, pair)
            lockup = _format_lockup(self.state, pair)

            row_values = [str(idx)]
            if self.is_multichain:
                chain_name = ChainId(pair.base.chain_id).get_name()
                row_values.append(chain_name)
            row_values.extend([symbol, exchange, price, tvl, deposits, redemptions, position, lockup])

            table.add_row(*row_values, key=str(idx))

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
