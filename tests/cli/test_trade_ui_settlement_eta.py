"""Test trade-ui Lockup column showing async vault deposit settlement ETA.

When an async vault deposit (Ostium V1.5, ERC-7540 Lagoon) has been requested
on-chain but not yet settled, the trade sits in
``TradeStatus.vault_settlement_pending`` and the position holds 0 shares.
The trade-ui reuses the Lockup column to surface the estimated settlement time
so the operator knows when the shares will appear, instead of just seeing
``0 oLP`` with no hint.
"""

import datetime
from decimal import Decimal

from tradeexecutor.cli.trade_ui_tui import _format_lockup, _get_position_info
from tradeexecutor.state.identifier import (
    AssetIdentifier,
    TradingPairIdentifier,
    TradingPairKind,
)
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeType


VAULT_DISPLAY_FLAGS = [
    {"severity": "red", "type": "bad_debt_unrealized", "source": "morpho"},
    {"severity": "yellow", "type": "not_whitelisted", "source": "morpho"},
]


def _make_vault_pair() -> TradingPairIdentifier:
    """Create a minimal ERC-4626 vault trading pair."""
    base = AssetIdentifier(
        chain_id=42161,
        address="0x0000000000000000000000000000000000000001",
        token_symbol="oLP",
        decimals=18,
    )
    quote = AssetIdentifier(
        chain_id=42161,
        address="0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
        token_symbol="USDC",
        decimals=6,
    )
    return TradingPairIdentifier(
        base=base,
        quote=quote,
        pool_address="0x0000000000000000000000000000000000000001",
        exchange_address="0x0000000000000000000000000000000000000000",
        internal_id=1,
        internal_exchange_id=1,
        fee=0.0,
        kind=TradingPairKind.vault,
        exchange_name="ostium",
        other_data={"vault_protocol": "erc4626", "vault_name": "Ostium Liquidity Pool Vault"},
    )


def _make_state_with_pending_deposit(
    pair: TradingPairIdentifier,
    estimated_at: str | None,
) -> State:
    """Build a State with one open vault position whose buy trade is settlement-pending.

    ``estimated_at`` is the stored ISO settlement estimate (or ``None`` for
    operator-driven vaults that have no on-chain ETA).
    """
    position = TradingPosition(
        position_id=1,
        pair=pair,
        opened_at=datetime.datetime(2023, 1, 1),
        last_pricing_at=datetime.datetime(2023, 1, 1),
        last_token_price=Decimal("1"),
        last_reserve_price=Decimal("1"),
        reserve_currency=pair.quote,
    )
    trade = TradeExecution(
        trade_id=1,
        position_id=1,
        trade_type=TradeType.rebalance,
        pair=pair,
        opened_at=datetime.datetime(2023, 1, 1),
        planned_quantity=Decimal("1.0"),  # positive -> buy / deposit
        planned_price=1.0,
        planned_reserve=Decimal("5"),
        reserve_currency=pair.quote,
    )
    # vault_settlement_pending_at set (and executed_at/failed_at unset) makes
    # get_status() return vault_settlement_pending.
    trade.vault_settlement_pending_at = datetime.datetime(2023, 1, 1)
    trade.other_data["vault_settlement_estimated_at"] = estimated_at
    position.trades[trade.trade_id] = trade

    state = State()
    state.portfolio.open_positions[position.position_id] = position
    return state


def _make_state_with_open_position(
    pair: TradingPairIdentifier,
    lockup_expires_at: str | None = None,
    lockup_estimated: bool | None = None,
) -> State:
    """Build a State with one open vault position."""
    position = TradingPosition(
        position_id=1,
        pair=pair,
        opened_at=datetime.datetime(2023, 1, 1),
        last_pricing_at=datetime.datetime(2023, 1, 1),
        last_token_price=Decimal("1"),
        last_reserve_price=Decimal("1"),
        reserve_currency=pair.quote,
    )
    if lockup_expires_at is not None:
        position.other_data["vault_lockup_expires_at"] = lockup_expires_at
    if lockup_estimated is not None:
        position.other_data["vault_lockup_estimated"] = lockup_estimated

    state = State()
    state.portfolio.open_positions[position.position_id] = position
    return state


def test_tui_lockup_shows_ostium_settlement_eta(monkeypatch) -> None:
    """A pending Ostium deposit shows its ETA and pending reserve amount.

    Steps:
    1. Freeze the clock so the remaining time is deterministic.
    2. Build a state with a settlement-pending deposit estimated ~18h ahead.
    3. Assert the Lockup cell shows an ``eligible`` countdown, not ``-``.
    4. Build a state whose estimate has already passed and assert ``settling``.
    """

    # 1. Freeze "now" so the countdown is deterministic.
    now = datetime.datetime(2026, 6, 9, 12, 0, 0)
    monkeypatch.setattr("tradeexecutor.cli.trade_ui_tui.native_datetime_utc_now", lambda: now)

    pair = _make_vault_pair()

    # 2. Estimate 18 hours in the future.
    future = (now + datetime.timedelta(hours=18)).isoformat()
    state = _make_state_with_pending_deposit(pair, future)

    # 3. Future estimate renders as an "eligible" countdown.
    cell = _format_lockup(state, pair)
    assert "eligible" in cell.plain
    assert "18.0h" in cell.plain
    assert _get_position_info(state, pair) == "5 USDC"

    # 4. A past estimate (settlement slipped) renders as "settling".
    past = (now - datetime.timedelta(hours=1)).isoformat()
    state_past = _make_state_with_pending_deposit(pair, past)
    assert _format_lockup(state_past, pair).plain == "settling"


def test_tui_lockup_shows_pending_when_no_eta(monkeypatch) -> None:
    """An operator-driven vault (no on-chain ETA) shows ``pending`` and reserve amount.

    Steps:
    1. Freeze the clock.
    2. Build a state with a settlement-pending deposit but no stored estimate.
    3. Assert the Lockup cell shows ``pending`` and the Position cell shows the reserve amount.
    """

    # 1. Freeze "now".
    now = datetime.datetime(2026, 6, 9, 12, 0, 0)
    monkeypatch.setattr("tradeexecutor.cli.trade_ui_tui.native_datetime_utc_now", lambda: now)

    pair = _make_vault_pair()

    # 2. No estimate stored (Lagoon / ERC-7540 operator settlement).
    state = _make_state_with_pending_deposit(pair, None)

    # 3. Renders as "pending".
    assert _format_lockup(state, pair).plain == "pending"
    assert _get_position_info(state, pair) == "5 USDC"


def test_tui_lockup_shows_vault_display_flags_when_empty() -> None:
    """Generic vault warning flags fill an otherwise empty Lockup cell.

    Steps:
    1. Build a vault pair with generic red and yellow display flags.
    2. Build a state without any open or pending position for the pair.
    3. Assert the Lockup cell shows compact flag counts using red priority.
    """

    # 1. Build a vault pair with generic red and yellow display flags.
    pair = _make_vault_pair()
    pair.other_data["vault_display_flags"] = VAULT_DISPLAY_FLAGS

    # 2. Build a state without any open or pending position for the pair.
    state = State()

    # 3. The empty Lockup cell falls back to generic warning counts.
    cell = _format_lockup(state, pair)
    assert cell.plain == "red 1 yellow 1"
    assert cell.style == "red"


def test_tui_lockup_prefers_expiry_over_vault_display_flags(monkeypatch) -> None:
    """A real lockup expiry takes priority over generic vault warning flags.

    Steps:
    1. Freeze the clock so the lockup countdown is deterministic.
    2. Build an open vault position with a future lockup expiry and warning flags.
    3. Assert the Lockup cell shows the countdown instead of warning counts.
    """

    # 1. Freeze "now".
    now = datetime.datetime(2026, 6, 9, 12, 0, 0)
    monkeypatch.setattr("tradeexecutor.cli.trade_ui_tui.native_datetime_utc_now", lambda: now)

    # 2. Build an open vault position with a future lockup expiry and warning flags.
    pair = _make_vault_pair()
    pair.other_data["vault_display_flags"] = VAULT_DISPLAY_FLAGS
    state = _make_state_with_open_position(pair, (now + datetime.timedelta(hours=6)).isoformat())

    # 3. The real lockup countdown takes priority over warning counts.
    cell = _format_lockup(state, pair)
    assert cell.plain == "6.0h"
    assert cell.style == "yellow"


def test_tui_lockup_marks_estimated_expiry(monkeypatch) -> None:
    """An estimated lockup expiry is shown with a ``~`` prefix.

    Steps:
    1. Freeze the clock so the countdown is deterministic.
    2. Build an open vault position with an estimated future lockup expiry.
    3. Assert the Lockup cell marks the countdown as estimated.
    """

    # 1. Freeze the clock so the countdown is deterministic.
    now = datetime.datetime(2026, 6, 9, 12, 0, 0)
    monkeypatch.setattr("tradeexecutor.cli.trade_ui_tui.native_datetime_utc_now", lambda: now)

    # 2. Build an open vault position with an estimated future lockup expiry.
    pair = _make_vault_pair()
    state = _make_state_with_open_position(
        pair,
        (now + datetime.timedelta(days=30)).isoformat(),
        lockup_estimated=True,
    )

    # 3. The Lockup cell marks the countdown as estimated.
    cell = _format_lockup(state, pair)
    assert cell.plain == "~30.0d"
    assert cell.style == "yellow"


def test_tui_lockup_shows_metadata_only_lockup_days() -> None:
    """A metadata-only vault lockup shows a static estimated duration.

    Steps:
    1. Build a vault pair with metadata lockup days.
    2. Build an open position without a stored lockup expiry.
    3. Assert the Lockup cell shows the static estimated duration.
    """

    # 1. Build a vault pair with metadata lockup days.
    pair = _make_vault_pair()
    pair.other_data["vault_lockup_days"] = 30.0

    # 2. Build an open position without a stored lockup expiry.
    state = _make_state_with_open_position(pair)

    # 3. The Lockup cell shows the static estimated duration.
    cell = _format_lockup(state, pair)
    assert cell.plain == "~30.0d"
    assert cell.style == "yellow"


def test_tui_lockup_prefers_pending_settlement_over_vault_display_flags(monkeypatch) -> None:
    """Pending vault settlement takes priority over other Lockup fallbacks.

    Steps:
    1. Freeze the clock so the settlement countdown is deterministic.
    2. Build a settlement-pending vault deposit with generic warning flags and lockup metadata.
    3. Assert the Lockup cell shows the pending settlement ETA, not fallback data.
    """

    # 1. Freeze "now".
    now = datetime.datetime(2026, 6, 9, 12, 0, 0)
    monkeypatch.setattr("tradeexecutor.cli.trade_ui_tui.native_datetime_utc_now", lambda: now)

    # 2. Build a settlement-pending vault deposit with generic warning flags and lockup metadata.
    pair = _make_vault_pair()
    pair.other_data["vault_display_flags"] = VAULT_DISPLAY_FLAGS
    pair.other_data["vault_lockup_days"] = 30.0
    state = _make_state_with_pending_deposit(pair, (now + datetime.timedelta(hours=2)).isoformat())

    # 3. The settlement ETA takes priority over fallback data.
    cell = _format_lockup(state, pair)
    assert cell.plain == "eligible 2.0h"
    assert cell.style == "yellow"
