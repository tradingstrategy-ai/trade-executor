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

from tradeexecutor.cli.trade_ui_tui import _format_lockup
from tradeexecutor.state.identifier import (
    AssetIdentifier,
    TradingPairIdentifier,
    TradingPairKind,
)
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeType


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


def test_tui_lockup_shows_ostium_settlement_eta(monkeypatch) -> None:
    """A pending Ostium deposit with a future estimate shows an eligibility countdown.

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

    # 4. A past estimate (settlement slipped) renders as "settling".
    past = (now - datetime.timedelta(hours=1)).isoformat()
    state_past = _make_state_with_pending_deposit(pair, past)
    assert _format_lockup(state_past, pair).plain == "settling"


def test_tui_lockup_shows_pending_when_no_eta(monkeypatch) -> None:
    """An operator-driven vault (no on-chain ETA) shows ``pending``.

    Steps:
    1. Freeze the clock.
    2. Build a state with a settlement-pending deposit but no stored estimate.
    3. Assert the Lockup cell shows ``pending``.
    """

    # 1. Freeze "now".
    now = datetime.datetime(2026, 6, 9, 12, 0, 0)
    monkeypatch.setattr("tradeexecutor.cli.trade_ui_tui.native_datetime_utc_now", lambda: now)

    pair = _make_vault_pair()

    # 2. No estimate stored (Lagoon / ERC-7540 operator settlement).
    state = _make_state_with_pending_deposit(pair, None)

    # 3. Renders as "pending".
    assert _format_lockup(state, pair).plain == "pending"
