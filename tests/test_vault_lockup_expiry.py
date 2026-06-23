"""Test estimated vault lockup expiry persistence."""

import datetime
from decimal import Decimal

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeType


def _make_vault_pair(lockup_days: float | None = 30.0) -> TradingPairIdentifier:
    """Create a minimal vault pair with optional lockup metadata."""
    base = AssetIdentifier(
        chain_id=42161,
        address="0x0000000000000000000000000000000000000001",
        token_symbol="vUSDC",
        decimals=18,
    )
    quote = AssetIdentifier(
        chain_id=42161,
        address="0x0000000000000000000000000000000000000002",
        token_symbol="USDC",
        decimals=6,
    )
    other_data = {"vault_protocol": "erc4626"}
    if lockup_days is not None:
        other_data["vault_lockup_days"] = lockup_days
    return TradingPairIdentifier(
        base=base,
        quote=quote,
        pool_address="0x0000000000000000000000000000000000000001",
        exchange_address="0x0000000000000000000000000000000000000000",
        internal_id=1,
        internal_exchange_id=1,
        fee=0.0,
        kind=TradingPairKind.vault,
        exchange_name="vault",
        other_data=other_data,
    )


def _make_position(pair: TradingPairIdentifier) -> TradingPosition:
    """Create a minimal vault position."""
    return TradingPosition(
        position_id=1,
        pair=pair,
        opened_at=datetime.datetime(2026, 1, 1),
        last_pricing_at=datetime.datetime(2026, 1, 1),
        last_token_price=Decimal("1"),
        last_reserve_price=Decimal("1"),
        reserve_currency=pair.quote,
    )


def _make_trade(
    pair: TradingPairIdentifier,
    executed_at: datetime.datetime,
    *,
    planned_quantity: Decimal = Decimal("10"),
) -> TradeExecution:
    """Create a minimal vault trade."""
    trade = TradeExecution(
        trade_id=1,
        position_id=1,
        trade_type=TradeType.rebalance,
        pair=pair,
        opened_at=executed_at - datetime.timedelta(hours=1),
        planned_quantity=planned_quantity,
        planned_price=1.0,
        planned_reserve=Decimal("10"),
        reserve_currency=pair.quote,
    )
    trade.executed_at = executed_at
    return trade


def test_estimated_vault_lockup_expiry_is_stored_after_buy() -> None:
    """A successful vault buy stores an estimated lockup expiry.

    Steps:
    1. Build a vault position and a successful buy with lockup metadata.
    2. Apply the lockup expiry helper.
    3. Assert the position stores the estimated expiry and marker.
    """

    # 1. Build a vault position and a successful buy with lockup metadata.
    pair = _make_vault_pair(lockup_days=30.0)
    position = _make_position(pair)
    executed_at = datetime.datetime(2026, 6, 1, 12, 0, 0)
    trade = _make_trade(pair, executed_at)

    # 2. Apply the lockup expiry helper.
    State().maybe_set_vault_lockup_expiry(position, trade)

    # 3. The position stores the estimated expiry and marker.
    assert position.other_data["vault_lockup_expires_at"] == "2026-07-01T12:00:00"
    assert position.other_data["vault_lockup_estimated"] is True


def test_estimated_vault_lockup_expiry_extends_existing_estimate() -> None:
    """A top-up buy extends an existing estimated lockup expiry.

    Steps:
    1. Build a vault position with an existing estimated expiry.
    2. Apply the helper with a later successful buy.
    3. Assert the later estimated expiry is stored.
    """

    # 1. Build a vault position with an existing estimated expiry.
    pair = _make_vault_pair(lockup_days=30.0)
    position = _make_position(pair)
    position.other_data["vault_lockup_expires_at"] = "2026-06-15T12:00:00"
    position.other_data["vault_lockup_estimated"] = True
    trade = _make_trade(pair, datetime.datetime(2026, 6, 10, 12, 0, 0))

    # 2. Apply the helper with a later successful buy.
    State().maybe_set_vault_lockup_expiry(position, trade)

    # 3. The later estimated expiry is stored.
    assert position.other_data["vault_lockup_expires_at"] == "2026-07-10T12:00:00"
    assert position.other_data["vault_lockup_estimated"] is True


def test_estimated_vault_lockup_expiry_does_not_overwrite_concrete_expiry() -> None:
    """A metadata estimate does not overwrite an existing concrete expiry.

    Steps:
    1. Build a vault position with an existing concrete expiry.
    2. Apply the helper with a later successful buy.
    3. Assert the concrete expiry is unchanged.
    """

    # 1. Build a vault position with an existing concrete expiry.
    pair = _make_vault_pair(lockup_days=30.0)
    position = _make_position(pair)
    position.other_data["vault_lockup_expires_at"] = "2026-06-15T12:00:00"
    position.other_data["vault_lockup_estimated"] = False
    trade = _make_trade(pair, datetime.datetime(2026, 6, 10, 12, 0, 0))

    # 2. Apply the helper with a later successful buy.
    State().maybe_set_vault_lockup_expiry(position, trade)

    # 3. The concrete expiry is unchanged.
    assert position.other_data["vault_lockup_expires_at"] == "2026-06-15T12:00:00"
    assert position.other_data["vault_lockup_estimated"] is False


def test_estimated_vault_lockup_expiry_does_not_overwrite_concrete_no_lockup() -> None:
    """A metadata estimate does not overwrite an existing concrete no-lockup marker.

    Steps:
    1. Build a vault position with an existing concrete no-lockup marker.
    2. Apply the helper with a successful buy carrying lockup metadata.
    3. Assert the concrete no-lockup marker is unchanged.
    """

    # 1. Build a vault position with an existing concrete no-lockup marker.
    pair = _make_vault_pair(lockup_days=30.0)
    position = _make_position(pair)
    position.other_data["vault_lockup_expires_at"] = None
    position.other_data["vault_lockup_estimated"] = False
    trade = _make_trade(pair, datetime.datetime(2026, 6, 10, 12, 0, 0))

    # 2. Apply the helper with a successful buy carrying lockup metadata.
    State().maybe_set_vault_lockup_expiry(position, trade)

    # 3. The concrete no-lockup marker is unchanged.
    assert position.other_data["vault_lockup_expires_at"] is None
    assert position.other_data["vault_lockup_estimated"] is False


def test_estimated_vault_lockup_expiry_ignores_sells() -> None:
    """A vault sell does not update lockup metadata.

    Steps:
    1. Build a vault position and a sell trade with lockup metadata.
    2. Apply the lockup expiry helper.
    3. Assert the position lockup metadata is untouched.
    """

    # 1. Build a vault position and a sell trade with lockup metadata.
    pair = _make_vault_pair(lockup_days=30.0)
    position = _make_position(pair)
    trade = _make_trade(
        pair,
        datetime.datetime(2026, 6, 1, 12, 0, 0),
        planned_quantity=Decimal("-10"),
    )

    # 2. Apply the lockup expiry helper.
    State().maybe_set_vault_lockup_expiry(position, trade)

    # 3. The position lockup metadata is untouched.
    assert "vault_lockup_expires_at" not in position.other_data
    assert "vault_lockup_estimated" not in position.other_data
