"""Tests for Hyperliquid redemption-delay helpers."""

import datetime
from dataclasses import dataclass

import pytest

from tradeexecutor.ethereum.vault.hypercore_vault import HLP_VAULT_ADDRESS
from tradeexecutor.exchange_account.allocation import (
    HYPERLIQUID_DEFAULT_REDEEM_DELAY,
    HYPERLIQUID_HLP_REDEEM_DELAY,
    get_redeemable_capital,
)


@dataclass
class DummyPair:
    pool_address: str | None
    other_data: dict[str, str]
    chain_id: int = 9999

    def is_hyperliquid_vault(self) -> bool:
        return self.chain_id == 9999 and self.other_data.get("vault_protocol") == "hypercore"

    def is_vault(self) -> bool:
        return self.other_data.get("vault_protocol") == "hypercore"


@dataclass
class DummyTrade:
    executed_at: datetime.datetime | None

    def is_buy(self) -> bool:
        return True


@dataclass
class DummyPosition:
    pair: DummyPair
    opened_at: datetime.datetime
    trades: dict[int, DummyTrade]
    value: float

    def get_value(self) -> float:
        return self.value


@dataclass
class DummyVaultPosition:
    """Dummy that mirrors the real ``TradingPosition`` redeemable-value path.

    Unlike :class:`DummyPosition`, this exposes ``get_available_trading_quantity()``
    (a share quantity) and ``get_current_price()`` (USD per share), so it
    exercises the branch that must convert quantity into USD.
    """

    pair: DummyPair
    opened_at: datetime.datetime
    trades: dict[int, DummyTrade]
    quantity: float
    price: float

    def get_value(self) -> float:
        return self.quantity * self.price

    def get_available_trading_quantity(self) -> float:
        return self.quantity

    def get_current_price(self) -> float:
        return self.price


@pytest.mark.timeout(300)
def test_get_redeemable_capital_uses_hyperliquid_lockup_windows() -> None:
    """Check Hyperliquid redeemable capital against the known lock-up windows.

    1. Build one HLP position and one generic Hypercore vault position with the same buy timestamp.
    2. Verify the generic position unlocks after the default one-day delay while HLP stays locked.
    3. Verify HLP only becomes redeemable after the longer four-day delay.
    """
    # 1. Build one HLP position and one generic Hypercore vault position with the same buy timestamp.
    buy_at = datetime.datetime(2026, 1, 1, 12, 0, 0)

    hlp_position = DummyPosition(
        pair=DummyPair(
            pool_address=HLP_VAULT_ADDRESS["mainnet"],
            other_data={"vault_protocol": "hypercore"},
        ),
        opened_at=buy_at,
        trades={1: DummyTrade(executed_at=buy_at)},
        value=100.0,
    )
    generic_position = DummyPosition(
        pair=DummyPair(
            pool_address="0x0000000000000000000000000000000000001234",
            other_data={"vault_protocol": "hypercore"},
        ),
        opened_at=buy_at,
        trades={1: DummyTrade(executed_at=buy_at)},
        value=100.0,
    )

    # 2. Verify the generic position unlocks after the default one-day delay while HLP stays locked.
    after_default_delay = buy_at + HYPERLIQUID_DEFAULT_REDEEM_DELAY + datetime.timedelta(minutes=1)
    assert get_redeemable_capital(generic_position, timestamp=after_default_delay) == pytest.approx(100.0)
    assert get_redeemable_capital(hlp_position, timestamp=after_default_delay) == pytest.approx(0.0)

    # 3. Verify HLP only becomes redeemable after the longer four-day delay.
    after_hlp_delay = buy_at + HYPERLIQUID_HLP_REDEEM_DELAY + datetime.timedelta(minutes=1)
    assert get_redeemable_capital(hlp_position, timestamp=after_hlp_delay) == pytest.approx(100.0)


@pytest.mark.timeout(300)
def test_get_redeemable_capital_returns_usd_not_share_quantity() -> None:
    """Redeemable capital must be USD, not the raw vault-share quantity.

    A vault whose share price is not 1.0 previously returned the share count as
    if it were dollars, inflating redeemable capital by the share-price factor.

    1. Build a generic Hypercore vault position holding 1,000,000 shares priced at 0.05 USD each.
    2. While still locked, verify redeemable capital is zero.
    3. After the default delay, verify redeemable capital equals 50,000 USD (quantity times price), not 1,000,000.
    """
    # 1. Build a generic Hypercore vault position holding 1,000,000 shares priced at 0.05 USD each.
    buy_at = datetime.datetime(2026, 1, 1, 12, 0, 0)
    position = DummyVaultPosition(
        pair=DummyPair(
            pool_address="0x0000000000000000000000000000000000005678",
            other_data={"vault_protocol": "hypercore"},
        ),
        opened_at=buy_at,
        trades={1: DummyTrade(executed_at=buy_at)},
        quantity=1_000_000.0,
        price=0.05,
    )

    # 2. While still locked, verify redeemable capital is zero.
    still_locked = buy_at + datetime.timedelta(minutes=1)
    assert get_redeemable_capital(position, timestamp=still_locked) == pytest.approx(0.0)

    # 3. After the default delay, verify redeemable capital equals 50,000 USD (quantity times price), not 1,000,000.
    after_default_delay = buy_at + HYPERLIQUID_DEFAULT_REDEEM_DELAY + datetime.timedelta(minutes=1)
    assert get_redeemable_capital(position, timestamp=after_default_delay) == pytest.approx(50_000.0)
