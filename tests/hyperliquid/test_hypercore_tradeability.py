"""Test Hypercore live tradeability checks."""

import datetime
from decimal import Decimal
from unittest.mock import MagicMock

from eth_defi.compat import native_datetime_utc_now
from eth_defi.hyperliquid.api import UserVaultEquity
from eth_defi.hyperliquid.vault import VaultInfo
import pytest

from tradeexecutor.ethereum.vault.hypercore_valuation import (
    HypercoreVaultPricing,
    get_hypercore_deposit_closed_reason,
)
from tradeexecutor.ethereum.vault.hypercore_vault import (
    HLP_VAULT_ADDRESS,
    create_hypercore_vault_pair,
)
from tradeexecutor.state.identifier import AssetIdentifier


def _make_pair() -> object:
    quote = AssetIdentifier(
        chain_id=999,
        address="0x0000000000000000000000000000000000000002",
        token_symbol="USDC",
        decimals=6,
    )
    return create_hypercore_vault_pair(
        quote=quote,
        vault_address=HLP_VAULT_ADDRESS["mainnet"],
    )


def _make_info(
    *,
    max_withdrawable: Decimal = Decimal("100"),
    is_closed: bool = False,
    allow_deposits: bool = True,
    relationship_type: str = "normal",
    leader_fraction: float | None = 0.10,
) -> VaultInfo:
    return VaultInfo(
        name="Test vault",
        vault_address=HLP_VAULT_ADDRESS["mainnet"],
        leader="0x0000000000000000000000000000000000000003",
        description="",
        followers=[],
        portfolio={},
        max_distributable=Decimal("0"),
        max_withdrawable=max_withdrawable,
        is_closed=is_closed,
        allow_deposits=allow_deposits,
        relationship_type=relationship_type,
        commission_rate=None,
        leader_fraction=leader_fraction,
        leader_commission=None,
        parent=None,
    )


@pytest.fixture()
def pricing() -> HypercoreVaultPricing:
    return HypercoreVaultPricing(
        value_func=lambda pair: Decimal("100"),
        safe_address_resolver=lambda pair: "0x0000000000000000000000000000000000000004",
        session_factory=lambda pair: MagicMock(),
    )


def test_hypercore_deposit_closed_when_vault_closed(
    pricing: HypercoreVaultPricing,
    monkeypatch: pytest.MonkeyPatch,
):
    pair = _make_pair()
    monkeypatch.setattr(pricing, "_get_vault_info", lambda pair: _make_info(is_closed=True))

    assert pricing.get_max_deposit(None, pair) == 0
    assert pricing.can_deposit(None, pair) is False


def test_hypercore_deposit_closed_when_leader_disables_deposits(
    pricing: HypercoreVaultPricing,
    monkeypatch: pytest.MonkeyPatch,
):
    pair = _make_pair()
    monkeypatch.setattr(pricing, "_get_vault_info", lambda pair: _make_info(allow_deposits=False))

    assert pricing.get_max_deposit(None, pair) == 0
    assert pricing.can_deposit(None, pair) is False


def test_hypercore_deposit_closed_when_leader_fraction_too_low(
    pricing: HypercoreVaultPricing,
    monkeypatch: pytest.MonkeyPatch,
):
    pair = _make_pair()
    monkeypatch.setattr(pricing, "_get_vault_info", lambda pair: _make_info(leader_fraction=0.04))

    assert pricing.get_max_deposit(None, pair) == 0
    assert pricing.can_deposit(None, pair) is False


def test_hypercore_parent_vault_ignores_allow_deposits_flag(
    pricing: HypercoreVaultPricing,
    monkeypatch: pytest.MonkeyPatch,
):
    pair = _make_pair()
    monkeypatch.setattr(
        pricing,
        "_get_vault_info",
        lambda pair: _make_info(allow_deposits=False, relationship_type="parent"),
    )

    assert pricing.get_max_deposit(None, pair) is None
    assert pricing.can_deposit(None, pair) is True


def test_hypercore_redemption_closed_without_position(
    pricing: HypercoreVaultPricing,
    monkeypatch: pytest.MonkeyPatch,
):
    pair = _make_pair()
    monkeypatch.setattr(pricing, "_get_vault_info", lambda pair: _make_info())
    monkeypatch.setattr(
        "tradeexecutor.ethereum.vault.hypercore_valuation.fetch_user_vault_equity",
        lambda session, user, vault_address, **kwargs: None,
    )

    assert pricing.get_max_redemption(None, pair) == 0
    assert pricing.can_redeem(None, pair) is False


def test_hypercore_redemption_closed_during_lockup(
    pricing: HypercoreVaultPricing,
    monkeypatch: pytest.MonkeyPatch,
):
    pair = _make_pair()
    monkeypatch.setattr(pricing, "_get_vault_info", lambda pair: _make_info())
    monkeypatch.setattr(
        "tradeexecutor.ethereum.vault.hypercore_valuation.fetch_user_vault_equity",
        lambda session, user, vault_address, **kwargs: UserVaultEquity(
            vault_address=vault_address,
            equity=Decimal("50"),
            locked_until=native_datetime_utc_now() + datetime.timedelta(hours=1),
        ),
    )

    assert pricing.get_max_redemption(None, pair) == 0
    assert pricing.can_redeem(None, pair) is False


def test_hypercore_redemption_closed_when_vault_has_no_withdrawable_liquidity(
    pricing: HypercoreVaultPricing,
    monkeypatch: pytest.MonkeyPatch,
):
    pair = _make_pair()
    monkeypatch.setattr(pricing, "_get_vault_info", lambda pair: _make_info(max_withdrawable=Decimal("0")))
    monkeypatch.setattr(
        "tradeexecutor.ethereum.vault.hypercore_valuation.fetch_user_vault_equity",
        lambda session, user, vault_address, **kwargs: UserVaultEquity(
            vault_address=vault_address,
            equity=Decimal("50"),
            locked_until=native_datetime_utc_now() - datetime.timedelta(hours=1),
        ),
    )

    assert pricing.get_max_redemption(None, pair) == 0
    assert pricing.can_redeem(None, pair) is False


def test_hypercore_redemption_allowed_when_lockup_expired(
    pricing: HypercoreVaultPricing,
    monkeypatch: pytest.MonkeyPatch,
):
    pair = _make_pair()
    monkeypatch.setattr(pricing, "_get_vault_info", lambda pair: _make_info(max_withdrawable=Decimal("40")))
    monkeypatch.setattr(
        "tradeexecutor.ethereum.vault.hypercore_valuation.fetch_user_vault_equity",
        lambda session, user, vault_address, **kwargs: UserVaultEquity(
            vault_address=vault_address,
            equity=Decimal("50"),
            locked_until=native_datetime_utc_now() - datetime.timedelta(hours=1),
        ),
    )

    assert pricing.get_max_redemption(None, pair) == Decimal("40")
    assert pricing.can_redeem(None, pair) is True
    assert pricing.is_tradeable(None, pair) is True


def test_hypercore_deposit_closed_reason_matches_parent_special_case():
    parent_info = _make_info(allow_deposits=False, relationship_type="parent")
    normal_info = _make_info(allow_deposits=False, relationship_type="normal")

    assert get_hypercore_deposit_closed_reason(parent_info) is None
    assert get_hypercore_deposit_closed_reason(normal_info) == "Vault deposits disabled by leader"
