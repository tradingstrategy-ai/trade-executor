"""Test Hypercore live tradeability checks."""

import datetime
from decimal import Decimal
from unittest.mock import MagicMock

from eth_defi.compat import native_datetime_utc_now
from eth_defi.hyperliquid.api import UserVaultEquity
from eth_defi.hyperliquid.vault import HyperliquidVault, VaultInfo
import pytest
from requests.exceptions import Timeout

from tradeexecutor.ethereum.vault.hypercore_valuation import (
    HypercoreVaultPricing,
    get_hypercore_deposit_closed_reason,
)
from tradeexecutor.ethereum.vault.hypercore_routing import HypercoreVaultRouting
from tradeexecutor.ethereum.vault.hypercore_vault import (
    HLP_VAULT_ADDRESS,
    create_hypercore_vault_pair,
)
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.strategy.redemption import (
    DepositBlockReason,
    DepositCheckStage,
    RedemptionBlockReason,
    RedemptionCheckResult,
    RedemptionCheckStage,
)


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
    is_closed: bool | None = False,
    allow_deposits: bool | None = True,
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
    monkeypatch.setattr(pricing, "_get_vault_info", lambda pair, user=None: _make_info(is_closed=True))

    assert pricing.get_max_deposit(None, pair) == 0
    assert pricing.can_deposit(None, pair) is False
    check = pricing.check_deposit(None, pair, stage=DepositCheckStage.buy_rebalance)
    assert check.reason_code == DepositBlockReason.vault_deposits_closed
    assert check.message == "Vault is permanently closed"


def test_hypercore_deposit_closed_when_leader_disables_deposits(
    pricing: HypercoreVaultPricing,
    monkeypatch: pytest.MonkeyPatch,
):
    pair = _make_pair()
    monkeypatch.setattr(pricing, "_get_vault_info", lambda pair, user=None: _make_info(allow_deposits=False))

    assert pricing.get_max_deposit(None, pair) == 0
    assert pricing.can_deposit(None, pair) is False


def test_hypercore_low_share_is_capacity_policy_not_closure(
    pricing: HypercoreVaultPricing,
    monkeypatch: pytest.MonkeyPatch,
):
    """A low leader share blocks buys without claiming the vault was closed.

    1. Supply an explicitly open vault with a low observed leader share.
    2. Check the zero policy cap, reason code and source snapshot.
    """
    # 1. Fix the response so changing live vault flags cannot affect the test.
    pair = _make_pair()
    monkeypatch.setattr(pricing, "_get_vault_info", lambda pair, user=None: _make_info(leader_fraction=0.04))

    # 2. Only our temporary capacity policy blocks a new deposit.
    assert pricing.get_max_deposit(None, pair) == 0
    assert pricing.can_deposit(None, pair) is False
    check = pricing.check_deposit(None, pair, stage=DepositCheckStage.buy_rebalance)
    assert check.reason_code == DepositBlockReason.vault_max_deposit_zero
    assert check.message.startswith("Leader share")
    assert get_hypercore_deposit_closed_reason(_make_info(leader_fraction=0.04)) is None
    assert check.used_vault_info.leader_fraction == pytest.approx(0.04)
    restored = type(check).from_json(check.to_json())
    assert restored.reason_code == DepositBlockReason.vault_max_deposit_zero
    assert restored.used_vault_info.leader_fraction == pytest.approx(0.04)


def test_hypercore_missing_deposit_flags_are_unknown(
    pricing: HypercoreVaultPricing,
    monkeypatch: pytest.MonkeyPatch,
):
    """An incomplete API response must not be reported as an open or closed vault.

    1. Supply a vault-details result with absent deposit flags.
    2. Verify live pricing blocks new capital with an unknown reason.
    """
    # 1. The parser now preserves missing source flags as None.
    pair = _make_pair()
    monkeypatch.setattr(pricing, "_get_vault_info", lambda pair, user=None: _make_info(is_closed=None, allow_deposits=None))

    # 2. Unknown is neither a confirmed closure nor executable permission.
    check = pricing.check_deposit(None, pair, stage=DepositCheckStage.buy_rebalance)
    assert check.can_deposit is False
    assert check.reason_code == DepositBlockReason.unknown
    assert check.max_deposit is None


def test_hypercore_execution_preflight_rechecks_live_status(monkeypatch: pytest.MonkeyPatch):
    """Block a changed vault before activation or USDC transfer starts.

    1. Construct a live routing model and planned buy without broadcasting.
    2. Check explicit open, low-share policy, actual closure and API failure.
    """
    # 1. Preflight only reads metadata. Supply each API outcome locally so the
    # test covers changes and timeouts without a network call or transaction.
    routing = object.__new__(HypercoreVaultRouting)
    routing.simulate = False
    routing._session = MagicMock()
    trade = MagicMock()
    trade.is_buy.return_value = True
    trade.pair.pool_address = HLP_VAULT_ADDRESS["mainnet"]

    # 2. The low-share block is distinct from source closure and API failure.
    monkeypatch.setattr(HyperliquidVault, "fetch_metadata", lambda vault: _make_info())
    assert routing.check_trade_before_execution(trade) is None
    monkeypatch.setattr(HyperliquidVault, "fetch_metadata", lambda vault: _make_info(leader_fraction=0.05))
    assert routing.check_trade_before_execution(trade).startswith("Leader share")
    monkeypatch.setattr(HyperliquidVault, "fetch_metadata", lambda vault: _make_info(allow_deposits=False))
    assert routing.check_trade_before_execution(trade) == "Vault deposits disabled by leader"

    def timeout(_vault):
        raise Timeout("vaultDetails unavailable")

    monkeypatch.setattr(HyperliquidVault, "fetch_metadata", timeout)
    assert routing.check_trade_before_execution(trade) == "Hyperliquid vaultDetails unavailable before deposit: Timeout"


def test_hypercore_parent_vault_ignores_allow_deposits_flag(
    pricing: HypercoreVaultPricing,
    monkeypatch: pytest.MonkeyPatch,
):
    pair = _make_pair()
    monkeypatch.setattr(
        pricing,
        "_get_vault_info",
        lambda pair, user=None: _make_info(allow_deposits=False, relationship_type="parent"),
    )

    assert pricing.get_max_deposit(None, pair) is None
    assert pricing.can_deposit(None, pair) is True


def test_hypercore_redemption_closed_without_position(
    pricing: HypercoreVaultPricing,
    monkeypatch: pytest.MonkeyPatch,
):
    pair = _make_pair()
    monkeypatch.setattr(pricing, "_get_vault_info", lambda pair, user=None: _make_info())
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
    """Check Hypercore lockup diagnostics report a blocked user lockup.

    1. Mock a vault with withdrawable liquidity but a user lockup still in force.
    2. Run both the legacy and structured redemption checks.
    3. Verify the structured result records the user-lockup reason and stage.
    """
    pair = _make_pair()
    monkeypatch.setattr(pricing, "_get_vault_info", lambda pair, user=None: _make_info())
    monkeypatch.setattr(
        "tradeexecutor.ethereum.vault.hypercore_valuation.fetch_user_vault_equity",
        lambda session, user, vault_address, **kwargs: UserVaultEquity(
            vault_address=vault_address,
            equity=Decimal("50"),
            locked_until=native_datetime_utc_now() + datetime.timedelta(hours=1),
        ),
    )

    # 1. Mock a vault with withdrawable liquidity but a user lockup still in force.
    result = pricing.check_redemption(None, pair, stage=RedemptionCheckStage.carry_forward)

    # 2. Run both the legacy and structured redemption checks.
    assert pricing.get_max_redemption(None, pair) == 0
    assert pricing.can_redeem(None, pair) is False
    assert result.can_redeem is False

    # 3. Verify the structured result records the user-lockup reason and stage.
    assert result.reason_code == RedemptionBlockReason.user_lockup_not_expired
    assert result.stage == RedemptionCheckStage.carry_forward
    assert result.user_lockup_expires_at is not None
    assert result.max_redemption == pytest.approx(0.0)


def test_hypercore_redemption_closed_when_vault_has_no_withdrawable_liquidity(
    pricing: HypercoreVaultPricing,
    monkeypatch: pytest.MonkeyPatch,
):
    """Check Hypercore diagnostics report a zero-withdrawable vault block.

    1. Mock a vault whose user lockup is expired but whose max withdrawable value is zero.
    2. Run both the legacy and structured redemption checks.
    3. Verify the structured result records the vault-liquidity block reason.
    """
    pair = _make_pair()
    monkeypatch.setattr(pricing, "_get_vault_info", lambda pair, user=None: _make_info(max_withdrawable=Decimal("0")))
    monkeypatch.setattr(
        "tradeexecutor.ethereum.vault.hypercore_valuation.fetch_user_vault_equity",
        lambda session, user, vault_address, **kwargs: UserVaultEquity(
            vault_address=vault_address,
            equity=Decimal("50"),
            locked_until=native_datetime_utc_now() - datetime.timedelta(hours=1),
        ),
    )

    # 1. Mock a vault whose user lockup is expired but whose max withdrawable value is zero.
    result = pricing.check_redemption(None, pair, stage=RedemptionCheckStage.sell_rebalance)

    # 2. Run both the legacy and structured redemption checks.
    assert pricing.get_max_redemption(None, pair) == 0
    assert pricing.can_redeem(None, pair) is False
    assert result.can_redeem is False

    # 3. Verify the structured result records the vault-liquidity block reason.
    assert result.reason_code == RedemptionBlockReason.vault_max_withdrawable_zero
    assert result.stage == RedemptionCheckStage.sell_rebalance
    assert result.max_withdrawable == pytest.approx(0.0)
    assert result.max_redemption == pytest.approx(0.0)


def test_hypercore_redemption_allowed_when_lockup_expired(
    pricing: HypercoreVaultPricing,
    monkeypatch: pytest.MonkeyPatch,
):
    """Check Hypercore diagnostics report an allowed redemption and round-trip cleanly.

    1. Mock a vault with an expired user lockup and positive withdrawable liquidity.
    2. Run both the legacy and structured redemption checks.
    3. Verify the structured result round-trips through dataclasses-json intact.
    """
    pair = _make_pair()
    monkeypatch.setattr(pricing, "_get_vault_info", lambda pair, user=None: _make_info(max_withdrawable=Decimal("40")))
    monkeypatch.setattr(
        "tradeexecutor.ethereum.vault.hypercore_valuation.fetch_user_vault_equity",
        lambda session, user, vault_address, **kwargs: UserVaultEquity(
            vault_address=vault_address,
            equity=Decimal("50"),
            locked_until=native_datetime_utc_now() - datetime.timedelta(hours=1),
        ),
    )

    # 1. Mock a vault with an expired user lockup and positive withdrawable liquidity.
    result = pricing.check_redemption(None, pair, stage=RedemptionCheckStage.sell_rebalance)

    # 2. Run both the legacy and structured redemption checks.
    assert pricing.get_max_redemption(None, pair) == Decimal("40")
    assert pricing.can_redeem(None, pair) is True
    assert pricing.is_tradeable(None, pair) is True
    assert result.can_redeem is True
    assert result.reason_code is None
    assert result.max_redemption == pytest.approx(40.0)
    assert result.raw_api_data is not None
    assert result.raw_api_data.vault_info is not None
    assert result.raw_api_data.user_equity is not None

    # 3. Verify the structured result round-trips through dataclasses-json intact.
    restored = RedemptionCheckResult.from_json(result.to_json())
    assert restored.can_redeem is True
    assert restored.stage == RedemptionCheckStage.sell_rebalance
    assert restored.max_redemption == pytest.approx(40.0)
    assert restored.raw_api_data is not None
    assert restored.raw_api_data.user_equity is not None

def test_hypercore_redemption_defaults_to_open_when_safe_unknown(
    monkeypatch: pytest.MonkeyPatch,
):
    pair = _make_pair()
    pricing = HypercoreVaultPricing(
        value_func=lambda pair: Decimal("100"),
        safe_address_resolver=None,
        session_factory=lambda pair: MagicMock(),
    )

    monkeypatch.setattr(pricing, "_get_vault_info", lambda pair, user=None: _make_info())

    assert pricing.get_max_redemption(None, pair) is None
    assert pricing.can_redeem(None, pair) is True


def test_hypercore_deposit_closed_reason_matches_parent_special_case():
    parent_info = _make_info(allow_deposits=False, relationship_type="parent")
    normal_info = _make_info(allow_deposits=False, relationship_type="normal")

    assert get_hypercore_deposit_closed_reason(parent_info) is None
    assert get_hypercore_deposit_closed_reason(normal_info) == "Vault deposits disabled by leader"
