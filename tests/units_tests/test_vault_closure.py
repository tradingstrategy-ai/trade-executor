"""Shared admission rules: protocol evidence must not be confused with policy."""

import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from eth_defi.erc_4626.core import ERC4626Feature
from eth_defi.vault.deposit_redeem import VaultFlowUnavailable

from tradeexecutor.strategy.redemption import (
    DepositBlockReason, DepositCheckResult, DepositCheckStage,
    RedemptionBlockReason, RedemptionCheckResult, VaultInfoSnapshot,
)
from tradeexecutor.strategy.vault_closure import (
    VaultClosure, VaultClosureOutcome as Outcome, can_interact_with_vault,
    check_live_deposit, get_vault_closure, is_vault_closed,
    is_backtesting_deposit_open,
)


def vault(feature: ERC4626Feature | None = None, native: bool = False) -> SimpleNamespace:
    """Supply identity metadata only; tests exercise the real admission rule."""
    return SimpleNamespace(
        get_vault_features=lambda: {feature} if feature else set(),
        get_vault_protocol=lambda: None,
        is_hyperliquid_vault=lambda: native,
        get_ticker=lambda: "TEST", pool_address="0x01",
    )


@pytest.mark.parametrize("feature", [ERC4626Feature.lagoon_like, ERC4626Feature.morpho_v2_like, ERC4626Feature.yearn_v3_like])
def test_non_admission_maxima(feature: ERC4626Feature) -> None:
    """Do not mistake claim/account-specific maxima for admission.

    1. Supply generic zero maxima for each affected protocol.
    2. Check unknown evidence separately from default interaction permission.
    3. Verify explicit evidence and unknown-policy overrides.
    """
    # 1. The identity double substitutes metadata, not protocol decisions.
    pair = vault(feature)
    state = {"deposits_open": False, "max_deposit": 0, "max_redeem": 0}
    # 2. Both operation assessments use the production rule.
    for operation in ("deposit", "redemption"):
        result = get_vault_closure(pair, historical_state=state, operation=operation)
        assert result.status is Outcome.unknown
        assert not is_vault_closed(result)
        assert can_interact_with_vault(result)
    # 3. Only explicit controls change these defaults.
    assert is_vault_closed(pair, historical_state=state, trust_protocol_state=True)
    result = get_vault_closure(pair, historical_state=state, unknown_is_closed=True)
    assert result.blocked and result.status is Outcome.unknown


@pytest.mark.parametrize("flag,cap,outcome,allowed", [
    (True, 0, Outcome.policy_blocked, False),
    (False, 0, Outcome.closed, False),
    (True, None, Outcome.open, True),
    (pd.NA, None, Outcome.unknown, False),
])
def test_native_permission_and_policy(flag, cap: float | None, outcome: Outcome, allowed: bool) -> None:
    """Retain evidence and policy through the AlphaModel's serialisable check.

    1. Classify post-cutoff native permission and capacity.
    2. Serialise and reuse the structured deposit check without losing evidence.
    3. Relax only strategy policy, retaining venue closure and unknown blocks.
    """
    # 1. BitRocket-like open vaults can have a zero conservative policy cap.
    pair = vault(native=True)
    kwargs = dict(timestamp=datetime.datetime(2026, 10, 20), historical_state={"deposits_open": flag, "max_deposit": cap})
    result = get_vault_closure(pair, **kwargs)
    assert result.status is outcome
    assert can_interact_with_vault(result) is allowed
    assert is_vault_closed(result) is (outcome is Outcome.closed)
    # 2. JSON is the state persistence boundary, not just an in-memory copy.
    check = DepositCheckResult.from_json(result.as_deposit_check(pair).to_json())
    restored = get_vault_closure(pair, deposit_check=check)
    assert restored == result
    # 3. Captured checks preserve the same optional policy semantics.
    relaxed = get_vault_closure(pair, deposit_check=check, include_strategy_policy=False)
    assert relaxed.status is outcome
    assert relaxed.blocked is (outcome in (Outcome.closed, Outcome.unknown))


def test_cutoff_unknown_and_invalid_controls() -> None:
    """Require historical time and prevent accidental assessment reuse.

    1. Confirm the documented pre-collection assumption and legacy field wrapper.
    2. Reject missing times, misspellings and controls on captured evidence.
    """
    # 1. Pre-collection is an assumption, never observed open permission.
    pair = vault(native=True)
    result = get_vault_closure(pair, timestamp=datetime.datetime(2025, 1, 1), historical_state={"deposits_open": False})
    assert result.status is Outcome.unknown and not result.blocked
    assert not is_backtesting_deposit_open(pair, False, 0)
    # 2. A new operation/time requires a fresh query.
    with pytest.raises(ValueError, match="require a timestamp"):
        get_vault_closure(pair, historical_state={"deposits_open": False})
    with pytest.raises(TypeError):
        is_vault_closed(pair, typo=True)
    with pytest.raises(ValueError):
        get_vault_closure(pair, operation="borrow")
    for kwargs in ({"unknown_is_closed": True}, {"operation": "redemption"}, {"timestamp": datetime.datetime(2026, 1, 1)}):
        with pytest.raises(ValueError, match="captured assessment"):
            can_interact_with_vault(result, **kwargs)
    check = result.as_deposit_check(pair)
    for kwargs in ({"timestamp": datetime.datetime(2026, 1, 1)}, {"position": object()}, {"trust_protocol_state": True}, {"apply_hypercore_cutoff": False}, {"operation": "redemption"}):
        with pytest.raises(ValueError):
            get_vault_closure(pair, deposit_check=check, **kwargs)
    with pytest.raises(TypeError):
        VaultClosure(False, "open", "test")


def test_live_missing_owner_and_adapter_failure_are_unknown() -> None:
    """A failed live admission check does not establish venue closure.

    1. Mock unavailable account/RPC inputs while retaining the real live adapter.
    2. Check both missing owner and unsupported flow remain unknown and blocked.
    """
    # 1. Mocking the deposit manager avoids RPC and makes failures deterministic.
    pair = vault()
    manager = Mock()
    manager.has_synchronous_deposit.return_value = False
    adapter = SimpleNamespace(get_deposit_manager=lambda: manager)
    # 2. Neither failure path supplies positive closure evidence.
    check = check_live_deposit(timestamp=None, pair=pair, vault=adapter, owner=None, stage=DepositCheckStage.unknown)
    outcome = get_vault_closure(pair, deposit_check=check)
    assert outcome.status is Outcome.unknown and outcome.blocked
    assert not is_vault_closed(outcome)
    manager.can_create_deposit_request.assert_not_called()
    manager.can_create_deposit_request.side_effect = VaultFlowUnavailable("Admission could not be checked")
    check = check_live_deposit(timestamp=None, pair=pair, vault=adapter, owner="0x01", stage=DepositCheckStage.unknown)
    outcome = get_vault_closure(pair, deposit_check=check)
    assert outcome.status is Outcome.unknown and outcome.blocked


def test_live_native_policy_reuses_the_captured_response() -> None:
    """Reuse the shared protocol classifier and preserve mandatory owner limits.

    1. Capture API-open permission with a low leader-share policy cap.
    2. Relax the policy, then remove native evidence and retain the owner block.
    """
    # 1. The snapshot is the recorded live response; no network query is needed.
    pair = vault(native=True)
    check = DepositCheckResult(can_deposit=False, reason_code=DepositBlockReason.vault_max_deposit_zero,
        max_deposit=0, used_vault_info=VaultInfoSnapshot(is_closed=False, allow_deposits=True, leader_fraction=.05, relationship_type="normal"))
    assessment = get_vault_closure(pair, deposit_check=check)
    assert assessment.status is Outcome.policy_blocked
    assert not is_vault_closed(assessment)
    assert not can_interact_with_vault(assessment)
    # 2. An ordinary owner-scoped zero cap cannot be waived as strategy policy.
    assert can_interact_with_vault(pair, deposit_check=check, include_strategy_policy=False)
    check.used_vault_info.relationship_type = None
    assert not can_interact_with_vault(pair, deposit_check=check, include_strategy_policy=False)
    check.used_vault_info = None
    assessment = get_vault_closure(pair, deposit_check=check, include_strategy_policy=False)
    assert assessment.status is Outcome.account_restricted and assessment.blocked
    check.reason_code = DepositBlockReason.deposit_request_unavailable
    assert get_vault_closure(pair, deposit_check=check).status is Outcome.account_restricted


@pytest.mark.parametrize("reason,outcome", [
    (RedemptionBlockReason.user_lockup_not_expired, Outcome.account_restricted),
    (RedemptionBlockReason.vault_max_withdrawable_zero, Outcome.account_restricted),
    (RedemptionBlockReason.safe_address_unavailable, Outcome.unknown),
    (RedemptionBlockReason.user_equity_fetch_failed, Outcome.unknown),
    (RedemptionBlockReason.redemption_window_closed, Outcome.closed),
    (None, Outcome.account_restricted),
])
def test_live_redemption_restrictions(reason: RedemptionBlockReason | None, outcome: Outcome) -> None:
    """Never waive real live lockups or label account limits as global closure.

    1. Supply a captured owner-aware response through a minimal pricing provider.
    2. Verify policy relaxation preserves hard restrictions and unknown controls.
    """
    # 1. The provider double stands in for RPC only; the assessment rule is real.
    result = RedemptionCheckResult(can_redeem=False, reason_code=reason, max_redemption=0)
    pricing = SimpleNamespace(check_redemption=lambda *args, **kwargs: result)
    # 2. Even permissive overrides cannot bypass an account lockup or closure.
    assessment = get_vault_closure(vault(native=True), operation="redemption", pricing_model=pricing,
        include_strategy_policy=False, unknown_is_closed=False)
    assert assessment.status is outcome
    assert assessment.blocked is (outcome is not Outcome.unknown)
    assert is_vault_closed(assessment) is (outcome is Outcome.closed)
