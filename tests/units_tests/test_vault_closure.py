"""Shared admission rules: protocol evidence must not be confused with policy."""
import datetime
from unittest.mock import Mock
from tradeexecutor.strategy.redemption import DepositCheckStage
from tradeexecutor.strategy.vault_closure import check_live_deposit
from eth_defi.vault.deposit_redeem import VaultFlowUnavailable
from tradeexecutor.strategy.redemption import DepositCheckResult, DepositBlockReason, VaultInfoSnapshot
from types import SimpleNamespace

import pandas as pd
import pytest
from eth_defi.erc_4626.core import ERC4626Feature

from tradeexecutor.strategy.vault_closure import (
    VaultClosure, VaultClosureOutcome as Outcome, can_interact_with_vault,
    get_vault_closure, is_vault_closed,
)


def vault(feature=None, native=False):
    return SimpleNamespace(
        get_vault_features=lambda: {feature} if feature else set(),
        get_vault_protocol=lambda: None,
        is_hyperliquid_vault=lambda: native,
    )


@pytest.mark.parametrize('feature', [ERC4626Feature.lagoon_like, ERC4626Feature.morpho_v2_like, ERC4626Feature.yearn_v3_like])
def test_non_admission_maxima(feature):
    """Zero maxima from claim/account-specific APIs do not establish closure."""
    pair = vault(feature)
    state = {'deposits_open': False, 'max_deposit': 0, 'max_redeem': 0}
    for operation in ('deposit', 'redemption'):
        result = get_vault_closure(pair, historical_state=state, operation=operation)
        assert result.status is Outcome.unknown
        assert not is_vault_closed(pair, assessment=result)
        assert can_interact_with_vault(pair, assessment=result)
    # An explicitly verified admission source can opt out of the assumption.
    assert is_vault_closed(pair, historical_state=state, trust_protocol_state=True)
    result = get_vault_closure(pair, historical_state=state, unknown_is_closed=True)
    assert result.blocked and result.status is Outcome.unknown


@pytest.mark.parametrize('flag,cap,outcome,allowed', [
    (True, 0, Outcome.policy_blocked, False),
    (False, 0, Outcome.closed, False),
    (True, None, Outcome.open, True),
    (pd.NA, None, Outcome.unknown, False),
])
def test_native_permission_and_policy(flag, cap, outcome, allowed):
    """Open BitRocket-like vaults with a zero policy cap are not closed venues."""
    pair = vault(native=True)
    kwargs = dict(timestamp=datetime.datetime(2026, 10, 20), historical_state={'deposits_open': flag, 'max_deposit': cap})
    result = get_vault_closure(pair, **kwargs)
    assert result.status is outcome
    assert can_interact_with_vault(pair, assessment=result) is allowed
    assert is_vault_closed(pair, assessment=result) is (outcome is Outcome.closed)
    relaxed = get_vault_closure(pair, **kwargs, include_strategy_policy=False)
    assert relaxed.status is outcome
    assert relaxed.blocked is (outcome in (Outcome.closed, Outcome.unknown))


def test_cutoff_unknown_and_invalid_controls():
    """No historical evidence is assumed open before collection, never observed open."""
    pair = vault(native=True)
    result = get_vault_closure(pair, timestamp=datetime.datetime(2025, 1, 1), historical_state={'deposits_open': False})
    assert result.status is Outcome.unknown and not result.blocked
    with pytest.raises(TypeError):
        is_vault_closed(pair, typo=True)
    with pytest.raises(ValueError):
        get_vault_closure(pair, operation='borrow')
    with pytest.raises(ValueError):
        get_vault_closure(pair, assessment=result, unknown_is_closed=True)
    with pytest.raises(TypeError):
        VaultClosure(False, 'open', 'test')


def test_live_missing_owner_and_adapter_failure_are_unknown():
    """A failed live admission check must not be reported as venue closure."""
    pair = SimpleNamespace(get_ticker=lambda: 'TEST', pool_address='0x01')
    manager = Mock()
    manager.has_synchronous_deposit.return_value = False
    adapter = SimpleNamespace(get_deposit_manager=lambda: manager)
    check = check_live_deposit(timestamp=None, pair=pair, vault=adapter,
                               owner=None, stage=DepositCheckStage.unknown)
    outcome = get_vault_closure(pair, deposit_check=check)
    assert outcome.status is Outcome.unknown and outcome.blocked
    assert not is_vault_closed(pair, assessment=outcome)
    manager.can_create_deposit_request.assert_not_called()
    manager.can_create_deposit_request.side_effect = VaultFlowUnavailable('Admission could not be checked')
    check = check_live_deposit(timestamp=None, pair=pair, vault=adapter,
                               owner='0x01', stage=DepositCheckStage.unknown)
    outcome = get_vault_closure(pair, deposit_check=check)
    assert outcome.status is Outcome.unknown and outcome.blocked
    assert not is_vault_closed(pair, assessment=outcome)


def test_live_native_policy_reuses_the_captured_response():
    """A live zero policy cap is distinct from leader-disabled deposits."""
    pair = vault(native=True)
    check = DepositCheckResult(can_deposit=False, reason_code=DepositBlockReason.vault_max_deposit_zero,
        max_deposit=0, used_vault_info=VaultInfoSnapshot(is_closed=False, allow_deposits=True, leader_fraction=.05))
    assessment = get_vault_closure(pair, deposit_check=check)
    assert assessment.status is Outcome.policy_blocked
    assert not is_vault_closed(pair, assessment=assessment)
    assert not can_interact_with_vault(pair, assessment=assessment)
    assert can_interact_with_vault(pair, deposit_check=check, include_strategy_policy=False)
