"""Public helpers preserve research semantics without changing execution policy."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from tradeexecutor.backtest.vault_fees import FEE_KEY

import pandas as pd
import pytest
from tradingstrategy.vault import VaultDepositPermission
from tradeexecutor.analysis.vault import mark_whitelisted_vaults_ignored
from tradeexecutor.curator.vault_universe_creation import is_native_public_deposit_closure

from tradeexecutor.analysis.vault import adjust_vault_period_returns, summarise_native_vault_fees
from tradeexecutor.curator.vault_universe_creation import private_vault_exclusion_reasons
from tradeexecutor.state.other_data import OtherData
from tradeexecutor.strategy.vault_fee_data import get_missing_vault_fee_data
from tradeexecutor.testing.vault_fee_sample import load_fee_sample


def test_latest_present_restored_cycles():
    """Keep weekly state available across skipped writes and JSON restoration.

    1. Restore sparse cycles. 2. Save a new cycle. 3. Reject ambiguous keys.
    """
    # 1. Restore sparse cycles, including a stored null.
    store = OtherData(data={'2': {'members': [7]}, '10': {'null': None}})
    assert store.load_latest_present('members') == [7]
    assert store.load_latest_present('null', 'missing') is None
    assert store.load_latest_present('absent', 'missing') == 'missing'
    assert OtherData().load_latest_present('absent') is None
    # 2. Save a new cycle after restoring JSON string keys.
    store.save(11, 'members', [8])
    assert store.load_latest_present('members') == [8]
    restored = OtherData.from_json(store.to_json())
    assert restored.load_latest_present('members') == [8]
    # 3. Reject ambiguous keys rather than choosing a winner.
    store.data['11'] = {}
    with pytest.raises(ValueError):
        store.load_latest_present('members')


def test_restriction_evidence():
    """Preserve legacy filtering without making unknown access a restriction.

    1. Test unknown/public cases. 2. Test each explicit restriction form.
    """
    # 1. Unknown evidence and private-credit descriptions are not prohibitions.
    for row in ({}, {'notes': None, 'whitelist': None}, {'notes': 'private credit'}):
        assert private_vault_exclusion_reasons(row) == []
    # 2. Explicit permission, flags and notes each give a reason.
    for row in ({'deposit_permission': 'WHITELISTED'}, {'whitelist': {'status': 'restricted'}},
                {'flags': ['private-vault']}, {'notes': 'A private Lagoon vault'}):
        assert len(private_vault_exclusion_reasons(row)) == 1


def test_fee_ranking_independent_arithmetic():
    """Preserve ranking arithmetic independently of the helper implementation.

    1. Load real externalised metadata. 2. Check net arithmetic. 3. Check unknown/zero charges.
    """
    # 1. Load the existing frozen fixture; no network or fee-model mocks.
    universe = load_fee_sample(Path(__file__).resolve().parents[1] / 'backtest/fixtures/vault_externalised_fees', 'hyperliquid')
    metadata = next(universe.iterate_pairs()).get_vault_metadata()
    gross = pd.Series([.1, -.1, float('nan')])
    metadata = replace(metadata, management_fee=.02, performance_fee=.1, deposit_fee=.01, withdrawal_fee=.005)
    assert not get_missing_vault_fee_data(metadata)
    # 2. Positive returns pay performance fees after prorated management fees.
    after = gross - .02 * 90 / 365.25
    expected = (1 + after.where(after <= 0, after * .9)) * .99 * .995 - 1
    pd.testing.assert_series_equal(adjust_vault_period_returns(gross, metadata, 90), expected)
    # 3. Missing entry data remains NaN, while explicit zero fees preserve returns.
    assert adjust_vault_period_returns(gross, replace(metadata, deposit_fee=None), 90).isna().all()
    zero = replace(metadata, management_fee=0, performance_fee=0, deposit_fee=0, withdrawal_fee=0)
    pd.testing.assert_series_equal(adjust_vault_period_returns(gross, zero, 90), (1 + gross) - 1)
    internal = next(load_fee_sample(Path(__file__).resolve().parents[1] / 'backtest/fixtures/vault_externalised_fees', 'lagoon').iterate_pairs()).get_vault_metadata()
    internal = replace(internal, fee_internalised=True)
    pd.testing.assert_series_equal(adjust_vault_period_returns(gross, internal, 90),
        (1 + gross) * (1 - internal.deposit_fee) * (1 - internal.withdrawal_fee) - 1)
    with pytest.raises(ValueError):
        adjust_vault_period_returns(gross, zero, 0)


def test_native_fee_report_missing_failed_and_pending():
    """1. Build report-only ledger stand-ins. 2. Verify scope, status and component totals.

    Minimal stand-ins isolate reporting from trading; actual fee execution is
    covered by the existing externalised-vault-fee integration fixture.
    """
    # 1. Include successful, unrecorded, failed and pending sales, plus non-native.
    def sale(success, record=None):
        return SimpleNamespace(is_sell=lambda: True, is_success=lambda: success,
            other_data={} if record is None else {FEE_KEY: {'realised': record}})
    native = SimpleNamespace(is_hyperliquid_vault=lambda: True,
        get_vault_name=lambda: 'Native', base=SimpleNamespace(token_symbol='USDC'))
    other = SimpleNamespace(is_hyperliquid_vault=lambda: False, is_lighter_pool=lambda: False)
    recorded = sale(True, {'management':'1.10','performance':'2.20','exit_fee':'0.30'})
    position = SimpleNamespace(pair=native, trades={1:recorded,2:sale(True),3:sale(False),4:sale(False)})
    calls = []
    def positions(pending=False):
        calls.append(pending)
        return [position, SimpleNamespace(pair=other, trades={})]
    state = SimpleNamespace(portfolio=SimpleNamespace(get_all_positions=positions))
    # 2. Count only successful exits; report missing evidence without deducting again.
    assert summarise_native_vault_fees(state) == [{'Vault':'Native','Successful sells':2,
        'Recorded fee sells':1,'Recorded external fees USD':3.6,'Sells without external-fee record':1}]
    assert calls == [True]
    assert recorded.other_data[FEE_KEY]['realised']['management'] == '1.10'


def test_native_closure_compatibility_preserves_real_access_restrictions():
    """Keep native closure histories without treating genuine private access as public.

    1. Recognise only the native producer compatibility label.
    2. Preserve the old default and genuine private restrictions with the opt-in.
    3. Exempt approved history identities from blanket ignores, retaining other reasons.
    """
    # 1. A currently closed native public pool is different from a private EVM vault.
    row = {'chain_id':9998, 'protocol_slug':'lighter', 'deposit_permission':'whitelisted',
           'whitelist':{'status':'whitelisted','notes':'Native perp DEX compatibility status: public deposits are unavailable; this does not imply an approved-account route.'}}
    assert is_native_public_deposit_closure(row)
    assert not is_native_public_deposit_closure(dict(row,chain_id=1))
    # 2. The legacy control stays intact; private flags and notes remain restrictions.
    assert private_vault_exclusion_reasons(row)
    assert not private_vault_exclusion_reasons(row,historical_native_permissions=True)
    assert private_vault_exclusion_reasons(dict(row,flags=['private-vault']),historical_native_permissions=True)
    assert private_vault_exclusion_reasons(dict(row,notes='private vault'),historical_native_permissions=True)
    # 3. Use a real loaded native pair with incomplete-fee ignore reason already applied.
    universe = load_fee_sample(Path(__file__).resolve().parents[1] / 'backtest/fixtures/vault_externalised_fees', 'hyperliquid')
    pair = next(universe.iterate_pairs())
    metadata = pair.get_vault_metadata()
    pair.other_data["token_metadata"] = replace(metadata,deposit_permission=VaultDepositPermission.whitelisted)
    pair.set_ignore_reason('lacks_fee_data')
    keys = {(int(pair.chain_id),pair.pool_address.lower())}
    assert mark_whitelisted_vaults_ignored(universe,historical_native_closure_keys=keys) == []
    assert pair.get_ignore_reason() == 'lacks_fee_data'
    assert mark_whitelisted_vaults_ignored(universe) == [pair]
