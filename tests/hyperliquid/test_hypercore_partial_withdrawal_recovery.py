"""Reconcile the real October withdrawal checkpoint without live transfers."""

import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tradeexecutor.ethereum.vault import hypercore_withdrawal_recovery as recovery
from tradeexecutor.ethereum.vault import hypercore_transit_recovery as transit
from tradeexecutor.state.state import State
from tradeexecutor.state.store import JSONFileStore
from tradeexecutor.state.repair import repair_tx_not_generated, repair_zero_quantity
from tradeexecutor.state.trade import TradeStatus, has_unresolved_hypercore_accounting
from tradeexecutor.strategy.account_correction import preflight_state_for_account_correction


@pytest.fixture()
def incident(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Use a reduced production checkpoint and mock only remote custody APIs."""
    source = Path(__file__).parent / "state" / "hyper-ai-oct26-withdrawal.json"
    state = State.read_json_file(source)
    repair_tx_not_generated(state, interactive=False)
    repair_zero_quantity(state, interactive=False)
    store = JSONFileStore(tmp_path / "state.json")
    store.sync(state)
    trade = state.portfolio.get_trade_by_id(1825)
    safe = "0xa8f8debb722c6174b814b432169bf569603f673f"
    initial_safe = state.portfolio.get_default_reserve_position().quantity
    custody = {"perp": Decimal("3950.977563"), "spot": Decimal("0.008466"), "safe": initial_safe}
    evidence = {"trade_id":1825, "first_execution_hash":"0xfirst", "second_execution_hash":"0xsecond"}
    def post(url: str, *, json: dict, timeout: int) -> MagicMock:
        if json["type"] == "txDetails":
            first = json["hash"] == evidence["first_execution_hash"]
            result = {"tx": {"hash": json["hash"], "user": safe,
                "time": 1792977883129 if first else 1792977902092,
                "action": {"type":"vaultTransfer", "vaultAddress":trade.pair.pool_address,
                    "isDeposit":False, "usd":3947465150 if first else 18336496},
                "error":None if first else "Vault deposits must be at least $5."}}
        else:
            result = [{"hash":"0xfirst", "delta":{"type":"vaultWithdraw", "vault":trade.pair.pool_address,
                "netWithdrawnUsd":"3947.465150", "requestedUsd":"3947.465150"}}]
        response = MagicMock()
        response.json.return_value = result
        return response
    monkeypatch.setattr(recovery.requests, "post", post)
    monkeypatch.setattr(recovery, "create_hyperliquid_session", MagicMock())
    monkeypatch.setattr(recovery, "fetch_user_vault_equity", lambda *a, **kw: SimpleNamespace(equity=Decimal("19.836495")))
    def snapshot(**kwargs) -> transit.HypercoreTransitBalanceSnapshot:
        return transit.HypercoreTransitBalanceSnapshot(safe, custody["safe"], custody["spot"], custody["spot"], custody["perp"], custody["perp"], 0)
    monkeypatch.setattr(transit, "fetch_hypercore_transit_balances", snapshot)
    web3 = MagicMock()
    web3.eth.chain_id = 999
    web3.eth.get_transaction_receipt.side_effect = lambda h: {"status":1, "blockNumber":1 if h == trade.blockchain_transactions[0].tx_hash else 2}
    web3.eth.get_block.side_effect = lambda b: {"timestamp":1792977879 if b == 1 else 1792977898}
    sync = MagicMock()
    sync.get_token_storage_address.return_value = safe
    return SimpleNamespace(state=state, store=store, evidence=evidence, web3=web3, sync=sync, custody=custody, initial_safe=initial_safe)


def test_partial_sale_preview_and_recovery(incident: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """Return proven cash and book the partial sale once, retaining the residual.

    1. Verify sibling repairs and evidence-led preview without state writes.
    2. Mock external transfers while keeping actual state accounting and saves.
    3. Verify one sale credit, surviving residual and a resolved marker.
    """
    # 1. Verify sibling repairs and evidence-led preview without state writes.
    i = incident
    preflight_state_for_account_correction(i.state, allow_at_risk_hypercore_withdrawal=True)
    assert all(i.state.portfolio.get_trade_by_id(n).get_status() == TradeStatus.repaired for n in (1822,1823,1824))
    before = i.store.path.read_bytes()
    proof = recovery.inspect_partial_withdrawal(i.state, i.sync, i.web3, i.evidence)
    recovery.recover_partial_withdrawal(proof, i.state, i.store, i.sync, i.web3, None, True)
    assert i.store.path.read_bytes() == before
    # 2. Mock external transfers while keeping actual state accounting and saves.
    def transfer(**kwargs) -> list[str]:
        actions = kwargs["actions"]
        i.custody.update(perp=Decimal("0.5"), spot=Decimal("0.01"), safe=i.initial_safe + actions[-1].amount)
        for action in actions:
            kwargs["on_completed"](action.action_kind, "0xrecovery")
        return [a.action_kind for a in actions]
    monkeypatch.setattr(transit, "execute_hypercore_transit_recovery_actions", transfer)
    recovery.recover_partial_withdrawal(proof, i.state, i.store, i.sync, i.web3, MagicMock(), False)
    # 3. Verify one sale credit, surviving residual and a resolved marker.
    t = i.state.portfolio.get_trade_by_id(1825)
    position = i.state.portfolio.get_position_by_id(554)
    assert t.get_status() == TradeStatus.success
    assert position.get_quantity() > 0
    assert position.get_value() == pytest.approx(19.836495)
    assert i.state.portfolio.get_default_reserve_position().quantity == i.initial_safe + t.executed_reserve
    assert not has_unresolved_hypercore_accounting(t)
    # Persisted transfer checkpoints intentionally remain unresolved until the
    # command's final balance checks and atomic accounting/slot save.
    assert has_unresolved_hypercore_accounting(i.store.load().portfolio.get_trade_by_id(1825))


def test_recovery_after_transfer_before_save(incident: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """A completed external transfer can be reconciled without sending twice.

    1. Persist completion evidence while leaving original sale accounting intact.
    2. Reinspect custody and reconcile without calling the transfer helper.
    3. Refuse ambiguous intermediate custody before any transfer.
    4. Reject a mistyped evidence trade ID with an operator-facing error.
    """
    # 1. Persist completion evidence while leaving original sale accounting intact.
    i = incident
    proof = recovery.inspect_partial_withdrawal(i.state, i.sync, i.web3, i.evidence)
    actions = transit.plan_hypercore_transit_recovery_actions(proof.snapshot)
    expected = actions[-1].amount
    proof.trade.other_data["hypercore_withdrawal_recovery"] = {
        "expected_return":str(expected), "completed":{"perp_to_spot":"0xa", "spot_to_evm":"0xb"}}
    i.store.sync(i.state)
    i.custody.update(perp=Decimal("0.5"), spot=Decimal("0.01"), safe=i.initial_safe + expected)
    # 2. Reinspect custody and reconcile without calling the transfer helper.
    transfer = MagicMock(side_effect=AssertionError("Must not send twice"))
    monkeypatch.setattr(transit, "execute_hypercore_transit_recovery_actions", transfer)
    proof = recovery.inspect_partial_withdrawal(i.state, i.sync, i.web3, i.evidence)
    assert proof.already_returned
    recovery.recover_partial_withdrawal(proof, i.state, i.store, i.sync, i.web3, None, False)
    transfer.assert_not_called()
    assert i.state.portfolio.get_trade_by_id(1825).get_status() == TradeStatus.success
    # 3. Refuse ambiguous intermediate custody before any transfer.
    i.state = i.store.load()
    i.custody.update(spot=Decimal("3950"), safe=i.initial_safe)
    with pytest.raises(RuntimeError, match="custody"):
        recovery.inspect_partial_withdrawal(i.state, i.sync, i.web3, i.evidence)

    # 4. Reject a mistyped evidence trade ID with an operator-facing error.
    with pytest.raises(RuntimeError, match="identify a HyperCore sell"):
        recovery.inspect_partial_withdrawal(i.state, i.sync, i.web3, dict(i.evidence, trade_id=999999))
