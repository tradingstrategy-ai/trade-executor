"""Reconcile the real October withdrawal checkpoint without live transfers."""

import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import launch_anvil
from eth_defi.provider.multi_provider import create_multi_provider_web3

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


def test_recovery_signs_with_initialised_nonce(incident: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise real wallet signing and broadcast rather than mocking the transit helper.

    1. Start local Anvil and an unsynchronised wallet.
    2. Replace HyperCore calls/polls only, because Anvil has no HyperCore backend.
    3. Run recovery through both real transaction broadcasts and verify consecutive nonces.
    """
    # 1. Start local Anvil and an unsynchronised wallet.
    i = incident
    legacy_audit = {"expected_return": "3950.476029", "completed": {}}
    i.state.portfolio.get_trade_by_id(1825).other_data["hypercore_withdrawal_recovery"] = legacy_audit
    proof = recovery.inspect_partial_withdrawal(
        i.state, i.sync, i.web3, dict(i.evidence, recovery_failed_before_signing=True),
    )
    anvil = launch_anvil()
    try:
        web3 = create_multi_provider_web3(anvil.json_rpc_url)
        wallet = HotWallet.from_private_key("0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80")
        assert wallet.current_nonce is None
        initial_nonce = web3.eth.get_transaction_count(wallet.address)
        # 2. Replace HyperCore calls/polls only, because Anvil has no HyperCore backend.
        # A bound call to a local empty account accepts calldata and yields a real
        # EVM receipt; it does not claim to validate HyperCore execution semantics.
        contract = web3.eth.contract(address=web3.eth.accounts[1], abi=[{
            "type": "function", "name": "recover", "inputs": [], "outputs": [], "stateMutability": "nonpayable",
        }])
        monkeypatch.setattr(transit, "build_hypercore_transfer_usd_class_call", lambda *a, **kw: contract.functions.recover())
        monkeypatch.setattr(transit, "build_hypercore_send_asset_to_evm_call", lambda *a, **kw: contract.functions.recover())
        i.sync.vault.underlying_token.convert_to_raw.side_effect = i.state.portfolio.get_default_reserve_position().asset.convert_to_raw_amount
        def spot_arrived(*args, **kwargs) -> Decimal:
            i.custody["perp"] -= kwargs["expected_increase"]
            i.custody["spot"] += kwargs["expected_increase"]
            return i.custody["spot"]
        def evm_arrived(*args, **kwargs) -> Decimal:
            i.custody["safe"] += kwargs["expected_increase"]
            i.custody["spot"] -= kwargs["expected_increase"]
            return i.custody["safe"]
        monkeypatch.setattr(transit, "wait_for_spot_free_balance", spot_arrived)
        monkeypatch.setattr(transit, "wait_for_evm_usdc_balance", evm_arrived)
        # 3. Run recovery through both real transaction broadcasts and verify consecutive nonces.
        recovery.recover_partial_withdrawal(proof, i.state, i.store, i.sync, web3, wallet, False)
        audit = proof.trade.other_data["hypercore_withdrawal_recovery"]
        assert audit["previous_attempt"] == legacy_audit
        transactions = [web3.eth.get_transaction(tx) for tx in audit["completed"].values()]
        assert [tx["nonce"] for tx in transactions] == [initial_nonce, initial_nonce + 1]
        assert proof.trade.get_status() == TradeStatus.success
        assert i.store.load().portfolio.get_trade_by_id(1825).executed_at is None
    finally:
        anvil.close()


def test_pre_signing_failure_retry_is_explicit(incident: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not silently retry an empty checkpoint or persist a failed wallet setup.

    1. Reproduce the legacy empty checkpoint and require explicit operator attestation.
    2. Verify preview is read-only and a wallet setup failure cannot replace the checkpoint.
    3. Refuse attestation for new attempts, completed legs or changed custody.
    """
    # 1. Reproduce the legacy empty checkpoint and require explicit operator attestation.
    i = incident
    trade = i.state.portfolio.get_trade_by_id(1825)
    trade.other_data["hypercore_withdrawal_recovery"] = {"expected_return": "3950.476029", "completed": {}}
    i.store.sync(i.state)
    with pytest.raises(RuntimeError, match="custody"):
        recovery.inspect_partial_withdrawal(i.state, i.sync, i.web3, i.evidence)
    evidence = dict(i.evidence, recovery_failed_before_signing=True)
    proof = recovery.inspect_partial_withdrawal(i.state, i.sync, i.web3, evidence)
    # 2. Verify preview is read-only and a wallet setup failure cannot replace the checkpoint.
    before = i.store.path.read_bytes()
    recovery.recover_partial_withdrawal(proof, i.state, i.store, i.sync, i.web3, None, True)
    wallet = MagicMock()
    wallet.sync_nonce.side_effect = RuntimeError("RPC unavailable")
    with pytest.raises(RuntimeError, match="RPC unavailable"):
        recovery.recover_partial_withdrawal(proof, i.state, i.store, i.sync, i.web3, wallet, False)
    assert i.store.path.read_bytes() == before
    # 3. Refuse attestation for new attempts, completed legs or changed custody.
    wallet.sync_nonce.side_effect = None
    wallet.current_nonce = 5
    i.web3.eth.get_transaction_count.return_value = 6
    with pytest.raises(RuntimeError, match="Pending wallet"):
        recovery.recover_partial_withdrawal(proof, i.state, i.store, i.sync, i.web3, wallet, False)
    assert i.store.path.read_bytes() == before
    audit = trade.other_data["hypercore_withdrawal_recovery"]
    audit["started_at"] = "2026-10-26T13:00:00"
    with pytest.raises(RuntimeError, match="legacy empty"):
        recovery.inspect_partial_withdrawal(i.state, i.sync, i.web3, evidence)
    del audit["started_at"]
    audit["completed"] = {"perp_to_spot": "0xrecovery"}
    with pytest.raises(RuntimeError, match="legacy empty"):
        recovery.inspect_partial_withdrawal(i.state, i.sync, i.web3, evidence)
    audit["completed"] = {}
    i.custody["spot"] = Decimal("3950")
    with pytest.raises(RuntimeError, match="custody"):
        recovery.inspect_partial_withdrawal(i.state, i.sync, i.web3, evidence)
