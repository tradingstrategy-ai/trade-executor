"""Exercise an atomic guard migration on a real Safe in an Anvil fork."""

import pytest
from eth_defi.hotwallet import HotWallet
from eth_defi.provider.receipt import wait_for_transaction_receipt_robust
from eth_defi.safe.deployment import deploy_safe
from eth_defi.safe.execute import execute_safe_tx
from hexbytes import HexBytes
from safe_eth.safe.api.transaction_service_api.transaction_service_api import TransactionServiceApi
from web3 import Web3

import tradeexecutor.ethereum.lagoon.guard_proposal as guard_proposal
from tradeexecutor.ethereum.lagoon.guard_proposal import prepare_guard_proposal, submit_guard_migration


@pytest.mark.timeout(300)
def test_guard_migration_safe_batch_executes_atomically(
    web3: Web3,
    deployer_hot_wallet: HotWallet,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute the exact signed Safe proposal and observe both module changes.

    1. Deploy a one-owner Safe and install two guard-shaped contracts on the fork.
    2. Enable the old module with a signed Safe transaction.
    3. Submit the migration to a mocked Transaction Service and execute its signed batch.
    4. Verify the old module is disabled, the new module is enabled, and the nonce advanced.
    """
    owner = deployer_hot_wallet.account
    old_guard = Web3.to_checksum_address("0x0000000000000000000000000000000000000101")
    new_guard = Web3.to_checksum_address("0x0000000000000000000000000000000000000102")

    # 1. Deploy a one-owner Safe and install two guard-shaped contracts on the fork.
    safe = deploy_safe(web3, owner, owners=[owner.address], threshold=1, post_deploy_delay_seconds=0)
    # The stub implements getGovernanceAddress() by returning the Safe address.
    # It is sufficient here because module enablement is governed by the Safe,
    # while the production guard deployment is covered by existing Lagoon tests.
    guard_runtime = "0x73" + safe.address[2:] + "60005260206000f3"
    assert "error" not in web3.provider.make_request("anvil_setCode", [old_guard, guard_runtime])
    assert "error" not in web3.provider.make_request("anvil_setCode", [new_guard, guard_runtime])

    # 2. Enable the old module with a signed Safe transaction.
    enable_old = safe.build_multisig_tx(
        to=safe.address,
        value=0,
        data=HexBytes(safe.contract.functions.enableModule(old_guard)._encode_transaction_data()),
    )
    enable_old.sign(owner.key.hex())
    enable_hash, _ = execute_safe_tx(enable_old, owner.key.hex(), tx_gas=300_000)
    assert wait_for_transaction_receipt_robust(web3, enable_hash)["status"] == 1
    assert safe.retrieve_modules() == [old_guard]
    old_nonce = safe.retrieve_nonce()
    with pytest.raises(ValueError, match="expected"):
        prepare_guard_proposal(web3, safe.address, owner.key.hex(), expected_old_guard_address=new_guard)
    captured = {}

    class FakeTransactionService:
        NETWORK_SHORTNAME = TransactionServiceApi.NETWORK_SHORTNAME

        def __init__(self, **_kwargs):
            pass

        def get_transactions(self, _address, **_kwargs):
            return []

        def post_transaction(self, safe_tx):
            captured["safe_tx"] = safe_tx

    # Mock the external Service only; the Safe, batch encoding and execution are real.
    monkeypatch.setattr(guard_proposal, "TransactionServiceApi", FakeTransactionService)

    # 3. Submit the migration to a mocked Transaction Service and execute its signed batch.
    context = prepare_guard_proposal(web3, safe.address, owner.key.hex(), expected_old_guard_address=old_guard)
    proposal = submit_guard_migration(context, new_guard, owner.key.hex())
    assert proposal["operation"] == 1
    assert captured["safe_tx"].safe_nonce == old_nonce
    migration_hash, _ = execute_safe_tx(captured["safe_tx"], owner.key.hex(), tx_gas=500_000)
    assert wait_for_transaction_receipt_robust(web3, migration_hash)["status"] == 1

    # 4. Verify the old module is disabled, the new module is enabled, and the nonce advanced.
    assert safe.retrieve_modules() == [new_guard]
    assert safe.contract.functions.isModuleEnabled(old_guard).call() is False
    assert safe.contract.functions.isModuleEnabled(new_guard).call() is True
    assert safe.retrieve_nonce() == old_nonce + 1
