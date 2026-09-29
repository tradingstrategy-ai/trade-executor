"""Safe Transaction Service proposals for Lagoon guard replacements."""

import json
import logging
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from eth_account import Account
from hexbytes import HexBytes
from safe_eth.safe.multi_send import MultiSendOperation
from tradingstrategy.chain import ChainId
from typer.main import get_command
from web3 import Web3

import tradeexecutor.cli.commands.lagoon_deploy_vault as deploy_command
import tradeexecutor.ethereum.lagoon.guard_proposal as guard_proposal
from tradeexecutor.cli.main import app
from tradeexecutor.ethereum.lagoon.guard_proposal import GuardProposalContext, submit_guard_migration


def _address(number: int) -> str:
    return Web3.to_checksum_address(f"0x{number:040x}")


def test_guard_proposal_batches_module_replacement_and_detects_nonce_conflict(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build one signed batch and reject a competing proposal at its Safe nonce.

    1. Arrange a Safe with one old guard and a governed replacement.
    2. Submit the signed migration through a mocked Transaction Service.
    3. Check the ordered zero-value calls, DelegateCall wrapper and returned URL.
    4. Retry against the same hash and then a conflicting pending hash.
    """
    captured: dict = {}
    old_guard, new_guard, safe_address, batch_address = (_address(i) for i in range(1, 5))
    owner = Account.create()
    transaction = SimpleNamespace(
        safe_tx_hash=HexBytes("0x" + "ab" * 32),
        data=HexBytes("0x1234"),
        safe_nonce=7,
        sign=lambda private_key: captured.setdefault("signed_by", private_key),
    )
    functions = SimpleNamespace(
        disableModule=lambda *_args: SimpleNamespace(_encode_transaction_data=lambda: "0x1111"),
        enableModule=lambda *_args: SimpleNamespace(_encode_transaction_data=lambda: "0x2222"),
    )
    safe = SimpleNamespace(
        address=safe_address,
        retrieve_modules=lambda: [old_guard],
        retrieve_nonce=lambda: 7,
        contract=SimpleNamespace(functions=functions),
        ethereum_client=object(),
        build_multisig_tx=lambda **kwargs: captured.setdefault("safe_tx", kwargs) and transaction,
    )
    web3 = SimpleNamespace(eth=SimpleNamespace(
        get_code=lambda _address: b"\x01",
        contract=lambda **_kwargs: SimpleNamespace(functions=SimpleNamespace(
            getGovernanceAddress=lambda: SimpleNamespace(call=lambda: safe_address),
        )),
    ))
    multisend = SimpleNamespace(
        address=batch_address,
        build_tx_data=lambda calls: captured.setdefault("calls", calls) and HexBytes("0x1234"),
    )
    context = GuardProposalContext(web3, safe, multisend, old_guard, guard_proposal.EthereumNetwork.BASE)
    pending = []

    class FakeTransactionService:
        NETWORK_SHORTNAME = guard_proposal.TransactionServiceApi.NETWORK_SHORTNAME

        def __init__(self, **kwargs):
            captured["service"] = kwargs

        def get_transactions(self, address, **kwargs):
            captured["lookup"] = (address, kwargs)
            return pending

        def post_transaction(self, safe_tx):
            captured["posted"] = safe_tx

    # 1. Arrange a Safe with one old guard and a governed replacement.
    monkeypatch.setattr(guard_proposal, "TransactionServiceApi", FakeTransactionService)

    # 2. Submit the signed migration through a mocked Transaction Service.
    proposal = submit_guard_migration(context, new_guard, owner.key.hex(), safe_api_key="test-key")

    # 3. Check the ordered zero-value calls, DelegateCall wrapper and returned URL.
    assert captured["signed_by"] == owner.key.hex()
    assert captured["posted"] is transaction
    assert captured["lookup"] == (safe_address, {"executed": False, "nonce": 7, "limit": 100, "offset": 0})
    assert captured["safe_tx"] == {
        "to": batch_address, "value": 0, "data": HexBytes("0x1234"), "operation": 1, "safe_nonce": 7,
    }
    assert [(call.operation, call.to, call.value, call.data) for call in captured["calls"]] == [
        (MultiSendOperation.CALL, safe_address, 0, HexBytes("0x1111")),
        (MultiSendOperation.CALL, safe_address, 0, HexBytes("0x2222")),
    ]
    assert proposal["status"] == "submitted"
    assert proposal["nonce"] == 7
    assert proposal["url"].startswith(f"https://app.safe.global/transactions/tx?safe=base:{safe_address}")
    with pytest.raises(RuntimeError, match="nonce changed since deployment"):
        submit_guard_migration(context, new_guard, owner.key.hex(), expected_proposal={"nonce": 6})

    # 4. Retry against the same hash and then a conflicting pending hash.
    pending.append({"safeTxHash": proposal["safe_tx_hash"]})
    submit_guard_migration(context, new_guard, owner.key.hex())
    pending.append({"safeTxHash": "0x" + "cd" * 32})
    with pytest.raises(RuntimeError, match="different pending proposal"):
        submit_guard_migration(context, new_guard, owner.key.hex())


def test_resubmit_guard_migration_updates_only_pending_chain(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Recover pending proposals and select one RPC for a single-chain record.

    1. Save one submitted chain and one pending chain in a deployment artefact.
    2. Retry with mocked chain validation and Transaction Service submission.
    3. Verify only the pending chain is submitted and persisted.
    4. Select one chain from several RPC connections for a single-chain record.
    """
    safe_address, old_guard, new_guard = (_address(i) for i in range(1, 4))
    existing = {"status": "submitted", "safe_tx_hash": "0x" + "ab" * 32, "url": "https://safe.invalid/existing"}
    migration = {"safe_address": safe_address, "old_guard_address": old_guard, "new_guard_address": new_guard, "safe_proposal": {"status": "pending"}}
    record = {"multichain": True, "deployments": {
        "base": {"guard_migration": {**migration, "safe_proposal": existing}},
        "arbitrum": {"guard_migration": migration},
    }}
    deployment_file = tmp_path / "vault.json"
    deployment_file.write_text(json.dumps(record))
    text_file = deployment_file.with_suffix(".txt")
    text_file.write_text("Chain: base\n    Safe proposal status: submitted\nChain: arbitrum\n    Safe proposal status: pending\nGuard report\n")
    calls = []
    monkeypatch.setattr(deploy_command, "prepare_guard_proposal", lambda *args, **kwargs: calls.append((args, kwargs)) or "context")
    monkeypatch.setattr(deploy_command, "submit_guard_migration", lambda *_args, **_kwargs: {"status": "submitted", "safe_tx_hash": "0x" + "cd" * 32, "url": "https://safe.invalid/tx"})

    # 1. Save one submitted chain and one pending chain in a deployment artefact.
    # 2. Retry with mocked chain validation and Transaction Service submission.
    results = deploy_command.resubmit_guard_migration(
        deployment_file,
        {"base": object(), "arbitrum": object()},
        "private-key",
    )

    # 3. Verify only the pending chain is submitted and persisted.
    assert len(calls) == 1
    assert calls[0][1]["expected_old_guard_address"] == old_guard
    assert results["base"] == existing
    assert results["arbitrum"]["status"] == "submitted"
    saved = json.loads(deployment_file.read_text())
    assert saved["deployments"]["arbitrum"]["guard_migration"]["safe_proposal"]["safe_tx_hash"] == "0x" + "cd" * 32
    updated_text = text_file.read_text()
    assert "Chain: base\n    Safe proposal status: submitted" in updated_text
    assert "Chain: arbitrum\n    Safe proposal status: submitted\n    Safe transaction: https://safe.invalid/tx" in updated_text
    assert "Safe proposal status: pending" not in updated_text

    # 4. Select one chain from several RPC connections for a single-chain record.
    deployment_file.write_text(json.dumps({
        "Guard migration": {**migration, "safe_proposal": {"status": "pending"}},
    }))
    with pytest.raises(ValueError, match="use --chain-name"):
        deploy_command.resubmit_guard_migration(deployment_file, {"base": object(), "arbitrum": object()}, "private-key")
    selected = deploy_command.resubmit_guard_migration(
        deployment_file,
        {"base": object(), "arbitrum": object()},
        "private-key",
        chain="base",
    )
    assert set(selected) == {"base"}
    assert len(calls) == 2


def test_retry_rejects_records_without_a_usable_guard_proposal(tmp_path: Path) -> None:
    """Give operators a useful error for a fresh vault or incomplete submission record.

    1. Save a fresh-vault record with no guard migration and reject retry.
    2. Save an incomplete submitted proposal and reject its misleading status.
    """
    deployment_file = tmp_path / "vault.json"

    # 1. Save a fresh-vault record with no guard migration and reject retry.
    deployment_file.write_text(json.dumps({"deployment_mode": "fresh deployment"}))
    with pytest.raises(ValueError, match="No guard migration"):
        deploy_command.resubmit_guard_migration(deployment_file, {"base": object()}, "key")

    # 2. Save an incomplete submitted proposal and reject its misleading status.
    deployment_file.write_text(json.dumps({"Guard migration": {"safe_proposal": {"status": "submitted"}}}))
    with pytest.raises(RuntimeError, match="missing its hash or URL"):
        deploy_command.resubmit_guard_migration(deployment_file, {"base": object()}, "key")


def test_registered_deploy_command_retries_without_deploying(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Route recovery through lagoon-deploy-vault without starting a new deployment.

    1. Arrange a saved record, one chain connection and mocked recovery boundary.
    2. Invoke the registered deployment command with the retry option.
    3. Check the record and chain reach recovery without deployment setup.
    """
    # 1. Arrange a saved record, one chain connection and mocked recovery boundary.
    record_file = tmp_path / "vault.json"
    record_file.write_text("{}")
    captured = {}
    web3config = SimpleNamespace(
        connections={ChainId.base: object()},
        has_any_connection=lambda: True,
        close=lambda: captured.setdefault("closed", True),
    )

    # Mock RPC setup and recovery because this test checks CLI routing; the
    # batch and saved-record behaviour are exercised in the other tests.
    monkeypatch.setattr(deploy_command, "setup_logging", lambda _level: logging.getLogger("guard-retry-test"))
    monkeypatch.setattr(deploy_command, "create_web3_config", lambda **_kwargs: web3config)
    monkeypatch.setattr(deploy_command, "prepare_cache", lambda *_args, **_kwargs: pytest.fail("retry must not prepare deployment cache"))
    monkeypatch.setattr(
        deploy_command,
        "resubmit_guard_migration",
        lambda deployment_file, chain_web3, private_key, **kwargs: captured.update(
            record=deployment_file,
            chains=chain_web3,
            key=private_key,
            options=kwargs,
        ) or {"base": {"safe_tx_hash": "0x" + "ab" * 32, "url": "https://safe.invalid/tx"}},
    )

    monkeypatch.setenv("SIMULATE", "false")
    monkeypatch.setenv("GENERATE_LIGHTER_API_KEY", "true")

    # 2. Invoke the registered deployment command with the retry option.
    get_command(app).main(
        args=[
            "lagoon-deploy-vault",
            "--retry-guard-proposal",
            "--vault-record-file", str(record_file),
            "--chain-name", "base",
            "--private-key", "0x123",
            "--json-rpc-base", "http://unused",
            "--log-level", "disabled",
        ],
        standalone_mode=False,
    )

    # 3. Check the record and chain reach recovery without deployment setup.
    assert captured["record"] == record_file
    assert set(captured["chains"]) == {"base"}
    assert captured["key"] == "0x123"
    assert captured["options"]["chain"] == "base"
    assert captured["closed"] is True


def test_multichain_guard_deployment_keeps_first_proposal_after_second_chain_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep the first chain's hand-off record when a later guard deployment fails.

    1. Arrange two guard-only chains with a deployment failure on the second.
    2. Run sequential deployment with a mocked Transaction Service boundary.
    3. Verify the first chain's submitted proposal survives in the checkpoint.
    """
    saved_records = []
    safe_address = _address(1)
    first_deployment = SimpleNamespace(
        safe_address=safe_address,
        trading_strategy_module=SimpleNamespace(address=_address(2)),
    )

    def fake_deploy(*, chain_web3, **_kwargs):
        slug = next(iter(chain_web3))
        if slug == "arbitrum":
            raise RuntimeError("deployment failed")
        return SimpleNamespace(deployments={slug: first_deployment})

    def fake_build(_result, _nonce, _configs, _report, *, safe_proposals):
        return "Checkpoint", {"proposals": deepcopy(safe_proposals)}

    # Mock chain deployment and the external Service because this test checks
    # checkpoint ordering, not contract deployment or network behaviour.
    monkeypatch.setattr(deploy_command, "deploy_multichain_lagoon_vault", fake_deploy)
    monkeypatch.setattr(deploy_command, "_build_multichain_artifact_payload", fake_build)
    monkeypatch.setattr(deploy_command, "_write_deployment_artifacts", lambda _path, **kwargs: saved_records.append(deepcopy(kwargs["public_json_payload"])))
    monkeypatch.setattr(deploy_command, "describe_pending_guard_migration", lambda *_args: {"status": "pending", "nonce": 4})
    monkeypatch.setattr(deploy_command, "submit_guard_migration", lambda *_args, **_kwargs: {"status": "submitted", "nonce": 4, "safe_tx_hash": "0x" + "ab" * 32, "url": "https://safe.invalid/tx"})

    # 1. Arrange two guard-only chains with a deployment failure on the second.
    # 2. Run sequential deployment with a mocked Transaction Service boundary.
    with pytest.raises(RuntimeError, match="Guard deployment failed on arbitrum"):
        deploy_command._deploy_and_propose_guard_only_chains(
            chain_web3={"base": object(), "arbitrum": object()},
            configs={"base": object(), "arbitrum": object()},
            deployer=object(),
            proposal_contexts={"base": object(), "arbitrum": object()},
            private_key="key",
            safe_api_key=None,
            vault_record_file=tmp_path / "deployment.json",
            safe_salt_nonce=None,
            logger=SimpleNamespace(info=lambda *_args: None),
        )

    # 3. Verify the first chain's submitted proposal survives in the checkpoint.
    assert [record["proposals"]["base"]["status"] for record in saved_records] == ["pending", "submitted"]
    assert saved_records[-1]["proposals"]["base"]["safe_tx_hash"] == "0x" + "ab" * 32
    assert saved_records[-1]["partial"] is True


def test_failed_submission_can_retry_from_saved_guard_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Retry the deployed guard after the Transaction Service initially fails.

    1. Arrange one deployed guard and a failing Service submission.
    2. Run guard-only deployment and inspect the pending record left by failure.
    3. Retry from that record without invoking the deployer again.
    4. Verify the accepted proposal is saved to the same record.
    """
    safe_address, old_guard, new_guard = (_address(i) for i in range(1, 4))
    deployment = SimpleNamespace(safe_address=safe_address, trading_strategy_module=SimpleNamespace(address=new_guard))
    deploy_calls = []

    def fake_deploy(**_kwargs):
        deploy_calls.append(True)
        return SimpleNamespace(deployments={"base": deployment})

    def fake_build(_result, _nonce, _configs, _report, *, safe_proposals):
        return "Guard checkpoint", {"multichain": True, "deployments": {"base": {"guard_migration": {
            "safe_address": safe_address,
            "old_guard_address": old_guard,
            "new_guard_address": new_guard,
            "safe_proposal": deepcopy(safe_proposals["base"]),
        }}}}

    def failing_submit(*_args, **_kwargs):
        raise RuntimeError("Service unavailable")

    # Mock contract deployment and the external Service because this test
    # verifies durable hand-off and retry after an API failure.
    monkeypatch.setattr(deploy_command, "deploy_multichain_lagoon_vault", fake_deploy)
    monkeypatch.setattr(deploy_command, "_build_multichain_artifact_payload", fake_build)
    monkeypatch.setattr(deploy_command, "describe_pending_guard_migration", lambda *_args: {"status": "pending", "nonce": 5, "data": "0x1234"})
    monkeypatch.setattr(deploy_command, "submit_guard_migration", failing_submit)
    record_file = tmp_path / "vault.json"
    logger = SimpleNamespace(info=lambda *_args: None)

    # 1. Arrange one deployed guard and a failing Service submission.
    # 2. Run guard-only deployment and inspect the pending record left by failure.
    with pytest.raises(RuntimeError, match="Service unavailable"):
        deploy_command._deploy_and_propose_guard_only_chains(
            chain_web3={"base": object()},
            configs={"base": object()},
            deployer=object(),
            proposal_contexts={"base": object()},
            private_key="key",
            safe_api_key=None,
            vault_record_file=record_file,
            safe_salt_nonce=None,
            logger=logger,
        )
    pending = json.loads(record_file.read_text())["deployments"]["base"]["guard_migration"]["safe_proposal"]
    assert pending == {"status": "pending", "nonce": 5, "data": "0x1234"}

    # 3. Retry from that record without invoking the deployer again.
    monkeypatch.setattr(deploy_command, "prepare_guard_proposal", lambda *_args, **_kwargs: "context")
    monkeypatch.setattr(deploy_command, "submit_guard_migration", lambda *_args, **_kwargs: {"status": "submitted", "nonce": 5, "safe_tx_hash": "0x" + "ab" * 32, "url": "https://safe.invalid/tx"})
    deploy_command.resubmit_guard_migration(record_file, {"base": object()}, "key")

    # 4. Verify the accepted proposal is saved to the same record.
    saved = json.loads(record_file.read_text())
    assert len(deploy_calls) == 1
    assert saved["deployments"]["base"]["guard_migration"]["safe_proposal"]["status"] == "submitted"
