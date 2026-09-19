"""Typer black-box coverage for Lagoon manual Safe settlement proposals."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from hexbytes import HexBytes
from typer.testing import CliRunner

import tradeexecutor.cli.commands.lagoon_manual_settle as manual_settle


def test_lagoon_manual_settle_proposes_preflighted_safe_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Propose the inspected direct settlement through the Typer command.

    1. Mock a configured Lagoon vault with a successful direct-call preflight.
    2. Invoke ``lagoon-manual-settle --propose-safe-transaction`` through Typer.
    3. Verify the Safe proposal uses the exact preflight target, calldata, Call operation and zero value.
    """
    captured: dict[str, object] = {}
    private_key = "0x" + "11" * 32
    proposal = SimpleNamespace(safe_tx_hash=HexBytes("0x" + "ab" * 32))
    vault = SimpleNamespace(safe="safe")
    report = {
        "settlement_required": True,
        "target_call_simulation": {"succeeds": True, "estimated_gas": 123_456},
        "gnosis_safe_transaction_fields": {
            "to": "0x0000000000000000000000000000000000000001",
            "value": "0",
            "operation": 0,
            "data": "0x1234",
            "contractMethod": {"name": "settleDeposit"},
            "contractInputsValues": {"_newTotalAssets": "42"},
        },
        "block_number": 123,
        "chain_id": 1,
        "vault": "0x0000000000000000000000000000000000000001",
        "safe": "0x0000000000000000000000000000000000000002",
        "denomination_token": {"symbol": "USDC"},
        "pending_deposit": "1",
        "pending_redemption_shares": "0",
        "safe_balance": "1",
        "settlement_abi": {"name": "settleDeposit"},
    }
    state = SimpleNamespace(
        sync=SimpleNamespace(
            deployment=SimpleNamespace(
                address="0x0000000000000000000000000000000000000001",
                block_number=1,
            ),
        ),
    )
    web3config = SimpleNamespace(
        has_any_connection=lambda: True,
        get_default=lambda: object(),
        close=lambda: captured.setdefault("web3config_closed", True),
    )
    store = SimpleNamespace(is_pristine=lambda: False, load=lambda: state)

    def fake_propose_safe_transaction(**kwargs):
        """Match the Safe proposal helper boundary with a signed SafeTx result."""
        captured["proposal"] = kwargs
        return proposal

    # The remaining command dependencies are mocked because this test verifies
    # Typer argument handling and proposal payload wiring without reading state,
    # querying RPC, or creating an external pending Safe transaction.
    monkeypatch.setattr(
        manual_settle, "prepare_executor_id", lambda _id, _strategy_file: "lighter-ai"
    )
    monkeypatch.setattr(manual_settle, "setup_logging", lambda **_kwargs: None)
    monkeypatch.setattr(
        manual_settle, "read_strategy_module", lambda _strategy_file: object()
    )
    monkeypatch.setattr(
        manual_settle, "create_web3_config", lambda **_kwargs: web3config
    )
    monkeypatch.setattr(manual_settle, "configure_default_chain", lambda *_args: None)
    monkeypatch.setattr(
        manual_settle,
        "resolve_state_store",
        lambda _id, _state_file: (Path("state.json"), store),
    )
    monkeypatch.setattr(manual_settle, "load_lagoon_vault", lambda *_args: vault)
    monkeypatch.setattr(
        manual_settle, "inspect_manual_lagoon_settlement", lambda *_args: report
    )
    monkeypatch.setattr(
        manual_settle, "propose_safe_transaction", fake_propose_safe_transaction
    )

    # 1. Mock a configured Lagoon vault with a successful direct-call preflight.
    runner = CliRunner()
    test_app = typer.Typer()
    test_app.command()(manual_settle.lagoon_manual_settle)

    # 2. Invoke lagoon-manual-settle --propose-safe-transaction through Typer.
    result = runner.invoke(
        test_app,
        [
            "--strategy-file",
            "strategies/lighter-ai.py",
            "--json-rpc-ethereum",
            "https://ethereum-rpc.invalid",
            "--private-key",
            private_key,
            "--propose-safe-transaction",
        ],
    )

    # 3. Verify the Safe proposal uses the exact preflight target, calldata, Call operation and zero value.
    assert result.exit_code == 0, result.output
    assert "Safe Transaction Service proposal created: 0x" + "ab" * 32 in result.output
    assert (
        "Safe transaction: https://app.safe.global/transactions/tx?"
        "safe=eth:0x0000000000000000000000000000000000000002&"
        "id=multisig_0x0000000000000000000000000000000000000002_0x"
        + "ab" * 32
    ) in result.output
    assert captured["proposal"] == {
        "safe": "safe",
        "address": "0x0000000000000000000000000000000000000001",
        "private_key": private_key,
        "data": HexBytes("0x1234"),
        "operation": 0,
        "value": 0,
    }
    assert captured["web3config_closed"] is True
