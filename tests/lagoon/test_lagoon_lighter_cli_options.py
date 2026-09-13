"""Typer black-box coverage for Lagoon Lighter CLI options and safety gates."""

import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from pytest import MonkeyPatch
from typer.main import get_command

from tradeexecutor.cli.main import app


def test_lighter_cli_options_reach_strategy_deployment(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
):
    """Parse Lighter options through the registered Typer command.

    1. Replace network/bootstrap work with a small strategy-path stub.
    2. Invoke the real Typer command with environment options.
    3. Verify the selected key slot and generation flag are forwarded.
    """
    # 1. Replace network/bootstrap work with a small strategy-path stub.
    captured = {}

    class DummyWeb3Config:
        connections = {"ethereum": object()}

        def has_any_connection(self):
            return True

        def close(self):
            pass

    def fake_deploy_multichain(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        "tradeexecutor.cli.commands.lagoon_deploy_vault.setup_logging",
        lambda _level: logging.getLogger("test-lighter-cli"),
    )
    monkeypatch.setattr(
        "tradeexecutor.cli.commands.lagoon_deploy_vault.prepare_cache",
        lambda *args, **kwargs: tmp_path,
    )
    monkeypatch.setattr(
        "tradeexecutor.cli.commands.lagoon_deploy_vault.prepare_token_cache",
        lambda *args, **kwargs: SimpleNamespace(filename=":memory:"),
    )
    monkeypatch.setattr(
        "tradeexecutor.cli.commands.lagoon_deploy_vault.create_web3_config",
        lambda **kwargs: DummyWeb3Config(),
    )
    monkeypatch.setattr(
        "tradeexecutor.cli.commands.lagoon_deploy_vault.create_hot_wallet",
        lambda web3, key: SimpleNamespace(address="0x0000000000000000000000000000000000000001"),
    )
    monkeypatch.setattr(
        "tradeexecutor.cli.commands.lagoon_deploy_vault._deploy_multichain",
        fake_deploy_multichain,
    )
    monkeypatch.setenv("PATH", os.environ["PATH"])
    monkeypatch.setenv("PRIVATE_KEY", "0x123")
    monkeypatch.setenv("JSON_RPC_ETHEREUM", "http://unused")
    monkeypatch.setenv("STRATEGY_FILE", __file__)
    monkeypatch.setenv("VAULT_RECORD_FILE", str(tmp_path / "record.txt"))
    monkeypatch.setenv("GENERATE_LIGHTER_API_KEY", "true")
    monkeypatch.setenv("LIGHTER_API_KEY_INDEX", "7")
    monkeypatch.setenv("UNIT_TESTING", "true")
    monkeypatch.setenv("LOG_LEVEL", "disabled")

    # 2. Invoke the real Typer command with environment options.
    get_command(app).main(args=["lagoon-deploy-vault"], standalone_mode=False)

    # 3. Verify the selected key slot and generation flag are forwarded.
    assert captured["generate_lighter_api_key"] is True
    assert captured["lighter_api_key_index"] == 7
    assert captured["private_json_path"] == tmp_path / "record.json"


def test_lighter_cli_rejects_simulation_before_network_bootstrap(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
):
    """Reject an impossible fork activation before creating Web3 config.

    1. Set both generation and simulation environment flags.
    2. Patch Web3 bootstrap to fail if it is reached.
    3. Invoke the real Typer command and verify the early error.
    """
    # 1. Set both generation and simulation environment flags.
    monkeypatch.setenv("PRIVATE_KEY", "0x123")
    monkeypatch.setenv("VAULT_RECORD_FILE", str(tmp_path / "record.txt"))
    monkeypatch.setenv("GENERATE_LIGHTER_API_KEY", "true")
    monkeypatch.setenv("SIMULATE", "true")
    monkeypatch.setattr(
        "tradeexecutor.cli.commands.lagoon_deploy_vault.create_web3_config",
        lambda **kwargs: pytest.fail("Web3 bootstrap must not run for simulated Lighter activation"),
    )

    # 2. Patch Web3 bootstrap to fail if it is reached.
    cli = get_command(app)

    # 3. Invoke the real Typer command and verify the early error.
    with pytest.raises(ValueError, match="cannot be used with --simulate"):
        cli.main(args=["lagoon-deploy-vault"], standalone_mode=False)
