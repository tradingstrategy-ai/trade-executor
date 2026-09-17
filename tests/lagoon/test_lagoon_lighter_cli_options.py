"""Typer black-box coverage for Lagoon Lighter CLI options and safety gates."""

import logging
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from eth_defi.erc_4626.vault_protocol.lagoon.deployment import LIGHTER_INITIAL_LAGOON_DEPOSIT
from pytest import LogCaptureFixture, MonkeyPatch
from typer.main import get_command

from tradeexecutor.cli.commands.lagoon_deploy_vault import _validate_lighter_initial_capital
from tradeexecutor.cli.main import app
from tradeexecutor.ethereum.lagoon.preflight_report import log_deployment_preflight_report

#: Non-default slot proving the Typer option reaches the deployment helper.
CUSTOM_LIGHTER_API_KEY_INDEX = 7


def test_lighter_initial_capital_preflight_rejects_insufficient_usdc() -> None:
    """Reject a Lighter deployment before confirmation when USDC is insufficient.

    1. Construct a token balance below the full initial Lagoon subscription.
    2. Run the pre-flight capital validation.
    3. Verify the error states the complete required subscription.
    """
    # 1. Construct a token balance below the full initial Lagoon subscription.
    token = SimpleNamespace(
        symbol="USDC",
        fetch_balance_of=lambda _address: LIGHTER_INITIAL_LAGOON_DEPOSIT - Decimal("0.01"),
    )
    hot_wallet = SimpleNamespace(
        address="0x0000000000000000000000000000000000000001",
        get_native_currency_balance=lambda _web3: Decimal(1),
    )
    web3 = SimpleNamespace()

    # 2. Run the pre-flight capital validation.
    # 3. Verify the error states the complete required subscription.
    with pytest.raises(ValueError, match="at least 20 USDC"):
        _validate_lighter_initial_capital(token, hot_wallet, web3)


def test_lighter_initial_capital_preflight_rejects_missing_native_gas() -> None:
    """Reject a Lighter deployment before confirmation without native gas.

    1. Construct the complete initial USDC subscription balance.
    2. Set the deployer's native Ethereum balance to zero.
    3. Verify pre-flight rejects the missing gas funding.
    """
    # 1. Construct the complete initial USDC subscription balance.
    token = SimpleNamespace(
        symbol="USDC",
        fetch_balance_of=lambda _address: LIGHTER_INITIAL_LAGOON_DEPOSIT,
    )
    # 2. Set the deployer's native Ethereum balance to zero.
    hot_wallet = SimpleNamespace(
        address="0x0000000000000000000000000000000000000001",
        get_native_currency_balance=lambda _web3: Decimal(0),
    )
    web3 = SimpleNamespace()

    # 3. Verify pre-flight rejects the missing gas funding.
    with pytest.raises(ValueError, match="native gas tokens"):
        _validate_lighter_initial_capital(token, hot_wallet, web3)


def test_lighter_preflight_report_shows_complete_initial_allocation(caplog: LogCaptureFixture) -> None:
    """Describe the full Lighter funding split before deployment confirmation.

    1. Construct minimal deployer and chain inputs for the report.
    2. Log a Lighter-enabled deployment pre-flight report.
    3. Verify the subscription, activation transfer and Safe reserve are shown.
    """
    # 1. Construct minimal deployer and chain inputs for the report.
    hot_wallet = SimpleNamespace(
        address="0x0000000000000000000000000000000000000001",
        current_nonce=1,
        sync_nonce=lambda _web3: None,
        get_native_currency_balance=lambda _web3: Decimal("1"),
    )
    web3 = SimpleNamespace(eth=SimpleNamespace(chain_id=1, block_number=1))

    # 2. Log a Lighter-enabled deployment pre-flight report.
    with caplog.at_level(logging.INFO):
        log_deployment_preflight_report(
            hot_wallet=hot_wallet,
            chain_web3={"ethereum": web3},
            fund_name="Lighter test",
            fund_symbol="LGT",
            asset_managers=[hot_wallet.address],
            multisig_owners=[hot_wallet.address],
            performance_fee=0,
            management_fee=0,
            lighter_api_key_generation=True,
            lighter_api_key_index=CUSTOM_LIGHTER_API_KEY_INDEX,
            lighter_deployment_address="0x0000000000000000000000000000000000000002",
            lighter_usdc_address="0x0000000000000000000000000000000000000003",
            lighter_activation_amount=Decimal(1),
            lighter_expected_safe_reserve=Decimal(19),
            lighter_deployer_usdc_balance=Decimal(20),
        )

    # 3. Verify the subscription, activation transfer and Safe reserve are shown.
    assert "Lighter initial Lagoon subscription: 20 USDC" in caplog.text
    assert "Lighter Safe-to-Lighter activation transfer: 1 USDC" in caplog.text
    assert "Lighter expected initial Safe reserve: 19 USDC" in caplog.text


def test_lighter_cli_options_reach_strategy_deployment(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Parse Lighter options through the registered Typer command.

    1. Replace network/bootstrap work with a small strategy-path stub.
    2. Invoke the real Typer command with environment options.
    3. Verify the selected key slot and generation flag are forwarded.
    """
    # 1. Replace network/bootstrap work with a small strategy-path stub.
    captured = {}

    class DummyWeb3Config:
        connections = {"ethereum": object()}

        def has_any_connection(self) -> bool:
            return True

        def close(self) -> None:
            pass

    def fake_deploy_multichain(**kwargs: Any) -> None:
        captured.update(kwargs)

    # Network bootstrap is mocked because this test exercises Typer option
    # parsing and forwarding, not deployment or provider behaviour.
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
    monkeypatch.setenv("PRIVATE_KEY", "0x123")
    monkeypatch.setenv("JSON_RPC_ETHEREUM", "http://unused")
    monkeypatch.setenv("STRATEGY_FILE", __file__)
    monkeypatch.setenv("VAULT_RECORD_FILE", str(tmp_path / "record.txt"))
    monkeypatch.setenv("GENERATE_LIGHTER_API_KEY", "true")
    monkeypatch.setenv(
        "LIGHTER_API_KEY_INDEX",
        str(CUSTOM_LIGHTER_API_KEY_INDEX),
    )
    monkeypatch.setenv("UNIT_TESTING", "true")
    monkeypatch.setenv("LOG_LEVEL", "disabled")

    # 2. Invoke the real Typer command with environment options.
    get_command(app).main(args=["lagoon-deploy-vault"], standalone_mode=False)

    # 3. Verify the selected key slot and generation flag are forwarded.
    assert captured["generate_lighter_api_key"] is True
    assert captured["lighter_api_key_index"] == CUSTOM_LIGHTER_API_KEY_INDEX
    assert captured["private_json_path"] == tmp_path / "record.json"


def test_lighter_cli_rejects_simulation_before_network_bootstrap(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
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

    # 2. Patch Web3 bootstrap to fail if it is reached.
    # This mock proves the safety validation happens before network setup.
    monkeypatch.setattr(
        "tradeexecutor.cli.commands.lagoon_deploy_vault.create_web3_config",
        lambda **kwargs: pytest.fail("Web3 bootstrap must not run for simulated Lighter activation"),
    )

    cli = get_command(app)

    # 3. Invoke the real Typer command and verify the early error.
    with pytest.raises(ValueError, match="cannot be used with --simulate"):
        cli.main(args=["lagoon-deploy-vault"], standalone_mode=False)
