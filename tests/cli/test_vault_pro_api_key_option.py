"""Tests for the vault dataset licence key command line option."""

import inspect
import os
from pathlib import Path

import pytest

from tradingstrategy.vault_data_client import VAULT_PRO_API_KEY_ENV_VAR

from tradingstrategy.client import Client

from tradeexecutor.cli.main import app


#: Commands that build a trading universe and therefore may load vault data.
#:
#: Kept as an explicit list rather than derived from the option itself, so that
#: a command silently losing the option fails this test instead of shrinking the
#: expectation with it.
VAULT_DATA_COMMANDS = {
    "backtest",
    "blacklist",
    "check_accounts",
    "check_position_triggers",
    "check_universe",
    "close_all",
    "close_position",
    "console",
    "correct_accounts",
    "distribute_gas_funds",
    "lagoon_settle",
    "perform_test_trade",
    "repair",
    "retry",
    "start",
    "trade_ui",
    "webapi",
}


def test_vault_data_commands_expose_the_licence_key_option() -> None:
    """Check every command that can load vault data accepts the licence key.

    Vault datasets are licence gated, so a command that builds a trading
    universe without a way to pass the key can only be configured through the
    environment, which is invisible in ``--help``.

    1. Collect the registered command names and their parameters.
    2. Verify each vault data command declares the option.
    3. Verify the option reads the documented environment variable.
    """

    # 1. Collect the registered command names and their parameters.
    commands = {command.callback.__name__: command.callback for command in app.registered_commands}

    for name in VAULT_DATA_COMMANDS:
        assert name in commands, f"Command {name} is no longer registered"

        # 2. Verify each vault data command declares the option.
        parameters = inspect.signature(commands[name]).parameters
        assert "vault_pro_api_key" in parameters, f"Command {name} does not accept --vault-pro-api-key"

        # 3. Verify the option reads the documented environment variable.
        assert parameters["vault_pro_api_key"].default.envvar == VAULT_PRO_API_KEY_ENV_VAR


def test_command_line_licence_key_reaches_the_vault_data_client(tmp_path: Path) -> None:
    """Check a key given on the command line is carried to the vault dataset client.

    Strategy modules construct no credentials of their own: they receive the
    oracle client and ask it for the vault client, so the licence key has to
    travel on that object rather than through the environment.

    1. Create an oracle client with a licence key, as the CLI does.
    2. Ask it for a vault dataset client, as universe loading does.
    3. Verify the key arrived without going through the environment.
    """

    # 1. Create an oracle client with a licence key, as the CLI does.
    client = Client(None, None, vault_pro_api_key="AAAAA-BBBBB-CCCCC-DDDDD-EEEEE")

    # 2. Ask it for a vault dataset client, as universe loading does.
    vault_data_client = client.get_vault_data_client(download_root=tmp_path)

    # 3. Verify the key arrived without going through the environment.
    assert vault_data_client.api_key == "AAAAA-BBBBB-CCCCC-DDDDD-EEEEE"
    assert vault_data_client.download_root == tmp_path
    assert os.environ.get(VAULT_PRO_API_KEY_ENV_VAR) != "AAAAA-BBBBB-CCCCC-DDDDD-EEEEE"


def test_client_without_a_licence_key_falls_back_to_the_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Check notebooks and tests can still configure the key through the environment.

    Not every caller goes through the command line, so a client created without
    an explicit key must still find one in the environment.

    1. Put a licence key in the environment only.
    2. Create an oracle client without one and ask it for a vault client.
    3. Verify the environment key was used.
    """

    # 1. Put a licence key in the environment only.
    monkeypatch.setenv(VAULT_PRO_API_KEY_ENV_VAR, "FROM-ENVIRONMENT")

    # 2. Create an oracle client without one and ask it for a vault client.
    client = Client(None, None)
    vault_data_client = client.get_vault_data_client(download_root=tmp_path)

    # 3. Verify the environment key was used.
    assert vault_data_client.api_key == "FROM-ENVIRONMENT"
