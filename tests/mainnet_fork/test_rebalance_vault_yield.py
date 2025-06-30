"""Test rebalance vault yield and get funds unstuck from Aave bug.
"""
import shutil
import os.path
import secrets
from pathlib import Path
from unittest import mock

import pytest
from _pytest.fixtures import FixtureRequest

from eth_defi.provider.anvil import AnvilLaunch, launch_anvil

from tradeexecutor.cli.commands.app import app


pytestmark = pytest.mark.skipif(not os.environ.get("JSON_RPC_BASE") or not os.environ.get("TRADING_STRATEGY_API_KEY"), reason="Set JSON_RPC_POLYGON and TRADING_STRATEGY_API_KEY environment variables to run this test")


@pytest.fixture()
def anvil(request: FixtureRequest) -> AnvilLaunch:
    """Do Ethereum mainnet fork from the damaged situation."""

    mainnet_rpc = os.environ["JSON_RPC_BASE"]

    anvil = launch_anvil(
        mainnet_rpc,
        fork_block_number=32092657,  # The timestamp on when the broken position was created
    )

    try:
        yield anvil
    finally:
        anvil.close()


@pytest.fixture()
def state_file(tmp_path) -> Path:
    """Make a copy of the state file with the broken vault on a new test cycle"""
    template = Path(__file__).resolve().parent / "yield-manager-aave-bug.json"
    assert template.exists(), f"State dump missing: {template}"
    p = tmp_path / Path("vault.json")
    shutil.copy(template, p)
    assert p.exists(), f"{p} missing"
    return p


@pytest.fixture()
def strategy_file() -> Path:
    """The strategy module where the broken accounting happened."""
    p = Path(__file__).resolve().parent / ".." / ".." / "strategies" /  "test_only" / "base-ath-ipor-aave-bug.py"
    assert p.exists(), f"{p.resolve()} missing"
    return p


@pytest.fixture()
def environment(
    anvil: AnvilLaunch,
    state_file: Path,
    strategy_file: Path,
    persistent_test_client,
    ) -> dict:
    """Passed to init and start commands as environment variables"""
    # Set up the configuration for the live trader
    environment = {
        "STRATEGY_FILE": strategy_file.as_posix(),
        "PRIVATE_KEY": "0x" + secrets.token_bytes(32).hex(),
        "JSON_RPC_BASE": anvil.json_rpc_url,
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "lagoon",
        "UNIT_TESTING": "true",
        "UNIT_TEST_FORCE_ANVIL": "true",  # check-wallet command legacy hack
        "LOG_LEVEL": "disabled",
        # "LOG_LEVEL": "info",
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "VAULT_ADDRESS": "0x7d8Fab3E65e6C81ea2a940c050A7c70195d1504f",
        "VAULT_ADAPTER_ADDRESS": "0x6DCCA7f34EB8F1a519ae690E9A3101f705bB0393",
        "SKIP_SAVE": "true",
        "AUTO_APPROVE": "true",  # skip y/n prompt
        "CACHE_PATH": str(persistent_test_client.transport.cache_path),  # Use unit test cache
        "RAISE_ON_UNCLEAN": "true",
        "RUN_SINGLE_CYCLE": "true",  # Run only one cycle"
        "MIN_GAS_BALANCE": "0.0",   # Disable gas balance check
        "DISABLE_BROADCAST": "true",  # Disable wait_and_broadcast_multiple_nodes() broadcast as we do not have real private key
    }
    return environment


@pytest.mark.slow_test_group
def test_rebalance_vault_yield(
    environment: dict,
    mocker,
):
    """Run one cycle and generate a rebalance transaction for the vault yield.

    - Make sure money is withdrawn from Aave and deposited to the vaults.
    """

    mocker.patch.dict("os.environ", environment, clear=True)
    app(["start"], standalone_mode=False)
