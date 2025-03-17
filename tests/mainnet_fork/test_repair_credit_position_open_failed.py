"""Test repair frozen credit position on Polygon fork.
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
        fork_block_number=27664435,  # The timestamp on when the broken position was created
    )

    try:
        yield anvil
    finally:
        anvil.close()


@pytest.fixture()
def state_file(tmp_path) -> Path:
    """Make a copy of the state file with the broken credit position on a new test cycle"""
    template = Path(__file__).resolve().parent / "credit-position-open-failed.json"
    assert template.exists(), f"State dump missing: {template}"
    p = tmp_path / Path("credit-position-open-failed.json.json")
    shutil.copy(template, p)
    assert p.exists(), f"{p} missing"
    return p


@pytest.fixture()
def strategy_file() -> Path:
    """The strategy module where the broken accounting happened."""
    p = Path(__file__).resolve().parent / ".." / ".." / "strategies" /  "test_only" / "base-ath.py"
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
        "JSON_RPC_ANVIL": anvil.json_rpc_url,
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "lagoon",
        "UNIT_TESTING": "true",
        "UNIT_TEST_FORCE_ANVIL": "true",  # check-wallet command legacy hack
        "LOG_LEVEL": "disabled",
        # "LOG_LEVEL": "info",
        # "CONFIRMATION_BLOCK_COUNT": "0",  # Needed for test backend, Anvil
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "VAULT_ADDRESS": "0x7d8Fab3E65e6C81ea2a940c050A7c70195d1504f",
        "VAULT_ADAPTER_ADDRESS": "0x3275Af9ce73665A1Cd665E5Fa0b48c25249219ac",
        "SKIP_SAVE": "true",
        "AUTO_APPROVE": "true",  # skip y/n prompt
        "CACHE_PATH": str(persistent_test_client.transport.cache_path),  # Use unit test cache
        "RAISE_ON_UNCLEAN": "true",
    }
    return environment


@pytest.mark.slow_test_group
def test_repair_credit_position_open_failed(
    environment: dict,
    mocker,
):
    """Fix a creidt position that failed to open.

    - Execution crashes in broadcasting phase
    """

    mocker.patch.dict("os.environ", environment, clear=True)
    app(["repair"], standalone_mode=False)
    app(["correct-accounts"], standalone_mode=False)



    # Check accounts now to verify if balance is good
    # with pytest.raises(SystemExit) as sys_exit:

    # assert sys_exit.value.code == 0
