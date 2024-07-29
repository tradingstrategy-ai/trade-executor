"""Test correct accounting for aUSDC position.

- Does Ethereum mainnet fork

- Archive node needed
"""
import logging
import os.path
import secrets
from pathlib import Path
from unittest import mock

import pytest
from _pytest.fixtures import FixtureRequest

from eth_defi.provider.anvil import AnvilLaunch, launch_anvil

from tradeexecutor.cli.commands.app import app


pytestmark = pytest.mark.skipif(not os.environ.get("JSON_RPC_ETHEREUM") or not os.environ.get("TRADING_STRATEGY_API_KEY"), reason="Set JSON_RPC_ETHEREUM and TRADING_STRATEGY_API_KEY environment variables to run this test")


@pytest.fixture()
def anvil(request: FixtureRequest) -> AnvilLaunch:
    """Do Ethereum mainnet fork from the damaged situation."""

    mainnet_rpc = os.environ["JSON_RPC_ETHEREUM"]

    anvil = launch_anvil(
        mainnet_rpc,
        fork_block_number=20409979,  # The timestamp on when the broken position was created
    )

    try:
        yield anvil
    finally:
        #anvil.close(log_level=logging.INFO)
        anvil.close()


@pytest.fixture()
def state_file() -> Path:
    """A sample of a state file where we should have an open spot position for WBTC.

    - This position was not opened due to Alchemy JSON-RPC error
    """
    p = Path(os.path.join(os.path.dirname(__file__), "ausdc-broken-position.json"))
    assert p.exists(), f"{p} missing"
    return p


@pytest.fixture()
def strategy_file() -> Path:
    """The strategy module where the broken accounting happened."""
    p = Path(os.path.join(os.path.dirname(__file__), "..", "..", "strategies", "test_only", "enzyme-ethereum-btc-eth-stoch-rsi.py"))
    assert p.exists(), f"{p.resolve()} missing"
    return p


@pytest.fixture()
def environment(
    anvil: AnvilLaunch,
    state_file: Path,
    strategy_file: Path,
    ) -> dict:
    """Passed to init and start commands as environment variables"""
    # Set up the configuration for the live trader
    environment = {
        "STRATEGY_FILE": strategy_file.as_posix(),
        "PRIVATE_KEY": "0x" + secrets.token_bytes(32).hex(),
        "JSON_RPC_ANVIL": anvil.json_rpc_url,
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "enzyme",
        "UNIT_TESTING": "true",
        "UNIT_TEST_FORCE_ANVIL": "true",  # check-wallet command legacy hack
        "LOG_LEVEL": "disabled",
        # "LOG_LEVEL": "info",
        # "CONFIRMATION_BLOCK_COUNT": "0",  # Needed for test backend, Anvil
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "VAULT_ADDRESS": "0x773C9f40a7aeCcB307dFFFD237Fc55e649bf375a",
        "VAULT_ADAPTER_ADDRESS": "0xf2be1e782a8512BEC56515c8b879a2BF0dC030A2",
        "VAULT_PAYMENT_FORWARDER_ADDRESS": "0x7424DaceaC1F64c266B85f9C43A0e0851EdB3234",
        "VAULT_DEPLOYMENT_BLOCK_NUMBER": "20362579",
        "SKIP_SAVE": "true",
    }
    return environment


def test_correct_accouting_closed_position_has_ausdc(
    environment: dict,
):
    """Fix missing aUSDC position.

    - We have closed aUSDC position, but not open one.
    """

    # TODO: How to cache data download between separate commands
    # to speed up the test a bit

    # Accounting is detect to be incorrect
    with mock.patch.dict('os.environ', environment, clear=True):
        with pytest.raises(SystemExit) as sys_exit:
            app(["check-accounts"], standalone_mode=False)
        assert sys_exit.value.code == 1

    # Fix issued
    with mock.patch.dict('os.environ', environment, clear=True):
        with pytest.raises(SystemExit) as sys_exit:
            app(["correct-accounts"], standalone_mode=False)
        assert sys_exit.value.code == 0
