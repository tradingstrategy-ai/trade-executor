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


pytestmark = pytest.mark.skipif(not os.environ.get("JSON_RPC_POLYGON") or not os.environ.get("TRADING_STRATEGY_API_KEY"), reason="Set JSON_RPC_POLYGON and TRADING_STRATEGY_API_KEY environment variables to run this test")


@pytest.fixture()
def anvil(request: FixtureRequest) -> AnvilLaunch:
    """Do Ethereum mainnet fork from the damaged situation."""

    mainnet_rpc = os.environ["JSON_RPC_POLYGON"]

    anvil = launch_anvil(
        mainnet_rpc,
        fork_block_number=60719175,  # The timestamp on when the broken position was created
    )

    try:
        yield anvil
    finally:
        anvil.close()


@pytest.fixture()
def state_file() -> Path:
    """Make a copy of the state file with the broken credit position on a new test cycle"""
    template = Path(__file__).resolve().parent / "state" / "frozen-credit-position.json"
    p = Path("/tmp/frozen-credit-position.json")
    shutil.copy(template, p)
    assert p.exists(), f"{p} missing"
    return p


@pytest.fixture()
def strategy_file() -> Path:
    """The strategy module where the broken accounting happened."""
    p = Path(__file__).resolve().parent / ".." / ".." / "strategies" /  "test_only" / "enzyme-polygon-eth-rolling-ratio.py"
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
        "VAULT_ADDRESS": "0x53b23bD0Ce01bAd74Ae31402608529451193738E",
        "VAULT_ADAPTER_ADDRESS": "0xB603CF9BfeC74Abffa139987AeB7BEaeF6FEC39C",
        "VAULT_PAYMENT_FORWARDER_ADDRESS": "0xc80D7FFeF34CbC95a49D106BbF5065e3Dff60907",
        "VAULT_DEPLOYMENT_BLOCK_NUMBER": "60338719",
        "SKIP_SAVE": "true",
        "AUTO_APPROVE": "true",  # skip y/n prompt
    }
    return environment


def test_repair_frozen_credit_position(
    environment: dict,
):
    """Fix frozen credit positions."""

    with mock.patch.dict("os.environ", environment, clear=True):
        app(["repair"], standalone_mode=False)

        # Check accounts now to verify if balance is good
        with pytest.raises(SystemExit) as sys_exit:
            app(["check-accounts"], standalone_mode=False)
        assert sys_exit.value.code == 0
