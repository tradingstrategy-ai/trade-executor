"""Test correct accounting for WBTC position.

- Does Ethereum mainnet fork

- Archive node needed
"""
import os.path
import secrets
from pathlib import Path
from unittest import mock

import pytest
from _pytest.fixtures import FixtureRequest

from eth_defi.chain import install_chain_middleware
from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import AnvilLaunch, launch_anvil
from web3 import Web3, HTTPProvider

from tradeexecutor.cli.commands.app import app
from tradeexecutor.state.state import State


pytestmark = pytest.mark.skipif(not os.environ.get("JSON_RPC_ETHEREUM") or not os.environ.get("TRADING_STRATEGY_API_KEY"), reason="Set JSON_RPC_ETHEREUM and TRADING_STRATEGY_API_KEY environment variables to run this test")


@pytest.fixture()
def anvil(request: FixtureRequest) -> AnvilLaunch:
    """Do Ethereum mainnet fork from the damaged situation."""

    mainnet_rpc = os.environ["JSON_RPC_ETHEREUM"]

    anvil = launch_anvil(
        mainnet_rpc,
        fork_block_number=20377193,  # The timestamp on when the broken position was created
    )

    try:
        yield anvil
    finally:
        anvil.close()


@pytest.fixture()
def web3(anvil: AnvilLaunch) -> Web3:
    web3 = Web3(HTTPProvider(anvil.json_rpc_url, request_kwargs={"timeout": 2}))
    web3.middleware_onion.clear()
    install_chain_middleware(web3)
    return web3


@pytest.fixture()
def state_file() -> Path:
    """A sample of a state file where we should have an open spot position for WBTC.

    - This position was not opened due to Alchemy JSON-RPC error
    """
    p = Path(os.path.join(os.path.dirname(__file__), "wbtc-broken-position.json"))
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
        "EXECUTOR_ID": "test_enzyme_guard_perform_test_trade",
        "STRATEGY_FILE": strategy_file.as_posix(),
        "PRIVATE_KEY": "0x" + secrets.token_bytes(32).hex(),
        "JSON_RPC_ANVIL": anvil.json_rpc_url,
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "enzyme",
        "UNIT_TESTING": "true",
        "UNIT_TEST_FORCE_ANVIL": "true",  # check-wallet command legacy hack
        "LOG_LEVEL": "disabled",
        # "CONFIRMATION_BLOCK_COUNT": "0",  # Needed for test backend, Anvil
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "VAULT_ADDRESS": "0x85091fF4fb529B8F8B3D63810ddB8ddC9913cD62",
        "VAULT_ADAPTER_ADDRESS": "0x0EFA7fcd31e07aa774d37013Ec3f3b658C89572d",
        "VAULT_PAYMENT_FORWARDER_ADDRESS": "0x06843Ac92D1902D8fA9013cD9923AA4A448F3E32",
        "VAULT_DEPLOYMENT_BLOCK_NUMBER": "59155136",
    }
    return environment


def test_correct_account_missing_open_spot_position(
    environment: dict,
):
    """Perform a test trade on Enzyme vault via CLI.

    - Use MATIC-ETH-USDC strategy

    - The vault is deployed via `enzyme-deploy-vault`

    - The deployer configures a guard for the vault

    - We do the checks and then perform a test trade on the vault for all trading pairs

    - See docs for the workflow https://tradingstrategy.ai/docs/deployment/vault-deployment.html
    """

    with mock.patch.dict('os.environ', environment, clear=True):
        app(["check-accounts"], standalone_mode=False)
