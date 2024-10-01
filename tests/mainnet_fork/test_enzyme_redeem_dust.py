"""Test Enzyme redemption where the redemption request has a closed position with dust."""
import json
import os
import secrets
from pathlib import Path
from unittest import mock

import pytest
from _pytest.fixtures import FixtureRequest

from eth_account import Account
from eth_typing import HexAddress
from hexbytes import HexBytes
from web3 import Web3, HTTPProvider

from eth_defi.provider.anvil import AnvilLaunch, launch_anvil
from eth_defi.chain import install_chain_middleware
from eth_defi.enzyme.vault import Vault
from eth_defi.hotwallet import HotWallet
from eth_defi.token import TokenDetails, fetch_erc20_details
from eth_defi.trace import assert_transaction_success_with_explanation

from tradeexecutor.cli.main import app
from tradeexecutor.monkeypatch.web3 import construct_sign_and_send_raw_middleware
from tradeexecutor.state.state import State


pytestmark = pytest.mark.skipif(not os.environ.get("JSON_RPC_POLYGON") or not os.environ.get("TRADING_STRATEGY_API_KEY"), reason="Set JSON_RPC_POLYGON and TRADING_STRATEGY_API_KEY environment variables to run this test")


@pytest.fixture()
def anvil(request: FixtureRequest) -> AnvilLaunch:
    """Do Ethereum mainnet fork from the damaged situation."""

    mainnet_rpc = os.environ["JSON_RPC_POLYGON"]

    anvil = launch_anvil(
        mainnet_rpc,
        fork_block_number=62513342,  # The timestamp on when the broken position was created
    )

    try:
        yield anvil
    finally:
        #anvil.close(log_level=logging.INFO)
        anvil.close()


@pytest.fixture()
def state_file() -> Path:
    p = Path(os.path.join(os.path.dirname(__file__), "redeem-dust.json"))
    assert p.exists(), f"{p} missing"
    return p


@pytest.fixture()
def strategy_file() -> Path:
    """The strategy module where the broken accounting happened."""
    p = Path(os.path.join(os.path.dirname(__file__), "..", "..", "strategies", "test_only", "enzyme-polygon-btc-eth-rsi.py"))
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
        "VAULT_ADDRESS": "0xbba6B781e0BAC1798e4E715ef9b1113Bf2387544",
        "VAULT_ADAPTER_ADDRESS": "0x519f26bE61889656e83262ab56D75f00DFDAAEc1",
        "VAULT_PAYMENT_FORWARDER_ADDRESS": "0x638241c16aB5002298B24ED2F0074B4662042258",
        "VAULT_DEPLOYMENT_BLOCK_NUMBER": "60554042",
        "SKIP_SAVE": "true",
    }
    return environment


def test_enzyme_redeem_dust(
    environment: dict,
    web3: Web3,
    state_file: Path,
    usdc: TokenDetails,
    hot_wallet: HotWallet,
    vault_record_file: Path,
    mocker,
):
    """Test Enzyme redemption where the redemption request has a closed position with dust.

    - The state file contains a closed aPolUSDC position

    - This position has small dust value left

    - Enzyme redemption request tries to redeem part of this dust (1 unit of aPolUSDC)

    - Make sure we can handle the dust redemption request on a closed position
    """

    mocker.patch.dict("os.environ", environment, clear=True)
    app(["check-accounts"], standalone_mode=False)
    app(["correct-accounts"], standalone_mode=False)





