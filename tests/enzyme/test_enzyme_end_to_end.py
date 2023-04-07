"""Eznyme end-to-end test."""
import datetime
import os
import secrets
from decimal import Decimal
from pathlib import Path

import pytest
from click.testing import CliRunner, Result
from eth_account import Account
from eth_defi.anvil import AnvilLaunch
from hexbytes import HexBytes

from eth_defi.enzyme.integration_manager import IntegrationManagerActionId
from eth_defi.enzyme.vault import Vault
from eth_defi.hotwallet import HotWallet
from eth_defi.trace import assert_transaction_success_with_explanation
from eth_defi.uniswap_v2.deployment import UniswapV2Deployment
from eth_typing import HexAddress

from tradeexecutor.cli.commands import app
from tradeexecutor.state.blockhain_transaction import BlockchainTransactionType
from tradingstrategy.pair import PandasPairUniverse
from web3 import Web3
from web3.contract import Contract

from eth_defi.event_reader.reorganisation_monitor import create_reorganisation_monitor

from tradeexecutor.ethereum.enzyme.tx import EnzymeTransactionBuilder
from tradeexecutor.ethereum.enzyme.vault import EnzymeVaultSyncModel

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.testing.ethereumtrader_uniswap_v2 import UniswapV2TestTrader



@pytest.fixture
def hot_wallet(web3, deployer, user_1, usdc: Contract) -> HotWallet:
    """Create hot wallet for the signing tests.

    Top is up with some gas money and 500 USDC.
    """
    private_key = HexBytes(secrets.token_bytes(32))
    account = Account.from_key(private_key)
    wallet = HotWallet(account)
    wallet.sync_nonce(web3)
    tx_hash = web3.eth.send_transaction({"to": wallet.address, "from": user_1, "value": 15 * 10**18})
    assert_transaction_success_with_explanation(web3, tx_hash)
    tx_hash = usdc.functions.transfer(wallet.address, 500 * 10**6).transact({"from": deployer})
    assert_transaction_success_with_explanation(web3, tx_hash)

    return wallet


@pytest.fixture()
def strategy_path() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.dirname(__file__)) / "../../strategies/test_only" / "enzyme_end_to_end.py"


@pytest.fixture()
def state_file() -> Path:
    """Strategy state file for this test run."""
    return Path("/tmp/test_enzyme_end_to_end.json")


def run_init(environment: dict) -> Result:
    """Run vault init command"""

    # https://typer.tiangolo.com/tutorial/testing/
    runner = CliRunner()

    result = runner.invoke(app, "init", env=environment)

    if result.exception:
        raise result.exception

    return result


def test_enzyme_live_trading_init(
    anvil: AnvilLaunch,
    deployer: HexAddress,
    vault: Vault,
    usdc: Contract,
    weth: Contract,
    usdc_asset: AssetIdentifier,
    weth_asset: AssetIdentifier,
    user_1: HexAddress,
    uniswap_v2: UniswapV2Deployment,
    weth_usdc_trading_pair: TradingPairIdentifier,
    pair_universe: PandasPairUniverse,
    hot_wallet: HotWallet,
    state_file: Path,
):
    """Initialize Enzyme vault for live trading.

    Provide faux chain using Anvil with one pool that a sample strategy is trading.
    """
    # Set up the configuration for the live trader
    environment = {
        "EXECUTOR_ID": "test_enzyme_live_trading_init",
        "NAME": "test_enzyme_live_trading_init",
        "STRATEGY_FILE": strategy_path.as_posix(),
        "PRIVATE_KEY": hot_wallet.account.key.hex(),
        "JSON_RPC_ANVIL": anvil.json_rpc_url,
        "STATE_FILE": state_file
        "RESET_STATE": "true",
        "EXECUTION_TYPE": "uniswap_v2_hot_wallet",
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",
    }

    result = run_init(environment)
    assert result.exit_code == 0

