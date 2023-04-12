"""Eznyme end-to-end test."""

import os
import secrets

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import Result
from eth_account import Account
from typer.main import get_command

from eth_defi.anvil import AnvilLaunch
from hexbytes import HexBytes
from typer.testing import CliRunner
from web3.contract import Contract

from eth_defi.enzyme.vault import Vault
from eth_defi.hotwallet import HotWallet
from eth_defi.trace import assert_transaction_success_with_explanation
from eth_defi.uniswap_v2.deployment import UniswapV2Deployment
from eth_typing import HexAddress

from tradeexecutor.cli.main import app
from tradeexecutor.state.blockhain_transaction import BlockchainTransactionType
from tradeexecutor.state.trade import TradeType
from tradingstrategy.pair import PandasPairUniverse

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.state import State


@pytest.fixture
def hot_wallet(web3, deployer, user_1, usdc: Contract, vault: Vault) -> HotWallet:
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

    # Promote the hot wallet to the asset manager
    tx_hash = vault.vault.functions.addAssetManagers([account.address]).transact({"from": user_1})
    assert_transaction_success_with_explanation(web3, tx_hash)

    return wallet


@pytest.fixture()
def strategy_file() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.dirname(__file__)) / "../../strategies/test_only" / "enzyme_end_to_end.py"


@pytest.fixture()
def state_file() -> Path:
    """The state file for the tests.

    Always start with an empty file.
    """
    path = Path("/tmp/test_enzyme_end_to_end.json")
    if path.exists():
        os.remove(path)
    return path


@pytest.fixture()
def environment(
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
    strategy_file: Path,
    ) -> dict:
    """Passed to init and start commands as environment variables"""
    # Set up the configuration for the live trader
    environment = {
        "EXECUTOR_ID": "test_enzyme_live_trading_init",
        "NAME": "test_enzyme_live_trading_init",
        "STRATEGY_FILE": strategy_file.as_posix(),
        "PRIVATE_KEY": hot_wallet.account.key.hex(),
        "JSON_RPC_ANVIL": anvil.json_rpc_url,
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "enzyme",
        "UNIT_TESTING": "true",
        # "LOG_LEVEL": "info",  # Set to info to get debug data for the test run
        "LOG_LEVEL": "disabled",
        "VAULT_ADDRESS": vault.address,
        "VAULT_ADAPTER_ADDRESS": vault.generic_adapter.address,
        "TEST_EVM_UNISWAP_V2_ROUTER": uniswap_v2.router.address,
        "TEST_EVM_UNISWAP_V2_FACTORY": uniswap_v2.factory.address,
        "TEST_EVM_UNISWAP_V2_INIT_CODE_HASH": uniswap_v2.init_code_hash,
        "CONFIRMATION_BLOCK_COUNT": "0",  # Needed for test backend, Anvil
        "MAX_CYCLES": "5",  # Run decide_trades() 5 times
    }
    return environment


def run_init(environment: dict) -> Result:
    """Run vault init command"""

    # https://typer.tiangolo.com/tutorial/testing/
    runner = CliRunner()

    # Need to use patch here, or parent shell env vars will leak in and cause random test failres
    with patch.dict(os.environ, environment, clear=True):
        result = runner.invoke(app, "init", env=environment)

    if result.exception:
        raise result.exception

    return result


def test_enzyme_live_trading_init(
    environment: dict,
    state_file: Path,
):
    """Initialize Enzyme vault for live trading.

    Provide faux chain using Anvil with one pool that a sample strategy is trading.
    """

    result = run_init(environment)
    assert result.exit_code == 0

    # Check the initial state sync set some of the variables
    with state_file.open("rt") as inp:
        state = State.from_json(inp.read())
        assert state.sync.deployment.vault_token_name is not None
        assert state.sync.deployment.vault_token_symbol is not None
        assert state.sync.deployment.block_number > 1


def test_enzyme_live_trading_start(
    environment: dict,
    state_file: Path,
    usdc: Contract,
    weth: Contract,
    vault: Vault,
    deployer: HexAddress,
):
    """Run Enzyme vaulted strategy for few cycles.

    - Set up local Anvil testnet with Uniswap v2 and Enzyme

    - Create a strategy that trade ETH-USDC pair and does few buys and sells

    - Run cycles of this strategy

    - Check that the state file output looks good

    - Check that the chain output looks good

    At the end of 5th cycle we should have

    - 1 open position, id 2

    - 1 closed position, id 1
    """


    # Need to be initialised first
    result = run_init(environment)
    assert result.exit_code == 0

    # Deposit some money in the vault
    usdc.functions.approve(vault.comptroller.address, 500 * 10**6).transact({"from": deployer})
    vault.comptroller.functions.buyShares(500 * 10**6, 1).transact({"from": deployer})

    # Run strategy for few cycles.
    # Manually call the main() function so that Typer's CliRunner.invoke() does not steal
    # stdin and we can still set breakpoints
    cli = get_command(app)
    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["start"])

        assert e.value.code == 0

    # Check that trades completed
    with state_file.open("rt") as inp:
        state = State.from_json(inp.read())

        # Show tx revert reason if possible
        if len(state.portfolio.frozen_positions) > 0:
            for p in state.portfolio.frozen_positions.values():
                raise AssertionError(f"Frozen position {p}: {p.get_freeze_reason()}")

        assert len(state.portfolio.closed_positions) == 1
        assert len(state.portfolio.open_positions) == 1

        # Pick an example trade to examine
        p = state.portfolio.open_positions[2]
        t = p.trades[3]
        assert t.is_success()
        assert t.lp_fees_estimated == pytest.approx(0.14991015720000014)
        assert t.lp_fees_paid in (0, None)  # Not supported yet
        assert t.trade_type == TradeType.rebalance
        assert t.slippage_tolerance == 0.02  # Set in enzyme_end_to_end.py strategy module

        tx = t.blockchain_transactions[0]
        assert tx.type == BlockchainTransactionType.enzyme_vault

    # Check on-chain balances
    usdc_balance = usdc.functions.balanceOf(vault.vault.address).call()
    weth_balance = weth.functions.balanceOf(vault.vault.address).call()

    assert usdc_balance == pytest.approx(10**6 * 449.730472)
    assert weth_balance == pytest.approx(10**18 * 0.03112978758721282)
