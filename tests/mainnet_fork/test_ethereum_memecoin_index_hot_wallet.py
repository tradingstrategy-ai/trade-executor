"""Test Memecoin index strategy on the mainnet fork.

"""

import os
from pathlib import Path

import pytest
from eth_account import Account
from eth_typing import HexAddress
from typer.main import get_command
from web3 import Web3

from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import AnvilLaunch, launch_anvil
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.token import TokenDetails, fetch_erc20_details
from eth_defi.trace import assert_transaction_success_with_explanation
from tradeexecutor.cli.commands.app import app
from tradeexecutor.state.state import State

JSON_RPC_ETHEREUM = os.environ.get("JSON_RPC_ETHEREUM")
TRADING_STRATEGY_API_KEY = os.environ.get("TRADING_STRATEGY_API_KEY")

pytestmark = pytest.mark.skipif(
    (not JSON_RPC_ETHEREUM or not TRADING_STRATEGY_API_KEY),
     reason="Set JSON_RPC_ETHEREUM and TRADING_STRATEGY_API_KEY needed to run this test"
)


@pytest.fixture()
def anvil(usdc_whale) -> AnvilLaunch:
    """Launch mainnet fork."""

    anvil = launch_anvil(
        fork_url=JSON_RPC_ETHEREUM,
        unlocked_addresses=[usdc_whale],
        fork_block_number=21_271_568,  # Keep test stable
    )
    try:
        yield anvil
    finally:
        anvil.close()


@pytest.fixture()
def web3(anvil) -> Web3:
    web3 = create_multi_provider_web3(anvil.json_rpc_url)
    assert web3.eth.chain_id == 1
    return web3


@pytest.fixture()
def usdc_whale() -> HexAddress:
    """A random account picked, holds a lot of stablecoin"""
    # https://etherscan.io/token/0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48#balances
    return HexAddress("0x37305B1cD40574E4C5Ce33f8e8306Be057fD7341")


@pytest.fixture
def usdc(web3) -> TokenDetails:
    details = fetch_erc20_details(web3, "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48")
    return details


@pytest.fixture
def weth(web3) -> TokenDetails:
    details = fetch_erc20_details(web3, "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2")
    return details


@pytest.fixture
def user_1(web3) -> Account:
    return web3.eth.accounts[3]


@pytest.fixture
def hot_wallet(
    web3,
    user_1,
    usdc: TokenDetails,
    usdc_whale,
) -> HotWallet:
    """Create hot wallet with a private key as we need to pass this key to forge, others commands.

    - Top up with 500 USDC
    """
    wallet = HotWallet.create_for_testing(web3)
    tx_hash = usdc.contract.functions.transfer(wallet.address, 500 * 10**6).transact({"from": usdc_whale})
    assert_transaction_success_with_explanation(web3, tx_hash)
    return wallet


@pytest.fixture()
def strategy_file() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.dirname(__file__)) / ".." / ".." / "strategies" / "test_only" / "memecoin_index.py"


@pytest.fixture()
def state_file(tmp_path) -> Path:
    path = tmp_path / "memecoin_index.json"
    return path


@pytest.fixture()
def environment(
    strategy_file,
    hot_wallet,
    anvil,
    state_file,
):
    environment = {
        "EXECUTOR_ID": "test_memecoin_index",
        "NAME": "test_memecoin_index",
        "STRATEGY_FILE": strategy_file.as_posix(),
        "PRIVATE_KEY": hot_wallet.account.key.hex(),
        "JSON_RPC_ETHEREUM": anvil.json_rpc_url,
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "hot_wallet",
        "UNIT_TESTING": "true",
        # "LOG_LEVEL": "info",  # Set to info to get debug data for the test run
        "LOG_LEVEL": "disabled",
        "RUN_SINGLE_CYCLE": "true",
        "TRADING_STRATEGY_API_KEY": TRADING_STRATEGY_API_KEY,
        "PATH": os.environ["PATH"], # Needs Forge bin
        "MAX_DATA_DELAY_MINUTES": str(10 * 60 * 24 * 365),  # 10 years or "disabled""
    }
    return environment


@pytest.mark.slow_test_group
def test_ethereum_mainnet_fork_memecoin_index(
    environment: dict,
    user_1,
    mocker,
    state_file,
    usdc,
    hot_wallet,
    web3,
):
    """Run a single cycle of Memecoin index strategy to see everything works.

    - Run against known block height of Ethereum mainnet

    - Ensure the strategy decide_trades loop does not crash

    - The test is not exactly stable, as blockchain is in a historical state,
      but loaded market data is the latest. There is no way to simulate
      "load live market data but in the point of history" because it would defeat
      the purpose of loading live data. When the test starts keep falling,
      just fix and bump the block height forward.
    """

    # Anvil is unbelieveable piece of crap and somehow fails the tx with nonce 0 that is approve.
    # We just manually approve infinity at the start of the test.
    # tx = usdc.contract.functions.approve("0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D", 2**32 - 1).build_transaction({
    #     "from": hot_wallet.address,
    # })
    # hot_wallet.fill_in_gas_price(web3, tx)
    # signed = hot_wallet.sign_transaction_with_new_nonce(tx)
    # tx_hash = web3.eth.send_raw_transaction(signed.rawTransaction)
    # assert_transaction_success_with_explanation(web3, tx_hash)

    cli = get_command(app)
    mocker.patch.dict("os.environ", environment, clear=True)
    cli.main(args=["init"], standalone_mode=False)
    cli.main(args=["start"], standalone_mode=False)

    state = State.read_json_file(state_file)
    assert len(state.visualisation.get_messages_tail(5)) == 1
    assert len(state.portfolio.frozen_positions) == 0


