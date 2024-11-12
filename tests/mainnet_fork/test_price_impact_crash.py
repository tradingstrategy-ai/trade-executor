"""Test that the strategy crashes if price impact tolerance is xeeced."""
import json
import os
import secrets
from pathlib import Path

import flaky
import pytest

from eth_account import Account
from eth_typing import HexAddress
from hexbytes import HexBytes
from web3 import Web3, HTTPProvider

from eth_defi.provider.anvil import AnvilLaunch, launch_anvil
from eth_defi.chain import install_chain_middleware
from eth_defi.hotwallet import HotWallet
from eth_defi.token import TokenDetails, fetch_erc20_details
from eth_defi.trace import assert_transaction_success_with_explanation

from tradeexecutor.cli.main import app
from tradeexecutor.monkeypatch.web3 import construct_sign_and_send_raw_middleware
from tradeexecutor.strategy.trade_pricing import PriceImpactToleranceExceeded


pytestmark = pytest.mark.skipif(
    not os.environ.get("JSON_RPC_POLYGON")
    or not os.environ.get("TRADING_STRATEGY_API_KEY"),
    reason="Set POLYGON_JSON_RPC and TRADING_STRATEGY_API_KEY environment variables to run this test",
)


@pytest.fixture()
def usdc_whale() -> HexAddress:
    """A random account picked from Polygon that holds a lot of USDC."""
    # https://polygonscan.com/token/0x2791bca1f2de4661ed88a30c99a7a9449aa84174#balances
    return HexAddress("0x72A53cDBBcc1b9efa39c834A540550e23463AAcB")


@pytest.fixture()
def anvil(usdc_whale) -> AnvilLaunch:
    """Launch Polygon fork."""
    rpc_url = os.environ["JSON_RPC_POLYGON"]

    anvil = launch_anvil(
        fork_url=rpc_url,
        unlocked_addresses=[usdc_whale],
        fork_block_number=58_000_000,
    )
    try:
        yield anvil
    finally:
        anvil.close()


@pytest.fixture()
def web3(anvil) -> Web3:
    """Set up the Anvil Web3 connection.

    Also perform the Anvil state reset for each test.
    """
    web3 = Web3(HTTPProvider(anvil.json_rpc_url, request_kwargs={"timeout": 10}))
    web3.middleware_onion.clear()
    install_chain_middleware(web3)
    return web3


@pytest.fixture()
def deployer(web3) -> HexAddress:
    """Deployer account.

    - This account will deploy all smart contracts

    - Starts with 10,000 ETH
    """
    return web3.eth.accounts[0]


@pytest.fixture
def usdc(web3) -> TokenDetails:
    details = fetch_erc20_details(web3, "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
    return details


@pytest.fixture
def hot_wallet(
    web3,
    deployer,
    usdc: TokenDetails,
    usdc_whale,
) -> HotWallet:
    """Create hot wallet for the signing tests.

    Top is up with some gas money and 500 USDC.
    """
    private_key = HexBytes(secrets.token_bytes(32))
    account = Account.from_key(private_key)
    wallet = HotWallet(account)
    wallet.sync_nonce(web3)
    tx_hash = web3.eth.send_transaction(
        {"to": wallet.address, "from": deployer, "value": 15 * 10**18}
    )
    assert_transaction_success_with_explanation(web3, tx_hash)
    tx_hash = usdc.contract.functions.transfer(
        wallet.address, 200_000 * 10**6
    ).transact({"from": usdc_whale})
    assert_transaction_success_with_explanation(web3, tx_hash)

    # Add to the local signer chain
    web3.middleware_onion.add(construct_sign_and_send_raw_middleware(account))

    return wallet


@pytest.fixture()
def strategy_file() -> Path:
    """Where do we load our strategy file."""
    return (
        Path(os.path.dirname(__file__))
        / ".."
        / ".."
        / "strategies"
        / "test_only"
        / "price_impact_crash.py"
    )

@pytest.fixture()
def state_file(tmp_path) -> Path:
    """The state file for the tests.

    Always start with an empty file.
    """
    path = Path(f"{tmp_path}/price_impact_crash.state.json")
    if path.exists():
        os.remove(path)
    return path


@pytest.fixture()
def environment(
    anvil: AnvilLaunch,
    hot_wallet: HotWallet,
    state_file: Path,
    strategy_file: Path,
) -> dict:
    """Passed to init and start commands as environment variables"""
    # Set up the configuration for the live trader
    environment = {
        "EXECUTOR_ID": "price_impact_crash",
        "STRATEGY_FILE": strategy_file.as_posix(),
        "PRIVATE_KEY": hot_wallet.account.key.hex(),
        "JSON_RPC_ANVIL": anvil.json_rpc_url,
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "hot_wallet",
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",
        "CONFIRMATION_BLOCK_COUNT": "0",  # Needed for test backend, Anvil
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "PATH": os.environ["PATH"],  # Needs forge
        "RUN_SINGLE_CYCLE": "true",  # Should crash on the first cycle
        "MAX_DATA_DELAY_MINUTES": str(24 * 60),
    }
    return environment


# ERROR tests/mainnet_fork/test_price_impact_crash.py::test_price_impact_crash - requests.exceptions.ReadTimeout: HTTPConnectionPool(host='localhost', port=25535): Read timed out. (read timeout=10)
@flaky.flaky
def test_price_impact_crash(
    environment: dict,
    state_file: Path,
    mocker,
):
    """Crash trade executor with a price impact trip wire.

    - Start a strategy against Anvil fork of mainnnet

    - Do a trade every second

    - Parameters.max_price_impact is set to such a low value that it will crash on the first trade
    """

    mocker.patch.dict("os.environ", environment, clear=True)

    # Initialise state.json
    app(["init"], standalone_mode=False)

    # Run a single cycle of the strategy
    with pytest.raises(PriceImpactToleranceExceeded):
        app(["start"], standalone_mode=False)

