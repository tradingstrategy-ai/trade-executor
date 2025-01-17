"""Hot wallet based synching tests."""
import datetime
import secrets

import pytest
from _pytest.fixtures import FixtureRequest
from eth_account import Account
from eth_defi.provider.anvil import launch_anvil, AnvilLaunch
from eth_defi.chain import install_chain_middleware
from eth_defi.hotwallet import HotWallet
from eth_defi.token import create_token, fetch_erc20_details
from eth_defi.trace import assert_transaction_success_with_explanation
from eth_typing import HexAddress
from hexbytes import HexBytes
from tradingstrategy.chain import ChainId
from web3 import Web3, HTTPProvider
from web3.contract import Contract

from tradeexecutor.ethereum.hot_wallet_sync_model import HotWalletSyncModel
from tradeexecutor.ethereum.token import translate_token_details
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.state import State


@pytest.fixture()
def anvil() -> AnvilLaunch:

    # London hardfork will enable EIP-1559 style gas fees
    anvil = launch_anvil(
        hardfork="london",
        gas_limit=15_000_000,  # Max 5M gas per block, or per transaction in test automining
    )
    try:
        # Make the initial snapshot ("zero state") to which we revert between tests
        # web3 = Web3(HTTPProvider(anvil.json_rpc_url))
        # snapshot_id = make_anvil_custom_rpc_request(web3, "evm_snapshot")
        # assert snapshot_id == "0x0"
        yield anvil
    finally:
        anvil.close()


@pytest.fixture()
def web3(anvil: AnvilLaunch) -> Web3:
    """Set up the Anvil Web3 connection.

    Also perform the Anvil state reset for each test.
    """
    web3 = Web3(HTTPProvider(anvil.json_rpc_url, request_kwargs={"timeout": 2}))
    install_chain_middleware(web3)
    return web3


@pytest.fixture()
def deployer(web3) -> HexAddress:
    """Deployer account.

    - This account will deploy all smart contracts

    - Starts with 10,000 ETH
    """
    return web3.eth.accounts[0]


@pytest.fixture()
def user_1(web3) -> HexAddress:
    """User account.

    Do some account allocation for tests.
    """
    return web3.eth.accounts[1]


@pytest.fixture()
def usdc(web3, deployer) -> Contract:
    """Mock USDC token.

    All initial $100M goes to `deployer`
    """
    token = create_token(web3, deployer, "USD Coin", "USDC", 100_000_000 * 10**6, decimals=6)
    return token


@pytest.fixture()
def usdc_asset(usdc: Contract) -> AssetIdentifier:
    """USDC as a persistent id.
    """
    details = fetch_erc20_details(usdc.w3, usdc.address)
    return translate_token_details(details)


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
    return wallet


def test_hot_wallet_sync_model_init(
    web3: Web3,
    hot_wallet,
):
    """Set up strategy sync mode on init."""

    sync_model = HotWalletSyncModel(web3, hot_wallet)
    state = State()
    sync_model.sync_initial(state)

    deployment = state.sync.deployment
    assert deployment.address == hot_wallet.address
    assert deployment.block_number > 0
    assert deployment.tx_hash is None
    assert deployment.block_mined_at is not None
    assert deployment.vault_token_symbol is None
    assert deployment.vault_token_name is None
    assert deployment.chain_id == ChainId.anvil


def test_hot_wallet_sync_model_deposit(
    web3: Web3,
    hot_wallet: HotWallet,
    deployer: HexAddress,
    usdc: Contract,
    usdc_asset: AssetIdentifier,
):
    """Update reserve balances."""

    sync_model = HotWalletSyncModel(web3, hot_wallet)
    state = State()
    sync_model.sync_initial(state)

    supported_reserves = [usdc_asset]

    sync_model.sync_treasury(
        datetime.datetime.utcnow(),
        state,
        supported_reserves
    )

    assert len(state.portfolio.reserves) == 0

    tx_hash = usdc.functions.transfer(hot_wallet.address, 500 * 10**6).transact({"from": deployer})
    assert_transaction_success_with_explanation(web3, tx_hash)

    sync_model.sync_treasury(
        datetime.datetime.utcnow(),
        state,
        supported_reserves
    )

    assert len(state.portfolio.reserves) == 1

    assert state.portfolio.calculate_total_equity() == 500


def test_hot_wallet_sync_model_sync_twice(
    web3: Web3,
    hot_wallet: HotWallet,
    deployer: HexAddress,
    usdc: Contract,
    usdc_asset: AssetIdentifier,
):
    """Check that we do not generate extra events."""

    sync_model = HotWalletSyncModel(web3, hot_wallet)
    state = State()
    sync_model.sync_initial(state)

    supported_reserves = [usdc_asset]

    sync_model.sync_treasury(
        datetime.datetime.utcnow(),
        state,
        supported_reserves
    )

    assert len(state.portfolio.reserves) == 0

    tx_hash = usdc.functions.transfer(hot_wallet.address, 500 * 10**6).transact({"from": deployer})
    assert_transaction_success_with_explanation(web3, tx_hash)


    sync_model.sync_treasury(
        datetime.datetime.utcnow(),
        state,
        supported_reserves
    )

    assert len(state.portfolio.reserves) == 1

    reserve_position = state.portfolio.get_default_reserve_position()
    assert len(reserve_position.balance_updates) == 1

    sync_model.sync_treasury(
        datetime.datetime.utcnow(),
        state,
        supported_reserves
    )

    assert len(reserve_position.balance_updates) == 1

