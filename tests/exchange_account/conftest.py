"""Exchange account test fixtures.

Provides fixtures needed for CLI tests in exchange_account directory.
"""

import logging
import pytest
import secrets

from hexbytes import HexBytes
from eth_typing import HexAddress
from web3 import Web3, HTTPProvider
from web3.contract import Contract
from eth_account import Account

from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import AnvilLaunch, launch_anvil
from eth_defi.chain import install_chain_middleware
from eth_defi.token import create_token
from eth_defi.trace import assert_transaction_success_with_explanation


logger = logging.getLogger(__name__)


@pytest.fixture()
def anvil() -> AnvilLaunch:
    """Launch local Anvil node."""
    anvil = launch_anvil()
    try:
        yield anvil
    finally:
        anvil.close()


@pytest.fixture()
def web3(anvil: AnvilLaunch) -> Web3:
    """Set up the Anvil Web3 connection."""
    web3 = Web3(HTTPProvider(anvil.json_rpc_url, request_kwargs={"timeout": 2}))
    web3.middleware_onion.clear()
    install_chain_middleware(web3)
    return web3


@pytest.fixture()
def deployer(web3) -> HexAddress:
    """Deployer account with ETH."""
    return web3.eth.accounts[0]


@pytest.fixture()
def user_1(web3) -> HexAddress:
    """User account."""
    return web3.eth.accounts[1]


@pytest.fixture()
def usdc(web3, deployer) -> Contract:
    """Mock USDC token."""
    token = create_token(web3, deployer, "USD Coin", "USDC", 100_000_000 * 10**6, decimals=6)
    return token


@pytest.fixture()
def dummy_token(web3, deployer) -> Contract:
    """Mock dummy token for trading pair."""
    token = create_token(web3, deployer, "Dummy Token", "DUMMY", 100_000_000 * 10**18, decimals=18)
    return token


@pytest.fixture
def hot_wallet(web3, deployer, user_1, usdc: Contract) -> HotWallet:
    """Create hot wallet for CLI tests.

    Top up with some gas money and 500 USDC.
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
