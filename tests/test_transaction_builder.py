"""Ethereum hot wallet transactio builder tests."""

import secrets


import pytest
from eth_account import Account
from eth_typing import HexAddress
from hexbytes import HexBytes
from web3 import EthereumTesterProvider, Web3
from web3.contract import Contract

from eth_defi.gas import estimate_gas_fees
from eth_defi.hotwallet import HotWallet
from eth_defi.token import create_token
from eth_defi.uniswap_v2.deployment import UniswapV2Deployment, deploy_trading_pair, deploy_uniswap_v2_like
from tradeexecutor.ethereum.tx import TransactionBuilder, APPROVE_GAS_LIMIT
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier


@pytest.fixture
def tester_provider():
    # https://web3py.readthedocs.io/en/stable/examples.html#contract-unit-tests-in-python
    return EthereumTesterProvider()


@pytest.fixture
def eth_tester(tester_provider):
    # https://web3py.readthedocs.io/en/stable/examples.html#contract-unit-tests-in-python
    return tester_provider.ethereum_tester


@pytest.fixture
def web3(tester_provider):
    """Set up a local unit testing blockchain."""
    # https://web3py.readthedocs.io/en/stable/examples.html#contract-unit-tests-in-python
    return Web3(tester_provider)


@pytest.fixture
def chain_id(web3) -> int:
    """The test chain id (67)."""
    return web3.eth.chain_id


@pytest.fixture()
def deployer(web3) -> HexAddress:
    """Deploy account.

    Do some account allocation for tests.
    """
    return web3.eth.accounts[0]


@pytest.fixture
def usdc_token(web3, deployer: HexAddress) -> Contract:
    """Create USDC with 10M supply."""
    token = create_token(web3, deployer, "Fake USDC coin", "USDC", 10_000_000 * 10**6, 6)
    return token


@pytest.fixture
def aave_token(web3, deployer: HexAddress) -> Contract:
    """Create AAVE with 10M supply."""
    token = create_token(web3, deployer, "Fake Aave coin", "AAVE", 10_000_000 * 10**18, 18)
    return token


@pytest.fixture()
def uniswap_v2(web3, deployer) -> UniswapV2Deployment:
    """Uniswap v2 deployment."""
    deployment = deploy_uniswap_v2_like(web3, deployer)
    return deployment


@pytest.fixture
def weth_token(uniswap_v2: UniswapV2Deployment) -> Contract:
    """Mock some assets"""
    return uniswap_v2.weth


@pytest.fixture
def asset_usdc(usdc_token, chain_id) -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(chain_id, usdc_token.address, usdc_token.functions.symbol().call(), usdc_token.functions.decimals().call())


@pytest.fixture
def asset_weth(weth_token, chain_id) -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(chain_id, weth_token.address, weth_token.functions.symbol().call(), weth_token.functions.decimals().call())


@pytest.fixture
def asset_aave(aave_token, chain_id) -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(chain_id, aave_token.address, aave_token.functions.symbol().call(), aave_token.functions.decimals().call())


@pytest.fixture
def aave_usdc_uniswap_trading_pair(web3, deployer, uniswap_v2, aave_token, usdc_token) -> HexAddress:
    """AAVE-USDC pool with 200k liquidity."""
    pair_address = deploy_trading_pair(
        web3,
        deployer,
        uniswap_v2,
        aave_token,
        usdc_token,
        1000 * 10**18,  # 1000 AAVE liquidity
        200_000 * 10**6,  # 200k USDC liquidity
    )
    return pair_address


@pytest.fixture
def weth_usdc_uniswap_trading_pair(web3, deployer, uniswap_v2, weth_token, usdc_token) -> HexAddress:
    """AAVE-USDC pool with 1.7M liquidity."""
    pair_address = deploy_trading_pair(
        web3,
        deployer,
        uniswap_v2,
        weth_token,
        usdc_token,
        1000 * 10**18,  # 1000 ETH liquidity
        1_700_000 * 10**6,  # 1.7M USDC liquidity
    )
    return pair_address


@pytest.fixture
def weth_usdc_pair(weth_usdc_uniswap_trading_pair, asset_usdc, asset_weth) -> TradingPairIdentifier:
    """WETH-USDC pair representation in the trade executor domain."""
    return TradingPairIdentifier(asset_weth, asset_usdc, weth_usdc_uniswap_trading_pair, internal_id=int(weth_usdc_uniswap_trading_pair, 16))


@pytest.fixture
def aave_usdc_pair(aave_usdc_uniswap_trading_pair, asset_usdc, asset_aave) -> TradingPairIdentifier:
    """AAVE-USDC pair representation in the trade executor domain."""
    return TradingPairIdentifier(asset_aave, asset_usdc, aave_usdc_uniswap_trading_pair, internal_id=int(aave_usdc_uniswap_trading_pair, 16))


@pytest.fixture()
def hot_wallet(web3: Web3, usdc_token: Contract, deployer: HexAddress) -> HotWallet:
    """Our trading Ethereum account.

    Start with 10,000 USDC cash and 2 ETH.
    """
    account = Account.create()
    web3.eth.send_transaction({"from": deployer, "to": account.address, "value": 2*10**18})
    usdc_token.functions.transfer(account.address, 10_000 * 10**6).transact({"from": deployer})
    wallet = HotWallet(account)
    wallet.sync_nonce(web3)
    return wallet


def test_build_erc20_approve(
        web3,
        hot_wallet,
        uniswap_v2,
        usdc_token,
    ):
    """Create approve() transaction and execute it."""

    fees = estimate_gas_fees(web3)

    tx_builder = TransactionBuilder(
        web3,
        hot_wallet,
        fees,
    )

    tx = tx_builder.create_transaction(
        usdc_token,
        "approve",
        (uniswap_v2.router.address, 2**256-1),
        APPROVE_GAS_LIMIT,
    )

    tx_hash = tx_builder.broadcast(tx)

    receipt = web3.eth.get_transaction_receipt(tx_hash)
    assert receipt.status == 1  # Success


def test_planned_gas_price(
        web3,
        hot_wallet,
        uniswap_v2,
        usdc_token,
    ):
    """We can read planned gas price back from the transaction."""

    fees = estimate_gas_fees(web3)

    tx_builder = TransactionBuilder(
        web3,
        hot_wallet,
        fees,
    )

    tx = tx_builder.create_transaction(
        usdc_token,
        "approve",
        (uniswap_v2.router.address, 2**256-1),
        APPROVE_GAS_LIMIT,
    )

    # 1844540158
    assert tx.get_planned_gas_price() >= 1_000_000








