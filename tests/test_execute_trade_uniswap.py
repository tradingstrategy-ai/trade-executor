"""Test trading against faux Uniswap pool."""

import datetime
import secrets
from decimal import Decimal
from typing import List

import pytest
from eth_account import Account
from eth_account.signers.local import LocalAccount
from eth_typing import HexAddress
from hexbytes import HexBytes
from web3 import EthereumTesterProvider, Web3
from web3.contract import Contract

from smart_contracts_for_testing.abi import get_deployed_contract
from smart_contracts_for_testing.token import create_token
from smart_contracts_for_testing.uniswap_v2 import UniswapV2Deployment, deploy_uniswap_v2_like, deploy_trading_pair, \
    estimate_price
from tradeexecutor.ethereum.execution import prepare_swaps, broadcast, wait_trades_to_complete, resolve_trades
from tradeexecutor.ethereum.wallet import sync_reserves, sync_portfolio
from tradeexecutor.state.state import AssetIdentifier, Portfolio, State, TradingPairIdentifier, TradeStatus
from tradeexecutor.testing.trader import TestTrader


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


@pytest.fixture()
def hot_wallet_private_key(web3) -> HexBytes:
    """Generate a private key"""
    return HexBytes(secrets.token_bytes(32))


@pytest.fixture
def usdc_token(web3, deployer: HexAddress) -> Contract:
    """Create USDC with 10M supply."""
    token = create_token(web3, deployer, "Fake USDC coin", "USDC", 10_000_000 * 10**6, 6)
    return token


@pytest.fixture
def usdc(usdc_token, web3) -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(web3.eth.chain_id, usdc_token.address, "USDC", 6)


@pytest.fixture
def weth(web3) -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(web3.eth.chain_id, "0x1", "WETH", 18)


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
def uniswap_trading_pair(web3, deployer, uniswap_v2, weth_token, usdc_token) -> HexAddress:
    """WETH-USDC pool with 1.7M liquidity."""
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
def weth_usdc_pair(uniswap_trading_pair, asset_usdc, asset_weth) -> TradingPairIdentifier:
    return TradingPairIdentifier(asset_weth, asset_usdc, uniswap_trading_pair)


@pytest.fixture
def start_ts() -> datetime.datetime:
    """Timestamp of action started"""
    return datetime.datetime(2022, 1, 1, tzinfo=None)


@pytest.fixture
def supported_reserves(usdc) -> List[AssetIdentifier]:
    """Timestamp of action started"""
    return [usdc]


@pytest.fixture()
def hot_wallet(web3: Web3, usdc_token: Contract, hot_wallet_private_key: HexBytes, deployer: HexAddress) -> LocalAccount:
    """Our trading Ethereum account.

    Start with 10,000 USDC cash and 2 ETH.
    """
    account = Account.from_key(hot_wallet_private_key)
    web3.eth.send_transaction({"from": deployer, "to": account.address, "value": 2*10**18})
    usdc_token.functions.transfer(account.address, 10_000 * 10**6).transact({"from": deployer})
    return account


@pytest.fixture
def supported_reserves(usdc) -> List[AssetIdentifier]:
    """The reserve currencies we support."""
    return [usdc]


@pytest.fixture()
def portfolio(web3, usdc, hot_wallet, start_ts, supported_reserves) -> Portfolio:
    """A portfolio loaded with the initial cash"""
    portfolio = Portfolio()
    events = sync_reserves(web3, 1, start_ts, hot_wallet.address, [], supported_reserves)
    sync_portfolio(portfolio, events)
    return portfolio


@pytest.fixture()
def state(portfolio) -> State:
    return State(portfolio=portfolio)


def test_execute_trade_instructions_buy_weth(
        web3: Web3,
        state: State,
        uniswap_v2: UniswapV2Deployment,
        hot_wallet: LocalAccount,
        usdc_token, weth_token,
        weth_usdc_pair,
        start_ts):
    """Sync reserves from one deposit."""

    portfolio = state.portfolio

    # We have everything in cash
    assert portfolio.get_total_equity() == 10_000
    assert portfolio.get_current_cash() == 10_000

    # Buy 500 USDC worth of WETH
    trader = TestTrader(state)

    buy_amount = 500

    # Estimate price
    raw_assumed_quantity = estimate_price(web3, uniswap_v2, weth_token, usdc_token, buy_amount*10**6)
    assumed_quantity = Decimal(raw_assumed_quantity) / Decimal(10**18)

    # 1: plan
    position, trade = trader.prepare_buy(weth_usdc_pair, assumed_quantity, 1700)
    assert state.portfolio.get_total_equity() == pytest.approx(10000.0)
    assert trade.get_status() == TradeStatus.planned

    ts = start_ts + datetime.timedelta(seconds=1)

    # 2: prepare
    # Prepare transactions
    prepare_swaps(
        web3,
        hot_wallet,
        uniswap_v2,
        ts,
        state,
        [trade]
    )

    assert trade.get_status() == TradeStatus.started
    assert trade.tx_info.tx_hash is not None
    assert trade.tx_info.details["from"] == hot_wallet.address
    assert trade.tx_info.signed_bytes is not None
    assert trade.tx_info.nonce == 0

    #: 3 broadcast
    ts = start_ts + datetime.timedelta(seconds=1)
    broadcasted = broadcast(web3, ts, [trade])
    assert trade.get_status() == TradeStatus.broadcasted
    assert trade.broadcasted_at is not None

    #: 4 process results
    ts = start_ts + datetime.timedelta(seconds=1)
    receipts = wait_trades_to_complete(web3, [trade])
    resolve_trades(state, ts, broadcasted, receipts)