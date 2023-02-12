"""Uniswap v2 routing model tests.

To run these tests, we need to connect to BNB Chain:

.. code-block::  shell

    export BNB_CHAIN_JSON_RPC="https://bsc-dataseed.binance.org/"
    pytest -k test_uniswap_v2_routing

"""

import datetime
import logging
import os
from decimal import Decimal

import flaky
import pytest
from eth_account import Account
from eth_defi.anvil import fork_network_anvil
from eth_defi.chain import install_chain_middleware

from eth_defi.gas import estimate_gas_fees, node_default_gas_price_strategy
from eth_defi.confirmation import wait_transactions_to_complete
from eth_typing import HexAddress, HexStr
from web3 import Web3, HTTPProvider
from web3.contract import Contract

from eth_defi.hotwallet import HotWallet
from eth_defi.uniswap_v3.deployment import (
    UniswapV3Deployment,
    deploy_uniswap_v3,
    deploy_pool,
    add_liquidity,
)
from eth_defi.token import create_token
from eth_defi.uniswap_v3.utils import get_default_tick_range

from tradeexecutor.ethereum.tx import TransactionBuilder
from tradeexecutor.ethereum.uniswap_v3_routing import (
    UniswapV3RoutingState,
    UniswapV3SimpleRoutingModel,
    OutOfBalance,
)
from tradeexecutor.ethereum.uniswap_v3_execution import UniswapV3ExecutionModel
from tradeexecutor.ethereum.universe import create_pair_universe
from tradeexecutor.ethereum.wallet import sync_reserves
from tradeexecutor.state.sync import apply_sync_events
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.state import State
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.position import TradingPosition

from tradeexecutor.cli.log import setup_pytest_logging


# https://docs.pytest.org/en/latest/how-to/skipping.html#skip-all-test-functions-of-a-class-or-module
from tradeexecutor.strategy.trading_strategy_universe import (
    create_pair_universe_from_code,
)
from tradeexecutor.testing.pairuniversetrader import PairUniverseTestTrader
from tradingstrategy.chain import ChainId
from tradingstrategy.pair import PandasPairUniverse


pytestmark = pytest.mark.skipif(
    os.environ.get("JSON_RPC_POLYGON") is None,
    reason="Set JSON_RPC_POLYGON environment variable to Polygon node to run this test",
)


@pytest.fixture()
def weth_usdc_fee() -> int:
    return 3000


@pytest.fixture()
def aave_usdc_fee() -> int:
    return 3000


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request)


@pytest.fixture()
def large_busd_holder() -> HexAddress:
    """A random account picked from BNB Smart chain that holds a lot of BUSD.

    This account is unlocked on Ganache, so you have access to good BUSD stash.

    `To find large holder accounts, use bscscan <https://bscscan.com/token/0xe9e7cea3dedca5984780bafc599bd69add087d56#balances>`_.
    """
    # Binance Hot Wallet 6
    return HexAddress(HexStr("0x8894E0a0c962CB723c1976a4421c95949bE2D4E3"))


@pytest.fixture()
def anvil_bnb_chain_fork(logger, large_busd_holder) -> str:
    """Create a testable fork of live BNB chain.

    :return: JSON-RPC URL for Web3
    """

    mainnet_rpc = os.environ["JSON_RPC_POLYGON"]

    launch = fork_network_anvil(mainnet_rpc, unlocked_addresses=[large_busd_holder])
    try:
        yield launch.json_rpc_url
    finally:
        launch.close(log_level=logging.INFO)


@pytest.fixture
def web3(anvil_bnb_chain_fork: str):
    """Set up a local unit testing blockchain."""
    # https://web3py.readthedocs.io/en/stable/examples.html#contract-unit-tests-in-python
    web3 = Web3(HTTPProvider(anvil_bnb_chain_fork, request_kwargs={"timeout": 5}))
    web3.eth.set_gas_price_strategy(node_default_gas_price_strategy)
    install_chain_middleware(web3)
    return web3


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
def hot_wallet(
    web3: Web3, busd_token: Contract, large_busd_holder: HexAddress
) -> HotWallet:
    """Our trading Ethereum account.

    Start with 10,000 USDC cash and 2 BNB.
    """
    account = Account.create()
    web3.eth.send_transaction(
        {"from": large_busd_holder, "to": account.address, "value": 2 * 10**18}
    )
    tx_hash = busd_token.functions.transfer(
        account.address, 10_000 * 10**18
    ).transact({"from": large_busd_holder})
    wait_transactions_to_complete(web3, [tx_hash])
    wallet = HotWallet(account)
    wallet.sync_nonce(web3)
    return wallet


@pytest.fixture
def usdc_token(web3, deployer: HexAddress) -> Contract:
    """Create USDC with 10M supply."""
    token = create_token(
        web3, deployer, "Fake USDC coin", "USDC", 10_000_000 * 10**6, 6
    )
    return token


@pytest.fixture
def aave_token(web3, deployer: HexAddress) -> Contract:
    """Create AAVE with 10M supply."""
    token = create_token(
        web3, deployer, "Fake Aave coin", "AAVE", 10_000_000 * 10**18, 18
    )
    return token


@pytest.fixture()
def uniswap_v3(web3, deployer) -> UniswapV3Deployment:
    """Uniswap v2 deployment."""
    deployment = deploy_uniswap_v3(web3, deployer)
    return deployment


@pytest.fixture
def weth_token(uniswap_v3: UniswapV3Deployment) -> Contract:
    """Mock some assets"""
    return uniswap_v3.weth


@pytest.fixture
def asset_usdc(usdc_token, chain_id) -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(
        chain_id,
        usdc_token.address,
        usdc_token.functions.symbol().call(),
        usdc_token.functions.decimals().call(),
    )


@pytest.fixture
def asset_weth(weth_token, chain_id) -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(
        chain_id,
        weth_token.address,
        weth_token.functions.symbol().call(),
        weth_token.functions.decimals().call(),
    )


@pytest.fixture
def asset_aave(aave_token, chain_id) -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(
        chain_id,
        aave_token.address,
        aave_token.functions.symbol().call(),
        aave_token.functions.decimals().call(),
    )


@pytest.fixture
def aave_usdc_uniswap_trading_pair(
    web3, deployer, uniswap_v3, aave_token, usdc_token, aave_usdc_fee
) -> HexAddress:
    """AAVE-USDC pool with 200k liquidity. Fee of 0.1%"""
    min_tick, max_tick = get_default_tick_range(aave_usdc_fee)

    pool_contract = deploy_pool(
        web3,
        deployer,
        deployment=uniswap_v3,
        token0=aave_token,
        token1=usdc_token,
        fee=aave_usdc_fee,
    )

    add_liquidity(
        web3,
        deployer,
        deployment=uniswap_v3,
        pool=pool_contract,
        amount0=1000 * 10**18,  # 1000 AAVE liquidity
        amount1=200_000 * 10**6,  # 200k USDC liquidity
        lower_tick=min_tick,
        upper_tick=max_tick,
    )
    return pool_contract.address


@pytest.fixture
def weth_usdc_uniswap_trading_pair(
    web3, deployer, uniswap_v3, weth_token, usdc_token, weth_usdc_fee
) -> HexAddress:
    """ETH-USDC pool with 1.7M liquidity."""
    min_tick, max_tick = get_default_tick_range(weth_usdc_fee)

    pool_contract = deploy_pool(
        web3,
        deployer,
        deployment=uniswap_v3,
        token0=weth_token,
        token1=usdc_token,
        fee=weth_usdc_fee,
    )

    add_liquidity(
        web3,
        deployer,
        deployment=uniswap_v3,
        pool=pool_contract,
        amount0=1000 * 10**18,  # 1000 ETH liquidity
        amount1=1_700_000 * 10**6,  # 1.7M USDC liquidity
        lower_tick=min_tick,
        upper_tick=max_tick,
    )
    return pool_contract.address


@pytest.fixture
def weth_usdc_pair(
    uniswap_v3, weth_usdc_uniswap_trading_pair, asset_usdc, asset_weth, weth_usdc_fee
) -> TradingPairIdentifier:
    return TradingPairIdentifier(
        asset_weth,
        asset_usdc,
        weth_usdc_uniswap_trading_pair,
        uniswap_v3.factory.address,
        fee=weth_usdc_fee,
    )


@pytest.fixture
def aave_usdc_pair(
    uniswap_v3, aave_usdc_uniswap_trading_pair, asset_usdc, asset_aave, aave_usdc_fee
) -> TradingPairIdentifier:
    return TradingPairIdentifier(
        asset_aave,
        asset_usdc,
        aave_usdc_uniswap_trading_pair,
        uniswap_v3.factory.address,
        fee=aave_usdc_fee,
    )


@pytest.fixture
def start_ts() -> datetime.datetime:
    """Timestamp of action started"""
    return datetime.datetime(2022, 1, 1, tzinfo=None)


@pytest.fixture
def supported_reserves(usdc) -> list[AssetIdentifier]:
    """Timestamp of action started"""
    return [usdc]


@pytest.fixture
def supported_reserves(asset_usdc) -> list[AssetIdentifier]:
    """The reserve currencies we support."""
    return [asset_usdc]


@pytest.fixture()
def pair_universe(web3, weth_usdc_pair, aave_usdc_pair) -> PandasPairUniverse:
    return create_pair_universe(web3, None, [weth_usdc_pair, aave_usdc_pair])


@pytest.fixture()
def portfolio(web3, hot_wallet, start_ts, supported_reserves) -> Portfolio:
    """A portfolio loaded with the initial cash.

    We start with 10,000 USDC.
    """
    portfolio = Portfolio()
    events = sync_reserves(web3, start_ts, hot_wallet.address, [], supported_reserves)
    apply_sync_events(portfolio, events)
    return portfolio


@pytest.fixture()
def state(portfolio) -> State:
    return State(portfolio=portfolio)


@pytest.fixture()
def routing_model(busd_asset):

    address_map = {
        "factory": "0x1F98431c8aD98523631AE4a59f267346ea31F984",
        "router": "0xE592427A0AEce92De3Edee1F18E0157C05861564",
        "position_manager": "0xC36442b4a4522E871399CD717aBDD847Ab11FE88",
        "quoter": "0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6"
        # "router02":"0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45",
        # "quoterV2":"0x61fFE014bA17989E743c5F6cB21bF9697530B21e"
    }

    allowed_intermediary_pairs = {
        # For WBNB pairs route thru (WBNB, BUSD) pool
        # https://tradingstrategy.ai/trading-view/binance/pancakeswap-v2/bnb-busd
        "0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c": "0x58f876857a02d6762e0101bb5c46a8c1ed44dc16",
    }

    return UniswapV3SimpleRoutingModel(
        address_map,
        allowed_intermediary_pairs,
        reserve_token_address=busd_asset.address,
    )


@pytest.fixture()
def execution_model(web3, hot_wallet) -> UniswapV3ExecutionModel:
    return UniswapV3ExecutionModel(web3, hot_wallet)


# Flaky because Ganache hangs
@flaky.flaky()
def test_simple_routing_one_leg(
    web3,
    hot_wallet,
    busd_asset,
    cake_token,
    routing_model,
    cake_busd_trading_pair,
    pair_universe,
):
    """Make 1x two way trade BUSD -> Cake.

    - Buy Cake with BUSD
    """

    # Get live fee structure from BNB Chain
    fees = estimate_gas_fees(web3)

    # Prepare a transaction builder
    tx_builder = TransactionBuilder(
        web3,
        hot_wallet,
        fees,
    )

    # Create
    routing_state = UniswapV3RoutingState(pair_universe, tx_builder)

    txs = routing_model.trade(
        routing_state,
        cake_busd_trading_pair,
        busd_asset,
        100 * 10**18,  # Buy Cake worth of 100 BUSD,
        check_balances=True,
    )

    # We should have 1 approve, 1 swap
    assert len(txs) == 2

    # Execute
    tx_builder.broadcast_and_wait_transactions_to_complete(
        web3, txs, revert_reasons=True
    )

    # Check all transactions succeeded
    for tx in txs:
        assert tx.is_success(), f"Transaction failed: {tx}"

    # We received the tokens we bought
    assert cake_token.functions.balanceOf(hot_wallet.address).call() > 0


def test_simple_routing_buy_sell(
    web3,
    hot_wallet,
    busd_asset,
    cake_asset,
    cake_token,
    busd_token,
    routing_model,
    cake_busd_trading_pair,
    pair_universe,
):
    """Make 2x two way trade BUSD -> Cake -> BUSD."""

    # Get live fee structure from BNB Chain
    fees = estimate_gas_fees(web3)

    # Prepare a transaction builder
    tx_builder = TransactionBuilder(
        web3,
        hot_wallet,
        fees,
    )

    # Create
    routing_state = UniswapV3RoutingState(pair_universe, tx_builder)

    txs = routing_model.trade(
        routing_state,
        cake_busd_trading_pair,
        busd_asset,
        100 * 10**18,  # Buy Cake worth of 100 BUSD,
        check_balances=True,
    )

    # We should have 1 approve, 1 swap
    assert len(txs) == 2

    # Execute
    tx_builder.broadcast_and_wait_transactions_to_complete(
        web3, txs, revert_reasons=True
    )

    assert all(tx.is_success() for tx in txs)

    # We received the tokens we bought
    cake_balance = cake_token.functions.balanceOf(hot_wallet.address).call()

    # Sell Cake we received
    txs = routing_model.trade(
        routing_state,
        cake_busd_trading_pair,
        cake_asset,
        cake_balance,  # Sell all cake
        check_balances=True,
    )
    assert len(txs) == 2
    # Execute
    tx_builder.broadcast_and_wait_transactions_to_complete(
        web3, txs, revert_reasons=True
    )
    assert all(tx.is_success() for tx in txs)

    # We started with 10_000 BUSD
    balance = busd_token.functions.balanceOf(hot_wallet.address).call()
    assert balance == pytest.approx(9999500634326300440503)


def test_simple_routing_not_enough_balance(
    web3,
    hot_wallet,
    busd_asset,
    routing_model,
    cake_busd_trading_pair,
):
    """Try to buy, but does not have cash."""

    # Get live fee structure from BNB Chain
    fees = estimate_gas_fees(web3)

    # Prepare a transaction builder
    tx_builder = TransactionBuilder(
        web3,
        hot_wallet,
        fees,
    )

    # Create
    routing_state = UniswapV3RoutingState(pair_universe, tx_builder)

    with pytest.raises(OutOfBalance):
        routing_model.trade(
            routing_state,
            cake_busd_trading_pair,
            busd_asset,
            1_000_000_000 * 10**18,  # Buy Cake worth of 10B BUSD,
            check_balances=True,
        )


def test_simple_routing_three_leg(
    web3,
    hot_wallet,
    busd_asset,
    bnb_asset,
    cake_asset,
    cake_token,
    routing_model,
    cake_bnb_trading_pair,
    bnb_busd_trading_pair,
    pair_universe,
):
    """Make 1x two way trade BUSD -> BNB -> Cake."""

    # Get live fee structure from BNB Chain
    fees = estimate_gas_fees(web3)

    # Prepare a transaction builder
    tx_builder = TransactionBuilder(
        web3,
        hot_wallet,
        fees,
    )

    routing_state = UniswapV3RoutingState(pair_universe, tx_builder)

    txs = routing_model.trade(
        routing_state,
        cake_bnb_trading_pair,
        busd_asset,
        100 * 10**18,  # Buy Cake worth of 100 BUSD,
        check_balances=True,
        intermediary_pair=bnb_busd_trading_pair,
    )

    # We should have 1 approve, 1 swap
    assert len(txs) == 2

    # Execute
    tx_builder.broadcast_and_wait_transactions_to_complete(
        web3, txs, revert_reasons=True
    )

    # Check all transactions succeeded
    for tx in txs:
        assert tx.is_success(), f"Transaction failed: {tx}"

    # We received the tokens we bought
    assert cake_token.functions.balanceOf(hot_wallet.address).call() > 0


def test_three_leg_buy_sell(
    web3,
    hot_wallet,
    busd_asset,
    bnb_asset,
    cake_asset,
    cake_token,
    busd_token,
    routing_model,
    cake_bnb_trading_pair,
    bnb_busd_trading_pair,
    pair_universe,
):
    """Make trades BUSD -> BNB -> Cake and Cake -> BNB -> BUSD."""

    # We start without Cake
    balance = cake_token.functions.balanceOf(hot_wallet.address).call()
    assert balance == 0

    # Get live fee structure from BNB Chain
    fees = estimate_gas_fees(web3)

    # Prepare a transaction builder
    tx_builder = TransactionBuilder(
        web3,
        hot_wallet,
        fees,
    )

    routing_state = UniswapV3RoutingState(pair_universe, tx_builder)

    txs = routing_model.trade(
        routing_state,
        cake_bnb_trading_pair,
        busd_asset,
        100 * 10**18,  # Buy Cake worth of 100 BUSD,
        check_balances=True,
        intermediary_pair=bnb_busd_trading_pair,
    )

    # We should have 1 approve, 1 swap
    assert len(txs) == 2

    # # Check for three legs
    buy_tx = txs[1]
    path = buy_tx.args[2]
    assert len(path) == 3

    # Execute
    tx_builder.broadcast_and_wait_transactions_to_complete(
        web3, txs, revert_reasons=True
    )

    # Check all transactions succeeded
    for tx in txs:
        assert tx.is_success(), f"Transaction failed: {tx}"

    # We received the tokens we bought
    balance = cake_token.functions.balanceOf(hot_wallet.address).call()
    assert balance > 0

    txs = routing_model.trade(
        routing_state,
        cake_bnb_trading_pair,
        cake_asset,
        balance,
        check_balances=True,
        intermediary_pair=bnb_busd_trading_pair,
    )

    # We should have 1 approve, 1 swap
    assert len(txs) == 2

    # Check for three legs
    sell_tx = txs[1]
    path = sell_tx.args[2]
    assert len(path) == 3, f"Bad sell tx {sell_tx}"

    # Execute
    tx_builder.broadcast_and_wait_transactions_to_complete(
        web3, txs, revert_reasons=True
    )

    # Check all transactions succeeded
    for tx in txs:
        assert tx.is_success(), f"Transaction failed: {tx}"

    # We started with 10_000 BUSD
    balance = busd_token.functions.balanceOf(hot_wallet.address).call()
    assert balance == pytest.approx(9999003745120046326850)


def test_three_leg_buy_sell_twice_on_chain(
    web3,
    hot_wallet,
    busd_asset,
    bnb_asset,
    cake_asset,
    cake_token,
    busd_token,
    routing_model,
    cake_bnb_trading_pair,
    bnb_busd_trading_pair,
    pair_universe,
):
    """Make trades 2x BUSD -> BNB -> Cake and Cake -> BNB -> BUSD.

    Because we do the round trip 2x, we should not need approvals
    on the second time and we need one less transactions.

    We reset the routing state between, forcing
    the routing state to read the approval information
    back from the chain.
    """

    # Get live fee structure from BNB Chain
    fees = estimate_gas_fees(web3)

    # Prepare a transaction builder
    tx_builder = TransactionBuilder(
        web3,
        hot_wallet,
        fees,
    )

    routing_state = None

    def trip():

        txs = routing_model.trade(
            routing_state,
            cake_bnb_trading_pair,
            busd_asset,
            100 * 10**18,  # Buy Cake worth of 100 BUSD,
            check_balances=True,
            intermediary_pair=bnb_busd_trading_pair,
        )

        # Execute
        tx_builder.broadcast_and_wait_transactions_to_complete(
            web3, txs, revert_reasons=True
        )

        # Check all transactions succeeded
        for tx in txs:
            assert tx.is_success(), f"Transaction failed: {tx}"

        # We received the tokens we bought
        balance = cake_token.functions.balanceOf(hot_wallet.address).call()
        assert balance > 0

        txs2 = routing_model.trade(
            routing_state,
            cake_bnb_trading_pair,
            cake_asset,
            balance,
            check_balances=True,
            intermediary_pair=bnb_busd_trading_pair,
        )

        # Execute
        tx_builder.broadcast_and_wait_transactions_to_complete(
            web3, txs2, revert_reasons=True
        )

        # Check all transactions succeeded
        for tx in txs2:
            assert tx.is_success(), f"Transaction failed: {tx}"

        return txs + txs2

    routing_state = UniswapV3RoutingState(pair_universe, tx_builder)
    txs_1 = trip()
    assert len(txs_1) == 4
    routing_state = UniswapV3RoutingState(pair_universe, tx_builder)
    txs_2 = trip()
    assert len(txs_2) == 2


def test_three_leg_buy_sell_twice(
    web3,
    hot_wallet,
    busd_asset,
    bnb_asset,
    cake_asset,
    cake_token,
    busd_token,
    routing_model,
    cake_bnb_trading_pair,
    bnb_busd_trading_pair,
    pair_universe,
):
    """Make trades 2x BUSD -> BNB -> Cake and Cake -> BNB -> BUSD.

    Because we do the round trip 2x, we should not need approvals
    on the second time and we need one less transactions.
    """

    # Get live fee structure from BNB Chain
    fees = estimate_gas_fees(web3)

    # Prepare a transaction builder
    tx_builder = TransactionBuilder(
        web3,
        hot_wallet,
        fees,
    )

    routing_state = UniswapV3RoutingState(pair_universe, tx_builder)

    def trip():

        txs = routing_model.trade(
            routing_state,
            cake_bnb_trading_pair,
            busd_asset,
            100 * 10**18,  # Buy Cake worth of 100 BUSD,
            check_balances=True,
            intermediary_pair=bnb_busd_trading_pair,
        )

        # Execute
        tx_builder.broadcast_and_wait_transactions_to_complete(
            web3, txs, revert_reasons=True
        )

        # Check all transactions succeeded
        for tx in txs:
            assert tx.is_success(), f"Transaction failed: {tx}"

        # We received the tokens we bought
        balance = cake_token.functions.balanceOf(hot_wallet.address).call()
        assert balance > 0

        txs2 = routing_model.trade(
            routing_state,
            cake_bnb_trading_pair,
            cake_asset,
            balance,
            check_balances=True,
            intermediary_pair=bnb_busd_trading_pair,
        )

        # Execute
        tx_builder.broadcast_and_wait_transactions_to_complete(
            web3, txs2, revert_reasons=True
        )

        # Check all transactions succeeded
        for tx in txs2:
            assert tx.is_success(), f"Transaction failed: {tx}"

        return txs + txs2

    txs_1 = trip()
    assert len(txs_1) == 4
    txs_2 = trip()
    assert len(txs_2) == 2


# Flaky becaues Ganache hangs
@flaky.flaky()
def test_stateful_routing_three_legs(
    web3,
    pair_universe,
    hot_wallet,
    busd_asset,
    bnb_asset,
    cake_asset,
    cake_token,
    routing_model,
    cake_bnb_trading_pair,
    bnb_busd_trading_pair,
    state: State,
    execution_model: UniswapV3ExecutionModel,
):
    """Perform 3-leg buy/sell using RoutingModel.execute_trades().

    This also shows how blockchain native transactions
    and state management integrate.
    """

    # Get live fee structure from BNB Chain
    fees = estimate_gas_fees(web3)

    # Prepare a transaction builder
    tx_builder = TransactionBuilder(web3, hot_wallet, fees)

    routing_state = UniswapV3RoutingState(pair_universe, tx_builder)

    trader = PairUniverseTestTrader(state)

    reserve = pair_universe.get_token(busd_asset.address)
    if not reserve:
        all_tokens = pair_universe.get_all_tokens()
        assert (
            reserve
        ), f"Reserve asset {busd_asset.address} missing in the universe {busd_asset}, we have {all_tokens}"

    # Buy Cake via BUSD -> BNB pool for 100 USD
    trades = [trader.buy(cake_bnb_trading_pair, Decimal(100))]

    t = trades[0]
    assert t.is_buy()
    assert t.reserve_currency == busd_asset
    assert t.pair == cake_bnb_trading_pair

    state.start_trades(datetime.datetime.utcnow(), trades)
    routing_model.execute_trades_internal(
        pair_universe, routing_state, trades, check_balances=True
    )
    execution_model.broadcast_and_resolve(state, trades, stop_on_execution_failure=True)

    # Check all all trades and transactions completed
    for t in trades:
        assert t.is_success()
        for tx in t.blockchain_transactions:
            assert tx.is_success()

    # We received the tokens we bought
    assert cake_token.functions.balanceOf(hot_wallet.address).call() > 0

    cake_position: TradingPosition = state.portfolio.open_positions[1]
    assert cake_position

    # Buy Cake via BUSD -> BNB pool for 100 USD
    trades = [trader.sell(cake_bnb_trading_pair, cake_position.get_quantity())]

    t = trades[0]
    assert t.is_sell()
    assert t.reserve_currency == busd_asset
    assert t.pair == cake_bnb_trading_pair
    assert t.planned_quantity == -cake_position.get_quantity()

    state.start_trades(datetime.datetime.utcnow(), trades)
    routing_model.execute_trades_internal(
        pair_universe, routing_state, trades, check_balances=True
    )
    execution_model.broadcast_and_resolve(state, trades, stop_on_execution_failure=True)

    # Check all all trades and transactions completed
    for t in trades:
        assert t.is_success()
        for tx in t.blockchain_transactions:
            assert tx.is_success()

    # On-chain balance is zero after the sell
    assert cake_token.functions.balanceOf(hot_wallet.address).call() == 0


def test_stateful_routing_two_legs(
    web3,
    pair_universe,
    hot_wallet,
    busd_asset,
    bnb_asset,
    cake_asset,
    cake_token,
    routing_model,
    cake_busd_trading_pair,
    state: State,
    execution_model: UniswapV3ExecutionModel,
):
    """Perform 2-leg buy/sell using RoutingModel.execute_trades().

    This also shows how blockchain native transactions
    and state management integrate.

    Routing is abstracted away - this test is not different from one above,
    except for the trading pair that we have changed.
    """

    # Get live fee structure from BNB Chain
    fees = estimate_gas_fees(web3)

    # Prepare a transaction builder
    tx_builder = TransactionBuilder(web3, hot_wallet, fees)

    routing_state = UniswapV3RoutingState(pair_universe, tx_builder)

    trader = PairUniverseTestTrader(state)

    # Buy Cake via BUSD -> BNB pool for 100 USD
    trades = [trader.buy(cake_busd_trading_pair, Decimal(100))]

    t = trades[0]
    assert t.is_buy()
    assert t.reserve_currency == busd_asset
    assert t.pair == cake_busd_trading_pair

    state.start_trades(datetime.datetime.utcnow(), trades)
    routing_model.execute_trades_internal(
        pair_universe, routing_state, trades, check_balances=True
    )
    execution_model.broadcast_and_resolve(state, trades, stop_on_execution_failure=True)

    # Check all all trades and transactions completed
    for t in trades:
        assert t.is_success()
        for tx in t.blockchain_transactions:
            assert tx.is_success()

    # We received the tokens we bought
    assert cake_token.functions.balanceOf(hot_wallet.address).call() > 0

    cake_position: TradingPosition = state.portfolio.open_positions[1]
    assert cake_position

    # Buy Cake via BUSD -> BNB pool for 100 USD
    trades = [trader.sell(cake_busd_trading_pair, cake_position.get_quantity())]

    t = trades[0]
    assert t.is_sell()
    assert t.reserve_currency == busd_asset
    assert t.pair == cake_busd_trading_pair
    assert t.planned_quantity == -cake_position.get_quantity()

    state.start_trades(datetime.datetime.utcnow(), trades)
    routing_model.execute_trades_internal(
        pair_universe, routing_state, trades, check_balances=True
    )
    execution_model.broadcast_and_resolve(state, trades, stop_on_execution_failure=True)

    # Check all all trades and transactions completed
    for t in trades:
        assert t.is_success()
        for tx in t.blockchain_transactions:
            assert tx.is_success()

    # On-chain balance is zero after the sell
    assert cake_token.functions.balanceOf(hot_wallet.address).call() == 0
