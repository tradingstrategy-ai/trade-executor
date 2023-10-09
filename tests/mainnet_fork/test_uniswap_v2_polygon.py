"""Uniswap v2 routing model tests.

To run these tests, we need to connect to polygon Chain:

.. code-block::  shell

    export polygon_CHAIN_JSON_RPC="https://bsc-dataseed.binance.org/"
    pytest -k test_uniswap_v2_routing

"""

import datetime
import os
from decimal import Decimal

import flaky
import pytest
from eth_account import Account
from eth_defi.provider.anvil import fork_network_anvil
from eth_defi.chain import install_chain_middleware
from eth_defi.abi import get_deployed_contract
from eth_defi.confirmation import wait_transactions_to_complete
from eth_typing import HexAddress, HexStr
from web3 import Web3, HTTPProvider
from web3.contract import Contract

from eth_defi.hotwallet import HotWallet
from eth_defi.uniswap_v2.deployment import (
    UniswapV2Deployment,
    fetch_deployment,
)

from tradeexecutor.ethereum.tx import HotWalletTransactionBuilder
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_routing import (
    UniswapV2RoutingState,
    UniswapV2SimpleRoutingModel,
    OutOfBalance,
)
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_execution import UniswapV2ExecutionModel
from tradeexecutor.ethereum.wallet import sync_reserves
from tradeexecutor.testing.dummy_wallet import apply_sync_events
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.state import State
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.strategy.account_correction import check_accounts
from tradeexecutor.strategy.execution_model import AssetManagementMode
from tradeexecutor.strategy.sync_model import SyncModel

from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.cli.bootstrap import create_sync_model


# https://docs.pytest.org/en/latest/how-to/skipping.html#skip-all-test-functions-of-a-class-or-module
from tradeexecutor.strategy.trading_strategy_universe import (
    create_pair_universe_from_code,
)
from tradeexecutor.testing.pairuniversetrader import PairUniverseTestTrader
from tradeexecutor.testing.pytest_helpers import is_failed_test
from tradingstrategy.chain import ChainId
from tradingstrategy.pair import PandasPairUniverse


pytestmark = pytest.mark.skipif(
    os.environ.get("JSON_RPC_POLYGON") is None,
    reason="Set JSON_RPC_POLYGON environment variable to Polygon node to run this test",
)


@pytest.fixture()
def quickswap(web3) -> UniswapV2Deployment:
    """Fetch live quickswap v3 deployment.

    See https://docs.quickswap.exchange/concepts/protocol-overview/03-smart-contracts for more information
    """
    deployment = fetch_deployment(
        web3,
        "0x5757371414417b8C6CAad45bAeF941aBc7d3Ab32",
        "0xa5E0829CaCEd8fFDD4De3c43696c57F7D7A678ff",
        "0x96e8ac4277198ff8b6f785478aa9a39f403cb768dd02cbee326c3e7da348845f",
    )
    return deployment


@pytest.fixture
def wmatic_token(quickswap: UniswapV2Deployment) -> Contract:
    """WMATIC is native token of Polygon."""
    return quickswap.weth


@pytest.fixture
def eth_matic_trading_pair_address() -> HexAddress:
    """See https://tradingstrategy.ai/trading-view/polygon/quickswap/matic-usdc"""
    return HexAddress(HexStr("0x86f1d8390222A3691C28938eC7404A1661E618e0"))


@pytest.fixture
def matic_usdc_trading_pair_address() -> HexAddress:
    """See https://tradingstrategy.ai/trading-view/polygon/quickswap/matic-usdc"""
    return HexAddress(HexStr("0x6e7a5fafcec6bb1e78bae2a1f0b612012bf14827"))


@pytest.fixture()
def hot_wallet(
    web3: Web3, usdc_token: Contract, large_usdc_holder: HexAddress
) -> HotWallet:
    """Our trading Ethereum account.

    Start with 10,000 USDC cash and 2 polygon.
    """
    account = Account.create()
    web3.eth.send_transaction(
        {"from": large_usdc_holder, "to": account.address, "value": 2 * 10**18}
    )
    tx_hash = usdc_token.functions.transfer(account.address, 10_000_000 * 10**6).transact(
        {"from": large_usdc_holder}
    )
    wait_transactions_to_complete(web3, [tx_hash])
    wallet = HotWallet(account)
    wallet.sync_nonce(web3)
    return wallet


@pytest.fixture
def eth_usdc_trading_pair(eth_asset, usdc_asset, quickswap) -> TradingPairIdentifier:
    """eth-usdc pair representation in the trade executor domain."""
    return TradingPairIdentifier(
        eth_asset,
        usdc_asset,
        "0x853ee4b2a13f8a742d64c8f088be7ba2131f670d",  #  https://tradingstrategy.ai/trading-view/polygon/quickswap/eth-usdc
        internal_id=1000,  # random number
        internal_exchange_id=1000,  # random number
        exchange_address=quickswap.factory.address,
        fee=0.003
    )


@pytest.fixture
def matic_usdc_trading_pair(
    matic_asset, usdc_asset, quickswap
) -> TradingPairIdentifier:
    return TradingPairIdentifier(
        matic_asset,
        usdc_asset,
        "0x6e7a5fafcec6bb1e78bae2a1f0b612012bf14827",  #  https://tradingstrategy.ai/trading-view/polygon/quickswap/matic-usdc
        internal_id=1001,  # random number
        internal_exchange_id=1000,  # random number
        exchange_address=quickswap.factory.address,
        fee=0.003
    )


@pytest.fixture
def eth_matic_trading_pair(eth_asset, matic_asset, quickswap) -> TradingPairIdentifier:
    """eth-usdc pair representation in the trade executor domain."""
    return TradingPairIdentifier(
        eth_asset,
        matic_asset,
        "0xadbf1854e5883eb8aa7baf50705338739e558e5b",  #  https://tradingstrategy.ai/trading-view/polygon/quickswap/matic-eth
        internal_id=1002,  # random number
        internal_exchange_id=1000,  # random number
        exchange_address=quickswap.factory.address,
        fee=0.003
    )


@pytest.fixture
def pair_universe(
    eth_usdc_trading_pair, matic_usdc_trading_pair, eth_matic_trading_pair
) -> PandasPairUniverse:
    """Pair universe needed for the trade routing."""
    return create_pair_universe_from_code(
        ChainId.polygon,
        [eth_usdc_trading_pair, matic_usdc_trading_pair, eth_matic_trading_pair],
    )


@pytest.fixture()
def routing_model(usdc_asset):

    # Allowed exchanges as factory -> router pairs
    factory_router_map = {
        "0x5757371414417b8C6CAad45bAeF941aBc7d3Ab32": (
        "0xa5E0829CaCEd8fFDD4De3c43696c57F7D7A678ff", "0x96e8ac4277198ff8b6f785478aa9a39f403cb768dd02cbee326c3e7da348845f"),
    }

    allowed_intermediary_pairs = {
        # Route WMATIC through USDC:WMATIC fee 0.05% pool,
        # https://tradingstrategy.ai/trading-view/polygon/quickswap/matic-usdc
        "0x0d500b1d8e8ef31e21c99d1db9a6444d3adf1270": "0x6e7a5fafcec6bb1e78bae2a1f0b612012bf14827",
        # Route WETH through USDC:WETH fee 0.05% pool,
        # https://tradingstrategy.ai/trading-view/polygon/quickswap/eth-usdc
        "0x7ceb23fd6bc0add59e62ac25578270cff1b9f619": "0x853ee4b2a13f8a742d64c8f088be7ba2131f670d",
    }

    return UniswapV2SimpleRoutingModel(
        factory_router_map,
        allowed_intermediary_pairs,
        reserve_token_address=usdc_asset.address,
        trading_fee=0.0025,  # https://docs.panethswap.finance/products/panethswap-exchange/panethswap-pools
    )


@pytest.fixture()
def execution_model(web3, hot_wallet) -> UniswapV2ExecutionModel:
    tx_builder = HotWalletTransactionBuilder(web3, hot_wallet)
    return UniswapV2ExecutionModel(tx_builder)


@pytest.fixture
def state(web3, hot_wallet, usdc_asset) -> State:
    """State used in the tests."""
    state = State()

    events = sync_reserves(
        web3, datetime.datetime.utcnow(), hot_wallet.address, [], [usdc_asset]
    )
    assert len(events) > 0
    apply_sync_events(state, events)
    reserve_currency, exchange_rate = state.portfolio.get_default_reserve_asset()
    assert reserve_currency == usdc_asset
    return state


@pytest.fixture
def sync_model(web3, hot_wallet) -> SyncModel:
    return create_sync_model(
        AssetManagementMode.hot_wallet,
        web3,
        hot_wallet,
    )


def test_simple_routing_three_leg(
        web3,
        hot_wallet,
        usdc_asset,
        matic_asset,
        eth_asset,
        eth_token,
        routing_model,
        eth_matic_trading_pair,
        matic_usdc_trading_pair,
        pair_universe,
        state: State,
        sync_model: SyncModel,
):
    """Make 1x two way trade USDC -> BNB -> Eth."""

    # Prepare a transaction builder
    tx_builder = HotWalletTransactionBuilder(
        web3,
        hot_wallet,
    )

    routing_state = UniswapV2RoutingState(pair_universe, tx_builder)

    # We start out with 0 Eth balance
    balance = eth_token.functions.balanceOf(hot_wallet.address).call()
    assert balance == 0, f"Expected balance of 0. Balance is {balance}" 

    txs = routing_model.trade(
        routing_state,
        eth_matic_trading_pair,
        usdc_asset,
        100 * 10 ** 6,  # Buy Eth worth of 100 USDC,
        check_balances=True,
        intermediary_pair=matic_usdc_trading_pair,
    )

    # We should have 1 approve, 1 swap
    assert len(txs) == 2

    # Execute
    tx_builder.broadcast_and_wait_transactions_to_complete(
        web3,
        txs,
        revert_reasons=True
    )

    # Check all transactions succeeded
    for tx in txs:
        assert tx.is_success(), f"Transaction failed: {tx}"

    # We received the tokens we bought
    assert eth_token.functions.balanceOf(hot_wallet.address).call() > 0

    clean, df = check_accounts(pair_universe, [usdc_asset], state, sync_model)
    
    assert clean is True, f"Accounts are not clean: {df}"


def test_three_leg_buy_sell(
        web3,
        hot_wallet,
        usdc_asset,
        matic_asset,
        eth_asset,
        eth_token,
        usdc_token,
        routing_model,
        eth_matic_trading_pair,
        matic_usdc_trading_pair,
        pair_universe,
        state,
        sync_model,
):
    """Make trades USDC -> BNB -> Eth and Eth -> BNB -> USDC."""

    # We start without Eth
    balance = eth_token.functions.balanceOf(hot_wallet.address).call()
    assert balance == 0

    # Prepare a transaction builder
    tx_builder = HotWalletTransactionBuilder(
        web3,
        hot_wallet,
    )

    routing_state = UniswapV2RoutingState(pair_universe, tx_builder)

    txs = routing_model.trade(
        routing_state,
        eth_matic_trading_pair,
        usdc_asset,
        100 * 10 ** 6,  # Buy Eth worth of 100 USDC,
        check_balances=True,
        intermediary_pair=matic_usdc_trading_pair,
    )

    # We should have 1 approve, 1 swap
    assert len(txs) == 2

    # # Check for three legs
    buy_tx = txs[1]
    path = buy_tx.transaction_args[2]
    assert len(path) == 3

    # Execute
    tx_builder.broadcast_and_wait_transactions_to_complete(
        web3,
        txs,
        revert_reasons=True
    )

    # Check all transactions succeeded
    for tx in txs:
        assert tx.is_success(), f"Transaction failed: {tx}"

    # We received the tokens we bought
    balance = eth_token.functions.balanceOf(hot_wallet.address).call()
    assert balance > 0

    txs = routing_model.trade(
        routing_state,
        eth_matic_trading_pair,
        eth_asset,
        balance,
        check_balances=True,
        intermediary_pair=matic_usdc_trading_pair,
    )

    # We should have 1 approve, 1 swap
    assert len(txs) == 2

    # Check for three legs
    sell_tx = txs[1]
    path = sell_tx.transaction_args[2]
    assert len(path) == 3, f"Bad sell tx {sell_tx}"

    # Execute
    tx_builder.broadcast_and_wait_transactions_to_complete(
        web3,
        txs,
        revert_reasons=True
    )

    # Check all transactions succeeded
    for tx in txs:
        assert tx.is_success(), f"Transaction failed: {tx}"

    # We started with 10_000 USDC
    balance = usdc_token.functions.balanceOf(hot_wallet.address).call()
    assert balance == pytest.approx(9999998805591)

    clean, df = check_accounts(pair_universe, [usdc_asset], state, sync_model)

    assert clean is True, f"Accounts are not clean: {df}"


# def test_three_leg_buy_sell_twice_on_chain(
#         web3,
#         hot_wallet,
#         usdc_asset,
#         matic_asset,
#         eth_asset,
#         eth_token,
#         usdc_token,
#         routing_model,
#         eth_matic_trading_pair,
#         matic_usdc_trading_pair,
#         pair_universe,
# ):
#     """Make trades 2x USDC -> BNB -> Eth and Eth -> BNB -> USDC.

#     Because we do the round trip 2x, we should not need approvals
#     on the second time and we need one less transactions.

#     We reset the routing state between, forcing
#     the routing state to read the approval information
#     back from the chain.
#     """

#     # Prepare a transaction builder
#     tx_builder = HotWalletTransactionBuilder(
#         web3,
#         hot_wallet,
#     )

#     routing_state = None

#     def trip():

#         txs = routing_model.trade(
#             routing_state,
#             eth_matic_trading_pair,
#             usdc_asset,
#             100 * 10 ** 18,  # Buy Eth worth of 100 USDC,
#             check_balances=True,
#             intermediary_pair=matic_usdc_trading_pair,
#         )

#         # Execute
#         tx_builder.broadcast_and_wait_transactions_to_complete(
#             web3,
#             txs,
#             revert_reasons=True
#         )

#         # Check all transactions succeeded
#         for tx in txs:
#             assert tx.is_success(), f"Transaction failed: {tx}"

#         # We received the tokens we bought
#         balance = eth_token.functions.balanceOf(hot_wallet.address).call()
#         assert balance > 0

#         txs2 = routing_model.trade(
#             routing_state,
#             eth_matic_trading_pair,
#             eth_asset,
#             balance,
#             check_balances=True,
#             intermediary_pair=matic_usdc_trading_pair,
#         )

#         # Execute
#         tx_builder.broadcast_and_wait_transactions_to_complete(
#             web3,
#             txs2,
#             revert_reasons=True
#         )

#         # Check all transactions succeeded
#         for tx in txs2:
#             assert tx.is_success(), f"Transaction failed: {tx}"

#         return txs + txs2

#     routing_state = UniswapV2RoutingState(pair_universe, tx_builder)
#     txs_1 = trip()
#     assert len(txs_1) == 4
#     routing_state = UniswapV2RoutingState(pair_universe, tx_builder)
#     txs_2 = trip()
#     assert len(txs_2) == 2


# def test_three_leg_buy_sell_twice(
#         web3,
#         hot_wallet,
#         usdc_asset,
#         matic_asset,
#         eth_asset,
#         eth_token,
#         usdc_token,
#         routing_model,
#         eth_matic_trading_pair,
#         matic_usdc_trading_pair,
#         pair_universe,
# ):
#     """Make trades 2x USDC -> BNB -> Eth and Eth -> BNB -> USDC.

#     Because we do the round trip 2x, we should not need approvals
#     on the second time and we need one less transactions.
#     """

#     # Prepare a transaction builder
#     tx_builder = HotWalletTransactionBuilder(
#         web3,
#         hot_wallet,
#     )

#     routing_state = UniswapV2RoutingState(pair_universe, tx_builder)

#     def trip():

#         txs = routing_model.trade(
#             routing_state,
#             eth_matic_trading_pair,
#             usdc_asset,
#             100 * 10 ** 18,  # Buy Eth worth of 100 USDC,
#             check_balances=True,
#             intermediary_pair=matic_usdc_trading_pair,
#         )

#         # Execute
#         tx_builder.broadcast_and_wait_transactions_to_complete(
#             web3,
#             txs,
#             revert_reasons=True
#         )

#         # Check all transactions succeeded
#         for tx in txs:
#             assert tx.is_success(), f"Transaction failed: {tx}"

#         # We received the tokens we bought
#         balance = eth_token.functions.balanceOf(hot_wallet.address).call()
#         assert balance > 0

#         txs2 = routing_model.trade(
#             routing_state,
#             eth_matic_trading_pair,
#             eth_asset,
#             balance,
#             check_balances=True,
#             intermediary_pair=matic_usdc_trading_pair,
#         )

#         # Execute
#         tx_builder.broadcast_and_wait_transactions_to_complete(
#             web3,
#             txs2,
#             revert_reasons=True
#         )

#         # Check all transactions succeeded
#         for tx in txs2:
#             assert tx.is_success(), f"Transaction failed: {tx}"

#         return txs + txs2

#     txs_1 = trip()
#     assert len(txs_1) == 4
#     txs_2 = trip()
#     assert len(txs_2) == 2


# def test_stateful_routing_three_legstest_stateful_routing_three_legs(
#         web3,
#         pair_universe,
#         hot_wallet,
#         usdc_asset,
#         matic_asset,
#         eth_asset,
#         eth_token,
#         routing_model,
#         eth_matic_trading_pair,
#         matic_usdc_trading_pair,
#         state: State,
#         execution_model: UniswapV2ExecutionModel
# ):
#     """Perform 3-leg buy/sell using RoutingModel.execute_trades().

#     This also shows how blockchain native transactions
#     and state management integrate.
#     """

#     # Prepare a transaction builder
#     tx_builder = HotWalletTransactionBuilder(web3, hot_wallet)

#     routing_state = UniswapV2RoutingState(pair_universe, tx_builder)

#     trader = PairUniverseTestTrader(state)

#     reserve = pair_universe.get_token(usdc_asset.address)
#     if not reserve:
#         all_tokens = pair_universe.get_all_tokens()
#         assert reserve, f"Reserve asset {usdc_asset.address} missing in the universe {usdc_asset}, we have {all_tokens}"

#     # Buy Eth via USDC -> BNB pool for 100 USD
#     trades = [
#         trader.buy(eth_matic_trading_pair, Decimal(100))
#     ]

#     t = trades[0]
#     assert t.is_buy()
#     assert t.reserve_currency == usdc_asset
#     assert t.pair == eth_matic_trading_pair

#     state.start_execution_all(datetime.datetime.utcnow(), trades)
#     routing_model.execute_trades_internal(pair_universe, routing_state, trades, check_balances=True)
#     execution_model.broadcast_and_resolve(state, trades, stop_on_execution_failure=True)

#     # Check all all trades and transactions completed
#     for t in trades:
#         assert t.is_success()
#         for tx in t.blockchain_transactions:
#             assert tx.is_success()

#     # We received the tokens we bought
#     assert eth_token.functions.balanceOf(hot_wallet.address).call() > 0

#     eth_position: TradingPosition = state.portfolio.open_positions[1]
#     assert eth_position

#     # Buy Eth via USDC -> BNB pool for 100 USD
#     trades = [
#         trader.sell(eth_matic_trading_pair, eth_position.get_quantity())
#     ]

#     t = trades[0]
#     assert t.is_sell()
#     assert t.reserve_currency == usdc_asset
#     assert t.pair == eth_matic_trading_pair
#     assert t.planned_quantity == -eth_position.get_quantity()

#     state.start_execution_all(datetime.datetime.utcnow(), trades)
#     routing_model.execute_trades_internal(pair_universe, routing_state, trades, check_balances=True)
#     execution_model.broadcast_and_resolve(state, trades, stop_on_execution_failure=True)

#     # Check all all trades and transactions completed
#     for t in trades:
#         assert t.is_success()
#         for tx in t.blockchain_transactions:
#             assert tx.is_success()

#     # On-chain balance is zero after the sell
#     assert eth_token.functions.balanceOf(hot_wallet.address).call() == 0


# def test_stateful_routing_out_of_balance(
#         web3,
#         pair_universe,
#         hot_wallet,
#         usdc_asset,
#         matic_asset,
#         eth_asset,
#         eth_token,
#         routing_model,
#         eth_usdc_trading_pair,
#         state: State,
#         execution_model: UniswapV2ExecutionModel,
#         usdc_token,
#         user_2
# ):
#     """Abort trade because we do not have enough tokens on-chain.

#     - Clear the tokens before the trade
#     """

#     # Prepare a transaction builder
#     tx_builder = HotWalletTransactionBuilder(web3, hot_wallet)

#     # Move all USDC expect 1 unit out from the wallet
#     balance = usdc_token.functions.balanceOf(hot_wallet.address).call()
#     remove_tokens_tx = usdc_token.functions.transfer(user_2, balance - 1)
#     signed_tx = hot_wallet.sign_bound_call_with_new_nonce(remove_tokens_tx)
#     web3.eth.send_raw_transaction(signed_tx.rawTransaction)

#     routing_state = UniswapV2RoutingState(pair_universe, tx_builder)

#     trader = PairUniverseTestTrader(state)

#     # Buy Eth via USDC -> BNB pool for 100 USD
#     trades = [
#         trader.buy(eth_usdc_trading_pair, Decimal(100))
#     ]

#     state.start_execution_all(datetime.datetime.utcnow(), trades)

#     with pytest.raises(OutOfBalance):
#         routing_model.execute_trades_internal(pair_universe, routing_state, trades, check_balances=True)


# def test_stateful_routing_adjust_epsilon(
#         web3,
#         pair_universe,
#         hot_wallet,
#         usdc_asset,
#         matic_asset,
#         eth_asset,
#         eth_token,
#         routing_model,
#         eth_usdc_trading_pair,
#         state: State,
#         execution_model: UniswapV2ExecutionModel,
#         usdc_token,
#         user_2,
# ):
#     """Perform a trade where we have a rounding error in our reserves.
#     """

#     # Prepare a transaction builder
#     tx_builder = HotWalletTransactionBuilder(web3, hot_wallet)

#     # Move 1 unit of USDC out from the wallet
#     balance = usdc_token.functions.balanceOf(hot_wallet.address).call()
#     diff = (balance - 100 * 10 ** 18) + 1
#     remove_tokens_tx = usdc_token.functions.transfer(user_2, diff)
#     signed_tx = hot_wallet.sign_bound_call_with_new_nonce(remove_tokens_tx)
#     web3.eth.send_raw_transaction(signed_tx.rawTransaction)
#     routing_state = UniswapV2RoutingState(pair_universe, tx_builder)

#     trader = PairUniverseTestTrader(state)

#     # Buy Eth via USDC -> BNB pool for 100 USD
#     trades = [
#         trader.buy(eth_usdc_trading_pair, Decimal(100))
#     ]

#     t = trades[0]
#     assert t.is_buy()
#     assert t.reserve_currency == usdc_asset
#     assert t.pair == eth_usdc_trading_pair

#     state.start_execution_all(datetime.datetime.utcnow(), trades)
#     routing_model.execute_trades_internal(pair_universe, routing_state, trades, check_balances=True)
#     execution_model.broadcast_and_resolve(state, trades, stop_on_execution_failure=True)

#     # Check all all trades and transactions completed
#     for t in trades:
#         assert t.is_success()
#         for tx in t.blockchain_transactions:
#             assert tx.is_success()

#     # Check that we recorded spending amount correctly
#     trade_tx = trades[0].blockchain_transactions[-1]
#     assert trade_tx.other["reserve_amount"] == str(100 * 10 ** 18)
#     assert trade_tx.other["adjusted_reserve_amount"] == str(100 * 10 ** 18 - 1)


# def test_stateful_routing_adjust_epsilon_sell(
#         web3,
#         pair_universe,
#         hot_wallet,
#         usdc_asset,
#         matic_asset,
#         eth_asset,
#         eth_token,
#         routing_model,
#         eth_usdc_trading_pair,
#         state: State,
#         execution_model: UniswapV2ExecutionModel,
#         user_2,
# ):
#     """Perform a trade where we have a rounding error in our reserves, sell side.
#     """

#     # Prepare a transaction builder
#     tx_builder = HotWalletTransactionBuilder(web3, hot_wallet)
#     routing_state = UniswapV2RoutingState(pair_universe, tx_builder)
#     trader = PairUniverseTestTrader(state)

#     # Buy Eth via USDC -> BNB pool for 100 USD
#     trades = [
#         trader.buy(eth_usdc_trading_pair, Decimal(100))
#     ]

#     state.start_execution_all(datetime.datetime.utcnow(), trades)
#     routing_model.execute_trades_internal(pair_universe, routing_state, trades, check_balances=True)
#     execution_model.broadcast_and_resolve(state, trades, stop_on_execution_failure=True)

#     # We received the tokens we bought
#     assert eth_token.functions.balanceOf(hot_wallet.address).call() > 0

#     eth_position: TradingPosition = state.portfolio.open_positions[1]

#     # Move 0.000001 Eth away to simulate rounding error
#     remove_tokens_tx = eth_token.functions.transfer(user_2, 1)
#     signed_tx = hot_wallet.sign_bound_call_with_new_nonce(remove_tokens_tx)
#     web3.eth.send_raw_transaction(signed_tx.rawTransaction)

#     t = trader.sell(eth_usdc_trading_pair, eth_position.get_quantity())
#     assert t.is_sell()

#     trades = [t]
#     state.start_execution_all(datetime.datetime.utcnow(), trades)
#     routing_model.execute_trades_internal(pair_universe, routing_state, trades, check_balances=True)
#     execution_model.broadcast_and_resolve(state, trades, stop_on_execution_failure=True)
#     assert t.is_success()

#     # On-chain balance is zero after the sell
#     assert eth_token.functions.balanceOf(hot_wallet.address).call() == eth_position.get_quantity() * 10 ** 187

#     # Check that we recorded spending amount correctly
#     trade_tx = t.blockchain_transactions[-1]
#     assert int(trade_tx.other["reserve_amount"]) == int(trade_tx.other["adjusted_reserve_amount"]) + 1
