"""Live trading three-way trade tests.

- USDC->WMATIC->ETH routing

"""

import datetime
import os

from decimal import Decimal

import flaky
import pytest
from eth_account import Account
from eth_defi.confirmation import wait_transactions_to_complete
from eth_typing import HexAddress, HexStr

from eth_defi.token import TokenDetails, fetch_erc20_details
from tradingstrategy.exchange import ExchangeUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe
from web3 import Web3
from web3.contract import Contract

from eth_defi.hotwallet import HotWallet
from eth_defi.uniswap_v2.deployment import (
    UniswapV2Deployment,
    fetch_deployment,
)

from tradeexecutor.ethereum.tx import HotWalletTransactionBuilder
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_live_pricing import UniswapV2LivePricing
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_routing import (
    UniswapV2RoutingState,
    UniswapV2Routing,
)
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_execution import UniswapV2Execution
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.state.state import State
from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier
from tradeexecutor.strategy.account_correction import check_accounts
from tradeexecutor.strategy.execution_model import AssetManagementMode
from tradeexecutor.strategy.sync_model import SyncModel
from tradeexecutor.cli.bootstrap import create_sync_model


# https://docs.pytest.org/en/latest/how-to/skipping.html#skip-all-test-functions-of-a-class-or-module
from tradeexecutor.strategy.trading_strategy_universe import (
    create_pair_universe_from_code, TradingStrategyUniverse,
)
from tradingstrategy.chain import ChainId
from tradingstrategy.pair import PandasPairUniverse

from tradeexecutor.testing.synthetic_exchange_data import generate_exchange

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
def sand_token(web3: Web3) -> TokenDetails:
    return fetch_erc20_details(web3, "0xbbba073c31bf03b8acf7c28ef0738decf3695683")


@pytest.fixture
def matic_token(web3: Web3) -> TokenDetails:
    return fetch_erc20_details(web3, "0x0d500b1d8e8ef31e21c99d1db9a6444d3adf1270")

@pytest.fixture
def eth_matic_trading_pair_address() -> HexAddress:
    return HexAddress(HexStr("0x86f1d8390222A3691C28938eC7404A1661E618e0"))


@pytest.fixture
def matic_usdc_trading_pair_address() -> HexAddress:
    return HexAddress(HexStr("0x6e7a5fafcec6bb1e78bae2a1f0b612012bf14827"))


@pytest.fixture
def sand_asset(sand_token, chain_id) -> AssetIdentifier:
    return AssetIdentifier(
        chain_id,
        sand_token.address,
        sand_token.symbol,
        sand_token.decimals,
    )


@pytest.fixture
def sand_matic_trading_pair(
    quickswap,
    sand_asset,
    matic_asset,
) -> TradingPairIdentifier:
    return TradingPairIdentifier(
        sand_asset,
        matic_asset,
        "0x369582d2010b6ed950b571f4101e3bb9b554876f",  # https://tradingstrategy.ai/trading-view/polygon/quickswap/sand-matic
        internal_id=2000,
        internal_exchange_id=1000,
        exchange_address=quickswap.factory.address,
        fee=0.003
    )


@pytest.fixture()
def hot_wallet(
    web3: Web3,
        usdc_token: Contract,
        large_usdc_holder: HexAddress
) -> HotWallet:
    """Our trading Ethereum account.

    Start with 10,000 USDC cash and 2 polygon.
    """
    account = Account.create()
    matic_amount = 15
    web3.eth.send_transaction(
        {"from": large_usdc_holder, "to": account.address, "value": matic_amount * 10**18}
    )

    # Let's hope Binance hot wallet has 200k
    tx_hash = usdc_token.functions.transfer(account.address, 200_000 * 10**6).transact(
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
    eth_usdc_trading_pair,
    matic_usdc_trading_pair,
    eth_matic_trading_pair,
    sand_matic_trading_pair,
) -> PandasPairUniverse:
    """Pair universe needed for the trade routing."""
    return create_pair_universe_from_code(
        ChainId.polygon,
        [eth_usdc_trading_pair, matic_usdc_trading_pair, eth_matic_trading_pair, sand_matic_trading_pair],
    )


@pytest.fixture()
def routing_model(usdc_asset) -> UniswapV2Routing:

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

    return UniswapV2Routing(
        factory_router_map,
        allowed_intermediary_pairs,
        reserve_token_address=usdc_asset.address,
        trading_fee=0.0025,  # https://docs.panethswap.finance/products/panethswap-exchange/panethswap-pools
    )


@pytest.fixture
def strategy_universe(
    quickswap,
    pair_universe,
    usdc_asset,
) -> TradingStrategyUniverse:
    """Construct a trading strategy universe with a single Uniswap v2 DEX and our pairs

    - Use real on-chain routing data and forked EVM

    - Set up USDC as a reserve asset
    """

    exchange = generate_exchange(exchange_id=1, chain_id=ChainId.polygon, address=quickswap.factory.address)
    exchange_universe = ExchangeUniverse({exchange.exchange_id: exchange})

    data_universe = Universe(
        time_bucket=TimeBucket.not_applicable,
        chains={ChainId.polygon},
        pairs=pair_universe,
        exchange_universe=exchange_universe,
    )

    strategy_universe = TradingStrategyUniverse(
        data_universe=data_universe,
        reserve_assets={usdc_asset}
    )

    return strategy_universe


@pytest.fixture()
def pricing_model(web3, pair_universe, routing_model) -> UniswapV2LivePricing:
    """Pull the live price data from on-chain.

    - The price feed is frozen to the point when we fork the chain,
      only our own trades will modify the price
    """
    return UniswapV2LivePricing(web3, pair_universe, routing_model)


@pytest.fixture()
def execution_model(web3, hot_wallet) -> UniswapV2Execution:
    """Create an execution model that waits zero blocks for confirmation.

    Because we are using Anvil mainnet fork, there are not going to be any new blocks.
    """
    tx_builder = HotWalletTransactionBuilder(web3, hot_wallet)
    return UniswapV2Execution(
        tx_builder,
        confirmation_timeout=datetime.timedelta(seconds=10.00),  # Anvil has total 10 seconds to mine all txs in a batch
        confirmation_block_count=0,
        mainnet_fork=True,
    )


@pytest.fixture
def state(web3, hot_wallet, usdc_asset, sync_model) -> State:
    """State used in the tests.

    Start with USDC and some gas MATIC in the wallet.
    """
    state = State()
    sync_model.sync_initial(state)
    sync_model.sync_treasury(datetime.datetime.utcnow(), state, supported_reserves=[usdc_asset])
    assert state.portfolio.get_default_reserve_position().get_value() == 200_000.0  # We are Anvil rich
    return state


@pytest.fixture
def sync_model(web3, hot_wallet) -> SyncModel:
    return create_sync_model(
        AssetManagementMode.hot_wallet,
        web3,
        hot_wallet,
    )


# Flaky because of shitty nodes
@flaky.flaky()
def test_simple_routing_three_leg_live(
    web3,
    strategy_universe: TradingStrategyUniverse,
    state: State,
    sync_model: SyncModel,
    execution_model: UniswapV2Execution,
    pricing_model: UniswapV2LivePricing,
    eth_matic_trading_pair: TradingPairIdentifier,
    matic_usdc_trading_pair: TradingPairIdentifier,
    sand_matic_trading_pair: TradingPairIdentifier,
    routing_model:  UniswapV2Routing,
    sand_token: TokenDetails,
    usdc_asset: AssetIdentifier,
    usdc_token: Contract,
    matic_token: TokenDetails,
    hot_wallet: HotWallet,
):
    """Perform a three-legged trade USDC->WMATIC-ETH on a live mainnet forked Quickswap.

    - Try to use as little as possible unit testing infrastructure
      and rely on real execution model.

    - We initialise all components except ``PandasStrategyRunner``
      as we do not have a strategy to run

    - Perform accounting checks in each phase
    """

    position_manager = PositionManager(
        datetime.datetime.utcnow(),
        strategy_universe,
        state,
        pricing_model,
    )

    routing_state = UniswapV2RoutingState(
        strategy_universe.data_universe.pairs,
        execution_model.tx_builder,
    )

    # Buy 100,000 USD worth of ETH, going thru ETH->MATIC
    trades = position_manager.open_spot(
        matic_usdc_trading_pair,
        Decimal(100),
    )

    trades += position_manager.open_spot(
        sand_matic_trading_pair,
        Decimal(100_000),
        slippage_tolerance=0.04,  # default tolerance 17bps is too small for this pool
    )

    execution_model.execute_trades(
        datetime.datetime.utcnow(),
        state,
        trades,
        routing_model,
        routing_state,
        check_balances=True,
    )

    assert all(t.is_success() for t in trades), f"Trades failed: {trades}"

    # Inspect SAND-WMATIC trade
    trade_2 = trades[1]
    assert trade_2.is_executed()
    assert trade_2.is_success(), f"Trade failed:\n {trade_2.get_revert_reason()}"

    swap_tx = trade_2.blockchain_transactions[0]
    path = swap_tx.args[2]
    assert len(path) == 3  # Three-legged trade

    # We received the tokens we bought
    assert sand_token.fetch_balance_of(hot_wallet.address) > 0

    # Check that expected amounts match when doing buy and hold
    clean, df = check_accounts(strategy_universe.data_universe.pairs, [usdc_asset], state, sync_model)
    assert clean is True, f"Accounts are not clean:\n{df}"

    #
    # Cycle 2
    #
    # - Close SAND position
    # - Increase WMATIC position
    #

    trades = []

    sand_position = position_manager.get_current_position_for_pair(sand_matic_trading_pair)
    trades += position_manager.close_position(sand_position)

    trades += position_manager.adjust_position(
        matic_usdc_trading_pair,
        100.0,  # +100 USD
        quantity_delta=None,
        weight=1.0,
    )

    execution_model.execute_trades(
        datetime.datetime.utcnow(),
        state,
        trades,
        routing_model,
        routing_state,
        check_balances=True,
    )

    assert all(t.is_success() for t in trades)

    # Check SAND sell planned vs. expected is good
    t = trades[0]
    assert t.planned_reserve == pytest.approx(t.executed_reserve, Decimal(0.01))

    # Check that expected amounts match after closing the position
    clean, df = check_accounts(strategy_universe.data_universe.pairs, [usdc_asset], state, sync_model)
    assert clean is True, f"Accounts are not clean:\n{df}"

    # We received the tokens we bought
    assert sand_token.fetch_balance_of(hot_wallet.address) == 0
    assert 0 < matic_token.fetch_balance_of(hot_wallet.address) < 3000
