"""Fixtures to set up Polygon fork generic router for uniswap v2 + uniswap v3 + aave + 1delta"""

import pandas as pd
import pytest as pytest

from tradeexecutor.ethereum.execution import EthereumExecution
from tradeexecutor.ethereum.tx import HotWalletTransactionBuilder
from tradingstrategy.chain import ChainId

from eth_defi.hotwallet import HotWallet
from eth_defi.uniswap_v2.deployment import UniswapV2Deployment
from eth_defi.uniswap_v3.deployment import UniswapV3Deployment
from tradingstrategy.exchange import ExchangeUniverse
from tradingstrategy.lending import LendingProtocolType
from tradingstrategy.timebucket import TimeBucket
from tradeexecutor.ethereum.one_delta.one_delta_live_pricing import OneDeltaLivePricing
from tradeexecutor.ethereum.one_delta.one_delta_routing import OneDeltaRouting
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_live_pricing import UniswapV2LivePricing
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_routing import UniswapV2Routing
from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_live_pricing import UniswapV3LivePricing
from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_routing import UniswapV3Routing
from tradeexecutor.ethereum.universe import create_exchange_universe, create_pair_universe
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.strategy.execution_context import unit_test_execution_context
from tradeexecutor.strategy.generic.generic_router import GenericRouting
from tradeexecutor.strategy.generic.generic_pricing_model import GenericPricing
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, load_partial_data
from tradeexecutor.strategy.universe_model import default_universe_options


@pytest.fixture()
def exchange_universe(web3, uniswap_v3_deployment: UniswapV3Deployment) -> ExchangeUniverse:
    """We trade on one uniswap v3 deployment on tester."""
    return create_exchange_universe(web3, [uniswap_v3_deployment])


@pytest.fixture()
def strategy_universe(
        chain_id,
        exchange_universe,
        asset_usdc,
        persistent_test_client
) -> TradingStrategyUniverse:
    """Universe that also contains data about our reserve assets."""

    pairs = [
        (ChainId.polygon, "uniswap-v3", "WETH", "USDC", 0.0005),
        (ChainId.polygon, "quickswap", "WMATIC", "USDC", 0.0030),
    ]

    reverses = [
        (ChainId.polygon, LendingProtocolType.aave_v3, "WETH"),
        (ChainId.polygon, LendingProtocolType.aave_v3, "USDC"),
    ]

    dataset = load_partial_data(
        persistent_test_client,
        execution_context=unit_test_execution_context,
        time_bucket=TimeBucket.d1,
        pairs=pairs,
        universe_options=default_universe_options,
        start_at=pd.Timestamp("2023-10-01"),
        end_at=pd.Timestamp("2023-10-30"),
        lending_reserves=reverses,
    )

    # Convert loaded data to a trading pair universe
    return TradingStrategyUniverse.create_from_dataset(dataset, asset_usdc.address)


@pytest.fixture()
def generic_routing_model(
    quickswap_deployment: UniswapV2Deployment,
    wmatic_usdc_spot_pair: TradingPairIdentifier,
    quickswap_routing_model: UniswapV2Routing,
    one_delta_routing_model: OneDeltaRouting,
    uniswap_v3_routing_model: UniswapV3Routing,
    asset_usdc: AssetIdentifier,
    asset_wmatic: AssetIdentifier,
) -> GenericRouting:
    """Create a routing model that trades Uniswap v2, v3 and 1delta + Aave.

    Live Polygon deployment addresses.
    """

    # Uses default router choose function
    return GenericRouting(
        routers={
            "quickswap": quickswap_routing_model,
            "uniswap-v3": uniswap_v3_routing_model,
            "1delta": one_delta_routing_model,
        }
    )


@pytest.fixture()
def generic_pricing_model(
    web3,
    strategy_universe: TradingStrategyUniverse,
    quickswap_routing_model: UniswapV2Routing,
    one_delta_routing_model: OneDeltaRouting,
    uniswap_v3_routing_model: UniswapV3Routing,
) -> GenericPricing:
    """Create a routing model that trades Uniswap v2, v3 and 1delta + Aave.

    Live Polygon deployment addresses.
    """

    pair_universe = strategy_universe.data_universe.pairs

    quickswap_pricing_model = UniswapV2LivePricing(web3, pair_universe, quickswap_routing_model)
    uniswap_v3_pricing_model = UniswapV3LivePricing(web3, pair_universe, uniswap_v3_routing_model)
    one_delta_pricing = OneDeltaLivePricing(web3, pair_universe, one_delta_routing_model)

    # Uses default router choose function
    return GenericPricing(
        pair_universe,
        routes = {
            "quickswap": quickswap_pricing_model,
            "uniswap-v3": uniswap_v3_pricing_model,
            "1delta": one_delta_pricing,
        }
    )


@pytest.fixture()
def execution_model(
        web3,
        hot_wallet: HotWallet,
        exchange_universe: ExchangeUniverse,
        weth_usdc_spot_pair,
) -> EthereumExecution:
    """Set EthereumExecutionModel in mainnet fork testing mode."""
    execution_model = EthereumExecution(
        HotWalletTransactionBuilder(web3, hot_wallet),
        mainnet_fork=True,
        confirmation_block_count=0,
    )
    return execution_model