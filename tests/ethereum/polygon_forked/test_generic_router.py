"""Test live routing of combined Uniswap v2, v3 spot and 1delta leveraged positions."""
import pandas as pd
import pytest as pytest
from eth_typing import ChainId
from tradingstrategy.exchange import ExchangeUniverse
from tradingstrategy.lending import LendingProtocolType
from tradingstrategy.pair import PandasPairUniverse
from tradingstrategy.timebucket import TimeBucket

from eth_defi.uniswap_v2.deployment import UniswapV2Deployment
from eth_defi.uniswap_v3.deployment import UniswapV3Deployment
from tradeexecutor.ethereum.one_delta.one_delta_routing import OneDeltaSimpleRoutingModel
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_routing import UniswapV2SimpleRoutingModel
from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_routing import UniswapV3SimpleRoutingModel
from tradeexecutor.ethereum.universe import create_exchange_universe, create_pair_universe
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.strategy.execution_context import unit_test_execution_context
from tradeexecutor.strategy.generic_router import GenericRouting
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, load_partial_data
from tradeexecutor.strategy.universe_model import default_universe_options


@pytest.fixture()
def exchange_universe(web3, uniswap_v3_deployment: UniswapV3Deployment) -> ExchangeUniverse:
    """We trade on one uniswap v3 deployment on tester."""
    return create_exchange_universe(web3, [uniswap_v3_deployment])


@pytest.fixture()
def pair_universe(web3, exchange_universe: ExchangeUniverse, weth_usdc_spot_pair) -> PandasPairUniverse:
    exchange = next(iter(exchange_universe.exchanges.values()))
    return create_pair_universe(web3, exchange, [weth_usdc_spot_pair])


@pytest.fixture()
def trading_strategy_universe(chain_id, exchange_universe, pair_universe, asset_usdc, persistent_test_client) -> TradingStrategyUniverse:
    """Universe that also contains data about our reserve assets."""

    pairs = [
        (ChainId.polygon, "uniswap-v3", "WETH", "USDC", 0.0005),
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
    return TradingStrategyUniverse.create_single_pair_universe(dataset)


@pytest.fixture()
def generic_routing_model(
    quickswap_deployment: UniswapV2Deployment,
    wmatic_usdc_spot_pair: TradingPairIdentifier,
    one_delta_routing_model: OneDeltaSimpleRoutingModel,
    uniswap_v3_routing_model: UniswapV3SimpleRoutingModel,
    asset_usdc: AssetIdentifier,
    asset_wmatic: AssetIdentifier,
) -> GenericRouting:
    """Create a routing model that trades Uniswap v2, v3 and 1delta + Aave.

    Live Polygon deployment addresses.
    """

    # Route WMATIC and USDC quoted pairs on Quickswap
    uniswap_v2_router = UniswapV2SimpleRoutingModel(
        factory_router_map={quickswap_deployment.factory.address: (quickswap_deployment.router.address, quickswap_deployment.init_code_hash)},
        allowed_intermediary_pairs={wmatic_usdc_spot_pair.base.address: wmatic_usdc_spot_pair.pool_address},
        reserve_token_address=asset_usdc.address,
    )

    # Uses default router choose function
    return GenericRouting(
        routers={
            "quickswap": uniswap_v2_router,
            "uniswap-v3": uniswap_v3_routing_model,
            "1delta": one_delta_routing_model,
        }
    )


@pytest.fixture()
def generic_pricing_model(
    quickswap_deployment: UniswapV2Deployment,
    wmatic_usdc_spot_pair: TradingPairIdentifier,
    one_delta_routing_model: OneDeltaSimpleRoutingModel,
    uniswap_v3_routing_model: UniswapV3SimpleRoutingModel,
    asset_usdc: AssetIdentifier,
    asset_wmatic: AssetIdentifier,
) -> GenericRouting:
    """Create a routing model that trades Uniswap v2, v3 and 1delta + Aave.

    Live Polygon deployment addresses.
    """

    # Route WMATIC and USDC quoted pairs on Quickswap
    uniswap_v2_router = UniswapV2SimpleRoutingModel(
        factory_router_map={quickswap_deployment.factory.address: (quickswap_deployment.router.address, quickswap_deployment.init_code_hash)},
        allowed_intermediary_pairs={wmatic_usdc_spot_pair.base.address: wmatic_usdc_spot_pair.pool_address},
        reserve_token_address=asset_usdc.address,
    )

    # Uses default router choose function
    return GenericRouting(
        routers={
            "quickswap": uniswap_v2_router,
            "uniswap-v3": uniswap_v3_routing_model,
            "1delta": one_delta_routing_model,
        }
    )


def test_generic_routing_open_all_positions(
    web3,
    generic_routing_model,
):
    """Open Uniswap v2, v3 and 1delta position in the same state."""
