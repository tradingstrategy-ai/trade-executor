"""Fixtures to set up Polygon fork generic router for uniswap v2 + uniswap v3 + aave v3"""

import pandas as pd
import pytest as pytest
from web3 import Web3

from tradeexecutor.ethereum.ethereum_protocol_adapters import EthereumPairConfigurator
from tradeexecutor.ethereum.execution import EthereumExecution
from tradeexecutor.ethereum.tx import HotWalletTransactionBuilder
from tradeexecutor.strategy.generic.generic_valuation import GenericValuation
from tradingstrategy.chain import ChainId

from eth_defi.hotwallet import HotWallet
from eth_defi.uniswap_v2.deployment import UniswapV2Deployment
from eth_defi.uniswap_v3.deployment import UniswapV3Deployment
from tradingstrategy.exchange import ExchangeUniverse
from tradingstrategy.lending import LendingProtocolType
from tradingstrategy.timebucket import TimeBucket
from tradeexecutor.ethereum.one_delta.one_delta_live_pricing import OneDeltaLivePricing
from tradeexecutor.ethereum.aave_v3.aave_v3_routing import AaveV3Routing
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
        (ChainId.polygon, "uniswap-v3", "WMATIC", "USDC", 0.0005),
    ]

    reverses = [
        (ChainId.polygon, LendingProtocolType.aave_v3, "WETH"),
        (ChainId.polygon, LendingProtocolType.aave_v3, "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"),
        (ChainId.polygon, LendingProtocolType.aave_v3, "WMATIC"),
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


# @pytest.fixture()
# def aave_v3_routing_model(
#     aave_v3_deployment,
#     uniswap_v3_deployment,
#     asset_usdc,
#     asset_weth,
# ) -> AaveV3Routing:
#     return AaveV3Routing(
#         address_map={
#             "aave_v3_pool": aave_v3_deployment.pool.address,
#             "aave_v3_data_provider": aave_v3_deployment.data_provider.address,
#             "aave_v3_oracle": aave_v3_deployment.oracle.address,
#             "factory": uniswap_v3_deployment.factory.address,
#             "router": uniswap_v3_deployment.swap_router.address,
#             "position_manager": uniswap_v3_deployment.position_manager.address,
#             "quoter": uniswap_v3_deployment.quoter.address
#         },
#         allowed_intermediary_pairs={},
#         reserve_token_address=asset_usdc.address.lower(),
#     )



@pytest.fixture()
def pair_configurator(
    web3: Web3,
    strategy_universe: TradingStrategyUniverse,
) -> EthereumPairConfigurator:
    return EthereumPairConfigurator(
        web3,
        strategy_universe,
    )


@pytest.fixture()
def generic_routing_model(pair_configurator) -> GenericRouting:
    """Create a routing model that trades Uniswap v2, v3 and 1delta + Aave.

    Live Polygon deployment addresses.
    """

    # Uses default router choose function
    return GenericRouting(pair_configurator)


@pytest.fixture()
def generic_pricing_model(pair_configurator) -> GenericPricing:
    """Create a routing model that trades Uniswap v2, v3 and 1delta + Aave.

    Live Polygon deployment addresses.
    """
    # Uses default router choose function
    return GenericPricing(pair_configurator)


@pytest.fixture()
def generic_valuation_model(pair_configurator) -> GenericValuation:
    return GenericValuation(pair_configurator)
