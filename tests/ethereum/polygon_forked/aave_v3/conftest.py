"""Fixtures to set up Polygon fork generic router for uniswap v2 + uniswap v3 + aave v3"""

import pandas as pd
import pytest as pytest
from web3 import Web3

from tradeexecutor.ethereum.ethereum_protocol_adapters import EthereumPairConfigurator
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.strategy.generic.generic_valuation import GenericValuation
from tradingstrategy.chain import ChainId

from eth_defi.uniswap_v3.deployment import UniswapV3Deployment
from tradingstrategy.exchange import ExchangeUniverse
from tradingstrategy.lending import LendingProtocolType
from tradingstrategy.timebucket import TimeBucket
from tradeexecutor.ethereum.universe import create_exchange_universe
from tradeexecutor.strategy.execution_context import unit_test_execution_context
from tradeexecutor.strategy.generic.generic_router import GenericRouting
from tradeexecutor.strategy.generic.generic_pricing_model import GenericPricing
from tradeexecutor.strategy.generic.pair_configurator import ProtocolRoutingId
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, load_partial_data
from tradeexecutor.strategy.universe_model import default_universe_options


class TestForkedPolygonPairConfigurator(EthereumPairConfigurator):
    """Override default EthereumPairConfigurator to force Aave v3 routing"""
    def match_router(self, pair: TradingPairIdentifier) -> ProtocolRoutingId:
        if pair.is_credit_supply():
            return ProtocolRoutingId(
                router_name="aave-v3",
                lending_protocol_slug="aave_v3",
            )    
        
        return super().match_router(pair)


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
        start_at=pd.Timestamp("2023-12-01"),
        end_at=pd.Timestamp("2023-12-15"),
        lending_reserves=reverses,
    )

    # Convert loaded data to a trading pair universe
    return TradingStrategyUniverse.create_from_dataset(dataset, asset_usdc.address)


@pytest.fixture()
def pair_configurator(
    web3: Web3,
    strategy_universe: TradingStrategyUniverse,
) -> EthereumPairConfigurator:
    return TestForkedPolygonPairConfigurator(
        web3,
        strategy_universe,
    )


@pytest.fixture()
def generic_routing_model(pair_configurator) -> GenericRouting:
    return GenericRouting(pair_configurator)


@pytest.fixture()
def generic_pricing_model(pair_configurator) -> GenericPricing:
    return GenericPricing(pair_configurator)


@pytest.fixture()
def generic_valuation_model(pair_configurator) -> GenericValuation:
    return GenericValuation(pair_configurator)
