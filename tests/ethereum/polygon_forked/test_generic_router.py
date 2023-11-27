"""Test live routing of combined Uniswap v2, v3 spot and 1delta leveraged positions."""
import datetime
from _decimal import Decimal

import pandas as pd
import pytest as pytest
from eth_typing import ChainId
from web3 import Web3

from eth_defi.hotwallet import HotWallet
from eth_defi.uniswap_v2.deployment import UniswapV2Deployment
from eth_defi.uniswap_v3.deployment import UniswapV3Deployment
from tradingstrategy.exchange import ExchangeUniverse
from tradingstrategy.lending import LendingProtocolType
from tradingstrategy.pair import PandasPairUniverse
from tradingstrategy.timebucket import TimeBucket
from tradeexecutor.ethereum.hot_wallet_sync_model import HotWalletSyncModel
from tradeexecutor.ethereum.one_delta.one_delta_live_pricing import OneDeltaLivePricing
from tradeexecutor.ethereum.one_delta.one_delta_routing import OneDeltaSimpleRoutingModel
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_live_pricing import UniswapV2LivePricing
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_routing import UniswapV2SimpleRoutingModel
from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_live_pricing import UniswapV3LivePricing
from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_routing import UniswapV3SimpleRoutingModel
from tradeexecutor.ethereum.universe import create_exchange_universe, create_pair_universe
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.strategy.execution_context import unit_test_execution_context
from tradeexecutor.strategy.generic.generic_router import GenericRouting
from tradeexecutor.strategy.generic.generic_pricing_model import GenericPricingModel
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
    quickswap_routing_model: UniswapV2SimpleRoutingModel,
    one_delta_routing_model: OneDeltaSimpleRoutingModel,
    uniswap_v3_routing_model: UniswapV3SimpleRoutingModel,
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
    pair_universe: PandasPairUniverse,
    quickswap_routing_model: UniswapV2SimpleRoutingModel,
    one_delta_routing_model: OneDeltaSimpleRoutingModel,
    uniswap_v3_routing_model: UniswapV3SimpleRoutingModel,
) -> GenericPricingModel:
    """Create a routing model that trades Uniswap v2, v3 and 1delta + Aave.

    Live Polygon deployment addresses.
    """

    quickswap_pricing_model = UniswapV2LivePricing(web3, pair_universe, quickswap_routing_model)
    uniswap_v3_pricing_model = UniswapV3LivePricing(web3, pair_universe, uniswap_v3_routing_model)
    one_delta_pricing = OneDeltaLivePricing(web3, pair_universe, one_delta_routing_model)

    # Uses default router choose function
    return GenericPricingModel(
        pair_universe,
        routes = {
            "quickswap": quickswap_pricing_model,
            "uniswap-v3": uniswap_v3_pricing_model,
            "1delta": one_delta_pricing,
        }
    )


def test_generic_routing_open_position_across_markets(
    web3: Web3,
    hot_wallet: HotWallet,
    pair_universe: PandasPairUniverse,
    generic_routing_model: GenericRouting,
    generic_pricing_model: GenericPricingModel,
    asset_usdc: AssetIdentifier,
):
    """Open Uniswap v2, v3 and 1delta position in the same state."""

    # Check that our preflight checks pass
    generic_routing_model.perform_preflight_checks_and_logging(pair_universe)

    sync_model = HotWalletSyncModel(
        web3,
        hot_wallet,
    )

    state = State()
    sync_model.sync_initial(state)

    # Strategy has its reserve balances updated
    sync_model.sync_treasury(datetime.datetime.utcnow(), state, supported_reserves=[asset_usdc])

    assert state.portfolio.get_reserve_position(asset_usdc).quantity == Decimal('10_000')


