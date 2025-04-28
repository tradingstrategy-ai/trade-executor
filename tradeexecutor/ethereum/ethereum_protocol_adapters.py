"""Default protocols support for EVM blockchains.

See :py:mod:`tradeexecutor.strategy.generic.pair_configurator`.
"""

import logging
from typing import Set

from web3 import Web3
from eth_defi.aave_v3.deployment import fetch_deployment as fetch_aave_deployment
from eth_defi.uniswap_v3.deployment import fetch_deployment as fetch_uniswap_v3_deployment
from eth_defi.one_delta.deployment import fetch_deployment as fetch_1delta_deployment
from eth_defi.aave_v3.constants import AAVE_V3_DEPLOYMENTS
from eth_defi.one_delta.constants import ONE_DELTA_DEPLOYMENTS
from tradeexecutor.ethereum.routing_data import base_uniswap_v3_address_map

from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import ExchangeUniverse, ExchangeType

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.generic.pair_configurator import PairConfigurator, ProtocolRoutingId, ProtocolRoutingConfig
from tradeexecutor.strategy.generic.default_protocols import default_match_router, default_supported_routers
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


logger = logging.getLogger(__name__)


def get_exchange_type(
    exchange_universe: ExchangeUniverse,
    pair: TradingPairIdentifier,
) -> ExchangeType:
    assert pair.exchange_address is not None, f"Pair missing exchange_address: {pair}"
    exchange = exchange_universe.get_by_chain_and_factory(ChainId(pair.chain_id), pair.exchange_address)
    assert exchange is not None, f"Exchange address {pair.exchange_address} for pair {pair}: data not loaded"
    return exchange.exchange_type


def create_uniswap_v2_adapter(
    web3: Web3,
    strategy_universe: TradingStrategyUniverse,
    routing_id: ProtocolRoutingId,
) -> ProtocolRoutingConfig:

    # TODO: Avoid circular imports for now
    from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_live_pricing import UniswapV2LivePricing
    from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_valuation import UniswapV2PoolRevaluator
    from tradeexecutor.ethereum.routing_data import create_uniswap_v2_compatible_routing

    logger.info("create_uniswap_v2_adapter(): %s", routing_id)

    assert routing_id.router_name == "uniswap-v2"
    assert len(strategy_universe.data_universe.chains) == 1
    assert len(strategy_universe.reserve_assets) == 1

    reserve = strategy_universe.get_reserve_asset()
    assert reserve.token_symbol in ("USDC", "USDT",)

    exchange_universe = strategy_universe.data_universe.exchange_universe
    chain_id = strategy_universe.get_single_chain()

    exchange = exchange_universe.get_by_chain_and_slug(chain_id, routing_id.exchange_slug)

    if exchange.exchange_slug == "quickswap":
        routing_model = create_uniswap_v2_compatible_routing(
            TradeRouting.quickswap_usdc,
            ReserveCurrency.usdc,
            chain_id,
        )
    elif exchange.exchange_slug == "sushi":
        routing_model = create_uniswap_v2_compatible_routing(
            TradeRouting.sushi_usdc,
            ReserveCurrency.usdc,
            chain_id,
        )
    elif exchange.exchange_slug == "pancakeswap-v2":
        routing_model = create_uniswap_v2_compatible_routing(
            TradeRouting.pancakeswap_usdt,
            ReserveCurrency.usdt,
            chain_id,
        )
    else:
        assert exchange.exchange_slug == "uniswap-v2", f"Expected uniswap-v2, got exchange slug {exchange.exchange_slug}"
        routing_model = create_uniswap_v2_compatible_routing(
            TradeRouting.uniswap_v2_usdc,
            ReserveCurrency.usdc,
            chain_id,
        )

    pricing_model = UniswapV2LivePricing(
        web3,
        strategy_universe.data_universe.pairs,
        routing_model,
    )

    valuation_model = UniswapV2PoolRevaluator(pricing_model)

    return ProtocolRoutingConfig(
        routing_id=routing_id,
        routing_model=routing_model,
        pricing_model=pricing_model,
        valuation_model=valuation_model,
    )


def create_uniswap_v3_adapter(
    web3: Web3,
    strategy_universe: TradingStrategyUniverse,
    routing_id: ProtocolRoutingId,
) -> ProtocolRoutingConfig:
    """Always the same."""

    # TODO: Avoid circular imports for now
    from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_live_pricing import UniswapV3LivePricing
    from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_routing import UniswapV3Routing
    from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_valuation import UniswapV3PoolRevaluator
    from tradeexecutor.ethereum.routing_data import uniswap_v3_address_map

    assert routing_id.router_name == "uniswap-v3"
    assert len(strategy_universe.data_universe.chains) == 1
    assert len(strategy_universe.reserve_assets) == 1
    chain_id = strategy_universe.get_single_chain()
    reserve_asset = strategy_universe.get_reserve_asset()

    allowed_intermediary_pairs = UNISWAP_V3_INTERMEDIATE.get(chain_id, {})

    if chain_id == ChainId.base:
        # Special case for Base chain
        address_map = base_uniswap_v3_address_map
    else:
        address_map = uniswap_v3_address_map

    # TODO: Add intermediate tokens
    routing_model = UniswapV3Routing(
        address_map=address_map,
        chain_id=chain_id,
        reserve_token_address=reserve_asset.address,
        allowed_intermediary_pairs=allowed_intermediary_pairs,
    )

    pricing_model = UniswapV3LivePricing(
        web3,
        strategy_universe.data_universe.pairs,
        routing_model,
    )

    valuation_model = UniswapV3PoolRevaluator(pricing_model)

    return ProtocolRoutingConfig(
        routing_id=routing_id,
        routing_model=routing_model,
        pricing_model=pricing_model,
        valuation_model=valuation_model,
    )


def create_1delta_adapter(
    web3: Web3,
    strategy_universe: TradingStrategyUniverse,
    routing_id: ProtocolRoutingId,
) -> ProtocolRoutingConfig:

    # TODO: Avoid circular imports for now
    from tradeexecutor.ethereum.one_delta.one_delta_live_pricing import OneDeltaLivePricing
    from tradeexecutor.ethereum.one_delta.one_delta_routing import OneDeltaRouting
    from tradeexecutor.ethereum.one_delta.one_delta_valuation import OneDeltaPoolRevaluator


    assert routing_id.router_name == "1delta"
    assert routing_id.lending_protocol_slug == "aave"
    assert routing_id.exchange_slug == "uniswap-v3"

    assert len(strategy_universe.data_universe.chains) == 1
    assert len(strategy_universe.reserve_assets) == 1
    chain_id = strategy_universe.get_single_chain()
    chain_slug = chain_id.get_slug()
    reserve_asset = strategy_universe.get_reserve_asset()

    assert chain_slug in AAVE_V3_DEPLOYMENTS, f"Chain {chain_slug} not supported for Aave v3"
    assert chain_slug in ONE_DELTA_DEPLOYMENTS, f"Chain {chain_slug} not supported for 1delta"

    uniswap_v3_deployment = fetch_uniswap_v3_deployment(
        web3,
        "0x1F98431c8aD98523631AE4a59f267346ea31F984",
        "0xE592427A0AEce92De3Edee1F18E0157C05861564",
        "0xC36442b4a4522E871399CD717aBDD847Ab11FE88",
        "0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6",
    )

    aave_v3_deployment = fetch_aave_deployment(
        web3,
        pool_address=AAVE_V3_DEPLOYMENTS[chain_slug]["pool"],
        data_provider_address=AAVE_V3_DEPLOYMENTS[chain_slug]["data_provider"],
        oracle_address=AAVE_V3_DEPLOYMENTS[chain_slug]["oracle"],
    )

    one_delta_deployment = fetch_1delta_deployment(
        web3,
        flash_aggregator_address=ONE_DELTA_DEPLOYMENTS[chain_slug]["broker_proxy"],
        broker_proxy_address=ONE_DELTA_DEPLOYMENTS[chain_slug]["broker_proxy"],
        quoter_address=ONE_DELTA_DEPLOYMENTS[chain_slug]["quoter"],
    )

    address_map = {
        "one_delta_broker_proxy": one_delta_deployment.broker_proxy.address,
        "one_delta_quoter": one_delta_deployment.quoter.address,
        "aave_v3_pool": aave_v3_deployment.pool.address,
        "aave_v3_data_provider": aave_v3_deployment.data_provider.address,
        "aave_v3_oracle": aave_v3_deployment.oracle.address,
        "factory": uniswap_v3_deployment.factory.address,
        "router": uniswap_v3_deployment.swap_router.address,
        "position_manager": uniswap_v3_deployment.position_manager.address,
        "quoter": uniswap_v3_deployment.quoter.address
    }

    # TODO: Add intermediate tokens
    routing_model = OneDeltaRouting(
        address_map=address_map,
        chain_id=chain_id,
        reserve_token_address=reserve_asset.address,
        allowed_intermediary_pairs={},
    )

    pricing_model = OneDeltaLivePricing(
        web3,
        strategy_universe.data_universe.pairs,
        routing_model,
    )

    valuation_model = OneDeltaPoolRevaluator(pricing_model)

    return ProtocolRoutingConfig(
        routing_id=routing_id,
        routing_model=routing_model,
        pricing_model=pricing_model,
        valuation_model=valuation_model,
    )


def create_aave_v3_adapter(
    web3: Web3,
    strategy_universe: TradingStrategyUniverse,
    routing_id: ProtocolRoutingId,
) -> ProtocolRoutingConfig:

    # TODO: Avoid circular imports for now
    from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_live_pricing import UniswapV3LivePricing
    from tradeexecutor.ethereum.eth_valuation import EthereumPoolRevaluator
    from tradeexecutor.ethereum.aave_v3.aave_v3_routing import AaveV3Routing

    assert routing_id.router_name == "aave-v3"
    assert routing_id.lending_protocol_slug == "aave_v3"

    assert len(strategy_universe.data_universe.chains) == 1
    assert len(strategy_universe.reserve_assets) == 1
    chain_id = strategy_universe.get_single_chain()
    chain_slug = chain_id.get_slug()
    reserve_asset = strategy_universe.get_reserve_asset()

    assert chain_slug in AAVE_V3_DEPLOYMENTS, f"Chain {chain_slug} not supported for Aave v3"

    aave_v3_deployment = fetch_aave_deployment(
        web3,
        pool_address=AAVE_V3_DEPLOYMENTS[chain_slug]["pool"],
        data_provider_address=AAVE_V3_DEPLOYMENTS[chain_slug]["data_provider"],
        oracle_address=AAVE_V3_DEPLOYMENTS[chain_slug]["oracle"],
        ausdc_address=AAVE_V3_DEPLOYMENTS[chain_slug].get("ausdc"),
    )

    address_map = {
        "aave_v3_pool": aave_v3_deployment.pool.address,
        "aave_v3_data_provider": aave_v3_deployment.data_provider.address,
        "aave_v3_oracle": aave_v3_deployment.oracle.address,
    }

    routing_model = AaveV3Routing(
        address_map=address_map,
        chain_id=chain_id,
        reserve_token_address=reserve_asset.address,
        allowed_intermediary_pairs={},
    )

    pricing_model = UniswapV3LivePricing(
        web3,
        strategy_universe.data_universe.pairs,
        routing_model,
    )

    valuation_model = EthereumPoolRevaluator(pricing_model)

    return ProtocolRoutingConfig(
        routing_id=routing_id,
        routing_model=routing_model,
        pricing_model=pricing_model,
        valuation_model=valuation_model,
    )


class EthereumPairConfigurator(PairConfigurator):
    """Set up routes for EVM trading pairs.

    Supported protocols

    - 1delta

    - Uniswap v2 likes

    - Uniswap v3 likes

    - Aave v3
    """

    def __init__(
        self,
        web3: Web3,
        strategy_universe: TradingStrategyUniverse | None,
    ):
        """Initialise pair configuration.

        :param web3:
            Web3 connection to the active blockchain.

        :param strategy_universe:
            The initial strategy universe.

            TODO: Currently only reserve currency, exchange and pair data is used.
            Candle data is discarded.
        """

        assert isinstance(web3, Web3)
        assert isinstance(strategy_universe, TradingStrategyUniverse)

        self.web3 = web3

        super().__init__(strategy_universe)

    def get_supported_routers(self) -> Set[ProtocolRoutingId]:
        return default_supported_routers(self.strategy_universe)

    def create_config(self, routing_id: ProtocolRoutingId) -> ProtocolRoutingConfig:
        if routing_id.router_name == "1delta":
            return create_1delta_adapter(self.web3, self.strategy_universe, routing_id)
        elif routing_id.router_name == "uniswap-v2":
            return create_uniswap_v2_adapter(self.web3, self.strategy_universe, routing_id)
        elif routing_id.router_name == "uniswap-v3":
            return create_uniswap_v3_adapter(self.web3, self.strategy_universe, routing_id)
        elif routing_id.router_name == "aave-v3":
            return create_aave_v3_adapter(self.web3, self.strategy_universe, routing_id)
        else:
            raise NotImplementedError(f"Cannot route exchange {routing_id}")

    def match_router(self, pair: TradingPairIdentifier) -> ProtocolRoutingId:
        return default_match_router(self.strategy_universe, pair)


UNISWAP_V3_INTERMEDIATE = {
    ChainId.ethereum: {
        # Route WETH through WETH-USDC 5 BPS
        # https://tradingstrategy.ai/trading-view/ethereum/uniswap-v3/eth-usdc-fee-5
        "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",
    },

    ChainId.base: {
        # Route WETH through WETH-USDC 5 BPS
        # https://coinmarketcap.com/dexscan/base/0xd0b53d9277642d899df5c87a3966a349a798f224/
        "0x4200000000000000000000000000000000000006": "0xd0b53d9277642d899df5c87a3966a349a798f224",
    }
}
