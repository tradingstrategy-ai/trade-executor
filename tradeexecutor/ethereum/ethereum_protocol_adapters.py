from typing import Set

from web3 import Web3

from tradeexecutor.ethereum.one_delta.one_delta_live_pricing import OneDeltaLivePricing
from tradeexecutor.ethereum.one_delta.one_delta_routing import OneDeltaRouting
from tradeexecutor.ethereum.one_delta.one_delta_valuation import OneDeltaPoolRevaluator
from tradeexecutor.ethereum.routing_data import uniswap_v3_address_map, create_uniswap_v2_compatible_routing
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_live_pricing import UniswapV2LivePricing
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_valuation import UniswapV2PoolRevaluator
from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_live_pricing import UniswapV3LivePricing
from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_routing import UniswapV3Routing
from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_valuation import UniswapV3PoolRevaluator
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.generic.pair_configurator import PairConfigurator
from tradeexecutor.strategy.generic.routing_function import LiveProtocolAdapter, DeFiTradingPairConfig, UnroutableTrade
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import ExchangeUniverse, ExchangeType, ExchangeNotFoundError


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
    pair: TradingPairIdentifier,
) -> DeFiTradingPairConfig:

    assert len(strategy_universe.data_universe.chains) == 1
    assert len(strategy_universe.reserve_assets) == 1

    reserve = strategy_universe.reserve_assets[0]
    assert reserve.token_symbol == "USDC"

    exchange_universe = strategy_universe.data_universe.exchange_universe
    exchange_type = get_exchange_type(exchange_universe, pair)
    assert exchange_type == ExchangeType.uniswap_v2

    exchange = exchange_universe.get_by_chain_and_factory(ChainId(pair.chain_id), pair.exchange_address)

    if exchange.exchange_slug == "quickswap":
        routing_model = create_uniswap_v2_compatible_routing(
            TradeRouting.quickswap_usdc,
            ReserveCurrency.usdc,
        )
    else:
        raise NotImplementedError(f"Exchange not yet supported: {exchange}")

    pricing_model = UniswapV2LivePricing(
        web3,
        strategy_universe.data_universe.pairs,
        routing_model,
    )

    valuation_model = UniswapV2PoolRevaluator(pricing_model)

    return DeFiTradingPairConfig(
        router_name="uniswap-v2",
        pair=pair,
        pricing_model=pricing_model,
        valuation_model=valuation_model,
    )


def create_uniswap_v3_adapter(
    web3: Web3,
    strategy_universe: TradingStrategyUniverse,
    pair: TradingPairIdentifier,
) -> DeFiTradingPairConfig:
    """Always the same."""

    assert len(strategy_universe.data_universe.chains) == 1
    assert len(strategy_universe.reserve_assets) == 1

    exchange_universe = strategy_universe.data_universe.exchange_universe
    exchange_type = get_exchange_type(exchange_universe, pair)
    assert exchange_type == ExchangeType.uniswap_v3

    # TODO: Add intermediate tokens
    routing_model = UniswapV3Routing(
        uniswap_v3_address_map=uniswap_v3_address_map,
        chain_id=strategy_universe.data_universe.chains[0],
        reserve_token_address=strategy_universe.reserve_assets[0].address,
    )

    pricing_model = UniswapV3LivePricing(
        web3,
        strategy_universe.data_universe.pairs,
        routing_model,
    )

    valuation_model = UniswapV3PoolRevaluator(pricing_model)

    return DeFiTradingPairConfig(
        router_name="uniswap-v3",
        pair=pair,
        pricing_model=pricing_model,
        valuation_model=valuation_model,
    )



def create_1delta_adapter(
    web3: Web3,
    strategy_universe: TradingStrategyUniverse,
    pair: TradingPairIdentifier,
) -> DeFiTradingPairConfig:

    assert pair.is_leverage()
    assert len(strategy_universe.data_universe.chains) == 1
    assert len(strategy_universe.reserve_assets) == 1

    exchange_universe = strategy_universe.data_universe.exchange_universe
    exchange_type = get_exchange_type(exchange_universe, pair.underlying_spot_pair)
    assert exchange_type == ExchangeType.uniswap_v3

    # TODO: Add intermediate tokens
    routing_model = OneDeltaRouting(
        uniswap_v3_address_map=uniswap_v3_address_map,
        chain_id=strategy_universe.data_universe.chains[0],
        reserve_token_address=strategy_universe.reserve_assets[0].address,
    )

    pricing_model = OneDeltaLivePricing(
        web3,
        strategy_universe.data_universe.pairs,
        routing_model,
    )

    valuation_model = OneDeltaPoolRevaluator(pricing_model)

    return DeFiTradingPairConfig(
        router_name="1delta",
        pair=pair,
        pricing_model=pricing_model,
        valuation_model=valuation_model,
    )


class EthereumPairConfigurator(PairConfigurator):

    def get_supported_routers(self) -> Set[str]:
        exchanges = self.strategy_universe.data_universe.exchange_universe
        assert exchanges.get_exchange_count() < 5, "Exchanges might not be configured correctly"
        slugs = {e.exchange_slug for e in exchanges.exchanges.values()}
        return slugs | {"1delta"}   # TODO: Have 1delta variations

    def create_pair_config(self, pair: TradingPairIdentifier):
        exchange_type = get_exchange_type(self.strategy_universe.data_universe.exchange_universe, pair)
        if pair.is_leverage():
            return create_1delta_adapter(self.web3, self.strategy_universe, pair)
        elif exchange_type == ExchangeType.uniswap_v2:
            return create_uniswap_v2_adapter(self.web3, self.strategy_universe, pair)
        elif exchange_type == ExchangeType.uniswap_v3:
            return create_uniswap_v3_adapter(self.web3, self.strategy_universe, pair)
        else:
            raise NotImplementedError(f"Cannot route exchange {exchange_type} for pair {pair}")

    def create_router(self, router_name: str):
        strategy_universe = self.strategy_universe
        chain_id = strategy_universe.get_single_chain()
        reserve_asset = strategy_universe.get_reserve_asset()

        if router_name == "1delta":
            # TODO: Add intermediate tokens
            return OneDeltaRouting(
                address_map=uniswap_v3_address_map,
                chain_id=chain_id,
                allowed_intermediary_pairs={},
                reserve_token_address=reserve_asset.address,
            )
        else:
            exchange = self.strategy_universe.data_universe.exchange_universe.get_by_chain_and_slug(chain_id, router_name)
            if exchange.exchange_type == ExchangeType.uniswap_v2:
                if exchange.exchange_slug == "quickswap":
                    return create_uniswap_v2_compatible_routing(
                        TradeRouting.quickswap_usdc,
                        ReserveCurrency.usdc,
                    )
                else:
                    raise NotImplementedError(f"Exchange not yet supported: {exchange}")
            else:
                return  UniswapV3Routing(
                    address_map=uniswap_v3_address_map,
                    allowed_intermediary_pairs={},
                    chain_id=chain_id,
                    reserve_token_address=reserve_asset.address,
                )

    def match_router(self, pair: TradingPairIdentifier):

        if pair.is_leverage() or pair.is_credit_supply():
            return "1delta"

        pair_universe = self.strategy_universe.data_universe.pairs

        try:
            exchange = pair_universe.exchange_universe.get_by_chain_and_factory(
                ChainId(pair.chain_id),
                pair.exchange_address
            )
        except ExchangeNotFoundError as e:
            raise UnroutableTrade(
                f"Could not find exchange for pair: {pair}, exchange address {pair.exchange_address}.\n"
                f"We have data for {pair_universe.exchange_universe.get_exchange_count()} exchanges.\n"
            ) from e

        assert exchange is not None, \
            f"Loaded exchange data does not have exchange for pair {pair}, exchange address {pair.exchange_address}\n" \
            f"We have data for {pair_universe.exchange_universe.get_exchange_count()} exchanges"

        return exchange.exchange_slug