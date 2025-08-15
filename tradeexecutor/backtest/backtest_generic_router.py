
from typing import Set

from tradeexecutor.backtest.backtest_pricing import BacktestPricing
from tradeexecutor.backtest.backtest_routing import BacktestRoutingIgnoredModel, BacktestRoutingModel
from tradeexecutor.backtest.backtest_valuation import BacktestValuationModel
from tradeexecutor.ethereum.routing_data import create_compatible_routing, get_reserve_currency_by_asset
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.generic.pair_configurator import PairConfigurator, ProtocolRoutingId, ProtocolRoutingConfig
from tradeexecutor.strategy.generic.default_protocols import default_match_router, default_supported_routers



class EthereumBacktestPairConfigurator(PairConfigurator):
    """Set up routes for EVM trading pairs in backtesting.

    See :py:class:`tradeexecutor.ethereum.ethereum_protocol_adapter.EthereumPairConfigurator`
    for live trading implementation.

    All routes

    Supported protocols

    - 1delta

    - Uniswap v2 likes

    - Uniswap v3 likes
    """


    def get_supported_routers(self) -> Set[ProtocolRoutingId]:
        return default_supported_routers(self.strategy_universe)

    def create_config(self, routing_id: ProtocolRoutingId):

        strategy_universe = self.strategy_universe

        # Resolve all different combinations of DEXes and stablecoins
        reserve = strategy_universe.get_reserve_asset()
        assert reserve.token_symbol in ("USDC", "USDT"), f"Expected USDT/USDC reserve, got {reserve.token_symbol}.\nTODO: Development assert. Please fix."

        # Get in parameters for our supported intermediate pairs
        reserve_currency = get_reserve_currency_by_asset(reserve)

        if routing_id.router_name == "vault":
            routing_model = BacktestRoutingIgnoredModel(
                reserve.address,
            )
        else:
            match routing_id.exchange_slug:
                case "uniswap-v2" | "pancakeswap-v2" | "my-dex":

                    if routing_id.exchange_slug in ("uniswap-v2", "my-dex"):
                        if reserve.token_symbol == "USDT":
                            routing_type = TradeRouting.uniswap_v2_usdt
                        else:
                            routing_type = TradeRouting.uniswap_v2_usdc
                    elif routing_id.exchange_slug == "pancakeswap-v2":
                        routing_type = TradeRouting.pancakeswap_usdt
                    else:
                        raise NotImplementedError(f"Unsupported Uniswap v2 routing {routing_id.router_name}")

                    real_routing_model = create_compatible_routing(routing_type, reserve_currency)
                    routing_model = BacktestRoutingModel(
                        real_routing_model.factory_router_map,
                        real_routing_model.allowed_intermediary_pairs,
                        real_routing_model.reserve_token_address,
                    )

                case "uniswap-v3":
                    if reserve.token_symbol == "USDT":
                        routing_type = TradeRouting.uniswap_v3_usdt
                    else:
                        routing_type = TradeRouting.uniswap_v3_usdc

                    real_routing_model = create_compatible_routing(routing_type, reserve_currency)

                    routing_model = BacktestRoutingModel(
                        factory_router_map={},
                        reserve_token_address=reserve.address,
                        allowed_intermediary_pairs=real_routing_model.allowed_intermediary_pairs,
                    )

                case _:
                    raise NotImplementedError(f"Unsupported exchange {routing_id}")

        pricing_model = BacktestPricing(
            strategy_universe.data_universe.candles,
            routing_model,
            data_delay_tolerance=self.data_delay_tolerance,
            liquidity_universe=strategy_universe.data_universe.liquidity,
            pairs=strategy_universe.data_universe.pairs,
        )

        valuation_model = BacktestValuationModel(pricing_model)

        return ProtocolRoutingConfig(
            routing_id=routing_id,
            routing_model=routing_model,
            pricing_model=pricing_model,
            valuation_model=valuation_model,
        )

    def match_router(self, pair: TradingPairIdentifier) -> ProtocolRoutingId:
        return default_match_router(self.strategy_universe, pair)
