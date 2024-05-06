
from typing import Set

from tradeexecutor.backtest.backtest_pricing import BacktestPricing
from tradeexecutor.backtest.backtest_routing import BacktestRoutingIgnoredModel
from tradeexecutor.backtest.backtest_valuation import BacktestValuationModel
from tradeexecutor.state.identifier import TradingPairIdentifier
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

        reserve = strategy_universe.reserve_assets[0]
        assert reserve.token_symbol in ("USDC", "USDT"), f"Expected USDT/USDC reserve, got {reserve.token_symbol}.\nTODO: Development assert. Please fix."

        routing_model = BacktestRoutingIgnoredModel(reserve.address)

        pricing_model = BacktestPricing(
            strategy_universe.data_universe.candles,
            routing_model,
            data_delay_tolerance=self.data_delay_tolerance,
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
