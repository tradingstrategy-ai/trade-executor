from abc import abstractmethod, ABC
from typing import Set

from web3 import Web3

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.strategy.generic.routing_function import DeFiTradingPairConfig
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.routing import RoutingModel
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.strategy.valuation import ValuationModel


class PairConfigurator(ABC):
    """

    - Pricing and valuation models are per price

    - Routing model is per supported DeFi protocol

    - Routing state is per cycle


    """

    def __init__(
        self,
        web3: Web3,
        strategy_universe: TradingStrategyUniverse,
    ):
        self.web3 = web3
        self.strategy_universe = strategy_universe
        self.pair_configs = {}
        self.router_configs = {}

    @abstractmethod
    def create_pair_config(self, pair):
        pass

    @abstractmethod
    def create_router(self, router_name: str):
        pass

    @abstractmethod
    def match_router(self, pair: TradingPairIdentifier) -> str:
        pass

    @abstractmethod
    def get_supported_routers(self) -> Set[str]:
        pass

    def get_valuation(self, pair: TradingPairIdentifier) -> ValuationModel:
        return self.get_pair_config(pair).valuation_model

    def get_pricing(self, pair: TradingPairIdentifier) -> PricingModel:
        return self.get_pair_config(pair).pricing_model

    def get_pair_config(self, pair: TradingPairIdentifier) -> DeFiTradingPairConfig:
        """Get cached config."""

        config = self.pair_configs.get(pair)
        if config is None:
            config = self.pair_configs[pair] = self.create_pair_config(pair)

        return config

    def get_router(self, router_name: str) -> RoutingModel:
        """Get cached config."""

        config = self.router_configs.get(router_name)
        if config is None:
            config = self.router_configs[router_name] = self.create_router(router_name)

        return config





