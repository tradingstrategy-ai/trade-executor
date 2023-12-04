from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Set

from web3 import Web3

from tradeexecutor.state.identifier import TradingPairIdentifier, ExchangeType
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.routing import RoutingModel
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.strategy.valuation import ValuationModel


@dataclass
class ProtocolRoutingId:
    router_name: str
    exchange_slug: str
    lending_protocol_slug: str | None = None

    def __hash__(self):
        return hash((self.router_name, self.exchange_slug, self.lending_protocol_slug))

    def __eq__(self, other):
        return other.router_name == self.router_name \
            and other.exchange_slug == self.exchange_slug \
            and other.lending_protocol_slug == self.lending_protocol_slug

    def __str__(self):
        return f"<ProtocolRoutingId {self.router_name} {self.exchange_slug} {self.lending_protocol_slug}>"


@dataclass
class ProtocolRoutingConfig:
    """Different components we need to deal trading on a protocol.

    - These are per-pair
    """
    routing_id: ProtocolRoutingId
    routing_model: RoutingModel
    valuation_model: ValuationModel
    pricing_model: PricingModel



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
        self.configs = {}

    @abstractmethod
    def create_config(self, routing_id: ProtocolRoutingId):
        pass

    @abstractmethod
    def get_supported_routers(self) -> Set[ProtocolRoutingId]:
        pass

    @abstractmethod
    def match_router(self, pair: TradingPairIdentifier) -> ProtocolRoutingId:
        pass

    def get_routing(self, pair: TradingPairIdentifier) -> RoutingModel:
        router = self.match_router(pair)
        return self.get_config(router).routing_model

    def get_valuation(self, pair: TradingPairIdentifier) -> ValuationModel:
        router = self.match_router(pair)
        return self.get_config(router).valuation_model

    def get_pricing(self, pair: TradingPairIdentifier) -> PricingModel:
        router = self.match_router(pair)
        return self.get_config(router).pricing_model

    def get_config(self, router: ProtocolRoutingId) -> ProtocolRoutingConfig:
        """Get cached config."""

        config = self.configs.get(router)
        if config is None:
            config = self.configs[router] = self.create_config(router)

        return config
