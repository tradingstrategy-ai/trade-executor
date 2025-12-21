"""Match trading pairs to their supported DeFI protocols."""

from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Set, Dict

import pandas as pd

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.routing import RoutingModel
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.strategy.valuation import ValuationModel
from tradingstrategy.pair import PandasPairUniverse


@dataclass
class ProtocolRoutingId:
    """Identify supported protocol for routing.

    Because we use composable :term:`DeFi` we need identify
    any combination of supported elements.
    """

    #: The type of routing.
    #:
    #: The major protocol used for trades.
    #:
    #: "uniswap-v2", "uniswap-v3" or "1delta" or "vault".
    #: Special id "cex" used in backtesting.
    router_name: str

    #: "quickswap" or "uniswap-v3" or "trader-joe"
    exchange_slug: str | None = None

    #: "aave"
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

    - These are a per supported protocol

    - This is initialised once in :py:meth`PairConfigurator.create_config`
    """

    #: Which protocol this config is for
    routing_id: ProtocolRoutingId

    #: Routing model to transform trades to blockchain transactions
    routing_model: RoutingModel

    #: Valuation model get the PnL of a position
    valuation_model: ValuationModel

    #: Estimate buy/sell price of an asset
    pricing_model: PricingModel



class PairConfigurator(ABC):
    """Match trading pairs to their routes.

    - Only applicable to live trading

    - Create routing configurations based on the loaded
      trading universe :py:meth:`get_supported_routers`.

    - Can read data and check smart contracts directly on a blockchain

    - See :py:class:tradeexecutor.ethereum.ethereum_protocol_adapters.EthereumPairConfigurator`
      for an implementation
    """

    def __init__(
        self,
        strategy_universe: TradingStrategyUniverse,
        data_delay_tolerance=pd.Timedelta("2d"),
    ):
        """Initialise pair configuration.

        :param strategy_universe:
            The initial strategy universe.

            TODO: Currently only reserve currency, exchange and pair data is used.
            Candle data is discarded.
        """

        assert isinstance(strategy_universe, TradingStrategyUniverse)
        self.strategy_universe = strategy_universe

        #: Our cached configs
        self.configs: Dict[ProtocolRoutingId, ProtocolRoutingConfig] = {}

        self.data_delay_tolerance = data_delay_tolerance  # See BacktestPricingModel

    @property
    def cross_chain(self) -> bool:
        """Are we trading cross-chain?

        :return:
            True if we are trading cross-chain
        """
        return self.strategy_universe.cross_chain

    @abstractmethod
    def create_config(self, routing_id: ProtocolRoutingId, three_leg_resolution=True, pairs=None) -> ProtocolRoutingConfig:
        """Create a protocol configuraiton

        - Initialise pricing, valuation and router models
          for a particular protocol

        - Called only once per process life cycle,
          created models are cached

        :param routing_id:
            The protocol we are initialising
        """
        pass

    @abstractmethod
    def get_supported_routers(self) -> Set[ProtocolRoutingId]:
        """Create supported routing options based on the loaded trading universe.

        :return:
            List of protocols we can handle in this trading universe
        """

    @abstractmethod
    def match_router(self, pair: TradingPairIdentifier) -> ProtocolRoutingId:
        """Map a trading pair to a supported protocol.

        :param pair:
            The trading pair.

            For spot, return `uniswap-v2` or `uniswap-v3` routing id.

            For leverage, return `1delta` routing id.

        :raise UnroutableTrade:
            In the case we do not have exchange data loaded in the trading universe to route the pair.
        """

    def get_routing(self, pair: TradingPairIdentifier) -> RoutingModel:
        """Get routing model for a pair.

        :return:
            Lazily created cached result.

        :raise UnroutableTrade:
            In the case we do not have exchange data loaded in the trading universe to route the pair.

        """
        router = self.match_router(pair)
        return self.get_config(router).routing_model

    def get_valuation(self, pair: TradingPairIdentifier) -> ValuationModel:
        """Get valuation model for a pair.

        :return:
            Lazily created cached result.

        :raise UnroutableTrade:
            In the case we do not have exchange data loaded in the trading universe to route the pair.

        """
        router = self.match_router(pair)
        return self.get_config(router).valuation_model

    def get_pricing(self, pair: TradingPairIdentifier) -> PricingModel:
        """Get pricing model for a pair.

        :return:
            Lazily created cached result.

        :raise UnroutableTrade:
            In the case we do not have exchange data loaded in the trading universe to route the pair.

        """
        router = self.match_router(pair)
        return self.get_config(router).pricing_model

    def get_config(
        self,
        router: ProtocolRoutingId,
        pairs: PandasPairUniverse | None = None,
        three_leg_resolution=True,
    ) -> ProtocolRoutingConfig:
        """Get cached config for a specific protocol.

        Lazily create the config if not yet available.

        :raise Exception:
            Various errors in the protocol set up code.
        """

        config = self.configs.get(router)
        if config is None:
            config = self.configs[router] = self.create_config(router, three_leg_resolution=three_leg_resolution, pairs=pairs)

        return config


class UnroutableTrade(Exception):
    """Trade cannot be routed, as we could not find a matching route.

    TODO: Refactor this exception to another module and remove the stub module.
    """
