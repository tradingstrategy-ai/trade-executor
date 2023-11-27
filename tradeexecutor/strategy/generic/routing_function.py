from typing import Protocol

from eth_typing import ChainId
from tradingstrategy.pair import PandasPairUniverse

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.trade import TradeExecution


class UnroutableTrade(Exception):
    """Trade cannot be routed, as we could not find a matching route."""


class RoutingFunction(Protocol):
    """A function protocol definition for router choose.

    """

    def __call__(
            self,
            pair_universe: PandasPairUniverse,
            pair: TradingPairIdentifier,
    ) -> str | None:
        """For each trade, return the name of the route that should be used.

        :param pair_univerwse:
            Give us loaded exchange and pair data to route the trade.

        :param pair:
            The trading pair for which we need to find the backend.

        :return:
            The route name that should be taken.

            If we do not know how to route the trade, return ``None``.
        """


def default_route_chooser(
    pair_universe: PandasPairUniverse,
    pair: TradingPairIdentifier,
) -> str | None:
    """Default router function.

    Comes with some DEXes and protocols prebuilt.

    Use smart contract addresses hardcoded in :py:mod:`tradeexecutor.ethereum.routing_data`.

    """
    exchange = pair_universe.exchange_universe.get_by_chain_and_factory(
        ChainId(pair.chain_id),
        pair.exchange_address
    )
    return exchange.exchange_slug
