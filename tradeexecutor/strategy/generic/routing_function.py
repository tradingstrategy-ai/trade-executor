"""Router functions.

- Interface for the route chooser function :py:class:`RoutingFunction`

- The default router choose :py:func:`default_route_chooser`
"""

from typing import Protocol

from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import ExchangeNotFoundError
from tradingstrategy.pair import PandasPairUniverse

from tradeexecutor.state.identifier import TradingPairIdentifier


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

    :return:
        Exchange slug if a spot market.

        ``1delta`` if we are doing a leveraged position.

    :raise UnroutableTrade:
        If we cannot figure out how to trade.

        Usually due to missing data.
    """

    assert isinstance(pair, TradingPairIdentifier), f"Expected TradingPairIdentifier, got {pair.__class__}"
    assert pair_universe.exchange_universe, f"PandasPairUniverse.exchange_universe not set"
    assert pair.exchange_address, f"Pair does not have exchange_address filled in: {pair}"

    if pair.is_leverage() or pair.is_credit_supply():
        return "1delta"

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
