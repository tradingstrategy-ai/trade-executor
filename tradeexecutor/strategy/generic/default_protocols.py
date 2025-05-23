"""Router functions.

- Interface for the route chooser function :py:class:`RoutingFunction`

- The default router choose :py:func:`default_route_chooser`
"""
from typing import Set

from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import ExchangeNotFoundError

from tradeexecutor.state.identifier import TradingPairIdentifier, ExchangeType
from tradeexecutor.strategy.generic.pair_configurator import UnroutableTrade, ProtocolRoutingId
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


def default_match_router(
    strategy_universe: TradingStrategyUniverse,
    pair: TradingPairIdentifier
) -> ProtocolRoutingId:
    """The default implementation of routing of protocols"""

    assert isinstance(pair, TradingPairIdentifier)

    if pair.is_leverage():
        return ProtocolRoutingId(
            router_name="1delta",
            exchange_slug="uniswap-v3",
            lending_protocol_slug="aave"
        )
    elif pair.is_credit_supply():
        # prefer 1delta whenever possible
        if pair.chain_id in [ChainId.polygon.value]:
            return ProtocolRoutingId(
                router_name="1delta",
                exchange_slug="uniswap-v3",
                lending_protocol_slug="aave"
            )
        else:
            return ProtocolRoutingId(
                router_name="aave-v3",
                lending_protocol_slug="aave_v3",
            )
    elif pair.is_vault():
        return ProtocolRoutingId(
            router_name="vault",
        )

    pair_universe = strategy_universe.data_universe.pairs

    assert pair_universe.exchange_universe, "exchange_universe attr not set in pair_universe"

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

    return ProtocolRoutingId(
        router_name="uniswap-v2" if exchange.exchange_type == ExchangeType.uniswap_v2 else "uniswap-v3",
        exchange_slug=exchange.exchange_slug,
    )


def default_supported_routers(strategy_universe: TradingStrategyUniverse) -> Set[ProtocolRoutingId]:
    """Default supported protocols.

    Read trading pairs and figure out what protocols we need to support,
    based on loaded trading pairs.
    """
    exchanges = strategy_universe.data_universe.exchange_universe
    chain_id = strategy_universe.get_single_chain()

    # Vaults count as exchanges, so multi vault strategy needs bump the number here
    assert exchanges.get_exchange_count() < 20, f"Exchanges might not be configured correctly, we have {exchanges.get_exchange_count()} exchanges"
    configs = set()

    vaults_done = False

    for xc in exchanges.exchanges.values():
        if xc.exchange_type == ExchangeType.erc_4626_vault:
            if not vaults_done:
                # All vaults use the same route
                configs.add(
                    ProtocolRoutingId(
                        router_name="vault",
                        exchange_slug=None,
                    )
                )
                vault_done = True
        else:
            configs.add(
                ProtocolRoutingId(
                    router_name="uniswap-v2" if xc.exchange_type == ExchangeType.uniswap_v2 else "uniswap-v3",
                    exchange_slug=xc.exchange_slug,
                )
            )

    # Enabled 1delta if lending candles are available
    if strategy_universe.data_universe.lending_candles:
        if chain_id == ChainId.polygon.value:
            configs.add(
                ProtocolRoutingId(
                    router_name="1delta",
                    exchange_slug="uniswap-v3",
                    lending_protocol_slug="aave",
                )
            )
        else:
            configs.add(
                ProtocolRoutingId(
                    router_name="aave-v3",
                    exchange_slug="uniswap-v3",
                    lending_protocol_slug="aave_v3",
                )
            )

    return configs
