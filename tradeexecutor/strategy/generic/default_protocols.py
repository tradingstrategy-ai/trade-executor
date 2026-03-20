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
        if pair.is_hyperliquid_vault():
            return ProtocolRoutingId(
                router_name="hypercore_vault",
            )
        return ProtocolRoutingId(
            router_name="vault",
        )
    elif pair.is_freqtrade():
        return ProtocolRoutingId(
            router_name="freqtrade",
        )
    elif pair.is_exchange_account():
        return ProtocolRoutingId(
            router_name="exchange_account",
        )
    elif pair.is_cctp_bridge():
        return ProtocolRoutingId(
            router_name="cctp-bridge",
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
    chain_id = strategy_universe.get_primary_chain()

    # Vaults count as exchanges, so multi vault strategy needs bump the number here
    assert exchanges.get_exchange_count() < 1000, f"Exchanges might not be configured correctly, we have {exchanges.get_exchange_count()} exchanges"
    configs = set()

    # Collect exchange IDs used by exchange account pairs (Derive, etc.)
    # and CCTP bridge pairs so we can skip them in the exchange loop below
    non_dex_exchange_ids = set()
    has_cctp_bridge = False
    pairs_df = strategy_universe.data_universe.pairs.df
    if "other_data" in pairs_df.columns:
        for _, row in pairs_df.iterrows():
            other_data = row.get("other_data")
            if isinstance(other_data, dict):
                if other_data.get("exchange_protocol"):
                    non_dex_exchange_ids.add(row["exchange_id"])
                if other_data.get("bridge_protocol") == "cctp":
                    has_cctp_bridge = True
                    non_dex_exchange_ids.add(row["exchange_id"])

    vaults_done = False
    hypercore_vault_done = False

    # Hypercore chain IDs — vault exchanges on these chains are Hypercore
    # vaults, even if they still carry the old erc_4626_vault exchange type
    # from stale cached data.
    hypercore_chain_ids = {ChainId.hypercore.value, ChainId.hyperliquid.value}

    # Only set up routing for exchanges that have actual loaded pairs.
    # Exchange metadata from vault universes may include exchanges on
    # other chains (e.g. Uniswap on Ethereum) that have no pairs in
    # the live universe and would cause RPC errors during routing setup.
    exchange_ids_with_pairs = set(pairs_df["exchange_id"].unique())

    for xc in exchanges.exchanges.values():
        if xc.exchange_id in non_dex_exchange_ids:
            continue
        if xc.exchange_id not in exchange_ids_with_pairs:
            continue

        # Detect Hypercore vaults by exchange type OR by chain_id fallback
        # for stale cached data that still uses erc_4626_vault type.
        is_hypercore_vault = (
            xc.exchange_type == ExchangeType.hypercore_vault
            or (xc.exchange_type == ExchangeType.erc_4626_vault and xc.chain_id in hypercore_chain_ids)
        )

        if is_hypercore_vault:
            if not hypercore_vault_done:
                configs.add(
                    ProtocolRoutingId(
                        router_name="hypercore_vault",
                        exchange_slug=None,
                    )
                )
                hypercore_vault_done = True
        elif xc.exchange_type == ExchangeType.erc_4626_vault:
            if not vaults_done:
                configs.add(
                    ProtocolRoutingId(
                        router_name="vault",
                        exchange_slug=None,
                    )
                )
                vaults_done = True
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

    # Add CCTP bridge routing if any bridge pairs are present
    if has_cctp_bridge:
        configs.add(
            ProtocolRoutingId(
                router_name="cctp-bridge",
            )
        )

    return configs
