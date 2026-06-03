"""Synthetic CCTP bridge helpers for multichain trading universes."""

import logging
from dataclasses import dataclass

import pandas as pd
from eth_defi.cctp.constants import TOKEN_MESSENGER_V2
from eth_defi.token import USDC_NATIVE_TOKEN
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import Exchange, ExchangeType, ExchangeUniverse
from tradingstrategy.pair import PandasPairUniverse

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GeneratedCCTPBridgeUniverse:
    """Result of synthetic forward CCTP bridge generation."""

    pair_universe: PandasPairUniverse
    exchange_universe: ExchangeUniverse
    generated_pairs: list[TradingPairIdentifier]


def can_auto_generate_cctp_bridges(
    reserve_asset: AssetIdentifier,
    primary_chain: ChainId | None,
) -> bool:
    """Check whether a universe can use synthetic forward CCTP bridges."""
    if primary_chain is None:
        return False

    primary_usdc = USDC_NATIVE_TOKEN.get(primary_chain.value)
    if primary_usdc is None:
        return False

    return (
        reserve_asset.chain_id == primary_chain.value
        and reserve_asset.address.lower() == primary_usdc.lower()
    )


def generate_primary_to_satellite_cctp_bridge_universe(
    pairs: PandasPairUniverse,
    exchange_universe: ExchangeUniverse,
    reserve_asset: AssetIdentifier,
    primary_chain: ChainId,
) -> GeneratedCCTPBridgeUniverse:
    """Generate forward-only CCTP bridge pairs for a multichain universe.

    The generated pairs always represent the route ``primary -> satellite``.
    Reverse withdrawals use sell trades on the same forward pair.
    """
    if not can_auto_generate_cctp_bridges(reserve_asset, primary_chain):
        return GeneratedCCTPBridgeUniverse(
            pair_universe=pairs,
            exchange_universe=exchange_universe,
            generated_pairs=[],
        )

    existing_routes = set()
    for _, row in pairs.df.iterrows():
        other_data = row.get("other_data")
        if not isinstance(other_data, dict):
            continue
        if other_data.get("bridge_protocol") != "cctp":
            continue
        destination_chain_id = other_data.get("_base_chain_id", other_data.get("destination_chain_id"))
        if destination_chain_id is None:
            continue
        existing_routes.add((int(row["chain_id"]), int(destination_chain_id)))

    chain_ids = {
        ChainId(int(chain_id))
        for chain_id in pairs.df["chain_id"].unique()
    }
    satellite_chain_ids = sorted(
        (
            chain_id
            for chain_id in chain_ids
            if chain_id != primary_chain and chain_id.value in USDC_NATIVE_TOKEN
        ),
        key=lambda chain_id: chain_id.value,
    )

    if not satellite_chain_ids:
        return GeneratedCCTPBridgeUniverse(
            pair_universe=pairs,
            exchange_universe=exchange_universe,
            generated_pairs=[],
        )

    next_pair_id = int(pairs.df["pair_id"].max()) + 1 if len(pairs.df) else 1
    next_exchange_id = max(exchange_universe.exchanges.keys(), default=0) + 1
    generated_pairs: list[TradingPairIdentifier] = []

    bridge_exchange = Exchange(
        chain_id=primary_chain,
        chain_slug=primary_chain.get_slug(),
        exchange_id=next_exchange_id,
        exchange_slug="cctp-bridge",
        address=TOKEN_MESSENGER_V2,
        exchange_type=ExchangeType.uniswap_v2,
        pair_count=0,
        name="CCTP Bridge",
    )

    # Resolve USDC decimals from the pair universe by address lookup.
    # The reserve_asset.decimals can be wrong (e.g. 18) when the
    # DEXPair symbol-based decimal resolution is ambiguous (both tokens
    # share the same symbol in vault pairs).
    primary_usdc_address = USDC_NATIVE_TOKEN[primary_chain.value]
    primary_usdc_token = pairs.get_token(primary_usdc_address.lower(), chain_id=primary_chain)
    if primary_usdc_token is not None and primary_usdc_token.decimals is not None:
        usdc_decimals = primary_usdc_token.decimals
    else:
        usdc_decimals = reserve_asset.decimals

    if usdc_decimals != reserve_asset.decimals:
        logger.warning(
            "Reserve asset decimals mismatch: reserve_asset.decimals=%d, "
            "pair universe USDC decimals=%d for %s on chain %s. "
            "Using pair universe value.",
            reserve_asset.decimals,
            usdc_decimals,
            primary_usdc_address,
            primary_chain.get_name(),
        )

    for satellite_chain in satellite_chain_ids:
        route = (primary_chain.value, satellite_chain.value)
        if route in existing_routes:
            continue

        # Resolve satellite USDC decimals from the pair universe too
        sat_usdc_address = USDC_NATIVE_TOKEN[satellite_chain.value]
        sat_usdc_token = pairs.get_token(sat_usdc_address.lower(), chain_id=satellite_chain)
        sat_decimals = sat_usdc_token.decimals if (sat_usdc_token and sat_usdc_token.decimals) else usdc_decimals

        destination_asset = AssetIdentifier(
            chain_id=satellite_chain.value,
            address=sat_usdc_address,
            token_symbol=reserve_asset.token_symbol,
            decimals=sat_decimals,
        )
        # Use corrected decimals for the quote (primary chain USDC) too
        quote_asset = AssetIdentifier(
            chain_id=reserve_asset.chain_id,
            address=reserve_asset.address,
            token_symbol=reserve_asset.token_symbol,
            decimals=usdc_decimals,
        )

        generated_pairs.append(
            TradingPairIdentifier(
                base=destination_asset,
                quote=quote_asset,
                pool_address=TOKEN_MESSENGER_V2,
                exchange_address=TOKEN_MESSENGER_V2,
                internal_id=next_pair_id,
                internal_exchange_id=next_exchange_id,
                fee=0.0,
                kind=TradingPairKind.cctp_bridge,
                exchange_name="CCTP Bridge",
                other_data={
                    "bridge_protocol": "cctp",
                    "destination_chain_id": satellite_chain.value,
                },
            )
        )
        next_pair_id += 1

    if not generated_pairs:
        return GeneratedCCTPBridgeUniverse(
            pair_universe=pairs,
            exchange_universe=exchange_universe,
            generated_pairs=[],
        )

    bridge_exchange.pair_count = len(generated_pairs)

    logger.info(
        "Generated %d synthetic forward CCTP bridge pair(s) from %s to %s",
        len(generated_pairs),
        primary_chain.get_name(),
        ", ".join(chain.get_name() for chain in satellite_chain_ids),
    )

    # Avoid a top-level import cycle with trading_strategy_universe.
    from tradeexecutor.strategy.trading_strategy_universe import create_pair_universe_from_code

    bridge_pair_universe = create_pair_universe_from_code(primary_chain, generated_pairs)
    merged_exchange_universe = ExchangeUniverse(exchanges=dict(exchange_universe.exchanges))
    merged_exchange_universe.add([bridge_exchange])

    all_columns = list(dict.fromkeys([*pairs.df.columns, *bridge_pair_universe.df.columns]))
    existing_pairs_df = pairs.df.reset_index(drop=True).reindex(columns=all_columns)
    generated_pairs_df = bridge_pair_universe.df.reset_index(drop=True).reindex(columns=all_columns)
    merged_pairs_df = pd.DataFrame.from_records(
        [
            *existing_pairs_df.to_dict("records"),
            *generated_pairs_df.to_dict("records"),
        ],
        columns=all_columns,
    )
    merged_pair_universe = PandasPairUniverse(
        merged_pairs_df,
        exchange_universe=merged_exchange_universe,
    )

    return GeneratedCCTPBridgeUniverse(
        pair_universe=merged_pair_universe,
        exchange_universe=merged_exchange_universe,
        generated_pairs=generated_pairs,
    )
