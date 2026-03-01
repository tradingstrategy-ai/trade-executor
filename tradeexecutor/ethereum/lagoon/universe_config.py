"""Translate a strategy's trading universe to Lagoon vault deployment configuration.

Bridges the strategy-level :class:`TradingStrategyUniverse` definition with the
Lagoon vault deployment infrastructure, producing per-chain :class:`LagoonConfig`
objects ready for :func:`deploy_multichain_lagoon_vault`.

Example::

    from tradeexecutor.ethereum.lagoon.universe_config import translate_trading_universe_to_lagoon_config

    configs = translate_trading_universe_to_lagoon_config(
        universe=strategy_universe,
        chain_web3={"arbitrum": web3_arb, "base": web3_base},
        asset_manager=deployer.address,
        safe_owners=[deployer.address],
        safe_threshold=1,
        safe_salt_nonce=42,
    )
"""

import logging
from copy import deepcopy

from eth_typing import HexAddress
from web3 import Web3

from eth_defi.cctp.constants import TESTNET_CHAIN_IDS
from eth_defi.cctp.whitelist import CCTPDeployment
from eth_defi.erc_4626.vault_protocol.lagoon.deployment import LagoonConfig, LagoonDeploymentParameters
from eth_defi.uniswap_v3.constants import UNISWAP_V3_DEPLOYMENTS
from eth_defi.uniswap_v3.deployment import fetch_deployment as fetch_deployment_uni_v3

from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import ExchangeType

from tradeexecutor.state.identifier import TradingPairKind
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse

logger = logging.getLogger(__name__)


def translate_trading_universe_to_lagoon_config(
    universe: TradingStrategyUniverse,
    chain_web3: dict[str, Web3],
    asset_manager: HexAddress,
    safe_owners: list[HexAddress],
    safe_threshold: int,
    safe_salt_nonce: int,
    fund_name: str = "Crosschain Strategy Vault",
    fund_symbol: str = "CSV",
    any_asset: bool = False,
) -> dict[str, LagoonConfig]:
    """Translate a trading universe into per-chain Lagoon vault deployment configs.

    Inspects the universe's pairs and exchanges to determine which chains
    need CCTP bridging, Uniswap v3 whitelisting, etc., and builds the
    appropriate :class:`LagoonConfig` for each chain.

    The source chain (where ``reserve_assets[0]`` lives) gets a full Lagoon
    vault deployment. All other chains become satellite deployments
    (Safe + guard only, no vault contract).

    When *any_asset* is ``False`` (the default), the function extracts all
    unique token addresses from the universe's trading pairs and whitelists
    them per chain via :attr:`LagoonConfig.assets`.

    :param universe:
        Strategy trading universe containing pairs, exchanges, and reserve assets.

    :param chain_web3:
        Web3 connections keyed by chain slug (e.g. ``{"arbitrum": web3_arb, "base": web3_base}``).

    :param asset_manager:
        Address that will manage the vault (execute trades).

    :param safe_owners:
        Addresses that own the Safe multisig.

    :param safe_threshold:
        Number of Safe owner signatures required.

    :param safe_salt_nonce:
        CREATE2 salt for deterministic Safe address across chains.

    :param fund_name:
        Vault token name.

    :param fund_symbol:
        Vault token symbol.

    :param any_asset:
        When ``True``, the vault guard allows any ERC-20 token.
        When ``False`` (default), only tokens from the trading universe
        are whitelisted.

    :return:
        Per-chain configs keyed by chain slug, ready for
        :func:`deploy_multichain_lagoon_vault`.
    """

    # Determine source chain from the first reserve asset
    reserve_asset = list(universe.reserve_assets)[0]
    source_chain_id = reserve_asset.chain_id
    source_chain_slug = ChainId(source_chain_id).get_slug()

    # Collect all chain IDs from pairs
    all_chain_ids = set()
    for pair in universe.iterate_pairs():
        all_chain_ids.add(pair.base.chain_id)
        all_chain_ids.add(pair.quote.chain_id)

    # Map chain IDs to slugs
    chain_id_to_slug = {cid: ChainId(cid).get_slug() for cid in all_chain_ids}
    slug_to_chain_id = {slug: cid for cid, slug in chain_id_to_slug.items()}

    # Validate chain_web3 has connections for all chains
    for chain_id, slug in chain_id_to_slug.items():
        assert slug in chain_web3, (
            f"No Web3 connection for chain {slug} (id={chain_id}). "
            f"Available connections: {list(chain_web3.keys())}"
        )

    # Detect testnet
    is_testnet = any(cid in TESTNET_CHAIN_IDS for cid in all_chain_ids)

    # Collect CCTP destination chain IDs per chain
    # For each chain, find which other chains it bridges to/from
    cctp_destinations: dict[int, set[int]] = {cid: set() for cid in all_chain_ids}
    has_cctp = False
    for pair in universe.iterate_pairs():
        if pair.kind == TradingPairKind.cctp_bridge:
            has_cctp = True
            src = pair.get_source_chain_id()
            dest = pair.get_destination_chain_id()
            cctp_destinations[src].add(dest)
            cctp_destinations[dest].add(src)

    # Collect Uniswap v3 exchange chain IDs
    uniswap_v3_chain_ids = set()
    exchanges = universe.data_universe.exchange_universe.exchanges
    for exchange in exchanges.values():
        if exchange.exchange_type == ExchangeType.uniswap_v3:
            uniswap_v3_chain_ids.add(exchange.chain_id.value)

    # Collect unique token addresses per chain (used when any_asset=False)
    chain_token_addresses: dict[int, set[str]] = {cid: set() for cid in all_chain_ids}
    if not any_asset:
        for pair in universe.iterate_pairs():
            chain_token_addresses[pair.base.chain_id].add(Web3.to_checksum_address(pair.base.address))
            chain_token_addresses[pair.quote.chain_id].add(Web3.to_checksum_address(pair.quote.address))

    logger.info(
        "Universe analysis: source_chain=%s, chains=%s, cctp=%s, uniswap_v3_chains=%s, testnet=%s, any_asset=%s",
        source_chain_slug,
        list(chain_id_to_slug.values()),
        has_cctp,
        [chain_id_to_slug[cid] for cid in uniswap_v3_chain_ids],
        is_testnet,
        any_asset,
    )

    if not any_asset:
        logger.info(
            "Token whitelist per chain: %s",
            {chain_id_to_slug[cid]: sorted(addrs) for cid, addrs in chain_token_addresses.items() if addrs},
        )

    # Build per-chain configs
    base_params = LagoonDeploymentParameters(
        underlying=None,  # Auto-resolved per chain from USDC_NATIVE_TOKEN
        name=fund_name,
        symbol=fund_symbol,
    )

    configs: dict[str, LagoonConfig] = {}
    for chain_id in all_chain_ids:
        slug = chain_id_to_slug[chain_id]
        is_source = (chain_id == source_chain_id)

        token_addresses = sorted(chain_token_addresses.get(chain_id, set())) if not any_asset else None

        config = LagoonConfig(
            parameters=deepcopy(base_params),
            asset_manager=asset_manager,
            safe_owners=list(safe_owners),
            safe_threshold=safe_threshold,
            safe_salt_nonce=safe_salt_nonce,
            any_asset=any_asset,
            assets=token_addresses if token_addresses else None,
            satellite_chain=not is_source,
        )

        # Source chain needs from_the_scratch on testnets (no factory)
        if is_source and is_testnet:
            config.from_the_scratch = True
            config.use_forge = True

        # Configure CCTP
        dest_chain_ids = cctp_destinations.get(chain_id, set())
        if dest_chain_ids:
            config.cctp_deployment = CCTPDeployment.create_for_chain(
                chain_id=chain_id,
                allowed_destinations=list(dest_chain_ids),
            )

        # Configure Uniswap v3
        if chain_id in uniswap_v3_chain_ids:
            deployment_data = UNISWAP_V3_DEPLOYMENTS.get(slug)
            if deployment_data:
                config.uniswap_v3 = fetch_deployment_uni_v3(
                    chain_web3[slug],
                    factory_address=deployment_data["factory"],
                    router_address=deployment_data["router"],
                    position_manager_address=deployment_data["position_manager"],
                    quoter_address=deployment_data["quoter"],
                    quoter_v2=deployment_data.get("quoter_v2", False),
                    router_v2=deployment_data.get("router_v2", False),
                )
            else:
                logger.warning("No Uniswap v3 deployment data for chain %s", slug)

        configs[slug] = config

    return configs
