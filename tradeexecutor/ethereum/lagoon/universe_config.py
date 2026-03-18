"""Translate a strategy's trading universe to Lagoon vault deployment configuration.

Bridges the strategy-level :class:`TradingStrategyUniverse` definition with the
Lagoon vault deployment infrastructure, producing per-chain :class:`LagoonConfig`
objects ready for :func:`deploy_multichain_lagoon_vault`.

Example::

    from tradeexecutor.ethereum.lagoon.universe_config import translate_trading_universe_to_lagoon_config

    configs = translate_trading_universe_to_lagoon_config(
        universe=strategy_universe,
        chain_web3={"arbitrum": web3_arb, "base": web3_base},
        asset_managers=[deployer.address],
        safe_owners=[deployer.address],
        safe_threshold=1,
        safe_salt_nonce=42,
    )
"""

import logging
from copy import deepcopy

from eth_defi.cctp.constants import TESTNET_CHAIN_IDS
from eth_defi.cctp.whitelist import CCTPDeployment
from eth_defi.erc_4626.vault_protocol.lagoon.deployment import (
    LagoonConfig, LagoonDeploymentParameters)
from eth_defi.gmx.whitelist import GMXDeployment, fetch_all_gmx_markets
from eth_defi.uniswap_v3.constants import UNISWAP_V3_DEPLOYMENTS
from eth_defi.uniswap_v3.deployment import \
    fetch_deployment as fetch_deployment_uni_v3
from eth_typing import HexAddress
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import ExchangeType
from web3 import Web3

from tradeexecutor.state.identifier import TradingPairKind
from tradeexecutor.strategy.trading_strategy_universe import \
    TradingStrategyUniverse

logger = logging.getLogger(__name__)

HYPERCORE_NATIVE_CHAIN_ID = 9999
HYPEREVM_CHAIN_ID = ChainId.hyperliquid.value  # 999


def normalise_deployment_chain_id(chain_id: int) -> int | None:
    """Normalise non-deployable or redirected chain ids to deployment chains."""
    if chain_id == HYPERCORE_NATIVE_CHAIN_ID:
        return HYPEREVM_CHAIN_ID
    return chain_id


def _collect_chain_ids(universe: TradingStrategyUniverse) -> set[int]:
    """Collect all deployable chain ids from the trading universe."""
    chain_ids: set[int] = set()
    for pair in universe.iterate_pairs():
        for chain_id in (pair.base.chain_id, pair.quote.chain_id):
            normalised = normalise_deployment_chain_id(chain_id)
            if normalised is not None:
                chain_ids.add(normalised)
    return chain_ids


def _build_chain_slug_maps(all_chain_ids: set[int], chain_web3: dict[str, Web3]) -> tuple[dict[int, str], dict[str, int]]:
    """Build and validate slug mappings for deployment chains."""
    chain_id_to_slug = {cid: ChainId(cid).get_slug() for cid in all_chain_ids}
    slug_to_chain_id = {slug: cid for cid, slug in chain_id_to_slug.items()}
    for chain_id, slug in chain_id_to_slug.items():
        assert slug in chain_web3, (
            f"No Web3 connection for chain {slug} (id={chain_id}). "
            f"Available connections: {list(chain_web3.keys())}"
        )
    return chain_id_to_slug, slug_to_chain_id


def _collect_universe_metadata(universe: TradingStrategyUniverse, all_chain_ids: set[int]) -> tuple[dict[int, set[int]], bool, set[int], set[int], dict[int, list[str]]]:
    """Collect cross-chain protocol metadata used to build Lagoon configs."""
    cctp_destinations: dict[int, set[int]] = {cid: set() for cid in all_chain_ids}
    has_cctp = False
    gmx_chain_ids: set[int] = set()
    hypercore_vaults_per_chain: dict[int, list[str]] = {}

    for pair in universe.iterate_pairs():
        if pair.kind == TradingPairKind.cctp_bridge:
            has_cctp = True
            src = normalise_deployment_chain_id(pair.get_source_chain_id())
            dest = normalise_deployment_chain_id(pair.get_destination_chain_id())
            if src is not None and dest is not None:
                cctp_destinations[src].add(dest)
                cctp_destinations[dest].add(src)

        if pair.is_exchange_account() and pair.get_exchange_account_protocol() == "gmx":
            gmx_chain_ids.add(normalise_deployment_chain_id(pair.base.chain_id))

        if pair.is_vault() and pair.other_data.get("vault_protocol") == "hypercore":
            vault_addr = pair.pool_address
            if vault_addr:
                chain_id = normalise_deployment_chain_id(pair.base.chain_id)
                if chain_id is not None:
                    hypercore_vaults_per_chain.setdefault(chain_id, []).append(vault_addr)

    uniswap_v3_chain_ids = {
        exchange.chain_id.value
        for exchange in universe.data_universe.exchange_universe.exchanges.values()
        if exchange.exchange_type == ExchangeType.uniswap_v3
    }
    return cctp_destinations, has_cctp, uniswap_v3_chain_ids, gmx_chain_ids, hypercore_vaults_per_chain


def _collect_chain_token_addresses(universe: TradingStrategyUniverse, all_chain_ids: set[int], any_asset: bool) -> dict[int, set[str]]:
    """Collect per-chain token whitelist addresses after chain normalisation."""
    chain_token_addresses: dict[int, set[str]] = {cid: set() for cid in all_chain_ids}
    if any_asset:
        return chain_token_addresses

    for pair in universe.iterate_pairs():
        for asset in (pair.base, pair.quote):
            chain_id = normalise_deployment_chain_id(asset.chain_id)
            if chain_id is not None:
                chain_token_addresses.setdefault(chain_id, set()).add(Web3.to_checksum_address(asset.address))
    return chain_token_addresses


def _apply_protocol_configs(
    *,
    config: LagoonConfig,
    chain_id: int,
    slug: str,
    chain_web3: dict[str, Web3],
    cctp_destinations: dict[int, set[int]],
    uniswap_v3_chain_ids: set[int],
    gmx_chain_ids: set[int],
    hypercore_vaults_per_chain: dict[int, list[str]],
    any_asset: bool,
) -> None:
    """Apply protocol-specific whitelist/deployment settings to one chain config."""
    dest_chain_ids = cctp_destinations.get(chain_id, set())
    if dest_chain_ids:
        config.cctp_deployment = CCTPDeployment.create_for_chain(
            chain_id=chain_id,
            allowed_destinations=list(dest_chain_ids),
        )

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

    if chain_id in gmx_chain_ids:
        if any_asset:
            market_addresses = []
        else:
            all_markets = fetch_all_gmx_markets(chain_web3[slug])
            market_addresses = list(all_markets.keys())

        if chain_id == ChainId.arbitrum_sepolia.value:
            from eth_defi.gmx.contracts import \
                get_contract_addresses as get_gmx_addresses
            testnet_addrs = get_gmx_addresses("arbitrum_sepolia")
            config.gmx_deployment = GMXDeployment(
                exchange_router=testnet_addrs.exchangerouter,
                synthetics_router=testnet_addrs.syntheticsrouter,
                order_vault=testnet_addrs.ordervault,
                markets=market_addresses,
            )
        else:
            config.gmx_deployment = GMXDeployment.create_arbitrum(markets=market_addresses)
        logger.info("GMX deployment configured for %s: %d market(s)%s", slug, len(market_addresses), " (skipped per-market whitelisting — any_asset=True)" if any_asset else " (all markets)")

    vault_addrs = hypercore_vaults_per_chain.get(chain_id, [])
    if vault_addrs:
        config.hypercore_vaults = vault_addrs
        logger.info("Hypercore vaults configured for %s: %s", slug, vault_addrs)


def translate_trading_universe_to_lagoon_config(
    universe: TradingStrategyUniverse,
    chain_web3: dict[str, Web3],
    safe_owners: list[HexAddress],
    safe_threshold: int,
    safe_salt_nonce: int | None,
    fund_name: str = "Crosschain Strategy Vault",
    fund_symbol: str = "CSV",
    any_asset: bool = False,
    asset_managers: list[HexAddress | str] | None = None,
    asset_manager: HexAddress | str | None = None,
    guard_only: bool = False,
    existing_vault_address: HexAddress | str | None = None,
    existing_safe_address: HexAddress | str | None = None,
) -> dict[str, LagoonConfig]:
    """Translate a trading universe into per-chain Lagoon vault deployment configs.

    Inspects the universe's pairs and exchanges to determine which chains
    need CCTP bridging, Uniswap v3 whitelisting, etc., and builds the
    appropriate :class:`LagoonConfig` for each chain.

    By default the source chain (where ``reserve_assets[0]`` lives) gets a
    full Lagoon vault deployment. All other chains become satellite
    deployments (Safe + guard only, no vault contract).

    When ``guard_only`` is enabled, the source chain reuses the existing
    Lagoon vault and Safe while satellite chains reuse the existing Safe
    only and deploy replacement guards/modules.

    When *any_asset* is ``False`` (the default), the function extracts all
    unique token addresses from the universe's trading pairs and whitelists
    them per chain via :attr:`LagoonConfig.assets`.

    :param universe:
        Strategy trading universe containing pairs, exchanges, and reserve assets.

    :param chain_web3:
        Web3 connections keyed by chain slug (e.g. ``{"arbitrum": web3_arb, "base": web3_base}``).

    :param asset_managers:
        Ordered list of addresses that may manage the vault and execute
        trades.

        The first address becomes the primary asset manager and Lagoon
        valuation manager. Any later addresses are secondary asset
        managers and only receive guard sender permissions.

    :param asset_manager:
        Backwards-compatible single-asset-manager input.

        Used only when ``asset_managers`` is not provided.

    :param safe_owners:
        Addresses that own the Safe multisig.

    :param safe_threshold:
        Number of Safe owner signatures required.

    :param safe_salt_nonce:
        CREATE2 salt for deterministic Safe address across chains.

        Not needed when ``guard_only`` is enabled and the deployment reuses an
        existing Safe across all chains.

    :param fund_name:
        Vault token name.

    :param fund_symbol:
        Vault token symbol.

    :param any_asset:
        When ``True``, the vault guard allows any ERC-20 token.
        When ``False`` (default), only tokens from the trading universe
        are whitelisted.

    :param guard_only:
        When ``True``, build configs for redeploying only the guard/module
        against an existing Safe. The source chain also reuses the existing
        Lagoon vault.

    :param existing_vault_address:
        Existing Lagoon vault address on the source chain. Required when
        ``guard_only`` is ``True``.

    :param existing_safe_address:
        Existing deterministic Safe address shared across all chains.
        Required when ``guard_only`` is ``True``.

    :return:
        Per-chain configs keyed by chain slug, ready for
        :func:`deploy_multichain_lagoon_vault`.
    """

    logger.info("translate_trading_universe_to_lagoon_config() called with universe: %s", universe)

    if asset_managers is None:
        assert asset_manager is not None, "Either asset_managers or asset_manager must be provided"
        asset_managers = [asset_manager]

    if guard_only:
        assert existing_safe_address, "existing_safe_address must be provided when guard_only=True"
        assert existing_vault_address, "existing_vault_address must be provided when guard_only=True"

    # Determine source chain from the first reserve asset
    reserve_asset = list(universe.reserve_assets)[0]
    source_chain_id = normalise_deployment_chain_id(reserve_asset.chain_id)
    source_chain_slug = ChainId(source_chain_id).get_slug()

    all_chain_ids = _collect_chain_ids(universe)
    chain_id_to_slug, _ = _build_chain_slug_maps(all_chain_ids, chain_web3)

    # Detect testnet
    is_testnet = any(cid in TESTNET_CHAIN_IDS for cid in all_chain_ids)

    cctp_destinations, has_cctp, uniswap_v3_chain_ids, gmx_chain_ids, hypercore_vaults_per_chain = _collect_universe_metadata(
        universe,
        all_chain_ids,
    )
    chain_token_addresses = _collect_chain_token_addresses(universe, all_chain_ids, any_asset)

    logger.info(
        "Universe analysis: source_chain=%s, chains=%s, cctp=%s, uniswap_v3_chains=%s, gmx_chains=%s, hypercore_vault_chains=%s, testnet=%s, any_asset=%s",
        source_chain_slug,
        list(chain_id_to_slug.values()),
        has_cctp,
        [chain_id_to_slug[cid] for cid in uniswap_v3_chain_ids],
        [chain_id_to_slug[cid] for cid in gmx_chain_ids],
        [chain_id_to_slug[cid] for cid in hypercore_vaults_per_chain],
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
            asset_managers=list(asset_managers),
            safe_owners=list(safe_owners),
            safe_threshold=safe_threshold,
            safe_salt_nonce=safe_salt_nonce,
            any_asset=any_asset,
            assets=token_addresses if token_addresses else None,
            satellite_chain=not is_source,
            guard_only=guard_only,
            existing_safe_address=existing_safe_address if guard_only else None,
            existing_vault_address=existing_vault_address if guard_only and is_source else None,
        )

        # Source chain needs from_the_scratch on testnets (no factory)
        if is_source and is_testnet:
            config.from_the_scratch = True
            config.use_forge = True

        _apply_protocol_configs(
            config=config,
            chain_id=chain_id,
            slug=slug,
            chain_web3=chain_web3,
            cctp_destinations=cctp_destinations,
            uniswap_v3_chain_ids=uniswap_v3_chain_ids,
            gmx_chain_ids=gmx_chain_ids,
            hypercore_vaults_per_chain=hypercore_vaults_per_chain,
            any_asset=any_asset,
        )

        configs[slug] = config

    return configs
