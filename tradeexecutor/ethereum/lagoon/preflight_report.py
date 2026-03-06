"""Pre-flight deployment report for Lagoon vault deployments.

Logs a structured summary of deployment parameters before the user
confirms.  Works for both single-chain and multichain (strategy-file)
deployment paths.
"""

import logging

from eth_defi.erc_4626.vault_protocol.lagoon.config import LagoonChainConfig
from eth_defi.erc_4626.vault_protocol.lagoon.deployment import LagoonConfig
from eth_defi.hotwallet import HotWallet
from eth_defi.token import TokenDetails
from web3 import Web3

_logger = logging.getLogger(__name__)


def log_deployment_preflight_report(
    *,
    hot_wallet: HotWallet,
    chain_web3: dict[str, Web3],
    fund_name: str,
    fund_symbol: str,
    asset_manager: str,
    multisig_owners: list[str],
    performance_fee: int,
    management_fee: int,
    etherscan_api_key: str | None = None,
    verifier: str = "etherscan",
    verifier_url: str | None = None,
    # Multichain: per-chain LagoonConfig objects
    chain_configs: dict[str, LagoonConfig] | None = None,
    # Legacy single-chain fields (when chain_configs is None)
    denomination_token: TokenDetails | None = None,
    any_asset: bool = False,
    whitelisted_asset_details: list | None = None,
    uniswap_v2: bool = False,
    uniswap_v3: bool = False,
    one_delta: bool = False,
    aave: bool = False,
    erc_4626_vaults: str | None = None,
    lagoon_chain_config: LagoonChainConfig | None = None,
    simulate: bool = False,
    logger=None,
):
    """Log a structured pre-flight report before vault deployment.

    Outputs a generic section (common to all chains) followed by
    per-chain sections showing balances, nonces, and protocol
    whitelisting details.

    :param chain_web3:
        ``{chain_slug: Web3}`` mapping.  Single-chain callers pass a
        one-entry dict.

    :param chain_configs:
        When deploying via strategy file, the per-chain
        :class:`LagoonConfig` objects.  When ``None``, the legacy
        single-chain keyword arguments are used instead.
    """

    if logger is None:
        logger = _logger

    sep = "-" * 80

    # ------------------------------------------------------------------
    # Generic section (same across all chains)
    # ------------------------------------------------------------------
    logger.info(sep)
    logger.info("DEPLOYMENT PRE-FLIGHT REPORT")
    logger.info(sep)

    logger.info("Deployer hot wallet: %s", hot_wallet.address)
    if asset_manager != hot_wallet.address:
        logger.info("Asset manager: %s", asset_manager)
    else:
        logger.info("Hot wallet set for the asset manager role")

    logger.info("Fund: %s (%s)", fund_name, fund_symbol)
    logger.info("Multisig owners: %s", multisig_owners)
    logger.info("Performance fee: %f %%", performance_fee / 100)
    logger.info("Management fee: %f %%", management_fee / 100)
    logger.info("Simulate: %s", simulate)

    if etherscan_api_key:
        logger.info("Etherscan API key: %s", etherscan_api_key)
    else:
        logger.warning("Etherscan API key: not provided")

    logger.info("Verifier: %s", verifier)
    if verifier_url:
        logger.info("Verifier URL: %s", verifier_url)

    # ------------------------------------------------------------------
    # Per-chain sections
    # ------------------------------------------------------------------
    for slug, web3 in chain_web3.items():
        logger.info(sep)
        logger.info("Chain: %s (id %d)", slug, web3.eth.chain_id)
        logger.info("  Latest block: %s", f"{web3.eth.block_number:,}")

        # Balance & nonce — sync nonce first so the value is up-to-date
        hot_wallet.sync_nonce(web3)
        balance = hot_wallet.get_native_currency_balance(web3)
        logger.info("  Deployer balance: %f, nonce %d", balance, hot_wallet.current_nonce)

        if chain_configs and slug in chain_configs:
            _log_chain_config_section(chain_configs[slug], logger)
        else:
            _log_legacy_single_chain_section(
                denomination_token=denomination_token,
                any_asset=any_asset,
                whitelisted_asset_details=whitelisted_asset_details or [],
                uniswap_v2=uniswap_v2,
                uniswap_v3=uniswap_v3,
                one_delta=one_delta,
                aave=aave,
                erc_4626_vaults=erc_4626_vaults,
                lagoon_chain_config=lagoon_chain_config,
                logger=logger,
            )

    logger.info(sep)


def _log_chain_config_section(config: LagoonConfig, logger):
    """Log details from a :class:`LagoonConfig` (multichain path)."""

    logger.info("  Role: %s", "satellite" if config.satellite_chain else "source (vault)")
    logger.info("  From the scratch deployment: %s", config.from_the_scratch)
    logger.info("  Use BeaconProxyFactory: %s", config.factory_contract)

    if config.parameters and config.parameters.underlying:
        logger.info("  Underlying token: %s", config.parameters.underlying)

    logger.info("  Whitelisting any asset: %s", config.any_asset)

    if config.assets:
        logger.info("  Whitelisted assets: %s", ", ".join(str(a) for a in config.assets))

    logger.info("  Uniswap v3: %s", config.uniswap_v3 is not None)
    logger.info("  Aave v3: %s", config.aave_v3 is not None)
    logger.info("  CowSwap: %s", config.cowswap)
    logger.info("  Velora: %s", config.velora)
    logger.info("  GMX: %s", config.gmx_deployment is not None)
    logger.info("  CCTP: %s", config.cctp_deployment is not None)

    if config.erc_4626_vaults:
        vault_addrs = ", ".join(str(getattr(v, "address", v)) for v in config.erc_4626_vaults)
        logger.info("  ERC-4626 vaults: %s", vault_addrs)

    if config.etherscan_api_key:
        logger.info("  Etherscan API key (chain-level): %s", config.etherscan_api_key)


def _log_legacy_single_chain_section(
    *,
    denomination_token,
    any_asset,
    whitelisted_asset_details,
    uniswap_v2,
    uniswap_v3,
    one_delta,
    aave,
    erc_4626_vaults,
    lagoon_chain_config,
    logger,
):
    """Log details from the legacy single-chain CLI flags."""

    if denomination_token:
        logger.info("  Underlying token: %s", denomination_token.symbol)

    logger.info("  Whitelisting any token: %s", any_asset)

    if whitelisted_asset_details:
        logger.info("  Whitelisted assets: %s", ", ".join(a.symbol for a in whitelisted_asset_details))

    logger.info("  Uniswap v2: %s", uniswap_v2)
    logger.info("  Uniswap v3: %s", uniswap_v3)
    logger.info("  1delta: %s", one_delta)
    logger.info("  Aave: %s", aave)
    logger.info("  ERC-4626 vaults: %s", erc_4626_vaults)

    if lagoon_chain_config:
        logger.info("  From the scratch deployment: %s", lagoon_chain_config.from_the_scratch)
        logger.info("  Use BeaconProxyFactory: %s", lagoon_chain_config.factory_contract)
