"""lagoon-deploy-vault CLI command.

See :ref:`vault deployment` for the full documentation how to use this command.

Example how to manually test:

.. code-block:: shell

    export SIMULATE=true
    export FUND_NAME="Up only and then more"
    export FUND_SYMBOL="UP"
    export VAULT_RECORD_FILE="/tmp/sample-vault-deployment.json"
    export OWNER_ADDRESS="0x238B0435F69355e623d99363d58F7ba49C408491"

    #
    # Asset configuration
    #

    # USDC
    export DENOMINATION_ASSET="0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
    # Whitelisted tokens for Polygon: WETH, WMATIC
    export WHITELISTED_ASSETS="0x7ceb23fd6bc0add59e62ac25578270cff1b9f619 0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270"

    #
    # Secret configuration
    #

    export JSON_RPC_POLYGON=
    export PRIVATE_KEY=
    # Is Polygonscan.com API key, passed to Forge
    export ETHERSCAN_API_KEY=

    #
    # Verifier configuration (optional)
    #

    # For Blockscout-based chains (e.g., Derive Chain):
    # export VERIFIER=blockscout
    # export VERIFIER_URL=https://explorer.derive.xyz/api

    #
    # Asset manager (optional, defaults to PRIVATE_KEY address)
    #

    # export ASSET_MANAGER=0x...

    trade-executor lagoon-deploy-vault
"""

import datetime
import json
import os.path
import random
import sys
from pathlib import Path
from typing import cast

from eth_defi.aave_v3.constants import AAVE_V3_DEPLOYMENTS
from eth_defi.aave_v3.deployment import \
    fetch_deployment as fetch_aave_deployment
from eth_defi.abi import ONE_ADDRESS_STR
from eth_defi.erc_4626.classification import create_vault_instance
from eth_defi.erc_4626.core import ERC4626Feature
from eth_defi.erc_4626.vault import ERC4626Vault
from eth_defi.erc_4626.vault_protocol.lagoon.config import \
    get_lagoon_chain_config
from eth_defi.erc_4626.vault_protocol.lagoon.deployment import (
    DEFAULT_MANAGEMENT_RATE, DEFAULT_PERFORMANCE_RATE,
    LagoonDeploymentParameters, deploy_automated_lagoon_vault,
    deploy_multichain_lagoon_vault)
from eth_defi.hotwallet import HotWallet
from eth_defi.token import fetch_erc20_details
from eth_defi.uniswap_v2.constants import UNISWAP_V2_DEPLOYMENTS
from eth_defi.uniswap_v2.deployment import fetch_deployment
from eth_defi.uniswap_v3.constants import UNISWAP_V3_DEPLOYMENTS
from eth_defi.uniswap_v3.deployment import \
    fetch_deployment as fetch_deployment_uni_v3
from tradingstrategy.chain import ChainId
from typer import Option
from web3 import Web3

from tradeexecutor.cli.bootstrap import (create_web3_config, prepare_cache,
                                         prepare_token_cache)
from tradeexecutor.cli.commands import shared_options
from tradeexecutor.cli.commands.app import app
from tradeexecutor.cli.commands.shared_options import \
    parse_comma_separated_list
from tradeexecutor.cli.log import setup_logging
from tradeexecutor.ethereum.lagoon.deploy_report import print_deployment_report
from tradeexecutor.ethereum.lagoon.preflight_report import log_deployment_preflight_report
from tradeexecutor.ethereum.lagoon.universe_config import \
    translate_trading_universe_to_lagoon_config
from tradeexecutor.ethereum.token_cache import get_default_token_cache
from tradeexecutor.ethereum.web3config import collect_rpc_kwargs
from tradeexecutor.monkeypatch.web3 import \
    construct_sign_and_send_raw_middleware
from tradeexecutor.strategy.execution_context import one_off_execution_context
from tradeexecutor.strategy.pandas_trader.create_universe_wrapper import \
    call_create_trading_universe
from tradeexecutor.strategy.strategy_module import read_strategy_module


@app.command()
def lagoon_deploy_vault(
    log_level: str = shared_options.log_level,
    json_rpc_binance: str | None = shared_options.json_rpc_binance,
    json_rpc_polygon: str | None = shared_options.json_rpc_polygon,
    json_rpc_avalanche: str | None = shared_options.json_rpc_avalanche,
    json_rpc_ethereum: str | None = shared_options.json_rpc_ethereum,
    json_rpc_base: str | None = shared_options.json_rpc_base,
    json_rpc_arbitrum: str | None = shared_options.json_rpc_arbitrum,
    json_rpc_anvil: str | None = shared_options.json_rpc_anvil,
    json_rpc_derive: str | None = shared_options.json_rpc_derive,
    json_rpc_arbitrum_sepolia: str | None = shared_options.json_rpc_arbitrum_sepolia,
    json_rpc_base_sepolia: str | None = shared_options.json_rpc_base_sepolia,
    json_rpc_hyperliquid: str | None = shared_options.json_rpc_hyperliquid,
    json_rpc_hyperliquid_testnet: str | None = shared_options.json_rpc_hyperliquid_testnet,
    json_rpc_monad: str | None = shared_options.json_rpc_monad,
    private_key: str = shared_options.private_key,

    # Vault options
    vault_record_file: Path = Option(..., envvar="VAULT_RECORD_FILE", help="Store vault data in this TXT file, paired with a JSON file."),
    fund_name: str | None = Option(None, envvar="FUND_NAME", help="On-chain name for the fund shares"),
    fund_symbol: str | None = Option(None, envvar="FUND_SYMBOL", help="On-chain token symbol for the fund shares"),
    denomination_asset: str | None = Option(None, envvar="DENOMINATION_ASSET", help="Stablecoin asset used for vault denomination"),
    multisig_owners: str | None = Option(None, callback=parse_comma_separated_list, envvar="MULTISIG_OWNERS", help="The list of acconts that are set to the cosigners of the Safe. The multisig threshold is number of cosigners - 1."),
    # terms_of_service_address: str | None = Option(None, envvar="TERMS_OF_SERVICE_ADDRESS", help="The address of the terms of service smart contract"),
    whitelisted_assets: str | None = Option(None, envvar="WHITELISTED_ASSETS", help="Space separarted list of ERC-20 addresses this vault can trade. Denomination asset does not need to be whitelisted separately."),
    any_asset: bool = Option(False, envvar="ANY_ASSET", help="Allow trading of any ERC-20 on Uniswap (unsecure)."),

    unit_testing: bool = shared_options.unit_testing,
    # production: bool = Option(False, envvar="PRODUCTION", help="Set production metadata flag true for the deployment."),
    simulate: bool = Option(False, envvar="SIMULATE", help="Simulate deployment using Anvil mainnet work, when doing manual deployment testing."),
    etherscan_api_key: str | None = Option(None, envvar="ETHERSCAN_API_KEY", help="Etherscan API key needed to verify contracts on a production deployment."),
    verifier: str = Option("etherscan", envvar="VERIFIER", help="Contract verifier to use: etherscan, blockscout, sourcify, oklink. Default: etherscan."),
    verifier_url: str | None = Option(None, envvar="VERIFIER_URL", help="Verifier API URL for Blockscout or custom verifiers (e.g., https://explorer.derive.xyz/api). Required when verifier=blockscout."),
    asset_manager_address: str | None = Option(None, envvar="ASSET_MANAGER", help="Address to use as vault asset manager. If not provided, uses the deployer address (derived from PRIVATE_KEY). Allows using a master deployer while assigning a different asset manager."),
    one_delta: bool = Option(False, envvar="ONE_DELTA", help="Whitelist 1delta interaction with GuardV0 smart contract."),
    aave: bool = Option(False, envvar="AAVE", help="Whitelist Aave aUSDC deposits"),
    uniswap_v2: bool = Option(False, envvar="UNISWAP_V2", help="Whitelist Uniswap v2"),
    uniswap_v3: bool = Option(False, envvar="UNISWAP_V3", help="Whitelist Uniswap v3"),
    cowswap: bool = Option(False, envvar="COWSWAP", help="Whitelist CoW Swap"),
    erc_4626_vaults: str | None = Option(None, envvar="ERC_4626_VAULTS", help="Whitelist ERC-4626 vaults, a comma separated list of addresses"),
    verbose: bool = Option(False, envvar="VERBOSE", help="Extra verbosity with deploy commands"),
    performance_fee: int = Option(DEFAULT_PERFORMANCE_RATE, envvar="PERFORMANCE_FEE", help="Performance fee in BPS"),
    management_fee: int = Option(DEFAULT_MANAGEMENT_RATE, envvar="MANAGEMENT_FEE", help="Management fee in BPS"),
    guard_only: bool = Option(False, envvar="GUARD_ONLY", help="Deploys a new TradingStrategyModuleV0 guard with new settings. Lagoon multisig owners must then perform the transaction to enable this guard."),
    existing_vault_address: str | None = Option(None, envvar="EXISTING_VAULT_ADDRESS", help="When deploying a guard only, get the existing vault address."),
    existing_safe_address: str | None = Option(None, envvar="EXISTING_SAFE_ADDRESS", help="When deploying a guard only, get the existing safe address."),
    vault_adapter_address: str = shared_options.vault_adapter_address,
    cache_path: Path | None = shared_options.cache_path,
    strategy_file: Path | None = Option(None, envvar="STRATEGY_FILE", help="Strategy module for multichain deployment. When provided, uses translate_trading_universe_to_lagoon_config() to generate per-chain configs."),
    safe_salt_nonce: int | None = Option(None, envvar="SAFE_SALT_NONCE", help="CREATE2 salt nonce for deterministic Safe address across chains. Random if not given."),
    trading_strategy_api_key: str | None = shared_options.trading_strategy_api_key,
    hypersync_api_key: str | None = shared_options.hypersync_api_key,
    chain_name: str | None = shared_options.chain_name,
):
    """Deploy a Lagoon vault or modify the vault deployment.

    Deploys a new Lagoon vault, Safe and TradingStrategyModuleV0 guard for automated trading.

    When --strategy-file is provided, performs a multichain deployment using the
    strategy's trading universe to determine per-chain configurations (CCTP bridging,
    Uniswap v3 whitelisting, etc.).

    TODO: Heavily under development.
    """

    assert private_key, "PRIVATE_KEY not set"

    logger = setup_logging(log_level)

    # Prepare cache for token metadata storage
    # Use a fixed executor ID for this deployment command
    executor_id = "lagoon-deploy"
    cache_path = prepare_cache(executor_id, cache_path, unit_testing=unit_testing)
    token_cache = prepare_token_cache(cache_path, unit_testing=unit_testing)

    rpc_kwargs = collect_rpc_kwargs(
        json_rpc_binance=json_rpc_binance,
        json_rpc_polygon=json_rpc_polygon,
        json_rpc_avalanche=json_rpc_avalanche,
        json_rpc_ethereum=json_rpc_ethereum,
        json_rpc_base=json_rpc_base,
        json_rpc_anvil=json_rpc_anvil,
        json_rpc_arbitrum=json_rpc_arbitrum,
        json_rpc_derive=json_rpc_derive,
        json_rpc_arbitrum_sepolia=json_rpc_arbitrum_sepolia,
        json_rpc_base_sepolia=json_rpc_base_sepolia,
        json_rpc_hyperliquid=json_rpc_hyperliquid,
        json_rpc_hyperliquid_testnet=json_rpc_hyperliquid_testnet,
        json_rpc_monad=json_rpc_monad,
        chain_name=chain_name,
    )
    web3config = create_web3_config(
        **rpc_kwargs,
        simulate=simulate,
        mev_endpoint_disabled=True,
    )

    if not web3config.has_any_connection():
        raise RuntimeError("Vault deploy requires that you pass JSON-RPC connection to one of the networks")

    hot_wallet = HotWallet.from_private_key(private_key)

    # Asset manager defaults to deployer, but can be overridden
    if asset_manager_address:
        asset_manager = Web3.to_checksum_address(asset_manager_address)
    else:
        asset_manager = hot_wallet.address

    assert not (strategy_file and denomination_asset), \
        f"Cannot use both --strategy-file and --denomination-asset. " \
        f"When --strategy-file is provided, the reserve asset is read from the strategy's create_trading_universe(). " \
        f"Remove --denomination-asset to use the strategy-file deployment path."

    # Strategy-file deployment path: use strategy file to generate per-chain configs
    # via translate_trading_universe_to_lagoon_config(). Handles both multichain
    # and single-chain strategies — protocol detection (GMX, CCTP, Uniswap v3, etc.)
    # is always driven by the strategy's trading universe.
    if strategy_file:
        _deploy_multichain(
            web3config=web3config,
            hot_wallet=hot_wallet,
            asset_manager=asset_manager,
            strategy_file=strategy_file,
            safe_salt_nonce=safe_salt_nonce,
            fund_name=fund_name,
            fund_symbol=fund_symbol,
            multisig_owners=multisig_owners,
            vault_record_file=vault_record_file,
            simulate=simulate,
            unit_testing=unit_testing,
            logger=logger,
            any_asset=any_asset,
            trading_strategy_api_key=trading_strategy_api_key,
            hypersync_api_key=hypersync_api_key,
            etherscan_api_key=etherscan_api_key,
            verifier=verifier,
            verifier_url=verifier_url,
        )
        web3config.close()
        logger.info("All ok.")
        return

    # Single-chain deployment path (original flow)
    web3config.choose_single_chain()

    web3 = web3config.get_default()
    chain_id = ChainId(web3.eth.chain_id)

    logger.info("Connected to chain %s", chain_id.name)

    hot_wallet.sync_nonce(web3)
    web3.middleware_onion.add(construct_sign_and_send_raw_middleware(hot_wallet.account))

    assert not whitelisted_assets, "whitelisted_assets: Not implemented"
    whitelisted_asset_details = []

    # Check the chain is online
    logger.info(f"  Chain id is {web3.eth.chain_id:,}")
    logger.info(f"  Latest block is {web3.eth.block_number:,}")

    lagoon_chain_config = get_lagoon_chain_config(chain_id)

    if existing_vault_address:
        existing_vault = create_vault_instance(
            web3,
            existing_vault_address,
            {ERC4626Feature.lagoon_like},
        )
        logger.info("Deploying for existing vault %s", existing_vault.name)
        denomination_token = existing_vault.denomination_token
    else:
        denomination_token = fetch_erc20_details(
            web3,
            denomination_asset,
        )

    if simulate:
        logger.info("Simulation deployment")
    else:
        logger.info("Ready to deploy")

    chain_slug = chain_id.get_slug()
    log_deployment_preflight_report(
        hot_wallet=hot_wallet,
        chain_web3={chain_slug: web3},
        fund_name=fund_name,
        fund_symbol=fund_symbol,
        asset_manager=asset_manager,
        multisig_owners=multisig_owners,
        performance_fee=performance_fee,
        management_fee=management_fee,
        etherscan_api_key=etherscan_api_key,
        verifier=verifier,
        verifier_url=verifier_url,
        denomination_token=denomination_token,
        any_asset=any_asset,
        whitelisted_asset_details=whitelisted_asset_details,
        uniswap_v2=uniswap_v2,
        uniswap_v3=uniswap_v3,
        one_delta=one_delta,
        aave=aave,
        erc_4626_vaults=erc_4626_vaults,
        lagoon_chain_config=lagoon_chain_config,
        simulate=simulate,
        logger=logger,
    )

    if not (simulate or unit_testing):

        # Require API key for etherscan verifier, or verifier_url for blockscout
        if verifier == "etherscan" and not etherscan_api_key:
            raise RuntimeError("Etherscan API key needed for production deployments with etherscan verifier")
        if verifier == "blockscout" and not verifier_url:
            raise RuntimeError("Verifier URL needed for production deployments with blockscout verifier")

        confirm = input("Ok [y/n]? ")
        if not confirm.lower().startswith("y"):
            print("Aborted")
            sys.exit(1)

    # Currently assumes HotWallet = asset manager
    # as the trade-executor that deploys the vault is going to
    # the assset manager for this vault
    parameters = LagoonDeploymentParameters(
        underlying=denomination_token.address,
        name=fund_name,
        symbol=fund_symbol,
        performanceRate=performance_fee,
        managementRate=management_fee,

    )

    chain_slug = chain_id.get_slug()

    if uniswap_v2:
        uniswap_v2_deployment = fetch_deployment(
            web3,
            factory_address=UNISWAP_V2_DEPLOYMENTS[chain_slug]["factory"],
            router_address=UNISWAP_V2_DEPLOYMENTS[chain_slug]["router"],
            init_code_hash=UNISWAP_V2_DEPLOYMENTS[chain_slug]["init_code_hash"],
        )
    else:
        uniswap_v2_deployment = None

    if uniswap_v3:
        chain_slug = chain_id.get_slug()
        deployment_data = UNISWAP_V3_DEPLOYMENTS[chain_slug]
        uniswap_v3_deployment= fetch_deployment_uni_v3(
            web3,
            factory_address=deployment_data["factory"],
            router_address=deployment_data["router"],
            position_manager_address=deployment_data["position_manager"],
            quoter_address=deployment_data["quoter"],
            quoter_v2=deployment_data.get("quoter_v2", False),
            router_v2=deployment_data.get("router_v2", False),
        )
    else:
        uniswap_v3_deployment = None

    if aave:
        chain_slug = chain_id.get_slug()
        deployment_data = AAVE_V3_DEPLOYMENTS[chain_slug]
        assert "ausdc" in deployment_data, f"No aUSDC configuration: {AAVE_V3_DEPLOYMENTS}"
        aave_v3_deployment = fetch_aave_deployment(
            web3,
            pool_address=deployment_data["pool"],
            data_provider_address=deployment_data["data_provider"],
            oracle_address=deployment_data["oracle"],
            ausdc_address=deployment_data["ausdc"],
        )
    else:
        aave_v3_deployment = None

    # Scanning ERC-4626 vaults on a startup for token details takes a long time
    # Token cache already prepared at the start of the command
    logger.info("Using token cache at %s", token_cache.filename)

    if erc_4626_vaults:
        erc_4626_vault_addresses = [Web3.to_checksum_address(a.strip()) for a in erc_4626_vaults.split(",")]
        erc_4626_vaults = []
        for addr in erc_4626_vault_addresses:
            logger.info("Resolving ERC-4626 vault at %s", addr)
            vault = cast(ERC4626Vault, create_vault_instance(web3, addr, token_cache=token_cache))
            assert vault.is_valid(), f"Invalid ERC-4626 vault at {addr}"
            logger.info("Preparing vault %s for whitelisting", vault.name)
            erc_4626_vaults.append(vault)

    # Capture block before deployment so the report can find guard config events
    # (deploy_info.block_number is set AFTER deployment, missing all events)
    pre_deploy_block = web3.eth.block_number

    deploy_info = deploy_automated_lagoon_vault(
        web3=web3,
        deployer=hot_wallet,
        asset_manager=asset_manager,
        parameters=parameters,
        safe_owners=multisig_owners,
        safe_threshold=len(multisig_owners) - 1,
        uniswap_v2=uniswap_v2_deployment,
        uniswap_v3=uniswap_v3_deployment,
        aave_v3=aave_v3_deployment,
        any_asset=any_asset,
        use_forge=True,
        etherscan_api_key=etherscan_api_key,
        verifier=verifier,
        verifier_url=verifier_url,
        guard_only=guard_only,
        existing_vault_address=existing_vault_address,
        existing_safe_address=existing_safe_address,
        erc_4626_vaults=erc_4626_vaults,
        factory_contract=lagoon_chain_config.factory_contract,
        from_the_scratch=lagoon_chain_config.from_the_scratch,
        cowswap=cowswap,
    )

    if vault_record_file and (not simulate):
        # Make a small file, mostly used to communicate with unit tests
        with open(vault_record_file, "wt") as out:
            out.write(deploy_info.pformat())

        with open(vault_record_file.with_suffix(".json"), "wt") as out:
            out.write(json.dumps(deploy_info.get_deployment_data()))

        logger.info("Wrote %s for vault details", os.path.abspath(vault_record_file))
    else:
        logger.info("Skipping record file because of simulation")

    logger.info("Token cache %s contains %d entries", token_cache.filename, len(token_cache))

    if not guard_only:
        logger.info("Lagoon deployed:\n%s", deploy_info.pformat())
    else:
        logger.info("New guard deployed: %s", deploy_info.trading_strategy_module.address)
        logger.info("Old guard address: %s", vault_adapter_address)
        logger.info("Safe address: %s", deploy_info.safe.address)
        logger.info("Vault address: %s", deploy_info.vault.address)
        mods = deploy_info.safe.retrieve_modules()
        logger.info("Currently enabled Safe modules: %s", mods)
        assert len(mods) == 1, f"Expected only one module enabled, got: {mods}"
        logger.info("Safe transactions needed:")
        logger.info("1. %s.disableModule(%s, %s)", deploy_info.safe.address, ONE_ADDRESS_STR, deploy_info.old_trading_strategy_module.address)
        logger.info("2. %s.enableModule(%s)", deploy_info.safe.address, deploy_info.trading_strategy_module.address)
        logger.info("Safe ABI needed: %s", SAFE_ABI_STR)


    # Print deployment guard configuration report
    print_deployment_report(
        safe_address=deploy_info.safe_address or deploy_info.safe.address,
        module_address=deploy_info.trading_strategy_module.address,
        web3=web3,
        hypersync_api_key=hypersync_api_key,
        simulate=simulate,
        from_block=pre_deploy_block,
    )

    web3config.close()

    logger.info("All ok.")


def _deploy_multichain(
    web3config,
    hot_wallet: HotWallet,
    asset_manager: str,
    strategy_file: Path,
    safe_salt_nonce: int | None,
    fund_name: str | None,
    fund_symbol: str | None,
    multisig_owners: list[str] | None,
    vault_record_file: Path,
    simulate: bool,
    unit_testing: bool,
    logger,
    any_asset: bool = False,
    trading_strategy_api_key: str | None = None,
    hypersync_api_key: str | None = None,
    etherscan_api_key: str | None = None,
    verifier: str = "etherscan",
    verifier_url: str | None = None,
):
    """Deploy multichain Lagoon vault from a strategy file.

    Uses the strategy's trading universe to determine per-chain configurations.
    """

    if safe_salt_nonce is None:
        safe_salt_nonce = random.randint(1, 2**32)
        logger.info("Generated random safe_salt_nonce: %d", safe_salt_nonce)

    if not multisig_owners:
        multisig_owners = [hot_wallet.address]

    safe_threshold = max(1, len(multisig_owners) - 1)

    # Create TradingStrategy client if API key is available
    # (needed by strategies that fetch exchange/pair data from the API)
    client = None
    if trading_strategy_api_key:
        from tradingstrategy.client import Client
        client = Client.create_live_client(api_key=trading_strategy_api_key)

    # Load strategy module and create trading universe
    mod = read_strategy_module(strategy_file)
    universe = call_create_trading_universe(
        mod.create_trading_universe,
        client=client,
        universe_options=mod.get_universe_options(),
        execution_context=one_off_execution_context,
    )

    # Build chain_web3 mapping: {chain_slug: web3}
    chain_web3 = {}
    for chain_id, web3 in web3config.connections.items():
        chain_web3[chain_id.get_slug()] = web3

    logger.info("Multichain deployment: chains=%s, strategy=%s", list(chain_web3.keys()), strategy_file)

    # Sync hot wallet nonce on each chain
    for slug, web3 in chain_web3.items():
        hot_wallet.sync_nonce(web3)
        web3.middleware_onion.add(construct_sign_and_send_raw_middleware(hot_wallet.account))
        logger.info("  Chain %s: block %d", slug, web3.eth.block_number)

    # Generate per-chain configs from the strategy universe
    configs = translate_trading_universe_to_lagoon_config(
        universe=universe,
        chain_web3=chain_web3,
        asset_manager=asset_manager,
        safe_owners=multisig_owners,
        safe_threshold=safe_threshold,
        safe_salt_nonce=safe_salt_nonce,
        fund_name=fund_name or "Strategy Vault",
        fund_symbol=fund_symbol or "CSV",
        any_asset=any_asset,
    )

    chain_word = "chain" if len(configs) == 1 else "chains"
    logger.info("Generated configs for %d %s:", len(configs), chain_word)

    log_deployment_preflight_report(
        hot_wallet=hot_wallet,
        chain_web3=chain_web3,
        fund_name=fund_name or "Strategy Vault",
        fund_symbol=fund_symbol or "CSV",
        asset_manager=asset_manager,
        multisig_owners=multisig_owners,
        performance_fee=configs[next(iter(configs))].parameters.performanceRate,
        management_fee=configs[next(iter(configs))].parameters.managementRate,
        etherscan_api_key=etherscan_api_key,
        verifier=verifier,
        verifier_url=verifier_url,
        chain_configs=configs,
        simulate=simulate,
        logger=logger,
    )

    if not (simulate or unit_testing):
        label = "multichain vault" if len(configs) > 1 else "vault"
        confirm = input(f"Deploy {label}? [y/n] ")
        if not confirm.lower().startswith("y"):
            print("Aborted")
            sys.exit(1)

    # Capture block before deployment so the report can find guard config events
    pre_deploy_blocks = {slug: w3.eth.block_number for slug, w3 in chain_web3.items()}

    # Deploy across all chains
    result = deploy_multichain_lagoon_vault(
        chain_web3=chain_web3,
        deployer=hot_wallet.account,
        chain_configs=configs,
    )

    logger.info("Deployment complete")
    logger.info("Safe address: %s", result.deployments[next(iter(result.deployments))].safe_address)

    # Write deployment record
    if vault_record_file and not simulate:
        deployment_data = {
            "multichain": True,
            "safe_salt_nonce": safe_salt_nonce,
            "deployments": {},
        }
        for slug, dep in result.deployments.items():
            deployment_data["deployments"][slug] = {
                "vault_address": dep.vault.address if hasattr(dep.vault, "address") else None,
                "safe_address": dep.safe_address,
                "module_address": dep.trading_strategy_module.address if dep.trading_strategy_module else None,
                "is_satellite": dep.is_satellite,
            }

        # Write human-readable summary
        with open(vault_record_file, "wt") as out:
            for slug, dep in result.deployments.items():
                out.write(f"Chain: {slug}\n")
                out.write(f"  Satellite: {dep.is_satellite}\n")
                out.write(f"  Safe: {dep.safe_address}\n")
                if not dep.is_satellite:
                    out.write(f"  Vault: {dep.vault.address}\n")
                if dep.trading_strategy_module:
                    out.write(f"  Module: {dep.trading_strategy_module.address}\n")
                out.write("\n")

        # Write machine-readable JSON
        with open(vault_record_file.with_suffix(".json"), "wt") as out:
            out.write(json.dumps(deployment_data, indent=2))

        logger.info("Wrote deployment record to %s", os.path.abspath(vault_record_file))

    for slug, dep in result.deployments.items():
        kind = "satellite" if dep.is_satellite else "source"
        logger.info("Lagoon deployed on %s (%s):\n%s", slug, kind, dep.pformat())

    # Print deployment guard configuration report for each chain
    for slug, dep in result.deployments.items():
        if dep.trading_strategy_module and slug in chain_web3:
            print_deployment_report(
                safe_address=dep.safe_address,
                module_address=dep.trading_strategy_module.address,
                web3=chain_web3[slug],
                hypersync_api_key=hypersync_api_key,
                simulate=simulate,
                from_block=pre_deploy_blocks.get(slug, 0),
            )


SAFE_ABI_STR = """
[
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "module",
          "type": "address"
        }
      ],
      "name": "enableModule",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "prevModule",
          "type": "address"
        },
        {
          "internalType": "address",
          "name": "module",
          "type": "address"
        }
      ],
      "name": "disableModule",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    }
]
"""
