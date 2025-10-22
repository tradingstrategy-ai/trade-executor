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

    trade-executor enzyme-deploy-vault
"""

import json
import os.path
import sys
from pathlib import Path
from typing import Optional, cast

from typer import Option
from web3 import Web3

from eth_defi.aave_v3.constants import AAVE_V3_DEPLOYMENTS
from eth_defi.abi import ONE_ADDRESS_STR
from eth_defi.erc_4626.classification import create_vault_instance
from eth_defi.erc_4626.core import ERC4626Feature
from eth_defi.erc_4626.vault import ERC4626Vault
from eth_defi.hotwallet import HotWallet
from eth_defi.lagoon.config import get_lagoon_chain_config
from eth_defi.lagoon.deployment import LagoonDeploymentParameters, deploy_automated_lagoon_vault, DEFAULT_PERFORMANCE_RATE, DEFAULT_MANAGEMENT_RATE
from eth_defi.token import fetch_erc20_details
from eth_defi.uniswap_v2.constants import UNISWAP_V2_DEPLOYMENTS
from eth_defi.uniswap_v2.deployment import fetch_deployment
from eth_defi.uniswap_v3.constants import UNISWAP_V3_DEPLOYMENTS
from eth_defi.uniswap_v3.deployment import fetch_deployment as fetch_deployment_uni_v3
from eth_defi.aave_v3.deployment import fetch_deployment as fetch_aave_deployment

from tradeexecutor.cli.commands.shared_options import parse_comma_separated_list
from tradeexecutor.ethereum.token_cache import get_default_token_cache
from tradeexecutor.monkeypatch.web3 import construct_sign_and_send_raw_middleware
from tradingstrategy.chain import ChainId

from tradeexecutor.cli.bootstrap import create_web3_config, prepare_cache
from tradeexecutor.cli.commands import shared_options
from tradeexecutor.cli.commands.app import app
from tradeexecutor.cli.log import setup_logging


@app.command()
def lagoon_deploy_vault(
    log_level: str = shared_options.log_level,
    json_rpc_binance: Optional[str] = shared_options.json_rpc_binance,
    json_rpc_polygon: Optional[str] = shared_options.json_rpc_polygon,
    json_rpc_avalanche: Optional[str] = shared_options.json_rpc_avalanche,
    json_rpc_ethereum: Optional[str] = shared_options.json_rpc_ethereum,
    json_rpc_base: Optional[str] = shared_options.json_rpc_base,
    json_rpc_arbitrum: Optional[str] = shared_options.json_rpc_arbitrum,
    json_rpc_anvil: Optional[str] = shared_options.json_rpc_anvil,
    private_key: str = shared_options.private_key,

    # Vault options
    vault_record_file: Optional[Path] = Option(..., envvar="VAULT_RECORD_FILE", help="Store vault data in this TXT file, paired with a JSON file."),
    fund_name: Optional[str] = Option(None, envvar="FUND_NAME", help="On-chain name for the fund shares"),
    fund_symbol: Optional[str] = Option(None, envvar="FUND_SYMBOL", help="On-chain token symbol for the fund shares"),
    denomination_asset: Optional[str] = Option(None, envvar="DENOMINATION_ASSET", help="Stablecoin asset used for vault denomination"),
    multisig_owners: Optional[str] = Option(None, callback=parse_comma_separated_list, envvar="MULTISIG_OWNERS", help="The list of acconts that are set to the cosigners of the Safe. The multisig threshold is number of cosigners - 1."),
    # terms_of_service_address: Optional[str] = Option(None, envvar="TERMS_OF_SERVICE_ADDRESS", help="The address of the terms of service smart contract"),
    whitelisted_assets: Optional[str] = Option(None, envvar="WHITELISTED_ASSETS", help="Space separarted list of ERC-20 addresses this vault can trade. Denomination asset does not need to be whitelisted separately."),
    any_asset: Optional[bool] = Option(False, envvar="ANY_ASSET", help="Allow trading of any ERC-20 on Uniswap (unsecure)."),

    unit_testing: bool = shared_options.unit_testing,
    # production: bool = Option(False, envvar="PRODUCTION", help="Set production metadata flag true for the deployment."),
    simulate: bool = Option(False, envvar="SIMULATE", help="Simulate deployment using Anvil mainnet work, when doing manual deployment testing."),
    etherscan_api_key: Optional[str] = Option(None, envvar="ETHERSCAN_API_KEY", help="Etherscan API key need to verify the contracts on a production deployment."),
    one_delta: bool = Option(False, envvar="ONE_DELTA", help="Whitelist 1delta interaction with GuardV0 smart contract."),
    aave: bool = Option(False, envvar="AAVE", help="Whitelist Aave aUSDC deposits"),
    uniswap_v2: bool = Option(False, envvar="UNISWAP_V2", help="Whitelist Uniswap v2"),
    uniswap_v3: bool = Option(False, envvar="UNISWAP_V3", help="Whitelist Uniswap v3"),
    erc_4626_vaults: str = Option(None, envvar="ERC_4626_VAULTS", help="Whitelist ERC-4626 vaults, a command separated list of addresses"),
    verbose: bool = Option(False, envvar="VERBOSE", help="Extra verbosity with deploy commands"),
    performance_fee: int = Option(DEFAULT_PERFORMANCE_RATE, envvar="PERFORMANCE_FEE", help="Performance fee in BPS"),
    management_fee: int = Option(DEFAULT_MANAGEMENT_RATE, envvar="MANAGEMENT_FEE", help="Management fee in BPS"),
    guard_only: bool = Option(False, envvar="GUARD_ONLY", help="Deploys a new TradingStrategyModuleV0 guard with new settings. Lagoon multisig owners must then perform the transaction to enable this guard."),
    existing_vault_address: str = Option(None, envvar="EXISTING_VAULT_ADDRESS", help="When deploying a guard only, get the existing vault address."),
    existing_safe_address: str = Option(None, envvar="EXISTING_SAFE_ADDRESS", help="When deploying a guard only, get the existing safe address."),
    vault_adapter_address: str = shared_options.vault_adapter_address,
    cache_path: Optional[Path] = shared_options.cache_path,
):
    """Deploy a Lagoon vault or modify the vault deployment.

    Deploys a new Lagoon vault, Safe and TradingStrategyModuleV0 guard for automated trading.

    TODO: Heavily under development.
    """

    assert any_asset, "Currently only any_asset configurations supported"
    assert private_key, "PRIVATE_KEY not set"

    logger = setup_logging(log_level)

    web3config = create_web3_config(
        json_rpc_binance=json_rpc_binance,
        json_rpc_polygon=json_rpc_polygon,
        json_rpc_avalanche=json_rpc_avalanche,
        json_rpc_ethereum=json_rpc_ethereum,
        json_rpc_base=json_rpc_base,
        json_rpc_anvil=json_rpc_anvil,
        json_rpc_arbitrum=json_rpc_arbitrum,
        simulate=simulate,
        mev_endpoint_disabled=True,
    )

    if not web3config.has_any_connection():
        raise RuntimeError("Vault deploy requires that you pass JSON-RPC connection to one of the networks")

    web3config.choose_single_chain()

    web3 = web3config.get_default()
    chain_id = ChainId(web3.eth.chain_id)

    logger.info("Connected to chain %s", chain_id.name)

    hot_wallet = HotWallet.from_private_key(private_key)
    hot_wallet.sync_nonce(web3)
    web3.middleware_onion.add(construct_sign_and_send_raw_middleware(hot_wallet.account))

    assert not whitelisted_assets, "whitelisted_assets: Not implemented"
    whitelisted_asset_details = []

    # Check the chain is online
    logger.info(f"  Chain id is {web3.eth.chain_id:,}")
    logger.info(f"  Latest block is {web3.eth.block_number:,}")

    # TODO: Asset manager is now always the deployer
    asset_manager = hot_wallet.address

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

    logger.info("-" * 80)
    logger.info("Deployer hot wallet: %s", hot_wallet.address)
    logger.info("Deployer balance: %f, nonce %d", hot_wallet.get_native_currency_balance(web3), hot_wallet.current_nonce)
    logger.info("Fund: %s (%s)", fund_name, fund_symbol)
    logger.info("Underlying token: %s", denomination_token.symbol)
    logger.info("Whitelisting any token: %s", aave)
    logger.info("Whitelisted assets: %s", ", ".join([a.symbol for a in whitelisted_asset_details]))
    logger.info("Whitelisting Uniswap v2: %s", uniswap_v2)
    logger.info("Whitelisting Uniswap v3: %s", uniswap_v3)
    logger.info("Whitelisting 1delta: %s", one_delta)
    logger.info("Whitelisting Aave: %s", aave)
    logger.info("Whitelisting vaults: %s", erc_4626_vaults)
    logger.info("Multisig owners: %s", multisig_owners)
    logger.info("Performance fee: %f %%", performance_fee / 100)
    logger.info("Management fee: %f %%", management_fee / 100)
    logger.info("From the scratch Lagoon deployment: %s", lagoon_chain_config.from_the_scratch)
    logger.info("Use Lagoon BeaconProxyFactory: %s", lagoon_chain_config.factory_contract)

    if etherscan_api_key:
        logger.info("Etherscan API key: %s", etherscan_api_key)
    else:
        logger.error("Etherscan API key: not provided")

    if asset_manager != hot_wallet.address:
        logger.info("Asset manager is %s", asset_manager)
    else:
        logger.info("Hot wallet set for the asset manager role")

    logger.info("-" * 80)

    if not (simulate or unit_testing):

        # TODO: Move this bit somewhere else
        if not etherscan_api_key:
            raise RuntimeError("Etherscan API key needed for production deployments")

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
    token_cache = get_default_token_cache()
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
        any_asset=True,
        use_forge=True,
        etherscan_api_key=etherscan_api_key,
        guard_only=guard_only,
        existing_vault_address=existing_vault_address,
        existing_safe_address=existing_safe_address,
        erc_4626_vaults=erc_4626_vaults,
        factory_contract=lagoon_chain_config.factory_contract,
        from_the_scratch=lagoon_chain_config.from_the_scratch,
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


    web3config.close()
=
    logger.info("All ok.")
