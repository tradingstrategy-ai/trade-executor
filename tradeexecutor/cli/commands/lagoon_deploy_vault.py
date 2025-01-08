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
from typing import Optional

from typer import Option


from eth_defi.hotwallet import HotWallet
from eth_defi.lagoon.deployment import LagoonDeploymentParameters, deploy_automated_lagoon_vault
from eth_defi.token import fetch_erc20_details
from eth_defi.uniswap_v2.constants import UNISWAP_V2_DEPLOYMENTS
from eth_defi.uniswap_v2.deployment import fetch_deployment
from eth_defi.uniswap_v3.constants import UNISWAP_V3_DEPLOYMENTS
from eth_defi.uniswap_v3.deployment import fetch_deployment as fetch_deployment_uni_v3

from tradeexecutor.cli.commands.shared_options import parse_comma_separated_list
from tradeexecutor.monkeypatch.web3 import construct_sign_and_send_raw_middleware
from tradingstrategy.chain import ChainId

from tradeexecutor.cli.bootstrap import create_web3_config
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
    vault_record_file: Optional[Path] = Option(..., envvar="VAULT_RECORD_FILE", help="Store vault and comptroller addresses in this JSON file. It's important to write down all contract addresses."),
    fund_name: Optional[str] = Option(..., envvar="FUND_NAME", help="On-chain name for the fund shares"),
    fund_symbol: Optional[str] = Option(..., envvar="FUND_SYMBOL", help="On-chain token symbol for the fund shares"),
    denomination_asset: Optional[str] = Option(..., envvar="DENOMINATION_ASSET", help="Stablecoin asset used for vault denomination"),
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
):
    """Deploy a new Lagoon vault.

    Deploys a new Lagoon vault, Safe and TradingStrategyModuleV0 guard for automated trading.

    TODO: Heavily under development.
    """

    assert any_asset, "Currently only any_asset configurations supported"

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
    logger.info("Multisig owners: %s", multisig_owners)
    logger.info("Etherscan API key: %s", etherscan_api_key)

    if etherscan_api_key:
        logger.info("Etherscan API key: %s", etherscan_api_key)
    else:
        logger.error("Etherscan API key missing")

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
            quoter_v2=deployment_data["quoter_v2"],
            router_v2=deployment_data["router_v2"],
        )

    else:
        uniswap_v3_deployment = None

    deploy_info = deploy_automated_lagoon_vault(
        web3=web3,
        deployer=hot_wallet.account,
        asset_manager=asset_manager,
        parameters=parameters,
        safe_owners=multisig_owners,
        safe_threshold=len(multisig_owners) - 1,
        uniswap_v2=uniswap_v2_deployment,
        uniswap_v3=uniswap_v3_deployment,
        any_asset=True,
        use_forge=True,
        etherscan_api_key=etherscan_api_key,
    )

    if vault_record_file and (not simulate):
        # Make a small file, mostly used to communicate with unit tests
        with open(vault_record_file, "wt") as out:
            vault_record = deploy_info.pformat()
            json.dump(vault_record, out, indent=4)
        logger.info("Wrote %s for vault details", os.path.abspath(vault_record_file))
    else:
        logger.info("Skipping record file because of simulation")

    logger.info("Lagoon deployed:\n%s", deploy_info.pformat())

    web3config.close()

