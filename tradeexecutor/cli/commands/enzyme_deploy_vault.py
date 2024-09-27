"""enzyme-deploy-vault CLI command.

See :ref:`vault deployment` for the full documentation how to use this command.

Example how to manually test:

.. code-block:: shell

    export SIMULATE=true
    export FUND_NAME="Up only and then more"
    export FUND_SYMBOL="UP"
    export TERMS_OF_SERVICE_ADDRESS="0x24BB78E70bE0fC8e93Ce90cc8A586e48428Ff515"
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
import logging
import os.path
import sys
from io import StringIO
from pathlib import Path
from pprint import pformat
from typing import Optional

from typer import Option

from eth_defi.abi import get_deployed_contract
from eth_defi.deploy import deploy_contract
from eth_defi.enzyme.deployment import POLYGON_DEPLOYMENT, EnzymeDeployment, ETHEREUM_DEPLOYMENT
from eth_defi.enzyme.generic_adapter_vault import deploy_vault_with_generic_adapter
from eth_defi.hotwallet import HotWallet
from eth_defi.token import fetch_erc20_details
from tradeexecutor.cli.guard import get_enzyme_deployment, generate_whitelist
from tradeexecutor.monkeypatch.web3 import construct_sign_and_send_raw_middleware
from tradingstrategy.chain import ChainId

from tradeexecutor.cli.bootstrap import create_web3_config
from tradeexecutor.cli.commands import shared_options
from tradeexecutor.cli.commands.app import app
from tradeexecutor.cli.log import setup_logging


@app.command()
def enzyme_deploy_vault(
    log_level: str = shared_options.log_level,
    json_rpc_binance: Optional[str] = shared_options.json_rpc_binance,
    json_rpc_polygon: Optional[str] = shared_options.json_rpc_polygon,
    json_rpc_avalanche: Optional[str] = shared_options.json_rpc_avalanche,
    json_rpc_ethereum: Optional[str] = shared_options.json_rpc_ethereum,
    json_rpc_arbitrum: Optional[str] = shared_options.json_rpc_arbitrum,
    json_rpc_anvil: Optional[str] = shared_options.json_rpc_anvil,
    private_key: str = shared_options.private_key,

    # Vault options
    vault_record_file: Optional[Path] = Option(..., envvar="VAULT_RECORD_FILE", help="Store vault and comptroller addresses in this JSON file. It's important to write down all contract addresses."),
    fund_name: Optional[str] = Option(..., envvar="FUND_NAME", help="On-chain name for the fund shares"),
    fund_symbol: Optional[str] = Option(..., envvar="FUND_SYMBOL", help="On-chain token symbol for the fund shares"),
    comptroller_lib: Optional[str] = shared_options.comptroller_lib,
    denomination_asset: Optional[str] = Option(None, envvar="DENOMINATION_ASSET", help="Stablecoin asset used for vault denomination"),

    owner_address: Optional[str] = Option(None, envvar="OWNER_ADDRESS", help="The protocol or multisig address that is set as the owner of the vault"),
    terms_of_service_address: Optional[str] = Option(None, envvar="TERMS_OF_SERVICE_ADDRESS", help="The address of the terms of service smart contract"),
    whitelisted_assets: Optional[str] = Option(..., envvar="WHITELISTED_ASSETS", help="Space separarted list of ERC-20 addresses this vault can trade. Denomination asset does not need to be whitelisted separately."),

    unit_testing: bool = shared_options.unit_testing,
    production: bool = Option(False, envvar="PRODUCTION", help="Set production metadata flag true for the deployment."),
    simulate: bool = Option(False, envvar="SIMULATE", help="Simulate deployment using Anvil mainnet work, when doing manual deployment testing."),
    etherscan_api_key: Optional[str] = Option(None, envvar="ETHERSCAN_API_KEY", help="Etherscan API key need to verify the contracts on a production deployment."),
    one_delta: bool = Option(False, envvar="ONE_DELTA", help="Whitelist 1delta interaction with GuardV0 smart contract."),
    aave: bool = Option(False, envvar="AAVE", help="Whitelist Aave aUSDC deposits"),
):
    """Deploy a new Enzyme vault.

    Deploys a new Enzyme vault that is configured to be run with Trading Strategy Protocol.

    Multiple contracts will be deployed and verified in a blockchain explorer.
    The vault is configured with custom guard, deposit and terms of service contracts.
    """

    logger = setup_logging(log_level)

    web3config = create_web3_config(
        json_rpc_binance=json_rpc_binance,
        json_rpc_polygon=json_rpc_polygon,
        json_rpc_avalanche=json_rpc_avalanche,
        json_rpc_ethereum=json_rpc_ethereum,
        json_rpc_anvil=json_rpc_anvil,
        json_rpc_arbitrum=json_rpc_arbitrum,
        simulate=simulate,
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

    # Build the list of whitelisted assets GuardV0 allows us to trade
    if denomination_asset not in whitelisted_assets:
        # Unit test legacy hack
        whitelisted_assets = denomination_asset + " " + whitelisted_assets
    whitelisted_asset_details = generate_whitelist(web3, whitelisted_assets)
    assert len(whitelisted_asset_details) >= 1, "You need to whitelist at least one token as a trading pair"
    denomination_token = whitelisted_asset_details[0]

    enzyme_deployment = get_enzyme_deployment(
        web3,
        chain_id,
        hot_wallet,
        comptroller_lib=comptroller_lib
    )

    # Check the chain is online
    logger.info(f"  Chain id is {web3.eth.chain_id:,}")
    logger.info(f"  Latest block is {web3.eth.block_number:,}")

    if terms_of_service_address is not None:
        terms_of_service = get_deployed_contract(
            web3,
            "terms-of-service/TermsOfService.json",
            terms_of_service_address,
        )
        terms_of_service.functions.latestTermsOfServiceVersion().call()  # Check ABI matches or crash
    else:
        terms_of_service = None

    asset_manager_address = hot_wallet.address

    if owner_address is None:
        owner_address = hot_wallet.address

    if simulate:
        logger.info("Simulation deployment")
    else:
        logger.info("Ready to deploy")
    logger.info("-" * 80)
    logger.info("Deployer hot wallet: %s", hot_wallet.address)
    logger.info("Deployer balance: %f, nonce %d", hot_wallet.get_native_currency_balance(web3), hot_wallet.current_nonce)
    logger.info("Enzyme FundDeployer: %s", enzyme_deployment.contracts.fund_deployer.address)
    if enzyme_deployment.usdc is not None:
        logger.info("USDC: %s", enzyme_deployment.usdc.address)
    logger.info("Terms of service: %s", terms_of_service.address if terms_of_service else "-")
    logger.info("Fund: %s (%s)", fund_name, fund_symbol)
    logger.info("Whitelisted assets: %s", ", ".join([a.symbol for a in whitelisted_asset_details]))

    logger.info("Whitelisting 1delta: %s", one_delta)
    logger.info("Whitelisting Aave: %s", aave)

    if owner_address != hot_wallet.address:
        logger.info("Ownership will be transferred to %s", owner_address)
    else:
        logger.warning("Ownership will be retained at the deployer %s", hot_wallet.address)

    if asset_manager_address != hot_wallet.address:
        logger.info("Asset manager is %s", asset_manager_address)
    else:
        logger.info("No separate asset manager role set: will use the current hot wallet as the asset manager for the vault")

    logger.info("-" * 80)

    if not (simulate or unit_testing):
        confirm = input("Ok [y/n]? ")
        if not confirm.lower().startswith("y"):
            print("Aborted")
            sys.exit(1)

    # TODO: Hack
    if chain_id == ChainId.arbitrum:
        uniswap_v2 = False
    else:
        uniswap_v2 = True

    try:
        # Currently assumes HotWallet = asset manager
        # as the trade-executor that deploys the vault is going to
        # the assset manager for this vault
        vault = deploy_vault_with_generic_adapter(
            enzyme_deployment,
            hot_wallet,
            asset_manager=hot_wallet.address,
            owner=owner_address,
            terms_of_service=terms_of_service,
            denomination_asset=denomination_token.contract,
            fund_name=fund_name,
            fund_symbol=fund_symbol,
            whitelisted_assets=whitelisted_asset_details,
            etherscan_api_key=etherscan_api_key if not simulate else None,  # Only verify when not simulating
            production=production,
            one_delta=one_delta,
            aave=aave,
            uniswap_v2=uniswap_v2,
        )

    except Exception as e:

        logger.error("Failed to deploy, is_mainnet_fork(): %s", web3config.is_mainnet_fork())

        if web3config.is_mainnet_fork():
            # Try to get some useful debug info from Anvil
            web3config.close(logging.ERROR)

        logger.exception(e)  # The fancy traceback formatter does not do nested

        raise RuntimeError(f"Deployment failed. Hot wallet: {hot_wallet.address}.\nException: {e}") from e

    if vault_record_file and (not simulate):
        # Make a small file, mostly used to communicate with unit tests
        with open(vault_record_file, "wt") as out:
            vault_record = {
                "fund_name": fund_name,
                "fund_symbol": fund_symbol,
                "vault": vault.address,
                "comptroller": vault.comptroller.address,
                "generic_adapter": vault.generic_adapter.address,
                "block_number": vault.deployed_at_block,
                "usdc_payment_forwarder": vault.payment_forwarder.address,
                "guard": vault.guard_contract.address,
                "deployer": hot_wallet.address,
                "denomination_token": denomination_token.address,
                "terms_of_service": terms_of_service_address,
                "whitelisted_assets": whitelisted_assets,
                "asset_manager_address": asset_manager_address,
                "owner_address": owner_address,
            }
            json.dump(vault_record, out, indent=4)
        logger.info("Wrote %s for vault details", os.path.abspath(vault_record_file))
    else:
        logger.info("Skipping record file because of simulation")

    buf = StringIO()
    for key, value in vault.get_deployment_info():
        print(f"{key}={value}", file=buf)

    logger.info(
        "Vault environment variables for trade-executor init command:\n%s",
        buf.getvalue()
    )

    web3config.close()

