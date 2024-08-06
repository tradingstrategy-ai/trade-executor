"""deploy-guard CLI command.

See :ref:`vault deployment` for the full documentation how to use this command.

Example how to manually test:

.. code-block:: shell

    export SIMULATE=true
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

"""

import json
import logging
import os.path
import sys
from pathlib import Path
from pprint import pformat
from typing import Optional

from typer import Option

from eth_defi.abi import get_deployed_contract
from eth_defi.deploy import deploy_contract
from eth_defi.enzyme.deployment import POLYGON_DEPLOYMENT, EnzymeDeployment, ETHEREUM_DEPLOYMENT
from eth_defi.enzyme.generic_adapter_vault import deploy_guard as _deploy_guard
from eth_defi.hotwallet import HotWallet
from tradeexecutor.cli.guard import generate_whitelist
from tradeexecutor.monkeypatch.web3 import construct_sign_and_send_raw_middleware
from tradingstrategy.chain import ChainId

from tradeexecutor.cli.bootstrap import create_web3_config
from tradeexecutor.cli.commands import shared_options
from tradeexecutor.cli.commands.app import app
from tradeexecutor.cli.log import setup_logging


@app.command()
def deploy_guard(
    log_level: str = shared_options.log_level,
    json_rpc_binance: Optional[str] = shared_options.json_rpc_binance,
    json_rpc_polygon: Optional[str] = shared_options.json_rpc_polygon,
    json_rpc_avalanche: Optional[str] = shared_options.json_rpc_avalanche,
    json_rpc_ethereum: Optional[str] = shared_options.json_rpc_ethereum,
    json_rpc_arbitrum: Optional[str] = shared_options.json_rpc_arbitrum,
    json_rpc_anvil: Optional[str] = shared_options.json_rpc_anvil,
    private_key: str = shared_options.private_key,

    denomination_asset: Optional[str] = Option(None, envvar="DENOMINATION_ASSET", help="Stablecoin asset used for vault denomination"),
    owner_address: Optional[str] = Option(None, envvar="OWNER_ADDRESS", help="The protocol or multisig address that is set as the owner of the vault"),
    whitelisted_assets: Optional[str] = Option(..., envvar="WHITELISTED_ASSETS", help="Space separarted list of ERC-20 addresses this vault can trade. Denomination asset does not need to be whitelisted separately."),

    unit_testing: bool = shared_options.unit_testing,
    production: bool = Option(False, envvar="PRODUCTION", help="Set production metadata flag true for the deployment."),
    simulate: bool = Option(False, envvar="SIMULATE", help="Simulate deployment using Anvil mainnet work, when doing manual deployment testing."),
    etherscan_api_key: Optional[str] = Option(None, envvar="ETHERSCAN_API_KEY", help="Etherscan API key need to verify the contracts on a production deployment."),
    one_delta: bool = Option(False, envvar="ONE_DELTA", help="Whitelist 1delta interaction with GuardV0 smart contract."),
    aave: bool = Option(False, envvar="AAVE", help="Whitelist Aave aUSDC deposits"),

    vault_address: Optional[str] = shared_options.vault_address,
    vault_adapter_address: Optional[str] = shared_options.vault_adapter_address,

):
    """Deploy a new Guard smart contract.

    - Can be assigned to Enzyme vault

    - Can be assigned to a standalone multisig
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
    whitelisted_asset_details = generate_whitelist(web3, whitelisted_assets)
    denomination_token = whitelisted_asset_details[0]

    if vault_address:
        assert vault_adapter_address, f"Both vault_address and vault_adapter_address must be given"
        allow_sender = vault_adapter_address
        allow_receiver = vault_address
    else:
        allow_receiver = None
        allow_sender = None

    # Check the chain is online
    logger.info(f"  Chain id is {web3.eth.chain_id:,}")
    logger.info(f"  Latest block is {web3.eth.block_number:,}")
    logger.info(f"  Vault is {web3.eth.block_number:,}")

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
    logger.info("Whitelisted assets: %s", ", ".join([a.symbol for a in whitelisted_asset_details]))
    logger.info("Whitelisting 1delta and Aave: %s", one_delta)

    if owner_address != hot_wallet.address:
        logger.info("Ownership will be transferred to %s", owner_address)
    else:
        logger.warning("Ownership will be retained at the deployer %s", hot_wallet.address)

    if asset_manager_address != hot_wallet.address:
        logger.info("Asset manager is %s", asset_manager_address)
    else:
        logger.info("No separate asset manager role set: will use the current hot wallet as the asset manager")

    logger.info("-" * 80)

    if not (simulate or unit_testing):
        confirm = input("Ok [y/n]? ")
        if not confirm.lower().startswith("y"):
            print("Aborted")
            sys.exit(1)
    try:
        guard = _deploy_guard(
            web3=web3,
            deployer=hot_wallet,
            asset_manager=hot_wallet.address,
            owner=owner_address,
            denomination_asset=denomination_token.contract,
            whitelisted_assets=whitelisted_asset_details,
            etherscan_api_key=etherscan_api_key if not simulate else None,  # Only verify when not simulating
            one_delta=one_delta,
            aave=aave,
            allow_receiver=allow_receiver,
            allow_sender=allow_sender,
        )

    except Exception as e:

        logger.error("Failed to deploy, is_mainnet_fork(): %s", web3config.is_mainnet_fork())

        if web3config.is_mainnet_fork():
            # Try to get some useful debug info from Anvil
            web3config.close(logging.ERROR)

        logger.exception(e)  # TODO: Typer workaround

        raise RuntimeError(f"Deployment failed. Hot wallet: {hot_wallet.address}, denomination asset: {denomination_token.address}\n{e}") from e

    logger.info("Guard deployed at %s", guard.address)
    logger.info("All ok")

    web3config.close()

