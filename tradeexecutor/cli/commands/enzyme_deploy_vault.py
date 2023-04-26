"""enzyme-asset-list CLi command."""

import json
from typing import Optional

from typer import Option

from eth_defi.deploy import deploy_contract
from eth_defi.enzyme.deployment import POLYGON_DEPLOYMENT, EnzymeDeployment
from eth_defi.hotwallet import HotWallet
from eth_defi.token import fetch_erc20_details
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
    vault_record_file: Optional[str] = Option(..., envvar="VAULT_RECORD_FILE", help="Store vault and comptroller addresses in this JSON file. It's important to write down all contract addresses."),
    fund_name: Optional[str] = Option(..., envvar="FUND_NAME", help="On-chain name for the fund shares"),
    fund_symbol: Optional[str] = Option(..., envvar="FUND_SYMBOL", help="On-chain token symbol for the fund shares"),
    comptroller_lib: Optional[str] = Option(None, envvar="COMPTROLLER_LIB", help="Enzyme's ComptrollerLib address for custom deployments"),
    denomination_asset: Optional[str] = Option(None, envvar="DENOMINATION_ASSET", help="Stablecoin asset used for vault denomination"),
):
    """Deploy a new Enzyme vault.

    Add adapters for the vault and configure it for automated trading.
    The account with the private key is set as the owner of the vault.
    Vault denomination asset is set to USDC.
    """

    logger = setup_logging(log_level)

    web3config = create_web3_config(
        json_rpc_binance=json_rpc_binance,
        json_rpc_polygon=json_rpc_polygon,
        json_rpc_avalanche=json_rpc_avalanche,
        json_rpc_ethereum=json_rpc_ethereum,
        json_rpc_anvil=json_rpc_anvil,
        json_rpc_arbitrum=json_rpc_arbitrum,
    )

    if not web3config.has_any_connection():
        raise RuntimeError("Vault deploy requires that you pass JSON-RPC connection to one of the networks")

    web3config.choose_single_chain()

    web3 = web3config.get_default()
    chain_id = ChainId(web3.eth.chain_id)

    logger.info("Connected to chain %s", chain_id.name)

    hot_wallet = HotWallet.from_private_key(private_key)
    web3.middleware_onion.add(construct_sign_and_send_raw_middleware(hot_wallet.account))

    # No other supported Enzyme deployments
    match chain_id:
        case ChainId.ethereum:
            raise NotImplementedError("Not supported yet")
        case ChainId.polygon:
            deployment_info = POLYGON_DEPLOYMENT
            enzyme_deployment = EnzymeDeployment.fetch_deployment(web3, POLYGON_DEPLOYMENT)
            denomination_token = fetch_erc20_details(web3, deployment_info["usdc"])
        case _:
            assert comptroller_lib, f"You need to give Enzyme's ComptrollerLib address for a chain {chain_id}"
            assert denomination_asset, f"You need to give denomination_asset for a chain {chain_id}"
            enzyme_deployment = EnzymeDeployment.fetch_deployment(web3, {"comptroller_lib": comptroller_lib})
            denomination_token = fetch_erc20_details(web3, denomination_asset)

    # Check the chain is online
    logger.info(f"  Chain id is {web3.eth.chain_id:,}")
    logger.info(f"  Latest block is {web3.eth.block_number:,}")

    # Check balances
    logger.info("Balance details")
    logger.info("  Hot wallet is %s", hot_wallet.address)
    gas_balance = web3.eth.get_balance(hot_wallet.address) / 10**18
    logger.info("  We have %f tokens for gas left", gas_balance)

    logger.info("Enzyme details")
    logger.info("  Integration manager deployed at %s", enzyme_deployment.contracts.integration_manager.address)
    logger.info("  %s is %s", denomination_token.symbol, denomination_token.address)

    logger.info("Deploying vault")
    try:
        # TODO: Fix this later, use create_new_vault() argument
        enzyme_deployment.deployer = hot_wallet.address

        comptroller_contract, vault_contract = enzyme_deployment.create_new_vault(
            hot_wallet.address,
            denomination_asset=denomination_token.contract,
            fund_name=fund_name,
            fund_symbol=fund_symbol,
        )
    except Exception as e:
        raise RuntimeError(f"Deployment failed. Hot wallet: {hot_wallet.address}, denomination asset: {denomination_token.address}") from e

    logger.info("Deploying VaultSpecificGenericAdapter")
    generic_adapter = deploy_contract(
        web3,
        f"VaultSpecificGenericAdapter.json",
        hot_wallet.address,
        enzyme_deployment.contracts.integration_manager.address,
        vault_contract.address,
    )

    block_number = web3.eth.block_number

    if vault_record_file:
        with open(vault_record_file, "wt") as out:
            vault_record = {
                "vault": vault_contract.address,
                "comptroller": comptroller_contract.address,
                "generic_adapter": generic_adapter.address,
                "block_number": block_number,
            }
            json.dump(vault_record, out)

    logger.info("Vault details")
    logger.info("  Vault at %s", vault_contract.address)
    logger.info("  Comptroller at %s", comptroller_contract.address)
    logger.info("  GenericAdapter at %s", generic_adapter.address)
    logger.info("  Deployment block number is %d", block_number)
