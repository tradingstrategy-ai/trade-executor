"""Vault security related common functions for CLI commands"""
from web3 import Web3

from eth_defi.enzyme.deployment import EnzymeDeployment, ETHEREUM_DEPLOYMENT, POLYGON_DEPLOYMENT, ARBITRUM_DEPLOYMENT
from eth_defi.hotwallet import HotWallet
from eth_defi.token import fetch_erc20_details, TokenDetails
from tradingstrategy.chain import ChainId

SUPPORTED_DENOMINATION_TOKENS  = ("USDC", "USDC.e", "USDT")


def generate_whitelist(web3, whitelisted_assets: str) -> list[TokenDetails]:
    """Look up token details we are about to whitelist.

    :param whitelisted_assets:
        Space separated list.

        Assets to whitelist, first must be the vault denomination token.

    :return:
        Tuple (Token list, address of USDC token)

    """
    # Build the list of whitelisted assets GuardV0 allows us to trade
    whitelisted_asset_details = []
    for token_address in whitelisted_assets.split():
        token_address = token_address.strip()
        if token_address:
            whitelisted_asset_details.append(fetch_erc20_details(web3, token_address))

    assert len(whitelisted_asset_details) >= 1, "You need to whitelist at least one token as a trading pair"
    assert whitelisted_asset_details[0].symbol in SUPPORTED_DENOMINATION_TOKENS, f"Unsuppored denomination token for Enzyme: {whitelisted_assets[0]}"
    return whitelisted_asset_details


def get_enzyme_deployment(
    web3: Web3,
    chain_id: ChainId,
    deployer: HotWallet,
    comptroller_lib: str | None = None,
    allowed_adapters_policy: str | None = None,
) -> EnzymeDeployment:
    """

    :param chain_id:

    :param comptroller_lib:
        For unit test deployment

    :return:
    """

    # No other supported Enzyme deployments
    match chain_id:
        case ChainId.ethereum:
            deployment_info = ETHEREUM_DEPLOYMENT
            enzyme_deployment = EnzymeDeployment.fetch_deployment(web3, ETHEREUM_DEPLOYMENT, deployer=deployer.address)
            #denomination_token = fetch_erc20_details(web3, deployment_info["usdc"])
        case ChainId.polygon:
            enzyme_deployment = EnzymeDeployment.fetch_deployment(web3, POLYGON_DEPLOYMENT, deployer=deployer.address)
            # denomination_token = fetch_erc20_details(web3, deployment_info["usdc"])
        case ChainId.arbitrum:
            enzyme_deployment = EnzymeDeployment.fetch_deployment(web3, ARBITRUM_DEPLOYMENT, deployer=deployer.address)
            # denomination_token = fetch_erc20_details(web3, deployment_info["usdc"])
        case _:
            # Local unit test deployment.
            # Because addresses are random, they need to be explicitly passed
            # to any command line command
            assert comptroller_lib, f"You need to give Enzyme's ComptrollerLib address for a chain {chain_id}"
            enzyme_deployment = EnzymeDeployment.fetch_deployment(
                web3,
                {
                    "comptroller_lib": comptroller_lib,
                    "allowed_adapters_policy": allowed_adapters_policy,
                },
                deployer=deployer.address
            )

    return enzyme_deployment