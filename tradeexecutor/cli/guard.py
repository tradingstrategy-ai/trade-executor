"""Vault security related common functions"""
from eth_defi.token import fetch_erc20_details, TokenDetails


def generate_whitelist(web3, whitelisted_assets: list[str]) -> tuple[list[TokenDetails], str]:
    """Look up token details we are about to whitelist.

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

    if whitelisted_asset_details[0].symbol == "USDC":
        # Unit test path
        usdc = whitelisted_asset_details[0].contract
        whitelisted_asset_details = whitelisted_asset_details[1:]
    else:
        # Will read from the chain
        usdc = None

    return whitelisted_asset_details, usdc