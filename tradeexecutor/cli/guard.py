"""Vault security related common functions"""
from eth_defi.token import fetch_erc20_details, TokenDetails

SUPPORTED_DENOMINATION_TOKENS  = ("USDC", "USDC.e",)


def generate_whitelist(web3, whitelisted_assets: list[str]) -> list[TokenDetails]:
    """Look up token details we are about to whitelist.

    :param whilisted_assets:
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
    assert whitelisted_asset_details[0].symbol in SUPPORTED_DENOMINATION_TOKENS, f"Got {whitelisted_assets[0]}"
    return whitelisted_asset_details