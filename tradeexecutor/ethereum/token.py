"""ERC-20 token data helpers."""

from web3 import Web3

from eth_defi.token import TokenDetails, fetch_erc20_details
from tradeexecutor.state.identifier import AssetIdentifier


def fetch_token_as_asset(web3: Web3, contract_address: str) -> AssetIdentifier:
    """Get ERC-20 token details suitable for persistent stroage.

    :param contract_address:
        Address of an ERC-20 contract

    :return:
        Asset identifier that can be use with persistent storage.
    """
    token = fetch_erc20_details(web3, contract_address)
    return translate_token_details(token)


def translate_token_details(token: TokenDetails) -> AssetIdentifier:
    """Translate on-chain fetched ERC-20 details to the persistent format.

    :param token:
        On-chain token data

    :return:
        Persistent asset identifier
    """
    web3 = token.contract.w3
    return AssetIdentifier(
        chain_id=web3.eth.chain_id,
        address=token.address,
        token_symbol=token.symbol,
        decimals=token.decimals,
        internal_id=None,
        info_url=None,
    )
