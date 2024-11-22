"""ERC-20 token data helpers."""

from web3 import Web3

from eth_defi.token import TokenDetails, fetch_erc20_details
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.types import JSONHexAddress
from tradingstrategy.pair import PandasPairUniverse


class NonSpotPosition(Exception):
    """Portfolio has non-spot position when spot-only is supported"""


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


def create_spot_token_map_existing_positions(
    portfolio: Portfolio,
    raise_on_unsupported=True,
) -> dict[JSONHexAddress, TradingPosition]:
    """Create a map of spot ERC-20 tokens in our portfolio.

    :return:
        ERC-20 address -> position map for all open/frozen spot positions.
    """

    position_map = {}

    for p in portfolio.get_open_and_frozen_positions():
        if not p.is_spot():
            if raise_on_unsupported:
                raise NonSpotPosition(f"Not ERC-20 spot position: {p}")
            else:
                continue

        position_map[p.pair.base.address] = p

    return position_map


def create_spot_token_map_potential_positions(
    pair_universe: PandasPairUniverse,
    portfolio: Portfolio,
    raise_on_unsupported=True,
) -> dict[AssetIdentifier, TradingPosition | None]:
    """Create a map of spot ERC-20 tokens in our portfolio.

    :return:
        ERC-20 address -> position map for all open/frozen spot positions and potential positions.

        IF there is not yet existing position, return None.
    """

    position_map = create_spot_token_map_existing_positions(portfolio, raise_on_unsupported)

    # Include tokens for which we do not have a position mapped yet
    for pair in pair_universe.iterate_pairs():
        if pair.base_token_address not in position_map:
            position_map[pair.base_token_address] = None

    return position_map


