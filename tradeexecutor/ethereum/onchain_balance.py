"""On-chain live balance reader"""

import logging
from typing import List, Iterable

from eth_defi.balances import fetch_erc20_balances_fallback
from eth_defi.chain import fetch_block_timestamp
from eth_defi.provider.broken_provider import get_block_tip_latency

from eth_typing import HexAddress
from web3 import Web3

from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.strategy.sync_model import OnChainBalance


logger = logging.getLogger(__name__)


def fetch_address_balances(
    web3: Web3,
    address: HexAddress | str,
    assets: List[AssetIdentifier],
    block_number: int | None = None,
    filter_zero=True,
) -> Iterable[OnChainBalance]:
    """Get token balances an address is holding.


    :param web3:
        Our web3 connection

    :param address:
        Ethereum address.

        Hot wallet or vault.

    :param assets:
        The asset list we checj.

    :param block_number:
        Optional historical block number when to do the scan.

        Needs an archive node for old blocks.

    :param filter_zero:
        Do not return zero balances
    """

    if not block_number:
        block_number = web3.eth.block_number - get_block_tip_latency(web3)

    timestamp = fetch_block_timestamp(web3, block_number)

    token_addresses = [asset.address for asset in assets]

    balances = fetch_erc20_balances_fallback(
        web3,
        address,
        token_addresses,
        block_identifier=block_number,
        decimalise=True,
    )

    for asset in assets:
        amount = balances[asset.address]

        if filter_zero and amount == 0:
            continue

        yield OnChainBalance(
            block_number=block_number,
            timestamp=timestamp,
            asset=asset,
            amount=amount,
        )
