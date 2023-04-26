"""Enzyme token support

- Manage lists of tokens Enzyme supported assets

- Run as a standlone script to product JSON copy-paste
"""
import datetime
import enum
import json
import os.path
from dataclasses import dataclass
from typing import Optional, Dict, List

from dataclasses_json import dataclass_json
from eth_typing import BlockNumber

from eth_defi.chain import fetch_block_timestamp
from tradingstrategy.chain import ChainId

from eth_defi.enzyme.price_feed import EnzymePriceFeed
from tradeexecutor.state.types import ZeroExAddress


DERIVATIVE_PREFIXES = (
    "am",  # Balancer? amUSDT
    "x",  # Staked sushi?  xSUSHI
    "mi", # ??? miMATIC
    "st",  # Staked matic? stMATIC
    "c",  # Curve deposit cUSDCv3
)

STABLECOIN_HINTS = (
    "USD",
    "EUR",
    "RAI"
)


class EnzymeAssetType(enum.Enum):

    #: Any token with normal price action
    token = "token"

    #: Stablecoin
    stablecoin = "stablecoin"

    #: Derivatice of some other token like staked ETH
    derivative = "derivative"

    #: Broken tokens
    broken = "broken"

    @staticmethod
    def classify_by_symbol(symbol: str | None) -> "EnzymeAssetType":
        """Classify a token

        .. note ::

            Work in progress

        :param symbol:
            Token symbol
        """

        if symbol is None:
            return EnzymeAssetType.broken

        if symbol.startswith(DERIVATIVE_PREFIXES):
            return EnzymeAssetType.derivative

        if any(hint in symbol for hint in STABLECOIN_HINTS):
            return EnzymeAssetType.stablecoin

        return EnzymeAssetType.token


@dataclass_json
@dataclass(frozen=True, slots=True)
class EnzymeAsset:
    """JSON'nable entry of Enzyme supported asset."""

    chain_id: ChainId

    #: Token symbol
    #:
    #: Note broken tokens may not have symbol
    #:
    symbol: Optional[str]

    type: EnzymeAssetType

    #: Token address
    primitive_address: ZeroExAddress

    chainlink_aggregator_address: ZeroExAddress

    added_block_number: BlockNumber

    added_at: datetime.datetime

    removed_block_number: Optional[BlockNumber] = None

    removed_at: Optional[datetime.datetime] = None

    def __hash__(self):
        return hash((self.chain_id.value, self.address))

    def __eq__(self, other):
        return self.chain_id == other.chain_id and self.address == other.address

    @staticmethod
    def convert_raw_feed(feed: EnzymePriceFeed) -> "EnzymeAsset":
        """Convert raw Enzyme price feed to JSON serialisable format."""
        web3 = feed.web3
        data = {
            "chain_id": ChainId(feed.web3.eth.chain_id),
            "type": EnzymeAssetType.classify_by_symbol(feed.primitive_token.symbol),
            "symbol": feed.primitive_token.symbol,
            "primitive_address": feed.primitive_token.address.lower(),
            "chainlink_aggregator_address": feed.chainlink_aggregator.address.lower(),
            "added_block_number": feed.added_block_number,
            "added_at": fetch_block_timestamp(web3, feed.added_block_number),
            "removed_block_number": feed.removed_block_number,
            "removed_at": fetch_block_timestamp(web3, feed.removed_block_number) if feed.removed_block_number else None,
        }
        return EnzymeAsset.from_dict(data)


def load_enzyme_asset_list(chain_id: ChainId) -> List[EnzymeAsset]:
    """Load hardcoded asset list for Enzyme.
    
    Regenerate asset files with:
    
    .. code-block:: shell

        poetry run trade-executor enzyme-asset-list

     
    :param chain_id: 

    :return:
        List of tokens
    """
    fname = chain_id.get_slug() + "_assets.json"
    path = os.path.join(os.path.dirname(__file__), fname)
    with open(path, "rt") as inp:
        data = json.load(inp)
        return [EnzymeAsset.from_dict(i) for i in data]


