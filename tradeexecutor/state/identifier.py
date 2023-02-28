"""Asset and trading pair identifiers.

How executor internally knows how to connect trading pairs in data and in execution environment (on-chain).
"""
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from dataclasses_json import dataclass_json
from eth_typing import HexAddress
from tradingstrategy.chain import ChainId
from web3 import Web3

from tradeexecutor.state.types import JSONHexAddress


@dataclass_json
@dataclass
class AssetIdentifier:
    """Identify a blockchain asset for trade execution.

    As internal token_ids and pair_ids may be unstable, trading pairs and tokens are explicitly
    referred by their smart contract addresses when a strategy decision moves to the execution.
    We duplicate data here to make sure we have a persistent record that helps to diagnose the sisues.
    """

    #: See https://chainlist.org/
    chain_id: int

    #: Smart contract address of the asset.
    #: Always lowercase.
    address: JSONHexAddress

    #: The ticker symbol of this token.
    token_symbol: str

    #: How many tokens this decimals.
    #: Must be always set and non-negative.
    decimals: int

    #: How this asset is referred in the internal database
    internal_id: Optional[int] = None

    #: Info page URL for this asset
    info_url: Optional[str] = None

    def __str__(self):
        return f"<{self.token_symbol} at {self.address}>"

    def __hash__(self):
        assert self.chain_id, "chain_id needs to be set to be hashable"
        assert self.address, "address needs to be set to be hashable"
        return hash((self.chain_id, self.address))

    def __eq__(self, other):
        assert isinstance(other, TradingPairIdentifier), f"Got {other}"
        return self.chain_id == other.chain_id and self.address == other.address

    def __post_init__(self):
        assert type(self.address) == str, f"Got address {self.address} as {type(self.address)}"
        assert self.address.startswith("0x")
        self.address= self.address.lower()
        assert type(self.chain_id) == int
        assert type(self.decimals) == int, f"Bad decimals {self.decimals}"
        assert self.decimals >= 0

    def get_identifier(self) -> str:
        """Assets are identified by their smart contract address."""
        return self.address.lower()

    @property
    def checksum_address(self) -> HexAddress:
        """Ethereum madness."""
        return Web3.toChecksumAddress(self.address)

    def __eq__(self, other: "AssetIdentifier") -> bool:
        """Assets are considered be identical if they share the same smart contract address."""
        assert isinstance(other, AssetIdentifier)
        return self.address.lower() == other.address.lower()

    def convert_to_raw_amount(self, amount: Decimal) -> int:
        """Return any amount in token native units.

        Convert decimal to fixed point integer.
        """
        assert isinstance(amount, Decimal), "Input only exact numbers for the conversion, not fuzzy ones like floats"
        assert self.decimals is not None, f"Cannot perform human to raw token amount conversion, because no decimals given: {self}"
        return int(amount * Decimal(10**self.decimals))

    def convert_to_decimal(self, raw_amount: int) -> Decimal:
        assert self.decimals is not None, f"Cannot perform human to raw token amount conversion, because no decimals given: {self}"
        return Decimal(raw_amount) / Decimal(10**self.decimals)



@dataclass_json
@dataclass
class TradingPairIdentifier:
    """Uniquely identify one trading pair across all tradeable blockchain assets.

    - Tokens are converted from machine readable token0 - token1 pair
      to more human-friendly base and quote token pair.
      See :ref:`conversion <trading pair>`.

    - This class is a data class that is copy-by-value.
      We copy both machine-readable information (smart contract addresses)
      and human readable information (symbols), as both are important
      to store for the persistent use - we do not expect to be able to lookup
      the information again with smart contract addresses in the future,
      as API access is expensive and blockchains may permanently be abandon.

    """

    #: Base token in this trading pair
    base: AssetIdentifier

    #: Quote token in this trading pair
    quote: AssetIdentifier

    #: Smart contract address of the pool contract.
    pool_address: str

    #: Exchange address.
    #: Identifies a decentralised exchange.
    #: Uniswap v2 likes are identified by their factor address.
    exchange_address: str

    #: How this asset is referred in the internal database
    #:
    #: Internal ids are not stable over the long duration.
    #: Internal ids are not also stable across different oracles.
    #: Always use `(chain_id, pool_address)` pair for persistent lookups.
    #:
    internal_id: Optional[int] = None

    #: What is the internal exchange id of this trading pair.
    internal_exchange_id: Optional[int] = None

    #: Info page URL for this trading pair e.g. with the price charts
    info_url: Optional[str] = None

    #: Trading fee for this pair.
    #:
    #: Uniswap v3 style fee where all trades share the same LP fees.
    #:
    fee: Optional[float] = None

    def __post_init__(self):
        assert self.base.chain_id == self.quote.chain_id, "Cross-chain trading pairs are not possible"
        assert (type(self.fee) in {float, type(None)}) or (self.fee == 0)

    def __repr__(self):
        fee = self.fee or 0
        return f"<Pair {self.base.token_symbol}-{self.quote.token_symbol} at {self.pool_address} ({fee * 100:.4f}% fee) on exchange {self.exchange_address}>"

    def __hash__(self):
        assert self.internal_id, "Internal id needed to be hashable"
        return self.internal_id

    def __eq__(self, other):
        assert isinstance(other, TradingPairIdentifier), f"Got {other}"
        return self.base == other.base and self.quote == other.quote

    @property
    def chain_id(self) -> int:
        """Return raw chain id.

        Get one from the base token, beacuse both tokens are on the same chain.
        """
        return self.base.chain_id

    def get_identifier(self) -> str:
        """We use the smart contract pool address to uniquely identify trading positions.

        Ethereum address is lowercased, not checksummed.
        """
        return self.pool_address.lower()

    def get_ticker(self) -> str:
        """Return base token symbol - quote token symbol human readable ticket."""
        return f"{self.base.token_symbol}-{self.quote.token_symbol}"

    def get_human_description(self) -> str:
        """Same as get_ticker()."""
        return self.get_ticker()

    def has_complete_info(self) -> bool:
        """Check if the pair has good information.

        Both base and quote token must have

        - Symbol

        - Decimals

        This check is mainly useful to filter out crap tokens
        from the trading decisions.
        """
        return (self.base.decimals > 0 and
                self.base.token_symbol and
                self.quote.decimals > 0 and
                self.quote.token_symbol)
