"""Asset and trading pair identifiers.

How executor internally knows how to connect trading pairs in data and in execution environment (on-chain).
"""
import datetime
import enum
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional, Literal, TypeAlias

import numpy as np
from web3 import Web3
from dataclasses_json import dataclass_json, config
from eth_typing import HexAddress

from eth_defi.erc_4626.core import ERC4626Feature
from tradingstrategy.chain import ChainId
from tradingstrategy.lending import LendingProtocolType
from tradingstrategy.stablecoin import is_stablecoin_like
from tradingstrategy.token_metadata import TokenMetadata
from tradingstrategy.types import PrimaryKey
from tradingstrategy.exchange import ExchangeType

from tradeexecutor.utils.accuracy import sum_decimal, ensure_exact_zero, SUM_EPSILON
from tradeexecutor.state.types import JSONHexAddress, USDollarAmount, LeverageMultiplier, USDollarPrice, Percent

#: Asset unique id as a human-readable string.
#:
#: chain id - address tuple as string.
#:
#: Can be persisted.
#: Can be used in JSON serialisation.
#:
AssetFriendlyId: TypeAlias = str


class AssetType(enum.Enum):
    """What kind of asset is this.

    We mark special tokens that are dynamically created
    by lending protocols.
    """

    #: Normal ERC-20
    token = "token"

    #: ERC-20 aToken with dynamic balance()
    collateral = "collateral"

    #: ERC-20 vToken with dynamic balance()
    borrowed = "borrowed"



@dataclass_json
@dataclass
class AssetIdentifier:
    """Identify a blockchain asset for trade execution.

    This is pass-by-copy (as opposite to pass-by-reference) asset identifier
    we use across the persistent state. Because we copy a lot of information
    about asset, not just its id, this makes data reads and diagnosing problems
    simpler.

    As internal token_ids and pair_ids may be unstable, trading pairs and tokens are explicitly
    referred by their smart contract addresses when a strategy decision moves to the execution.
    We duplicate data here to make sure we have a persistent record that helps to diagnose the issues.

    Setting custom data:

    - Both :py:class:`AssetIdentifier` and :py:class:`TradingPairIdentifier` offer :py:attr:`AssetIdentifier.other_data` allowing you to set custom attributes.

    - Most useful of them is to categorise tokens with different tags, see :py:meth:`get_tags` and :py:meth:`set_tags`.

    - You must set these attributes in `create_trading_universe` function.

    - For more usage examples see `test_custom_labels`.

    A custom asset tagging example:

    .. code-block:: python

        def create_trading_universe(
            ts: datetime.datetime,
            client: Client,
            execution_context: ExecutionContext,
            universe_options: UniverseOptions,
        ) -> TradingStrategyUniverse:

            assert universe_options.start_at

            pairs = [
                (ChainId.polygon, "uniswap-v3", "WETH", "USDC", 0.0005)
            ]

            dataset = load_partial_data(
                client,
                execution_context=unit_test_execution_context,
                time_bucket=TimeBucket.d1,
                pairs=pairs,
                universe_options=default_universe_options,
                start_at=universe_options.start_at,
                end_at=universe_options.end_at,
            )

            strategy_universe = TradingStrategyUniverse.create_single_pair_universe(dataset)

            # IMPORTANT
            # Warm up must be called before any tags are set
            strategy_universe.warm_up_data()

            # Set custom labels on the token WETH on the trading pair
            weth_usdc = strategy_universe.get_pair_by_human_description(pairs[0])
            weth_usdc.base.set_tags({"L1", "EVM", "bluechip"})

            assert strategy_universe.data_universe.pairs.pair_map is not None, "Cache data structure missing?"

            weth_usdc = strategy_universe.get_pair_by_human_description(pairs[0])
            assert len(weth_usdc.base.get_tags()) > 0

            return strategy_universe

    Then how to read tag data and use it in trade decision making:

    .. code-block:: python

        def decide_trades(input: StrategyInput) -> list[TradeExecution]:
            # Show how to read pair and asset labels in decide_trade()
            for pair in input.strategy_universe.iterate_pairs():
                if "L1" in pair.get_tags():
                    # Do some trading logic for L1 tokens only
                    pass

            return []


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
    internal_id: Optional[PrimaryKey] = None

    #: Info page URL for this asset
    info_url: Optional[str] = None

    #: The underlying asset for aTokens, vTokens and such
    underlying: Optional["AssetIdentifier"] = None

    #: What kind of asset is this
    #:
    #: Legacy data will default to ``None``.
    #:
    type: Optional[AssetType] = None

    #: Aave liquidation threhold for this asset
    #:
    #: Set on aTokens that are used as collateral.
    #:
    liquidation_threshold: float | None = None

    #: User storeable properties.
    #:
    #: You can add any of your own metadata on the assets here.
    #:
    #: Be wary of the life cycle of the instances. The life time of the class instances
    #: tied to the trading universe that is recreated for every strategy cycle.
    #:
    #: See also :py:meth:`get_tags`.
    #:
    other_data: Optional[dict] = field(default_factory=dict)

    def __str__(self):
        if self.underlying:
            return f"<{self.token_symbol} ({self.underlying.token_symbol}) at {self.address}>"
        else:
            return f"<{self.token_symbol} at {self.address}>"

    def __hash__(self):
        assert self.chain_id is not None, "chain_id needs to be set to be hashable"
        assert self.address, "address needs to be set to be hashable"
        return hash((self.chain_id, self.address))

    def __eq__(self, other):
        assert isinstance(other, TradingPairIdentifier), f"Got {other}"
        return self.chain_id == other.chain_id and self.address == other.address

    def __post_init__(self):
        """Validate asset description initialisation."""
        assert type(self.address) == str, f"Got address {self.address} as {type(self.address)}"
        assert self.address.startswith("0x")
        self.address= self.address.lower()
        assert type(self.chain_id) == int
        if isinstance(self.decimals, np.int64):
            self.decimals = int(self.decimals)
        assert type(self.decimals) == int, f"Bad decimals {type(self.decimals)}: {self.decimals}"
        assert self.decimals >= 0

        if self.type:
            assert isinstance(self.type, AssetType), f"Got {self.type.__class__}: {self.type}"

    def get_identifier(self) -> AssetFriendlyId:
        """Assets are identified by their smart contract address.

        JSON/Human friendly format to give hash keys to assets,
        in the format chain id-address.

        :return:
            JSON friendly hask key
        """
        return f"{self.chain_id}-{self.address.lower()}"

    @property
    def checksum_address(self) -> HexAddress:
        """Ethereum madness."""
        return Web3.to_checksum_address(self.address)

    def __eq__(self, other: "AssetIdentifier") -> bool:
        """Assets are considered be identical if they share the same smart contract address."""
        assert isinstance(other, AssetIdentifier), f"Compared to wrong class: {other} {other.__class__}"
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

    def is_stablecoin(self) -> bool:
        """Do we think this asset reprents a stablecoin"""
        return is_stablecoin_like(self.token_symbol)

    def is_interest_accruing(self) -> bool:
        """Will this token gain on-chain interest thru rebase"""

        # TODO: this condition may change in the future when new asset types are introduced
        return self.underlying is not None

    def is_credit(self) -> bool:
        """Is this a credit asset that accrue interest for us"""
        return self.underlying and self.token_symbol.startswith("a")  # TODO: Hardcoded Aave v3

    def is_debt(self) -> bool:
        """Is this a credit asset that accrue interest for us"""
        assert self.underlying
        return self.token_symbol.startswith("v")  # TODO: Hardcoded Aave v3

    def get_pricing_asset(self) -> "AssetIdentifier":
        """Get the asset that delivers price for this asset.

        :return:

            If this asset is a derivative of another,
            then get the underlying, otherwise return self.
        """
        return self.underlying if self.underlying else self

    def get_tags(self) -> set[str]:
        """Return list of tags associated with this asset.

        - Used in basket construction strategies

        - Cen be source from CoinGecko, CoinMarketCap or hand labelled

        - Is Python :py:class:`set`

        - See also :py:meth:`TradingPairIdentifier.get_tags`

        - See also :py:meth:`set_tags`

        To set tags:

            asset.other_data["tags"] = {"L1", "memecoin"}

        :return:
            For WETH return e.g. [`L1`, `bluechip`]
        """
        return self.other_data.get("tags", set())

    def set_tags(self, tags: set[str]):
        """Set tags for this asset.

        - See also :py:meth:`get_tags`

        - See also :py:meth:`other_data`

        - Must be called in `create_trading_universe`

        - Be wary of `AssetIdentifier` life time as it is passed by value, not be reference,
          so you cannot update instance data after it has been copied to open positions, etc.

        - `translate_trading_pair()` is the critical method for understanding and managing identifier life times
        """
        assert type(tags) == set
        self.other_data["tags"] = tags

    def get_buy_tax(self) -> Percent | None:
        """Get buy tax associated with this token if any.

        .. note ::

            This property should be per trading pair, but all other DEX systems use per token tax values.
        """
        return self.other_data.get("buy_tax")

    def get_sell_tax(self) -> Percent | None:
        """Get sell tax associated with this token if any.

        .. note ::

            This property should be per trading pair, but all other DEX systems use per token tax values.

        """
        return self.other_data.get("sell_tax")


class TradingPairKind(enum.Enum):
    """What kind of trading position this is.

    - Spot markets are base:quote token pairs

    - Credit supplies are aToken:token pairs
    """

    #: Bought tokens from DEX
    #:
    spot_market_hold = "spot_market_hold"

    #: Bought rebalancing tokens from DEX
    #:
    #: E.g. buy stETH or aUSD directly through DEX,
    #: instead of thru vault/reserves deposit.
    #:
    spot_market_hold_rebalancing_token = "spot_market_rebalancing token"

    #: Supplying credit to Aave reserves/gaining interest
    #:
    credit_supply = "credit_supply"

    #: Leveraged long constructd using lending protocols
    #:
    lending_protocol_long = "lending_protocol_long"

    #: Leveraged short constructd using lending protocols
    #:
    lending_protocol_short = "lending_protocol_short"

    #: ERC-4626 vault or similar
    vault = "vault"

    #: Cash.
    #:
    #: Not a real trading pair, but expresses cash place holder in weighting calculations.
    cash = "cash"

    #: Freqtrade bot managed position.
    #:
    #: Capital allocated to an external Freqtrade bot instance.
    freqtrade = "freqtrade"

    def is_interest_accruing(self) -> bool:
        """Do base or quote or both gain interest during when the position is open."""
        return self in (TradingPairKind.lending_protocol_short, TradingPairKind.lending_protocol_long, TradingPairKind.credit_supply)

    def is_credit_based(self) -> bool:
        return self.is_interest_accruing()

    def is_credit_supply(self) -> bool:
        """This trading pair is for gaining interest."""
        return self == TradingPairKind.credit_supply

    def is_shorting(self) -> bool:
        """This trading pair is for shorting."""
        return self == TradingPairKind.lending_protocol_short

    def is_longing(self) -> bool:
        """This trading pair is for shorting."""
        return self == TradingPairKind.lending_protocol_long

    def is_leverage(self) -> bool:
        """This is a leverage trade on a lending protocol."""
        return self.is_shorting() or self.is_longing()

    def is_spot(self) -> bool:
        """This is a spot market pair."""
        return self == TradingPairKind.spot_market_hold or self == TradingPairKind.spot_market_hold_rebalancing_token

    def is_vault(self):
        return self == TradingPairKind.vault

    def is_cash(self):
        return self == TradingPairKind.cash

    def is_freqtrade(self):
        """This is a Freqtrade managed position."""
        return self == TradingPairKind.freqtrade


_TRANSIENT_OTHER_DATA_KEYS = {
    # Coingecko + TokenSniffer data combined
    "token_metadata",
    # Externally loaded Token Risk data
    "token_risk_data",
    # Unserialisable because uses raw Python objects
    "lagoon_compat_check_data",
}

def _reduce_other_data(val):
    """See translate_trading_pair() on how TradingPairIdentifier.other_data is populated"""
    if isinstance(val, dict):
        # Remove remote data from the dict
        for key in _TRANSIENT_OTHER_DATA_KEYS:
            if key in val:
                del val[key]
    return val


@dataclass_json
@dataclass(slots=True)
class TradingPairIdentifier:
    """Uniquely identify one trading pair across all tradeable blockchain assets.

    This is pass-by-copy (as opposite to pass-by-reference) trading pair identifier
    we use across the persistent state. Because we copy a lot of information
    about asset, not just its id, this makes data reads and diagnosing problems
    simpler.

    - Tokens are converted from machine readable token0 - token1 pair
      to more human-friendly base and quote token pair.
      See :ref:`conversion <trading pair>`.

    - This class is a data class that is a copy-by-value in the persistent state:
      We copy both machine-readable information (smart contract addresses)
      and human readable information (symbols), as both are important
      to store for the persistent use - we do not expect to be able to lookup
      the information again with smart contract addresses in the future,
      as API access is expensive and blockchains may permanently be abandon.


    - This class is preferred to be used as immutable, but
      sometimes it is handy to manually override :py:attr`fee`
      for different backtesting scenarios

    - This identifier is also used for :term:`lending protocols <lending protocol>`.
      In this case :py:attr:`base` is aToken like aUSDC. and :py:attr:`quote`
      is USDC.
    """

    #: Base token in this trading pair
    #:
    #: E.g. `WETH`.
    #:
    #: In leveraged positions this is borrowed asset with :py:attr:`AssetIdentifier.underlying` set.
    #:
    #:
    base: AssetIdentifier

    #: Quote token in this trading pair
    #:
    #: E.g. `USDC`
    #:
    #: In leveraged positions and credit supply positions, this is borrowed asset with :py:attr:`AssetIdentifier.underlying` set.
    #:
    quote: AssetIdentifier

    #: Smart contract address of the pool contract.
    #:
    #: - Uniswap v2 pair contract address
    #:
    #: - Uniswap v3 pool contract address
    #:
    #: Set to asset address for Aave pools.
    #:
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
    #: For synthetic pairs, like leveraged pairs on lending protocols,
    #: the internal id is the same as the underlying spot pair id.
    #: TODO: Confirm this, or missing?
    #:
    internal_id: Optional[PrimaryKey] = None

    #: What is the internal exchange id of this trading pair.
    internal_exchange_id: Optional[PrimaryKey] = None

    #: Info page URL for this trading pair e.g. with the price charts
    info_url: Optional[str] = None

    #: Trading fee for this pair.
    #:
    #: Liquidity provider fee expressed as the percent of the trade.
    #:
    #: E.g. `0.0030` for 0.30% fee.
    #:
    #: Should be filled for all Uniswap v2 and Uniswap v3 pairs.
    #: If the smaller Uni v2 forks do not have good data, 0.0030% is assumed.
    #:
    fee: Optional[float] = None

    #: The underlying token0/token1 for Uniswap pair is flipped compared to base token/quote token.
    #:
    #: Use :py:meth:`has_reverse_token_order` to access - might not be set.
    #: This is set when :py:class:`TradingPairIdentifier` is constructed.
    #:
    reverse_token_order: Optional[bool] = None

    #: What kind of position this is
    #:
    kind: TradingPairKind = TradingPairKind.spot_market_hold

    #: Underlying spot trading pair
    #: 
    #: This is used e.g. by alpha models to track the underlying pairs
    #: when doing leveraged positions.
    #:
    underlying_spot_pair: Optional["TradingPairIdentifier"] = None

    #: Exchange name where this pair trades on.
    #:
    #: May or may not be filled.
    #:
    exchange_name: Optional[str] = None

    #: User storeable properties.
    #:
    #: You can add any of your own metadata on the assets here.
    #:
    #: Be wary of the life cycle of the instances. The life time of the class instances
    #: tied to the trading universe that is recreated for every strategy cycle.
    #:
    #: .. warning::
    #:
    #:     To keep state size in check, other_data is not serialised.
    #:
    other_data: Optional[dict] = field(default_factory=dict, metadata=config(encoder=_reduce_other_data))

    def __post_init__(self):
        assert self.base.chain_id == self.quote.chain_id, "Cross-chain trading pairs are not possible"

        # float/int zero fix
        # TODO: Can be carefully removed later
        if self.fee == 0:
            self.fee = 0.0

        assert (type(self.fee) in {float, type(None)}) or (self.fee == 0)

        if self.reverse_token_order is None:
            # TODO: Make this lazy property
            self.reverse_token_order = int(self.base.address, 16) > int(self.quote.address, 16)

        if self.internal_id:
            assert isinstance(self.internal_id, int), f"Expected int, got {self.internal_id} of type {type(self.internal_id)}: {self.internal_id}"

    def __repr__(self):
        fee = self.fee or 0
        type_name = self.kind.name if self.kind else "spot"
        exchange_name = self.exchange_name if self.exchange_name else f"{self.exchange_address}"

        if self.is_vault():
            return f"<Pair for vault {self.get_vault_name()} on {self.get_vault_protocol()},  {self.base.token_symbol}-{self.quote.token_symbol} {exchange_name}, id #{self.internal_id}>"
        elif self.chain_id not in (ChainId.unknown, ChainId.centralised_exchange):
            # DEX pair
            return f"<Pair {self.base.token_symbol}-{self.quote.token_symbol} {type_name} at {self.pool_address} ({fee * 100:.4f}% fee) on exchange {exchange_name}>"
        else:
            # Backtesting with CEX data
            return f"<Pair {self.base.token_symbol}-{self.quote.token_symbol} {type_name} at {exchange_name}>"

    def __hash__(self):
        """Trading pair hash is hash(base, quote, fee).

        This might not hold true for all upcoming markets.
        """
        return hash((self.base.address, self.quote.address, self.fee))

    def __eq__(self, other: "TradingPairIdentifier | None"):
        # TODO: Include exchange slug in comparison
        if other is None:
            return False

        assert isinstance(other, TradingPairIdentifier), f"TradingPairIdentifier comparison failed, the other object is: {type(other)}"
        return self.base == other.base and self.quote == other.quote and self.fee == other.fee

    @property
    def chain_id(self) -> int:
        """Return raw chain id.

        Get one from the base token, beacuse both tokens are on the same chain.

        See also :py:class:`tradingstrategy.chain.ChainId`
        """
        return self.base.chain_id

    def get_identifier(self) -> str:
        """We use the smart contract pool address to uniquely identify trading positions.

        Ethereum address is lowercased, not checksummed.
        """
        return self.pool_address.lower()

    def get_ticker(self) -> str:
        """Return base token symbol - quote token symbol human readable ticket.

        Example: ``WETH-USDC``, ``

        See also :py:meth:`get_human_description`.
        """
        return f"{self.base.token_symbol}-{self.quote.token_symbol}"

    def get_human_description_for_code(self) -> str:
        """Get reverse of get_pair_byhuman_description() function.

        - Handle mainnet fork issues

        :return:
            E.g "(ChainId.polygon, "quickswap", "WETH", "USDC", 0.0030)

        """
        chain_id = ChainId(self.chain_id)
        assert chain_id != ChainId.anvil, f"Cannot use with anvil: {self}"
        return f"(ChainId.{chain_id.name}, {self.base.token_symbol}, {self.quote.token_symbol}, {self.fee})"

    def get_chart_label(self) -> str:
        """Get the short name used in charting for this pair.

        - Base token symbol if spot
        - Vault name if vault
        """
        if self.is_vault():
            return self.get_vault_name()
        return self.base.token_symbol

    def get_lending_protocol(self) -> LendingProtocolType | None:
        """Is this pair on a particular lending protocol."""
        if self.kind in (TradingPairKind.lending_protocol_short, TradingPairKind.lending_protocol_long):
            return LendingProtocolType.aave_v3
        return None

    def get_human_description(self, describe_type=False) -> str:
        """Get short ticker human description for this pair.

        :param describe_type:
            Handle spot, short and such pairs.

        See :py:meth:`get_ticker`.
        """

        if describe_type:
            underlying = self.underlying_spot_pair or self
            if self.is_short():
                return f"{underlying.get_ticker()} short"
            elif self.is_spot():
                return f"{self.get_ticker()} spot"
            elif self.is_credit_supply():
                return f"{underlying.get_ticker()} credit"
            elif self.is_vault():
                return self.get_vault_name()

        return self.get_ticker()

    def get_diagnostics_info(self, extended=False) -> dict:
        """Return data for debugginb about this pair.

        - Used for table formatting

        .. warning ::

            When TradingPairIdentifier is serialised, other_data is dicarded.

        :return:
            Human readable dict of various data points
        """

        metadata = self.get_token_metadata()

        data = {
            "internal_id": self.internal_id,
            "kind": self.kind.value,
            "chain": self.chain_id,
            "exchange": self.exchange_name,
            "address": self.pool_address,
            "base": self.base.token_symbol,
            "base_decimals": self.base.decimals,
            "base_address": self.base.address,
            "quote": self.quote.token_symbol,
            "quote_address": self.quote.address,
            "fee": self.fee,
            "buy_tax": self.get_buy_tax(),
            "sell_tax": self.get_sell_tax(),
            "risk_score": self.get_risk_score(),
            "vault_features": self.get_vault_features(),
        }

        if extended:
            data["coingecko_categories"] = self.get_coingecko_categories()

            if metadata:
                data["metadata_cached_on_disk"] = metadata.cached
                data["metadata_queried_at"] = metadata.queried_at

                tokensniffer_data = metadata.token_sniffer_data
                if tokensniffer_data:
                    data["tokensniffer_fetched_at"] = tokensniffer_data["data_fetched_at"]
                    data["tokensniffer_refreshed_at"] = datetime.datetime.utcfromtimestamp(int(tokensniffer_data["refreshed_at"] / 1000))
                    data["tokensniffer_token_created_at"] = tokensniffer_data["created_at"]
                    data["tokensniffer_token_flagged_at"] = tokensniffer_data["flagged_at"]

                coingecko_data = metadata.coingecko_data
                if coingecko_data:
                    data["coingecko_queried_at"] = coingecko_data["queried_at"]
                    data["coingecko_last_updated"] = coingecko_data["last_updated"]
                    data["coingecko_id"] = coingecko_data["id"]
                    data["coingecko_symbol"] = coingecko_data["symbol"]
                    data["coingecko_slug"] = coingecko_data["web_slug"]
                    data["coingecko_public_notice"] = coingecko_data["public_notice"]

        data = {k: str(v) for k, v in data.items()}

        return data

    def get_token_metadata(self) -> TokenMetadata | None:
        """Get full metadata for the base token.

        Not serialised.
        """
        return self.other_data.get("token_metadata")

    def get_minimal_tokesniffer_data(self) -> dict | None:
        """Always serialised if avail.

        See translate_trading_pair().
        """
        data = self.other_data.get("token_sniffer_data")
        if data:
            return data
        return None

    def get_token_risk_data(self) -> dict | None:
        """From Token Risk API by Hexen"""
        data = self.other_data.get("token_risk_data")
        if data:
            return data
        return None

    def get_lagoon_compat_check_data(self) -> "eth_defi.lagoon.lagoon_compatibility.LagoonTokenCompatibilityData | None":
        """Get Lagoon compatibility check data.

        - Internally generated by :py:func:`eth_defi.lagoon.lagoon_compatibility.check_lagoon_compatibility_with_database`
        """
        return self.other_data.get("lagoon_compat_check_data")

    def get_tokensniffer_data(self) -> dict | None:
        """From TokenSniffer APY by Solidly"""
        meta = self.get_token_metadata()
        if meta:
            return meta.token_sniffer_data
        return None

    def get_coingecko_categories(self) -> set | None:
        meta = self.get_token_metadata()
        if meta:
            return meta.get_coingecko_categories()
        return None

    def get_risk_score(self) -> int | None:
        """Refers to TokenSniffe risk score."""
        tokensniffer_data = self.get_minimal_tokesniffer_data()
        if tokensniffer_data:
            return tokensniffer_data["score"]
        return None

    def get_token_risk_score(self) -> int | None:
        """Refers to Token Risk API score."""
        data = self.get_token_risk_data()
        if data:
            return data["score"]
        return None

    def get_token_risk_flags(self) -> set[str] | None:
        """Get all flags set by Token Risk API."""
        data = self.get_token_risk_data()
        if not data:
            return None
        flags = set()
        for flag in data["results"]:
            if flag["value"] in ("true", True):
                flags.add(flag["key"])
        return flags

    def get_vault_features(self) -> set[ERC4626Feature] | None:
        """Get list of vault feature flags if the pair is a vault.

        Needed for ERC-4626 compatibility.
        """
        feats = self.other_data.get("vault_features")
        if feats:
            return set(feats)
        return None

    def get_vault_protocol(self) -> str | None:
        """Get the vault protocol name.

        - Lowercased, slug
        """
        return self.other_data.get("vault_protocol")

    def get_vault_name(self) -> str | None:
        """Get the vault name,

        - Human readable
        """
        return self.other_data.get("vault_name")

    def get_performance_fee(self) -> Percent | None:
        """Get the vault performance fee.

        - None if unknown
        """
        return self.other_data.get("vault_performance_fee")

    def get_management_fee(self) -> Percent | None:
        """Get the vault management fee.

        - None if unknown
        """
        return self.other_data.get("vault_management_fee")

    def has_complete_info(self) -> bool:
        """Check if the pair has good information.

        Because of the open-ended  nature a lot of irrelevant broken
        data can be found on blockchains.

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

    def has_reverse_token_order(self) -> bool:
        """Has Uniswap smart contract a flipped token order.

        - Is token0 base token or token0 is the quote token

        See :py:func:`eth_defi.uniswap_v3.price.get_onchain_price`
        """
        assert self.reverse_token_order is not None, \
            f"reverse_token_order not set for: {self}.\n" \
            f"This is needed for Uniswap routing.\n" \
            f"If you construct TradingPairIdentifier by hand remember to set TradingPairIdentifier.reverse_token_order"
        return self.reverse_token_order

    def get_max_leverage_at_open(
        self,
        side: Literal["long", "short"] = "short",
    ) -> LeverageMultiplier:
        """Return the max leverage we can set for this position at open.

        E.g. for AAVE WETH short this is 0.8 because we can supply
        1000 USDC to get 800 USDC loan. This gives us the health factor
        of 1.13 on open.

        Max Leverage in pair: l=1/(1-cfBuy); cfBuy = collateralFacor of Buy Asset

        - `See 1delta documentation <https://docs.1delta.io/lenders/metrics>`__.

        :param side:
            Order side: long or short
        """
        assert self.kind in (TradingPairKind.lending_protocol_short, TradingPairKind.lending_protocol_long)

        max_long_leverage = 1 / (1 - self.get_collateral_factor())
        max_short_leverage = max_long_leverage - 1

        return max_short_leverage if side == "short" else max_long_leverage

    def is_leverage(self) -> bool:
        return self.kind.is_leverage()

    def is_short(self) -> bool:
        """Leveraged short."""
        return self.kind.is_shorting()

    def is_long(self) -> bool:
        """Leveraged long, not spot."""
        return self.kind.is_longing()

    def is_spot(self) -> bool:
        return self.kind.is_spot()

    def is_cash(self) -> bool:
        return self.kind.is_cash()

    def is_vault(self) -> bool:
        return self.kind.is_vault()

    def is_freqtrade(self) -> bool:
        return self.kind.is_freqtrade()

    def is_credit_supply(self) -> bool:
        return self.kind.is_credit_supply()

    def is_volume_generating(self) -> bool:
        """Should this pair to be included in total volume calculations"""
        return not self.is_credit_supply()

    def get_liquidation_threshold(self) -> Percent:
        """What's the liqudation threshold for this leveraged pair"""
        assert self.kind.is_leverage()
        # Liquidation threshold comes from the collateral token
        threshold = self.quote.liquidation_threshold
        assert type(threshold) == float, f"Liquidation theshold must be 0..1, got {threshold}, {type(threshold)}, pair: {self}"
        assert 0 < threshold < 1, f"Liquidation theshold must be 0..1, got {threshold}, pair: {self}"
        return threshold

    def get_collateral_factor(self) -> Percent:
        """Same as liquidation threshold.

        Alias for :py:meth:`get_liquidation_threshold`
        """
        return self.get_liquidation_threshold()

    def get_pricing_pair(self) -> Optional["TradingPairIdentifier"]:
        """Get the the trading pair that determines the price for the asset.

        - For spot pairs this is the trading pair itself

        - For pairs that may lack price feed data like USDC/USD
          pairs used in credit supply, return None

        :return:
            The trading pair we can use to query underlying asset price.

            Return ``None`` if the trading pair does not have price information.
        """
        if self.is_spot() | self.is_vault():
            return self
        elif self.is_credit_supply():
            # Credit supply does not have a real trading pair,
            # but any position price is simply the amount of collateral
            return self
        elif self.is_leverage():
            assert self.underlying_spot_pair is not None, f"For a leveraged pair, we lack the price feed for the underlying spot: {self}"
            return self.underlying_spot_pair
        raise AssertionError(f"Cannot figure out how to get the underlying pricing pair for: {self}")

    def get_tags(self) -> set[str]:
        """Get tags asssociated with the base asset of this trading pair.

        - See :py:meth:`AssetIdentifier.get_tags`
        """
        underlying = self.underlying_spot_pair or self
        return underlying.base.get_tags()

    def can_have_tax(self) -> bool:
        """Can this token pair have a tax.

        Token tax is usually only applicable to Uniswap v2 pairs.
        """
        return self.exchange_name != "uniswap-v3"

    def get_buy_tax(self) -> Percent | None:
        """Get buy tax associated with this token if any.

        .. note ::

            This property should be per trading pair, but all other DEX systems use per token tax values.
        """

        if not self.can_have_tax():
            # Shortcut for untaxable pairs
            return 0

        return self.base.get_buy_tax()

    def get_sell_tax(self) -> Percent | None:
        """Get sell tax associated with this token if any.

        .. note ::

            This property should be per trading pair, but all other DEX systems use per token tax values.

        """

        if not self.can_have_tax():
            # Shortcut for untaxable pairs
            return 0

        return self.base.get_sell_tax()

    def set_tradeable(self, value: bool, reason: str | None = None):
        """Set the tradeable flag for this trading pair.

        - Used in backtesting to disable trading on some pairs
          that are not available in the backtesting data.

        - By default pairs are tradeable

        - Indicator data will be generared for non-tradeabl pairs by default (TODO: how to di
        """
        assert isinstance(value, bool), f"Tradeable value must be boolean, got {value}"
        self.other_data["tradeable"] = value
        self.other_data["tradeable_reason"] = reason

    def is_tradeable(self) -> bool:
        """Check the tradeable flag.

        Use :py:meth:`tradeexecutor.strategy.state.State.is_good_pair()` to access.
        """
        return self.other_data.get("tradeable", True)


@dataclass_json
@dataclass(slots=True)
class AssetWithTrackedValue:
    """Track one asset with a value.

    - Track asset quantity \

    - The asset can be vToken/aToken for interest based tracking,
      in this case :py:attr:`presentation` is set

    - Any tracked asset must get USD oracle price from somewhere
    """

    #: Asset we are tracking
    #:
    #: The is aToken or vToken asset.
    #:
    #: Use ``asset.underlying`` to get the token.
    #:
    asset: AssetIdentifier

    #: How many token units we have.
    #:
    #: In the case of loans this represents the underlying asset (WETH),
    #: not any gained interest (vWETH).
    #:
    quantity: Decimal

    #: What was the last known USD price of a single unit of quantity
    last_usd_price: USDollarPrice

    #: When the last pricing happened
    last_pricing_at: datetime.datetime

    #: Strategy cycle time stamp when the tracking was started
    #:
    created_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)

    #: Strategy cycle time stamp when the tracking was started
    #:
    created_strategy_cycle_at: datetime.datetime | None = None

    #: Supply or borrow interest rate for this asset when the loan is created
    #: 
    #: This is recorded using lending candles data when position is created
    #:
    interest_rate_at_open: Percent | None = None

    #: Latest supply or borrow interest rate for this asset
    #: 
    #: This is recorded using lending candles data every time `sync_interests` is called
    #:
    last_interest_rate: Percent | None = None

    def __repr__(self):
        return f"<AssetWithTrackedValue {self.asset.token_symbol} {self.quantity} at price {self.last_usd_price} USD>"

    def __post_init__(self):
        assert isinstance(self.quantity, Decimal), f"Got {self.quantity.__class__}"
        # __post_init__ is also called on de-serialisation
        # Quantity si ze
        assert self.quantity >= 0, f"Any tracked asset must have positive quantity, received {self.asset} = {self.quantity}"
        assert self.last_usd_price is not None, "Price is None - asset price must set during initialisation"
        assert self.last_usd_price > 0

    def get_usd_value(self) -> USDollarAmount:
        """Rrturn the approximate value of this tracked asset.

        Priced in the `last_usd_price`
        """
        return float(self.quantity) * self.last_usd_price

    def revalue(self, price: USDollarPrice, when: datetime.datetime):
        """Update the latest known price of the asset."""
        assert isinstance(when, datetime.datetime)
        assert type(price) == float
        assert 0 < price < 1_000_000, f"Price sanity check {price}"
        self.last_usd_price = price
        self.last_pricing_at = when

    def change_quantity_and_value(
        self,
        delta: Decimal,
        price: USDollarPrice,
        when: datetime.datetime,
        allow_negative: bool = False,
        available_accrued_interest: Decimal = Decimal(0),
        epsilon: Decimal = SUM_EPSILON,
        close_position=False,
    ):
        """The tracked asset amount is changing due to position increase/reduce.

        :param allow_negative:
            Backtesting helper parameter.

            Bail out with an exception if delta is too high and balance would go negative.

        :param available_accrued_interest:
            How much interest we have gained.

            To be used with ``allow_negative``.
        """
        assert delta is not None, "Asset delta must be given"
        self.revalue(price, when)

        if not allow_negative:
            total_available = self.quantity + available_accrued_interest
            s = sum_decimal((total_available, delta,), epsilon=epsilon)

            # See close_position=True
            #
            # Round loan value to zero
            #
            if close_position and (abs(s) < abs(delta * epsilon)) and s != 0:
                delta = -self.quantity
            else:
                assert s >= 0, f"Tracked asset cannot go negative: {self}. delta: {delta}, total available: {total_available}, sum: {s}, quantity: {self.quantity}, interest: {available_accrued_interest}"

        self.quantity += delta

        # Fix decimal math issues
        self.quantity = ensure_exact_zero(self.quantity, epsilon=epsilon)

        # TODO: this is a temp hack for testing to make sure the borrowed quantity can be minimum 0
        if self.quantity < 0:
            self.quantity = Decimal(0)

    def reset(self, quantity: Decimal):
        """Reset the loan quantity.

        See also :py:func:`tradeexecutor.strategy.lending_protocol_leverage.reset_credit_supply_loan`.
        """
        assert isinstance(quantity, Decimal)
        self.quantity = quantity
        self.interest_rate_at_open = None
        self.last_interest_rate = None




