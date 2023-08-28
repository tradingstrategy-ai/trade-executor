"""Asset and trading pair identifiers.

How executor internally knows how to connect trading pairs in data and in execution environment (on-chain).
"""
import datetime
import enum
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

from dataclasses_json import dataclass_json
from eth_typing import HexAddress

from tradeexecutor.utils.accuracy import sum_decimal, ensure_exact_zero
from tradingstrategy.chain import ChainId
from web3 import Web3

from tradeexecutor.state.types import JSONHexAddress, USDollarAmount, LeverageMultiplier, USDollarPrice, Percent
from tradingstrategy.lending import LendingProtocolType
from tradingstrategy.stablecoin import is_stablecoin_like
from tradingstrategy.types import PrimaryKey


@dataclass_json
@dataclass
class AssetType:
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
    internal_id: Optional[PrimaryKey] = None

    #: Info page URL for this asset
    info_url: Optional[str] = None

    #: The underlying asset for aTokens, vTokens and such
    underlying: Optional["AssetIdentifier"] = None

    #: What kind of asset is this
    #:
    #: Legacy data will default to ``None``.
    #:
    type: AssetType | None = None

    #: Aave liquidation threhold for this asset
    #:
    #: Set on aTokens that are used as collateral.
    #:
    liquidation_threshold: float | None = None

    def __str__(self):
        if self.underlying:
            return f"<{self.token_symbol} ({self.underlying.token_symbol}) at {self.address}>"
        else:
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


@dataclass_json
@dataclass(slots=True)
class TradingPairIdentifier:
    """Uniquely identify one trading pair across all tradeable blockchain assets.

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
    internal_id: Optional[int] = None

    #: What is the internal exchange id of this trading pair.
    internal_exchange_id: Optional[int] = None

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
    #:
    reverse_token_order: Optional[bool] = None

    #: Whether the pair is based on a Uniswap v3 pool (or uniswap v3 equivalent).
    uniswap_v3_like: Optional[bool] = None

    #: What kind of position this is
    #:
    kind: TradingPairKind = TradingPairKind.spot_market_hold

    def __post_init__(self):
        assert self.base.chain_id == self.quote.chain_id, "Cross-chain trading pairs are not possible"

        # float/int zero fix
        # TODO: Can be carefully removed later
        if self.fee == 0:
            self.fee = 0.0

        assert (type(self.fee) in {float, type(None)}) or (self.fee == 0)

    def __repr__(self):
        fee = self.fee or 0
        type_name = self.kind.name if self.kind else "spot"
        return f"<Pair {self.base.token_symbol}-{self.quote.token_symbol} {type_name} at {self.pool_address} ({fee * 100:.4f}% fee) on exchange {self.exchange_address}>"

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

        Example: `WETH-USDC`.
        """
        return f"{self.base.token_symbol}-{self.quote.token_symbol}"

    def get_human_description(self) -> str:
        """Same as get_ticker()."""
        return self.get_ticker()

    def get_lending_protocol(self) -> LendingProtocolType | None:
        """Is this pair on a particular lending protocol."""
        if self.kind in (TradingPairKind.lending_protocol_short, TradingPairKind.lending_protocol_long):
            return LendingProtocolType.aave_v3
        return None

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
        assert self.reverse_token_order is not None, f"reverse_token_order not set for: {self}"
        return self.reverse_token_order

    def get_max_leverage_at_open(self) -> LeverageMultiplier:
        """Return the max leverage we can set for this position at open.

        E.g. for AAVE WETH short this is 0.8 because we can supply
        1000 USDC to get 800 USDC loan. This gives us the health factor
        of 1.13 on open.

        Max Leverage in pair: l=1/(1-cfBuy); cfBuy = collateralFacor of Buy Asset

        - `See 1delta documentation <https://docs.1delta.io/lenders/metrics>`__.
        """
        assert self.kind in (TradingPairKind.lending_protocol_short, TradingPairKind.lending_protocol_long)
        return 1 / (1 - self.get_collateral_factor())

    def is_leverage(self) -> bool:
        return self.kind.is_leverage()

    def is_spot(self) -> bool:
        return self.kind.is_spot()

    def is_credit_supply(self) -> bool:
        return self.kind.is_credit_supply()

    def get_liquidation_threshold(self) -> Percent:
        """What's the liqudation threshold for this leveraged pair"""
        assert self.kind.is_leverage()
        # Liquidation threshold comes from the collateral token
        return self.quote.liquidation_threshold

    def get_collateral_factor(self) -> Percent:
        """Same as liquidation threshold.

        Alias for :py:meth:`get_liquidation_threshold`
        """
        return self.get_liquidation_threshold()


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

    def __repr__(self):
        return f"<AssetWithTrackedValue {self.asset.token_symbol} {self.quantity} at price {self.last_usd_price} USD>"

    def __post_init__(self):
        assert self.quantity > 0, f"Any tracked asset must have positive quantity, received {self.asset} = {self.quantity}"
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
        allow_negative=False,
    ):
        """The tracked asset amount is changing due to position increase/reduce."""
        assert delta is not None, "Asset delta must be given"
        self.revalue(price, when)

        if not allow_negative:
            assert sum_decimal((self.quantity, delta,)) >= 0, f"Tracked asset cannot go negative: {self}. Quantity: {self.quantity}, delta: {delta}"
        self.quantity += delta

        # Fix decimal math issues
        self.quantity = ensure_exact_zero(self.quantity)


