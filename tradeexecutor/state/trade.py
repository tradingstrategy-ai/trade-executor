"""Trade execution state info."""

import datetime
import enum
import pprint
import logging
from dataclasses import dataclass, field, asdict
from decimal import Decimal
from typing import Optional, Tuple, List
from types import NoneType

from dataclasses_json import dataclass_json

from eth_defi.tx import AssetDelta

from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier
from tradeexecutor.state.types import USDollarAmount, USDollarPrice, BPS
from tradeexecutor.strategy.trade_pricing import TradePricing


logger = logging.getLogger()


#: Absolute minimum units we are willing to trade regardless of an asset
#:
#: Used to catch floating point rounding errors
QUANTITY_EPSILON = Decimal(10**-18)


class TradeType(enum.Enum):
    """What kind of trade execution this was."""

    #: A normal trade with strategy decision
    rebalance = "rebalance"

    #: The trade was made because stop loss trigger reached
    stop_loss = "stop_loss"

    #: The trade was made because take profit trigger reached
    take_profit = "take_profit"

    #: This is an accounting counter trade to cancel a broken trade.
    #:
    #: - The original trade is marked as repaied
    #:
    #: - This trade contains any reverse accounting variables needed to fix the position total
    repair = "repair"


class TradeStatus(enum.Enum):

    #: Trade has been put to the planning pipeline.
    #: The trade instance has been created and stored in the state,
    #: but no internal accounting changes have been made.
    planned = "planned"

    #: Trade has txid allocated.
    #: Any capital have been debited from the reserved and credited on the trade.
    started = "started"

    #: Trade has been pushed to the network
    broadcasted = "broadcasted"

    #: Trade was executed ok
    #: Any capital on sell transaction have been credited back to the reserves.
    success = "success"

    #: Trade was reversed e.g. due to too much slippage.
    #: Trade can be retries.
    failed = "failed"

    #: This trade was originally failed, but later repaired.
    #:
    #: A counter entry was made in the position and this trade was marked as repaired.
    repaired = "repaired"

    #: A virtual trade to reverse any balances of a repaired trade.
    #:
    repair_entry = "repair_entry"


@dataclass_json
@dataclass()
class TradeExecution:
    """Trade execution tracker. 
    
    - One TradeExecution instance can only represent one swap

    Each trade has a reserve currency that we use to trade the token (usually USDC).

    Each trade can be

    - Buy: swap quote token -> base token

    - Sell: swap base token -> quote token

    When doing a buy `planned_reserve` (fiat) is the input. This yields to `executed_quantity` of tokens
    that may be different from `planned_quantity`.

    When doing a sell `planned_quantity` (token) is the input. This yields to `executed_reserve`
    of fiat that might be different from `planned_reserve.

    Trade execution has four states

    - Planning: The execution object is prepared

    - Capital allocation and transaction creation: We move reserve from out portfolio to the trade in internal accounting

    - Transaction broadcast: trade cannot be cancelled in this point

    - Resolving the trade: We check the Ethereum transaction receipt to see how well we succeeded in the trade

    There trade state is resolved based on the market variables (usually timestamps).
    """

    #: Trade id is unique among all trades in the same portfolio
    trade_id: int

    #: Position id is unique among all trades in the same portfolio
    position_id: int

    #: Spot, margin, lending, etc.
    trade_type: TradeType

    #: Which trading pair this trade was for
    pair: TradingPairIdentifier

    #: What was the strategy cycle timestamp for it was created.
    #:
    #: Naive UTC timestamp.
    #:
    #: If the trade was executed by a take profit/stop loss trigger
    #: then this is the trigger timestamp (not wall clock time)
    #:
    #: See also
    #:
    #: - :py:attr:`started_at`
    #:
    #: - :py:meth:`get_execution_lag`
    #:
    #: - :py:meth:`get_decision_lag`
    #:
    opened_at: datetime.datetime

    #: Positive for buy, negative for sell.
    #:
    #: Always accurately known for sells.
    planned_quantity: Decimal

    #: How many reserve tokens (USD) we use in this trade
    #:
    #: - Always a position number (only the sign of :py:attr:`planned_quantity` changes between buy/sell)
    #:
    #: - For buys, Always known accurately for buys.
    #:
    #: - For sells, an estimation based on :py:attr:`planned_price`
    #:
    #: Expressed in :py:attr:`reserve_currency`.
    #:
    planned_reserve: Decimal

    #: What we thought the execution price for this trade would have been
    #: at the moment of strategy decision.
    #:
    #: This price includes any fees we pay for LPs,
    #: and should become executed_price if the execution is perfect.
    #:
    #: For the market price see :py:attr:`planned_mid_price`.
    planned_price: USDollarPrice

    #: Which reserve currency we are going to take.
    #: Note that pair.quote might be different from reserve currency.
    #: This is because we can do three-way trades like BUSD -> BNB -> Cake
    #: when our routing model supports this.
    reserve_currency: AssetIdentifier

    #: What is the reserve currency exchange rate used for this trade
    #:
    #: - Access using :py:attr:`get_reserve_currency_exchange_rate`.
    #:
    #: - USDC/USD exchange rate.
    #:
    #: - If not set (legacy) assume 1.0 reset assets / USD
    #:
    reserve_currency_exchange_rate: Optional[USDollarPrice] = None

    #: What we thought was the mid-price when we made the decision to tale this trade
    #:
    #: This is the market price of the asset at the time of the trade decision.
    planned_mid_price: Optional[USDollarPrice] = None

    #: How much slippage we could initially tolerate,
    #: 0.01 is 1% slippage.
    planned_max_slippage: Optional[BPS] = None

    #: When this trade was decided
    #:
    #: Wall clock time.
    #:
    #: For backtested trades, this is always set to
    #: opened_at.
    started_at: Optional[datetime.datetime] = None

    #: How much reserves was moved on this trade before execution
    reserve_currency_allocated: Optional[Decimal] = None

    #: When this trade entered mempool
    broadcasted_at: Optional[datetime.datetime] = None

    #: Timestamp of the block where the txid was first mined.
    #:
    #: For failed trades, this is not set until repaired,
    #: but instead we set :py:attr:`failed_at`.
    executed_at: Optional[datetime.datetime] = None

    #: The trade did not go through.
    #: The timestamp when we figured this out.
    failed_at: Optional[datetime.datetime] = None

    #: What was the actual price we received
    executed_price: Optional[USDollarPrice] = None

    #: How much underlying token we traded, the actual realised amount.
    #: Positive for buy, negative for sell
    executed_quantity: Optional[Decimal] = None

    #: How much reserves we spend for this traded, the actual realised amount.
    executed_reserve: Optional[Decimal] = None

    #: Slippage tolerance for this trade.
    #:
    #: Examples
    #:
    #: - `0`: no slippage toleranc eallowed at all
    #:
    #: - `0.01`: 1% slippage tolerance
    #:
    #: - `1`: MEV bots can steal all your money
    #:
    #: We estimate `executed_quantity = planned_quantity * slippage_tolerance`.
    #: If any trade outcome exceeds the slippage tolerance the trade fails.
    #:
    #: If you are usinga vault-based trading, slippage tolerance must be always set
    #: to calculate the asset delta.
    #:
    #: See also :py:meth:`calculate_asset_deltas`.
    #:
    slippage_tolerance: Optional[float] = None

    #: LP fee % recorded before the execution starts.
    #:
    #: Recorded as multiplier
    #:
    #: Not available in the case this is ignored
    #: in backtesting or not supported by routers/trading pairs.
    #:
    #: Used to calculate :py:attr:`lp_fees_estimated`.
    #:
    #: Sourced from Uniswap v2 router or Uniswap v3 pool information.
    #:
    fee_tier: Optional[float] = None

    #: LP fees paid, currency convereted to the USD.
    #:
    #: The value is read back from the realised trade.
    #: LP fee is usually % of the trade. For Uniswap style exchanges
    #: fees are always taken from `amount in` token
    #: and directly passed to the LPs as the part of the swap,
    #: these is no separate fee information.
    lp_fees_paid: Optional[USDollarAmount] = None

    #: LP fees estimated in the USD
    #:
    #: This is set before the execution and is mostly useful
    #: for backtesting.
    lp_fees_estimated: Optional[USDollarAmount] = None

    #: What is the conversation rate between quote token and US dollar used in LP fee conversion.
    #:
    #: We set this exchange rate before the trade is started.
    #: Both `lp_fees_estimated` and `lp_fees_paid` need to use the same exchange rate,
    #: even though it would not be timestamp accurte.
    lp_fee_exchange_rate: Optional[USDollarPrice] = None

    #: USD price per blockchain native currency unit, at the time of execution
    #:
    #: Used for converting tx fees and gas units to dollars
    native_token_price: Optional[USDollarPrice] = None

    # Trade retries
    retry_of: Optional[int] = None

    #: Associated blockchain transaction details.
    #: Each trade contains 1 ... n blockchain transactions.
    #: Typically this is approve() + swap() for Uniswap v2
    #: or just swap() if we have the prior approval and approve does not need to be
    #: done for the hot wallet anymore.
    blockchain_transactions: List[BlockchainTransaction] = field(default_factory=list)

    #: Human readable notes about this trade
    #:
    #: Used to mark test trades from command line.
    #: Special case; not worth to display unless the field is filled in.
    notes: Optional[str] = None

    #: Trade was manually repaired
    #:
    #: E.g. failed broadcast issue was fixed.
    #: Marked when the repair command is called.
    repaired_at: Optional[datetime.datetime] = None

    #: Which is the trade that this trade is repairing.
    #:
    #: This trade makes a opposing trade to the trade referred here,
    #: making accounting match again and unfreezing the position.
    #:
    #: For the repair trade
    #:
    #: - Strategy cycle is set to the original broken trade
    #:
    repaired_trade_id: Optional[datetime.datetime] = None

    #: Related TradePricing instance
    #:
    #: TradePricing instance can refer to more than one swap
    price_structure: Optional[TradePricing] = None
    

    def __repr__(self):
        if self.is_buy():
            return f"<Buy #{self.trade_id} {self.planned_quantity} {self.pair.base.token_symbol} at {self.planned_price}, {self.get_status().name}>"
        else:
            return f"<Sell #{self.trade_id} {abs(self.planned_quantity)} {self.pair.base.token_symbol} at {self.planned_price}, {self.get_status().name}>"

    def pretty_print(self) -> str:
        """Get diagnostics output for the trade.

        Use Python `pprint` module.
        """
        d = asdict(self)
        return pprint.pformat(d)

    def get_full_debug_dump_str(self):
        return pprint.pformat(asdict(self))

    def __hash__(self):
        # TODO: Hash better?
        return hash(self.trade_id)

    def __eq__(self, other):
        """Note that we do not support comparison across different portfolios ATM."""
        assert isinstance(other, TradeExecution)
        return self.trade_id == other.trade_id

    def __post_init__(self):

        assert self.trade_id > 0

        if self.trade_type != TradeType.repair:
            assert self.planned_quantity != 0

        assert abs(self.planned_quantity) > QUANTITY_EPSILON, f"We got a planned quantity that does look like a good number: {self.planned_quantity}, trade is: {self}"

        assert self.planned_price > 0
        assert self.planned_reserve >= 0
        assert type(self.planned_price) in {float, int}, f"Price was given as {self.planned_price.__class__}: {self.planned_price}"
        assert self.opened_at.tzinfo is None, f"We got a datetime {self.opened_at} with tzinfo {self.opened_at.tzinfo}"
        
    @property
    def strategy_cycle_at(self):
        """Alias for oepned_at"""
        return self.opened_at
    
    @property
    def fee_tier(self) -> (float | None):
        """LP fee % recorded before the execution starts.
        
        :return:
            float (fee multiplier) or None if no fee was provided.

        """
        return self._fee_tier

    @fee_tier.setter
    def fee_tier(self, value):
        """Setter for fee_tier.
        Ensures fee_tier is a float"""
        
        if type(value) is property:
            # hack
            # See comment on this post: https://florimond.dev/en/posts/2018/10/reconciling-dataclasses-and-properties-in-python/
            value = None
        
        assert (type(value) in {float, NoneType}) or (value == 0), "If fee tier is specified, it must be provided as a float to trade execution"

        if value is None and (self.pair.fee or self.pair.fee == 0):
            assert type(self.pair.fee) == float, f"trading pair fee not in float format, got {self.pair.fee} ({type(self.pair.fee)}"
            # Low verbosity as otherwise this message is filling test logs
            logger.debug("No fee_tier provided but fee was found on associated trading pair, using trading pair fee")
            self._fee_tier = self.pair.fee
        else:
            self._fee_tier = value

    @property
    def lp_fees_estimated(self) -> float:
        """LP fees estimated in the USD.
        This is set before the execution and is mostly useful
        for backtesting.
        """
        return self._lp_fees_estimated

    @lp_fees_estimated.setter
    def lp_fees_estimated(self, value):
        """Setter for lp_fees_estimated"""
        
        if type(value) is property:
            # hack
            # See comment on this post: https://florimond.dev/en/posts/2018/10/reconciling-dataclasses-and-properties-in-python/
            value = None
        
        # TODO standardize
        # assert type(value) == float, f"Received lp_fees_estimated: {value} - {type(value)}"
        self._lp_fees_estimated = value
    
    def get_human_description(self) -> str:
        """User friendly description for this trade"""
        if self.is_buy():
            return f"Buy {self.planned_quantity} {self.pair.base.token_symbol} <id:{self.pair.base.internal_id}> at {self.planned_price}"
        else:
            return f"Sell {abs(self.planned_quantity)} {self.pair.base.token_symbol} <id:{self.pair.base.internal_id}> at {self.planned_price}"

    def get_reserve_currency_exchange_rate(self) -> USDollarPrice:
        """What was the reserve stablecoin exchange trade for this trade.

        :return:
            1.0 if not set
        """
        return self.reserve_currency_exchange_rate or 1.0

    def is_sell(self) -> bool:
        assert self.planned_quantity != 0, "Buy/sell concept does not exist for zero quantity"
        return self.planned_quantity < 0

    def is_buy(self) -> bool:
        assert self.planned_quantity != 0, "Buy/sell concept does not exist for zero quantity"
        return self.planned_quantity >= 0

    def is_success(self) -> bool:
        """This trade was succcessfully completed."""
        return self.executed_at is not None

    def is_failed(self) -> bool:
        """This trade was succcessfully completed."""
        return (self.failed_at is not None) and (self.repaired_at is None)

    def is_pending(self) -> bool:
        """This trade was succcessfully completed."""
        return self.get_status() in (TradeStatus.started, TradeStatus.broadcasted)

    def is_planned(self) -> bool:
        """This trade is still in planning, unallocated."""
        return self.get_status() in (TradeStatus.planned,)

    def is_started(self) -> bool:
        """This trade has a txid allocated."""
        return self.get_status() in (TradeStatus.started,)

    def is_rebalance(self) -> bool:
        """This trade is part of the normal strategy rebalance."""
        return self.trade_type == TradeType.rebalance

    def is_stop_loss(self) -> bool:
        """This trade is made to close stop loss on a position."""
        return self.trade_type == TradeType.stop_loss

    def is_take_profit(self) -> bool:
        """This trade is made to close take profit on a position."""
        return self.trade_type == TradeType.take_profit

    def is_triggered(self) -> bool:
        """Was this trade based on a trigger signal."""
        return self.trade_type == TradeType.take_profit or self.trade_type == TradeType.stop_loss

    def is_accounted_for_equity(self) -> bool:
        """Does this trade contribute towards the trading position equity.

        Failed trades are reverted. Only their fees account.
        """
        return self.get_status() in (TradeStatus.started, TradeStatus.broadcasted, TradeStatus.success)

    def is_unfinished(self) -> bool:
        """We could not confirm this trade back from the blockchain after broadcasting."""
        return self.get_status() in (TradeStatus.broadcasted,)

    def is_repaired(self) -> bool:
        """The automatic execution failed and this was later repaired.

        A manual repair command was issued and it manaeged to correctly repair this trade
        and underlying transactions.
        """
        return self.repaired_at is not None

    def is_repair_trade(self) -> bool:
        """This trade repairs another trade in the same position."""
        return self.repaired_trade_id is not None

    def is_executed(self) -> bool:
        """Did this trade ever execute."""
        return self.executed_at is not None

    def is_repair_needed(self) -> bool:
        """This trade needs repair, but is not repaired yet."""
        return self.is_failed() and not self.is_repaired()

    def is_repair_trade(self) -> bool:
        """This trade is fixes a frozen position and counters another trade.
        """
        return self.repaired_trade_id is not None

    def is_repair_trade(self) -> bool:
        """This trade is fixes a frozen position and counters another trade.
        """
        return self.repaired_trade_id is not None

    def is_redemption(self) -> bool:
        """This trade marks a redemption balance update on a position"""

    def get_status(self) -> TradeStatus:
        """Resolve the trade status.

        Based on the different state variables set on this item,
        figure out what is the best status for this trade.
        """
        if self.repaired_trade_id:
            # Bookkeeping trades are only internal and thus always success
            return TradeStatus.success
        elif self.repaired_at:
            return TradeStatus.repaired
        elif self.failed_at:
            return TradeStatus.failed
        elif self.executed_at:
            return TradeStatus.success
        elif self.broadcasted_at:
            return TradeStatus.broadcasted
        elif self.started_at:
            return TradeStatus.started
        else:
            return TradeStatus.planned

    def get_executed_value(self) -> USDollarAmount:
        return abs(float(self.executed_quantity) * self.executed_price)

    def get_planned_value(self) -> USDollarAmount:
        return abs(self.planned_price * float(abs(self.planned_quantity)))

    def get_planned_reserve(self) -> Decimal:
        return self.planned_reserve

    def get_raw_planned_reserve(self) -> int:
        """Return the amount of USD token for the buy as raw token units."""
        return self.reserve_currency.convert_to_raw_amount(self.planned_reserve)

    def get_raw_planned_quantity(self) -> int:
        """Return the amount of USD token for the buy as raw token units."""
        return self.pair.base.convert_to_raw_amount(self.planned_quantity)

    def get_allocated_value(self) -> USDollarAmount:
        return self.reserve_currency_allocated

    def get_position_quantity(self) -> Decimal:
        """Get the planned or executed quantity of the base token.

        Positive for buy, negative for sell.
        """

        if self.repaired_trade_id or self.repaired_at:
            # Repaired trades are shortcuted to zero
            return Decimal(0)
        elif self.executed_quantity is not None:
            return self.executed_quantity
        else:
            return self.planned_quantity

    def get_reserve_quantity(self) -> Decimal:  
        """Get the planned or executed quantity of the quote token.

        Negative for buy, positive for sell.
        """
        if self.repaired_trade_id or self.repaired_at:
            # Repaired trades are shortctted to zero
            return Decimal(0)
        elif self.executed_reserve is not None:
            return self.executed_quantity
        else:
            return self.planned_reserve

    def get_equity_for_position(self) -> Decimal:
        """Get the planned or executed quantity of the base token.

        Positive for buy, negative for sell.
        """
        if self.executed_quantity is not None:
            return self.executed_quantity
        return Decimal(0)

    def get_equity_for_reserve(self) -> Decimal:
        """Get the planned or executed quantity of the quote token.

        Negative for buy, positive for sell.
        """

        if self.is_buy():
            if self.get_status() in (TradeStatus.started, TradeStatus.broadcasted):
                return self.reserve_currency_allocated

        return Decimal(0)

    def get_value(self) -> USDollarAmount:
        """Get estimated or realised value of this trade.

        Value is always a positive number.
        """

        if self.repaired_at:
            # Repaired trades have their value set to zero
            return 0.0
        elif self.executed_at:
            return self.get_executed_value()
        elif self.failed_at:
            return self.get_planned_value()
        elif self.started_at:
            # Trade is being planned, but capital has already moved
            # from the portfolio reservs to this trade object in internal accounting
            return self.get_planned_value()
        else:
            # Trade does not have value until capital is allocated to it
            return 0.0

    def get_credit_debit(self) -> Tuple[Decimal, Decimal]:
        """Returns the token quantity and reserve currency quantity for this trade.

        If buy this is (+trading position quantity/-reserve currency quantity).

        If sell this is (-trading position quantity/-reserve currency quantity).
        """
        return self.get_position_quantity(), self.get_reserve_quantity()

    def get_fees_paid(self) -> USDollarAmount:
        """
        Get total swap fees paid for trade. Returns 0 instead of `None`
        
        :return: total amount of lp fees (swap fees) paid in US dollars
        """
        
        status = self.get_status()
        if status == TradeStatus.success:
            return self.lp_fees_paid or 0
        elif status == TradeStatus.failed:
            return 0
        else:
            raise AssertionError(f"Unsupported trade state to query fees: {self.get_status()}")

    def get_execution_sort_position(self) -> int:
        """When this trade should be executed.

        Lower, negative, trades should be executed first.

        We need to execute sells first because we need to have cash in hand to execute buys.

        :return:
            Sortable int
        """
        if self.is_sell():
            return -self.trade_id
        else:
            return self.trade_id

    def get_decision_lag(self) -> datetime.timedelta:
        """How long it took between strategy decision cycle starting and the strategy to make a decision."""
        return self.started_at - self.opened_at

    def get_execution_lag(self) -> datetime.timedelta:
        """How long it took between strategy decision cycle starting and the trade executed."""
        return self.started_at - self.opened_at

    def mark_broadcasted(self, broadcasted_at: datetime.datetime):
        assert self.get_status() == TradeStatus.started, f"Trade in bad state: {self.get_status()}"
        self.broadcasted_at = broadcasted_at

    def mark_success(self,
                     executed_at: datetime.datetime,
                     executed_price: USDollarAmount,
                     executed_quantity: Decimal,
                     executed_reserve: Decimal,
                     lp_fees: USDollarAmount,
                     native_token_price: USDollarAmount,
                     force=False,
                     ):
        """Mark trade success.

        - Called by execution engine when we get a confirmation from the blockchain our blockchain txs where good

        - Called by repair to force trades to good state
        """
        if not force:
            assert self.get_status() == TradeStatus.broadcasted, f"Cannot mark trade success if it is not broadcasted. Current status: {self.get_status()}"
        assert isinstance(executed_quantity, Decimal)
        assert type(executed_price) == float, f"Received executed price: {executed_price} {type(executed_price)}"
        assert executed_at.tzinfo is None
        self.executed_at = executed_at
        self.executed_quantity = executed_quantity
        self.executed_reserve = executed_reserve
        self.executed_price = executed_price
        self.lp_fees_paid = lp_fees
        self.native_token_price = native_token_price
        self.reserve_currency_allocated = Decimal(0)

    def mark_failed(self, failed_at: datetime.datetime):
        assert self.get_status() == TradeStatus.broadcasted
        assert failed_at.tzinfo is None
        self.failed_at = failed_at

    def set_blockchain_transactions(self, txs: List[BlockchainTransaction]):
        """Set the physical transactions needed to perform this trade."""
        assert not self.blockchain_transactions
        self.blockchain_transactions = txs

    def get_planned_max_gas_price(self) -> int:
        """Get the maximum gas fee set to all transactions in this trade."""
        return max([t.get_planned_gas_price() for t in self.blockchain_transactions])

    def calculate_asset_deltas(self) -> List[AssetDelta]:
        """Get the expected amount fo token balance change in a wallet for this trade.

        Needed for vault based trading to work, as they run
        slippage tolerance checks on expectd inputs and outputs.

        Each trade has minimum of two asset deltas

        - The token you spent to buu/sell (input)

        - The token you receive (output)

        The output will always have :py:attr:`slippage_tolerance` applied.
        Input is passed as is.

        :return:

            List of asset deltas [input, output]

        """

        # TODO: slippage tolerance currently ignores multihop trades

        assert self.slippage_tolerance is not None, "Slippage tolerance must be set before we can calculate_asset_deltas()"
        assert 0 <= self.slippage_tolerance <= 1.0, f"Slippage tolerance must be 0...1, got {self.slippage_tolerance}"

        if self.is_buy():
            input_asset = self.reserve_currency
            input_amount = self.planned_reserve

            output_asset = self.pair.base
            output_amount = self.planned_quantity

            assert input_amount > 0, "Buy missing input amount"
            assert output_amount > 0, "Buy missing output amount"
        else:
            input_asset = self.pair.base
            input_amount = -self.planned_quantity

            output_asset = self.reserve_currency
            output_amount = self.planned_reserve

            assert input_amount > 0, "Sell missing input amount"
            assert output_amount > 0, "Sell missing output amount"

        assert input_amount > 0
        assert output_amount > 0

        return [
            AssetDelta(input_asset.address, -input_asset.convert_to_raw_amount(input_amount)),
            AssetDelta(output_asset.address, output_asset.convert_to_raw_amount(output_amount * Decimal(1 - self.slippage_tolerance))),
        ]
