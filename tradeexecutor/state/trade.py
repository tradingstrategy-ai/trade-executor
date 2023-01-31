"""Trade execution state info."""

import datetime
import enum
import pprint
from dataclasses import dataclass, field, asdict
from decimal import Decimal
from typing import Optional, Tuple, List

from dataclasses_json import dataclass_json

from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier
from tradeexecutor.state.types import USDollarAmount, USDollarPrice, BPS


class TradeType(enum.Enum):
    """What kind of trade execution this was."""

    #: A normal trade with strategy decision
    rebalance = "rebalance"

    #: The trade was made because stop loss trigger reached
    stop_loss = "stop_loss"

    #: The trade was made because take profit trigger reached
    take_profit = "take_profit"


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


@dataclass_json
@dataclass()
class TradeExecution:
    """Trade execution tracker.

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
    #: Always accurately known for sells.
    planned_quantity: Decimal

    #: How many reserve tokens (USD) we use in this trade
    #: Always known accurately for buys.
    #: Expressed in `reserve_currency`.
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

    #: Timestamp of the block where the txid was first mined
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

    #: LP fee % recorded before the execution starts.
    #:
    #: Not available in the case this is ignored
    #: in backtesting or not supported by routers/trading pairs.
    #:
    #: Used to calculate :py:attr:`lp_fees_estimated`.
    #:
    #: Sourced from Uniswap v2 router or Uniswap v3 pool information.
    #:
    fee_tier: Optional[BPS] = None

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

    #: Trade was manually repaird
    #:
    #: E.g. failed broadcast issue was fixed.
    #: Marked when the repair command is called.
    repaired_at: Optional[datetime.datetime] = None

    def __repr__(self):
        if self.is_buy():
            return f"<Buy #{self.trade_id} {self.planned_quantity} {self.pair.base.token_symbol} at {self.planned_price}, {self.get_status().name}>"
        else:
            return f"<Sell #{self.trade_id} {abs(self.planned_quantity)} {self.pair.base.token_symbol} at {self.planned_price}, {self.get_status().name}>"

    def get_full_debug_dump_str(self):
        return pprint.pformat(asdict(self))

    def __hash__(self):
        # TODO: Hash better?
        return hash(str(self))

    def __eq__(self, other):
        assert isinstance(other, TradeExecution)
        return self.trade_id == other.trade_id

    def __post_init__(self):
        assert self.trade_id > 0
        assert self.planned_quantity != 0
        assert self.planned_price > 0
        assert self.planned_reserve >= 0
        assert type(self.planned_price) == float, f"Price was given as {self.planned_price.__class__}: {self.planned_price}"
        assert self.opened_at.tzinfo is None, f"We got a datetime {self.opened_at} with tzinfo {self.opened_at.tzinfo}"

        if self.lp_fees_estimated is not None:
            assert type(self.lp_fees_estimated) == float

        if self.fee_tier is not None:
            assert type(self.fee_tier) == float

    @property
    def strategy_cycle_at(self):
        """Alias for oepned_at"""
        return self.opened_at

    def get_human_description(self) -> str:
        """User friendly description for this trade"""
        if self.is_buy():
            return f"Buy {self.planned_quantity} {self.pair.base.token_symbol} <id:{self.pair.base.internal_id}> at {self.planned_price}"
        else:
            return f"Sell {abs(self.planned_quantity)} {self.pair.base.token_symbol} <id:{self.pair.base.internal_id}> at {self.planned_price}"

    def is_sell(self):
        assert self.planned_quantity != 0, "Buy/sell concept does not exist for zero quantity"
        return self.planned_quantity < 0

    def is_buy(self):
        assert self.planned_quantity != 0, "Buy/sell concept does not exist for zero quantity"
        return self.planned_quantity >= 0

    def is_success(self):
        """This trade was succcessfully completed."""
        return self.executed_at is not None

    def is_failed(self):
        """This trade was succcessfully completed."""
        return self.failed_at is not None

    def is_pending(self):
        """This trade was succcessfully completed."""
        return self.get_status() in (TradeStatus.started, TradeStatus.broadcasted)

    def is_planned(self):
        """This trade is still in planning, unallocated."""
        return self.get_status() in (TradeStatus.planned,)

    def is_started(self):
        """This trade has a txid allocated."""
        return self.get_status() in (TradeStatus.started,)

    def is_rebalance(self):
        """This trade is part of the normal strategy rebalance."""
        return self.trade_type == TradeType.rebalance

    def is_stop_loss(self):
        """This trade is made to close stop loss on a position."""
        return self.trade_type == TradeType.stop_loss

    def is_take_profit(self):
        """This trade is made to close take profit on a position."""
        return self.trade_type == TradeType.take_profit

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

    def get_status(self) -> TradeStatus:
        if self.failed_at:
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
        if self.executed_quantity is not None:
            return self.executed_quantity
        else:
            return self.planned_quantity

    def get_reserve_quantity(self) -> Decimal:
        """Get the planned or executed quantity of the quote token.

        Negative for buy, positive for sell.
        """
        if self.executed_reserve is not None:
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

        if self.executed_at:
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
        TODO: Make this functio to behave
        :return:
        """
        status = self.get_status()
        if status == TradeStatus.success:
            return self.lp_fees_paid
        elif status == TradeStatus.failed:
            return 0
        else:
            raise AssertionError(f"Unsupported trade state to query fees: {self.get_status()}")

    def get_execution_sort_position(self) -> int:
        """When this trade should be executed.

        Lower, negative, trades should be executed first.

        We need to execute sells first because we need to have cash in hand to execute buys.
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

    def mark_success(self, executed_at: datetime.datetime, executed_price: USDollarAmount, executed_quantity: Decimal, executed_reserve: Decimal, lp_fees: USDollarAmount, native_token_price: USDollarAmount):
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
