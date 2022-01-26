import enum
from dataclasses import dataclass, field
import datetime
from decimal import Decimal
from typing import List, Optional, Dict, Callable, Iterable, Tuple

from dataclasses_json import dataclass_json

from tradingstrategy.pair import PandasPairUniverse, DEXPair
from tradingstrategy.types import PrimaryKey, USDollarAmount

from tradingstrategy.chain import ChainId


class PositionType(enum.Enum):
    token_hold = "token_hold"
    lending_pool_hold = "lending_pool_hold"


class TradeType(enum.Enum):
    rebalance = "rebalance"
    stop_loss = "stop_loss"
    take_profit = "take_profit"


class TradeStatus(enum.Enum):

    #: Trade has been put to the planning pipeline
    planned = "planned"

    #: Trade has txid allocated
    started = "started"

    #: Trade has been pushed to the network
    broadcasted = "broadcasted"

    #: Trade was executed ok
    success = "success"

    #: Trade was reversed e.g. due to too much slippage
    failed = "failed"


@dataclass_json
@dataclass
class AssetIdentifier:
    """Identify a blockchain asset for trade execution.

    As internal token_ids and pair_ids may be unstable, trading pairs and tokens are explicitly
    referred by their smart contract addresses when a strategy decision moves to the execution.
    We duplicate data here to make sure we have a persistent record that helps to diagnose the sisues.
    """
    chain_id: ChainId
    address: str
    token_symbol: str
    decimals: Optional[int] = None

    def get_identifier(self) -> str:
        """Assets are identified by their smart contract address."""
        return self.address


@dataclass_json
@dataclass
class TradingPairIdentifier:
    base: AssetIdentifier
    quote: AssetIdentifier

    #: Internal pair_id might not be stable across differenet dataset versions
    pair_id: PrimaryKey

    #: Smart contract address of the pool contract
    pool_address: str

    def get_identifier(self) -> str:
        """We use the smart contract pool address to uniquely identify trading positions."""
        return self.pool_address

    def get_trading_pair(self, pair_universe: PandasPairUniverse) -> DEXPair:
        """Reverse resolves the smart contract address to trading pair data in the current trading pair universe."""
        return pair_universe.get_pair_by_smart_contract(self.pool_address)


@dataclass_json
@dataclass
class ReservePosition:
    asset: AssetIdentifier
    quantity: Decimal
    current_usd_price: float
    last_pricing_at: datetime.datetime

    def get_identifier(self) -> str:
        return self.asset.get_identifier()

    def get_current_value(self) -> USDollarAmount:
        return float(self.quantity) * self.current_usd_price


@dataclass_json
@dataclass
class TradeExecution:

    trade_id: int
    position_id: int
    trade_type: TradeType
    pair: TradingPairIdentifier
    opened_at: datetime.datetime

    planned_quantity: Decimal
    planned_price: USDollarAmount
    planned_reserve_currency_amount: Decimal

    #: Which reserve currency we are going to take
    reserve_currency: AssetIdentifier

    #: When capital is allocated for this trade
    capital_allocated_at: Optional[datetime.datetime] = None
    reserve_currency_allocated: Optional[Decimal] = None

    #: When this trade entered mempool
    broadcasted_at: Optional[datetime.datetime] = None

    #: Timestamp of the block where the txid was first mined
    executed_at: Optional[datetime.datetime] = None
    failed_at: Optional[datetime.datetime] = None

    executed_price: Optional[USDollarAmount] = None
    executed_quantity: Optional[USDollarAmount] = None

    #: LP fees estimated in the USD
    lp_fees_paid: Optional[USDollarAmount] = None

    #: Gas consumed by the tx
    gas_units_consumed: Optional[int] = None

    #: Gas price for the tx in gwei
    gas_price: Optional[int] = None

    #: USD price per blockchain native currency unit, at the time of execution
    native_token_price: Optional[USDollarAmount] = None

    # Blockchain bookkeeping
    txid: Optional[str] = None
    nonce: Optional[int] = None

    # Trade retries
    retry_of: Optional[int] = None

    def __post_init__(self):
        assert self.trade_id > 0
        assert self.planned_quantity != 0
        assert self.planned_price > 0
        assert self.planned_reserve_currency_amount > 0

    def is_sell(self):
        return self.planned_quantity < 0

    def is_buy(self):
        return self.planned_quantity > 0

    def is_success(self):
        """This trade was succcessfully completed."""
        return self.executed_at != None

    def is_pending(self):
        """This trade was succcessfully completed."""
        return self.get_status() in (TradeStatus.started, TradeStatus.broadcasted)

    def is_planned(self):
        """This trade is still in planning, unallocated."""
        return self.get_status() in (TradeStatus.planned,)

    def get_status(self) -> TradeStatus:
        if self.failed_at:
            return TradeStatus.failed
        elif self.executed_at:
            return TradeStatus.success
        elif self.broadcasted_at:
            return TradeStatus.broadcasted
        elif self.capital_allocated_at:
            return TradeStatus.started
        else:
            return TradeStatus.planned

    def allocate_reserves(self, ts: datetime.datetime, amount: Decimal):
        self.capital_allocated_at = ts
        self.reserve_currency_allocated = amount

    def get_executed_value(self) -> USDollarAmount:
        return float(self.executed_quantity) * self.executed_price

    def get_planned_value(self) -> USDollarAmount:
        return self.planned_price * self.planned_quantity

    def get_allocated_value(self) -> USDollarAmount:
        return self.reserve_currency_allocated

    def get_value(self) -> USDollarAmount:
        """Get estimated or realised value of this trade."""

        if self.executed_at:
            return self.get_executed_value()
        elif self.failed_at:
            return self.get_planned_value()
        elif self.capital_allocated_at:
            # Trade is being planned, but capital has already moved
            # from the portfolio reservs to this trade object in internal accounting
            return float(self.reserve_currency_allocated)
        else:
            # Trade does not have value until capital is allocated to it
            return 0.0

    def get_quantity(self) -> Decimal:
        """Get estimated or realised value of this trade."""
        if self.closed_at:
            return self.executed_quantity
        else:
            return self.planned_quantity

    def get_gas_fees_paid(self) -> USDollarAmount:
        native_token_consumed = self.gas_units_consumed * self.gas_price / (10**18)
        return self.native_token_price * native_token_consumed

    def get_fees_paid(self) -> USDollarAmount:
        status = self.get_status()
        if status == TradeStatus.success:
            return self.lp_fees_paid + self.get_gas_fees_paid()
        elif status == TradeStatus.failed:
            return self.get_gas_fees_paid()
        else:
            raise AssertionError(f"Unsupported trade state to query fees: {self.get_status()}")

    def mark_success(self, executed_at: datetime.datetime, executed_price: USDollarAmount, executed_quantity: Decimal, lp_fees: USDollarAmount, gas_price: int, gas_units_consumed: int, native_token_price: USDollarAmount):
        assert self.get_status() == TradeStatus.broadcasted
        self.executed_at = executed_at
        self.executed_quantity = executed_quantity
        self.executed_price = executed_price
        self.lp_fees_paid = lp_fees
        self.gas_price = gas_price
        self.gas_units_consumed = gas_units_consumed
        self.native_token_price = native_token_price
        self.reserve_currency_allocated = Decimal(0)

    def mark_failed(self, failed_at: datetime.datetime):
        assert self.get_status() == TradeStatus.broadcasted
        self.failed_at = failed_at


@dataclass_json
@dataclass
class TradingPosition:
    position_id: int
    pair: TradingPairIdentifier
    # type: PositionType
    opened_at: datetime.datetime

    last_usd_price: USDollarAmount
    last_pricing_at: datetime.datetime

    reserve_currency: AssetIdentifier

    trades: Dict[int, TradeExecution] = field(default_factory=dict)

    closed_at: Optional[datetime.datetime] = None
    last_trade_at: Optional[datetime.datetime] = None

    next_trade_id = 1

    def __post_init__(self):
        assert self.position_id > 0
        assert self.last_usd_price > 0
        assert self.reserve_currency is not None

    def get_quantity(self) -> Decimal:
        """Get the tied up token quantity in all successfully executed trades.

        Does not account for trades that are currently being executd.
        """
        sum([t.quantity for t in self.trades.values() if t.is_success()])

    def get_live_quantity(self) -> Decimal:
        """Get all tied up token quantity.

        If there are trades being executed, us their estimated amounts.
        """
        sum([t.quantity for t in self.trades.values() if not t.is_failed()])

    def has_unexecuted_trades(self) -> bool:
        return any([t for t in self.trades.values() if t.is_pending()])

    def has_planned_trades(self) -> bool:
        return any([t for t in self.trades.values() if t.is_planned()])

    def get_identifier(self) -> str:
        """One trading pair may have multiple open positions at the same time."""
        return f"{self.pair.get_identifier()}-{self.position_id}"

    def get_successful_trades(self) -> List[TradeExecution]:
        """Get all trades that have been successfully executed and contribute to this position"""
        return [t for t in self.trades.values() if t.is_success()]

    def get_value(self) -> USDollarAmount:
        return sum([t.get_value() for t in self.trades.values()])

    def revalue(self, ts: datetime.datetime, price: USDollarAmount):
        assert isinstance(ts, datetime.datetime)
        self.last_usd_price = price
        self.last_pricing_at = ts

    def open_trade(self, ts: datetime.datetime, quantity: Decimal, assumed_price: USDollarAmount, trade_type: TradeType, reserve_currency: AssetIdentifier) -> TradeExecution:
        trade = TradeExecution(
            trade_id=self.next_trade_id,
            position_id=self.position_id,
            trade_type=trade_type,
            pair=self.pair,
            opened_at=ts,
            planned_quantity=quantity,
            planned_price=assumed_price,
            planned_reserve_currency_amount=Decimal(quantity * assumed_price),
            reserve_currency=self.reserve_currency,
        )
        self.trades[trade.trade_id] = trade
        self.next_trade_id += 1
        return trade

    def has_trade(self, trade: TradeExecution):
        """Check if a trade belongs to this position."""
        if trade.position_id != self.position_id:
            return False
        return trade.trade_id in self.trades


@dataclass
class RevalueEvent:
    """Describe how asset was revalued"""
    position_id: str
    revalued_at: datetime.datetime
    quantity: Decimal
    old_price: USDollarAmount
    new_price: USDollarAmount


@dataclass_json
@dataclass
class Portfolio:
    """Represents a trading portfolio on DEX markets.

    Multiple trading pair issue: We do not identify the actual assets we have purchased,
    but the trading pairs that we used to purchase them. Thus, each position marks
    "openness" in a certain trading pair. For example open position of WBNB-BUSD
    means we have bought X BNB tokens through WBNB-BUSD trading pair.
    But because this is DEX market, we could enter and exit through WBNB-USDT,
    WBNB-ETH, etc. and our position is not really tied to a trading pair.
    However, because all TradFi trading view the world through trading pairs,
    we keep the tradition here.
    """

    next_position_id = 1

    #: Currently open trading positions
    open_positions: Dict[int, TradingPosition] = field(default_factory=dict)

    #: Currently held reserve assets
    reserves: Dict[str, ReservePosition] = field(default_factory=dict)

    #: Trades completed in the past
    past_positions: List[TradingPosition] = field(default_factory=list)

    def is_empty(self):
        return len(self.open_positions) == 0 and len(self.reserves) == 0 and len(self.past_positions) == 0

    def get_open_position_for_pair(self, pair: TradingPairIdentifier) -> TradingPosition:
        return self.open_positions.get(pair.pool_address)

    def open_new_position(self, ts: datetime.datetime, pair: TradingPairIdentifier, assumed_price: USDollarAmount, reserve_currency: AssetIdentifier) -> TradingPosition:
        p = TradingPosition(
            position_id=self.next_position_id,
            opened_at=ts,
            pair=pair,
            last_pricing_at=ts,
            last_usd_price=assumed_price,
            reserve_currency=reserve_currency,
        )
        self.open_positions[p.position_id] = p
        self.next_position_id += 1
        return p

    def create_trade(self, ts: datetime.datetime, pair: TradingPairIdentifier, quantity: Decimal, assumed_price: USDollarAmount, trade_type: TradeType, reserve_currency: AssetIdentifier) -> Tuple[TradingPosition, TradeExecution]:
        position = self.open_positions.get(pair.pool_address)
        if position is None:
            position = self.open_new_position(ts, pair, assumed_price, reserve_currency)

        trade = position.open_trade(ts, assumed_price, quantity, trade_type, reserve_currency)
        return position, trade

    def get_current_cash(self) -> USDollarAmount:
        """Get how much reserve stablecoins we have."""
        return sum([r.get_current_value() for r in self.reserves.values()])

    def get_open_position_equity(self) -> USDollarAmount:
        """Get the value of current trading positions."""
        return sum([p.get_value() for p in self.open_positions.values()])

    def get_total_equity(self) -> USDollarAmount:
        """Get the value of the portfolio based on the latest pricing."""
        return self.get_open_position_equity() + self.get_current_cash()

    def revalue_portfolio(self, timestamp: datetime.datetime, revaluation_method: Callable) -> List[RevalueEvent]:
        """Revalue the assets based on the latest trading universe prices.

        Mutates the `TradingPosition` objects in-place.
        """
        assert isinstance(timestamp, datetime.datetime)
        events = []
        # TODO: Revalue reserves after we have stablecoin price feeds
        for position in self.open_positions.values():
            old_price = position.last_usd_price
            new_price = revaluation_method(timestamp, position)
            events.append(RevalueEvent(timestamp, position.quantiy, old_price, new_price))
            position.revalue(timestamp, new_price)
        return events

    def find_position_for_trade(self, trade) -> Optional[TradingPosition]:
        """Find a position tha trade belongs for."""
        return self.open_positions[trade.position_id]

    def get_reserve_position(self, asset: AssetIdentifier) -> ReservePosition:
        return self.reserves[asset.get_identifier()]

    def adjust_reserves(self, asset: AssetIdentifier, amount: Decimal):
        """Remove currency from reserved"""
        reserve = self.get_reserve_position(asset)
        reserve.quantity += amount

    def has_unexecuted_trades(self) -> bool:
        """Do we have any trades that have capital allocated, but not executed yet."""
        return any([p.has_unexecuted_trades() for p in self.open_positions])

    def update_reserves(self, new_reserves: List[ReservePosition]):
        """Update current reserves.

        Overrides current amounts of reserves.

        E.g. in the case users have deposited more capital.
        """

        assert not self.has_unexecuted_trades(), "Updating reserves while there are trades in progress will mess up internal account"

        for r in new_reserves:
            self.reserves[r.get_identifier()] = r


@dataclass_json
@dataclass
class State:
    """The current state of the trading strategy execution."""

    portfolio: Portfolio = field(default_factory=Portfolio)

    #: Strategy can store its internal thinking over different signals
    strategy_thinking: dict = field(default_factory=dict)

    def is_empty(self):
        return self.portfolio.is_empty()

    def create_trade(self, ts: datetime.datetime, pair: TradingPairIdentifier, quantity: Decimal, assumed_price: USDollarAmount, trade_type: TradeType, reserve_currency: AssetIdentifier) -> Tuple[TradingPosition, TradeExecution]:
        """Creates a request for a new trade.

        If there is no open position, marks a position open.

        When the trade is created no balances are suff
        """
        trade = self.portfolio.create_trade(ts, pair, quantity, assumed_price, trade_type, reserve_currency)
        return trade

    def start_execution(self, ts: datetime.datetime, trade: TradeExecution, txid: str, nonce: int):
        """Update our balances and mark the trade execution as started.

        Called before a transaction is broadcasted.
        """

        assert trade.get_status() == TradeStatus.planned

        position = self.portfolio.find_position_for_trade(trade)
        assert position, f"Trade does not belong to an open position {trade}"

        if trade.is_buy():
            value = trade.get_planned_value()
            trade.allocate_reserves(ts, value)
            self.portfolio.adjust_reserves(trade.reserve_currency, -value)

        trade.txid = txid
        trade.nonce = nonce

    def mark_broadcasted(self, broadcasted_at: datetime.datetime, trade: TradeExecution):
        """"""
        assert trade.get_status() == TradeStatus.started
        trade.broadcasted_at = broadcasted_at

    def mark_trade_success(self, executed_at: datetime.datetime, trade: TradeExecution, executed_price: USDollarAmount, executed_amount: Decimal, lp_fees: USDollarAmount, gas_price: USDollarAmount, gas_used: Decimal, native_token_price: USDollarAmount):
        """"""
        trade.mark_success(executed_at, executed_price, executed_amount, lp_fees, gas_price, gas_used, native_token_price)

    def mark_trade_failed(self, failed_at: datetime.datetime, trade: TradeExecution):
        """Unroll the allocated capital."""
        trade.mark_failed(failed_at)
        self.portfolio.adjust_reserves(trade.reserve_currency, trade.reserve_currency_allocated)

    def update_reserves(self, new_reserves: List[ReservePosition]):
        self.portfolio.update_reserves(new_reserves)



