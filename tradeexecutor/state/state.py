import enum
from dataclasses import dataclass, field
import datetime
from decimal import Decimal
from typing import List, Optional, Dict, Callable

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


class TradeOutcome(enum.Enum):
    success = "success"
    reverted = "reverted"

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

    def get_current_value(self) -> USDollarAmount:
        return self.quantity * self.current_usd_price


@dataclass_json
@dataclass
class TradeExecution:

    trade_id: int
    position_id: int
    trade_type: TradeType
    pair: TradingPairIdentifier
    opened_at: datetime.datetime

    requested_quantity: Decimal
    requested_price: USDollarAmount

    reserve_currency_used: AssetIdentifier

    broadcasted_at: Optional[datetime.datetime] = None
    closed_at: Optional[datetime.datetime] = None
    failed_at: Optional[datetime.datetime] = None
    fill_price: Optional[USDollarAmount] = None

    txid: Optional[str] = None
    replay_of: Optional[int] = None

    def is_sell(self):
        return self.requested_quantity < 0

    def is_buy(self):
        return self.requested_quantity > 0

    def __post_init__(self):
        assert self.trade_id > 0
        assert self.requested_quantity != 0


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

    next_trade_id = 1

    def __post_init__(self):
        assert self.position_id > 0
        assert self.quantity != 0
        assert self.last_usd_price > 0

    @property
    def quantity(self) -> Decimal:
        sum([t.quantity for t in self.trades])

    def get_identifier(self) -> str:
        """One trading pair may have multiple open positions at the same time."""
        return f"{self.pair.get_identifier()}-{self.position_id}"

    def get_value(self) -> USDollarAmount:
        return self.last_usd_price * self.quantity

    def revalue(self, ts: datetime.datetime, price: USDollarAmount):
        assert isinstance(ts, datetime.datetime)
        self.last_usd_price = price
        self.last_pricing_at = ts

    def open_trade(self, ts: datetime.datetime, quantity: Decimal, assumed_price: USDollarAmount, trade_type: TradeType, reserve_currency: AssetIdentifier) -> TradeExecution:
        trade = TradeExecution(
            trade_id=self.next_trade_id,
            trade_type=trade_type,
            pair=self.pair,
            opened_at=ts,
            requested_quantity=quantity,
            requested_price=assumed_price,
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

    def is_closing_trade(self, trade: TradeExecution) -> bool:
        pass


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
    open_positions: Dict[int, TradingPosition] = field(default_factory=list)

    #: Currently held reserve assets
    reserves: List[ReservePosition] = field(default_factory=list)

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
        self.open_positions[pair.get_identifier()] = p
        self.next_position_id += 1
        return p

    def create_trade(self, ts: datetime.datetime, pair: TradingPairIdentifier, quantity: Decimal, assumed_price: USDollarAmount, trade_type: TradeType, reserve_currency: AssetIdentifier) -> TradeExecution:
        position = self.open_positions.get(pair.pool_address)
        if position is None:
            position = self.open_new_position(ts, pair, assumed_price)

        position.open_trade(ts, assumed_price, quantity, trade_type, reserve_currency)
        return position

    def get_current_cash(self) -> USDollarAmount:
        """Get how much reserve stablecoins we have."""
        return sum([r.get_current_value() for r in self.reserves])

    def get_open_position_equity(self) -> USDollarAmount:
        """Get the value of current trading positions."""
        return sum([p.get_current_value() for p in self.open_positions])

    def get_total_equity(self) -> USDollarAmount:
        """Get the value of the portfolio based on the latest pricing."""
        return self.get_equity() + self.get_current_cash()

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
        for p in self.open_positions:
            if p.has_trade(trade):
                return p
        return None

    def start_execution(self, ts: datetime.datetime, trade: TradeExecution, txid: str):
        """The trade execution has started.

        Assume reverse currency is tied to the trade, so adjust the balances accordingly.
        """
        position = self.find_position_for_trade(trade)



@dataclass_json
@dataclass
class State:
    """The current state of the trading strategy execution."""

    portfolio: Portfolio

    #: Trades completed in the past
    past_positions: List[TradingPosition] = field(default_factory=list)

    #: Strategy can store its internal thinking over different signals
    strategy_thinking: dict = field(default_factory=dict)

    def create_trade(self, ts: datetime.datetime, pair: TradingPairIdentifier, quantity: Decimal, trade_type: TradeType) -> TradeExecution:
        """Creates a request for a new trade.

        If there is no open position, marks a position open.

        When the trade is created no balances are suff
        """
        trade = self.portfolio.create_trade(ts, pair, quantity, trade_type)
        return trade

    def start_execution(self, ts: datetime.datetime, trade: TradeExecution, txid: str):
        """Update our balances and mark the trade execution as started.

        Called before a transaction is broadcasted.
        """


    def finish_execution(self, trade: TradeExecution, outcome: TradeOutcome):
        """

        Close any positions if the trade was closing position.

        :param trade:
        :param outcome:
        :return:
        """
        pass





