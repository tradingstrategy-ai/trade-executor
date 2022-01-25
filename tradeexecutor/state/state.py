import enum
from dataclasses import dataclass, field
import datetime
from decimal import Decimal
from typing import List, Optional, Dict, Callable
import pandas as pd

from dataclasses_json import dataclass_json

from tradingstrategy.pair import PandasPairUniverse, DEXPair
from tradingstrategy.types import PrimaryKey, USDollarAmount

from tradingstrategy.chain import ChainId
from tradingstrategy.universe import Universe


class PositionType(enum.Enum):
    token_hold = "token_hold"
    lending_pool_hold = "lending_pool_hold"


class TradeType(enum.Enum):
    rebalance = "rebalance"
    stop_loss = "stop_loss"
    take_profit = "take_profit"


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
class TradingPosition:
    position_id: int
    pair: TradingPairIdentifier
    quantity: Decimal
    type: PositionType
    opened_at: datetime.datetime
    closed_at: datetime.datetime
    current_usd_price: float

    def get_identifier(self) -> str:
        """One trading pair may have multiple open positions at the same time."""
        return f"{self.pair.get_identifier()}-{self.position_id}"

    def get_value(self) -> USDollarAmount:
        return self.current_usd_price


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
    trade_type: TradeType
    clock_at: datetime.datetime
    trading_pair: TradingPairIdentifier

    requested_quantity: Decimal

    started_at: datetime.datetime
    broadcasted_at: Optional[datetime.datetime] = None
    ended_at: Optional[datetime.datetime] = None
    failed_At: Optional[datetime.datetime] = None
    txid: List[str] = field(default=list)
    retried_trade_int: Optional[int] = None

    def is_sell(self):
        return self.requested_quantity < 0

    def is_buy(self):
        return self.requested_quantity > 0

    def __post_init__(self):
        assert self.trade_id > 0
        assert self.requested_quantity != 0


@dataclass
class RevalueEvent:
    """Describe how asset was revalued"""

    pair_address: str
    revalued_at: pd.Timestamp
    old_value: USDollarAmount
    new_value: USDollarAmount


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

    #: Currently open trading positions
    open_positions: Dict[str, TradingPosition] = field(default_factory=list)

    #: Currently held reserve assets
    reserves: List[ReservePosition] = field(default_factory=list)

    def get_current_cash(self) -> USDollarAmount:
        """Get how much reserve stablecoins we have."""
        return sum([r.get_current_value() for r in self.reserves])

    def get_equity(self) -> USDollarAmount:
        """Get the value of current trading positions."""
        return sum([p.get_current_value() for p in self.open_positions])

    def get_total_equity(self) -> USDollarAmount:
        """Get the value of the portfolio based on the latest pricing."""
        return self.get_equity() + self.get_current_cash()

    def revalue_portfolio(self, timestamp: pd.Timestamp, revaluation_method: Callable) -> List[RevalueEvent]:
        """Revalue the assets based on the latest trading universe prices.

        Mutates the `TradingPosition` objects in-place.
        """

        events = []
        # TODO: Revalue reserves after we have stablecoin price feeds
        for position in self.open_positions.values():
            old_value = position.get_value()
            new_value = revaluation_method(timestamp, position)

            events.append(RevalueEvent(timestamp, positionold_value, new_value))



@dataclass_json
@dataclass
class State:
    """The current state of the trading strategy execution."""

    portfolio: Portfolio

    next_trade_id = 1

    next_position_id = 1

    #: Currently open trades
    open_trades: List[TradeExecution] = field(default_factory=list)

    #: Trades completed in the past
    completed_trades: List[TradeExecution] = field(default_factory=list)

    #: Strategy can store its internal thinking over different signals
    strategy_thinking: dict = field(default_factory=dict)

    def allocate_trade_id(self):
        try:
            return self.next_trade_id
        finally:
            self.next_trade_id += 1


