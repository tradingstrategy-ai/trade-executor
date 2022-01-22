import enum
from dataclasses import dataclass, field
import datetime
from decimal import Decimal
from typing import List, Optional

from dataclasses_json import dataclass_json
from tradingstrategy.types import PrimaryKey

from tradingstrategy.chain import ChainId


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


@dataclass_json
@dataclass
class Position:
    asset: AssetIdentifier
    quantity: Decimal
    type: PositionType
    opened_at: datetime.datetime
    closed_at: datetime.datetime


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


@dataclass_json
@dataclass
class Position:
    pass


@dataclass_json
@dataclass
class State:
    """The current state of the trading strategy execution."""

    next_trade_id = 1

    #: Currently held assets
    open_positions: List[Position] = field(default_factory=list)

    #: Currently open trades
    open_trades: List[TradeExecution] = field(default_factory=list)

    #: Trades completed in the past
    completed_trades: List[TradeExecution] = field(default_factory=list)

    #: Strategy can store its internal thinkign over different signals
    strategy_thinking: dict = field(default_factory=dict)

    def allocate_trade_id(self):
        try:
            return self.next_trade_id
        finally:
            self.next_trade_id += 1

