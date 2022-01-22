import enum
from dataclasses import dataclass, field
import datetime
from decimal import Decimal
from typing import List

from dataclasses_json import dataclass_json


class PositionType(enum.Enum):
    token_hold = "token_hold"
    lending_pool_hold = "lending_pool_hold"


@dataclass_json
@dataclass
class AssetIdentifier:
    address: str



@dataclass_json
@dataclass
class Position:
    address: str
    quantity: Decimal
    type: PositionType
    opened_at: datetime.datetime
    closed_at: datetime.datetime


@dataclass_json
@dataclass
class TradeExecution:

    trade_id: int
    clock_at: datetime.datetime
    asset: AssetIdentifier

    requested_quantity: Decimal
    price_impact: Decimal
    slippage: Decimal

    started_at: datetime.datetime
    broadcasted_at: datetime.datetime
    ended_at: datetime.datetime

    txid: List[str]
    retried_trade_int: int


@dataclass_json
@dataclass
class Position:
    pass


@dataclass_json
@dataclass
class State:
    """The current state of the trading strategy execution."""

    #: Currently held assets
    open_positions: List[Position] = field(default_factory=list)

    #: Currently open trades
    open_trades: List[TradeExecution] = field(default_factory=list)

    #: Trades completed in the past
    completed_trades: List[TradeExecution] = field(default_factory=list)

    #: Strategy can store its internal thinkign over different signals
    strategy_thinking: dict = field(default_factory=dict)

