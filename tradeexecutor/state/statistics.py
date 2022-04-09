"""Various statistics calculated across portfolios and positions.

Statistics are calculated/refreshed on the server-side and exported as a part of the state.
This way the clients (JavaScript) do not need to reconstruct this information.

Any statistics are optional: they are not needed to make any state transitions, they are
purely there for profit and loss calculations.
"""
import datetime
from dataclasses import field
from typing import Dict

from dataclasses_json import dataclass_json

from tradeexecutor.utils import dataclass
from tradingstrategy.types import USDollarAmount


@dataclass_json
@dataclass
class PositionStatistics:

    #: Real-time clock when these stats were calculated
    refreshed_at: datetime.datetime

    #: Position duration in seconds
    duration: float

    #: Profitability %
    profitability: float

    #: How much profit we made
    profit_usd: USDollarAmount

    #: All value tied to this position
    equity: USDollarAmount


@dataclass_json
@dataclass
class PortfolioStatistics:
    #: Real-time clock when these stats were calculated
    refreshed_at: datetime.datetime
    total_equity: USDollarAmount
    position_equity: USDollarAmount
    total_cash: USDollarAmount
    all_time_profit: float
    trade_count: int
    first_trade_at: datetime.datetime
    last_trade_at: datetime.datetime


@dataclass_json
@dataclass
class Statistics:

    #: Per portfolio statistics.
    #: Note that now we have only one portfolio per state.
    portfolio: PortfolioStatistics

    #: Per position statistics
    position: Dict[int, PortfolioStatistics] = field(default_factory=dict)