"""Various statistics calculated across portfolios and positions.

Statistics are calculated/refreshed on the server-side and exported as a part of the state.
This way the clients (JavaScript) do not need to reconstruct this information.

Any statistics are optional: they are not needed to make any state transitions, they are
purely there for profit and loss calculations.
"""
import datetime
from collections import defaultdict
from dataclasses import field, dataclass
from typing import Dict, List, Optional

from dataclasses_json import dataclass_json

from tradingstrategy.types import USDollarAmount


@dataclass_json
@dataclass
class PositionStatistics:

    #: Real-time clock when these stats were calculated
    calculated_at: datetime.datetime

    #: When this position was revalued last time.
    #: Should not be far off from `calculated_at`
    #: because we should revalue positions always before calculating their stats.
    last_valuation_at: datetime.datetime

    #: Profitability %
    profitability: float

    #: How much profit we made so far
    profit_usd: USDollarAmount

    #: All value tied to this position
    equity: USDollarAmount


@dataclass_json
@dataclass
class FinalPositionStatistics:
    """When position is closed, its final statistics are calculated.

    These statistics contain fields that is not present in open positions.
    """

    #: Real-time clock when these stats were calculated
    calculated_at: datetime.datetime

    #: When this position was opened
    #: Only calculated after the position is closed.
    first_trade_at: datetime.datetime

    #: When this position was closed
    #: Only calculated after the position is closed.
    last_trade_at: datetime.datetime

    #: How many trades we have made
    trade_count: int



@dataclass_json
@dataclass
class PortfolioStatistics:
    #: Real-time clock when these stats were calculated
    calculated_at: datetime.datetime
    total_equity: USDollarAmount

    open_position_count: int
    open_position_equity: USDollarAmount
    frozen_position_count: int
    frozen_position_equity: USDollarAmount
    closed_position_count: int

    free_cash: USDollarAmount

    unrealised_profit_usd: USDollarAmount
    closed_profit_usd: USDollarAmount


@dataclass_json
@dataclass
class Statistics:
    """Statistics for a trade execution state.

    We calculate various statistics on the server-side and make them part of the state,
    so that JS clients can easily display this information.

    Statistics are collected over time and more often than trading ticks.
    We store historical statistics for each position as the part of the state.
    """

    #: Per portfolio statistics.
    #: Contains list of statistics for the portfolio over time.
    #: The first timestamp is the first entry in the list.
    #: Note that now we have only one portfolio per state.
    portfolio: List[PortfolioStatistics] = field(default_factory=list)

    #: Per position statistics.
    #: We look them up by position id.
    #: Each position contains list of statistics for the position over time.
    #: The first timestamp is the first entry in the list.
    positions: Dict[int, List[PositionStatistics]] = field(default_factory=lambda: defaultdict(list))

    #: Per position statistics for closed positions.
    closed_positions: Dict[int, FinalPositionStatistics] = field(default_factory=dict)

    def get_latest_portfolio_stats(self) -> PortfolioStatistics:
        return self.portfolio[-1]

    def get_latest_position_stats(self, position_id: int) -> PositionStatistics:
        return self.positions[position_id][-1]