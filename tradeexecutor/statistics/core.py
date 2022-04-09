"""Core portfolio statistics calculations."""

import datetime
from dataclasses import dataclass, field
from typing import Dict

from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.statistics import Statistics, PortfolioStatistics, PositionStatistics, FinalPositionStatistics


@dataclass
class NewStatistics:
    """New statistics calcualted for the portfolio on each stats cycle."""
    portfolio: PortfolioStatistics
    positions: Dict[int, PositionStatistics] = field(default_factory=dict)


def calculate_position_statistics(clock: datetime.datetime, position: TradingPosition) -> PositionStatistics:
    stats = PositionStatistics(
        calculated_at=clock,
        last_valuation_at=position.last_pricing_at,
        profitability=position.get_total_profit_percent(),
        profit_usd=position.get_total_profit_usd(),
        equity=float(position.get_equity_for_position()),
    )
    return stats


def calculate_closed_position_statistics(clock: datetime.datetime, position: TradingPosition) -> PositionStatistics:
    stats = FinalPositionStatistics(
        calculated_at=clock,
        first_trade_at=position.get_first_trade().executed_at,
        last_trade_at=position.get_last_trade().executed_at,
        trade_count=len(position.trades),
    )
    return stats


def calculate_statistics(clock: datetime.datetime, portfolio: Portfolio) -> NewStatistics:
    """Calculate statistics for a portfolio."""

    pf_stats = PortfolioStatistics(
        calculated_at=clock,
        total_equity=portfolio.get_total_equity(),
        open_position_count=len(portfolio.open_positions),
        open_position_equity=portfolio.get_open_position_equity(),
        frozen_position_equity=portfolio.get_frozen_position_equity(),
        frozen_position_count=len(portfolio.frozen_positions),
        closed_position_count=len(portfolio.closed_positions),
        free_cash=portfolio.get_current_cash(),
        unrealised_profit_usd=portfolio.get_unrealised_profit_usd(),
        closed_profit_usd=portfolio.get_closed_profit_usd(),
    )

    stats = NewStatistics(portfolio=pf_stats)

    for position in portfolio.open_positions.values():
        stats.positions[position.position_id] = calculate_position_statistics(clock, position)

    for position in portfolio.frozen_positions.values():
        stats.positions[position.position_id] = calculate_position_statistics(clock, position)

    return stats



def update_statistics(clock: datetime.datetime, stats: Statistics, portfolio: Portfolio):
    """Update statistics in a portfolio with a new cycle."""

    new_stats = calculate_statistics(clock, portfolio)
    stats.portfolio.append(new_stats.portfolio)
    for position_id, position_stats in new_stats.positions.items():
        stats.positions[position_id].append(position_stats)

    # Check if we have missing closed position statistics
    # by comparing which closed ids are missing from the stats list
    closed_position_stat_ids = set(stats.closed_positions.keys())
    all_closed_position_ids = set(portfolio.closed_positions.keys())

    missing_closed_position_stats = all_closed_position_ids - closed_position_stat_ids
    for position_id in missing_closed_position_stats:
        position = portfolio.closed_positions[position_id]
        # Calculate the closing value for the profitability
        stats.positions[position_id].append(calculate_position_statistics(clock, position))
        # Calculate stats that are only available for closed positions
        stats.closed_positions[position_id] = calculate_closed_position_statistics(clock, position)
