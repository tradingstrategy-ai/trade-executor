"""Core portfolio statistics calculations."""

import logging
import datetime
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional
from tradeexecutor.analysis.trade_analyser import build_trade_analysis

from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.statistics import Statistics, PortfolioStatistics, PositionStatistics, FinalPositionStatistics
from tradeexecutor.state.sync import Treasury
from tradeexecutor.state.types import LegacyDataException, USDollarPrice
from tradeexecutor.strategy.execution_context import ExecutionMode
from tradeexecutor.statistics.statistics_table import StatisticsTable
from tradeexecutor.visual.equity_curve import calculate_compounding_unrealised_trading_profitability

logger = logging.getLogger(__name__)


@dataclass
class TimestampedStatistics:
    """Statistics calculated for the portfolio on each stats cycle.

    Can be collected at

    - And of the strategy cycle

    - During the position revaluation
    """

    #: Statistics for the whole portfolio
    #:
    #: Inc. quant finance metrics
    #:
    portfolio: PortfolioStatistics

    #: Statistics for each individual position that was open at the time of colleciton
    positions: Dict[int, PositionStatistics] = field(default_factory=dict)

    #: Strategy cycle timestamp
    #:
    #: Not available if the statistics where calculated
    #: out of rebalance
    #:
    strategy_cycle_at: datetime.datetime | None = None

    #: Number of issued shares
    share_count: Optional[Decimal] = None

    #: Share price
    #:
    #: Derived from net asset value / share count
    #:
    share_price_usd: Optional[USDollarPrice] = None


def calculate_position_statistics(clock: datetime.datetime, position: TradingPosition) -> PositionStatistics:
    """Generate a historical record of the current position state."""
    # first_trade = position.get_first_trade()

    value = position.get_value()

    # TODO: Ignore. Common occurence. Position may lack open value if it was never properly opened.
    #
    #if position.is_open() and not position.is_repaired():
        # Normally opened positions should always have some value
    #if value == 0:
    #        logger.warning(f"Position {position} reported value {value}. Last token price: {position.last_token_price}. Last reserve price: {position.last_reserve_price}")


    if position.is_credit_supply():
        # Special path to calculate accured interest.
        # This will be mean %
        profitability = position.estimate_gained_interest()
    else:
        # TODO: Update this to new code paths
        profitability = position.get_total_profit_percent()

    stats = PositionStatistics(
        calculated_at=clock,
        last_valuation_at=position.last_pricing_at,
        profitability=profitability,
        profit_usd=position.get_total_profit_usd(),
        quantity=float(position.get_quantity_old()),
        value=value,
    )

    return stats


def calculate_closed_position_statistics(
        clock: datetime.datetime,
        position: TradingPosition,
        position_stats: List[PositionStatistics]) -> FinalPositionStatistics:
    """Calculate the statistics for closed positions."""

    # This position should have at least some relevant value calculated
    # for it
    first_trade = position.get_first_trade()

    # TODO: Is also recorded at position.portfolio_value_at_open
    # after v311
    value_at_open = position_stats[0].value

    if first_trade.is_success():
        if value_at_open == 0:
            # TODO:
            # We should really abort here with an exception
            # but we have some legacy data that prevents this
            logger.info(f"PositionStatistics lacked value at open for {position}.\n"
                           f"Stats:\n"
                           f"{position_stats}")

    value_at_max = max([(p.value or 0) for p in position_stats])
    stats = FinalPositionStatistics(
        calculated_at=clock,
        trade_count=len(position.trades),
        value_at_open=value_at_open,
        value_at_max=value_at_max,
    )
    return stats


def calculate_statistics(
    clock: datetime.datetime,
    portfolio: Portfolio,
    execution_mode: ExecutionMode,
    strategy_cycle_at: datetime.datetime | None = None,
    treasury: Treasury = None,
) -> TimestampedStatistics:
    """Calculate statistics for a portfolio.

    :param trasury:
        Needed to track share price.
    """

    first_trade, last_trade = portfolio.get_first_and_last_executed_trade()

    total_equity = portfolio.calculate_total_equity()

    # May not available for non-live execution and non-vault strategies
    if treasury:
        share_count = treasury.share_count
        if share_count:
            share_price_usd = total_equity / float(share_count)
        else:
            share_price_usd = None
    else:
        share_count = None
        share_price_usd = None

    # comprehensenhive statistics after each trade are not needed for backtesting
    if (execution_mode != ExecutionMode.backtesting):

        trade_analysis = build_trade_analysis(portfolio)

        try:
            profitability_series = calculate_compounding_unrealised_trading_profitability(portfolio)
        except LegacyDataException:
            # Unit tests running on legacy state files
            profitability_series = []
        
        pf_stats = PortfolioStatistics(
            calculated_at=clock,
            total_equity=total_equity,
            net_asset_value=portfolio.get_net_asset_value(),
            unrealised_profitability=float(profitability_series.iloc[-1] if len(profitability_series) > 0 else 0),
            free_cash=float(portfolio.get_cash()),
            open_position_count=len(portfolio.open_positions),
            open_position_equity=portfolio.get_position_equity_and_loan_nav(),
            frozen_position_equity=portfolio.get_frozen_position_equity(),
            frozen_position_count=len(portfolio.frozen_positions),
            closed_position_count=len(portfolio.closed_positions),
            unrealised_profit_usd=portfolio.get_unrealised_profit_usd(),
            realised_profit_usd=portfolio.get_closed_profit_usd(),
            first_trade_at=first_trade and first_trade.executed_at or None,
            last_trade_at=last_trade and last_trade.executed_at or None,
            summary=trade_analysis.calculate_summary_statistics(),
        )

    else:
        pf_stats = PortfolioStatistics(
            calculated_at=clock,
            total_equity=total_equity,
            net_asset_value=portfolio.get_net_asset_value(),
            free_cash=float(portfolio.get_cash()),
        )

    stats = TimestampedStatistics(
        portfolio=pf_stats,
        strategy_cycle_at=strategy_cycle_at,
        share_count=share_count,
        share_price_usd=share_price_usd,
    )

    for position in portfolio.open_positions.values():
        stats.positions[position.position_id] = calculate_position_statistics(clock, position)

    for position in portfolio.frozen_positions.values():
        stats.positions[position.position_id] = calculate_position_statistics(clock, position)

    return stats


def update_statistics(
    clock: datetime.datetime,
    stats: Statistics,
    portfolio: Portfolio,
    execution_mode: ExecutionMode,
    strategy_cycle_or_wall_clock: datetime.datetime | None = None,
    long_short_metrics_latest: StatisticsTable | None = None,
    treasury: Treasury | None = None,
):
    """Update statistics in a portfolio.

    TODO: Rename to `update_trading_statistics()`.

    Can be collected at

    - And of the strategy cycle

    - During the position revaluation

    - Other ad-hoc timestamp

    .. note ::

        You *must* make sure this function is called any time a position has been opened
        or closed, or relevant statistics may get corrupted due to holes in them.

    :param clock:
        Real-time clock of the update.

        Always available.

    :param strategy_cycle_or_wall_clock:
        The strategy cycle timestamp.

        Wall clock time if called outside the strategy cycle.

        Only available when `update_statistics()` is run
        at the end of live trading cycle.
    """
    logger.info(
        "update_statistics(), real-time clock at %s, strategy cycle at %s",
        clock,
        strategy_cycle_or_wall_clock
    )
    
    if execution_mode.is_live_trading() and execution_mode not in (ExecutionMode.unit_testing_trading, ExecutionMode.one_off):
        assert long_short_metrics_latest, f"long short metrics should be provided in live trading, trading mode is: {execution_mode}"
    
    if long_short_metrics_latest:
        logger.info("Serialising long_short_metrics_latest")
    
    stats.long_short_metrics_latest = long_short_metrics_latest
    
    new_stats = calculate_statistics(
        clock,
        portfolio,
        execution_mode,
        treasury=treasury,
    )
    stats.portfolio.append(new_stats.portfolio)
    for position_id, position_stats in new_stats.positions.items():
        stats.add_positions_stats(position_id, position_stats)

    # Check if we have missing closed position statistics
    # by comparing which closed ids are missing from the stats list
    closed_position_stat_ids = set(stats.closed_positions.keys())
    all_closed_position_ids = set(portfolio.closed_positions.keys())

    missing_closed_position_stats = all_closed_position_ids - closed_position_stat_ids
    for position_id in missing_closed_position_stats:
        position = portfolio.closed_positions[position_id]
        # Calculate the closing value for the profitability
        position_stats = calculate_position_statistics(clock, position)
        stats.add_positions_stats(position_id, position_stats)
        # Calculate stats that are only available for closed positions
        stats.closed_positions[position_id] = calculate_closed_position_statistics(clock, position, stats.positions[position_id])

    logger.info("update_statistics() complete")
