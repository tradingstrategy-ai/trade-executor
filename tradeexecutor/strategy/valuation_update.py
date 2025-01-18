"""Valuation update logic handling.

- Internal valuation calculations

- Posting new valuation onchain
"""

import datetime
import logging

from tradeexecutor.state.state import State
from tradeexecutor.statistics.core import update_statistics
from tradeexecutor.statistics.statistics_table import StatisticsTable
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.routing import RoutingState
from tradeexecutor.strategy.universe_model import StrategyExecutionUniverse
from tradeexecutor.strategy.valuation import revalue_state, ValuationModel

logger = logging.getLogger(__name__)


def update_position_valuations(
    timestamp: datetime.datetime,
    state: State,
    universe: StrategyExecutionUniverse,
    execution_context: ExecutionContext,
    valuation_model: ValuationModel,
    routing_state: RoutingState,
    long_short_metrics_latest: StatisticsTable | None = None,
):
    """Revalue positions and update statistics.

    - Revalue all positions

    - Push new valuation update to onchain if needed (Lagoon)

    - Update statistics

    A new statistics entry is calculated for portfolio and all of its positions
    and added to the state.

    Example:



    :param timestamp:
        Real-time or historical clock

    :param long_short_metrics_latest:
        Needed to calculate short statistics.

        Can be None - statistics skipped.
    """

    # Set up the execution to perform the valuation

    assert isinstance(universe, StrategyExecutionUniverse)
    assert isinstance(execution_context, ExecutionContext)
    assert isinstance(routing_state, RoutingState)

    timed_task_context_manager = execution_context.timed_task_context_manager

    if len(state.portfolio.reserves) == 0:
        logger.info("The strategy has no reserves or deposits yet")

    # TODO: this seems to be duplicated in tick()
    with timed_task_context_manager("revalue_portfolio_statistics"):
        logger.info("Updating position valuations")
        revalue_state(state, timestamp, valuation_model)

    with timed_task_context_manager("update_statistics"):
        logger.info("Updating position statistics after revaluation")

        update_statistics(
            timestamp,
            state.stats,
            state.portfolio,
            execution_context.mode,
            long_short_metrics_latest=long_short_metrics_latest,
        )

