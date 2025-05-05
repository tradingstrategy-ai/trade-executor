"""Retry failed trades.
"""
import datetime
import logging
from itertools import chain

from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeType, TradeStatus
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.routing import RoutingModel, RoutingState
from .repair import RepairAborted, RepairResult, unfreeze_position

logger = logging.getLogger(__name__)


def find_trades_to_be_retried(state: State) -> list[TradeExecution]:
    trades_to_be_repaired = []
    # Closed trades do not need attention
    for p in chain(state.portfolio.open_positions.values(), state.portfolio.frozen_positions.values()):
        t: TradeExecution
        for t in p.trades.values():
            if t.is_repair_needed():
                logger.info("Found a trade needing repair: %s", t)
                trades_to_be_repaired.append(t)

    return trades_to_be_repaired


def rebroadcast_trade(
    t: TradeExecution,
    *,
    state: State,
    execution_model: ExecutionModel,
    routing_model: RoutingModel,
    routing_state: RoutingState,
):
    assert t.get_status() == TradeStatus.failed

    logger.info("Rebroadcasting trade: %s", t)

    execution_model.execute_trades(
        datetime.datetime.utcnow(),
        state,
        [t],
        routing_model,
        routing_state,
        rebroadcast=True,
    )

    t.repaired_at = datetime.datetime.utcnow()
    t.add_note(f"Failed trade rebroadcasted at {t.repaired_at}")

    return t


def retry_trades(
    *,
    state: State,
    execution_model: ExecutionModel,
    routing_model: RoutingModel,
    routing_state: RoutingState,
    attempt_repair=True,
    interactive=True
) -> RepairResult:
    """Retry trade.

    - Find frozen positions and trades in them

    - Rebroadcast failed trades

    :param attempt_repair:
        If not set, only list broken trades and frozen positions.

        Do not attempt to repair them.

    :param interactive:
        Command line interactive user experience.

        Allows press `n` for abort.

    :raise RepairAborted:
        User chose no
    """

    logger.info("Repairing trades")

    frozen_positions = list(state.portfolio.frozen_positions.values())

    logger.info("Strategy has %d frozen positions", len(frozen_positions))

    trades_to_be_retried = find_trades_to_be_retried(state)

    logger.info("Found %d trades to be retried", len(trades_to_be_retried))

    if len(trades_to_be_retried) == 0 or not attempt_repair:
        return RepairResult(
            frozen_positions,
            [],
            trades_to_be_retried,
            [],
        )

    if interactive:
        for t in trades_to_be_retried:
            print("Needs repair:", t)

        confirmation = input("Attempt to repair [y/n] ").lower()
        if confirmation != "y":
            raise RepairAborted()
        
    new_trades = []
    for t in trades_to_be_retried:
        new_trades.append(
            rebroadcast_trade(
                t,
                state=state,
                execution_model=execution_model,
                routing_model=routing_model,
                routing_state=routing_state,
            )
        )
    
    unfrozen_positions = []
    for p in frozen_positions:
        if unfreeze_position(state.portfolio, p):
            unfrozen_positions.append(p)
            logger.info("Position unfrozen: %s", p)

    for t in new_trades:
        logger.info("Correction trade made: %s", t)

    return RepairResult(
        frozen_positions,
        unfrozen_positions,
        trades_to_be_retried,
        new_trades,
    )
