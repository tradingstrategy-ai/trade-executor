"""Rebroadcast unstarted/unfinished txs in trades."""
import datetime
from typing import Tuple, List
import logging

from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.routing import RoutingModel, RoutingState


logger = logging.getLogger(__name__)


def rebroadcast_all(
    state: State,
    execution_model: ExecutionModel,
    routing_model: RoutingModel,
    routing_state: RoutingState,
) -> Tuple[List[TradeExecution], List[BlockchainTransaction]]:

    txs = []
    trades = []

    for t in state.portfolio.get_all_trades():
        if t.is_unfinished():
            logger.info("Marking trade %s for rebroadcast", t)
            assert t.blockchain_transactions, f"Trade marked unfinished, did not have any txs: {t}"
            for tx in t.blockchain_transactions:

                now = datetime.datetime.utcnow()
                t.add_note(f"Rebroadcasting transaction at {now}")

                txs.append(tx)
                trades.append(t)

    execution_model.execute_trades(
        datetime.datetime.utcnow(),
        state,
        trades,
        routing_model,
        routing_state,
        rebroadcast=True,
    )

    return trades, txs
