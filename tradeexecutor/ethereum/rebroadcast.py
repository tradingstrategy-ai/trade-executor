"""Rebroadcast unstarted/unfinished txs in trades."""
import datetime
from typing import Tuple, List
import logging

from eth_typing import HexStr
from web3 import Web3
from web3.exceptions import TransactionNotFound

from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.routing import RoutingModel, RoutingState


logger = logging.getLogger(__name__)


def rebroadcast_all(
    web3: Web3,
    state: State,
    execution_model: ExecutionModel,
    routing_model: RoutingModel,
    routing_state: RoutingState,
) -> Tuple[List[TradeExecution], List[BlockchainTransaction]]:
    """Check the state for unconfirmed transactions and attempt to fix them.

    - Check if tx was correctly broadcasted, but we never received a receipt
      from a node

    - Re-sign txs and rebroadcast if needed

    :return:
        Trades we fixed, old broken txs.

        Old broken txs might have been replaced with new txs.
    """
    txs = []
    trades = []

    for t in state.portfolio.get_all_trades():
        if t.is_unfinished():
            logger.info("Marking trade %s for rebroadcast", t)
            assert t.blockchain_transactions, f"Trade marked unfinished, did not have any txs: {t}"
            for tx in t.blockchain_transactions:

                now = datetime.datetime.utcnow()
                t.add_note(f"Rebroadcasting transaction at {now}")

                # TODO: we should call make_trade_success() / failed here directly,
                # but we do not have examples of such txs
                try:
                    web3.eth.get_transaction_receipt(HexStr(tx.tx_hash))
                    raise NotImplementedError(f"The tx is on a chain already: {tx.tx_hash}, we do not have a code path to handle this yet")
                except TransactionNotFound:
                    pass

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
