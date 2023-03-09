"""Maanually repair broken states.

- Recover from failed trades

Trade failure modes may include

- Broadcasted but not confirmed

- Broadcasted, confirmed, but not marked as executed

- Executed, failed

"""
import datetime
import logging
from dataclasses import dataclass
from itertools import chain
from typing import List, TypedDict

from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution

logger = logging.getLogger(__name__)


class RepairAborted(Exception):
    """User chose no"""


@dataclass(slots=Trueno)
class RepairStats:
    """"""

    #: How many frozen positions we encountered
    frozen_positions: int = 0

    #: How many individal trades we repaired
    repaired_trades: int = 0

    #: How many positions we recovered
    recovered_positions: int = 0


def make_counter_trade(t: TradeExecution) -> TradeExecution:
    """Make a virtual trade that fixes the total balances of a position and unwinds the broken trade."""



def repair_trade(t: TradeExecution) -> TradeExecution:
    counter_trade = make_counter_trade(t)
    t.repaired_at = datetime.datetime.utcnow()
    return counter_trade


def find_trades_to_be_repaired(state: State) -> List[TradeExecution]:

    trades_to_be_repaired = []

    # Closed trades do not need attention
    for p in chain(state.portfolio.open_positions.values(), state.portfolio.frozen_positions.values()):
        t: TradeExecution
        for t in p.trades.values():
            if t.is_repair_needed():
                logger.info("Found a trade needing repair: %s", t)
                trades_to_be_repaired.append(t)

    return trades_to_be_repaired


def reconfirm_trade(reconfirming_needed_trades: List[TradeExecution]):

    raise NotImplementedError("Unfinished")

    for t in reconfirming_needed_trades:
        assert t.get_status() == TradeStatus.broadcasted

        receipt_data = wait_trades_to_complete(
            self.web3,
            [t],
            max_timeout=self.confirmation_timeout,
            confirmation_block_count=self.confirmation_block_count,
        )

        assert len(receipt_data) > 0, f"Got bad receipts: {receipt_data}"

        tx_data = {tx.tx_hash: (t, tx) for tx in t.blockchain_transactions}

        self.resolve_trades(
            datetime.datetime.now(),
            state,
            tx_data,
            receipt_data,
            stop_on_execution_failure=True)

        t.repaired_at = datetime.datetime.utcnow()
        if not t.notes:
            # Add human readable note,
            # but don't override any other notes
            t.notes = "Failed broadcast repaired"

        repaired.append(t)


def repair_trades(state: State, interactive=True) -> List[TradeExecution]:
    """Repair trade.

    - Find frozen positions and trades in them

    - Mark trades invalidated

    - Make the necessary counter trades to fix the total balances

    - Does not actually broadcast any transactions - only fixes the internal accounting

    :param interactive:
        Command line interactive user experience

    :raise RepairAborted:
        User chose no
    """

    stats = RepairStats()

    trades_to_be_repaired = []

    logger.info("Repairing trades, using execution model %s", self)

    total_trades = 0

    stats.frozen_positions = len(list(state.portfolio.frozen_positions))

    logger.info("Strategy has %d frozen positions", stats.frozen_positions)\

    trades_to_be_repaired = find_trades_to_be_repaired(state)

    logger.info("Found %d trades to be repaired", len(trades_to_be_repaired))

    if not trades_to_be_repaired:
        return []

    for t in trades_to_be_repaired:
        print(t)

    if interactive:
        confirmation = input("Attempt to repair [y/n]").lower()
        if confirmation != "y":
            raise RepairAborted()

    repair_trades(trades_to_be_repaired)

    return trades_to_be_repaired
