"""Frozen position management."""
import logging
from typing import List, Tuple

from tradeexecutor.state.state import TradeExecution, State


logger = logging.getLogger(__name__)


def freeze_position_on_failed_trade(state: State, trades: List[TradeExecution]) -> Tuple[List[TradeExecution], List[TradeExecution]]:
    """Move positions of failed sell trades to freeze list.

    Any base tokens will be added to the blacklist.

    :return: Tuple List of succeeded trades, List of failed trades
    """

    failed = []
    succeeded = []

    portfolio = state.portfolio

    for t in trades:
        # TODO: Check if we need special logic for failed buy trades
        if t.is_failed():

            logger.warning("Freezing position for a failed trade: %s", t)

            position = portfolio.get_position_by_trading_pair(t.pair)
            assert position.is_open(), "Cannot freeze position that is not open"
            portfolio.frozen_positions[position.position_id] = position
            del portfolio.open_positions[position.position_id]

            state.blacklist_asset(position.pair.base)

            failed.append(t)
        else:
            succeeded.append(t)

    return succeeded, failed


