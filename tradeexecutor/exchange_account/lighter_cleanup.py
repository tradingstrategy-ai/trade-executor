"""Close Lighter positions of accounts the strategy no longer declares.

Lighter's own close and dust rules, separate from the HyperCore ones in
``tradeexecutor.ethereum.vault.hypercore_small_position_cleanup``. Lighter
cannot withdraw less than :data:`~tradeexecutor.exchange_account.lighter.LIGHTER_DUST_USDC`,
so a residual below it can never return to the Safe. When the strategy stops
declaring a sub-account, ``correct-accounts`` closes its position once the
account holds less than that and writes the residual off. An account still
holding more is left open for the operator to drain: a Lighter withdrawal
takes about half an hour to become claimable, so it is never started
automatically.
"""

import datetime
import logging

from tradeexecutor.exchange_account.lighter import LIGHTER_DUST_USDC, is_lighter_exchange_account_pair
from tradeexecutor.exchange_account.state import close_exchange_account_position
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse

logger = logging.getLogger(__name__)


def close_undeclared_lighter_positions(
    state: State,
    universe: TradingStrategyUniverse,
    executed_at: datetime.datetime,
) -> list[TradingPosition]:
    """Close the positions of Lighter accounts the strategy no longer declares.

    Run after the exchange-account sync has booked each account's live
    equity on its position, so a position's quantity is the account's
    equity. Declared accounts are never touched; the strategy manages them.
    A position left open or frozen here still tracks an undeclared account,
    which :py:func:`~tradeexecutor.exchange_account.lighter.check_lighter_account_set`
    reports, so the caller exits unclean.

    :param state:
        Strategy state, changed in place.
    :param universe:
        Strategy universe; its Lighter pairs are the declared accounts.
    :param executed_at:
        Timestamp of the closes.
    :return:
        The closed positions.
    """
    declared = {pair for pair in universe.iterate_pairs() if is_lighter_exchange_account_pair(pair)}
    closed = []
    for position in state.portfolio.frozen_positions.values():
        if is_lighter_exchange_account_pair(position.pair) and position.pair not in declared:
            logger.warning("Lighter position #%d of undeclared account %s is frozen; run repair first", position.position_id, position.pair.get_exchange_account_id())
    for position in list(state.portfolio.open_positions.values()):
        if not is_lighter_exchange_account_pair(position.pair) or position.pair in declared:
            continue
        account_index = position.pair.get_exchange_account_id()
        equity = position.get_quantity()
        if equity < LIGHTER_DUST_USDC:
            close_exchange_account_position(
                state,
                position,
                executed_at,
                notes=f"Lighter account {account_index} is no longer declared by the strategy; {equity} USDC below the {LIGHTER_DUST_USDC} USDC withdrawal minimum written off",
            )
            logger.info("Closed Lighter position #%d of undeclared account %s, wrote off %s USDC", position.position_id, account_index, equity)
            closed.append(position)
        else:
            logger.error(
                "Lighter account %s holds %s USDC but the strategy no longer declares it; drain it with "
                "lighter-move-funds --lighter-account-index %s and rerun correct-accounts",
                account_index,
                equity,
                account_index,
            )
    return closed
