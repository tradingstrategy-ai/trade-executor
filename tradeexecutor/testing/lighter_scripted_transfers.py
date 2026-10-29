"""Test support for strategies that script Lighter custody transfers.

The example strategy ``strategies/test_only/lighter_scripted_transfers_strategy.py``
moves funds in, out and between its Lighter accounts with the
``PositionManager`` custody methods, one schedule step per cycle. This module
provides what the tutorial and the black-box test need around it: the step
bookkeeping, the expected schedule and a verifier that checks every movement was booked as its own trade
between two positions.

Progress lives in the trades themselves: each scripted trade records its
schedule step, so a rerun continues from the first step without successful
trades and a failed step stops the script.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from tradeexecutor.exchange_account.lighter import LIGHTER_MASTER_ACCOUNT_ID
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeFlag

logger = logging.getLogger(__name__)

#: Label of the Lagoon Safe, the executor's reserve, in a schedule.
ROLE_SAFE = "safe"

#: ``other_data`` key holding the schedule step of a scripted trade.
SCRIPTED_STEP_KEY = "lighter_scripted_step"

#: Base amount of the scripted movements: twice Lighter's 1 USDC minimum, so a 0.5 USDC fee leaves every movement above it.
SCRIPTED_TRANSFER_UNIT_USDC = Decimal(2)

#: Steps in the script, one per cycle.
SCRIPTED_TRANSFER_STEPS = 5


@dataclass(frozen=True, slots=True)
class ScriptedTransfer:
    """One expected custody movement."""

    #: Account the value leaves: ``safe``, ``master`` or a sub-account id.
    source: str

    #: Account the value arrives at.
    destination: str

    #: USDC moved, or ``None`` when the strategy moves everything available on the source.
    amount: Decimal | None

    def is_funding(self) -> bool:
        """Return whether this is a transfer between two Lighter accounts."""
        return ROLE_SAFE not in (self.source, self.destination)


def build_lighter_transfer_schedule(sub_account_ids: list[str]) -> list[list[ScriptedTransfer]]:
    """Return the movements the example strategy makes, one list per cycle.

    With the 1 USDC activation deposit and a ``5 * unit - 1`` USDC
    subscription, the master holds ``5 * unit`` after the first step and every
    later movement is at least ``unit``. A transfer fee reduces the balances,
    so ``unit`` needs margin above the fee when one is charged.

    1. Deposit all idle Safe cash to the master.
    2. Fund both sub-accounts from the master with ``2 * unit`` each.
    3. Move ``unit`` from the first sub-account to the second, signed by the first.
    4. Move ``unit`` from the second sub-account back to the master, signed by the second.
    5. Withdraw everything from both sub-accounts and the master to the Safe.

    :param sub_account_ids:
        The two sub-account ids in declaration order.
    :return:
        Schedule steps in execution order.
    """
    unit = SCRIPTED_TRANSFER_UNIT_USDC
    first, second = sub_account_ids
    master = LIGHTER_MASTER_ACCOUNT_ID
    return [
        [ScriptedTransfer(ROLE_SAFE, master, None)],
        [ScriptedTransfer(master, first, 2 * unit), ScriptedTransfer(master, second, 2 * unit)],
        [ScriptedTransfer(first, second, unit)],
        [ScriptedTransfer(second, master, unit)],
        [ScriptedTransfer(first, ROLE_SAFE, None), ScriptedTransfer(second, ROLE_SAFE, None), ScriptedTransfer(master, ROLE_SAFE, None)],
    ]


def get_scripted_trades(state: State) -> dict[int, list[TradeExecution]]:
    """Return the scripted trades in state, keyed by schedule step."""
    by_step: dict[int, list[TradeExecution]] = defaultdict(list)
    for trade in state.portfolio.get_all_trades():
        step = (trade.other_data or {}).get(SCRIPTED_STEP_KEY)
        if step is not None:
            by_step[int(step)].append(trade)
    return dict(by_step)


def get_scripted_progress(state: State, step_count: int) -> tuple[int, str | None]:
    """Return how many schedule steps have fully succeeded, and why progress stopped.

    :param state:
        Executor state.
    :param step_count:
        Number of steps in the schedule.
    :return:
        ``(completed_steps, problem)``; ``problem`` is ``None`` while the
        script can continue.
    """
    by_step = get_scripted_trades(state)
    completed = 0
    for step in range(1, step_count + 1):
        trades = by_step.get(step, [])
        if not trades:
            break
        if all(trade.is_success() for trade in trades):
            completed = step
            continue
        if any(trade.is_exchange_account_transfer_pending() for trade in trades):
            return completed, f"step {step} has a pending transfer"
        failed = [trade.trade_id for trade in trades if not trade.is_success()]
        return completed, f"step {step} has failed or expired trades {failed}"
    return completed, None


def get_next_scripted_step(state: State, step_count: int) -> int | None:
    """Return the next schedule step to run, or ``None`` when the script is done or stopped."""
    completed, problem = get_scripted_progress(state, step_count)
    if problem is not None:
        logger.error("Lighter scripted transfers stopped: %s", problem)
        return None
    if completed >= step_count:
        logger.info("Lighter scripted transfers completed all %d steps", step_count)
        return None
    return completed + 1


def tag_scripted_trades(trades: list[TradeExecution], step: int) -> list[TradeExecution]:
    """Record the schedule step on each trade and return them."""
    for trade in trades:
        trade.other_data[SCRIPTED_STEP_KEY] = step
    return trades


def verify_lighter_scripted_transfers(
    state: State,
    schedule: list[list[ScriptedTransfer]],
    accounts: dict[str, int],
) -> list[dict[str, Any]]:
    """Check that every scheduled movement was booked as one successful trade between the right positions.

    Checks, per movement: exactly one trade of its step moves between its
    accounts; the trade succeeded; a fixed amount was moved exactly; a
    withdrawal claimed what it moved; and a funding's source position carries
    balance updates for the amount and the fee. Then checks that no transfer
    is pending, that every Lighter position is empty, that the Lighter
    inflows, revaluations, outflows and fees add up, and that each account's
    flow-neutral USD profit is minus the fees it paid: the script only moves
    capital, so no account earns anything.

    :param state:
        Executor state after the vault has been wound up.
    :param schedule:
        Movements the strategy was expected to make.
    :param accounts:
        Account id to Lighter account index, including ``master``.
    :return:
        One row per movement for reporting.
    :raise AssertionError:
        With every problem found, if any.
    """
    positions = [position for position in state.portfolio.get_all_positions() if position.is_exchange_account()]
    by_step = get_scripted_trades(state)
    problems: list[str] = []
    rows: list[dict[str, Any]] = []

    for step, transfers in enumerate(schedule, 1):
        trades = by_step.get(step, [])
        if len(trades) != len(transfers):
            problems.append(f"step {step}: expected {len(transfers)} trades, found {len(trades)}")
        for transfer in transfers:
            matching = [trade for trade in trades if _matches(trade, transfer, accounts)]
            if len(matching) != 1:
                problems.append(f"step {step}: expected one trade {transfer.source} to {transfer.destination}, found {len(matching)}")
                continue
            row = _verify_transfer(state, matching[0], transfer, problems)
            row["step"] = step
            rows.append(row)

    pending = [trade.trade_id for trade in state.portfolio.get_all_trades() if trade.is_exchange_account_transfer_pending()]
    if pending:
        problems.append(f"pending transfers remain: {pending}")
    for position in positions:
        if position.get_quantity() != 0:
            problems.append(f"Lighter account {position.pair.get_exchange_account_id()} position #{position.position_id} still holds {position.get_quantity()} USDC")
    _check_conservation(positions, problems)
    _check_flow_neutral_profit(state, positions, problems)

    if problems:
        raise AssertionError("Scripted Lighter transfers were not booked correctly:\n" + "\n".join(problems))
    state.check_if_clean()
    return rows


def _matches(trade: TradeExecution, transfer: ScriptedTransfer, accounts: dict[str, int]) -> bool:
    """Return whether a trade moves between the accounts of a scheduled transfer."""
    account = int(trade.pair.get_exchange_account_id())
    if trade.is_internal_account_transfer():
        return transfer.is_funding() and account == accounts[transfer.destination] and int(trade.other_data.get("funding_source_account_id", -1)) == accounts[transfer.source]
    if not trade.is_external_account_transfer():
        return False
    if trade.is_buy():
        return transfer.source == ROLE_SAFE and account == accounts[transfer.destination]
    return transfer.destination == ROLE_SAFE and account == accounts[transfer.source]


def _check_conservation(positions: list[TradingPosition], problems: list[str]) -> None:
    """Check that Lighter inflows, revaluations, outflows and fees explain what remains.

    Deposits add, withdrawals and transfer fees remove, and transfers between
    accounts net to zero. Balance updates not booked by a funding trade are
    revaluations from Lighter, such as the activation deposit made before the
    executor existed.
    """
    inflow = outflow = fees = revalued = Decimal(0)
    funding_update_ids: set[tuple[int, int]] = set()
    for position in positions:
        for trade in position.trades.values():
            if not trade.is_success():
                continue
            if trade.is_internal_account_transfer():
                fees += Decimal(trade.other_data.get("funding_fee_usdc", "0"))
                source_id = int(trade.other_data["funding_source_position_id"])
                funding_update_ids.update((source_id, update_id) for update_id in trade.other_data.get("funding_source_balance_update_ids", []))
            elif trade.is_buy():
                inflow += trade.executed_quantity
            else:
                outflow += -trade.executed_quantity
    for position in positions:
        for update in position.balance_updates.values():
            if (position.position_id, update.balance_update_id) not in funding_update_ids:
                revalued += update.quantity
    remaining = sum((position.get_quantity() for position in positions), Decimal(0))
    if inflow + revalued - outflow - fees != remaining:
        problems.append(f"Lighter inflow {inflow} plus revaluations {revalued} minus outflow {outflow} and fees {fees} is not the remaining {remaining}")


def _check_flow_neutral_profit(state: State, positions: list[TradingPosition], problems: list[str]) -> None:
    """Check that moving capital left each account with a profit of minus the fees it paid.

    The fees are the Lighter transfer fees of the funding trades the account
    sent, and the shortfall of its withdrawals, where the Safe received less
    than the account was debited. The scripted run starts from a fresh vault,
    so every tracked position must know its withdrawn total; an unknown one
    is a problem rather than a skipped check.
    """
    fees_paid: dict[int, Decimal] = {}
    for trade in state.portfolio.get_all_trades():
        if not trade.is_success():
            continue
        if trade.is_internal_account_transfer():
            source_id = int(trade.other_data["funding_source_position_id"])
            fees_paid[source_id] = fees_paid.get(source_id, Decimal(0)) + Decimal(trade.other_data.get("funding_fee_usdc", "0"))
        elif trade.is_external_account_transfer() and trade.is_sell():
            shortfall = -trade.executed_quantity - trade.executed_reserve
            fees_paid[trade.position_id] = fees_paid.get(trade.position_id, Decimal(0)) + shortfall
    for position in positions:
        if position.share_price_state is None:
            continue
        profit = position.get_flow_neutral_profit_usd()
        expected = -float(fees_paid.get(position.position_id, Decimal(0)))
        if profit is None:
            problems.append(f"Lighter account {position.pair.get_exchange_account_id()} position #{position.position_id} has no withdrawn total, so its USD profit is unknown")
        elif abs(profit - expected) > 1e-6:
            problems.append(f"Lighter account {position.pair.get_exchange_account_id()} position #{position.position_id} reports a USD profit of {profit}, expected minus its fees {expected}")


def _verify_transfer(state: State, trade: TradeExecution, transfer: ScriptedTransfer, problems: list[str]) -> dict[str, Any]:
    """Check one scripted trade against its movement and return its report row."""
    label = f"trade #{trade.trade_id} {transfer.source} to {transfer.destination}"
    amount = abs(trade.executed_quantity or Decimal(0))
    fee = Decimal(0)
    if not trade.is_success():
        problems.append(f"{label} did not succeed: {trade.get_status().value}")
    if transfer.source == ROLE_SAFE:
        kind = "deposit"
    elif transfer.destination == ROLE_SAFE:
        kind = "withdrawal"
        if Decimal(trade.other_data.get("lighter_claimable_usdc", "0")) != amount:
            problems.append(f"{label} claimed {trade.other_data.get('lighter_claimable_usdc')} USDC, expected {amount}")
    else:
        kind = "funding"
        fee = Decimal(trade.other_data.get("funding_fee_usdc", "0"))
        source_position = state.portfolio.get_position_by_id(int(trade.other_data["funding_source_position_id"]))
        update_ids = trade.other_data.get("funding_source_balance_update_ids", [])
        if not update_ids:
            problems.append(f"{label} has no source balance updates")
        else:
            booked = sum((source_position.balance_updates[update_id].quantity for update_id in update_ids if update_id in source_position.balance_updates), Decimal(0))
            if booked != -(amount + fee):
                problems.append(f"{label} booked {booked} on the source, expected {-(amount + fee)}")
    if transfer.amount is not None and amount != transfer.amount:
        problems.append(f"{label} moved {amount} USDC, expected {transfer.amount}")
    return {
        "trade_id": trade.trade_id,
        "kind": kind,
        "source": transfer.source,
        "destination": transfer.destination,
        "amount": amount,
        "fee": fee,
        "status": trade.get_status().value,
    }
