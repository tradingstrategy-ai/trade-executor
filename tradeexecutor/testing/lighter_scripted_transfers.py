"""Scripted custody transfers between a Safe, a Lighter master account and its sub-accounts.

A production strategy sizes its Lighter custody movements from balances,
see :py:func:`tradeexecutor.exchange_account.lighter_strategy.decide_lighter_cash_management`.
A tutorial or a test instead needs a fixed, repeatable sequence of movements
whose accounting can be checked afterwards. This module provides that
sequence as a :py:func:`schedule <build_lighter_transfer_schedule>`, a
``decide_trades()`` helper that emits one schedule step per cycle, and a
verifier that checks every movement was booked as its own trade moving value
between two positions.

Every movement is one :py:class:`~tradeexecutor.state.trade.TradeExecution`:

- Safe to master: an external-account deposit, a buy on the master position
  paid from reserves.
- Master or sub-account to the Safe: an external-account withdrawal, a sell
  on that position credited to reserves.
- Any account to another account of the same owner: an internal-account
  funding, a buy on the destination position paid by balance updates on the
  source position. A transfer sent from a sub-account needs that
  sub-account's signer, see ``LIGHTER_SUB_ACCOUNT_OPERATOR_RECORD_FILES``.

Progress lives in the trades themselves: each scripted trade records its
schedule step and the roles it moves between, so a rerun continues from the
first step without successful trades and a failed step stops the script.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from typing import Any

from tradeexecutor.exchange_account.lighter import LIGHTER_PROTOCOL
from tradeexecutor.exchange_account.lighter_strategy import ensure_lighter_positions, get_lighter_account_pairs
from tradeexecutor.exchange_account.state import create_exchange_account_funding, create_exchange_account_transfer
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeFlag
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput

logger = logging.getLogger(__name__)

#: Role name of the Lagoon Safe, the executor's reserve.
ROLE_SAFE = "safe"

#: Role name of the Lighter master account.
ROLE_MASTER = "master"

#: ``other_data`` key holding the schedule step of a scripted trade.
SCRIPTED_STEP_KEY = "lighter_scripted_step"

#: ``other_data`` key holding the role the value leaves.
SCRIPTED_SOURCE_KEY = "lighter_scripted_source"

#: ``other_data`` key holding the role the value arrives at.
SCRIPTED_DESTINATION_KEY = "lighter_scripted_destination"

#: Smallest USDC unit Lighter and Ethereum USDC can move.
USDC_QUANTUM = Decimal("0.000001")


@dataclass(frozen=True, slots=True)
class ScriptedTransfer:
    """One custody movement in the schedule."""

    #: Role the value leaves: ``safe``, ``master`` or ``sub_<n>``.
    source: str

    #: Role the value arrives at.
    destination: str

    #: USDC moved, or ``None`` to move everything available on the source.
    amount: Decimal | None

    def is_funding(self) -> bool:
        """Return whether this is a transfer between two Lighter accounts."""
        return ROLE_SAFE not in (self.source, self.destination)


def get_sub_account_role(number: int) -> str:
    """Return the role name of a sub-account by its 1-based deployment order."""
    return f"sub_{number}"


def build_lighter_transfer_schedule(unit: Decimal) -> list[list[ScriptedTransfer]]:
    """Return the scripted movements, one list of transfers per cycle.

    With the 1 USDC activation deposit and a ``5 * unit - 1`` USDC
    subscription, the master holds ``5 * unit`` after the first step and every
    later movement is at least ``unit``, so ``unit = 1`` runs the schedule at
    Lighter's 1 USDC minimum. A transfer fee reduces the balances, so give
    ``unit`` enough margin above the fee when one is charged.

    1. Deposit all idle Safe cash to the master.
    2. Fund both sub-accounts from the master with ``2 * unit`` each.
    3. Move ``unit`` from sub-account 1 to sub-account 2, signed by sub-account 1.
    4. Move ``unit`` from sub-account 2 back to the master, signed by sub-account 2.
    5. Withdraw everything from both sub-accounts and the master to the Safe.

    :param unit:
        Base amount of the fixed movements.
    :return:
        Schedule steps in execution order.
    """
    if unit <= 0:
        raise ValueError("The schedule unit must be positive")
    sub_1, sub_2 = get_sub_account_role(1), get_sub_account_role(2)
    return [
        [ScriptedTransfer(ROLE_SAFE, ROLE_MASTER, None)],
        [ScriptedTransfer(ROLE_MASTER, sub_1, 2 * unit), ScriptedTransfer(ROLE_MASTER, sub_2, 2 * unit)],
        [ScriptedTransfer(sub_1, sub_2, unit)],
        [ScriptedTransfer(sub_2, ROLE_MASTER, unit)],
        [ScriptedTransfer(sub_1, ROLE_SAFE, None), ScriptedTransfer(sub_2, ROLE_SAFE, None), ScriptedTransfer(ROLE_MASTER, ROLE_SAFE, None)],
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


def get_role_pairs(pairs: list[TradingPairIdentifier]) -> dict[str, TradingPairIdentifier]:
    """Map account roles to the universe's Lighter pairs.

    :param pairs:
        Strategy universe pairs.
    :return:
        ``master`` and ``sub_<n>`` in universe order, which is the deployment order.
    """
    master_pair, sub_pairs = get_lighter_account_pairs(pairs)
    return {ROLE_MASTER: master_pair, **{get_sub_account_role(number): pair for number, pair in enumerate(sub_pairs, 1)}}


def decide_lighter_scripted_transfers(
    input: StrategyInput,
    schedule: list[list[ScriptedTransfer]],
    minimum: Decimal,
) -> list[TradeExecution]:
    """Emit the next schedule step as custody trades.

    One step runs per cycle. A cycle with a pending transfer, a completed
    schedule or a failed step emits nothing.

    :param input:
        Strategy cycle input.
    :param schedule:
        Movements from :py:func:`build_lighter_transfer_schedule`.
    :param minimum:
        Smallest movement Lighter accepts; a smaller scripted amount is an error.
    :return:
        This cycle's trades.
    """
    if not ensure_lighter_positions(input):
        return []
    state = input.state
    portfolio = state.portfolio
    if any(trade.is_exchange_account_transfer_pending() for position in portfolio.get_open_and_frozen_positions() for trade in position.trades.values()):
        return []
    completed, problem = get_scripted_progress(state, len(schedule))
    if problem is not None:
        logger.error("Lighter scripted transfers stopped: %s", problem)
        return []
    if completed >= len(schedule):
        logger.info("Lighter scripted transfers completed all %d steps", len(schedule))
        return []

    step = completed + 1
    role_pairs = get_role_pairs(list(input.strategy_universe.iterate_pairs()))
    positions = {role: portfolio.get_open_position_for_pair(pair) for role, pair in role_pairs.items()}
    position_manager = input.get_position_manager()
    reserve_asset = input.strategy_universe.get_reserve_asset()
    timestamp = input.timestamp.to_pydatetime()

    trades = []
    for transfer in schedule[step - 1]:
        metadata: dict[str, str | int] = {
            "protocol": LIGHTER_PROTOCOL,
            SCRIPTED_STEP_KEY: step,
            SCRIPTED_SOURCE_KEY: transfer.source,
            SCRIPTED_DESTINATION_KEY: transfer.destination,
        }
        notes = f"Scripted Lighter transfer step {step}: {transfer.source} to {transfer.destination}"
        if transfer.source == ROLE_SAFE:
            amount = transfer.amount if transfer.amount is not None else Decimal(str(position_manager.get_current_cash()))
            amount = _check_amount(amount, minimum, transfer)
            trades.append(create_exchange_account_transfer(state=state, position=positions[transfer.destination], strategy_cycle_at=timestamp, reserve_currency=reserve_asset, amount=amount, deposit=True, notes=notes, metadata={**metadata, "direction": "deposit"}))
        elif transfer.destination == ROLE_SAFE:
            amount = transfer.amount if transfer.amount is not None else Decimal(str(position_manager.get_exchange_account_available_balance(role_pairs[transfer.source])))
            amount = _check_amount(amount, minimum, transfer)
            trades.append(create_exchange_account_transfer(state=state, position=positions[transfer.source], strategy_cycle_at=timestamp, reserve_currency=reserve_asset, amount=amount, deposit=False, notes=notes, metadata={**metadata, "direction": "withdraw"}))
        else:
            if transfer.amount is None:
                raise ValueError(f"A transfer between accounts needs a fixed amount: {transfer}")
            amount = _check_amount(transfer.amount, minimum, transfer)
            trades.append(create_exchange_account_funding(state=state, source_position=positions[transfer.source], destination_position=positions[transfer.destination], strategy_cycle_at=timestamp, reserve_currency=reserve_asset, amount=amount, notes=notes, metadata=metadata))
    logger.info("Lighter scripted transfers step %d of %d: %d trade(s)", step, len(schedule), len(trades))
    return trades


def _check_amount(amount: Decimal, minimum: Decimal, transfer: ScriptedTransfer) -> Decimal:
    """Round an amount down to USDC units and reject one below Lighter's minimum."""
    amount = amount.quantize(USDC_QUANTUM, rounding=ROUND_DOWN)
    if amount < minimum:
        raise ValueError(f"Scripted transfer {transfer.source} to {transfer.destination} of {amount} USDC is below the {minimum} USDC minimum")
    return amount


def verify_lighter_scripted_transfers(
    state: State,
    schedule: list[list[ScriptedTransfer]],
    account_indices: list[int],
) -> list[dict[str, Any]]:
    """Check that every scheduled movement was booked as one successful trade between the right positions.

    Checks, per movement: exactly one trade carries its step and roles; the
    trade succeeded; it is a deposit, withdrawal or funding on the position
    of the right account; a fixed amount was moved exactly; and a funding's
    source position carries balance updates for the amount and the fee.
    Then checks that no transfer is pending, that every Lighter position is
    empty, and that the Lighter inflows, revaluations, outflows and fees add
    up.

    :param state:
        Executor state after the vault has been wound up.
    :param schedule:
        Movements the strategy was given.
    :param account_indices:
        Master account index followed by the sub-account indices, in the
        order given to the strategy.
    :return:
        One row per movement for reporting.
    :raise AssertionError:
        With every problem found, if any.
    """
    roles = {ROLE_MASTER: account_indices[0], **{get_sub_account_role(number): index for number, index in enumerate(account_indices[1:], 1)}}
    positions = [position for position in state.portfolio.get_all_positions() if position.is_exchange_account()]
    by_step = get_scripted_trades(state)
    problems: list[str] = []
    rows: list[dict[str, Any]] = []

    for step, transfers in enumerate(schedule, 1):
        trades = by_step.get(step, [])
        if len(trades) != len(transfers):
            problems.append(f"step {step}: expected {len(transfers)} trades, found {len(trades)}")
        for transfer in transfers:
            matching = [trade for trade in trades if trade.other_data.get(SCRIPTED_SOURCE_KEY) == transfer.source and trade.other_data.get(SCRIPTED_DESTINATION_KEY) == transfer.destination]
            if len(matching) != 1:
                problems.append(f"step {step}: expected one trade {transfer.source} to {transfer.destination}, found {len(matching)}")
                continue
            trade = matching[0]
            row = _verify_transfer(state, trade, transfer, roles, problems)
            row["step"] = step
            rows.append(row)

    pending = [trade.trade_id for trade in state.portfolio.get_all_trades() if trade.is_exchange_account_transfer_pending()]
    if pending:
        problems.append(f"pending transfers remain: {pending}")
    for position in positions:
        if position.get_quantity() != 0:
            problems.append(f"Lighter account {position.pair.get_exchange_account_id()} position #{position.position_id} still holds {position.get_quantity()} USDC")

    # Lighter inflows and outflows: deposits add, withdrawals and transfer fees
    # remove, and transfers between accounts net to zero. Balance updates not
    # booked by a funding trade are revaluations from Lighter, such as the
    # activation deposit made before the executor existed.
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

    if problems:
        raise AssertionError("Scripted Lighter transfers were not booked correctly:\n" + "\n".join(problems))
    state.check_if_clean()
    return rows


def _verify_transfer(
    state: State,
    trade: TradeExecution,
    transfer: ScriptedTransfer,
    roles: dict[str, int],
    problems: list[str],
) -> dict[str, Any]:
    """Check one scripted trade against its movement and return its report row."""
    label = f"trade #{trade.trade_id} {transfer.source} to {transfer.destination}"
    flags = trade.flags or set()
    amount = abs(trade.executed_quantity or Decimal(0))
    fee = Decimal(0)
    account = int(trade.pair.get_exchange_account_id())
    if not trade.is_success():
        problems.append(f"{label} did not succeed: {trade.get_status().value}")
    if transfer.source == ROLE_SAFE:
        if TradeFlag.external_account_transfer not in flags or not trade.is_buy():
            problems.append(f"{label} is not an external-account deposit")
        if account != roles[transfer.destination]:
            problems.append(f"{label} credits account {account}, expected {roles[transfer.destination]}")
        kind = "deposit"
    elif transfer.destination == ROLE_SAFE:
        if TradeFlag.external_account_transfer not in flags or trade.is_buy():
            problems.append(f"{label} is not an external-account withdrawal")
        if account != roles[transfer.source]:
            problems.append(f"{label} debits account {account}, expected {roles[transfer.source]}")
        if Decimal(trade.other_data.get("lighter_claimable_usdc", "0")) != amount:
            problems.append(f"{label} claimed {trade.other_data.get('lighter_claimable_usdc')} USDC, expected {amount}")
        kind = "withdrawal"
    else:
        if not trade.is_internal_account_transfer():
            problems.append(f"{label} is not an internal-account funding")
        if account != roles[transfer.destination]:
            problems.append(f"{label} credits account {account}, expected {roles[transfer.destination]}")
        source_index = int(trade.other_data.get("funding_source_account_id", -1))
        if source_index != roles[transfer.source]:
            problems.append(f"{label} is funded by account {source_index}, expected {roles[transfer.source]}")
        fee = Decimal(trade.other_data.get("funding_fee_usdc", "0"))
        source_position = state.portfolio.get_position_by_id(int(trade.other_data["funding_source_position_id"]))
        update_ids = trade.other_data.get("funding_source_balance_update_ids", [])
        if not update_ids:
            problems.append(f"{label} has no source balance updates")
        else:
            booked = sum((source_position.balance_updates[update_id].quantity for update_id in update_ids if update_id in source_position.balance_updates), Decimal(0))
            if booked != -(amount + fee):
                problems.append(f"{label} booked {booked} on the source, expected {-(amount + fee)}")
        kind = "funding"
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
