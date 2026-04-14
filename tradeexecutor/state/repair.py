"""Maanually repair broken states.

- Recover from failed trades

Trade failure modes may include

- Broadcasted but not confirmed

- Broadcasted, confirmed, but not marked as executed

- Executed, failed

Failure trades may be

- Buy e.g. first trade failed: open position -> closed position, allocated capital returned

- Sell e.g. closing trade failed: position stays open, the assets are marked to be available
  for the future sell


"""
import datetime
import logging
from dataclasses import dataclass
from decimal import Decimal
from itertools import chain
from typing import List, TypedDict

from eth_defi.compat import native_datetime_utc_now

from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeType, TradeStatus, TradeFlag
from eth_defi.compat import native_datetime_utc_now

logger = logging.getLogger(__name__)


class RepairAborted(Exception):
    """User chose no"""


class HypercoreDuplicateCloseError(Exception):
    """A Hypercore duplicate group was not safe to close automatically."""


@dataclass(slots=True)
class RepairResult:
    """The report of the repair results.

    Note that repair might not have done anything - every list is empty.
    """

    #: How many frozen positions we encountered
    frozen_positions: List[TradingPosition]

    #: What positions we managed to unfreeze
    unfrozen_positions: List[TradingPosition]

    #: How many individual trades we repaired
    trades_needing_repair: List[TradeExecution]

    #: New trades we made to fix the accounting
    new_trades: List[TradeExecution]


@dataclass(slots=True)
class HypercoreDuplicateCloseCandidate:
    """One safe Hypercore duplicate-clone close candidate."""

    vault_address: str
    vault_name: str
    survivor_position: TradingPosition
    clone_position: TradingPosition
    close_reason: str


def _get_hypercore_vault_address(position: TradingPosition) -> str:
    """Get the canonical vault address for a Hypercore position."""

    return (position.pair.pool_address or position.pair.base.address).lower()


def _get_position_expected_usd_equity(position: TradingPosition) -> Decimal:
    """Get the expected USD equity using the position valuation semantics."""

    value = position.get_value(include_interest=False)
    return Decimal(str(value))


def _is_close_enough(left: Decimal, right: Decimal, tolerance: Decimal = Decimal("0.000001")) -> bool:
    """Check whether two Decimal values are close enough for duplicate repair."""

    return abs(left - right) <= tolerance


def _has_repaired_close_trade(position: TradingPosition) -> bool:
    """Check whether a position has a repaired closing trade in its history."""

    return any(
        trade.is_repaired() and trade.is_sell()
        for trade in position.trades.values()
    )


def _validate_hypercore_duplicate_clone_group(
    positions: list[TradingPosition],
) -> HypercoreDuplicateCloseCandidate:
    """Validate that a duplicate group is safe to close automatically."""

    if len(positions) != 2:
        raise HypercoreDuplicateCloseError(
            f"Expected exactly 2 duplicate positions, got {len(positions)}"
        )

    older_position, newer_position = sorted(positions, key=lambda p: p.position_id)

    base_reasons: list[str] = []
    for position in (older_position, newer_position):
        if position.is_frozen():
            base_reasons.append("group contains frozen positions")
        if position.has_planned_trades():
            base_reasons.append("group contains planned trades")
        if position.is_about_to_close():
            base_reasons.append("group contains positions about to close")
        if position.loan is not None:
            base_reasons.append("group contains loan-backed state")
        if position.is_loan_based():
            base_reasons.append("group contains loan-based positions")

    older_trade = older_position.get_first_trade()
    newer_trade = newer_position.get_first_trade()

    if older_position.pair.pool_address != newer_position.pair.pool_address:
        base_reasons.append("pool addresses differ")
    if older_position.pair.base != newer_position.pair.base:
        base_reasons.append("base assets differ")
    if older_position.pair.quote != newer_position.pair.quote:
        base_reasons.append("quote assets differ")
    if older_trade.reserve_currency != newer_trade.reserve_currency:
        base_reasons.append("reserve currencies differ")

    if older_position.get_quantity(planned=False) != newer_position.get_quantity(planned=False):
        base_reasons.append("current quantities differ")
    if older_position.get_quantity(planned=True) != newer_position.get_quantity(planned=True):
        base_reasons.append("planned quantities differ")
    if older_position.last_token_price != newer_position.last_token_price:
        base_reasons.append("last_token_price differs")
    if older_position.last_reserve_price != newer_position.last_reserve_price:
        base_reasons.append("last_reserve_price differs")
    if not _is_close_enough(
        _get_position_expected_usd_equity(older_position),
        _get_position_expected_usd_equity(newer_position),
    ):
        base_reasons.append("expected USD equity differs")

    if base_reasons:
        raise HypercoreDuplicateCloseError(
            f"Vault {_get_hypercore_vault_address(older_position)} "
            f"positions #{older_position.position_id} and #{newer_position.position_id} "
            f"are not safe to close automatically: {', '.join(dict.fromkeys(base_reasons))}"
        )

    older_has_repaired_close = _has_repaired_close_trade(older_position)
    newer_has_repaired_close = _has_repaired_close_trade(newer_position)

    # Case 1: a later phantom clone opened alongside the older canonical position.
    clone_reasons: list[str] = []
    if len(newer_position.trades) != 1:
        clone_reasons.append("clone candidate does not have exactly one trade")
    else:
        clone_trade = newer_position.get_first_trade()
        clone_flags = clone_trade.flags or set()
        if not clone_trade.is_success():
            clone_reasons.append("clone candidate opening trade is not successful")
        if clone_trade.is_sell():
            clone_reasons.append("clone candidate trade is a sell")
        if clone_trade.is_failed():
            clone_reasons.append("clone candidate trade is failed")
        if clone_trade.is_repair_trade() or clone_trade.trade_type == TradeType.repair:
            clone_reasons.append("clone candidate trade is a repair trade")
        if TradeFlag.ignore_open not in clone_flags:
            clone_reasons.append("clone candidate opening trade lacks ignore_open flag")

    if not newer_position.balance_updates:
        pass
    elif len(newer_position.balance_updates) == 1:
        update = next(iter(newer_position.balance_updates.values()))
        if getattr(update.cause, "value", update.cause) != "vault_flow":
            clone_reasons.append("clone candidate has non-vault-flow balance updates")
    else:
        clone_reasons.append("clone candidate has multiple balance updates")

    if not clone_reasons and not older_has_repaired_close and not newer_has_repaired_close:
        return HypercoreDuplicateCloseCandidate(
            vault_address=_get_hypercore_vault_address(older_position),
            vault_name=older_position.pair.get_vault_name() or older_position.pair.get_ticker(),
            survivor_position=older_position,
            clone_position=newer_position,
            close_reason="later phantom duplicate clone",
        )

    # Case 2: an older stale repaired-close position remained open and a newer reopen became the live position.
    reopen_reasons: list[str] = []
    if not older_has_repaired_close:
        reopen_reasons.append("older position does not contain a repaired close trade")
    if newer_has_repaired_close:
        reopen_reasons.append("newer position unexpectedly contains a repaired close trade")
    if len(newer_position.trades) != 1:
        reopen_reasons.append("newer survivor candidate does not have exactly one trade")
    else:
        reopen_trade = newer_position.get_first_trade()
        if not reopen_trade.is_success():
            reopen_reasons.append("newer survivor candidate opening trade is not successful")
        if reopen_trade.is_sell():
            reopen_reasons.append("newer survivor candidate trade is a sell")
        if reopen_trade.is_failed():
            reopen_reasons.append("newer survivor candidate trade is failed")
        if reopen_trade.is_repair_trade() or reopen_trade.trade_type == TradeType.repair:
            reopen_reasons.append("newer survivor candidate trade is a repair trade")

    if len(newer_position.balance_updates) != 1:
        reopen_reasons.append("newer survivor candidate does not have exactly one balance update")
    else:
        update = next(iter(newer_position.balance_updates.values()))
        if getattr(update.cause, "value", update.cause) != "vault_flow":
            reopen_reasons.append("newer survivor candidate balance update is not vault_flow")

    repaired_close_trade = next(
        (trade for trade in older_position.trades.values() if trade.is_repaired() and trade.is_sell()),
        None,
    )
    if repaired_close_trade is None:
        reopen_reasons.append("older position repaired close trade could not be located")
    else:
        if newer_trade.opened_at is not None and repaired_close_trade.executed_at is not None:
            if newer_trade.opened_at <= repaired_close_trade.executed_at:
                reopen_reasons.append("newer survivor candidate was not opened after the repaired close")

    if not reopen_reasons:
        return HypercoreDuplicateCloseCandidate(
            vault_address=_get_hypercore_vault_address(older_position),
            vault_name=older_position.pair.get_vault_name() or older_position.pair.get_ticker(),
            survivor_position=newer_position,
            clone_position=older_position,
            close_reason="stale repaired-close duplicate after later reopen",
        )

    reasons = clone_reasons or reopen_reasons
    raise HypercoreDuplicateCloseError(
        f"Vault {_get_hypercore_vault_address(older_position)} "
        f"positions #{older_position.position_id} and #{newer_position.position_id} "
        f"are not safe to close automatically: {', '.join(dict.fromkeys(reasons))}"
    )


def find_hypercore_duplicate_close_candidates(
    portfolio: Portfolio,
) -> tuple[list[HypercoreDuplicateCloseCandidate], list[str]]:
    """Find Hypercore duplicate groups that are safe to close as later clones."""

    positions_by_vault: dict[str, list[TradingPosition]] = {}
    for position in portfolio.get_open_and_frozen_positions():
        if not position.pair.is_hyperliquid_vault():
            continue
        positions_by_vault.setdefault(_get_hypercore_vault_address(position), []).append(position)

    candidates: list[HypercoreDuplicateCloseCandidate] = []
    rejected_groups: list[str] = []

    for positions in positions_by_vault.values():
        if len(positions) < 2:
            continue
        try:
            candidates.append(_validate_hypercore_duplicate_clone_group(positions))
        except HypercoreDuplicateCloseError as e:
            rejected_groups.append(str(e))

    return candidates, rejected_groups


def close_hypercore_duplicate_clone(
    portfolio: Portfolio,
    candidate: HypercoreDuplicateCloseCandidate,
    now: datetime.datetime | None = None,
) -> TradeExecution:
    """Close a later Hypercore duplicate clone with a flagged repair trade."""

    now = now or native_datetime_utc_now()
    survivor_position = candidate.survivor_position
    clone_position = candidate.clone_position
    opening_trade = clone_position.get_first_trade()
    note = (
        f"Closed as duplicate Hypercore clone of position #{survivor_position.position_id} "
        f"because {candidate.close_reason} ({now.isoformat()})"
    )

    position, counter_trade, created = portfolio.create_trade(
        strategy_cycle_at=opening_trade.strategy_cycle_at,
        pair=clone_position.pair,
        quantity=Decimal(0),
        assumed_price=opening_trade.planned_price,
        trade_type=TradeType.repair,
        reserve_currency=opening_trade.reserve_currency,
        planned_mid_price=opening_trade.planned_mid_price,
        price_structure=opening_trade.price_structure,
        reserve=None,
        reserve_currency_price=opening_trade.get_reserve_currency_exchange_rate(),
        position=clone_position,
    )
    counter_trade.started_at = now
    assert created is False
    assert position == clone_position

    counter_trade.flags = (counter_trade.flags or set()) | {
        TradeFlag.close,
        TradeFlag.hypercore_duplicate_close,
    }
    counter_trade.mark_success(
        now,
        opening_trade.planned_price,
        Decimal(0),
        Decimal(0),
        0,
        opening_trade.native_token_price,
        force=True,
    )
    assert counter_trade.is_success()

    counter_trade.repaired_trade_id = opening_trade.trade_id
    opening_trade.add_note(
        f"Closed duplicate Hypercore clone at {now.strftime('%Y-%m-%d %H:%M')}, by #{counter_trade.trade_id}"
    )
    counter_trade.add_note(note)
    clone_position.add_notes_message(note)
    survivor_position.add_notes_message(
        f"Kept surviving Hypercore position while closing duplicate clone position "
        f"#{clone_position.position_id} because {candidate.close_reason} ({now.isoformat()})"
    )
    clone_position.other_data["closed_duplicate_survivor_position_id"] = survivor_position.position_id
    clone_position.other_data["hypercore_duplicate_close_reason"] = candidate.close_reason
    portfolio.close_position(clone_position, now)
    logger.info(
        "Closed Hypercore duplicate clone position #%d with repair trade #%d and kept survivor #%d for vault %s at %s",
        clone_position.position_id,
        counter_trade.trade_id,
        survivor_position.position_id,
        candidate.vault_name,
        candidate.vault_address,
    )
    return counter_trade


def make_counter_trade(portfolio: Portfolio, p: TradingPosition, t: TradeExecution) -> TradeExecution:
    """Make a virtual trade that fixes the total balances of a position and unwinds the broken trade."""

    # Note: we do not negate the values of the original trade,
    # because get_quantity() and others will return 0 to repaired spot trades for now.
    # This behavior may change in the future for more complex trades.
    position, counter_trade, created = portfolio.create_trade(
        strategy_cycle_at=t.strategy_cycle_at,
        pair=t.pair,
        quantity=-t.planned_quantity,
        assumed_price=t.planned_price,
        trade_type=TradeType.repair,
        reserve_currency=t.reserve_currency,
        planned_mid_price=t.planned_mid_price,
        price_structure=t.price_structure,
        reserve=None,
        reserve_currency_price=t.get_reserve_currency_exchange_rate(),
        position=p,
    )
    counter_trade.started_at = native_datetime_utc_now()
    assert created is False
    assert position == p

    counter_trade.mark_success(
        native_datetime_utc_now(),
        t.planned_price,
        Decimal(0),
        Decimal(0),
        0,
        t.native_token_price,
        force=True,
    )
    assert counter_trade.is_success()
    assert counter_trade.get_value() == 0
    assert counter_trade.get_position_quantity() == 0
    assert counter_trade.trade_type == TradeType.repair
    return counter_trade


def repair_trade(portfolio: Portfolio, t: TradeExecution) -> TradeExecution:
    """Repair a trade.

    - Make a counter trade for bookkeeping

    - Set the original trade to repaired state (instead of failed state)
    """
    p = portfolio.get_position_by_id(t.position_id)

    c = make_counter_trade(portfolio, p, t)
    now = native_datetime_utc_now()
    t.repaired_at = t.executed_at = native_datetime_utc_now()
    t.executed_quantity = 0
    t.executed_reserve = 0
    assert c.trade_id
    c.repaired_trade_id = t.trade_id
    t.add_note(f"Repaired at {now.strftime('%Y-%m-%d %H:%M')}, by #{c.trade_id}")
    c.add_note(f"Repairing trade #{c.repaired_trade_id}")
    assert t.get_status() == TradeStatus.repaired
    assert t.get_value() == 0
    assert t.get_position_quantity() == 0
    assert t.planned_quantity != 0

    # Unwind capital allocation
    if t.is_buy():
        if not t.is_credit_supply():
            # only need to adjust reserve for spot trades
            portfolio.adjust_reserves(
                t.reserve_currency,
                +t.planned_reserve,
                f"Repairing position {p}",
            )
        t.planned_reserve = 0

    return c


def repair_tx_missing(portfolio: Portfolio, t: TradeExecution) -> TradeExecution:
    """Repair a trade which failed to generate new transactions..

    - Make a counter trade for bookkeeping

    - Set the original trade to repaired state (instead of planned state)
    """
    p = portfolio.get_position_by_id(t.position_id)

    c = make_counter_trade(portfolio, p, t)
    now = native_datetime_utc_now()
    t.repaired_at = t.executed_at = native_datetime_utc_now()
    t.executed_quantity = 0
    t.executed_reserve = 0
    assert c.trade_id
    c.repaired_trade_id = t.trade_id
    t.add_note(f"Repaired at {now.strftime('%Y-%m-%d %H:%M')}, by #{c.trade_id}")
    c.add_note(f"Repairing trade #{c.repaired_trade_id}")
    assert t.get_status() == TradeStatus.repaired
    assert t.get_value() == 0
    assert t.get_position_quantity() == 0
    assert t.planned_quantity != 0
    return c


def close_position_with_empty_trade(portfolio: Portfolio, p: TradingPosition) -> TradeExecution:
    """Make a trade that closes the position.

    - Closes an open position that has lost it tokens,
      in accounting correction

    - This trade has size of 0 and pricing data from the opening trade

    - :py:attr:`TradeExecution.repaired_trade_id` is set for this trade to be the opening trade

    - We assume closed positions must have at least 2 trades,
      so this function will generate the final trade and now
      the position has at least opening trade + this trade
      (TODO: This assumption should be changed)

    """

    assert p.pair.is_spot() or p.pair.is_credit_supply() or p.pair.is_vault() or p.pair.is_cctp_bridge(), f"Only spot / vault / credit / cctp_bridge position supported for now"

    # TODO: Cannot honour this in some cases?
    # assert len(p.trades) == 1, f"Can only fix one failed trade, got: {p}:\n{p.trades}"

    opening_trade = p.get_first_trade()

    assert opening_trade.is_success(), f"Cannot make a repairing trade, because opening trade {t} was not success"

    # We copy any price structure from opening trade, though it should be meaningfull
    position, counter_trade, created = portfolio.create_trade(
        strategy_cycle_at=opening_trade.strategy_cycle_at,
        pair=p.pair,
        quantity=Decimal(0),
        assumed_price=opening_trade.planned_price,
        trade_type=TradeType.repair,
        reserve_currency=opening_trade.reserve_currency,
        planned_mid_price=opening_trade.planned_mid_price,
        price_structure=opening_trade.price_structure,
        reserve=None,
        reserve_currency_price=opening_trade.get_reserve_currency_exchange_rate(),
        position=p,
    )
    counter_trade.started_at = native_datetime_utc_now()
    assert created is False
    assert position == p

    counter_trade.mark_success(
        native_datetime_utc_now(),
        opening_trade.planned_price,
        Decimal(0),
        Decimal(0),
        0,
        opening_trade.native_token_price,
        force=True,
    )
    assert counter_trade.is_success()
    assert counter_trade.get_value() == 0
    assert counter_trade.get_position_quantity() == 0
    assert counter_trade.trade_type == TradeType.repair

    c = counter_trade

    now = native_datetime_utc_now()

    assert c.trade_id
    c.repaired_trade_id = opening_trade.trade_id
    opening_trade.add_note(f"Repaired at {now.strftime('%Y-%m-%d %H:%M')}, by #{c.trade_id}")
    c.add_note(f"Repairing to close the position, full position size gone missing")

    portfolio.close_position(position, native_datetime_utc_now())

    # The position is now cleared
    assert p.is_closed()
    assert not p.can_be_closed()
    assert p.position_id in portfolio.closed_positions

    return c


def close_hypercore_dust_positions(
    portfolio: Portfolio,
    now: datetime.datetime | None = None,
) -> List[TradeExecution]:
    """Close Hypercore vault dust positions with repair trades.

    Hypercore full withdrawals intentionally leave a small residual balance
    because the protocol rejects exact full redemptions when vault NAV drifts
    between quote and execution.  These dust leftovers should not stay open in
    the state because later cycles may try to trade around them as if they were
    still meaningful live positions.

    1. Scan open and frozen positions for Hypercore vault entries.
    2. Identify positions that are already within the close epsilon.
    3. Close each dust position locally with a zero-quantity repair trade.

    :return:
        Repair trades created for the auto-closed dust positions.
    """

    now = now or native_datetime_utc_now()
    created_trades: List[TradeExecution] = []

    positions = list(chain(
        portfolio.open_positions.values(),
        portfolio.frozen_positions.values(),
    ))

    for position in positions:
        if not position.pair.is_hyperliquid_vault():
            continue

        if not position.can_be_closed():
            continue

        if position.is_frozen():
            del portfolio.frozen_positions[position.position_id]
            portfolio.open_positions[position.position_id] = position
            position.unfrozen_at = now

        trade = close_position_with_empty_trade(portfolio, position)
        position.add_notes_message(
            "Auto-closed Hypercore dust position because Hypercore vault "
            f"withdrawals cannot fully redeem the final residual balance ({now.isoformat()})"
        )
        logger.info(
            "Auto-closed Hypercore dust position %s with repair trade %s",
            position,
            trade,
        )
        created_trades.append(trade)

    return created_trades


def find_trades_to_be_repaired(state: State) -> List[TradeExecution]:
    trades_to_be_repaired = []
    # Closed trades do not need attention
    for p in chain(state.portfolio.open_positions.values(), state.portfolio.frozen_positions.values()):
        t: TradeExecution
        for t in p.trades.values():
            if t.is_repair_needed():
                logger.info("Found a trade needing repair: %s", t)
                if t.is_short():
                    logger.error("Failed short trade can't be repaired using this command yet")
                else:  
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

        t.repaired_at = native_datetime_utc_now()
        t.add_note(f"Failed broadcast repaired at {t.repaired_at}")

        repaired.append(t)


def unfreeze_position(portfolio: Portfolio, position: TradingPosition) -> bool:
    """Attempt to unfreeze positions.

    - All failed trades on a position must be cleared

    :return:
        if we managed to unfreeze the position
    """

    # Double check trade status look good and we have no longer failed trades
    trades = list(position.trades.values())
    assert all([t.is_success() for t in trades]), f"All trades where not successful: {trades}"
    assert all([not t.is_failed() for t in trades]), f"Some trades were still failed: {trades}"
    assert any([t.is_repaired() for t in trades])

    # Based on if the last failing trade was open or close,
    # the position should ended up in open or closed
    total_equity = position.get_quantity_old()
    if total_equity > 0:
        portfolio.open_positions[position.position_id] = position
    elif total_equity == 0:
        assert position.can_be_closed()
        portfolio.closed_positions[position.position_id] = position
        position.closed_at = native_datetime_utc_now()
    else:
        raise RuntimeError("Not gonna happen")

    position.unfrozen_at = native_datetime_utc_now()
    del portfolio.frozen_positions[position.position_id]

    if position.notes is None:
        position.notes = ""

    position.add_notes_message(f"Unfrozen at {native_datetime_utc_now()}")

    return True


def repair_trades(
        state: State,
        attempt_repair=True,
        interactive=True) -> RepairResult:
    """Repair trade.

    - Find frozen positions and trades in them

    - Mark trades invalidated

    - Make the necessary counter trades to fix the total balances

    - Does not actually broadcast any transactions - only fixes the internal accounting

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

    trades_to_be_repaired = find_trades_to_be_repaired(state)

    logger.info("Found %d trades to be repaired", len(trades_to_be_repaired))

    if len(trades_to_be_repaired) == 0 or not attempt_repair:
        return RepairResult(
            frozen_positions,
            [],
            trades_to_be_repaired,
            [],
        )

    if interactive:

        for t in trades_to_be_repaired:
            print("Needs repair:", t)

        confirmation = input("Attempt to repair [y/n]").lower()
        if confirmation != "y":
            raise RepairAborted()

    new_trades = []
    for t in trades_to_be_repaired:
        new_trades.append(repair_trade(state.portfolio, t))

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
        trades_to_be_repaired,
        new_trades,
    )


def repair_tx_not_generated(state: State, interactive=True):
    """Repair command to fix trades that did not generate tranasctions.

    - Reasons include

    - Currently only manually callable from console

    - Simple deletes trades that have an empty transaction list

    Example exception:

    .. code-block:: text

          File "/usr/src/trade-executor/tradeexecutor/ethereum/routing_model.py", line 395, in trade
            return self.make_direct_trade(
          File "/usr/src/trade-executor/tradeexecutor/ethereum/uniswap_v3/uniswap_v3_routing.py", line 257, in make_direct_trade
            return super().make_direct_trade(
          File "/usr/src/trade-executor/tradeexecutor/ethereum/routing_model.py", line 112, in make_direct_trade
            adjusted_reserve_amount = routing_state.adjust_spend(
          File "/usr/src/trade-executor/tradeexecutor/ethereum/routing_state.py", line 283, in adjust_spend
            raise OutOfBalance(
        tradeexecutor.ethereum.routing_state.OutOfBalance: Not enough tokens for <USDC at 0x2791bca1f2de4661ed88a30c99a7a9449aa84174> to perform the trade. Required: 3032399763, on-chain balance for 0x375A8Cd0A654E0eCa46F81c1E5eA5200CC6A737C is 87731979.

    :param interactive:
        Use console interactive prompts to ask the user to confirm the repair

    :return:
        Repair trades generated.

    :raise RepairAborted:
        Interactive operation was aborted by the user
    """

    tx_missing_trades = set()
    portfolio = state.portfolio

    for t in portfolio.get_all_trades():

        if t.repaired_trade_id:
            # This is an accounting repair for some other trade
            continue

        if t.get_status() == TradeStatus.repaired:
            # Already repaired
            continue

        if t.get_status() == TradeStatus.success:
            # Already repaired
            continue

        if not t.blockchain_transactions:
            assert t.get_status() in (TradeStatus.planned, TradeStatus.started), f"Trade missing tx, but status is not planned/repaired {t}"
            tx_missing_trades.add(t)

    if not tx_missing_trades:
        if interactive:
            print("No trades with missing blockchain transactions detected")
        return []

    if interactive:

        print("Trade missing TX report")
        print("-" * 80)

        print("Trade to repair:")
        for t in tx_missing_trades:
            print(t)

        confirm = input("Confirm repair with counter trades [y/n]? ")
        if confirm.lower() != "y":
            raise RepairAborted()
    else:
        print("Auto-approve is active, repairing trades")

    repair_trades_generated = [repair_tx_missing(portfolio, t) for t in tx_missing_trades]
    
    print("Counter-trades:")
    for t in repair_trades_generated:
        position = portfolio.get_position_by_id(t.position_id)
        print("Position ", position)
        print("Trade that was repaired ", portfolio.get_trade_by_id(t.repaired_trade_id))
        print("Repair trade ", t)
        print("-")

    if interactive:
        confirm = input("Looks fixed [y/n]? ")
        if confirm.lower() != "y":
            raise RepairAborted()

    return repair_trades_generated


def repair_zero_quantity(state: State, interactive=True):
    """Scan for positions that failed to open and need to be cleaned up."""
    zero_quantity_positions = [p for p in state.portfolio.get_open_positions() if p.get_quantity() == 0]
    if not zero_quantity_positions:
        print("No zero quantity positions found")
        return

    print("Positions with zero quantity that need to be fixed")
    for p in zero_quantity_positions:
        print("   ", p)

    if interactive:
        confirm = input("Fix [y/n]? ")
        if confirm.lower() != "y":
            raise RepairAborted()

    portfolio = state.portfolio
    repair_trades_generated = []
    for p in zero_quantity_positions:
        # The position has gone to zero
        if p.can_be_closed():
            # In a lot of places we assume that a position with 1 trade cannot be closed
            # Make a 0-sized trade so that we know the position is closed
            t = close_position_with_empty_trade(portfolio, p)
            logger.info("Position %s closed with a trade %s", p, t)
            assert p.is_closed()

            repair_trades_generated.append(t)

    return repair_trades_generated
