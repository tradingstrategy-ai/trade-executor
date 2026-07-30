"""Clean undersized tracked HyperCore vault positions.

This module is deliberately limited to **HyperCore native vault** positions.
It must not be used for ERC-4626, ERC-7540, or Ostium vault positions: those
protocols have different redemption and settlement semantics.  The current
caller uses Lagoon's HyperCore execution route, but this module never handles
Lagoon vault shares.

HyperCore ``vaultTransfer`` has no withdraw-all operation and silently no-ops
for an amount that is even slightly above live equity. A small position is
therefore redeemed through the normal HyperCore router, with a full-close flag
and progressively larger retry margins if the first ``vaultTransfer`` silently
no-ops. ``MINIMUM_VAULT_DEPOSIT`` is enforced by the deposit encoder; the
withdrawal encoder has no equivalent client-side check. The cleaner therefore
never tops up a position it is trying to close. Settlement checks detect a
backend no-op. Cleanup uses an adaptive 0.10 USDC initial margin instead of the
normal 1.50 USDC close margin, so almost all of a small position is recovered.
If too little remains to verify all withdrawal phases, the position is closed
locally as protocol dust. If a routed trade fails, the updated state is saved
and the normal failed-trade repair flow takes over.
"""

import copy
import datetime
import logging
from dataclasses import dataclass, replace
from decimal import Decimal

from eth_defi.hyperliquid.api import fetch_user_vault_equity
from tradeexecutor.ethereum.vault.hypercore_routing import (
    HYPERCORE_SMALL_POSITION_CLEANUP_MIN_REDEMPTION_RAW,
    HypercoreWithdrawalPreflightError,
    raw_to_usdc,
)
from tradeexecutor.state.freeze import freeze_position_on_failed_trade
from tradeexecutor.state.repair import close_position_with_empty_trade
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeType
from tradeexecutor.strategy.dust import (
    HYPERCORE_SMALL_POSITION_CLEANUP_CLOSE_EPSILON,
    HYPERCORE_SMALL_POSITION_CLEANUP_PENDING_REDEEM,
)
from tradeexecutor.strategy.execution_model import ExecutionHaltableIssue


logger = logging.getLogger(__name__)

USDC_QUANTUM = Decimal("0.000001")
HYPERCORE_SMALL_POSITION_MINIMUM_REDEEMABLE_EQUITY = (
    HYPERCORE_SMALL_POSITION_CLEANUP_CLOSE_EPSILON
)
assert HYPERCORE_SMALL_POSITION_MINIMUM_REDEEMABLE_EQUITY == raw_to_usdc(
    HYPERCORE_SMALL_POSITION_CLEANUP_MIN_REDEMPTION_RAW
)


#: Strategy parameters that express the lowest useful allocation.  HyperAI
#: uses the first name; the second preserves compatibility with older strategy
#: modules using the generic alpha-model setting.
MINIMUM_ALLOCATION_PARAMETER_NAMES = (
    "individual_rebalance_min_threshold_usd",
    "minimum_rebalance_trade_threshold",
)


@dataclass(slots=True)
class HypercoreSmallPositionCandidate:
    """One tracked HyperCore position eligible for small-position cleanup."""

    position_id: int
    vault_name: str
    vault_address: str
    live_equity: Decimal
    minimum_allocation: Decimal
    is_pending_cleanup: bool


@dataclass(slots=True)
class HypercoreSmallPositionCleanupReport:
    """Outcome of one small-position cleanup pass.

    ``executed_trades`` and ``closed_position_ids`` contain live outcomes only.
    ``simulated_state`` contains dry-run local closures and expired plans, but
    not the replacement trade or its signed transaction.
    """

    candidates: list[HypercoreSmallPositionCandidate]
    executed_trades: list[TradeExecution]
    closed_position_ids: list[int]
    simulated_state: State | None = None


def get_hypercore_minimum_allocation(parameters) -> Decimal | None:
    """Read the strategy's minimum useful allocation for HyperCore cleanup.

    Strategies without either parameter opt out safely.  This keeps the
    correct-accounts default-on option from inventing an allocation threshold.
    """

    if parameters is None:
        return None

    for name in MINIMUM_ALLOCATION_PARAMETER_NAMES:
        value = parameters.get(name)
        if value is not None:
            minimum_allocation = Decimal(str(value))
            return minimum_allocation if minimum_allocation > 0 else None

    return None


def discover_hypercore_small_positions(
    state,
    minimum_allocation: Decimal,
) -> list[HypercoreSmallPositionCandidate]:
    """Find live tracked HyperCore positions below the strategy allocation floor.

    ``correct-accounts`` revalues HyperCore positions from the Hyperliquid API
    immediately before calling this function, so ``position.get_value()`` is a
    live-equity value rather than the stale quantity held in state.
    """

    candidates: list[HypercoreSmallPositionCandidate] = []

    for position in state.portfolio.get_open_positions():
        if not position.pair.is_hyperliquid_vault():
            continue
        if position.has_unexecuted_trades():
            logger.warning(
                "Skipping HyperCore small-position cleanup for position %d: "
                "it has a started or broadcast trade",
                position.position_id,
            )
            continue

        live_equity = Decimal(str(position.get_value(include_interest=False))).quantize(USDC_QUANTUM)
        is_pending_cleanup = (
            position.other_data.get(
                HYPERCORE_SMALL_POSITION_CLEANUP_PENDING_REDEEM
            )
            is True
        )
        if live_equity <= 0 or (
            not is_pending_cleanup and live_equity >= minimum_allocation
        ):
            continue

        vault_address = position.pair.pool_address or position.pair.base.address
        candidates.append(
            HypercoreSmallPositionCandidate(
                position_id=position.position_id,
                vault_name=position.pair.get_vault_name() or position.pair.get_ticker(),
                vault_address=vault_address,
                live_equity=live_equity,
                minimum_allocation=minimum_allocation,
                is_pending_cleanup=is_pending_cleanup,
            )
        )

    return candidates


def _expire_planned_hypercore_trades(
    position,
    timestamp: datetime.datetime,
) -> None:
    """Expire stale plans before cleanup evaluates a candidate.

    Started and broadcast trades are filtered out during candidate discovery.
    A planned trade has not entered execution and cannot execute by itself, but
    its quantity still affects ``get_quantity(planned=True)``.
    """

    planned_trades = [
        trade
        for trade in position.trades.values()
        if trade.is_planned()
    ]
    for trade in planned_trades:
        trade.mark_expired(timestamp)
        trade.add_note(
            "Expired by correct-accounts before HyperCore small-position cleanup "
            "evaluated this position."
        )
        logger.warning(
            "Expired planned trade %d before evaluating HyperCore position %d",
            trade.trade_id,
            position.position_id,
        )


def plan_hypercore_small_position_cleanup(
    state,
    candidate: HypercoreSmallPositionCandidate,
    timestamp: datetime.datetime,
) -> TradeExecution:
    """Create a full-close trade for a cleanup candidate with redeemable equity.

    ``MINIMUM_VAULT_DEPOSIT`` is deliberately irrelevant here because the
    withdrawal encoder does not apply that client-side deposit check. Dust
    cleanup must not add capital or extend the vault lock-up.
    """

    position = state.portfolio.open_positions[candidate.position_id]
    assert position.pair.is_hyperliquid_vault()
    assert position.last_token_price > 0, (
        f"HyperCore cleanup position {position.position_id} has no usable price"
    )
    assert not position.has_unexecuted_trades(), (
        f"HyperCore cleanup position {position.position_id} has a trade in execution"
    )
    assert not position.has_planned_trades(), (
        f"HyperCore cleanup position {position.position_id} has a stale planned trade"
    )

    reserve_asset = state.portfolio.get_default_reserve_position().asset
    if candidate.is_pending_cleanup:
        cleanup_reason = "a prior small-position cleanup is awaiting redemption"
    else:
        cleanup_reason = (
            f"live equity {candidate.live_equity:.6f} USDC is below strategy minimum "
            f"allocation {candidate.minimum_allocation:.6f} USDC"
        )
    note_prefix = f"correct-accounts HyperCore small-position cleanup: {cleanup_reason}."

    close_quantity = -position.get_quantity()
    assert close_quantity < 0, (
        f"HyperCore cleanup position {position.position_id} has no positive quantity to close"
    )
    _position, close_trade, _created = state.portfolio.create_trade(
        strategy_cycle_at=timestamp,
        pair=position.pair,
        quantity=close_quantity,
        reserve=None,
        assumed_price=position.last_token_price,
        trade_type=TradeType.rebalance,
        reserve_currency=reserve_asset,
        reserve_currency_price=1.0,
        position=position,
        closing=True,
        notes=(
            f"{note_prefix} Redeeming the full position quantity with the "
            "HyperCore small-position retry margin."
        ),
    )
    close_trade.other_data["hypercore_small_position_cleanup"] = True
    return close_trade


def _close_confirmed_hypercore_residual(
    *,
    state,
    candidate: HypercoreSmallPositionCandidate,
    session,
    safe_address: str,
) -> Decimal | None:
    """Close safety-margin dust, or return a material residual to redeem again."""

    position = state.portfolio.open_positions.get(candidate.position_id)
    if position is None:
        return None

    live_equity = fetch_user_vault_equity(
        session,
        user=safe_address,
        vault_address=candidate.vault_address,
        bypass_cache=True,
    )
    residual_equity = Decimal(0) if live_equity is None else live_equity.equity
    if residual_equity > HYPERCORE_SMALL_POSITION_MINIMUM_REDEEMABLE_EQUITY:
        position.other_data[HYPERCORE_SMALL_POSITION_CLEANUP_PENDING_REDEEM] = True
        logger.warning(
            "HyperCore cleanup left %.6f USDC in position %d, above the %.2f "
            "dust write-off threshold; leaving it open for a later cleanup run",
            residual_equity,
            position.position_id,
            HYPERCORE_SMALL_POSITION_MINIMUM_REDEEMABLE_EQUITY,
        )
        return residual_equity

    repair_trade = close_position_with_empty_trade(state.portfolio, position)
    position.add_notes_message(
        "correct-accounts closed the final HyperCore safety-margin residual "
        f"({residual_equity:.6f} USDC) after small-position redemption"
    )
    logger.info(
        "Closed HyperCore cleanup residual for position %d with repair trade %d",
        candidate.position_id,
        repair_trade.trade_id,
    )
    return None


def run_hypercore_small_position_cleanup(
    *,
    state,
    timestamp: datetime.datetime,
    candidates: list[HypercoreSmallPositionCandidate],
    execution_model,
    routing_model,
    routing_state,
    session,
    safe_address: str,
    store,
    dry_run: bool = False,
) -> HypercoreSmallPositionCleanupReport:
    """Redeem tracked small HyperCore positions and return proceeds to reserves.

    Trades use the normal HyperCore execution router, so all three withdrawal
    phases are verified and an actual successful sell credits the portfolio
    reserve. The cleaner never tops up a position because the local 5 USDC
    check belongs to deposits and a deposit would create a new lock-up.
    Positions too small for meaningful phase verification are closed locally
    without broadcasting.

    :param dry_run:
        Rehearse local position closure or the complete trade preparation path
        on an isolated state copy. No transactions are broadcast and neither
        the supplied state nor its store is changed.
    """

    working_state = copy.deepcopy(state) if dry_run else state
    executed_trades: list[TradeExecution] = []
    closed_position_ids: list[int] = []

    for candidate in candidates:
        working_position = working_state.portfolio.open_positions.get(
            candidate.position_id
        )
        if working_position is None:
            continue

        try:
            live_equity_result = fetch_user_vault_equity(
                session,
                user=safe_address,
                vault_address=candidate.vault_address,
                bypass_cache=True,
            )
        except Exception as e:
            logger.warning(
                "HyperCore small-position cleanup skipped position %d: could not refresh "
                "live vault equity: %s",
                candidate.position_id,
                e,
            )
            continue

        _expire_planned_hypercore_trades(working_position, timestamp)
        live_equity = Decimal(0) if live_equity_result is None else live_equity_result.equity
        if live_equity <= 0:
            close_position_with_empty_trade(working_state.portfolio, working_position)
            if dry_run:
                logger.info(
                    "Dry run: validated local closure of zero-equity HyperCore "
                    "position %d (%s)",
                    candidate.position_id,
                    candidate.vault_name,
                )
            else:
                store.sync(working_state)
                closed_position_ids.append(candidate.position_id)
            continue
        if live_equity >= candidate.minimum_allocation and not candidate.is_pending_cleanup:
            logger.info(
                "HyperCore small-position cleanup skipped position %d: live equity %.6f USDC "
                "now meets the %.6f USDC allocation minimum",
                candidate.position_id,
                live_equity,
                candidate.minimum_allocation,
            )
            continue
        if not live_equity_result.is_lockup_expired:
            logger.info(
                "HyperCore small-position cleanup deferred for position %d: vault lock-up "
                "remains until %s",
                candidate.position_id,
                live_equity_result.locked_until,
            )
            continue

        current_candidate = replace(candidate, live_equity=live_equity)

        if live_equity <= HYPERCORE_SMALL_POSITION_MINIMUM_REDEEMABLE_EQUITY:
            close_position_with_empty_trade(working_state.portfolio, working_position)
            working_position.add_notes_message(
                "correct-accounts closed the HyperCore position locally because "
                f"{live_equity:.6f} USDC was not enough to verify all withdrawal phases"
            )
            if dry_run:
                logger.info(
                    "Dry run: validated local closure of HyperCore position %d "
                    "(%s) because %.6f USDC is not enough to verify all "
                    "withdrawal phases",
                    candidate.position_id,
                    candidate.vault_name,
                    live_equity,
                )
            else:
                store.sync(working_state)
                closed_position_ids.append(candidate.position_id)
            continue

        preparation_state = copy.deepcopy(working_state) if dry_run else working_state

        trade = plan_hypercore_small_position_cleanup(
            preparation_state,
            current_candidate,
            timestamp,
        )
        trades = [trade]
        if dry_run:
            assert execution_model is not None
            assert routing_model is not None
            assert routing_state is not None
            execution_model.validate_confirmation_configuration()
            preparation_state.start_execution(
                timestamp,
                trade,
                underflow_check=False,
            )
            if execution_model.max_slippage is not None:
                trade.planned_max_slippage = execution_model.max_slippage
            routing_model.setup_trades(
                state=preparation_state,
                routing_state=routing_state,
                trades=trades,
                check_balances=False,
                rebroadcast=False,
            )
            logger.info(
                "Dry run: validated HyperCore small-position withdrawal %d (%s), "
                "live equity %.6f USDC, strategy minimum allocation %.6f USDC, "
                "prepared %d transaction(s) without broadcasting",
                current_candidate.position_id,
                current_candidate.vault_name,
                current_candidate.live_equity,
                current_candidate.minimum_allocation,
                len(trade.blockchain_transactions),
            )
            continue

        logger.info(
            "Cleaning HyperCore small position %d (%s): equity %.6f USDC, "
            "minimum allocation %.6f USDC, action=redeem",
            current_candidate.position_id,
            current_candidate.vault_name,
            current_candidate.live_equity,
            current_candidate.minimum_allocation,
        )
        try:
            execution_model.execute_trades(
                timestamp,
                working_state,
                trades,
                routing_model,
                routing_state,
                check_balances=False,
            )
        except (ExecutionHaltableIssue, HypercoreWithdrawalPreflightError) as e:
            if isinstance(e, HypercoreWithdrawalPreflightError) and trade.is_started():
                working_state.mark_trade_failed(timestamp, trade)
                freeze_position_on_failed_trade(timestamp, working_state, [trade])
            elif isinstance(e, HypercoreWithdrawalPreflightError) and trade.is_planned():
                trade.mark_expired(timestamp)
            logger.error(
                "HyperCore small-position cleanup failed for position %d: %s. "
                "The updated state has been saved; repair the failed trade before retrying.",
                current_candidate.position_id,
                e,
            )
            executed_trades.extend(trades)
            continue
        else:
            executed_trades.extend(trades)
        finally:
            store.sync(working_state)

        if not trade.is_success():
            continue

        residual_equity = _close_confirmed_hypercore_residual(
            state=working_state,
            candidate=current_candidate,
            session=session,
            safe_address=safe_address,
        )
        store.sync(working_state)
        if residual_equity is None:
            closed_position_ids.append(candidate.position_id)

    return HypercoreSmallPositionCleanupReport(
        candidates=candidates,
        executed_trades=executed_trades,
        closed_position_ids=closed_position_ids,
        simulated_state=working_state if dry_run else None,
    )
