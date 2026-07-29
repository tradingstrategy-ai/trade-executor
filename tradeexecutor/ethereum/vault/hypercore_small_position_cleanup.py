"""Clean undersized tracked HyperCore vault positions.

This module is deliberately limited to **HyperCore native vault** positions.
It must not be used for ERC-4626, ERC-7540, or Ostium vault positions: those
protocols have different redemption and settlement semantics.  The current
caller uses Lagoon's HyperCore execution route, but this module never handles
Lagoon vault shares.

HyperCore ``vaultTransfer`` has no withdraw-all operation and rejects an
amount that is even slightly above live equity.  A small position is therefore
redeemed through the normal HyperCore router, with a full-close flag and a
larger retry margin if the first vaultTransfer silently no-ops.  When the
position is too small to leave the protocol's 5 USDC withdrawal floor after
the safety margin, the cleaner first tops it up by enough to make every retry
withdrawable from fresh live equity.  The subsequent full close returns the
usable capital to reserves.  A deposit locks the whole vault position, so the
top-up and the later redemption deliberately happen in separate runs.
Any retry residual above the close epsilon remains state-tracked and is
redeemed on a later run; only final protocol dust is written off in state.
If a routed trade fails, the updated state is saved and the normal failed-trade
repair flow takes over.
"""

import datetime
import logging
from dataclasses import dataclass, replace
from decimal import Decimal

from eth_defi.hyperliquid.api import fetch_user_vault_equity
from eth_defi.hyperliquid.core_writer import MINIMUM_VAULT_DEPOSIT

from tradeexecutor.ethereum.vault.hypercore_routing import (
    HYPERCORE_SMALL_POSITION_CLEANUP_RETRY_SAFETY_MARGINS_RAW,
    HypercoreWithdrawalPreflightError,
    raw_to_usdc,
    usdc_to_raw,
)
from tradeexecutor.state.freeze import freeze_position_on_failed_trade
from tradeexecutor.state.repair import close_position_with_empty_trade
from tradeexecutor.state.trade import TradeExecution, TradeType
from tradeexecutor.strategy.dust import HYPERLIQUID_VAULT_CLOSE_EPSILON
from tradeexecutor.strategy.execution_model import ExecutionHaltableIssue


logger = logging.getLogger(__name__)

USDC_QUANTUM = Decimal("0.000001")


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
    top_up_reserve: Decimal | None
    is_pending_cleanup: bool


@dataclass(slots=True)
class HypercoreSmallPositionCleanupReport:
    """Outcome of one small-position cleanup pass."""

    candidates: list[HypercoreSmallPositionCandidate]
    executed_trades: list[TradeExecution]
    closed_position_ids: list[int]


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


def get_hypercore_small_position_top_up_reserve(
    live_equity: Decimal,
) -> Decimal | None:
    """Get the temporary top-up that makes every cleanup retry withdrawable now."""

    live_equity = raw_to_usdc(usdc_to_raw(live_equity))
    minimum_deposit = raw_to_usdc(MINIMUM_VAULT_DEPOSIT)
    largest_retry_margin = raw_to_usdc(
        max(HYPERCORE_SMALL_POSITION_CLEANUP_RETRY_SAFETY_MARGINS_RAW)
    )
    minimum_equity_for_retry_ladder = minimum_deposit + largest_retry_margin
    if live_equity >= minimum_equity_for_retry_ladder:
        return None
    return max(minimum_deposit, minimum_equity_for_retry_ladder - live_equity)


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
                "it has unfinished trades",
                position.position_id,
            )
            continue

        live_equity = Decimal(str(position.get_value(include_interest=False))).quantize(USDC_QUANTUM)
        is_pending_cleanup = (
            position.other_data.get("hypercore_small_position_cleanup_pending_redeem") is True
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
                top_up_reserve=get_hypercore_small_position_top_up_reserve(live_equity),
                is_pending_cleanup=is_pending_cleanup,
            )
        )

    return candidates


def plan_hypercore_small_position_cleanup(
    state,
    candidate: HypercoreSmallPositionCandidate,
    timestamp: datetime.datetime,
) -> list[TradeExecution]:
    """Create one top-up or full-close trade for a cleanup candidate.

    A HyperCore deposit locks the whole vault position, so a top-up must be
    persisted and allowed to unlock before its later redemption.
    """

    position = state.portfolio.open_positions[candidate.position_id]
    assert position.pair.is_hyperliquid_vault()
    assert position.last_token_price > 0, (
        f"HyperCore cleanup position {position.position_id} has no usable price"
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

    if candidate.top_up_reserve is not None:
        _position, top_up_trade, _created = state.portfolio.create_trade(
            strategy_cycle_at=timestamp,
            pair=position.pair,
            quantity=None,
            reserve=candidate.top_up_reserve,
            assumed_price=position.last_token_price,
            trade_type=TradeType.rebalance,
            reserve_currency=reserve_asset,
            reserve_currency_price=1.0,
            position=position,
            notes=(
                f"{note_prefix} Topping up {candidate.top_up_reserve:.6f} USDC so every "
                "HyperCore retry can meet the withdrawal floor."
            ),
        )
        top_up_trade.other_data["hypercore_small_position_cleanup"] = True
        return [top_up_trade]

    close_quantity = -position.get_quantity(planned=True)
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
            f"{note_prefix} Redeeming the full planned quantity with the "
            "HyperCore small-position retry margin."
        ),
    )
    close_trade.other_data["hypercore_small_position_cleanup"] = True
    return [close_trade]


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
    if residual_equity > HYPERLIQUID_VAULT_CLOSE_EPSILON:
        position.other_data["hypercore_small_position_cleanup_pending_redeem"] = True
        logger.warning(
            "HyperCore cleanup left %.6f USDC in position %d, above the %.2f "
            "dust write-off threshold; leaving it open for a later cleanup run",
            residual_equity,
            position.position_id,
            HYPERLIQUID_VAULT_CLOSE_EPSILON,
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
) -> HypercoreSmallPositionCleanupReport:
    """Redeem tracked small HyperCore positions and return proceeds to reserves.

    Trades use the normal HyperCore execution router, so all three withdrawal
    phases are verified and an actual successful sell credits the portfolio
    reserve.  A top-up is persisted by itself because it locks the whole vault
    position until HyperCore's lock-up expires.
    """

    executed_trades: list[TradeExecution] = []
    closed_position_ids: list[int] = []

    for candidate in candidates:
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

        live_equity = Decimal(0) if live_equity_result is None else live_equity_result.equity
        if live_equity <= 0:
            position = state.portfolio.open_positions.get(candidate.position_id)
            if position is not None:
                close_position_with_empty_trade(state.portfolio, position)
                store.sync(state)
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

        current_candidate = replace(
            candidate,
            live_equity=live_equity,
            top_up_reserve=get_hypercore_small_position_top_up_reserve(live_equity),
        )
        is_top_up = current_candidate.top_up_reserve is not None
        if is_top_up:
            reserve_quantity = state.portfolio.get_default_reserve_position().quantity
            if reserve_quantity < current_candidate.top_up_reserve:
                logger.warning(
                    "HyperCore small-position cleanup skipped position %d: needs %.6f USDC "
                    "temporary top-up but only %.6f USDC is in reserves",
                    current_candidate.position_id,
                    current_candidate.top_up_reserve,
                    reserve_quantity,
                )
                continue

        trades = plan_hypercore_small_position_cleanup(state, current_candidate, timestamp)
        logger.info(
            "Cleaning HyperCore small position %d (%s): equity %.6f USDC, minimum "
            "allocation %.6f USDC, action=%s",
            current_candidate.position_id,
            current_candidate.vault_name,
            current_candidate.live_equity,
            current_candidate.minimum_allocation,
            "top_up" if is_top_up else "redeem",
        )
        try:
            execution_model.execute_trades(
                timestamp,
                state,
                trades,
                routing_model,
                routing_state,
                check_balances=False,
            )
        except (ExecutionHaltableIssue, HypercoreWithdrawalPreflightError) as e:
            trade = trades[0]
            if isinstance(e, HypercoreWithdrawalPreflightError) and trade.is_started():
                state.mark_trade_failed(timestamp, trade)
                freeze_position_on_failed_trade(timestamp, state, [trade])
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
            store.sync(state)

        trade = trades[0]
        if not trade.is_success():
            continue
        if is_top_up:
            position = state.portfolio.open_positions.get(current_candidate.position_id)
            if position is None:
                logger.error(
                    "HyperCore small-position cleanup top-up trade %d succeeded but position %d "
                    "is no longer open",
                    trade.trade_id,
                    current_candidate.position_id,
                )
                continue
            position.other_data["hypercore_small_position_cleanup_pending_redeem"] = True
            store.sync(state)
            logger.info(
                "HyperCore small-position cleanup topped up position %d; redemption will run "
                "after the vault lock-up expires",
                current_candidate.position_id,
            )
            continue

        residual_equity = _close_confirmed_hypercore_residual(
            state=state,
            candidate=current_candidate,
            session=session,
            safe_address=safe_address,
        )
        store.sync(state)
        if residual_equity is None:
            closed_position_ids.append(candidate.position_id)

    return HypercoreSmallPositionCleanupReport(
        candidates=candidates,
        executed_trades=executed_trades,
        closed_position_ids=closed_position_ids,
    )
