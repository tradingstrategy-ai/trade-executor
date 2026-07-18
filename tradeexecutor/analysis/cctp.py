"""CCTP bridge trade analysis for cross-chain vault strategies."""
from decimal import Decimal
from itertools import chain as ichain
from typing import Iterable

import pandas as pd

from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeStatus
from tradeexecutor.state.types import USDollarAmount


def _format_planning_reasons(planning_amounts: dict | None) -> str:
    """Render a bridge trade's per-reason planning breakdown as a short string.

    :param planning_amounts:
        ``trade.other_data["cctp_planning_amounts"]`` — reason -> Decimal-as-string
        amount. ``None`` / missing for bridge trades from older state files.

    :return:
        e.g. ``"net_sell:12.5, idle_sweep:88.1"``, or empty string when absent or
        all-zero.
    """
    if not planning_amounts:
        return ""
    parts = []
    for reason, amount in planning_amounts.items():
        if Decimal(str(amount)) != 0:
            parts.append(f"{reason}:{amount}")
    return ", ".join(parts)


def build_bridge_trade_dataframe(
    bridge_trades: Iterable[TradeExecution],
) -> pd.DataFrame:
    """Build a DataFrame of CCTP bridge trades with direction, chain, and value.

    :param bridge_trades:
        Trades where ``trade.pair.is_cctp_bridge()`` is True.

    :return:
        DataFrame sorted by opened_at with columns: trade_id, opened_at,
        executed_at, direction, source_chain_id, destination_chain_id,
        value_usd, quantity, status, reasons, position_id.
    """
    rows = []
    for trade in bridge_trades:
        rows.append({
            "trade_id": trade.trade_id,
            "opened_at": trade.opened_at,
            "executed_at": trade.executed_at,
            "direction": "bridge out" if trade.is_buy() else "bridge back",
            "source_chain_id": trade.pair.get_source_chain_id(),
            "destination_chain_id": trade.pair.get_destination_chain_id(),
            "value_usd": trade.get_value(),
            "quantity": float(trade.executed_quantity or 0),
            "status": trade.get_status().value,
            "reasons": _format_planning_reasons(trade.other_data.get("cctp_planning_amounts")),
            "position_id": trade.position_id,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["opened_at", "trade_id"])
    return df


def build_bridge_trade_summary(
    bridge_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise CCTP bridge trades by destination chain and direction.

    :param bridge_df:
        Output of :py:func:`build_bridge_trade_dataframe`.

    :return:
        DataFrame grouped by destination_chain_id and direction with
        trade count, total value, and first/last trade timestamps.
    """
    if bridge_df.empty:
        return pd.DataFrame(
            columns=["destination_chain_id", "direction", "trades", "value_usd", "first_trade", "last_trade"],
        )
    return bridge_df.groupby(["destination_chain_id", "direction"]).agg(
        trades=("trade_id", "count"),
        value_usd=("value_usd", "sum"),
        first_trade=("opened_at", "min"),
        last_trade=("opened_at", "max"),
    ).reset_index()


def assert_bridge_coverage(
    trades: Iterable[TradeExecution],
    primary_chain_id: int,
) -> pd.DataFrame:
    """Assert a cross-chain vault run has satellite vaults and CCTP bridge trades.

    This validates that the run exercised the bridge-trade injection path. A
    satellite vault cycle no longer needs a bridge trade in the same cycle:
    existing idle satellite bridge capital can fund the vault trade, and the
    CCTP planner should bridge only the missing amount.

    :param trades:
        All trades from the backtest portfolio.

    :param primary_chain_id:
        Chain ID of the primary reserve chain (e.g. ``ChainId.ethereum.value``).

    :return:
        Per-cycle summary DataFrame with trade counts and a boolean flag for
        satellite vault cycles that reused existing bridge capital.

    :raise AssertionError:
        If the run has no satellite vault trades or no CCTP bridge trades.
    """
    rows = []
    for trade in trades:
        rows.append({
            "cycle": trade.strategy_cycle_at,
            "trade_id": trade.trade_id,
            "kind": trade.pair.kind.value,
            "chain_id": trade.pair.chain_id,
            "is_bridge": trade.pair.is_cctp_bridge(),
            "is_satellite_vault": trade.pair.kind.is_vault() and trade.pair.chain_id != primary_chain_id,
            "value_usd": trade.get_value(),
            "status": trade.get_status().value,
        })

    trade_df = pd.DataFrame(rows)
    cycle_df = trade_df.groupby("cycle").agg(
        trades=("trade_id", "count"),
        bridge_trades=("is_bridge", "sum"),
        satellite_vault_trades=("is_satellite_vault", "sum"),
        traded_value_usd=("value_usd", "sum"),
    ).reset_index()
    cycle_df["satellite_vaults_without_same_cycle_bridge"] = (
        (cycle_df["satellite_vault_trades"] > 0)
        & (cycle_df["bridge_trades"] == 0)
    )

    cycles_with_satellite_vaults = cycle_df[cycle_df["satellite_vault_trades"] > 0]

    assert len(cycles_with_satellite_vaults) > 0, "Expected satellite vault trades"
    assert cycle_df["bridge_trades"].sum() > 0, "Expected CCTP bridge trades"

    return cycle_df


def _position_in_transit_value(position) -> float:
    """Sum ``planned_reserve`` of a position's ``cctp_in_transit`` trades."""
    total = 0.0
    for trade in position.trades.values():
        if trade.get_status() == TradeStatus.cctp_in_transit:
            total += float(trade.planned_reserve or 0)
    return total


def _chain_has_pending_async_deposit(state: State, chain_id: int) -> bool:
    """Does any satellite position on this chain have an unsettled async deposit?

    ``bridge_capital_allocated`` alone cannot answer this — every satellite buy
    (synchronous included) allocates bridge capital for the lifetime of the
    position, so a positive allocation is the normal deployed state, not a
    settlement queue. Only an actual ``vault_settlement_pending`` buy means
    bridge capital is reserved for a deposit that has not settled yet.
    """
    for position in ichain(
        state.portfolio.open_positions.values(),
        state.portfolio.pending_positions.values(),
    ):
        if position.pair.chain_id != chain_id or position.pair.is_cctp_bridge():
            continue
        for trade in position.trades.values():
            if trade.is_buy() and trade.get_status() == TradeStatus.vault_settlement_pending:
                return True
    return False


def analyse_idle_bridge_capital(
    state: State,
    bridge_sweep_min_usd: USDollarAmount = 1.0,
    sweep_enabled: bool = True,
) -> pd.DataFrame:
    """Report idle capital left on CCTP bridge positions and why it was not swept.

    The idle-capital sweep (issue #1562) returns settled satellite bridge capital
    to the hub so it does not sit idle. This is the end-of-run acceptance check:
    one row per open bridge position, classifying any remaining available capital.

    Strategy parameters are not persisted in state, so ``sweep_enabled`` and
    ``bridge_sweep_min_usd`` must be passed by the caller (the notebook/test uses
    its own ``Parameters`` values); the function never infers them from state.

    ``why_not_swept`` classification:

    - ``swept`` — available capital is at or below one raw token unit (cleared).
    - ``below_min_sweep`` — available capital is between one raw unit and
      ``bridge_sweep_min_usd``, deliberately left as the configured dust buffer.
    - ``reserved_for_async_deposit`` — the chain has an unsettled async vault
      deposit (an actual ``vault_settlement_pending`` buy), so part of the
      bridge capital is committed and only the sub-threshold remainder is idle.
      Note a plain positive ``bridge_capital_allocated`` is NOT enough — every
      satellite buy allocates for the position's lifetime.
    - ``in_transit`` — a bridge-back is mid-flight for this position.
    - ``sweep_disabled`` — the sweep is off and idle capital at or above the
      threshold remains (this reproduces the pre-fix leak).
    - ``not_swept`` — available capital at or above the threshold remains with the
      sweep enabled; flags a leak to investigate. Two benign causes exist:
      capital that settled *after* the final planner run (e.g. an async
      redemption settling on the last tick — it would be swept next cycle), and
      a chain that also has a pending async deposit (the threshold branch wins
      over ``reserved_for_async_deposit``). Persistent ``not_swept`` across
      cycles is the genuine-leak signal.

    :param state:
        Portfolio state after a run.

    :param bridge_sweep_min_usd:
        The sweep threshold the run used.

    :param sweep_enabled:
        Whether the run had the sweep enabled.

    :return:
        DataFrame with columns: destination_chain_id, quantity,
        bridge_capital_allocated, available_idle, in_transit_value,
        why_not_swept. Empty (with those columns) when no bridge positions are
        open.
    """
    columns = [
        "destination_chain_id",
        "quantity",
        "bridge_capital_allocated",
        "available_idle",
        "in_transit_value",
        "why_not_swept",
    ]
    rows = []
    for position in state.portfolio.open_positions.values():
        if not position.pair.is_cctp_bridge():
            continue

        raw_unit = position.pair.base.convert_to_decimal(1)
        available = position.get_available_bridge_capital()
        allocated = position.bridge_capital_allocated
        in_transit = _position_in_transit_value(position)
        threshold = Decimal(str(bridge_sweep_min_usd))

        if in_transit > 0:
            why = "in_transit"
        elif available >= threshold:
            why = "sweep_disabled" if not sweep_enabled else "not_swept"
        elif allocated > raw_unit and _chain_has_pending_async_deposit(
            state, position.pair.get_destination_chain_id(),
        ):
            why = "reserved_for_async_deposit"
        elif available <= raw_unit:
            why = "swept"
        else:
            why = "below_min_sweep"

        rows.append({
            "destination_chain_id": position.pair.get_destination_chain_id(),
            "quantity": float(position.get_quantity()),
            "bridge_capital_allocated": float(allocated),
            "available_idle": float(available),
            "in_transit_value": in_transit,
            "why_not_swept": why,
        })

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values("destination_chain_id").reset_index(drop=True)
