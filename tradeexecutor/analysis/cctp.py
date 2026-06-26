"""CCTP bridge trade analysis for cross-chain vault strategies."""
from typing import Iterable

import pandas as pd

from tradeexecutor.state.trade import TradeExecution


def build_bridge_trade_dataframe(
    bridge_trades: Iterable[TradeExecution],
) -> pd.DataFrame:
    """Build a DataFrame of CCTP bridge trades with direction, chain, and value.

    :param bridge_trades:
        Trades where ``trade.pair.is_cctp_bridge()`` is True.

    :return:
        DataFrame sorted by opened_at with columns: trade_id, opened_at,
        executed_at, direction, source_chain_id, destination_chain_id,
        value_usd, quantity, status, position_id.
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
