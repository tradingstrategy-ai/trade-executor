"""Display trading positions as Pandas notebook items."""
import datetime
import textwrap
from typing import Iterable

import pandas as pd

from tradeexecutor.cli.commands.shared_options import TradeType
from tradeexecutor.ethereum.revert import clean_revert_reason_message
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.reserve import ReservePosition
from tradeexecutor.state.trade import TradeExecution


def _ftime(v: datetime.datetime) -> str:
    """Format times"""
    if not v:
        return ""
    return v.strftime('%Y-%m-%d %H:%M')


def display_positions(
    positions: Iterable[TradingPosition],
    trade_type: TradeType | None = None
) -> pd.DataFrame:
    """Format trading positions for Jupyter Notebook table output.

    Display in one table

    - All positions

    - Their underlying trades (conditionally based on trade_type)

    :param positions: Positions to display
    :param trade_type: Controls whether trades are displayed. If None or TradeType.all,
        trades are shown. If TradeType.none, trades are hidden.

    :return:
        DataFrame containing positions and optionally trades, values as string formatted
    """

    items = []
    idx = []
    for p in positions:
        idx.append(p.position_id)
        flags = []

        if p.is_frozen():
            flags.append("F")

        if p.is_repaired():
            flags.append("R")

        if p.has_unexecuted_trades():
            flags.append("UE")

        if p.is_stop_loss():
            flags.append("SL")

        notes = p.pair.base.address + "\n"
        long_notes = (p.notes or "")[0:100]  # Limit to 100 characters
        notes += "\n".join(textwrap.wrap(long_notes, width=20))

        items.append({
            "Flags": ", ".join(flags),
            "Ticker": p.pair.get_ticker(),
            "Value USD": f"${p.get_value():,.2f}",
            # f"Size": p.get_quantity(),
            # "Profit": p.get_realised_profit_percent() * 100 if p.is_closed() else "",
            "Opened": _ftime(p.opened_at),
            "Closed": _ftime(p.closed_at),
            "Qty": f"{p.get_quantity():,.4f}",
            "Notes": notes,
        })

        # Only display trades if trade_type is not "none"
        if trade_type is None or trade_type.value != "none":
            for t in p.trades.values():
                idx.append(p.position_id)

                flags = []

                flags.append("T")

                if t.is_buy():
                    flags.append("B")

                if t.is_sell():
                    flags.append("S")

                if t.is_stop_loss():
                    flags.append("SL")

                if t.is_repair_trade():
                    flags.append("R2")

                if t.is_repaired():
                    flags.append("R1")

                text = []
                if t.notes:
                    text.append(t.notes)

                revert_reason = clean_revert_reason_message(t.get_revert_reason())
                if revert_reason:
                    text.append(revert_reason)

                items.append({
                    "Flags": ", ".join(flags),
                    "Ticker": "‎ ‎ ‎ ‎ ‎ ┗",
                    "Trade id": str(t.trade_id),  # Mixed NA/number column fix
                    "Price": f"{t.executed_price:.6f}" if t.executed_price else "-",
                    "Qty": f"{t.get_position_quantity():,.4f}",
                    "Opened": _ftime(t.opened_at),
                    "Executed": _ftime(t.executed_at),
                    "Notes": "\n".join(text)[0:20],
                })

    df = pd.DataFrame(items, index=idx)
    df = df.fillna("")
    df = df.replace({pd.NaT: ""})
    return df


def display_reserve_position_events(position: ReservePosition) -> pd.DataFrame:
    """Display events that cause the balance of the reserve position.

    """

    items = []
    idx = []

    for event in position.get_balance_update_events():
        idx.append(event.balance_update_id)
        items.append({
            "Cause": event.cause.name,
            "At": event.block_mined_at,
            "Quantity": event.quantity,
            "Dollar value": event.usd_value,
            "Address": event.owner_address,
            "Notes": event.notes,
        })

    df = pd.DataFrame(items, index=idx)
    df = df.fillna("")
    df = df.replace({pd.NaT: ""})
    return df



def display_transactions(trades: Iterable[TradeExecution]) -> pd.DataFrame:
    """Format blockchain transactions for console table output.

    Display in one table

    - Transaction data with associated trade information

    :return:
        DataFrame containing positions and trades, values as string formatted
    """

    items = []
    idx = []
    for t in trades:
        ticker = t.pair.get_ticker()
        trade_id = t.trade_id
        for tx in t.blockchain_transactions:
            idx.append(trade_id)
            flags = []

            if tx.is_reverted():
                flags.append("R")

            if t.is_buy():
                flags.append("B")

            if t.is_sell():
                flags.append("S")

            items.append({
                "F": "".join(flags),
                "Id": t.trade_id,
                "Trade": ticker,
                # "Broadcasted": _ftime(tx.broadcasted_at),
                "Block": f"{tx.block_number or 0:,}",
                "Hash": tx.tx_hash,
                "Gas": tx.realised_gas_units_consumed,
                "Price (GWei)": tx.realised_gas_price // (10**9) if tx.realised_gas_price else "-",
                "Revert reason": _format_long_string(tx.revert_reason),
                # "Notes": (tx.notes or "")[0:20],
            })

    df = pd.DataFrame(items, index=idx)
    df = df.fillna("")
    df = df.replace({pd.NaT: ""})
    df = df.sort_values(by=["Id", "Block"]).set_index("Id")
    return df


def _format_long_string(text, max_length=20):
    """
    Splits a long string into multiple lines.

    Args:
    text (str): The string to split.
    max_length (int): Maximum number of characters per line. Default is 80.

    Returns:
    str: A multi-line string where no line exceeds max_length characters.
    """

    if text is None:
        text = ""

    # Split the text into words
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        # If adding the word would exceed max_length, start a new line
        if len(current_line) + len(word) + 1 > max_length:
            lines.append(current_line.strip())
            current_line = word
        else:
            # Add the word to the current line
            if current_line:
                current_line += " " + word
            else:
                current_line = word

    # Append the last line if it's not empty
    if current_line:
        lines.append(current_line.strip())

    # Join all lines with newline characters
    return "\n".join(lines)


def display_position_valuations(positions: Iterable[TradingPosition]) -> pd.DataFrame:
    """Format position valuations for Jupyter Notebook/console table output.

    Display in one table

    - All positions

    - Their values

    :return:
        DataFrame containing positions and their valuatio data
    """

    sorted_positions = [p for p in positions]
    sorted_positions.sort(key=lambda p: p.get_value(), reverse=True)

    items = []
    idx = []
    for p in positions:
        idx.append(p.position_id)
        flags = []
        success_trades = [t for t in p.trades.values() if t.is_success()]
        failed_trades = [t for t in p.trades.values() if (t.is_failed() or t.is_repaired())]
        if success_trades:
            last_trade_at = success_trades[-1].executed_at
        else:
            last_trade_at = None

        if failed_trades:
            last_failed_trade_at = failed_trades[-1].executed_at
        else:
            last_failed_trade_at = None

        items.append({
            "Id": p.position_id,
            # "Flags": ", ".join(flags),
            "Ticker": p.pair.get_ticker(),
            "Value USD": p.get_value(),
            "Qty": f"{p.get_quantity():,.4f}",
            "Token price": f"{p.last_token_price:,.9f}",
            "Valued at": _ftime(p.last_pricing_at),
            "Last trade at": _ftime(last_trade_at),
            "Last failed trade at": _ftime(last_failed_trade_at),
        })

    df = pd.DataFrame(items, index=idx)
    df = df.fillna("")
    if len(df) > 0:
        df = df.set_index("Id")
        df = df.replace({pd.NaT: ""})
    return df


def build_chain_exposure_snapshot(
    positions: Iterable[TradingPosition],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a per-position and per-chain exposure summary from open positions.

    Bridge positions (CCTP) are excluded so the result shows only
    vault/spot exposure.

    :param positions:
        Open and frozen positions, e.g. from
        ``state.portfolio.get_open_and_frozen_positions()``.

    :return:
        Tuple of (positions_df, chain_exposure_df).

        *positions_df* has columns: position_id, chain_id, pair, value_usd,
        opened_at — one row per non-bridge position, sorted by value descending.

        *chain_exposure_df* has columns: chain_id, positions, value_usd,
        weight — one row per chain.
    """
    rows = []
    for position in positions:
        if position.pair.is_cctp_bridge():
            continue
        rows.append({
            "position_id": position.position_id,
            "chain_id": position.pair.chain_id,
            "pair": position.pair.get_ticker(),
            "value_usd": position.get_value(),
            "opened_at": position.opened_at,
        })

    positions_df = pd.DataFrame(rows)
    if positions_df.empty:
        chain_exposure_df = pd.DataFrame(columns=["chain_id", "positions", "value_usd", "weight"])
        return positions_df, chain_exposure_df

    positions_df = positions_df.sort_values("value_usd", ascending=False)

    chain_exposure_df = positions_df.groupby("chain_id").agg(
        positions=("position_id", "count"),
        value_usd=("value_usd", "sum"),
    ).reset_index()
    total = chain_exposure_df["value_usd"].sum()
    chain_exposure_df["weight"] = chain_exposure_df["value_usd"] / total if total > 0 else 0

    return positions_df, chain_exposure_df


def position_duration_summary(
    state,
    backtest_end: datetime.datetime | None = None,
) -> dict:
    """Compute position duration statistics from a backtest state.

    :param state:
        Completed backtest state.

    :param backtest_end:
        End timestamp for open positions that have no ``closed_at``.
        Defaults to now if not given.

    :return:
        Dict with keys: position_count, positions_lt_3d,
        share_positions_lt_3d, median_position_days.
    """
    if backtest_end is None:
        backtest_end = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)

    positions = [p for p in state.portfolio.get_all_positions() if not p.is_credit_supply()]
    durations = []
    for position in positions:
        closed_at = position.closed_at if position.closed_at is not None else backtest_end
        durations.append((closed_at - position.opened_at).total_seconds() / 86400)

    duration_series = pd.Series(durations, dtype="float64")
    return {
        "position_count": len(positions),
        "positions_lt_3d": int((duration_series < 3).sum()),
        "share_positions_lt_3d": float((duration_series < 3).mean()) if len(duration_series) else float("nan"),
        "median_position_days": float(duration_series.median()) if len(duration_series) else float("nan"),
    }