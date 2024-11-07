"""Display trading positions as Pandas notebook items."""
import datetime
from typing import Iterable

import pandas as pd

from tradeexecutor.ethereum.revert import clean_revert_reason_message
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.reserve import ReservePosition


def _ftime(v: datetime.datetime) -> str:
    """Format times"""
    if not v:
        return ""
    return v.strftime('%Y-%m-%d %H:%M')


def display_positions(positions: Iterable[TradingPosition]) -> pd.DataFrame:
    """Format trading positions for Jupyter Notebook table output.

    Display in one table

    - All positions

    - Their underlying trades

    :return:
        DataFrame containing positions and trades, values as string formatted
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

        items.append({
            "Flags": ", ".join(flags),
            "Ticker": p.pair.get_ticker(),
            # f"Size": p.get_quantity(),
            # "Profit": p.get_realised_profit_percent() * 100 if p.is_closed() else "",
            "Opened": _ftime(p.opened_at),
            "Closed": _ftime(p.closed_at),
            "Qty": p.get_quantity(),
            "Notes": (p.notes or "")[0:20],
        })

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
                f"Trade size": f"{t.get_position_quantity():,.2f}",
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
