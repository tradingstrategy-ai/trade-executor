"""Display trading positions as Pandas notebook items."""
import datetime
from typing import Iterable

import pandas as pd

from tradeexecutor.ethereum.revert import clean_revert_reason_message
from tradeexecutor.state.position import TradingPosition


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

        if p.is_repaired():
            flags.append("R")

        if p.has_unexecuted_trades():
            flags.append("UE")

        items.append({
            "Flags": ", ".join(flags),
            "Ticker": p.pair.get_ticker(),
            "Profit": p.get_realised_profit_percent() * 100 if p.is_closed() else "",
            "Opened at": _ftime(p.opened_at),
            "Closed at": _ftime(p.closed_at),
            "Notes": p.notes,
        })

        for t in p.trades.values():
            idx.append(p.position_id)

            revert_reason = clean_revert_reason_message(t.get_revert_reason())
            text = (t.notes or "") + revert_reason

            items.append({
                "Trade id": str(t.trade_id),  # Mixed NA/number column fix
                "Price": t.executed_price,
                "Trade opened": _ftime(t.opened_at),
                "Trade executed": _ftime(t.executed_at),
                "Trade notes": text,
            })

    df = pd.DataFrame(items, index=idx)
    df = df.fillna("")
    df = df.replace({pd.NaT: ""})
    return df


