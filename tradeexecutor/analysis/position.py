"""Display trading positions as Pandas notebook items."""
from typing import Iterable

import pandas as pd

from tradeexecutor.state.position import TradingPosition


def display_positions(positions: Iterable[TradingPosition]) -> pd.DataFrame:
    """Format trading positions for Jupyter Notebook table output."""

    items = []
    idx = []
    for p in positions:
        first_trade = p.get_first_trade()
        last_trade = p.get_last_trade()
        idx.append(p.position_id)
        flags = []
        if p.is_repaired():
            flags.append("R")
        items.append({
            "Flags": flags,
            "Trades": len(p.trades),
            "Opened at": p.opened_at,
            "First trade": first_trade.executed_at,
            "Last trade": last_trade.executed_at,
        })

    return pd.DataFrame(items, index=idx)


