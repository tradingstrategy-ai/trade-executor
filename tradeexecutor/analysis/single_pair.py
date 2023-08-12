"""Single trading pair analysis"""
import pandas as pd

from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State


def expand_entries_and_exists(
    state: State,
) -> pd.DataFrame:
    """Write out a table containing entries and exists of every taken position.

    - Made for single pair strategies

    - Entry and exit are usually done using the close value
      of the previous candle

    - Assume each position contains only one entry and one exit trade

    :return:
        DataFrame indexed by position entries
    """

    items = []
    idx = []

    p: TradingPosition
    for p in state.portfolio.get_all_positions():

        first_trade = p.get_first_trade()
        last_trade = p.get_last_trade()

        # Open position at the end
        if first_trade == last_trade:
            last_trade = None

        volume = sum(t.get_executed_value() for t in p.trades.values())
        fee = sum(t.lp_fees_paid or 0 for t in p.trades.values())

        idx.append(first_trade.strategy_cycle_at)
        items.append({
            "Entry": first_trade.strategy_cycle_at,
            "Entry mid price": first_trade.price_structure.mid_price,
            "Exit": last_trade.strategy_cycle_at if last_trade else None,
            "Exit mid price": last_trade.price_structure.mid_price if last_trade else None,
            "PnL": p.get_total_profit_usd(),
            "Volume": volume,
            "LP fee": fee,
        })

    df = pd.DataFrame(items, index=idx)
    df = df.fillna("")
    df = df.replace({pd.NaT: ""})
    return df
