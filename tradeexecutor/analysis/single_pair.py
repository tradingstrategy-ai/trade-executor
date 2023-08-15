"""Single trading pair analysis"""
from _decimal import Decimal

import pandas as pd

from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State


def expand_entries_and_exits(
    state: State,
    token_quantizer=Decimal("0.000001"),
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

        symbol = p.pair.base.token_symbol

        first_trade = p.get_first_trade()
        last_trade = p.get_last_trade()

        # Open position at the end
        if first_trade == last_trade:
            last_trade = None

        volume = sum(t.get_volume() for t in p.trades.values())
        volume_token = sum(abs(t.get_position_quantity()) for t in p.trades.values())
        fee = sum(t.lp_fees_paid or 0 for t in p.trades.values())

        idx.append(first_trade.strategy_cycle_at)
        items.append({
            "Entry": first_trade.strategy_cycle_at,
            "Entry mid price": first_trade.price_structure.mid_price,
            "Exit": last_trade.strategy_cycle_at if last_trade else None,
            "Exit mid price": last_trade.price_structure.mid_price if last_trade else None,
            "PnL": p.get_total_profit_usd(),
            "Vol USD": volume,
            f"Vol {symbol}": volume_token.quantize(token_quantizer),
            "LP fee USD": fee,
            "Portfolio size": p.portfolio_value_at_open,
        })

    df = pd.DataFrame(items, index=idx)
    df = df.fillna("")
    df = df.replace({pd.NaT: ""})
    return df
