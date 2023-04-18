"""Analyser trading fees."""
import numpy as np
import pandas as pd

from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution


def analyse_trading_fees(
        state: State,
) -> pd.DataFrame:
    """Create a table containing trading fees for every trade.

    :return:
        DataFrame with the following columns.

        - strategy_cycle_at

        - pair_ticker

        - trade_type ("buy" or "sell")

        - pair_id

        - value_usd

        - fee_tier

        - fees_paid_usd

        - fees_paid_pct

        - fees_estimated_usd

        - fees_estimated_pct

    """

    rows = []

    trade: TradeExecution
    for trade in state.portfolio.get_all_trades():
        pair = trade.pair
        type = "buy" if trade.is_buy() else "sell"
        estimated_fees = trade.lp_fees_estimated or 0
        row = {
            "strategy_cycle_at": trade.strategy_cycle_at,
            "pair_id": pair.internal_id,
            "pair_ticker": pair.get_ticker(),
            "trade_type": type,
            "value_usd": trade.get_value(),
            "fee_tier": trade.fee_tier,
            "fees_paid_usd": trade.get_fees_paid(),
            "fees_paid_pct": trade.get_fees_paid() / trade.get_value(),
            "fees_estimated_usd": estimated_fees,
            "fees_estimated_pct": estimated_fees / trade.get_value(),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def create_pair_trading_fee_summary_table(analysis: pd.DataFrame) -> pd.DataFrame:
    """Creates a summary table that break down trading fees per pair and trade type.

    :param analysis:
        DataFrame of all trades analysed.

        Output from :py:func:`analyse_trading_fees`.

    """

    # https://pandas.pydata.org/docs/reference/api/pandas.pivot_table.html
    agg_funcs = {
        "fee_tier": [np.max, np.min],
        "value_usd": [np.sum, np.max, np.min, np.mean],
        "fees_paid_usd": [np.sum, np.max, np.min, np.mean],
        "fees_paid_pct": [np.max, np.min, np.mean],
        "fees_estimated_usd": [np.max, np.min, np.mean],
        "fees_estimated_pct": [np.max, np.min, np.mean],

    }

    return pd.pivot_table(
        analysis,
        index=['pair_ticker', 'trade_type'],
        values=['fee_tier', 'value_usd', 'fees_paid_usd', 'fees_paid_pct', 'fees_estimated_usd', 'fees_estimated_pct'],
        aggfunc=agg_funcs)
