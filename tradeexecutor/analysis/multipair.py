"""Multipair strategy analyses.

Designed for strategies trading > 5 assets.
"""
import numpy as np
import pandas as pd

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.state import State
from tradingstrategy.utils.format import format_percent, format_value, format_percent_2_decimals



def _format_value(v: float) -> str:
    """Format US dollar value, no dollar sign.
    """
    return f"{v:,.2f}"


def analyse_pair_trades(pair: TradingPairIdentifier, portfolio: Portfolio) -> dict:
    """Write a single analysis row for a specific pair.

    :return:
        Dict with raw value

    """

    positions = [p for p in portfolio.get_all_positions() if p.pair == pair]
    trades = [t for t in portfolio.get_all_trades() if t.pair == pair]

    profits = [p.get_total_profit_percent() for p in positions]
    best= max(profits)
    worst = min(profits)
    mean = float(np.mean(profits))
    median = float(np.median(profits))
    volume = sum([t.get_value() for t in trades])
    total_usd_profit = sum([p.get_total_profit_usd() for p in positions])
    wins = sum([1 for p in positions if p.get_total_profit_usd() >= 0])
    losses = sum([1 for p in positions if p.get_total_profit_usd() >= 0])
    take_profits = sum([1 for p in positions if p.is_take_profit()])
    stop_losses = sum([1 for p in positions if p.is_stop_loss()])
    trailing_stop_losses = sum([1 for p in positions if p.is_trailing_stop_loss()])

    return {
        "Trading pair": pair.get_human_description(),
        "Positions": len(positions),
        "Trades": len(trades),
        "Total PnL USD": total_usd_profit,
        "Best": best,
        "Worst": worst,
        "Avg": mean,
        "Median": median,
        "Volume": volume,
        "Wins": wins,
        "Losses": losses,
        "Take profits": take_profits,
        "Stop losses": stop_losses,
        "Trailing stop losses": trailing_stop_losses,
    }


def analyse_multipair(state: State) -> pd.DataFrame:
    """Build an analysis table.

    Create a table where 1 row = 1 trading pair.

    :param state:

    :return:
        Raw dataframe.
    """

    pairs = state.portfolio.get_all_traded_pairs()
    rows = [analyse_pair_trades(p, state.portfolio) for p in pairs]
    return pd.DataFrame(rows)


def format_multipair_summary(df: pd.DataFrame, sort_column="Total PnL USD", ascending=False) -> pd.DataFrame:
    """Format the multipair summary table.

    Convert raw numbers to preferred human format.

    :param df:
        Input table.

        See :py:func:`analyse_pair_trades`.

    :return:
        Dataframe with formatted values for each trading pair.

        If there are no trades return empty dataframe.
    """

    if len(df) == 0:
        return pd.DataFrame()

    df = df.sort_values(by=[sort_column], ascending=ascending)

    formatters = {
        "Trading pair": str,
        "Positions": str,
        "Trades": str,
        "Total PnL USD": _format_value,
        "Best": format_percent_2_decimals,
        "Worst": format_percent_2_decimals,
        "Avg": format_percent_2_decimals,
        "Median": format_percent_2_decimals,
        "Volume": _format_value,
        "Wins": str,
        "Losses": str,
        "Take profits": str,
        "Stop losses": str,
    }

    for col, format_func in formatters.items():
        df[col] = df[col].apply(format_func)

    return df

