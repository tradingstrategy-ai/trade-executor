"""Multipair strategy analyses.

Designed for strategies trading > 5 assets.
"""
import numpy as np
import pandas as pd
from IPython.display import HTML

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.state import State
from tradingstrategy.utils.format import format_percent_2_decimals
from tradingstrategy.utils.jupyter import make_clickable


def _format_value(v: float) -> str:
    """Format US dollar value, no dollar sign."""
    return f"{v:,.2f}"


def analyse_pair_trades(pair: TradingPairIdentifier, portfolio: Portfolio) -> dict:
    """Write a single analysis row for a specific pair.

    :return:
        Dict with raw value

    """

    positions = [p for p in portfolio.get_all_positions() if p.pair == pair]
    trades = [t for t in portfolio.get_all_trades() if t.pair == pair]

    profits = [p.get_total_profit_percent() for p in positions]
    best = max(profits)
    worst = min(profits)
    mean = float(np.mean(profits))
    median = float(np.median(profits))
    volume = sum([t.get_value() for t in trades])
    total_usd_profit = sum([p.get_total_profit_usd() for p in positions])
    wins = sum([1 for p in positions if p.get_total_profit_usd() >= 0])
    losses = sum([1 for p in positions if p.get_total_profit_usd() < 0])
    take_profits = sum([1 for p in positions if p.is_take_profit()])
    stop_losses = sum([1 for p in positions if p.is_stop_loss()])
    trailing_stop_losses = sum([1 for p in positions if p.is_trailing_stop_loss()])
    total_return = sum(profits)
    volatility = np.std(profits)

    return {
        "Trading pair": pair.get_human_description(describe_type=True),
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
        "Volatility": volatility,
        "Total return %": total_return,
    }


def analyse_multipair(state: State) -> pd.DataFrame:
    """Build an analysis table.

    Create a table where 1 row = 1 trading pair.

    :param state:

    :return:
        Datframe of the results.

        Sorted by the best return.
    """

    pairs = state.portfolio.get_all_traded_pairs()
    rows = [analyse_pair_trades(p, state.portfolio) for p in pairs]
    df = pd.DataFrame(rows)
    return df


def format_multipair_summary(
    df: pd.DataFrame,
    sort_column="Total return %",
    ascending=False,
    format_columns=True
) -> pd.DataFrame:
    """Format the multipair summary table.

    Convert raw numbers to preferred human format.

    :param df:
        Input table.

        See :py:func:`analyse_pair_trades`.

    :param format_columns:
        If True, format columns with clickable links. Provided as option since some users may want to export the tables without html markup.

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
        "Trailing stop losses": str,
        "Volatility": format_percent_2_decimals,
        "Total return %": format_percent_2_decimals,
    }

    for col, format_func in formatters.items():
        df[col] = df[col].apply(format_func)

    if format_columns:
        headings = [
            ("Trading pair", "https://tradingstrategy.ai/glossary/trading-pair"),
            ("Positions", "https://tradingstrategy.ai/glossary/position"),
            ("Trades", "https://tradingstrategy.ai/glossary/swap"),
            ("Total PnL USD", None),
            ("Best", None),
            ("Worst", None),
            ("Avg", None),
            ("Median", None),
            ("Volume", None),
            ("Wins", None),
            ("Losses", None),
            ("Take profits", "https://tradingstrategy.ai/glossary/take-profit"),
            ("Stop losses", "https://tradingstrategy.ai/glossary/stop-loss"),
            ("Trailing stop losses", "https://tradingstrategy.ai/glossary/trailing-stop-loss"),
            ("Volatility", None),
            ("Total return %", "https://tradingstrategy.ai/glossary/aggregate-return")
        ]

        df.columns = [make_clickable(h, url) if url else h for h, url in headings]

        return HTML(df.to_html(escape=False))

    return df
