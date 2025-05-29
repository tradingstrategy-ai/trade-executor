"""Blacklist helpers"""
import pandas as pd

from tradeexecutor.state.state import State
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradingstrategy.chain import ChainId


def display_blacklist(state: State, strategy_universe: TradingStrategyUniverse) -> pd.DataFrame:
    """Display the blacklisted assets in a DataFrame."""

    rows = []
    all_trades = list(state.portfolio.get_all_trades())
    all_trades.reverse()
    for asset in state.blacklisted_assets:

        last_trade = None
        for trade in all_trades:
            if trade.pair.base == asset:
                last_trade = trade
                pair = trade.pair
                break

        reason = state.blacklist_reason.get(asset.get_identifier())

        entry = {
            "Chain": ChainId(asset.chain_id).get_name(),
            "Token": asset.token_symbol,
            "Address": asset.address,
            "Last trade id": last_trade.trade_id if last_trade else "-",
            "Last trade at": last_trade.executed_at if last_trade else "-",
            "Reason": reason or "-",
            "Risk score": pair.get_risk_score() if pair else "-",
        }
        rows.append(entry)

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.set_index("Address")
    return df

