"""Daily returns and related calculations."""
import pandas as pd

from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


def calculate_returns(
    strategy_universe: TradingStrategyUniverse,
) -> pd.DataFrame:
    """Calculate returns to every asset in the trading universe.

    :return:
        DataFrame of price candle returns, with MultiIndex (pair_id, timestamp).

        For the first time frame, NaN is set as returns.

        DataFrame will also contain the original OHLCV candle data with columns
        "open", "close", etc.
    """

    candles_raw_df = strategy_universe.data_universe.candles.df
    df = candles_raw_df.set_index(["pair_id", "timestamp"], drop=False)
    df['returns'] = df.groupby(level='pair_id')['close'].pct_change()
    return df
