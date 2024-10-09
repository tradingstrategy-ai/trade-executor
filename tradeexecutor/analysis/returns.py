"""Daily returns and related calculations."""
import pandas as pd

from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


def calculate_returns(
    strategy_universe: TradingStrategyUniverse,
    column="close",
) -> pd.Series:
    """Calculate returns to every asset in the trading universe.

    Example:

    .. code-block: python

        import plotly.express as px

        from tradeexecutor.analysis.returns import calculate_returns

        returns_series = calculate_returns(strategy_universe)

        pair = strategy_universe.get_pair_by_human_description([
            ChainId.centralised_exchange,
            "binance",
            "ETH",
            "USDT"
        ])

        pair_returns = returns_series.loc[pair.internal_id]
        start = pair_returns.index[0]
        end = pair_returns.index[-1]

        display(pair_returns.head(5))

        fig = px.histogram(
            pair_returns,
            title=f"Returns for {pair.base.token_symbol}, {start} - {end}",
        )
        display(fig)

    :return:
        DataFrame of price candle returns, with MultiIndex (pair_id, timestamp).

        For the first time frame, NaN is set as returns.

        DataFrame will also contain the original OHLCV candle data with columns
        "open", "close", etc.
    """

    candles_raw_df = strategy_universe.data_universe.candles.df
    df = candles_raw_df.set_index(["pair_id", "timestamp"], drop=False)
    series = df.groupby(level='pair_id')[column].pct_change()
    return series
