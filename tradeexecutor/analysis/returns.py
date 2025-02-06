"""Daily returns and related calculations."""
import json
from dataclasses import asdict
from os import makedirs
from pathlib import Path

import pandas as pd

from tradeexecutor.state.state import State
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.visual.equity_curve import calculate_daily_returns
from tradingstrategy.client import Client


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



def get_default_returns_folder(client: Client) -> Path:
    """Where do we save return series."""
    folder = Path(client.transport.cache_path) / "returns"
    return folder


def save_daily_returns(
    state: State,
    client: Client = None,
    folder: Path = None,
    verbose=True,
    strategy_universe: TradingStrategyUniverse | None = None,
):
    """Save daily returns as a parquet file.

    - To be opened in another notebook for a benchmark

    - DataFrame contains one column "returns"

    - DataFrame.attrs will contain metadata about the backtest run, like trading pairs

    :param state:
        Backtest state.

    :param client:
        Use to resolve the default save folder

    :param folder:
        Save in this folder.

        The resulting name is "daily-returns-{state.name}.parquet".

        If not given use client cache path / returns.

    :param strategy_universe:
        Store pair metadata

    :param verbose:
        Print out the saved filename

    :return:
        Saved DataFrame.
    """

    assert (folder or client), "Give either folder or client"

    if folder is None:
        folder = get_default_returns_folder(client)

    assert state.name

    # Slugify the filename
    name = state.name.replace("_", "-").replace(" ", "-")
    path = folder / f"daily-returns-{name}.parquet"
    makedirs(folder, exist_ok=True)
    returns = calculate_daily_returns(state)
    assert returns is not None, "Got empty returns"

    df = pd.DataFrame({
        "returns": returns,
    })

    df.to_parquet(path)

    # Add some metadata
    df.attrs["name"] = state.name
    df.attrs["trading_start"] = state.get_trading_time_range()[0]
    df.attrs["trading_end"] = state.get_trading_time_range()[0]

    if strategy_universe is not None:
        pairs = [p.to_dict() for p in strategy_universe.iterate_pairs()]
        df.attrs["pairs"] = json.dumps(pairs)

    if verbose:
        print(f"Saved {path}, {path.stat().st_size:,} bytes")
