import pandas as pd
import datetime
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.lending import LendingProtocolType
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, load_partial_data
from tradingstrategy.client import Client
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.universe_model import UniverseOptions
import numpy as np

def get_indices_of_uneven_intervals(df: pd.DataFrame | pd.Series) -> bool:
    """Checks if a time series contains perfectly evenly spaced time intervals with no gaps.

    :param df: Pandas dataframe or series
    :return: True if time series is perfectly evenly spaced, False otherwise
    """
    assert type(df.index) == pd.DatetimeIndex, "Index must be a DatetimeIndex"

    numeric_representation = df.index.astype(np.int64)

    differences = np.diff(numeric_representation)

    not_equal_to_first = differences != differences[0]

    return np.where(not_equal_to_first)[0]


def is_missing_data(df: pd.DataFrame | pd.Series) -> bool:
    """Checks if a time series contains perfectly evenly spaced time intervals with no gaps.
    
    :param df: Pandas dataframe or series
    :return: False if time series is perfectly evenly spaced, True otherwise
    """
    return len(get_indices_of_uneven_intervals(df)) > 0


CANDLE_TIME_BUCKET = TimeBucket.h1

TRADING_PAIR = (ChainId.polygon, "uniswap-v3", "WETH", "USDC", 0.0005)

LENDING_RESERVES = [
    (ChainId.polygon, LendingProtocolType.aave_v3, "WETH"),
    (ChainId.polygon, LendingProtocolType.aave_v3, "USDC"),
]

STOP_LOSS_TIME_BUCKET = TimeBucket.m15

START_AT = datetime.datetime(2022, 10, 1)
END_AT = datetime.datetime(2023, 11, 15)


client = Client.create_jupyter_client()

universe_options = UniverseOptions(
    start_at=START_AT - datetime.timedelta(days=50),
    end_at=END_AT,
)

execution_context = ExecutionContext(mode=ExecutionMode.data_preload)

dataset = load_partial_data(
    client,
    execution_context=execution_context,
    time_bucket=CANDLE_TIME_BUCKET,
    pairs=[TRADING_PAIR],
    universe_options=universe_options,
    start_at=universe_options.start_at,
    end_at=universe_options.end_at,
    lending_reserves=LENDING_RESERVES,  # NEW
    stop_loss_time_bucket=STOP_LOSS_TIME_BUCKET,
)

strategy_universe = TradingStrategyUniverse.create_single_pair_universe(dataset)

df = strategy_universe.universe.lending_candles.variable_borrow_apr.df

# np.where(df['timestamp'] == '2023-01-10 23:00:00')

def test_missing_lending_data():
    problem_df = df[7294:7300]

    assert not is_missing_data(problem_df)
           