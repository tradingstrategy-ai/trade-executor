"""Check we clip history correctly."""
import datetime

from tradeexecutor.strategy.execution_context import ExecutionMode, ExecutionContext, notebook_execution_context
from tradeexecutor.strategy.trading_strategy_universe import load_pair_data_for_single_exchange
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket


def test_clip_dataset(persistent_test_client):
    client = persistent_test_client

    TRADING_PAIR = (ChainId.polygon, "uniswap-v3", "WETH", "USDC", 0.0005)

    CANDLE_TIME_BUCKET = TimeBucket.m15

    START_AT = datetime.datetime(2022, 9, 1)

    # Backtest range
    END_AT = datetime.datetime(2023, 4, 1)

    execution_context = notebook_execution_context
    universe_options = UniverseOptions()

    # Fetch backtesting datasets from the server
    dataset = load_pair_data_for_single_exchange(
        client,
        time_bucket=CANDLE_TIME_BUCKET,
        pair_tickers=[TRADING_PAIR],
        execution_context=execution_context,
        universe_options=universe_options,
        start_time=START_AT,
        end_time=END_AT,
        stop_loss_time_bucket=CANDLE_TIME_BUCKET,
    )

    assert dataset.backtest_stop_loss_candles is not None, "No stop loss candles provided"
    assert dataset.backtest_stop_loss_time_bucket == CANDLE_TIME_BUCKET, "Stop loss time bucket is incorrect"
    assert dataset.candles.iloc[0]["timestamp"] >= START_AT
    assert dataset.candles.iloc[-1]["timestamp"] <= END_AT
