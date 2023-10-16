"""Integration testings for data availability trigger for live oracle."""
import datetime
import os

import pytest

from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode, unit_test_execution_context
from tradeexecutor.strategy.pandas_trader.decision_trigger import wait_for_universe_data_availability_jsonl, \
    validate_latest_candles
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, load_all_data, load_partial_data
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradeexecutor.utils.timer import timed_task
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket

pytestmark = pytest.mark.skipif(os.environ.get("TRADING_STRATEGY_API_KEY") is None, reason="Set TRADING_STRATEGY_API_KEY environment variable to run this test")


@pytest.fixture()
def execution_context(request) -> ExecutionContext:
    """Setup backtest execution context."""
    return ExecutionContext(mode=ExecutionMode.backtesting, timed_task_context_manager=timed_task)


@pytest.mark.slow_test_group
def test_decision_trigger_ready_data(persistent_test_client):
    """Test that we can immedidately trigger trades for old data.

    We do not need to wait.
    """

    # Moved here to avoid pytest memory leaks
    def _inner():

        client = persistent_test_client

        # Time bucket for our candles
        candle_time_bucket = TimeBucket.d1

        # Which chain we are trading
        chain_id = ChainId.bsc

        # Which exchange we are trading on.
        exchange_slug = "pancakeswap-v2"

        # Which trading pair we are trading
        trading_pair = ("WBNB", "BUSD")

        # Load all datas we can get for our candle time bucket
        dataset = load_all_data(client, candle_time_bucket, execution_context, UniverseOptions())

        # Filter down to the single pair we are interested in
        universe = TradingStrategyUniverse.create_single_pair_universe(
            dataset,
            chain_id,
            exchange_slug,
            trading_pair[0],
            trading_pair[1],
        )

        return universe

    return _inner

    # Memory leak hack
    universe = _inner()

    client = persistent_test_client
    timestamp = datetime.datetime(2023, 1, 1)
    updated_universe_result = wait_for_universe_data_availability_jsonl(
        timestamp,
        client,
        universe,
    )

    assert updated_universe_result.ready_at <=  datetime.datetime.utcnow()
    assert updated_universe_result.poll_cycles == 1

    pair = updated_universe_result.updated_universe.data_universe.pairs.get_single()
    candles = updated_universe_result.updated_universe.data_universe.candles.get_candles_by_pair(pair.pair_id)

    last_possible_timestamp = timestamp -  TimeBucket.d1.to_timedelta()

    validate_latest_candles(
        {pair},
        candles,
        last_possible_timestamp
    )


@pytest.mark.slow_test_group
def test_decision_trigger_multipair(persistent_test_client):
    """Wait for the multipair decision trigger to be ready."""

    # Moved here to avoid pytest memory leaks
    def _inner():

        client = persistent_test_client

        trading_pairs = [
            (ChainId.polygon, "uniswap-v3", "WETH", "USDC", 0.0005), # Ether-USD Coin (PoS) https://tradingstrategy.ai/trading-view/polygon/uniswap-v3/eth-usdc-fee-5
            (ChainId.polygon, "uniswap-v3", "WMATIC", "USDC", 0.0005), # Wrapped Matic-USD Coin (PoS) https://tradingstrategy.ai/trading-view/polygon/uniswap-v3/matic-usdc-fee-5
            (ChainId.polygon, "uniswap-v3", "XSGD", "USDC", 0.0005), # XSGD-USD Coin (PoS) https://tradingstrategy.ai/trading-view/polygon/uniswap-v3/xsgd-usdc-fee-5
        ]

        # Load data for our trading pair whitelist
        dataset = load_partial_data(
            client=client,
            pairs=trading_pairs,
            time_bucket=TimeBucket.d1,
            execution_context=unit_test_execution_context,
            universe_options=UniverseOptions(),
            start_at=datetime.datetime(2023, 1, 1),
            end_at=datetime.datetime(2023, 2, 1),
        )

        # Filter down the dataset to the pairs we specified
        universe = TradingStrategyUniverse.create_multichain_universe_by_pair_descriptions(
            dataset,
            trading_pairs,
            reserve_token_symbol="USDC"
        )

        return universe

    universe = _inner()

    client = persistent_test_client
    timestamp = datetime.datetime(2023, 1, 1)
    updated_universe_result = wait_for_universe_data_availability_jsonl(
        timestamp,
        client,
        universe,
    )

    assert updated_universe_result.ready_at <= datetime.datetime.utcnow()
    assert updated_universe_result.poll_cycles == 1

    for pair in universe.data_universe.pairs.iterate_pairs():
        candles = updated_universe_result.updated_universe.data_universe.candles.get_candles_by_pair(pair.pair_id)

        last_possible_timestamp = timestamp - TimeBucket.d1.to_timedelta()

        validate_latest_candles(
            {pair},
            candles,
            last_possible_timestamp
        )

