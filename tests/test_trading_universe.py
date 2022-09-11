"""Trading Universe creation tests."""
import os

import pandas as pd
import pytest

from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.trading_strategy_universe import load_all_data, \
    TradingStrategyUniverse
from tradeexecutor.utils.timer import timed_task
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket


# https://docs.pytest.org/en/latest/how-to/skipping.html#skip-all-test-functions-of-a-class-or-module
pytestmark = pytest.mark.skipif(os.environ.get("TRADING_STRATEGY_API_KEY") is None, reason="Set TRADING_STRATEGY_API_KEY environment variable to run this test")


def test_create_multipair_universe(persistent_test_client):
    """Create a trading universe with multiple pairs."""

    client = persistent_test_client
    candle_time_bucket = TimeBucket.d7

    execution_context = ExecutionContext(
        ExecutionMode.unit_testing_trading,
        timed_task,
    )

    dataset = load_all_data(
        client,
        candle_time_bucket,
        execution_context)

    factory_router_map = {
        # PancakeSwap
        "0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73": ("0x10ED43C718714eb63d5aA57B78B54704E256024E",
                                                       "0x00fb7f630766e6a796048ea87d01acd3068e8ff67d078148a3fa3f4a84f69bd5")
    }

    universe = TradingStrategyUniverse.create_multipair_universe(
        dataset,
        [ChainId.bsc],
        ["pancakeswap-v2"],
        # WBNB, BUSD
        ["0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c", "0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56"],
        # BUSD
        "0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56",
        factory_router_map,
    )

    assert universe.universe.pairs.get_count() > 1000
    assert universe.reserve_assets[0].address.lower() == "0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56".lower()
    range = universe.universe.candles.get_timestamp_range()
    assert range[0] < pd.Timestamp('2022-01-01 00:00:00')
    assert range[1] > pd.Timestamp('2022-01-01 00:00:00')
