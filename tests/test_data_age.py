"""Check we correctly handle fresh and expired data."""
import os
import datetime

import pytest

from tradeexecutor.strategy.execution_context import ExecutionMode, unit_test_execution_context, unit_test_trading_execution_context
from tradeexecutor.strategy.universe_model import StrategyExecutionUniverse, DataTooOld, UniverseOptions
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client

from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverseModel, load_partial_data, TradingStrategyUniverse
from tradeexecutor.utils.timer import timed_task
from tradingstrategy.timebucket import TimeBucket


# https://docs.pytest.org/en/latest/how-to/skipping.html#skip-all-test-functions-of-a-class-or-module
pytestmark = pytest.mark.skipif(os.environ.get("TRADING_STRATEGY_API_KEY") is None, reason="Set TRADING_STRATEGY_API_KEY environment variable to run this test")


class DataAgeTestUniverseModel(TradingStrategyUniverseModel):
    """Load 6 months data."""

    def construct_universe(self, ts: datetime.datetime, mode: ExecutionMode) -> StrategyExecutionUniverse:
        assert isinstance(mode, ExecutionMode)

        client = self.client

        dataset = load_partial_data(
            client,
            unit_test_trading_execution_context,
            TimeBucket.d30,
            pairs=((ChainId.ethereum, "uniswap-v3", "WETH", "USDC", 0.0005),),
            universe_options=UniverseOptions(history_period=datetime.timedelta(days=6*30)),
        )

        # Pair index takes long time to construct and is not needed for the test
        universe = TradingStrategyUniverse.create_from_dataset(dataset)
        return universe


@pytest.mark.slow_test_group
def test_data_fresh(persistent_test_client):
    """Fresh data passes our data check."""
    # d1 data is used by other tests and cached
    universe_model = DataAgeTestUniverseModel(persistent_test_client, timed_task)
    best_before_duration = datetime.timedelta(weeks=1000)  # Our unit test is good for next 1000 years
    ts = datetime.datetime.utcnow()
    universe = universe_model.construct_universe(ts, ExecutionMode.unit_testing_trading)
    universe_model.check_data_age(ts, universe, best_before_duration)


@pytest.mark.slow_test_group
def test_data_aged(persistent_test_client):
    """Aged data raises an exception."""
    universe_model = DataAgeTestUniverseModel(persistent_test_client, timed_task)
    ts = datetime.datetime.utcnow()
    best_before_duration = datetime.timedelta(seconds=1)  # We can never have one second old data
    universe = universe_model.construct_universe(ts, ExecutionMode.backtesting)
    with pytest.raises(DataTooOld):
        universe_model.check_data_age(ts, universe, best_before_duration)






