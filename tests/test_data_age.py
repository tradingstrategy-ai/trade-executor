"""Check we correctly handle fresh and expired data."""
import os
import datetime

import pytest

from tradeexecutor.strategy.execution_context import ExecutionMode
from tradeexecutor.strategy.universe_model import StrategyExecutionUniverse, DataTooOld
from tradingstrategy.client import Client

from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverseModel
from tradeexecutor.utils.timer import timed_task
from tradingstrategy.timebucket import TimeBucket


# https://docs.pytest.org/en/latest/how-to/skipping.html#skip-all-test-functions-of-a-class-or-module
pytestmark = pytest.mark.skipif(os.environ.get("TRADING_STRATEGY_API_KEY") is None, reason="Set TRADING_STRATEGY_API_KEY environment variable to run this test")


class DataAgeTestUniverseModel(TradingStrategyUniverseModel):

    def construct_universe(self, ts: datetime.datetime, mode: ExecutionMode) -> StrategyExecutionUniverse:
        assert isinstance(mode, ExecutionMode)
        # d1 data is used by other tests and cached
        dataset = self.load_data(TimeBucket.d1, mode)
        # Pair index takes long time to construct and is not needed for the test
        universe = TradingStrategyUniverseModel.create_from_dataset(dataset, [], [], pairs_index=False)
        return universe


@pytest.fixture()
def universe_model(persistent_test_client: Client) -> DataAgeTestUniverseModel:
    return DataAgeTestUniverseModel(persistent_test_client, timed_task)


def test_data_fresh(universe_model: DataAgeTestUniverseModel):
    """Fresh data passes our data check."""
    # d1 data is used by other tests and cached
    best_before_duration = datetime.timedelta(weeks=1000)  # Our unit test is good for next 1000 years
    ts = datetime.datetime.utcnow()
    universe = universe_model.construct_universe(ts, ExecutionMode.real_trading)
    universe_model.check_data_age(ts, universe, best_before_duration)


def test_data_aged(universe_model: DataAgeTestUniverseModel):
    """Aged data raises an exception."""
    ts = datetime.datetime.utcnow()
    best_before_duration = datetime.timedelta(seconds=1)  # We can never have one second old data
    universe = universe_model.construct_universe(ts, ExecutionMode.backtesting)
    with pytest.raises(DataTooOld):
        universe_model.check_data_age(ts, universe, best_before_duration)






