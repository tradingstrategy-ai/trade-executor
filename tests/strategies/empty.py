import datetime

from tradeexecutor.state.state import State
from tradeexecutor.strategy.runner import StrategyRunner, Dataset


class EmptyStrategyRunner(StrategyRunner):
    """A strategy runner that does nothing.

    This is a dummy strategy runner and used in the unit testing.
    """

    def load_data(self, time_frame, client, lookback):
        return None

    def get_strategy_time_frame(self):
        return None

    def construct_universe(self, dataset: Dataset):
        return None

    def preflight_check(self, client, universe, now_):
        pass

    def on_clock(self, clock: datetime.datetime, universe, state: State):
        # Always return "no trades"
        return []


def strategy_executor_factory(**kwargs):
    strategy_runner = EmptyStrategyRunner(**kwargs)
    return strategy_runner