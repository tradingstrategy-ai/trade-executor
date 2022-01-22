import datetime

from tradeexecutor.state.state import State
from tradeexecutor.strategy.runner import StrategyRunner


class DummyStrategyRunner(StrategyRunner):
    """A strategy exercised by unit tests."""

    def on_clock(self, clock: datetime.datetime, state: State):
        return []


strategy_runner = DummyStrategyRunner()