from dataclasses import dataclass

from tradeexecutor.strategy.universe_model import UniverseModel
from tradeexecutor.strategy.runner import StrategyRunner
from tradingstrategy.timebucket import TimeBucket


@dataclass
class StrategyRunDescription:
    """Describe the strategy for the runner.

    Tell details like what currencies, candles, etc. to use.

    This data class is returned from the strategy factory.
    """

    #: What candles this strategy uses: 1d, 1h, etc.
    time_bucket: TimeBucket

    #: How to refresh the trading universe for the each tick
    universe_constructor: UniverseModel

    #: What kind of a strategy runner this strategy is using
    runner: StrategyRunner
