from dataclasses import dataclass
from typing import Optional

from tradeexecutor.strategy.universe_model import UniverseModel
from tradeexecutor.strategy.runner import StrategyRunner
from tradingstrategy.timebucket import TimeBucket


@dataclass
class StrategyExecutionDescription:
    """Describe how a strategy will be execuetd.

    -universe_model: What currencies, candles, etc. to use and how to refresh this
    -runner: Alpha model, communicating with the external environment, executing trades

    This data class is returned from the strategy_factory through :py:func:`tradeexecutor.strategy.bootstrap.import_strategy_file`.
    """

    #: How to refresh the trading universe for the each tick
    universe_model: UniverseModel

    #: What kind of a strategy runner this strategy is using
    runner: StrategyRunner

    # TODO: Deprecate?
    #: What candles this strategy uses: 1d, 1h, etc.
    time_bucket: Optional[TimeBucket] = None

