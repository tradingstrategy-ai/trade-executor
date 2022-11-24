"""Strategy execution description.

Describe core parameters for the strategy execution for the main loop.
"""

from dataclasses import dataclass
from typing import Optional

from tradingstrategy.chain import ChainId

from tradeexecutor.strategy.cycle import CycleDuration
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

    #: As read from the strategy
    trading_strategy_engine_version: Optional[str] = None

    #: How long is the strategy execution cycle
    cycle_duration: Optional[CycleDuration] = None

    # TODO: Deprecate?
    #: What candles this strategy uses: 1d, 1h, etc.
    time_bucket: Optional[TimeBucket] = None

    #: Blockchain id for this strategy.
    #:
    #: Necessary for single chain strategies to know
    #: on which chain we operate.
    #:
    chain_id: Optional[ChainId] = None

