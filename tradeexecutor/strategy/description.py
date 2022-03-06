from dataclasses import dataclass
from typing import List

from tradeexecutor.state.revaluation import RevaluationMethod
from tradeexecutor.state.state import AssetIdentifier
from tradeexecutor.state.sync import SyncMethod
from tradeexecutor.strategy.pricing_method import PricingMethod
from tradeexecutor.strategy.universe_constructor import UniverseConstructionMethod

from tradeexecutor.strategy.runner import StrategyRunner
from tradeexecutor.strategy.universe import UniverseModel
from tradingstrategy.timebucket import TimeBucket


@dataclass
class StrategyRunDescription:
    """Describe the strategy for the runner.

    Tell details like what currencies, candles, etc. to use.

    This data class is returned from the strategy factory.
    """

    #: What candles this strategy uses: 1d, 1h, etc.
    time_bucket: TimeBucket

    #: How do we estimate prices before buy
    pricing_method: PricingMethod

    #: How do revalue our portfolio at the start of a strategy tick
    revaluation_method: RevaluationMethod

    #: What reserve assets this straegy uses e.g. BUSD
    reserve_asset: List[AssetIdentifier]

    #: How to refresh the trading universe for the each tick
    universe_constructor: UniverseModel

    #: What kind of a strategy runner this strategy is using
    runner: StrategyRunner
