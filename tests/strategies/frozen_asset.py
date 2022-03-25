"""A unit test strategy that buys/sells frozen/blacklisted assets."""
import logging
from contextlib import AbstractContextManager
from typing import Dict, Any

import pandas as pd

from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

from tradeexecutor.ethereum.uniswap_v2_execution import UniswapV2ExecutionModel
from tradeexecutor.state.revaluation import RevaluationMethod
from tradeexecutor.state.state import State
from tradeexecutor.state.sync import SyncMethod
from tradeexecutor.strategy.approval import ApprovalModel
from tradeexecutor.strategy.description import StrategyExecutionDescription
from tradeexecutor.strategy.pricing_model import PricingModelFactory
from tradeexecutor.strategy.qstrader.alpha_model import AlphaModel
from tradeexecutor.strategy.qstrader.runner import QSTraderRunner
from tradeexecutor.strategy.universe_model import StaticUniverseModel


# Cannot use Python __name__ here because the module is dynamically loaded
logging = logging.getLogger("frozen_asset")


class BadAlpha(AlphaModel):

    def __call__(self, ts: pd.Timestamp, universe: Universe, state: State, debug_details: Dict) -> Dict[Any, float]:
        pancake = universe.get_single_exchange()
        wbnb_busd = universe.pairs.get_one_pair_from_pandas_universe(pancake.exchange_id, "WBNB", "BUSD")
        bit_busd = universe.pairs.get_one_pair_from_pandas_universe(pancake.exchange_id, "BIT", "BUSD")

        assert wbnb_busd
        assert bit_busd

        cycle = debug_details["cycle"]

        if cycle == 1:
            # Buy 50%/50%
            return {
                wbnb_busd.pair_id: 0.5,
                wbnb_busd.pair_id: 0.5,
            }
        elif cycle == 2:
            # Sell all
            return {
                wbnb_busd.pair_id: 0.5,
                wbnb_busd.pair_id: 0.5,
            }
        elif cycle == 3:
            assert
            return {wbnb_busd.pair_id: 0.5}
        else:
            raise NotImplementedError()


def strategy_factory(
        *ignore,
        execution_model: UniswapV2ExecutionModel,
        sync_method: SyncMethod,
        pricing_model_factory: PricingModelFactory,
        revaluation_method: RevaluationMethod,
        client,
        timed_task_context_manager: AbstractContextManager,
        approval_model: ApprovalModel,
        universe_model: StaticUniverseModel,
        cash_buffer: float,
        **kwargs) -> StrategyExecutionDescription:

    if ignore:
        # https://www.python.org/dev/peps/pep-3102/
        raise TypeError("Only keyword arguments accepted")

    # Use static universe passed from the tests
    assert isinstance(universe_model, StaticUniverseModel)

    runner = QSTraderRunner(
        alpha_model=BadAlpha(),
        timed_task_context_manager=timed_task_context_manager,
        execution_model=execution_model,
        approval_model=approval_model,
        revaluation_method=revaluation_method,
        sync_method=sync_method,
        pricing_model_factory=pricing_model_factory,
        cash_buffer=cash_buffer,
    )

    return StrategyExecutionDescription(
        time_bucket=TimeBucket.d1,
        universe_model=universe_model,
        runner=runner,
    )


__all__ = [strategy_factory]
