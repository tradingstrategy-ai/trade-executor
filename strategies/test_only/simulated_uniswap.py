"""Example strategy that trades on Uniswap v2 using the local EthereumTester EVM deployment."""
import logging
from contextlib import AbstractContextManager
from typing import Dict, Any

import pandas as pd

from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_execution_v0 import UniswapV2ExecutionModelVersion0
from tradeexecutor.state.state import State
from tradeexecutor.strategy.sync_model import SyncMethodV0
from tradeexecutor.strategy.approval import ApprovalModel
from tradeexecutor.strategy.description import StrategyExecutionDescription
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.pricing_model import PricingModelFactory
from tradeexecutor.strategy.qstrader.alpha_model import AlphaModel
from tradeexecutor.strategy.qstrader.runner import QSTraderRunner
from tradeexecutor.strategy.routing import RoutingModel
from tradeexecutor.strategy.universe_model import StaticUniverseModel


# Cannot use Python __name__ here because the module is dynamically loaded
from tradeexecutor.strategy.valuation import ValuationModelFactory

logging = logging.getLogger("simulated_uniswap")


class SomeTestBuysAlphaModel(AlphaModel):
    """A test alpha model that switches between ETH/AAVE positions."""

    def calculate_day_number(self, ts: pd.Timestamp):
        """Get how many days has past since 1-1-1970"""
        return (ts - pd.Timestamp("1970-1-1")).days

    def __call__(self, ts: pd.Timestamp, universe: Universe, state: State, debug_details: Dict) -> Dict[Any, float]:
        """
        Produce the dictionary of scalar signals for
        each of the Asset instances within the Universe.

        :param ts: Candle timestamp iterator

        :return: Dict(pair_id, alpha signal)
        """

        assert debug_details
        assert isinstance(universe, Universe)
        assert isinstance(state, State)

        # Because this is a test strategy, we assume we have fixed 3 assets
        assert len(universe.exchanges) == 1
        uniswap = universe.get_single_exchange()
        assert universe.pairs.get_count() == 2

        weth_usdc = universe.pairs.get_one_pair_from_pandas_universe(uniswap.exchange_id, "WETH", "USDC")
        aave_usdc = universe.pairs.get_one_pair_from_pandas_universe(uniswap.exchange_id, "AAVE", "USDC")

        assert weth_usdc
        assert aave_usdc

        # The test strategy rebalanced between three different positions daily
        day = self.calculate_day_number(ts)
        day_kind = day % 3
        # Expose our daily routine to unit tests
        debug_details["day_kind"] = day_kind

        # Switch between modes
        if day_kind == 0:
            # On 1/3 days buy 50%/50% ETH and AAVE
            return {
                weth_usdc.pair_id: 0.3,
                aave_usdc.pair_id: 0.3,
            }
        elif day_kind == 1:
            # On 1/3 days buy 100% ETH
            return {
                weth_usdc.pair_id: 1,
            }
        else:
            # On 1/3 days buy 100% AAVE
            return {
                aave_usdc.pair_id: 1,
            }


def strategy_factory(
        *ignore,
        execution_model: UniswapV2ExecutionModelVersion0,
        sync_method: SyncMethodV0,
        pricing_model_factory: PricingModelFactory,
        valuation_model_factory: ValuationModelFactory,
        client,
        timed_task_context_manager: AbstractContextManager,
        approval_model: ApprovalModel,
        universe_model: StaticUniverseModel,
        routing_model: RoutingModel,
        cash_buffer: float,
        **kwargs) -> StrategyExecutionDescription:

    if ignore:
        # https://www.python.org/dev/peps/pep-3102/
        raise TypeError("Only keyword arguments accepted")

    # Use static universe passed from the tests
    assert isinstance(universe_model, StaticUniverseModel)

    execution_context = kwargs.get("execution_context", ExecutionContext(ExecutionMode.unit_testing_trading))

    runner = QSTraderRunner(
        alpha_model=SomeTestBuysAlphaModel(),
        timed_task_context_manager=timed_task_context_manager,
        execution_model=execution_model,
        approval_model=approval_model,
        valuation_model_factory=valuation_model_factory,
        sync_method=sync_method,
        pricing_model_factory=pricing_model_factory,
        cash_buffer=cash_buffer,
        routing_model=routing_model,
        execution_context=execution_context,
    )

    return StrategyExecutionDescription(
        time_bucket=TimeBucket.d1,
        universe_model=universe_model,
        runner=runner,
    )


__all__ = [strategy_factory]
