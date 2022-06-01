"""A unit test strategy that buys/sells frozen/blacklisted assets."""
import logging
from contextlib import AbstractContextManager
from typing import Dict, Any

import pandas as pd

from tradeexecutor.ethereum.uniswap_v2_routing import UniswapV2SimpleRoutingModel
from tradeexecutor.strategy.trading_strategy_universe import translate_trading_pair

from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

from tradeexecutor.ethereum.uniswap_v2_execution_v0 import UniswapV2ExecutionModelVersion0
from tradeexecutor.state.state import State
from tradeexecutor.state.sync import SyncMethod
from tradeexecutor.strategy.approval import ApprovalModel
from tradeexecutor.strategy.description import StrategyExecutionDescription
from tradeexecutor.strategy.pricing_model import PricingModelFactory
from tradeexecutor.strategy.qstrader.alpha_model import AlphaModel
from tradeexecutor.strategy.qstrader.runner import QSTraderRunner
from tradeexecutor.strategy.universe_model import StaticUniverseModel


# Cannot use Python __name__ here because the module is dynamically loaded
from tradeexecutor.strategy.valuation import ValuationModelFactory

logging = logging.getLogger("frozen_asset")


#
# Trade routing info
#

# Keep everything internally in BUSD
reserve_token_address = "0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56".lower()

# Allowed exchanges as factory -> router pairs,
# by their smart contract addresses
factory_router_map = {
    "0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73": ("0x10ED43C718714eb63d5aA57B78B54704E256024E", "0x00fb7f630766e6a796048ea87d01acd3068e8ff67d078148a3fa3f4a84f69bd5")
}

# For three way trades, which pools we can use
allowed_intermediary_pairs = {}



class BadAlpha(AlphaModel):
    """Alpha model that buys Biconomy BIT tokan with transfer tax."""

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
                bit_busd.pair_id: 0.5,
            }
        elif cycle == 2:
            # Sell tick
            return {
                wbnb_busd.pair_id: 0.9,
                bit_busd.pair_id: 0.1,
            }
        elif cycle == 3:
            assert state.is_good_pair(translate_trading_pair(wbnb_busd))
            assert not state.is_good_pair(translate_trading_pair(bit_busd))
            return {wbnb_busd.pair_id: 1.0}
        else:
            raise NotImplementedError(f"Bad cycle: {cycle}")


def strategy_factory(
        *ignore,
        execution_model: UniswapV2ExecutionModelVersion0,
        sync_method: SyncMethod,
        pricing_model_factory: PricingModelFactory,
        valuation_model_factory: ValuationModelFactory,
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
        valuation_model_factory=valuation_model_factory,
        sync_method=sync_method,
        pricing_model_factory=pricing_model_factory,
        cash_buffer=cash_buffer,
        routing_model=UniswapV2SimpleRoutingModel(
            factory_router_map,
            allowed_intermediary_pairs,
            reserve_token_address,
        ),
    )

    return StrategyExecutionDescription(
        time_bucket=TimeBucket.d1,
        universe_model=universe_model,
        runner=runner,
    )


__all__ = [strategy_factory]
