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
            # Sell all
            return {
            }
        elif cycle == 3:
            assert state.is_good_pair(translate_trading_pair(wbnb_busd))
            assert not state.is_good_pair(translate_trading_pair(bit_busd))
            return {wbnb_busd.pair_id: 1.0}
        else:
            raise NotImplementedError(f"Bad cycle: {cycle}")


# Keep everything internally in BUSD
reserve_token_address = "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c"

# Allowed exchanges as factory -> router pairs,
# by their smart contract addresses
factory_router_map = {
    "0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73": "0x10ED43C718714eb63d5aA57B78B54704E256024E",
}

# For three way trades, which pools we can use
allowed_intermediary_pairs = {
    # BUSD -> WBNB
    "0x58f876857a02d6762e0101bb5c46a8c1ed44dc16",  # https://tradingstrategy.ai/trading-view/binance/pancakeswap-v2/bnb-busd
}


def strategy_factory(
        *ignore,
        execution_model: UniswapV2ExecutionModelVersion0,
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
        routing_model=UniswapV2SimpleRoutingModel(
            factory_router_map,
            allowed_intermediary_pairs,
            reserve_token_address,
            max_slippage=0.01,
        ),
    )

    return StrategyExecutionDescription(
        time_bucket=TimeBucket.d1,
        universe_model=universe_model,
        runner=runner,
    )


__all__ = [strategy_factory]
