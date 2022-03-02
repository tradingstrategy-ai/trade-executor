"""An example strategy that does a random buy or sell every day using WETH-USDC pair."""
import logging
import random
from typing import Dict, Any

import pandas as pd

from tradeexecutor.state.state import State
from tradeexecutor.strategy.qstrader.alpha_model import AlphaModel

from tradeexecutor.strategy.qstrader.runner import QSTraderRunner
from tradeexecutor.strategy.runner import Dataset
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe


# Cannot use Python __name__ here because the module is dynamically loaded
logging = logging.getLogger("uniswap_simulatead_example")


class DummyAlphaModel(AlphaModel):
    """Hold random % of portfolio in ETH-USDC, random in cash."""
    def __call__(self, ts: pd.Timestamp, universe: Universe, state: State, debug_details: Dict) -> Dict[Any, float]:
        # Because this is a test strategy, we assume we have fixed 3 assets
        assert len(universe.exchanges) == 1
        uniswap = universe.get_single_exchange()
        assert universe.pairs.get_count() == 1

        weth_usdc = universe.pairs.get_one_pair_from_pandas_universe(uniswap.exchange_id, "WETH", "USDC")
        return {weth_usdc.pair_id: random.random()}


class SimulatedUniswapV2LiveTrader(QSTraderRunner):
    """Live strategy set up."""

    def get_strategy_time_frame(self) -> TimeBucket:
        """Strategy is run on the daily candles."""
        return TimeBucket.d1

    def construct_universe(self, dataset: Dataset) -> Universe:
        """Sets up pairs, candles and liquidity samples.

        :param client: Client instance. Note that this cannot be stable across ticks, as e.g. API keys can change. Client is recreated for every tick.
        :return:
        """

def strategy_executor_factory(*ignore, **kwargs):
    if ignore:
        # https://www.python.org/dev/peps/pep-3102/
        raise TypeError("Only keyword arguments accepted")
    return SimulatedUniswapV2LiveTrader(alpha_model=DummyAlphaModel(), **kwargs)


__all__ = [strategy_executor_factory]