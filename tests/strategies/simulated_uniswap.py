"""Example strategy that trades on Uniswap v2 using the local EthereumTester EVM deployment."""
import logging
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
    return SimulatedUniswapV2LiveTrader(alpha_model=SomeTestBuysAlphaModel(), **kwargs)


__all__ = [strategy_executor_factory]