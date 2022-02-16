"""Example strategy that trades on Uniswap v2 using the local EthereumTester EVM deployment."""
import datetime
import logging
from collections import Counter, defaultdict
from contextlib import AbstractContextManager
from typing import Dict

import pandas as pd

from qstrader.alpha_model.alpha_model import AlphaModel

from tradeexecutor.strategy.qstrader.livetrader import QSTraderLiveTrader
from tradeexecutor.strategy.runner import Dataset
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.frameworks.qstrader import prepare_candles_for_qstrader
from tradingstrategy.liquidity import GroupedLiquidityUniverse, LiquidityDataUnavailable
from tradingstrategy.pair import filter_for_exchanges, PandasPairUniverse, DEXPair
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.utils.groupeduniverse import filter_for_pairs
from tradingstrategy.universe import Universe


# Cannot use Python __name__ here because the module is dynamically loaded
logging = logging.getLogger("uniswap_simulatead_example")


class BuyEverySecondDayAlpha(AlphaModel):
    """An alpha model that switches between buying and selling ETH.

    On every secodn day, we buy with the full portfolio. On another day, we sell.
    """

    def __init__(
            self,
            universe: Universe,
            data_handler=None
    ):
        self.exchange_universe = universe.exchanges
        self.pair_universe = universe.pairs
        self.candle_universe = universe.candles
        self.liquidity_universe = universe.liquidity
        self.data_handler = data_handler
        self.liquidity_reached_state = {}

    def __call__(self, ts: pd.Timestamp, debug_details: Dict) -> Dict[int, float]:
        """
        Produce the dictionary of scalar signals for
        each of the Asset instances within the Universe.

        :param ts: Candle timestamp iterator

        :return: Dict(pair_id, alpha signal)
        """

        assert debug_details
        ts = fix_qstrader_date(ts)
        return dict(weighed_signals)


class SimulatedUniswapV2LiveTrader(QSTraderLiveTrader):
    """Live strategy set up."""

    def get_strategy_time_frame(self) -> TimeBucket:
        """Strategy is run on the daily candles."""
        return TimeBucket.d1

    def construct_universe(self, dataset: Dataset) -> Universe:
        """Sets up pairs, candles and liquidity samples.

        :param client: Client instance. Note that this cannot be stable across ticks, as e.g. API keys can change. Client is recreated for every tick.
        :return:
        """


def strategy_executor_factory(*args, **kwargs):
    return SimulatedUniswapV2LiveTrader(*args, **kwargs)


__all__ = [strategy_executor_factory]