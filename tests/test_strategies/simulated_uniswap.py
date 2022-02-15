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

    def __init__(self, timed_task_context_manager: AbstractContextManager):
        super().__init__(SimulatedUniswapV2LiveTrader, timed_task_context_manager)

    def get_strategy_time_frame(self) -> TimeBucket:
        """Strategy is run on the daily candles."""
        return TimeBucket.d1

    def construct_universe(self, dataset: Dataset) -> Universe:
        """Sets up pairs, candles and liquidity samples.

        :param client: Client instance. Note that this cannot be stable across ticks, as e.g. API keys can change. Client is recreated for every tick.
        :return:
        """

        exchange_universe = dataset.exchanges

        our_exchanges = [
            exchange_universe.get_by_chain_and_slug(ChainId.bsc, "pancakeswap"),
            exchange_universe.get_by_chain_and_slug(ChainId.bsc, "pancakeswap-v2"),
        ]

        # Choose all pairs that trade on exchanges we are interested in
        pairs_df = filter_for_exchanges(dataset.pairs, our_exchanges)

        # Get daily candles as Pandas DataFrame
        all_candles = dataset.candles
        # filtered_candles = all_candles.loc[all_candles["pair_id"].isin(wanted_pair_ids)]
        filtered_candles = filter_for_pairs(all_candles, pairs_df)
        candle_universe = GroupedCandleUniverse(prepare_candles_for_qstrader(filtered_candles), timestamp_column="Date")

        all_liquidity = dataset.liquidity
        filtered_liquidity = filter_for_pairs(all_liquidity, pairs_df)
        #filtered_liquidity = all_liquidity.loc[all_liquidity["pair_id"].isin(wanted_pair_ids)]
        # filtered_liquidity = filtered_liquidity.set_index(filtered_liquidity["timestamp"])
        #filtered_liquidity = filtered_liquidity.set_index(filtered_liquidity["timestamp"])
        liquidity_universe = GroupedLiquidityUniverse(filtered_liquidity)

        return Universe(
            time_frame=dataset.time_frame,
            chains=[ChainId.bsc],
            pairs=PandasPairUniverse(pairs_df),
            exchanges=our_exchanges,
            candles=candle_universe,
            liquidity=liquidity_universe,
        )


def strategy_executor_factory(**kwargs):
    return SimulatedUniswapV2LiveTrader(**kwargs)


__all__ = [strategy_executor_factory]