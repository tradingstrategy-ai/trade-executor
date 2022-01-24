"""Example strategy that trades on PancakeSwap"""

import pandas as pd

from tradeexecutor.strategy.qstrader import QSTraderLiveTrader
from tradeexecutor.strategy.runner import Dataset
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.frameworks.qstrader import prepare_candles_for_qstrader
from tradingstrategy.liquidity import GroupedLiquidityUniverse
from tradingstrategy.pair import filter_for_exchanges, PandasPairUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.utils.groupeduniverse import filter_for_pairs
from tradingstrategy.universe import Universe


# Not relevant for live execution
# start = pd.Timestamp('2020-11-01 00:00')

# Not relevant for live execution
# end = pd.Timestamp('2021-12-30 00:00')
#end = pd.Timestamp('2020-12-01 00:00')

# Start backtesting with $10k in hand


initial_cash = 10_000

# Prefiltering to limit the pair set to speed up computations
# How many USD all time buy volume the pair must have had
# to be included in the backtesting
prefilter_min_buy_volume = 5_000_000

# The liquidity threshold for a token to be considered
# risk free enough to be purchased
min_liquidity_threshold = 750_000

# Any trading pair we enter must have
# at least portflio total market value * portfolio_base_liquidity_threshold liquidity available
portfolio_base_liquidity_threshold = 0.66

# How many tokens we can hold in our portfolio
# If there are more new tokens coming to market per day,
# we just ignore those with less liquidity
max_assets_per_portfolio = 4

# How many % of all value we hold in cash all the time,
# so that we can sustain hits
cash_buffer = 0.50

# Use daily candles to run the algorithm
candle_time_frame = TimeBucket.d1

# Print algorithm internal state while it is running to debug issues
debug = False


class PancakeLiveTrade(QSTraderLiveTrader):
    """Live strategy set up.
    """

    def construct_universe(self, dataset: Dataset) -> Universe:
        """Sets up pairs, candles and liquidity samples.

        :param client: Client instance. Note that this cannot be stable across ticks, as e.g. API keys can change. Client is recreated for every tick.
        :return:
        """

        exchange_universe = dataset.exchanges

        all_pairs_dataframe = dataset.pairs

        our_exchanges = [
            exchange_universe.get_by_chain_and_slug(ChainId.bsc, "pancakeswap"),
            exchange_universe.get_by_chain_and_slug(ChainId.bsc, "pancakeswap-v2"),
        ]

        # Only choose pairs on exchanges we are interested in
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

    def preflight_check(self, client: Client, universe: Universe, now_: datetime.datetime):
        pass

