"""Example strategy that trades on PancakeSwap"""
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
logging = logging.getLogger("pancakeswap_example")


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


def fix_qstrader_date(ts: pd.Timestamp) -> pd.Timestamp:
    """Quick-fix for Qstrader to use its internal hour system.

    TODO: Fix QSTrader framework in long run
    """
    return ts.replace(hour=0, minute=0)


class MomentumAlphaModel(AlphaModel):
    """An alpha model that ranks pairs by the daily upwords price change %."""

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

    def is_funny_price(self, usd_unit_price: float) -> bool:
        """Avoid taking positions in tokens with too funny prices.

        Might cause good old floating point to crap out.
        """
        return (usd_unit_price < 0.0000001) or (usd_unit_price > 100_000)

    def translate_pair(self, pair_id: int) -> str:
        """Make pari ids human readable for logging."""
        pair_info = self.pair_universe.get_pair_by_id(pair_id)
        return pair_info.get_friendly_name(self.exchange_universe)

    def filter_duplicate_base_tokens(self, alpha_signals: Counter, debug_data: dict) -> Counter:
        """Filter duplicate alpha signals for trading pairs sharing the same base token.

        This is because on DEX markets, the base token may trade with several quote tokens.

        For example, we can end up holding BNB by buying
        - WBNB-BUSD
        - WBNB-USDT

        For the portfolio, it does not matter which route we ended up buying the token
        (although this matters for the price impact).

        We use the already resolved pair data to check for the duplicates.
        We'll just filter out the second entry (worse alpha).
        """
        accumulated_quote_tokens = set()
        filtered_alpha = Counter()
        for pair_id, alpha in alpha_signals.most_common():
            pair: DEXPair = debug_data[pair_id]["pair"]
            base_token = pair.base_token_symbol
            if base_token in accumulated_quote_tokens:
                continue
            filtered_alpha[pair_id] = alpha
            accumulated_quote_tokens.add(base_token)
        return filtered_alpha

    def __call__(self, ts: pd.Timestamp, debug_details: Dict) -> Dict[int, float]:
        """
        Produce the dictionary of scalar signals for
        each of the Asset instances within the Universe.

        :param ts: Candle timestamp iterator

        :return: Dict(pair_id, alpha signal)
        """

        assert debug_details

        ts = fix_qstrader_date(ts)

        # Calculate momentum based on the candles one day before today.
        # For the simulation to the coherent, we need to make trading decisions
        # at the start of the day based on the momentum we have withnessed
        # yesterday.
        ts_yesterday = ts - pd.Timedelta(days=1)

        # For each pair, check the the diff between opening and closingn price
        timepoint_candles = self.candle_universe.get_all_samples_by_timestamp(ts_yesterday)
        alpha_signals = Counter()

        if len(timepoint_candles) == 0:
            print(f"No candles at {ts}")

        ts_: pd.Timestamp
        candle: pd.Series

        extra_debug_data = defaultdict(dict)

        # We have a absolute minimum liquidity floor (min_liquidity_threshold),
        # but we also have a minimum liquidity floor related to our portfolio size.
        # With high value portfolio, we will no longer invest in tokens with less liquidity.
        # TODO: Fix QSTrader to pass this information directly.
        portflio_value = debug_details["broker"]["portfolios"]["000001"]["total_equity"]
        liquidity_threshold = max(min_liquidity_threshold, portflio_value * portfolio_base_liquidity_threshold)

        # Iterate over all candles for all pairs in this timestamp (ts)
        for ts_, candle in timepoint_candles.iterrows():
            pair_id = candle["pair_id"]
            open = candle["Open"]  # QStrader data frames are using capitalised version of OHLCV core variables
            close = candle["Close"]  # QStrader data frames are using capitalised version of OHLCV core variables
            pair = pair_universe.get_pair_by_id(pair_id)

            if self.is_funny_price(open) or self.is_funny_price(close):
                # This trading pair is too funny that we do not want to play with it
                continue

            # We define momentum as how many % the trading pair price gained yesterday
            momentum = (close - open) / open
            momentum = max(0, momentum)

            # This pair has positive momentum, check if it has enough available liquidity
            available_liquidity_for_pair = 0
            if momentum > 0:

                # Check for the liquidity requirement
                try:
                    available_liquidity_for_pair = self.liquidity_universe.get_closest_liquidity(pair_id, ts)
                except LiquidityDataUnavailable as e:
                    # There might be holes in the data, because BSC network not syncing properly,
                    # BSC blockchain was halted or because BSC nodes themselves had crashed.
                    # In this case, we just assume the liquidity was zero and don't backtest.
                    # logger.warning(f"No liquidity data for pair {pair}, currently backtesting at {ts}")
                    available_liquidity_for_pair = 0

            if available_liquidity_for_pair >= liquidity_threshold:
                alpha_signals[pair_id] = momentum
            else:
                # No alpha for illiquid pairs
                alpha_signals[pair_id] = 0

            extra_debug_data[pair_id] = {
                "pair": pair,
                "open": open,
                "close": close,
                "momentum": momentum,
                "liquidity": available_liquidity_for_pair,
                "liquidity_threshold": liquidity_threshold,
            }

        # Pick top 10 momentum asset and filter out DEX duplicates
        filtered_signals = self.filter_duplicate_base_tokens(alpha_signals, extra_debug_data)
        top_signals = filtered_signals.most_common(max_assets_per_portfolio)

        # Use 1/N weighting system to reduce risk,
        # otherwise the algo could go 99% in to some token that had 10,000% paper gain
        weighed_signals = {}
        for idx, tuple in enumerate(top_signals, 1):
            pair_id, alpha = tuple
            weighed_signals[pair_id] = 1 / idx

        # Debug dump status
        if debug:
            for pair_id, momentum in top_signals:
                debug_data = extra_debug_data[pair_id]
                pair = debug_data["pair"]
                print(f"{ts}: Signal for {pair.get_ticker()} (#{pair.pair_id}) is {momentum * 100:,.2f}%, open: {debug_data['open']:,.8f}, close: {debug_data['close']:,.8f}")

        debug_details["signals"]: weighed_signals.copy()
        debug_details["pair_details"]: extra_debug_data

        return dict(weighed_signals)


class PancakeLiveTrader(QSTraderLiveTrader):
    """Live strategy set up."""

    def __init__(self, timed_task_context_manager: AbstractContextManager, max_data_age: datetime.timedelta):
        super().__init__(MomentumAlphaModel, timed_task_context_manager, max_data_age)

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
            time_bucket=dataset.time_bucket,
            chains=[ChainId.bsc],
            pairs=PandasPairUniverse(pairs_df),
            exchanges=our_exchanges,
            candles=candle_universe,
            liquidity=liquidity_universe,
        )


def strategy_executor_factory(**kwargs):
    return PancakeLiveTrader(**kwargs)


__all__ = [strategy_executor_factory]