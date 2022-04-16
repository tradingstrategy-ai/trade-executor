"""Live trading implementation of PancakeSwap v2 momentum strategy.

Constructs the trading universe from TradingStrategy.ai client and implements a real momentum strategy.
The universe considers only BUSD quoted PancakeSwap v2 pairs.
"""
import datetime
import logging
from collections import Counter, defaultdict
from contextlib import AbstractContextManager
from typing import Dict

import pandas as pd

from tradeexecutor.ethereum.uniswap_v2_execution import UniswapV2ExecutionModel
from tradeexecutor.state.revaluation import RevaluationMethod
from tradeexecutor.state.state import State
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.sync import SyncMethod
from tradeexecutor.strategy.approval import ApprovalModel
from tradeexecutor.strategy.description import StrategyExecutionDescription
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.pricing_model import PricingModelFactory
from tradeexecutor.strategy.qstrader.alpha_model import AlphaModel
from tradeexecutor.strategy.qstrader.runner import QSTraderRunner
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverseModel, \
    TradingStrategyUniverse, translate_trading_pair, Dataset
from tradingstrategy.client import Client

from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.frameworks.qstrader import prepare_candles_for_qstrader
from tradingstrategy.liquidity import GroupedLiquidityUniverse, LiquidityDataUnavailable
from tradingstrategy.pair import filter_for_exchanges, PandasPairUniverse, DEXPair, filter_for_quote_tokens, \
    StablecoinFilteringMode, filter_for_stablecoins
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.utils.groupeduniverse import filter_for_pairs
from tradingstrategy.universe import Universe


# Cannot use Python __name__ here because the module is dynamically loaded
logger = logging.getLogger("pancakeswap_8h_momentum")

# Use daily candles to run the algorithm
candle_time_frame = TimeBucket.h4

# We are making a decision based on 8 hours (2 candles)
# 1. The current 4h candle
# 2. The next 4h candle
lookback = pd.Timedelta(hours=8)

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
cash_buffer = 0.90


def fix_qstrader_date(ts: pd.Timestamp) -> pd.Timestamp:
    """Quick-fix for Qstrader to use its internal hour system.

    TODO: Fix QSTrader framework in long run
    """
    return ts.replace(hour=0, minute=0)


class MomentumAlphaModel(AlphaModel):
    """An alpha model that ranks pairs by the daily upwords price change %.

    We expose a lot of internal in debug data to make this code testable.
    """

    def is_funny_price(self, usd_unit_price: float) -> bool:
        """Avoid taking positions in tokens with too funny prices.

        Might cause good old floating point to crap out.
        """
        return (usd_unit_price < 0.0000001) or (usd_unit_price > 100_000)

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

    def is_accepted_risk(self, state: State, pair: DEXPair, price: float, liquidity: float) -> bool:
        """Do the risk check for the trading pair.

        - There needs to be enough liquidity
        - The price unit must look sensible

        :param price: The current closing price
        :param price: The current closing price
        :param liquidity: The trading pair liquidity in USD
        :return: True if the pair should be traded
        """

        if not state.is_good_pair(translate_trading_pair(pair)):
            return False

        if liquidity < min_liquidity_threshold:
            return False

        if self.is_funny_price(price):
            return False

        return True

    def __call__(self, ts: pd.Timestamp, universe: Universe, state: State, debug_details: Dict) -> Dict[int, float]:

        assert isinstance(ts, pd.Timestamp), f"Got {ts}"

        pair_universe = universe.pairs
        candle_universe = universe.candles
        liquidity_universe = universe.liquidity

        # The time range end is the current candle
        # The time range start is 2 * 4 hours back, and turn the range
        # exclusive instead of inclusive
        start = ts - lookback - datetime.timedelta(minutes=59)
        end = ts

        debug_details["candle_range_start"] = start
        debug_details["candle_range_end"] = end

        logger.info("Entering the alpha model at timestamp %s, we know %d pairs, our evaluation range is %s - %s", ts, pair_universe.get_count(), start, end)

        alpha_signals = Counter()

        # For each pair, check the the diff between opening and closingn price
        candle_data = candle_universe.iterate_samples_by_pair_range(start, end)

        extra_debug_data = {}
        problem_candle_count = good_candle_count = low_liquidity_count = bad_momentum_count = funny_price_count = 0

        # Iterate over all candles for all pairs in this timestamp (ts)
        for pair_id, pair_df in candle_data:

            # We have 0 or 1 candles in the range
            if len(pair_df) < 2:
                problem_candle_count += 1
                continue

            first_candle = pair_df.iloc[-2]
            # mid-candle ignored
            last_candle = pair_df.iloc[-1]

            open = first_candle["Open"]  # QStrader data frames are using capitalised version of OHLCV core variables
            close = last_candle["Close"]  # QStrader data frames are using capitalised version of OHLCV core variables
            pair = pair_universe.get_pair_by_id(pair_id)

            # We need 8h data to calculate the momentum.
            # We have have
            # opening of the first candle -> 4h -> opening of the second candle -> 4h -> closing of the second candle
            lookback_duration = last_candle["Date"] - first_candle["Date"]
            if lookback_duration < datetime.timedelta(hours=4):
                logger.info("Bad lookback duration %s for %s, our range is %s - %s", lookback_duration, pair, start, end)
                problem_candle_count += 1
                continue

            if self.is_funny_price(close):
                # This trading pair is too funny that we do not want to play with it
                funny_price_count += 1

            # We define momentum as how many % the trading pair price gained yesterday
            momentum = (close - open) / open
            momentum = max(0, momentum)

            # This pair has positive momentum, check if it has enough available liquidity
            available_liquidity_for_pair = 0
            if momentum > 0:

                # Check for the liquidity requirement
                try:
                    available_liquidity_for_pair = liquidity_universe.get_closest_liquidity(pair_id, ts)
                except LiquidityDataUnavailable as e:
                    # There might be holes in the data, because BSC network not syncing properly,
                    # BSC blockchain was halted or because BSC nodes themselves had crashed.
                    # In this case, we just assume the liquidity was zero and don't backtest.
                    # logger.warning(f"No liquidity data for pair {pair}, currently backtesting at {ts}")
                    logger.info("Holes in liquidity data for %s? %s", pair, ts)
                    available_liquidity_for_pair = 0
            else:
                #logger.info("Pair %s non-positive momentum for range %s - %s", pair, first_candle["Date"], last_candle["Date"])
                bad_momentum_count += 1
                continue

            if self.is_accepted_risk(state, pair, close, available_liquidity_for_pair):
                # Do candle check only after we know the pair is "good" liquidity wise
                # and should have candles
                candle_count = len(pair_df)
                alpha_signals[pair_id] = momentum
                good_candle_count += 1
            else:
                # No alpha for illiquid pairs
                # logger.info("Pair %s lacks liquidity, liquidity %s, needed %s", pair, available_liquidity_for_pair, liquidity_threshold)
                alpha_signals[pair_id] = 0
                candle_count = None
                low_liquidity_count += 1

            # Extra debug details are available for pairs for which a buy decision can be made
            extra_debug_data[pair_id] = {
                "pair": pair,
                "open": open,
                "close": close,
                "momentum": momentum,
                "liquidity": available_liquidity_for_pair,
                "candle_count": candle_count,
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

        # Some debug dump for unit test ipdb tracking
        for pair_id, momentum in top_signals:
            debug_data = extra_debug_data[pair_id]
            pair = debug_data["pair"]
            logger.info(f"{ts}: Signal for {pair.get_ticker()} (#{pair.pair_id}) is {momentum * 100:,.2f}%, open: {debug_data['open']:,.8f}, close: {debug_data['close']:,.8f}")

        logger.info("Got signals %s", weighed_signals)
        debug_details["signals"]: weighed_signals.copy()
        debug_details["pair_details"]: extra_debug_data
        debug_details["problem_candle_count"] = problem_candle_count
        debug_details["good_candle_count"] = good_candle_count
        debug_details["extra_debug_data"] = extra_debug_data
        debug_details["bad_momentum_count"] = bad_momentum_count
        debug_details["low_liquidity_count"] = low_liquidity_count
        debug_details["funny_price_count"] = funny_price_count
        return dict(weighed_signals)


class OurUniverseModel(TradingStrategyUniverseModel):
    """Create PancakeSwap v2 trading universe."""

    def filter_universe(self, dataset: Dataset) -> TradingStrategyUniverse:
        """Filter data streams we are interested in."""

        with self.timed_task_context_manager("filter_universe"):

            # We only trade on Pancakeswap v2
            exchange_universe = dataset.exchanges
            pancake_v2 = exchange_universe.get_by_chain_and_slug(ChainId.bsc, "pancakeswap-v2")
            assert pancake_v2, "PancakeSwap v2 missing in the dataset"

            busd_address = "0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56".lower()

            our_exchanges = [
                pancake_v2,
            ]

            # Choose BUSD pairs on PancakeSwap v2
            # that are not stablecoin pairs
            pairs_df = filter_for_exchanges(dataset.pairs, our_exchanges)
            pairs_df = filter_for_quote_tokens(pairs_df, [busd_address])
            pairs_df = filter_for_stablecoins(pairs_df, StablecoinFilteringMode.only_volatile_pairs)

            # Create trading pair database
            pairs = PandasPairUniverse(pairs_df)

            # We do a bit detour here as we need to address the assets by their trading pairs first
            bnb_busd = pairs.get_one_pair_from_pandas_universe(pancake_v2.exchange_id, "WBNB", "BUSD")
            assert bnb_busd, "We do not have BNB-BUSD, something wrong with the dataset"

            # Get daily candles as Pandas DataFrame
            all_candles = dataset.candles
            filtered_candles = filter_for_pairs(all_candles, pairs_df)
            candle_universe = GroupedCandleUniverse(prepare_candles_for_qstrader(filtered_candles), timestamp_column="Date")

            # Get liquidity candles as Pandas Dataframe
            all_liquidity = dataset.liquidity
            filtered_liquidity = filter_for_pairs(all_liquidity, pairs_df)
            liquidity_universe = GroupedLiquidityUniverse(filtered_liquidity)

            # We are using BUSD as the reserve asset, pick it through BNB-BUSD pair
            bnb_busd_pair = translate_trading_pair(bnb_busd)
            reserve_assets = [
                bnb_busd_pair.quote,
            ]

            universe = Universe(
                time_frame=dataset.time_frame,
                chains=[ChainId.bsc],
                pairs=pairs,
                exchanges=our_exchanges,
                candles=candle_universe,
                liquidity=liquidity_universe,
            )

            return TradingStrategyUniverse(universe=universe, reserve_assets=reserve_assets)

    def construct_universe(self, execution_model: ExecutionModel, live) -> TradingStrategyUniverse:
        dataset = self.load_data(TimeBucket.h4, live)
        universe = self.filter_universe(dataset)
        self.log_universe(universe.universe)
        return universe


def strategy_factory(
        *ignore,
        execution_model: UniswapV2ExecutionModel,
        sync_method: SyncMethod,
        pricing_model_factory: PricingModelFactory,
        revaluation_method: RevaluationMethod,
        client: Client,
        timed_task_context_manager: AbstractContextManager,
        approval_model: ApprovalModel,
        **kwargs) -> StrategyExecutionDescription:

    if ignore:
        # https://www.python.org/dev/peps/pep-3102/
        raise TypeError("Only keyword arguments accepted")

    # assert isinstance(execution_model, UniswapV2ExecutionModel), f"This strategy is compatible only with UniswapV2ExecutionModel, got {execution_model}"
    # assert execution_model.chain_id == 1337, f"This strategy is hardcoded to ganache-cli test chain, got chain {execution_model.chain_id}"

    universe_model = OurUniverseModel(client, timed_task_context_manager)

    runner = QSTraderRunner(
        alpha_model=MomentumAlphaModel(),
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