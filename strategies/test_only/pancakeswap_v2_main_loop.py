"""Live trading implementation of PancakeSwap v2 momentum strategy.

Constructs the trading universe from TradingStrategy.ai client and implements a real momentum strategy.
The universe considers only BUSD quoted PancakeSwap v2 pairs.
"""

import logging
from collections import Counter, defaultdict
from contextlib import AbstractContextManager
from typing import Dict

import pandas as pd

from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_execution import UniswapV2Execution
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_execution_v0 import UniswapV2ExecutionModelVersion0
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_routing import UniswapV2Routing
from tradeexecutor.state.state import State
from tradeexecutor.strategy.sync_model import SyncMethodV0
from tradeexecutor.strategy.approval import ApprovalModel
from tradeexecutor.strategy.description import StrategyExecutionDescription
from tradeexecutor.strategy.execution_context import ExecutionMode, ExecutionContext
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.pricing_model import PricingModelFactory
from tradeexecutor.strategy.qstrader.alpha_model import AlphaModel
from tradeexecutor.strategy.qstrader.runner import QSTraderRunner
from tradeexecutor.strategy.routing import RoutingModel
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverseModel, \
    TradingStrategyUniverse, translate_trading_pair, Dataset
from tradingstrategy.client import Client

from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.frameworks.qstrader import prepare_candles_for_qstrader
from tradingstrategy.liquidity import GroupedLiquidityUniverse, LiquidityDataUnavailable
from tradingstrategy.pair import filter_for_exchanges, PandasPairUniverse, DEXPair, filter_for_quote_tokens
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.utils.groupeduniverse import filter_for_pairs
from tradingstrategy.universe import Universe


# Cannot use Python __name__ here because the module is dynamically loaded
from tradeexecutor.strategy.valuation import ValuationModelFactory

logger = logging.getLogger("pancakeswap_example")


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
prefilter_min_buy_volume = 10_000_000

# The liquidity threshold for a token to be considered
# risk free enough to be purchased
min_liquidity_threshold = 5_750_000

# Any trading pair we enter must have
# at least portflio total market value * portfolio_base_liquidity_threshold liquidity available
portfolio_base_liquidity_threshold = 0.66

# How many tokens we can hold in our portfolio
# If there are more new tokens coming to market per day,
# we just ignore those with less liquidity
max_assets_per_portfolio = 2

# How many % of all value we hold in cash all the time,
# so that we can sustain hits
cash_buffer = 0.50

# Use daily candles to run the algorithm
candle_time_frame = TimeBucket.d1

#
# Routing options
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


def fix_qstrader_date(ts: pd.Timestamp) -> pd.Timestamp:
    """Quick-fix for Qstrader to use its internal hour system.

    TODO: Fix QSTrader framework in long run
    """
    return ts.replace(hour=0, minute=0)


class MomentumAlphaModel(AlphaModel):
    """An alpha model that ranks pairs by the daily upwords price change %."""

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

    def __call__(self, ts: pd.Timestamp, universe: Universe, state: State, debug_details: Dict) -> Dict[int, float]:
        """
        Produce the dictionary of scalar signals for
        each of the Asset instances within the Universe.

        :param ts: Candle timestamp iterator

        :return: Dict(pair_id, alpha signal)
        """

        assert debug_details
        assert isinstance(ts, pd.Timestamp), f"Got {ts}"

        pair_universe = universe.pairs
        candle_universe = universe.candles
        liquidity_universe = universe.liquidity

        logger.info("Entering the alpha model, we know %d pairs", pair_universe.get_count())

        ts = fix_qstrader_date(ts)

        # Calculate momentum based on the candles one day before today.
        # For the simulation to the coherent, we need to make trading decisions
        # at the start of the day based on the momentum we have withnessed
        # yesterday.
        ts_yesterday = ts - pd.Timedelta(days=1)

        # For each pair, check the the diff between opening and closingn price
        timepoint_candles = candle_universe.get_all_samples_by_timestamp(ts_yesterday)
        alpha_signals = Counter()

        if len(timepoint_candles) == 0:
            logger.error(f"No candles available at %s", ts_yesterday)
            return {}

        ts_: pd.Timestamp
        candle: pd.Series

        extra_debug_data = defaultdict(dict)

        # We have a absolute minimum liquidity floor (min_liquidity_threshold),
        # but we also have a minimum liquidity floor related to our portfolio size.
        # With high value portfolio, we will no longer invest in tokens with less liquidity.
        # TODO: Fix QSTrader to pass this information directly.
        portflio_value = state.portfolio.calculate_total_equity()
        liquidity_threshold = max(min_liquidity_threshold, portflio_value * portfolio_base_liquidity_threshold)

        # Expose how many candles we evaluated to the testing
        debug_details["timepoint_candles_count"] = len(timepoint_candles)

        # Iterate over all candles for all pairs in this timestamp (ts)
        for ts_, candle in timepoint_candles.iterrows():
            pair_id = candle["pair_id"]
            open = candle["Open"]  # QStrader data frames are using capitalised version of OHLCV core variables
            close = candle["Close"]  # QStrader data frames are using capitalised version of OHLCV core variables
            pair = pair_universe.get_pair_by_id(pair_id)

            # logger.debug("Checking pair %s", pair)

            if self.is_funny_price(open) or self.is_funny_price(close):
                # This trading pair is too funny that we do not want to play with it
                continue

            if (not pair.base_token_symbol) or (not pair.quote_token_symbol):
                # Questionable pair
                continue

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
                    available_liquidity_for_pair = 0

            if available_liquidity_for_pair >= liquidity_threshold:
                alpha_signals[pair_id] = momentum
            else:
                # No alpha for illiquid pairs
                # logger.warning("Liquidity not enough. Pair %s, liquidity %s, needed %s", pair, available_liquidity_for_pair, liquidity_threshold)
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

        # Some debug dump for unit test ipdb tracking
        for pair_id, momentum in top_signals:
            debug_data = extra_debug_data[pair_id]
            pair = debug_data["pair"]
            logger.info(f"{ts}: Signal for {pair.get_ticker()} (#{pair.pair_id}) is {momentum * 100:,.2f}%, open: {debug_data['open']:,.8f}, close: {debug_data['close']:,.8f}")

        logger.info("Got signals %s", weighed_signals)
        debug_details["signals"]: weighed_signals.copy()
        debug_details["pair_details"]: extra_debug_data

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
            pairs_df = filter_for_exchanges(dataset.pairs, our_exchanges)
            pairs_df = filter_for_quote_tokens(pairs_df, [busd_address])

            # Create trading pair database
            pairs = PandasPairUniverse(pairs_df)

            # We do a bit detour here as we need to address the assets by their trading pairs first
            # https://tradingstrategy.ai/trading-view/binance/pancakeswap-v2/bnb-busd
            bnb_busd = pairs.get_pair_by_smart_contract("0x58f876857a02d6762e0101bb5c46a8c1ed44dc16")
            bnb_busd.fee = 25
            assert bnb_busd, "We do not have BNB-BUSD, something wrong with the dataset"

            # Get daily candles as Pandas DataFrame
            all_candles = dataset.candles
            filtered_candles = filter_for_pairs(all_candles, pairs_df)
            candle_universe = GroupedCandleUniverse(
                prepare_candles_for_qstrader(filtered_candles),
                timestamp_column="Date",
                fix_wick_threshold=None,
            )

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
                time_bucket=dataset.time_bucket,
                chains=[ChainId.bsc],
                pairs=pairs,
                exchanges=our_exchanges,
                candles=candle_universe,
                liquidity=liquidity_universe,
            )

            return TradingStrategyUniverse(data_universe=universe, reserve_assets=reserve_assets)

    def construct_universe(self, execution_model: ExecutionModel, mode: ExecutionMode, universe_options) -> TradingStrategyUniverse:
        assert isinstance(mode, ExecutionMode)
        dataset = self.load_data(TimeBucket.d1, mode)
        universe = self.filter_universe(dataset)
        self.log_universe(universe.data_universe)
        return universe


def strategy_factory(
        *ignore,
        execution_model: UniswapV2ExecutionModelVersion0,
        sync_model,
        pricing_model_factory: PricingModelFactory,
        valuation_model_factory: ValuationModelFactory,
        client: Client,
        timed_task_context_manager: AbstractContextManager,
        approval_model: ApprovalModel,
        routing_model: RoutingModel,
        **kwargs) -> StrategyExecutionDescription:

    if ignore:
        # https://www.python.org/dev/peps/pep-3102/
        raise TypeError("Only keyword arguments accepted")

    assert isinstance(execution_model, (UniswapV2Execution, UniswapV2ExecutionModelVersion0)), f"This strategy is compatible only with UniswapV2ExecutionModel, got {execution_model}"

    assert execution_model.chain_id in (56, 1337), f"This strategy is hardcoded to ganache-cli test chain, got chain {execution_model.chain_id}"

    universe_model = OurUniverseModel(client, timed_task_context_manager)

    routing_model = UniswapV2Routing(
        factory_router_map,
        allowed_intermediary_pairs,
        reserve_token_address,
    )

    runner = QSTraderRunner(
        alpha_model=MomentumAlphaModel(),
        timed_task_context_manager=timed_task_context_manager,
        execution_model=execution_model,
        approval_model=approval_model,
        valuation_model_factory=valuation_model_factory,
        sync_model=sync_model,
        pricing_model_factory=pricing_model_factory,
        cash_buffer=cash_buffer,
        routing_model=routing_model,
        execution_context=ExecutionContext(ExecutionMode.unit_testing_trading),
    )

    return StrategyExecutionDescription(
        time_bucket=TimeBucket.d1,
        universe_model=universe_model,
        runner=runner,
    )


__all__ = [strategy_factory]