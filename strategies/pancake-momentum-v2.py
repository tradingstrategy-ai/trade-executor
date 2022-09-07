"""PancakeSwap v2 momentum strategy build on the top of the new trading framework.

- Trades BNB and stablecoin pairs

- PancakeSwap https://tradingstrategy.ai/trading-view/binance/pancakeswap-v2

- Contains tradeable checks and does not touch tokens with transfer fees as a risk mitigation  
"""

import datetime
import logging
import os
from collections import Counter, defaultdict
from contextlib import AbstractContextManager
from typing import Dict

import pandas as pd

from tradeexecutor.ethereum.uniswap_v2_routing import UniswapV2SimpleRoutingModel
from tradeexecutor.strategy.execution_context import ExecutionMode
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

from tradeexecutor.ethereum.uniswap_v2_execution_v0 import UniswapV2ExecutionModelVersion0
from tradeexecutor.state.state import State
from tradeexecutor.state.sync import SyncMethod
from tradeexecutor.strategy.approval import ApprovalModel
from tradeexecutor.strategy.description import StrategyExecutionDescription
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.pricing_model import PricingModelFactory
from tradeexecutor.strategy.qstrader.alpha_model import AlphaModel
from tradeexecutor.strategy.qstrader.runner import QSTraderRunner
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverseModel, \
    TradingStrategyUniverse, translate_trading_pair, Dataset, translate_token
from tradeexecutor.strategy.valuation import ValuationModelFactory
from tradeexecutor.utils.price import is_legit_price_value

# Create a Python logger to help pinpointing issues during development
logger = logging.getLogger("bnb_chain_16h_momentum")

# Use daily candles to run the algorithm
candle_time_frame = TimeBucket.h4

# We are making a decision based on 16 hours (4 candles)
lookback = pd.Timedelta(hours=16)

# The liquidity threshold for a token to be considered
# risk free enough to be purchased
min_liquidity_threshold = 750_000

# We need to present at least 2% of liquidity of any trading pair we enter
portfolio_base_liquidity_threshold = 0.02

# Keep 6 positions open at once
# TODO: env var MAX_POSITIONS hack because Ganache is so unstable
max_assets_per_portfolio = int(os.environ.get("MAX_POSITIONS", 6))

# How many % of all value we hold in cash all the time,
# so that we do not risk our trading capital
cash_buffer = 0.80

# Trade only against these tokens
allowed_quote_tokens = {
    "WBNB": "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c".lower(),
    "BUSD": "0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56".lower(),
 }

# Keep everything internally in BUSD
reserve_token_address = "0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56".lower()

# Allowed exchanges as factory -> router pairs,
# by their smart contract addresses
factory_router_map = {
    # PancakeSwap
    "0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73": ("0x10ED43C718714eb63d5aA57B78B54704E256024E", "0x00fb7f630766e6a796048ea87d01acd3068e8ff67d078148a3fa3f4a84f69bd5")
}

# For three way trades, which pools we can use
allowed_intermediary_pairs = {
    # Route WBNB through BUSD:WBNB pool,
    "0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c": "0x58f876857a02d6762e0101bb5c46a8c1ed44dc16",
}


import datetime
from typing import Dict, List, Optional

import pandas as pd

from tradeexecutor.strategy.pricing_model import PricingModel
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.strategy_type import StrategyType
from tradeexecutor.state.visualisation import Visualisation, PlotKind
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, load_all_data
from tradingstrategy.client import Client

# Tell what trade execution engine version this strategy needs to use
trading_strategy_engine_version = "0.1"

# What kind of strategy we are running.
# This tells we are going to use
trading_strategy_type = StrategyType.managed_positions

# How our trades are routed.
# PancakeSwap basic routing supports two way trades with BUSD
# and three way trades with BUSD-BNB hop.
trade_routing = TradeRouting.pancakeswap_basic

# How often the strategy performs the decide_trades cycle.
# We do it for every 16h.
trading_strategy_cycle = CycleDuration.cycle_16h

# Strategy keeps its cash in BUSD
reserve_currency = ReserveCurrency.busd

# Time bucket for our candles
candle_time_bucket = TimeBucket.h4

# Which chain we are trading
chain_id = ChainId.bsc

# Which exchange we are trading on.
exchange_slug = "pancakeswap-v2"

# Which trading pair we are trading
trading_pair = ("WBNB", "BUSD")

# Use 4h candles for trading
candle_time_frame = TimeBucket.h4

# How much of the cash to put on a single trade
position_size = 0.10

#
# Strategy thinking specific parameter
#

batch_size = 90

slow_ema_candle_count = 39

fast_ema_candle_count = 15


def decide_trades(
        timestamp: pd.Timestamp,
        universe: Universe,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: Dict) -> List[TradeExecution]:
    """The brain function to decide the trades on each trading strategy cycle.

    - Reads incoming execution state (positions, past trades)

    - Reads the current universe (candles)

    - Decides what to do next

    - Outputs strategy thinking for visualisation and debug messages

    :param timestamp:
        The Pandas timestamp object for this cycle. Matches
        trading_strategy_cycle division.
        Always truncated to the zero seconds and minutes, never a real-time clock.

    :param universe:
        Trading universe that was constructed earlier.

    :param state:
        The current trade execution state.
        Contains current open positions and all previously executed trades, plus output
        for statistics, visualisation and diangnostics of the strategy.

    :param pricing_model:
        Pricing model can tell the buy/sell price of the particular asset at a particular moment.

    :param cycle_debug_data:
        Python dictionary for various debug variables you can read or set, specific to this trade cycle.
        This data is discarded at the end of the trade cycle.

    :return:
        List of trade instructions in the form of :py:class:`TradeExecution` instances.
        The trades can be generated using `position_manager` but strategy could also hand craft its trades.
    """

    # The pair we are trading
    pair = universe.pairs.get_single()

    # How much cash we have in the hand
    cash = state.portfolio.get_current_cash()

    # Get OHLCV candles for our trading pair as Pandas Dataframe.
    # We could have candles for multiple trading pairs in a different strategy,
    # but this strategy only operates on single pair candle.
    # We also limit our sample size to N latest candles to speed up calculations.
    candles: pd.DataFrame = universe.candles.get_single_pair_data(sample_count=batch_size)

    # We have data for open, high, close, etc.
    # We only operate using candle close values in this strategy.
    close = candles["close"]

    # Calculate exponential moving averages based on slow and fast sample numbers.
    # More information about calculating expotential averages in Pandas:
    #
    # - https://www.statology.org/exponential-moving-average-pandas/
    #
    # - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html
    #
    slow_ema = close.ewm(span=slow_ema_candle_count).mean().iloc[-1]
    fast_ema = close.ewm(span=fast_ema_candle_count).mean().iloc[-1]

    # Get the last close price from close time series
    # that's Pandas's Series object
    # https://pandas.pydata.org/docs/reference/api/pandas.Series.iat.html
    current_price = close.iloc[-1]

    # List of any trades we decide on this cycle.
    # Because the strategy is simple, there can be
    # only zero (do nothing) or 1 (open or close) trades
    # decides
    trades = []

    # Create a position manager helper class that allows us easily to create
    # opening/closing trades for different positions
    position_manager = PositionManager(timestamp, universe, state, pricing_model)

    if current_price >= slow_ema:
        # Entry condition:
        # Close price is higher than the slow EMA
        if not position_manager.is_any_open():
            buy_amount = cash * position_size
            trades += position_manager.open_1x_long(pair, buy_amount)
    elif fast_ema >= slow_ema:
        # Exit condition:
        # Fast EMA crosses slow EMA
        if position_manager.is_any_open():
            trades += position_manager.close_all()

    # Visualize strategy
    # See available colours here
    # https://matplotlib.org/stable/gallery/color/named_colors.html
    visualisation = state.visualisation
    visualisation.plot_indicator(timestamp, "Slow EMA", PlotKind.technical_indicator_on_price, slow_ema, colour="forestgreen")
    visualisation.plot_indicator(timestamp, "Fast EMA", PlotKind.technical_indicator_on_price, fast_ema, colour="limegreen")

    return trades


def create_trading_universe(
        ts: datetime.datetime,
        client: Client,
        execution_context: ExecutionContext,
        candle_time_frame_override: Optional[TimeBucket]=None,
) -> TradingStrategyUniverse:
    """Creates the trading universe where the strategy trades."""

    # Load all datas we can get for our candle time bucket
    dataset = load_all_data(
        client,
        candle_time_frame_override or candle_time_bucket,
        execution_context)

    # Filter down to the single pair we are interested in
    universe = TradingStrategyUniverse.create_single_pair_universe(
        dataset,
        chain_id,
        exchange_slug,
        trading_pair[0],
        trading_pair[1],
    )

    return universe
