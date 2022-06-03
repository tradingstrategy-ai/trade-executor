"""Live trading strategy implementation fo ra single pair exponential moving average model.

- Trades a single trading pair

- Does long positions only
"""

import logging
from contextlib import AbstractContextManager
from typing import Dict

import pandas as pd

from tradeexecutor.ethereum.uniswap_v2_execution_v0 import UniswapV2ExecutionModelVersion0
from tradeexecutor.state.revaluation import RevaluationMethod
from tradeexecutor.state.state import State

from tradeexecutor.state.sync import SyncMethod
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.approval import ApprovalModel
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.description import StrategyExecutionDescription
from tradeexecutor.strategy.execution_model import ExecutionModel, ExecutionContext
from tradeexecutor.strategy.factory import StrategyType, TradeRouting
from tradeexecutor.strategy.pandas_trader.runner import PandasTraderRunner
from tradeexecutor.strategy.pandas_trader.output import StrategyOutput
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.pricing_model import PricingModelFactory
from tradeexecutor.strategy.strategy_module import ReserveCurrency
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverseModel, \
    TradingStrategyUniverse, load_all_data
from tradingstrategy.client import Client

from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

# Tell what trade execution engine version this strategy needs to use
trading_strategy_engine_version = "0.1"

trading_strategy_type = StrategyType.position_manager

trading_strategy_routing = TradeRouting.pancakeswap_basic

#: We operate on 16h cycles
trading_strategy_cycle = CycleDuration.cycle_16h

reserve_currency = ReserveCurrency.busd

# Time bucket for our candles
candle_time_bucket = TimeBucket.h4

# Which chain we are trading
chain_id = ChainId.bsc

# Which exchange we are trading
exchange_slug = "pancakeswap-v2"

# Which token we are trading
base_token = "WBNB"

# Using which currency
quote_token = "BUSD"

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


def decide_trade(
        timestamp: pd.Timestamp,
        universe: Universe,
        state: State,
        position_manager: PositionManager,
        output: StrategyOutput,
        cycle_debug_data: Dict):
    """The brain functoin to decide on trades.

    - Reads incoming execution state (positions, past trades)

    - Reads the current universe (candles)

    - Decides what to do next

    - Outputs strategy thinking for visualisation and debug messages
    """

    # The pair we are trading
    pair = universe.pairs.get_single()

    # How much cash we have in the hand
    cash = state.portfolio.get_current_cash()

    # Get OHLCV candles for our trading pair as Pandas Dataframe.
    # We could have candles for multiple trading pairs in a different strategy,
    # but this strategy only operates on single pair candle.
    # We also limit our sample size to N latest candles to speed up calculations.
    candles: pd.DataFrame = universe.candles.get_single(sample_count=batch_size)

    # We have data for open, high, close, etc.
    # We only operate using candle close values in this strategy.
    close = candles["close"]

    # Calculate exponential moving averages based on
    # https://www.statology.org/exponential-moving-average-pandas/
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html
    slow_ema = close.ewm(span=slow_ema_candle_count)
    fast_ema = close.ewm(span=fast_ema_candle_count)

    # Calculate strategy boolean status flags
    # on the last candle close and diagnostics
    current_price = close[-1]

    # List of any trades we decide on this cycle.
    # Because the strategy is simple, there can be
    # only zero (do nothing) or 1 (open or close) trades
    # decides
    trades: TradeExecution = []

    if current_price >= slow_ema[-1]:
        # Entry condition:
        # Close price is higher than the slow EMA
        if not position_manager.is_any_open():
            buy_amount = cash * position_size
            trades += position_manager.open_1x_long(pair, buy_amount)
    elif fast_ema[-1] >= slow_ema:
        # Exit condition:
        # Fast EMA crosses slow EMA
        if position_manager.is_any_open():
            trades += position_manager.close_all()

    # Visualize strategy
    # See available colours here
    # https://matplotlib.org/stable/gallery/color/named_colors.html
    output.visualise("Slow EMA", slow_ema, colour="forestgreen")
    output.visualise("Fast EMA", fast_ema, colour="limegreen")

    return trades, output


def create_trading_universe(
        client: Client,
        execution_context: ExecutionContext) -> TradingStrategyUniverse:
    """Creates the trading universe where we are trading.

    If `execution_context.live_trading` is true then this function is called for
    every execution cycle. If we are backtesting, then this function is
    called only once at the start of backtesting and the `decide_trades`
    need to deal with new and deprecated trading pairs.

    As we are only trading a single pair, load data for the single pair only.
    """

    # Load all datas we can get for our candle time bucket
    dataset = load_all_data(client, candle_time_bucket, execution_context)

    # Filter down to the single pair we are interested in
    universe = TradingStrategyUniverse.create_single_pair_universe(
        dataset,
        chain_id,
        exchange_slug,
        base_token,
        quote_token,
    )

    return universe

