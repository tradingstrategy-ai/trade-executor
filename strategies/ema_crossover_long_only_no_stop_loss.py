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
from tradeexecutor.strategy.description import StrategyExecutionDescription
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.pandas_trader.runner import PandasTraderRunner
from tradeexecutor.strategy.pandas_trader.output import StrategyOutput
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.pricing_model import PricingModelFactory
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverseModel, \
    TradingStrategyUniverse
from tradingstrategy.client import Client

from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

# Cannot use Python __name__ here because the module is dynamically loaded
logger = logging.getLogger("ema_crossover")

# Time bucket
time_bucket = TimeBucket.h4

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


class SinglePairUniverseModel(TradingStrategyUniverseModel):
    """Create a trading universe that contains only single trading pair.

    This trading pair is selected based on the parameters in the script above.
    """

    def construct_universe(self, execution_model: ExecutionModel, live) -> TradingStrategyUniverse:
        dataset = self.load_data(time_bucket, live)
        universe = TradingStrategyUniverse.create_single_pair_universe(
            dataset,
            chain_id,
            exchange_slug,
            base_token,
            quote_token,
        )
        self.log_universe(universe.universe)
        return universe


def strategy_factory(
        *ignore,
        execution_model: UniswapV2ExecutionModelVersion0,
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
    universe_model = SinglePairUniverseModel(client, timed_task_context_manager)

    runner = PandasTraderRunner(
        brain=decide_trade,
        timed_task_context_manager=timed_task_context_manager,
        execution_model=execution_model,
        approval_model=approval_model,
        revaluation_method=revaluation_method,
        sync_method=sync_method,
        pricing_model_factory=pricing_model_factory,
    )

    return StrategyExecutionDescription(
        time_bucket=TimeBucket.d1,
        universe_model=universe_model,
        runner=runner,
    )


__all__ = [strategy_factory]