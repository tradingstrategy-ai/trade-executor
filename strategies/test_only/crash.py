"""See test_webhook_main_loop_crash.py"""

import datetime
import time

import pandas as pd

from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.strategy_module import StrategyType, TradeRouting, ReserveCurrency

from typing import Optional, List, Dict
from tradeexecutor.strategy.trading_strategy_universe import load_pair_data_for_single_exchange, TradingStrategyUniverse
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradingstrategy.client import Client

from tradeexecutor.strategy.universe_model import UniverseOptions

TRADING_STRATEGY_ENGINE_VERSION = "0.1"

TRADING_STRATEGY_TYPE = StrategyType.managed_positions

TRADE_ROUTING = TradeRouting.pancakeswap_usdc

TRADING_STRATEGY_CYCLE = CycleDuration.cycle_1d

RESERVE_CURRENCY = ReserveCurrency.usdc

CANDLE_TIME_BUCKET = TimeBucket.d1

CHAIN_ID = ChainId.bsc

EXCHANGE_SLUG = "pancakeswap-v2"

TRADING_PAIR = ("ETH", "USDC")


class CrashTest(Exception):
    """Crash the strategy."""


def decide_trades(
        timestamp: pd.Timestamp,
        universe: Universe,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: Dict) -> List[TradeExecution]:
    time.sleep(2)   # We need to wait the pytest test case / web server to catch up with the output
    raise CrashTest("Boom")


def create_trading_universe(
        ts: datetime.datetime,
        client: Client,
        execution_context: ExecutionContext,
        universe_options: UniverseOptions,
) -> TradingStrategyUniverse:

    dataset = load_pair_data_for_single_exchange(
        client,
        execution_context,
        CANDLE_TIME_BUCKET,
        CHAIN_ID,
        EXCHANGE_SLUG,
        [TRADING_PAIR],
        universe_options,
    )

    # Filter down to the single pair we are interested in
    universe = TradingStrategyUniverse.create_single_pair_universe(
        dataset,
        CHAIN_ID,
        EXCHANGE_SLUG,
        TRADING_PAIR[0],
        TRADING_PAIR[1],
    )

    return universe

