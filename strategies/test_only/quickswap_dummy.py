"""A strategy that does not do any trades

- Unit test trading strategy implementation

- Load up Polygon WMATIC-USDC pair

- Load 1h data for 1 minute candles

- Used in test_strategy_cycle_trigger.py
"""

import datetime
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

# NOTE: this setting has currently no effect
TRADING_STRATEGY_ENGINE_VERSION = "0.1"

# NOTE: this setting has currently no effect
TRADING_STRATEGY_TYPE = StrategyType.managed_positions

TRADE_ROUTING = TradeRouting.quickswap_usdc

TRADING_STRATEGY_CYCLE = CycleDuration.cycle_1m

RESERVE_CURRENCY = ReserveCurrency.usdc

CANDLE_TIME_BUCKET = TimeBucket.m1

CHAIN_ID = ChainId.polygon

EXCHANGE_SLUG = "quickswap"

TRADING_PAIR = ("WMATIC", "USDC")


def decide_trades(
        timestamp: pd.Timestamp,
        universe: Universe,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: Dict) -> List[TradeExecution]:
    return []


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
        required_history_period=datetime.timedelta(hours=1),
    )

    # Filter down to the single pair we are interested in
    universe = TradingStrategyUniverse.create_single_pair_universe(
        dataset,
        CHAIN_ID,
        EXCHANGE_SLUG,
        TRADING_PAIR[0],
        TRADING_PAIR[1],
    )
    # We use 1 minutes candles and download 1 hour historical data
    universe.required_history_period = datetime.timedelta(hours=1)
    return universe