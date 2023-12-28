"""Dummy strategy used in generic routing end-to-end tests.

- Spot + short trades
"""
import datetime
from typing import List

import pandas as pd

from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, load_partial_data
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client, BaseClient
from tradingstrategy.lending import LendingProtocolType

from tradingstrategy.timebucket import TimeBucket

from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager

trading_strategy_engine_version = "0.3"
trading_strategy_cycle = CycleDuration.cycle_1s


def decide_trades(
    timestamp: pd.Timestamp,
    strategy_universe: TradingStrategyUniverse,
    state: State,
    pricing_model: PricingModel,
    cycle_debug_data: dict
) -> List[TradeExecution]:
    # Every second day buy spot,
    # every second day short

    trades = []
    position_manager = PositionManager(timestamp, strategy_universe, state, pricing_model)
    cycle = cycle_debug_data["cycle"]
    pairs = strategy_universe.data_universe.pairs
    spot_eth = pairs.get_pair_by_human_description((ChainId.polygon, "uniswap-v3", "WETH", "USDC", 0.0005))

    if position_manager.is_any_open():
        trades += position_manager.close_all()

    if cycle % 2 == 0:
        # Spot day
        trades += position_manager.open_spot(spot_eth, 100.0)
    else:
        # Short day
        trades += position_manager.open_short(spot_eth, 150.0)

    return trades


def create_trading_universe(
        ts: datetime.datetime,
        client: BaseClient,
        execution_context: ExecutionContext,
        universe_options: UniverseOptions,
):

    pairs = [
        (ChainId.polygon, "uniswap-v3", "WETH", "USDC", 0.0005),
        (ChainId.polygon, "quickswap", "WMATIC", "USDC", 0.0030),
    ]

    reverses = [
        (ChainId.polygon, LendingProtocolType.aave_v3, "WETH"),
        (ChainId.polygon, LendingProtocolType.aave_v3, "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"),
    ]

    dataset = load_partial_data(
        client,
        execution_context=execution_context,
        time_bucket=TimeBucket.h1,
        pairs=pairs,
        universe_options=universe_options,
        lending_reserves=reverses,
        required_history_period=datetime.timedelta(days=7),
    )

    strategy_universe = TradingStrategyUniverse.create_from_dataset(
        dataset,
        reserve_asset="USDC",
    )

    return strategy_universe
