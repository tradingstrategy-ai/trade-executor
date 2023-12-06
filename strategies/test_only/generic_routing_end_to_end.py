"""Dummy strategy used in generic routing end-to-end tests.

- Spot + short trades
"""
import datetime
from typing import Dict, List

import pandas as pd

from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.trading_strategy_universe import load_all_data, TradingStrategyUniverse
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradingstrategy.client import Client, BaseClient
from tradingstrategy.pair import DEXPair
from tradingstrategy.testing.uniswap_v2_mock_client import UniswapV2MockClient
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.strategy_type import StrategyType
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.reserve_currency import ReserveCurrency

trading_strategy_engine_version = "0.3"
trading_strategy_type = StrategyType.managed_positions
trade_routing = TradeRouting.default
trading_strategy_cycle = CycleDuration.cycle_1s
reserve_currency = ReserveCurrency.usdc


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
    assert isinstance(client, UniswapV2MockClient), f"Looks like we are not running on EVM testing backend. Got: {client}"

    # Load exchange and pair data for a single pair
    dataset = load_all_data(
        client,
        TimeBucket.not_applicable,
        execution_context,
        universe_options,
    )

    # Create a trading universe for our test EVM backend Uniswap v2 deployment
    assert len(dataset.pairs) == 1
    pair_data = dataset.pairs.iloc[0]
    pair: DEXPair = DEXPair.from_dict(pair_data.to_dict())

    universe = TradingStrategyUniverse.create_single_pair_universe(
        dataset,
        pair.chain_id,
        pair.exchange_slug,
        pair.base_token_symbol,
        pair.quote_token_symbol,
    )
    return universe
