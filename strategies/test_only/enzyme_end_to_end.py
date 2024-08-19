"""Dummy strategy used in Enzyme end-to-end tests.

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

trading_strategy_engine_version = "0.1"
trading_strategy_type = StrategyType.managed_positions
trade_routing = TradeRouting.user_supplied_routing_model
trading_strategy_cycle = CycleDuration.cycle_1s
reserve_currency = ReserveCurrency.usdc

management_fee=0.01
trading_strategy_protocol_fee=0.01
strategy_developer_fee=0.01

def decide_trades(
        timestamp: pd.Timestamp,
        universe: Universe,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: Dict) -> List[TradeExecution]:

    # Create a position manager helper class that allows us easily to create
    # opening/closing trades for different positions
    position_manager = PositionManager(timestamp, universe, state, pricing_model, default_slippage_tolerance=0.02)

    pair = universe.pairs.get_single()

    assert pair.pair_id > 0

    cash = state.portfolio.get_cash()

    cycle_number = cycle_debug_data["cycle"]

    trades = []

    # For odd seconds buy, for even seconds sell
    if cycle_number % 2 == 0:
        # buy on even days
        if not position_manager.is_any_open():
            position_size = 0.10
            buy_amount = cash * position_size
            trades += position_manager.open_1x_long(pair, buy_amount)
    else:
        # sell on odd days
        if position_manager.is_any_open():
            trades += position_manager.close_all()

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