"""Dummy strategy used in Enzyme end-to-end tests.

"""

from typing import Dict, List

import pandas as pd

from tradeexecutor.strategy.pricing_model import PricingModel
from tradingstrategy.universe import Universe

from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.strategy_type import StrategyType
from tradeexecutor.state.visualisation import PlotKind
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.reserve_currency import ReserveCurrency

trading_strategy_engine_version = "0.1"

trading_strategy_type = StrategyType.managed_positions

reserve_currency = ReserveCurrency.usdc

# How much of the cash to put on a single trade
position_size = 0.10


def decide_trades(
        timestamp: pd.Timestamp,
        universe: Universe,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: Dict) -> List[TradeExecution]:

    assert timestamp.minute == 0
    assert timestamp.second == 0
    assert timestamp.hour == 0

    # Create a position manager helper class that allows us easily to create
    # opening/closing trades for different positions
    position_manager = PositionManager(timestamp, universe, state, pricing_model)

    pair = universe.pairs.get_single()

    assert pair.pair_id > 0

    cash = state.portfolio.get_current_cash()

    # https://stackoverflow.com/a/623312/315168
    day_number = timestamp.to_pydatetime().timetuple().tm_yday

    trades = []

    if day_number % 2 == 0:
        # buy on even days
        if not position_manager.is_any_open():
            buy_amount = cash * position_size
            trades += position_manager.open_1x_long(pair, buy_amount)
    else:
        # sell on odd days
        if position_manager.is_any_open():
            trades += position_manager.close_all()

    return trades