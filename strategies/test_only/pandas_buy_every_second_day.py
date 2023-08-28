"""Unit test trading strategy implementation.

A strategy framework test that

- operates on a single trading pair

- buys on odd days

- sells on even days

- draws EMA lines (even though not used), so can be used to test visualisation

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

# Tell what trade execution engine version this strategy needs to use
trading_strategy_engine_version = "0.1"

# What kind of strategy we are running.
# This tells we are going to use
trading_strategy_type = StrategyType.managed_positions

# Strategy keeps its cash in BUSD
reserve_currency = ReserveCurrency.busd

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

    cash = state.portfolio.get_cash()
    assert cash > 0, "You did not top up the backtest simulation"

    # 30 days EMA
    candles: pd.DataFrame = universe.candles.get_single_pair_data(sample_count=30)
    close = candles["close"]

    slow_ema = close.ewm(span=30).mean().iloc[-1]
    fast_ema = close.ewm(span=10).mean().iloc[-1]

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

    visualisation = state.visualisation
    visualisation.plot_indicator(timestamp, "Slow EMA", PlotKind.technical_indicator_on_price, slow_ema, colour="forestgreen")
    visualisation.plot_indicator(timestamp, "Fast EMA", PlotKind.technical_indicator_on_price, fast_ema, colour="limegreen")

    return trades