"""Stop loss trade logic.

Logic for managing position stop loss/take profit signals.
"""
import datetime
from typing import List, Dict

from tradingstrategy.candle import CandleSampleUnavailable

from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeType
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.pricing_model import PricingModel


def check_position_triggers(
        position_manager: PositionManager,
) -> List[TradeExecution]:
    """Generate trades that depend on real-time price signals.

    - Stop loss

    - Take profit

    Does not do anything, unless the execution model overrides this.

    - Get the real-time price of an assets that are currently hold in the portfolio

    - Generate stop loss/take profit signals for trades

    :param position_manager:
        Encapsulates the current state, universe for closing positions

    :param cycle_debug_data:
        The debug data of previous cycle
    """

    ts: datetime.datetime = position_manager.timestamp
    state: State = position_manager.state
    pricing_model: PricingModel = position_manager.pricing_model

    positions = state.portfolio.get_open_positions()

    trades = []

    for p in positions:

        if not p.needs_real_time_price():
            # This position does not have take profit/stop loss set
            continue

        assert p.is_long(), "Stop loss supported only for long positions"

        try:
            price = pricing_model.get_sell_price(ts, p.pair, p.get_quantity())
        except CandleSampleUnavailable:
            # Backtest does not have price available for this timestamp,
            # because there has not been any trades.
            # Because there has not been any trades, we assume price has not moved
            # and any position trigger does not need to be executed.
            price = None

        if price is not None:
            if p.take_profit:
                if price >= p.take_profit:
                    trades.append(position_manager.close_position(p, TradeType.take_profit))

            if p.stop_loss:
                if price <= p.stop_loss:
                    trades.append(position_manager.close_position(p, TradeType.stop_loss))

    return trades



