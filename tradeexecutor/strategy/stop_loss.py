"""Stop loss trade logic.

Logic for managing position stop loss/take profit signals.
"""
import datetime
import logging
from io import StringIO
from typing import List, Dict

from tradingstrategy.candle import CandleSampleUnavailable

from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeType
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.pricing_model import PricingModel


logger = logging.getLogger(__name__)



def report_position_triggered(
        position: TradingPosition,
        condition: TradeType,
        trigger_price: float,
        mid_price: float,
        expected_price: float,
):
    """Write a trade logging output report on a position trigger."""

    buf = StringIO()

    name = "Stop loss" if condition == TradeType.stop_loss else "Trade profit"

    size = position.get_quantity()

    print(f"{name} triggered", file=buf)
    print("", file=buf)
    print(f"   Pair: {position.pair}", file=buf)
    print(f"   Size: {size} {position.pair.base.token_symbol}", file=buf)
    print(f"   Trigger price: {trigger_price} USD", file=buf)
    print(f"   Current mid price: {mid_price} USD", file=buf)
    print(f"   Expected avg closing price: {expected_price} USD", file=buf)

    logger.trade(buf.getvalue())


def check_position_triggers(
        position_manager: PositionManager,
) -> List[TradeExecution]:
    """Generate trades that depend on real-time price signals.

    - Stop loss

    - Take profit

    What dots this do.

    - Get the real-time price of an assets that are currently hold in the portfolio

    - Use mid-price to check for the trigger price threshold

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

        if not p.has_trigger_conditions():
            # This position does not have take profit/stop loss set
            continue

        assert p.is_long(), "Stop loss supported only for long positions"

        size = p.get_quantity()

        try:
            mid_price = pricing_model.get_mid_price(ts, p.pair)
            expected_sell_price = pricing_model.get_sell_price(ts, p.pair, size)
        except CandleSampleUnavailable:
            # Backtest does not have price available for this timestamp,
            # because there has not been any trades.
            # Because there has not been any trades, we assume price has not moved
            # and any position trigger does not need to be executed.
            continue

        trigger_type = trigger_price = None

        if p.take_profit:
            if mid_price >= p.take_profit:
                trigger_type = TradeType.take_profit
                trigger_price = p.take_profit
                trades.append(position_manager.close_position(p, TradeType.take_profit))
        elif p.stop_loss:
            if mid_price <= p.stop_loss:
                trigger_type = TradeType.stop_loss
                trigger_price = p.stop_loss
                trades.append(position_manager.close_position(p, TradeType.stop_loss))

        if trigger_type:
            # We got triggered
            report_position_triggered(
                p,
                trigger_type,
                trigger_price,
                mid_price,
                expected_sell_price,
            )

    return trades



