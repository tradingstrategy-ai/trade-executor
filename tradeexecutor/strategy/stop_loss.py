"""Stop loss trade logic.

Logic for managing position stop loss/take profit signals.
"""
import datetime
import logging
from decimal import Decimal
from io import StringIO
from typing import List, Dict

from tradingstrategy.candle import CandleSampleUnavailable

from tradeexecutor.strategy.generic_pricing_model import GenericPricingModel
from tradeexecutor.state.position import TradingPosition, TriggerPriceUpdate, CLOSED_POSITION_DUST_EPSILON
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

    What does this do:

    - Get the real-time price of an assets that are currently hold in the portfolio

    - Update dynamic stop loss/take profits like trailing stop loss

    - Use mid-price to check for the trigger price threshold

    - Generate stop loss/take profit signals for trades

    See related position attributes

    - :py:attr:`tradeexecutor.state.position.TradingPosition.stop_loss`

    - :py:attr:`tradeexecutor.state.position.TradingPosition.take_profit`

    - :py:attr:`tradeexecutor.state.position.TradingPosition.trailing_stop_loss`

    :param position_manager:
        Encapsulates the current state, universe for closing positions

    :param epsilon:
        The rounding error to zero

    :return:
        List of triggered trades for all positions, like market sell on a stop loss triggered.
    """

    ts: datetime.datetime = position_manager.timestamp
    state: State = position_manager.state

    positions = state.portfolio.get_open_positions()

    trades = []

    for p in positions:

        pricing_model = position_manager.pricing_model
        
        assert isinstance(pricing_model, PricingModel | GenericPricingModel), f"Got bad pricing model: {pricing_model} {type(pricing_model)}"

        if not p.has_trigger_conditions():
            # This position does not have take profit/stop loss set
            continue

        assert any([
            p.is_long(),
            p.is_short() and p.is_leverage(),
        ]), "Trigger only supports long and leveraged short positions"

        size = p.get_quantity()

        if size == 0:
            logger.warning("Encountered open position without token quantity: %s. Quantity is %s.", p, size)
            continue

        # TODO: Tracking down math bug
        if not isinstance(size, Decimal):
            logger.warning("Got bad size %s: %s", size, size.__class__)
            size = Decimal(size)

        spot_pair = p.pair
        if p.is_short():
            spot_pair = spot_pair.underlying_spot_pair
            
        try:
            mid_price = pricing_model.get_mid_price(ts, spot_pair)
            expected_sell_price = pricing_model.get_sell_price(ts, spot_pair, abs(size))
        except CandleSampleUnavailable:
            # Backtest does not have price available for this timestamp,
            # because there has not been any trades.
            # Because there has not been any trades, we assume price has not moved
            # and any position trigger does not need to be executed.
            continue

        assert type(mid_price) == float, f"Received bad mid-price: {mid_price} {type(mid_price)}"

        trigger_type = trigger_price = None
        stop_loss_before = stop_loss_after = None

        # Check for trailing stop loss updates
        if p.trailing_stop_loss_pct:
            assert p.is_long(), "Traing stop loss only supported for long positions at the moment"
            new_stop_loss = mid_price * p.trailing_stop_loss_pct
            if not p.stop_loss or (new_stop_loss > p.stop_loss):
                stop_loss_before = p.stop_loss
                stop_loss_after = new_stop_loss

        # Update dynamic triggers if needed
        if stop_loss_after is not None:
            assert stop_loss_after > 0
            trigger_update = TriggerPriceUpdate(
                ts,
                mid_price,
                stop_loss_before,
                stop_loss_after,
                None,  # No trailing take profits yet
                None,
            )
            p.trigger_updates.append(trigger_update)
            p.stop_loss = stop_loss_after

        # Check we need to close position for take profit
        if p.take_profit:
            if p.is_long():
                if mid_price >= p.take_profit:
                    trigger_type = TradeType.take_profit
                    trigger_price = p.take_profit
                    trades.extend(position_manager.close_position(p, TradeType.take_profit))
            else:
                if mid_price <= p.take_profit:
                    trigger_type = TradeType.take_profit
                    trigger_price = p.take_profit
                    trades.extend(position_manager.close_position(p, TradeType.take_profit))

        # Check we need to close position for stop loss
        if p.stop_loss:
            if p.is_long():
                if mid_price <= p.stop_loss:
                    trigger_type = TradeType.stop_loss
                    trigger_price = p.stop_loss
                    trades.extend(position_manager.close_position(p, TradeType.stop_loss))
            else:
                if mid_price >= p.stop_loss:
                    trigger_type = TradeType.stop_loss
                    trigger_price = p.stop_loss
                    trades.extend(position_manager.close_position(p, TradeType.stop_loss))

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



