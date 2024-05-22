"""Stop loss trade logic.

Logic for managing position stop loss/take profit signals.
"""
import datetime
import logging
from decimal import Decimal
from io import StringIO
from typing import List, Dict

from tradingstrategy.candle import CandleSampleUnavailable

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

    name = "Stop loss" if condition == TradeType.stop_loss else "Take profit"

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

    - Any trade.triggers like market limit

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
    pricing_model: PricingModel = position_manager.pricing_model

    positions = state.portfolio.get_open_positions()

    trades = []

    for p in positions:

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

        spot_pair = p.pair.get_pricing_pair()
            
        try:
            mid_price = pricing_model.get_mid_price(ts, spot_pair)
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
            stop_loss_before = p.stop_loss
            
            if p.is_long():
                new_stop_loss = mid_price * p.trailing_stop_loss_pct
            else:
                new_stop_loss = mid_price * (2 - p.trailing_stop_loss_pct)

            if any([
                not p.stop_loss,
                p.is_long() and new_stop_loss > p.stop_loss,
                p.is_short() and new_stop_loss < p.stop_loss,
            ]):
                stop_loss_before = p.stop_loss
                stop_loss_after = new_stop_loss

        # Update dynamic triggers if needed
        if stop_loss_after is not None:
            assert stop_loss_after > 0
            if p.is_long():
                assert stop_loss_after > stop_loss_before
            else:
                assert stop_loss_after < stop_loss_before

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
            if any([
                p.is_long() and mid_price >= p.take_profit,
                p.is_short() and mid_price <= p.take_profit,
            ]):
                trigger_type = TradeType.take_profit
                trigger_price = p.take_profit
                trades.extend(position_manager.close_position(p, TradeType.take_profit))
            

        # Check we need to close position for stop loss
        if p.stop_loss:
            if any([
                p.is_long() and mid_price <= p.stop_loss,
                p.is_short() and mid_price >= p.stop_loss,
            ]):
                trigger_type = TradeType.stop_loss
                trigger_price = p.stop_loss
                notes = f"Stoploss:{trigger_price} mid-price:{mid_price}"
                trades.extend(position_manager.close_position(p, TradeType.stop_loss, notes=notes))

        if trigger_type:
            # We got triggered
            expected_sell_price = pricing_model.get_sell_price(ts, spot_pair, abs(size))
            report_position_triggered(
                p,
                trigger_type,
                trigger_price,
                mid_price,
                expected_sell_price.price,
            )

    return trades



