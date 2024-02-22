"""Stop loss analysis

- Analyse how we do trigger stop loss

"""
import pandas as pd

from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradingstrategy.utils.format import format_price, format_percent, format_percent_2_decimals


def _extract_stop_loss_data(p: TradingPosition) -> dict:
    """Create a single line of stop-loss analytics."""

    trigger_updates = p.trigger_updates
    first_update = trigger_updates[0]
    last_update = trigger_updates[-1]
    # start_price = first_update.mid_price
    opening_price = p.get_opening_price()
    closing_price = p.get_closing_price()
    lowest_price = min(t.mid_price for t in trigger_updates)
    highest_price = max(t.mid_price for t in trigger_updates)
    opening_stop_loss = first_update.stop_loss_after
    lowest_stop_loss = min(t.stop_loss_after for t in trigger_updates)
    highest_stop_loss = max(t.stop_loss_after for t in trigger_updates)

    initial_stop_loss =  (opening_stop_loss - opening_price) / opening_price

    duration = p.get_duration()

    drift_up = (highest_stop_loss - opening_stop_loss) / opening_stop_loss
    drift_down = (lowest_stop_loss - opening_stop_loss) / opening_stop_loss
    if p.is_closed():
        profit = p.get_realised_profit_percent()
    else:
        profit = 0

    return {
        "position_id": p.position_id,
        "trading_pair": p.pair.get_ticker(),
        "triggered": p.is_stop_loss(),
        "updates": len(trigger_updates),
        "duration": duration,
        "profit": format_percent_2_decimals(profit),
        "initial": format_percent_2_decimals(initial_stop_loss),
        "drift_up": format_percent_2_decimals(drift_up),
        "drift_down": format_percent_2_decimals(drift_down),
        "opening_stop_loss": format_price(opening_stop_loss, decimals=2),
        "lowest_stop_loss": format_price(lowest_stop_loss, decimals=0),
        "highest_stop_loss": format_price(highest_stop_loss, decimals=0),
        "opening_price": format_price(opening_price, decimals=2),
        "closing_price": format_price(closing_price, decimals=2),
        "lowest_price": format_price(lowest_price, decimals=0),
        "highest_price": format_price(highest_price, decimals=0),
    }


def analyse_stop_losses(state: State) -> pd.DataFrame:
    """Create a table with stop loss data.

    - Allows manually to examine stop loss trigger performance per each position

    - This table can ge aggregated to create statistics of stop loss performance

    :return:
        DataFrame where each row is a position containing human-readable data about the stop loss performance of this position.
    """


    lines = [_extract_stop_loss_data(p) for p in state.portfolio.get_all_positions()]
    df = pd.DataFrame(lines)
    df = df.set_index("position_id").sort_index()
    return df


def analyse_trigger_updates(position: TradingPosition) -> pd.DataFrame:
    """Analyse trigger updates of a single position.

    - Figure out what's going on with the stop loss trigger of a single position

    :return:
        DataFrame where each row is a position containing human-readable data about the stop loss performance of this position.
    """

    assert isinstance(position, TradingPosition)
    lines = [u.to_dict() for u in position.trigger_updates]
    df = pd.DataFrame(lines)
    df = df.set_index("timestamp").sort_index()
    return df
