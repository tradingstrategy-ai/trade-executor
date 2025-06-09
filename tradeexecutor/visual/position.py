"""Single position visualisation."""

import logging
import math
from datetime import datetime

import pandas as pd
from plotly.graph_objs import Figure

from tradeexecutor.state.position import TradingPosition
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradingstrategy.timebucket import TimeBucket

logger = logging.getLogger(__name__)



def calculate_position_curve(
    strategy_universe: TradingStrategyUniverse,
    position: TradingPosition,
    time_bucket: TimeBucket,
    end_at: datetime.datetime | None = None,
) -> pd.DataFrame:
    """Calculatea visualisation dato for a single position.

    - Price
    - Position size
    - PnL

    :param end_at:
        End at timestamp for positions that are still open/were open at the end of the backtest.

    :return:
        DataFrame with columns:
    """

    assert isinstance(strategy_universe, TradingStrategyUniverse), f"Expected TradingStrategyUniverse, got {type(strategy_universe)}"
    assert isinstance(position, TradingPosition), f"Expected TradingPosition, got {type(position)}"

    start = position.opened_at
    end = position.closed_at or end_at

    assert end is not None, f"Position must have an end date or be closed at the end of the backtest: {position}"

    start = time_bucket.floor(pd.Timestamp(start))
    end = time_bucket.ceil(pd.Timestamp(end))
    pair = position.pair

    candles = strategy_universe.data_universe.candles.get_samples_by_pair(pair.internal_id)
    candles = candles.loc[start:end]

    assert len(candles) > 0, f"No candles found for {pair.internal_id} between {start} and {end} when plotting position {position}"

    entries = []

    cumulative_value = cumulative_quantity = pnl = 0

    for trade in position.trades.values():

        cumulative_quantity += trade.executed_quantity
        cumulative_value += trade.get_value() * math.copysign(trade.executed_quantity)

        entry = {
            "timestamp": trade.opened_at,
            "value": cumulative_value,
            "quantity": cumulative_quantity,
        }

        entries.append(entry)

    df = pd.DataFrame(entries)
    df = df.set_index("timestamp")
    df["price"] = candles["close"]

    df = df.resample(time_bucket.to_pandas_timedelta()).agg({
        "value": "last",
        "quantity": "last",
        "price": "last",
    })

    return df