"""Single position visualisation."""

import logging
import math
import datetime

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
    # TODO: index type may vary here,because get_samples_by_pair() gives (pair_id, timestamp) as index
    # use slow filtering method

    candles = candles.loc[(candles["timestamp"] >= start) & (candles["timestamp"] <= end)]

    assert len(candles) > 0, f"No candles found for {pair.internal_id} between {start} and {end} when plotting position {position}"

    entries = []

    cumulative_value = cumulative_quantity = pnl = 0

    for trade in position.trades.values():
        cumulative_quantity += trade.executed_quantity
        cumulative_value +=  math.copysign(trade.get_value(), trade.executed_quantity)
        entry = {
            "timestamp": trade.opened_at,
            "value": cumulative_value,
            "quantity": cumulative_quantity,
            "delta": trade.executed_quantity,
        }
        entries.append(entry)

    if not position.closed_at:
        # Mark the unrealised position at the end of the backtest
        entry = {
            "timestamp": end_at,
            "value": cumulative_value,
            "quantity": cumulative_quantity,
            "delta": 0,
        }
        entries.append(entry)

    df = pd.DataFrame(entries)
    df = df.set_index("timestamp")

    # Get price data for the position
    price_df = pd.DataFrame({
        "price": candles["close"],
        "timestamp": candles["timestamp"],
    })
    price_df = price_df.set_index("timestamp")

    joined_df = pd.merge(df, price_df, left_index=True, right_index=True, how='left')
    joined_df["delta"] = joined_df["delta"].fillna(0)

    joined_df = joined_df.resample(time_bucket.to_pandas_timedelta()).ffill()

    joined_df["pnl"] = joined_df["delta"].shift(1) * joined_df["price"]

    import ipdb ; ipdb.set_trace()

    return joined_df
