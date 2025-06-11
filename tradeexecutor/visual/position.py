"""Single position visualisation."""

import logging
import math
import datetime

import numpy as np
import pandas as pd
from pyasn1_modules.rfc6031 import id_pskc_valueMAC

from tradeexecutor.state.position import TradingPosition
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradingstrategy.timebucket import TimeBucket

logger = logging.getLogger(__name__)


def _realised(row):
    if row["delta"] < 0:
        avg_entry_price = row["value"] / row["quantity"] if row["quantity"] != 0 else 0
        return (row["executed_price"] - avg_entry_price) * abs(row["delta"]) - row["fee"]
    return 0

def calculate_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate PnL for a position DataFrame.

    - Add PnL columns to the the DataFrame

    :param df:
        DataFrame with columns: timestamp, value, quantity, delta, price

    :return:
        DataFrame with additional columns: pnl, cumulative_pnl
    """
    # Track cost basis of the position
    # df['trade_cost'] = df['delta'] * df['executed_price']
    # Cumulative cost of the position
    df["unrealised_pnl"] = (df["mark_price"] - df["avg_price"]) * df["quantity"]
    df["realised_pnl"] = (df["mark_price"] - df["avg_price"]) * df["delta"]
    return df




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

    TODO: Missing vault perf and managemetn fees.

    Example data:

    .. code-block:: plain-text



    :param end_at:
        End at timestamp for positions that are still open/were open at the end of the backtest.

    :return:
        DataFrame with columns:
    """

    assert isinstance(strategy_universe, TradingStrategyUniverse), f"Expected TradingStrategyUniverse, got {type(strategy_universe)}"
    assert isinstance(position, TradingPosition), f"Expected TradingPosition, got {type(position)}"

    assert position.is_spot() or position.is_vault(), "Interest calculations missing"
    assert not position.is_short(), "This function has not been tested for short positions yet"

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

    cumulative_quantity = cumulative_cost = avg_price = 0

    for trade in position.trades.values():
        delta = float(trade.executed_quantity)

        if delta > 0:
            # Buy: increase cost basis
            cumulative_cost += trade.get_value()
            cumulative_quantity += delta
        elif delta < 0:
            # Sell: reduce cost basis proportionally
            if cumulative_quantity > 0:
                avg_price = cumulative_cost / float(cumulative_quantity)
                cost_reduction = avg_price * abs(delta)
                cumulative_cost -= cost_reduction
                cumulative_quantity += delta  # delta is negative here
        else:
            raise NotImplementedError(f"Got a trade withe executed quantity zero: {trade}")

        entry = {
            "timestamp": trade.opened_at,  # When we made a trade
            "quantity": cumulative_quantity,  # Currently hold quantity if assetes
            "delta": delta,  # Positive if we bought, negative if we sold
            "executed_price": trade.executed_price,  # Price we got, fees included
            "fee": trade.lp_fees_paid,  # How much fees we paid
            "cumulative_cost": cumulative_cost,  # The cost of maintaining this position
            "avg_price": avg_price,  # Volume-weighted average price of the position
        }
        entries.append(entry)

    if not position.closed_at:
        # Mark the unrealised position at the end of the backtest.
        # Generate timestamp entry at the end of the backtest,
        # so we can calculate unrealised PnL for this row
        entry = {
            "timestamp": end_at,
            "quantity": cumulative_quantity,
            "delta": 0,
            "executed_price": np.nan,
            "fee": 0,
            "avg_price": avg_price
        }
        entries.append(entry)

    df = pd.DataFrame(entries)
    df = df.set_index("timestamp")

    # Get price data for the position.
    # We create column mark_price which we use to calculate realised and unrealised pnl
    price_df = pd.DataFrame({
        "mark_price": candles["close"],  # Mark price
        "timestamp": candles["timestamp"],
    })
    price_df = price_df.set_index("timestamp")

    joined_df = pd.merge(df, price_df, left_index=True, right_index=True, how='outer')
    joined_df["delta"] = joined_df["delta"].fillna(0)
    joined_df["fee"] = joined_df["fee"].fillna(0)
    joined_df = joined_df.ffill()
    joined_df["value"] = joined_df["mark_price"] * joined_df["quantity"] * (1 - pair.fee)

    joined_df["mark_price"] = joined_df["mark_price"].astype(float)
    joined_df["delta"] = joined_df["delta"].astype(float)
    joined_df["quantity"] = joined_df["quantity"].astype(float)

    def _realised_pnl(row):
        if row["delta"] < 0:
            return (row["mark_price"] - row["avg_price"]) * abs(row["delta"])
        return 0

    joined_df["realised_delta"] = joined_df.apply(_realised_pnl, axis=1)
    joined_df["realised_pnl"] = joined_df["realised_delta"].cumsum()
    joined_df["unrealised_pnl"] = joined_df["value"] - joined_df["cumulative_cost"]
    joined_df["pnl"] = joined_df["unrealised_pnl"] + joined_df["realised_pnl"]

    return joined_df
