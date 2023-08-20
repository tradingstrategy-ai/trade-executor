import datetime
from decimal import Decimal

import pandas as pd

from eth_defi.aave_v3.rates import SECONDS_PER_YEAR

from tradingstrategy.lending import LendingReserve, LendingCandleType
from tradingstrategy.client import Client
from tradingstrategy.timebucket import TimeBucket


def get_aave_v3_candles_for_period(
    client: Client,
    token: str,
    chain_id: int,
    start_time: datetime.datetime,
    end_time: datetime.datetime = datetime.datetime.utcnow(),
):
    reserve_universe = client.fetch_lending_reserve_universe()
    reserve: LendingReserve = reserve_universe.get_reserve_by_symbol_and_chain(token, chain_id)

    lending_candles = client.fetch_lending_candles_by_reserve_id(
        reserve.reserve_id,
        bucket=TimeBucket.h1,
        candle_type=LendingCandleType.variable_borrow_apr,
        start_time=start_time,
        end_time=end_time,
    )

    return lending_candles


def get_aave_v3_raw_data_for_period(
    client: Client,
    token: str,
    chain_id: int,
    start_time: datetime.datetime,
    end_time: datetime.datetime = datetime.datetime.utcnow(),
):
    reserve_universe = client.fetch_lending_reserve_universe()
    reserve: LendingReserve = reserve_universe.get_reserve_by_symbol_and_chain(token, chain_id)

    # NOTE: This is very slow the 1st time
    pq_table = client.fetch_lending_reserves_all_time()

    # NOTE: This is very slow
    df = pq_table.to_pandas()
    df = df.set_index(pd.DatetimeIndex(df["timestamp"]))

    # filter df by reserve_id
    df = df[
        (df["reserve_id"] == reserve.reserve_id)
        & (df["timestamp"] >= start_time)
        & (df["timestamp"] <= end_time)
    ]

    df.sort_index(inplace=True)

    return df


def calculate_loan_interests_raw(
    client: Client,
    token: str,
    chain_id: int,
    amount: int,
    start_time: datetime.datetime,
    end_time: datetime.datetime = datetime.datetime.utcnow(),
) -> Decimal:
    """
    Adapted from eth_defi.aave_v3.rates

    Calculate the accrued interest for a loan in a period of time with variable borrow rate

    Note that this is only for backtesting at the moment since it depends on historical data
    """

    df = get_aave_v3_raw_data_for_period(client, token, chain_id, start_time, end_time)

    if len(df) <= 0:
        raise ValueError(f"No data found in date range {start_time} - {end_time}")
    
    # Loan starts on first row of the DataFrame
    actual_start_time = df.index[0]
    start_variable_borrow_index = Decimal(df["variable_borrow_index"][0])

    # Loan ends on last row of the DataFrame
    actual_end_time = df.index[-1]
    end_variable_borrow_index = Decimal(df["variable_borrow_index"][-1])

    interest = (end_variable_borrow_index / start_variable_borrow_index) * amount - amount

    return interest


def estimate_loan_interests_raw(
    client: Client,
    token: str,
    chain_id: int,
    amount: int,
    start_time: datetime.datetime,
    end_time: datetime.datetime = datetime.datetime.utcnow(),
) -> Decimal:
    df = get_aave_v3_candles_for_period(client, token, chain_id, start_time, end_time)

    # print(df)
    
    if len(df) <= 0:
        raise ValueError(f"No data found in date range {start_time} - {end_time}")
    
    duration = Decimal((end_time - start_time).total_seconds())
    
    df["avg"] = df[["high", "low"]].mean(axis=1)
    interest = amount * Decimal(df["avg"].mean() / 100) * duration / SECONDS_PER_YEAR
    # print(f"Estimated interest using avg high-low APR: {interest}")

    # df["hourly_interest"] = amount * (df["avg"] / 100) * 3600 / SECONDS_PER_YEAR_INT
    # interest2 = Decimal(df["hourly_interest"].sum())
    # print(f"Estimated interest using hourly estimate: {interest2}")

    return interest
