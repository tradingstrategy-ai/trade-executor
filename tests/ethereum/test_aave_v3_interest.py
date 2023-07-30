import datetime

import pytest

from tradeexecutor.ethereum.aave_v3.interests import (
    calculate_loan_interests_raw, 
    estimate_loan_interests_raw, 
    get_aave_v3_candles_for_period, 
    get_aave_v3_raw_data_for_period,
)


def test_get_aave_v3_candle_for_period(persistent_test_client):
    client = persistent_test_client

    candles = get_aave_v3_candles_for_period(
        client,
        "USDC",
        137,
        datetime.datetime(2023, 1, 1),
        datetime.datetime(2023, 2, 1),
    )

    print(candles)
    assert len(candles) > 0


def test_get_aave_v3_raw_data_for_period(persistent_test_client):
    client = persistent_test_client

    df = get_aave_v3_raw_data_for_period(
        client,
        "USDC",
        137,
        datetime.datetime(2023, 1, 1),
        datetime.datetime(2023, 2, 1),
    )

    print(df)
    assert len(df) > 0

@pytest.mark.parametrize(
    "token, chain_id, amount, start_time, end_time",
    (
        # (
        #     "USDC",
        #     137,
        #     10000 * 10**6,
        #     datetime.datetime(2023, 1, 1),
        #     datetime.datetime(2023, 1, 3),
        # ),
        # (
        #     "USDT",
        #     137,
        #     10000 * 10**6,
        #     datetime.datetime(2023, 2, 1),
        #     datetime.datetime(2023, 2, 2),
        # ),
        # (
        #     "USDT",
        #     137,
        #     10000 * 10**6,
        #     datetime.datetime(2023, 3, 10),
        #     datetime.datetime(2023, 3, 13),
        # ),
        (
            "USDT",
            1,
            5000000000,
            datetime.datetime(2023, 2, 13, 18, 51, 35),
            datetime.datetime(2023, 2, 13, 18, 58, 11),
        ),
        (
            "USDC",
            1,
            100000000000,
            datetime.datetime(2023, 2, 7, 18, 56, 11),
            datetime.datetime(2023, 2, 7, 20, 34, 59),
        ),
        (
            "DAI",
            1,
            3100000000000000000000,
            datetime.datetime(2023, 2, 10, 3, 44, 23),
            datetime.datetime(2023, 2, 13, 16, 30, 23),
        ),
        (
            "USDT",
            137,
            10000 * 10**6,
            datetime.datetime(2023, 3, 10),
            datetime.datetime(2023, 3, 13),
        ),
    ),
)
def test_estimate_loan_interests(persistent_test_client, token, chain_id, amount, start_time, end_time):
    """
    Compare interests calculated from "correct" method and estimation
    """
    client = persistent_test_client

    print()
    print(f"Borrow {token} with amount {amount} from {start_time} to {end_time}")

    calculated_interests = calculate_loan_interests_raw(
        client,
        token,
        chain_id,
        amount,
        start_time,
        end_time,
    )

    print(f"Calculated: {calculated_interests}")
    assert calculated_interests > 0

    estimated_interests = estimate_loan_interests_raw(
        client,
        token,
        chain_id,
        amount,
        start_time,
        end_time,
    )

    print(f"Estimated: {estimated_interests}")
    assert estimated_interests > 0

    # assert estimated_interests > calculated_interests

    print("Diff:", estimated_interests * 100 / calculated_interests - 100, "%")
    print()
