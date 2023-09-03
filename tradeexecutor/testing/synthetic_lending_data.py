import datetime
import random

import pandas as pd
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.lending import (
    LendingProtocolType, LendingReserve, LendingCandleUniverse,
    LendingCandleType, LendingReserveUniverse, LendingReserveAdditionalDetails,
)

from tradeexecutor.state.state import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address


def generate_lending_reserve(
    token: AssetIdentifier,
    chain_id: ChainId = ChainId.ethereum,
    internal_id = random.randint(1, 1000),
) -> TradingPairIdentifier:
    """Generate a random lending reserve.

    :param token:
        Underlying asset of the reserve
    
    :param chain_id:
        Chain ID to of the reserve
    
    :param internal_id:
        Internal ID of the reserve
    """

    atoken = AssetIdentifier(
        chain_id.value,
        generate_random_ethereum_address(),
        f"a{token.token_symbol}",
        token.decimals,
        random.randint(1, 1000),
    )
    vtoken = AssetIdentifier(
        chain_id.value,
        generate_random_ethereum_address(),
        f"v{token.token_symbol}",
        token.decimals,
        random.randint(1, 1000),
    )

    return LendingReserve(
        reserve_id=internal_id,
        reserve_slug=token.token_symbol.lower(),
        protocol_slug=LendingProtocolType.aave_v3,
        chain_id=chain_id,
        chain_slug=chain_id.get_slug(),

        asset_id=token.internal_id,
        asset_name=token.token_symbol,
        asset_symbol=token.token_symbol,
        asset_address=token.address,
        asset_decimals=token.decimals,

        atoken_id=atoken.internal_id,
        atoken_symbol=atoken.token_symbol,
        atoken_address=atoken.address,
        atoken_decimals=atoken.decimals,

        vtoken_id=vtoken.internal_id,
        vtoken_symbol=vtoken.token_symbol,
        vtoken_address=vtoken.address,
        vtoken_decimals=vtoken.decimals,

        additional_details=LendingReserveAdditionalDetails(
            ltv=80,
            liquidation_threshold=85,
        ),
    )


def generate_lending_universe(
    bucket: TimeBucket,
    start: datetime.datetime,
    end: datetime.datetime,
    reserves: list[LendingReserve],
    aprs: dict[str, float],
) -> tuple[LendingReserveUniverse, LendingCandleUniverse]:
    """Generate sample lending time series data.

    The output candles are deterministic: the same input parameters result to the same output parameters.

    :param bucket: 
        Time bucket to use for the candles

    :param start: 
        Start time for the candles

    :param end: 
        End time for the candles

    :param reserves: 
        List of reserves to generate candles for

    :param aprs: 
        APRs to use for the candles
    """

    time_delta = bucket.to_timedelta()

    supply_df = None
    variable_borrow_df = None
    for type, apr in aprs.items():
        data = []
        now = start
        while now < end:
            for reserve in reserves:
                data.append({
                    "reserve_id": reserve.reserve_id,
                    "timestamp": now,
                    "open": apr,
                    "close": apr,
                    "high": apr,
                    "low": apr,
                })

            now += time_delta

        df = pd.DataFrame(data)
        df.set_index("timestamp", drop=False, inplace=True)

        if type == "supply":
            supply_df = df
        else:
            variable_borrow_df = df

    reserve_universe = LendingReserveUniverse(reserves={
        r.reserve_id: r
        for r in reserves
    })

    return reserve_universe, LendingCandleUniverse(
        candle_type_dfs={
            LendingCandleType.variable_borrow_apr: variable_borrow_df,
            LendingCandleType.supply_apr: supply_df,
        },
        lending_reserve_universe=reserve_universe,
    )
