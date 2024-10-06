"""Demeter integration functions.

- See https://github.com/zelos-alpha/demeter

- Do not import this module unless you have Demeter installed
"""
import logging
import datetime

try:
    import demeter
except ImportError as e:
    raise ImportError("Demeter framework not installed, see https://github.com/zelos-alpha/demeter")

import pandas as pd

from demeter import TokenInfo
from demeter.uniswap import UniV3Pool, UniLpMarket
from demeter.uniswap.data import fillna

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradingstrategy.pair import DEXPair
from tradingstrategy.token import Token


logger = logging.getLogger(__name__)


def to_demeter_token(asset: AssetIdentifier | Token) -> TokenInfo:
    """Adapt Trading Strategy asset types to Demeter TokeInfo type."""
    match asset:
        case AssetIdentifier():
            return TokenInfo(
                name=asset.token_symbol,
                decimal=asset.decimals,
                address=asset.address,
            )
        case Token():
            return TokenInfo(
                name=asset.symbol,
                decimal=asset.decimals,
                address=asset.address,
            )
        case _:
            raise NotImplementedError(f"Unsupported asset type: {asset.__class__}")


def to_demeter_uniswap_v3_pool(trading_pair: TradingPairIdentifier | DEXPair) -> UniV3Pool:
    """Adapt Trading Strategy trading pair types to Demeter pool type."""

    match trading_pair:
        case TradingPairIdentifier():
            base = to_demeter_token(trading_pair.base)
            quote = to_demeter_token(trading_pair.quote)
            pool = UniV3Pool(token0=base, token1=quote, fee=trading_pair.fee, quote_token=quote)
            return pool
        case DEXPair():
            base = to_demeter_token(trading_pair.get_base_token())
            quote = to_demeter_token(trading_pair.get_quote_token())
            fee = trading_pair.fee_tier * 100  # 0.05 = 5 bps in Demeter
            assert fee > 0
            pool = UniV3Pool(
                token0=base,
                token1=quote,
                fee=fee,
                quote_token=quote
            )
            assert pool.tick_spacing != 0
            return pool

        case _:
            raise NotImplementedError(f"Unsupported trading pair type: {trading_pair.__class__}")


def load_clmm_data_to_uni_lp_market(
    market: UniLpMarket,
    df: pd.DataFrame,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
):
    assert isinstance(market, UniLpMarket)
    assert isinstance(df, pd.DataFrame)

    logger.info(
        "Loading data to Demeter market %s, entries %d",
        market,
        len(df)
    )

    #
    # Remap columns
    #

    # columns = [
    #     "timestamp",
    #     "netAmount0",
    #     "netAmount1",
    #     "closeTick",
    #     "openTick",
    #     "lowestTick",
    #     "highestTick",
    #     "inAmount0",
    #     "inAmount1",
    #     "currentLiquidity",
    # ]

    # Index(['pair_id', 'bucket', 'open_tick', 'close_tick', 'high_tick', 'low_tick',
    #        'current_liquidity', 'net_amount0', 'net_amount1', 'in_amount0',
    #        'in_amount1'],

    df = df.rename(columns={
        "bucket": "timestamp",
        "close_tick": "closeTick",
        "open_tick": "openTick",
        "low_tick": "lowestTick",
        "high_tick": "highestTick",
        "in_amount0": "inAmount0",
        "in_amount1": "inAmount1",
        "current_liquidity": "currentLiquidity",
        "net_amount0": "netAmount0",
        "net_amount1": "netAmount1",
    })

    del df["pair_id"]  # Fails in fillna() below

    #
    # Following is copy paste from market.py from Demeter
    # I have no idea what it is supposed to do
    #

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    # fill empty row (first minutes in a day, might be blank)
    full_indexes = pd.date_range(
        start=start_date,
        end=datetime.datetime.combine(end_date, datetime.time(0, 0, 0)) + datetime.timedelta(days=1) - datetime.timedelta(minutes=1),
        freq="1min",
    )
    df = df.reindex(full_indexes)
    # df = Lines.from_dataframe(df)
    # df = df.fillna()
    df: pd.DataFrame = fillna(df)
    if pd.isna(df.iloc[0]["closeTick"]):
        df = df.bfill()

    market.add_statistic_column(df)
    market.data = df

