"""Prepare a filtered Parquet file containing candles for pairs on Polygon.

- Include pairs from Uniswap and Quickswap

- Create file for hourly and daily candles
"""
import os

from tradeexecutor.strategy.execution_context import python_script_execution_context
from tradeexecutor.strategy.trading_strategy_universe import load_all_data
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.pair import PandasPairUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.utils.forward_fill import forward_fill
from tradingstrategy.utils.groupeduniverse import fix_bad_wicks

client = Client.create_jupyter_client()

chain_id = ChainId.polygon
time_buckets = [TimeBucket.d1, TimeBucket.h1]
exchange_slugs = ["uniswap-v3", "quickswap"]

max_price_threshold = 100_000
min_price_threshold = 0.000001

exchanges = client.fetch_exchange_universe()
exchange_ids = [exchanges.get_by_chain_and_slug(ChainId.polygon, s).exchange_id for s in exchange_slugs]

for time_bucket in time_buckets:

    dataset = load_all_data(
        client,
        time_frame=time_bucket,
        execution_context=python_script_execution_context,
        universe_options=UniverseOptions(),
        with_liquidity=False,
    )

    # Filter out pair ids that belong to our target dataset
    pairs_df = dataset.pairs
    pair_ids = pairs_df.loc[pairs_df["exchange_id"].isin(exchange_ids)]["pair_id"]
    candles_df = dataset.candles.loc[dataset.candles["pair_id"].isin(pair_ids)]
    print(f"Total {len(pair_ids)} pairs")

    # Clean bad price data
    # When the initial liquidity is provided, the price might be funky=
    print(f"Unfilterd candles {time_bucket.value}: {len(candles_df):,}")
    candles_df = candles_df.loc[(candles_df["open"] < max_price_threshold) & (candles_df["close"] < max_price_threshold)]
    candles_df = candles_df.loc[(candles_df["open"] > min_price_threshold) & (candles_df["close"] > min_price_threshold)]
    print(f"Filtered candles {time_bucket.value}: {len(candles_df):,}")

    # Sanitise price data
    candles_df = candles_df.set_index("timestamp")
    candles_df = fix_bad_wicks(candles_df)

    # Forward fill data
    # Make sure there are no gaps in the data
    candles_df = candles_df.groupby("pair_id")
    candles_df = forward_fill(
        candles_df,
        freq=time_bucket.to_frequency(),
        columns=("open", "high", "low", "close", "volume"),
    )

    slug_str = "-and-".join(exchange_slugs)

    # Wrote Parquest file under /tmp
    fpath = f"/tmp/{chain_id.get_slug()}-{slug_str}-candles-{time_bucket.value}.parquet"
    flattened_df = candles_df.obj
    flattened_df = flattened_df.reset_index().set_index("timestamp")  # Get rid of grouping
    flattened_df.to_parquet(fpath)
    print(f"Wrote {fpath} {os.path.getsize(fpath):,} bytes")

    # Print sample price data
    pair_universe = PandasPairUniverse(pairs_df, exchange_universe=dataset.exchanges)
    examined_pair = pair_universe.get_pair_by_human_description((ChainId.polygon, "quickswap", "pSWAMP", "WMATIC"))
    single_df = flattened_df.loc[flattened_df["pair_id"] == examined_pair.pair_id]
    print(single_df[0:30])



