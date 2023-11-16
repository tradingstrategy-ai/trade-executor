"""Prepare a filtered Parquet file containing candles for pairs on Polygon.

- Include pairs from Uniswap and Quickswap
"""
import os

from tradeexecutor.strategy.execution_context import python_script_execution_context
from tradeexecutor.strategy.trading_strategy_universe import load_all_data
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.utils.forward_fill import forward_fill
from tradingstrategy.utils.groupeduniverse import fix_bad_wicks

client = Client.create_jupyter_client()

chain_id = ChainId.polygon
time_bucket = TimeBucket.d1
exchange_slugs = ["uniswap-v3", "quickswap"]

exchanges = client.fetch_exchange_universe()
exchange_ids = [exchanges.get_by_chain_and_slug(ChainId.polygon, s).exchange_id for s in exchange_slugs]

dataset = load_all_data(
    client,
    time_frame=TimeBucket.d1,
    execution_context=python_script_execution_context,
    universe_options=UniverseOptions(),
    with_liquidity=False,
)

# Filter out pair ids that belong to our target dataset
pair_universe = dataset.pairs
pair_ids = pair_universe.loc[pair_universe["exchange_id"].isin(exchange_ids)]["pair_id"]
filtered_df = dataset.candles.loc[dataset.candles["pair_id"].isin(pair_ids)]

print(f"Total {len(pair_ids)} pairs")

# Sanitise price data
filtered_df = filtered_df.set_index("timestamp")
filtered_df = fix_bad_wicks(filtered_df)

# Forward fill data
# Make sure there are no gaps in the data
filtered_df = filtered_df.groupby("pair_id")
pairs_df = forward_fill(
    filtered_df,
    freq=time_bucket.to_frequency(),
    columns=("open", "high", "low", "close", "volume"),
)

slug_str = "-and-".join(exchange_slugs)

# Wrote Parquest file under /tmp
fpath = f"/tmp/{chain_id.get_slug()}-{slug_str}-candles-{time_bucket.value}.parquet"
flattened_df = pairs_df.obj
flattened_df = flattened_df.reset_index().set_index("timestamp")  # Get rid of grouping
flattened_df.to_parquet(fpath)
print(f"Wrote {fpath} {os.path.getsize(fpath):,} bytes")


