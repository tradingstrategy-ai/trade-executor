"""Download all Base daily OHLCV and liquidity data and store it in Parquet.

- FinRL preparetion

- As the amount of data is large, do this only once to speed up the backtesting in liquidity-risk-analysis

- Large amount of RAM needed
"""

import datetime
from pathlib import Path

import pandas as pd

from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.timebucket import TimeBucket


chain_id = ChainId.polygon
time_bucket = TimeBucket.h4
client = Client.create_jupyter_client()
start = datetime.datetime(2023, 1, 1)
end = datetime.datetime(2025, 1, 1)

cache_path = client.transport.cache_path

combined_output_fname = Path(f"{cache_path}/{chain_id.get_slug()}-price-tvl-prefiltered.parquet")

# If the pair does not have this liquidity, skip the pair.
# All price before the min liquidity is reached is set to zero.
high_liquidity_prefilter = 100_000

# We need pair metadata to know which pairs belong to Polygon
print("Downloading/opening pairs dataset")
pairs_df = client.fetch_pair_universe().to_pandas()
our_chain_pair_ids = pairs_df[pairs_df.chain_id == chain_id.value]["pair_id"].unique()

print(f"We have data for {len(our_chain_pair_ids)} trading pairs on {chain_id.name}")

# Download all liquidity data, extract
# trading pairs that exceed our prefiltering threshold
print("Downloading/opening liquidity dataset")
liquidity_df = client.fetch_all_liquidity_samples(time_bucket).to_pandas()
print(f"Filtering out liquidity for chain {chain_id.name}")
liquidity_df = liquidity_df.loc[liquidity_df.pair_id.isin(our_chain_pair_ids)]
liquidity_per_pair = liquidity_df.groupby(liquidity_df.pair_id)
print(f"Chain {chain_id.name} has liquidity data for {len(liquidity_per_pair.groups)}")

passed_pair_ids = set()
liquidity_output_chunks = []

for pair_id, pair_df in liquidity_per_pair:
    if pair_df["high"].max() > high_liquidity_prefilter:
        liquidity_output_chunks.append(pair_df)
        passed_pair_ids.add(pair_id)

print(f"After filtering for {high_liquidity_prefilter:,} USD min liquidity we have {len(passed_pair_ids)} pairs")

liquidity_out_df = pd.concat(liquidity_output_chunks)

print("Downloading/opening OHLCV dataset")
price_df = client.fetch_all_candles(time_bucket).to_pandas()
price_df = price_df.loc[price_df.pair_id.isin(passed_pair_ids)]

# Combine dataframes

import ipdb ; ipdb.set_trace()

# Zero out price

print(f"Wrote {price_output_fname}, {price_output_fname.stat().st_size:,} bytes")
