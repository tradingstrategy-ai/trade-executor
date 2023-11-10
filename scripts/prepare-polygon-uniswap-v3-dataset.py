"""Prepare a filtered Parquet file containing candles for all pairs on Uniswap v3 on Polygon."""
import os

from tradeexecutor.strategy.execution_context import ExecutionContext, python_script_execution_context
from tradeexecutor.strategy.trading_strategy_universe import load_all_data
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.timebucket import TimeBucket

client = Client.create_jupyter_client()

exchanges = client.fetch_exchange_universe()
uni = exchanges.get_by_chain_and_slug(ChainId.polygon, "uniswap-v3")

dataset = load_all_data(
    client,
    time_frame=TimeBucket.d1,
    execution_context=python_script_execution_context,
    universe_options=UniverseOptions(),
    with_liquidity=False,
)


pair_df = dataset.pairs
pair_ids = pair_df.loc[pair_df["exchange_id"] == uni.exchange_id]["pair_id"]

df = dataset.candles

filtered_df = df.loc[df["pair_id"].isin(pair_ids)]

fpath = "/tmp/polygon-uni-v3-candles.parquet"

filtered_df.to_parquet(fpath)

print(f"Wrote {fpath} {os.path.getsize(fpath):,} bytes")

