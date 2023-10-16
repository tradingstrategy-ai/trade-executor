import gc
import sys
from pathlib import PosixPath

import psutil
import logging

from pyarrow import parquet as pq

from tradeexecutor.strategy.execution_context import unit_test_execution_context
from tradeexecutor.strategy.trading_strategy_universe import load_all_data
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradingstrategy.client import Client
from tradingstrategy.timebucket import TimeBucket

client = Client.create_test_client("/tmp/trading-strategy-tests")


logging.getLogger().setLevel(logging.WARNING)
logging.basicConfig(stream=sys.stdout)

TEST_PARQUET_FILE = PosixPath('/tmp/trading-strategy-tests/memory-leak.parquet')

for i in range(0, 40):

    #dataset = load_all_data(
    #    client,
    #   TimeBucket.d7,
    #   unit_test_execution_context,
    #    UniverseOptions(),
    #)

    candles = client.fetch_all_candles(TimeBucket.d7).to_pandas()

    # pairs = client.fetch_pair_universe().to_pandas()
    # data = pq.read_table(TEST_PARQUET_FILE, memory_map=True, use_threads=False, pre_buffer=False)

    # f = open(TEST_PARQUET_FILE, "rb")
    #data = pq.read_table(f, memory_map=True, use_threads=False, pre_buffer=False)
    # f.close()

    p = psutil.Process()
    rss = p.memory_info().rss
    # print(f"RSS is {rss:,}")

    # del data

    # del dataset

    gc.collect()
    import pyarrow
    pool = pyarrow.default_memory_pool()
    pool.release_unused()

    p = psutil.Process()
    rss = p.memory_info().rss

    print(f"Pool bytes {pool.bytes_allocated():,} max memory {pool.max_memory():,}, RSS after cleaning is {rss:,}")

    # del dataset

