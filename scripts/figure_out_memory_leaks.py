import gc
import psutil

from tradeexecutor.strategy.execution_context import unit_test_execution_context
from tradeexecutor.strategy.trading_strategy_universe import load_all_data
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradingstrategy.client import Client
from tradingstrategy.timebucket import TimeBucket

client = Client.create_test_client("/tmp/trading-strategy-tests")

for i in range(0, 10):

    dataset = load_all_data(
        client,
        TimeBucket.d7,
        unit_test_execution_context,
        UniverseOptions(),
    )

    p = psutil.Process()
    rss = p.memory_info().rss
    print(f"RSS is {rss:,}")

    del dataset

    gc.collect()
    import pyarrow
    pool = pyarrow.default_memory_pool()
    pool.release_unused()

    p = psutil.Process()
    rss = p.memory_info().rss
    print(f"RSS after cleaning is {rss:,}")

