"""Create features for FinRL data.

- Run prefilter-finrl.py first

"""
import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from IPython.display import HTML

from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet, IndicatorSource
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.strategy.pandas_trader.indicator import calculate_and_load_indicators_inline
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorDependencyResolver
from tradeexecutor.state.types import USDollarAmount
from tradeexecutor.strategy.pandas_trader.indicator_decorator import IndicatorRegistry
from tradeexecutor.analysis.indicator import display_indicators
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.timebucket import TimeBucket

chain_id = ChainId.ethereum
time_bucket = TimeBucket.h4
liquidity_time_bucket = TimeBucket.d1
exchange_slugs = {"uniswap-v3", "uniswap-v2"}
min_prefilter_liquidity = 1_000_000
client = Client.create_jupyter_client()
start = datetime.datetime(2023, 1, 1)
end = datetime.datetime(2025, 1, 1)
cache_path = client.transport.cache_path
compression_level = 15
combined_prefilter_fname = Path(f"{cache_path}/{chain_id.get_slug()}-price-tvl-prefiltered.parquet")
combined_feature_fname = Path(f"{cache_path}/{chain_id.get_slug()}-features.parquet")

df = pd.read_parquet(combined_output_fname)