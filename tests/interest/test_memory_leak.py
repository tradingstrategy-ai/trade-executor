"""Manual memory leak test."""

import datetime
import sys
import gc

import pandas as pd
import pytest

from tradeexecutor.analysis.universe import analyse_long_short_universe
from tradeexecutor.state.identifier import TradingPairKind
from tradeexecutor.strategy.execution_context import unit_test_execution_context
from tradeexecutor.strategy.trading_strategy_universe import load_partial_data, TradingStrategyUniverse, load_trading_and_lending_data, translate_trading_pair
from tradeexecutor.strategy.universe_model import default_universe_options, UniverseOptions
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.lending import LendingProtocolType, UnknownLendingReserve
from tradingstrategy.timebucket import TimeBucket


def test_memory_leak(persistent_test_client: Client):
    """Load trading pair and lending data for the same backtest"""
    import psutil
    client = persistent_test_client

    p = psutil.Process()

    for i in range(0, 10):
        rss = p.memory_info().rss
        data = client.fetch_pair_universe()
        print("RSS is ", rss)
        gc.collect()


