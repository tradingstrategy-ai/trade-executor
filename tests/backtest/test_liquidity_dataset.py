"""Load liquidity data"""
import pytest
import pandas as pd

from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.timebucket import TimeBucket

from tradeexecutor.strategy.execution_context import unit_test_execution_context
from tradeexecutor.strategy.trading_strategy_universe import load_partial_data
from tradeexecutor.strategy.universe_model import default_universe_options


def test_load_liquidity_dataset(persistent_test_client: Client):
    """Load partial price + liquidity data."""
    client = persistent_test_client

    pairs = [
        (ChainId.polygon, "uniswap-v3", "WETH", "USDC", 0.0005),
        (ChainId.polygon, "quickswap", "WETH", "USDC")
    ]

    dataset = load_partial_data(
        client,
        execution_context=unit_test_execution_context,
        time_bucket=TimeBucket.d1,
        pairs=pairs,
        universe_options=default_universe_options,
        start_at=pd.Timestamp("2024-01-01"),
        end_at=pd.Timestamp("2024-02-01"),
        liquidity=True,
    )

    # Liquidity data loaded
    assert len(dataset.pairs) == 2
    assert dataset.liquidity_time_bucket == TimeBucket.d1
    assert len(dataset.liquidity) == 64  # (31 + 1) * 2
