"""Trading Universe creation tests."""
import os

import pandas as pd
import pytest

from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.trading_strategy_universe import load_all_data, \
    TradingStrategyUniverse
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradeexecutor.utils.timer import timed_task
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket


# https://docs.pytest.org/en/latest/how-to/skipping.html#skip-all-test-functions-of-a-class-or-module
# pytestmark = pytest.mark.skipif(os.environ.get("TRADING_STRATEGY_API_KEY") is None, reason="Set TRADING_STRATEGY_API_KEY environment variable to run this test")

pytestmark = pytest.mark.skip("Disabled as too RAM hungry for parallel testing")


def test_create_multipair_universe(persistent_test_client):
    """Create a trading universe with multiple pairs."""

    client = persistent_test_client
    candle_time_bucket = TimeBucket.d7

    execution_context = ExecutionContext(
        ExecutionMode.unit_testing_trading,
        timed_task,
    )

    dataset = load_all_data(
        client,
        candle_time_bucket,
        execution_context,
        UniverseOptions(),
    )

    factory_router_map = {
        # PancakeSwap
        "0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73": ("0x10ED43C718714eb63d5aA57B78B54704E256024E",
                                                       "0x00fb7f630766e6a796048ea87d01acd3068e8ff67d078148a3fa3f4a84f69bd5")
    }

    universe = TradingStrategyUniverse.create_multipair_universe(
        dataset,
        [ChainId.bsc],
        ["pancakeswap-v2"],
        # WBNB, BUSD
        ["0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c", "0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56"],
        # BUSD
        "0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56",
        factory_router_map,
    )

    assert universe.data_universe.pairs.get_count() > 1000
    assert universe.reserve_assets[0].address.lower() == "0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56".lower()
    range = universe.data_universe.candles.get_timestamp_range()
    assert range[0] < pd.Timestamp('2022-01-01 00:00:00')
    assert range[1] > pd.Timestamp('2022-01-01 00:00:00')


def test_create_multipair_universe_by_pair_descriptions(persistent_test_client):
    """Create a trading universe with multiple pairs using human pair descriptions."""

    client = persistent_test_client
    candle_time_bucket = TimeBucket.d7

    execution_context = ExecutionContext(
        ExecutionMode.unit_testing_trading,
        timed_task,
    )

    dataset = load_all_data(
        client,
        candle_time_bucket,
        execution_context,
        UniverseOptions(),
    )

    pairs = (
        (ChainId.ethereum, "uniswap-v2", "WETH", "USDC"),  # ETH
        (ChainId.ethereum, "uniswap-v2", "AAVE", "WETH"),  # AAVE
        (ChainId.ethereum, "uniswap-v2", "UNI", "WETH"),  # UNI
        (ChainId.ethereum, "uniswap-v2", "CRV", "WETH"),  # Curve
        (ChainId.ethereum, "sushi", "SUSHI", "WETH"),  # Sushi
        (ChainId.bsc, "pancakeswap-v2", "WBNB", "BUSD"),  # BNB
        (ChainId.bsc, "pancakeswap-v2", "Cake", "BUSD"),  # Cake
        (ChainId.polygon, "quickswap", "WMATIC", "USDC"),  # Matic
        (ChainId.avalanche, "trader-joe", "WAVAX", "USDC"),  # Avax
        (ChainId.avalanche, "trader-joe", "JOE", "WAVAX"),  # TraderJoe
    )

    universe = TradingStrategyUniverse.create_multichain_universe_by_pair_descriptions(
        dataset,
        pairs,
        "USDC",
    )

    assert universe.data_universe.pairs.get_count() == 10
    assert universe.data_universe.candles.get_pair_count() == 10
    assert universe.reserve_assets[0].token_symbol == "USDC"


def test_reverse_token_order(persistent_test_client):
    """Check for a reverse token order."""

    client = persistent_test_client
    candle_time_bucket = TimeBucket.d7

    execution_context = ExecutionContext(
        ExecutionMode.unit_testing_trading,
        timed_task,
    )

    dataset = load_all_data(
        client,
        candle_time_bucket,
        execution_context,
        UniverseOptions(),
    )

    pairs = (
        (ChainId.polygon, "uniswap-v3", "WETH", "USDC", 0.0005),  # https://tradingstrategy.ai/trading-view/polygon/uniswap-v3/eth-usdc-fee-5
        (ChainId.polygon, "uniswap-v3", "WMATIC", "USDC", 0.0005),  # https://tradingstrategy.ai/trading-view/polygon/uniswap-v3/matic-usdc-fee-5
    )

    universe = TradingStrategyUniverse.create_multichain_universe_by_pair_descriptions(
        dataset,
        pairs,
        "USDC",
    )

    pair = universe.get_pair_by_address("0x45dda9cb7c25131df268515131f647d726f50608")
    assert pair.has_reverse_token_order()

    pair = universe.get_pair_by_address("0xa374094527e1673a86de625aa59517c5de346d32")
    assert not pair.has_reverse_token_order()
