"""Load OHLCV data directly for a single pair Parquet file."""
import os
import random
from pathlib import Path

import pytest

import pandas as pd

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.strategy.trading_strategy_universe import create_pair_universe_from_code
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange
from tradeexecutor.testing.synthetic_price_data import load_ohlcv_parquet_file
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe


@pytest.fixture
def ohlcv_sample_path() -> Path:
    p = os.path.join(os.path.dirname(__file__), "..", "notebooks", "pool_1_hourly_candles.parquet")
    return Path(p)


def test_load_parquet(ohlcv_sample_path):
    """Load candle data from external Parquet file."""

    # Set up fake assets
    mock_chain_id = ChainId.osmosis

    mock_exchange = generate_exchange(
        exchange_id=random.randint(1, 1000),
        chain_id=mock_chain_id,
        address=generate_random_ethereum_address())
    # Cosmos tokens use micro token (u-token) as the smallest unit
    osmo = AssetIdentifier(ChainId.osmosis.value, generate_random_ethereum_address(), "OSMO", 6, 1)
    atom = AssetIdentifier(ChainId.osmosis.value, generate_random_ethereum_address(), "ATOM", 6, 2)
    atom_osmo = TradingPairIdentifier(
        atom,
        osmo,
        generate_random_ethereum_address(),
        mock_exchange.address,
        internal_id=random.randint(1, 1000),
        internal_exchange_id=mock_exchange.exchange_id)

    time_bucket = TimeBucket.h1

    pair_universe = create_pair_universe_from_code(mock_chain_id, [atom_osmo])

    # Load candles for backtesting
    candles = load_ohlcv_parquet_file(
        ohlcv_sample_path,
        mock_chain_id,
        mock_exchange.exchange_id,
        atom_osmo.internal_id)
    candle_universe = GroupedCandleUniverse.create_from_single_pair_dataframe(candles)

    universe = Universe(
        time_bucket=time_bucket,
        chains={mock_chain_id},
        exchanges={mock_exchange},
        pairs=pair_universe,
        candles=candle_universe,
        liquidity=None
    )

    candle_range = universe.candles.get_timestamp_range()

    assert candle_range[0] == pd.Timestamp('2021-12-25 00:00:00')
    assert candle_range[1] == pd.Timestamp('2022-04-28 12:00:00')
