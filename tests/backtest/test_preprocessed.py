import datetime

import pytest

from tradeexecutor.backtest.preprocessed_backtest import Dataset, prepare_dataset, SavedDataset
from tradeexecutor.utils.dedent import dedent_any
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.timebucket import TimeBucket


@pytest.fixture
def integration_test_dataset():
    """Sample 1 months worth of data."""
    return Dataset(
        chain=ChainId.avalanche,
        slug="integration_test_dataset",
        name="Binance Chain, Pancakeswap, 2021-2025, hourly",
        description=dedent_any("""
            PancakeSwap DEX hourly trades.

            - Contains bull and bear market data with mixed set of tokens
            - Binance smart chain is home of many fly-by-night tokens, 
              and very few of tokens on this chain have long term prospects 
            """),
        start=datetime.datetime(2023, 1, 1),
        end=datetime.datetime(2023, 2, 1),
        time_bucket=TimeBucket.d1,
        min_tvl=1_000_000,
        exchanges={"trader-joe"},
        always_included_pairs=[],
    )

def test_preprocessed_dataset(
    persistent_test_client: Client,
    tmp_path,
    integration_test_dataset,
):
    """Test backtest set generation with small Avalanche dataset."""

    client = persistent_test_client

    saved_dataset = prepare_dataset(
        client=client,
        dataset=integration_test_dataset,
        output_folder=tmp_path,
    )
    assert isinstance(saved_dataset, SavedDataset)
    assert saved_dataset.get_pair_count() == 8