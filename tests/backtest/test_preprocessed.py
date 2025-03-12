"""Preproccessed backtest dataset integration tests."""

import datetime

import pytest

from tradeexecutor.backtest.preprocessed_backtest import BacktestDatasetDefinion, prepare_dataset, SavedDataset, AVAX_QUOTE_TOKEN
from tradeexecutor.utils.dedent import dedent_any
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.timebucket import TimeBucket


@pytest.fixture
def integration_test_dataset_daily():
    """Sample 1 months worth of data.

    - Avalanche datasets are small, fastest to download
    """
    return BacktestDatasetDefinion(
        chain=ChainId.avalanche,
        slug="integration_test_dataset",
        name="Avalanche test set",
        description=dedent_any("""
            TODO 
            """),
        start=datetime.datetime(2023, 1, 1),
        end=datetime.datetime(2023, 2, 1),
        time_bucket=TimeBucket.d1,
        min_tvl=500_000,
        min_weekly_volume=10_000,
        exchanges={"trader-joe"},
        always_included_pairs=[],
        reserve_token_address=AVAX_QUOTE_TOKEN,
    )


@pytest.fixture
def integration_test_dataset_hourly():
    """Sample 1 months worth of data.

    - Avalanche datasets are small, fastest to download
    """
    return BacktestDatasetDefinion(
        chain=ChainId.avalanche,
        slug="integration_test_dataset",
        name="Avalanche test set",
        description=dedent_any("""
            TODO 
            """),
        start=datetime.datetime(2023, 3, 1),
        end=datetime.datetime(2023, 3, 10),
        time_bucket=TimeBucket.h1,
        min_tvl=500_000,
        min_weekly_volume=10_000,
        exchanges={"trader-joe"},
        always_included_pairs=[],
        reserve_token_address=AVAX_QUOTE_TOKEN,
    )

def test_preprocessed_dataset_daily(
    persistent_test_client: Client,
    tmp_path,
    integration_test_dataset_daily,
):
    """Test backtest set generation with small Avalanche dataset."""

    client = persistent_test_client

    saved_dataset = prepare_dataset(
        client=client,
        dataset=integration_test_dataset_daily,
        output_folder=tmp_path,
    )
    assert isinstance(saved_dataset, SavedDataset)
    assert saved_dataset.get_pair_count() == 14


def test_preprocessed_dataset_hourly(
    persistent_test_client: Client,
    tmp_path,
    integration_test_dataset_hourly,
):
    """Test backtest set generation with small Avalanche dataset."""

    client = persistent_test_client

    saved_dataset = prepare_dataset(
        client=client,
        dataset=integration_test_dataset_hourly,
        output_folder=tmp_path,
    )
    assert isinstance(saved_dataset, SavedDataset)
    assert saved_dataset.get_pair_count() == 13
