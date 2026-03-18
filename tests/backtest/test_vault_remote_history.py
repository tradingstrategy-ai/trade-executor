"""Trading Strategy website vault history integration tests."""

import datetime
import os
from pathlib import Path

import pytest

from tradeexecutor.strategy.execution_context import unit_test_execution_context
from tradeexecutor.strategy.trading_strategy_universe import (
    load_partial_data,
    load_vault_universe_with_metadata,
)
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.exchange import ExchangeType
from tradingstrategy.timebucket import TimeBucket


pytestmark = pytest.mark.skipif(os.environ.get("TRADING_STRATEGY_API_KEY") is None, reason="Set TRADING_STRATEGY_API_KEY environment variable to run this test")


REMOTE_VAULTS = [
    (ChainId.base, "0x45aa96f0b3188d47a1dafdbefce1db6b37f58216"),
    (ChainId.base, "0xad20523a7dc37babc1cc74897e4977232b3d02e5"),
]

SUPPORTING_PAIRS = [
    (ChainId.ethereum, "uniswap-v3", "WETH", "USDC", 0.0005),
]


def test_load_partial_data_with_remote_vault_history(
    persistent_test_client: Client,
    tmp_path: Path,
) -> None:
    """Test Trading Strategy website vault history loading in ``load_partial_data()``.

    1. Load remote vault metadata and history into a pytest-provided temporary location.
    2. Build a partial dataset with Trading Strategy website vault history enabled.
    3. Confirm vault pairs, candles, liquidity and downloaded files are present.
    """
    download_root = tmp_path / "vault-downloads"

    # 1. Load remote vault metadata and history into a pytest-provided temporary location.
    client = persistent_test_client
    vault_universe = load_vault_universe_with_metadata(
        client,
        vaults=REMOTE_VAULTS,
        download_root=download_root,
    )

    # 2. Build a partial dataset with Trading Strategy website vault history enabled.
    dataset = load_partial_data(
        client=client,
        execution_context=unit_test_execution_context,
        time_bucket=TimeBucket.d1,
        pairs=SUPPORTING_PAIRS,
        universe_options=UniverseOptions(
            start_at=datetime.datetime(2025, 3, 1),
            end_at=datetime.datetime(2025, 5, 1),
        ),
        liquidity=True,
        liquidity_time_bucket=TimeBucket.d1,
        vaults=vault_universe,
        vault_history_source="trading-strategy-website",
        vault_history_download_root=download_root,
    )

    # 3. Confirm vault pairs, candles, liquidity and downloaded files are present.
    vault_pairs_df = dataset.pairs.loc[dataset.pairs["dex_type"] == ExchangeType.erc_4626_vault]
    assert len(vault_pairs_df) == len(REMOTE_VAULTS)

    vault_pair_ids = set(vault_pairs_df["pair_id"].unique())
    assert len(dataset.candles.loc[dataset.candles["pair_id"].isin(vault_pair_ids)]) > 0
    assert dataset.liquidity is not None
    assert len(dataset.liquidity.loc[dataset.liquidity["pair_id"].isin(vault_pair_ids)]) > 0
    assert (download_root / "vault-universe.json").exists()
    assert (download_root / "vault-price-history.parquet").exists()
