"""Hyper AI Trading Strategy website vault data integration tests."""

import os
from pathlib import Path
from types import ModuleType

import pytest

from tradeexecutor.strategy.execution_context import unit_test_execution_context
from tradeexecutor.strategy.pandas_trader.trading_universe_input import CreateTradingUniverseInput
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client


pytestmark = pytest.mark.skipif(os.environ.get("TRADING_STRATEGY_API_KEY") is None, reason="Set TRADING_STRATEGY_API_KEY environment variable to run this test")


FIXED_HYPERCORE_VAULTS = [
    (ChainId.hypercore, "0x010461c14e146ac35fe42271bdc1134ee31c703a"),
]


def test_hyper_ai_strategy_create_trading_universe_uses_remote_vault_data(
    persistent_test_client: Client,
    hyper_ai_strategy_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test Hyper AI Trading Strategy website vault universe construction.

    1. Replace the curated Hypercore builder and vault download helpers with tmp-path backed variants.
    2. Build the strategy trading universe through the real strategy module.
    3. Confirm the resulting universe contains remote vault metadata, candles, liquidity and downloaded files.
    """
    download_root = tmp_path / "vault-downloads"

    # 1. Replace the curated Hypercore builder and vault download helpers with tmp-path backed variants.
    hyper_ai_strategy_module.VAULTS = None
    monkeypatch.setattr(
        hyper_ai_strategy_module,
        "build_hyperliquid_vault_universe",
        lambda min_tvl, min_age: FIXED_HYPERCORE_VAULTS,
    )
    original_load_vault_universe_with_metadata = hyper_ai_strategy_module.load_vault_universe_with_metadata
    original_load_partial_data = hyper_ai_strategy_module.load_partial_data

    def _load_vault_universe_with_metadata(*args, **kwargs):
        kwargs.setdefault("download_root", download_root)
        return original_load_vault_universe_with_metadata(*args, **kwargs)

    def _load_partial_data(*args, **kwargs):
        kwargs.setdefault("vault_history_download_root", download_root)
        return original_load_partial_data(*args, **kwargs)

    monkeypatch.setattr(
        hyper_ai_strategy_module,
        "load_vault_universe_with_metadata",
        _load_vault_universe_with_metadata,
    )
    monkeypatch.setattr(
        hyper_ai_strategy_module,
        "load_partial_data",
        _load_partial_data,
    )

    # 2. Build the strategy trading universe through the real strategy module.
    universe_options = UniverseOptions.from_strategy_parameters_class(
        hyper_ai_strategy_module.Parameters,
        unit_test_execution_context,
    )
    input_data = CreateTradingUniverseInput(
        client=persistent_test_client,
        timestamp=hyper_ai_strategy_module.Parameters.backtest_end,
        parameters=hyper_ai_strategy_module.Parameters,
        execution_context=unit_test_execution_context,
        execution_model=None,
        universe_options=universe_options,
    )
    strategy_universe = hyper_ai_strategy_module.create_trading_universe(input_data)

    # 3. Confirm the resulting universe contains remote vault metadata, candles, liquidity and downloaded files.
    raw_pair = strategy_universe.data_universe.pairs.get_pair_by_smart_contract(FIXED_HYPERCORE_VAULTS[0][1])
    assert raw_pair is not None
    assert raw_pair.get_vault_metadata() is not None

    pair_candles = strategy_universe.data_universe.candles.get_candles_by_pair(raw_pair)
    assert len(pair_candles) > 0

    liquidity_df = strategy_universe.data_universe.liquidity.df
    assert len(liquidity_df.loc[liquidity_df["pair_id"] == raw_pair.pair_id]) > 0
    assert (download_root / "vault-universe.json").exists()
    assert (download_root / "vault-price-history.parquet").exists()
