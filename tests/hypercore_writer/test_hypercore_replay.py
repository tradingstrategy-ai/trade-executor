"""Unit-style tests for Hypercore replay market data."""

from __future__ import annotations

import datetime
import importlib.util
import os
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest
from tradingstrategy.chain import ChainId
from web3 import Web3

from tradeexecutor.curator import hyperliquid_vault_universe as vault_universe_module
from tradeexecutor.ethereum.ethereum_protocol_adapters import EthereumPairConfigurator
from tradeexecutor.ethereum.vault.hypercore_valuation import HypercoreVaultPricing
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.testing.hypercore_replay import HypercoreDailyMetricsReplay


def test_hypercore_replay_snapshot_asof(
    hypercore_replay_source: HypercoreDailyMetricsReplay,
    hypercore_vault_pair: TradingPairIdentifier,
) -> None:
    """Verify the replay snapshot lookup rules.

    1. Ask the replay source for a snapshot after the middle data point.
    2. Confirm the replay picks the latest available historical row on or before that date.
    3. Confirm the hardcoded v1 defaults still fill the missing Hypercore fields.
    """
    # 1. Ask the replay source for a snapshot after the middle data point.
    # We use the replay source because Hypercore does not expose historical simulation for this test.
    snapshot = hypercore_replay_source.get_snapshot(
        datetime.datetime(2026, 1, 30),
        hypercore_vault_pair,
        net_deposited_usdc=Decimal("100"),
    )

    # 2. Confirm the replay picks the latest available historical row on or before that date.
    assert snapshot.data_date == datetime.date(2026, 1, 21)
    assert snapshot.tvl_usd == pytest.approx(Decimal("269004391.733933"))
    assert snapshot.account_pnl_usd == pytest.approx(Decimal("119191482.033933"))

    # 3. Confirm the hardcoded v1 defaults still fill the missing Hypercore fields.
    # These fields are mocked because historical daily metrics only give us TVL and account PnL.
    assert snapshot.leader_fraction == Decimal("0.10")
    assert snapshot.allow_deposits is True
    assert snapshot.is_closed is False
    assert snapshot.relationship_type == "normal"


def test_hypercore_replay_share_price_calculation(
    hypercore_replay_source: HypercoreDailyMetricsReplay,
    hypercore_vault_pair: TradingPairIdentifier,
) -> None:
    """Verify the mock share price reconstruction.

    1. Load one early replay snapshot and one later replay snapshot for the same net deposit.
    2. Check that the reconstructed share price grows across the replay window.
    3. Check that replay equity is derived from ``net_deposited_usdc * share_price``.
    """
    # 1. Load one early replay snapshot and one later replay snapshot for the same net deposit.
    # We use replay snapshots because share price is reconstructed by the mock from historical inputs.
    first = hypercore_replay_source.get_snapshot(
        datetime.datetime(2026, 1, 21),
        hypercore_vault_pair,
        net_deposited_usdc=Decimal("125"),
    )
    later = hypercore_replay_source.get_snapshot(
        datetime.datetime(2026, 2, 3),
        hypercore_vault_pair,
        net_deposited_usdc=Decimal("125"),
    )

    # 2. Check that the reconstructed share price grows across the replay window.
    assert first.share_price > Decimal("1")
    assert later.share_price > first.share_price

    # 3. Check that replay equity is derived from ``net_deposited_usdc * share_price``.
    assert first.equity_usd == pytest.approx(Decimal("125") * first.share_price)
    assert later.equity_usd == pytest.approx(Decimal("125") * later.share_price)


def test_hypercore_pricing_uses_replay_tvl_and_gating(
    hypercore_replay_source: HypercoreDailyMetricsReplay,
    hypercore_daily_metrics_frame: pd.DataFrame,
    hypercore_vault_pair: TradingPairIdentifier,
) -> None:
    """Verify replay-backed pricing and redemption gating.

    1. Build a Hypercore pricing model that reads TVL and gating data from the replay source.
    2. Check that TVL, deposit permission and redeem permission come from the replay snapshot.
    3. Rebuild the source with a future lock-up expiry and confirm redemption is blocked.
    """
    # 1. Build a Hypercore pricing model that reads TVL and gating data from the replay source.
    # We inject the replay source because the live Hypercore API cannot simulate this history for tests.
    # The other callables are minimal mocks because this test only cares about replay TVL and gating behaviour.
    pricing = HypercoreVaultPricing(
        value_func=lambda pair: Decimal("150"),
        safe_address_resolver=lambda pair: "0x0000000000000000000000000000000000000001",
        session_factory=lambda pair: None,
        simulate=False,
        market_data_source=hypercore_replay_source,
    )

    timestamp = datetime.datetime(2026, 2, 3)

    # 2. Check that TVL, deposit permission and redeem permission come from the replay snapshot.
    assert pricing.get_usd_tvl(timestamp, hypercore_vault_pair) == pytest.approx(308585706.340077)
    assert pricing.can_deposit(timestamp, hypercore_vault_pair) is True
    assert pricing.can_redeem(timestamp, hypercore_vault_pair) is True
    assert pricing.get_max_redemption(timestamp, hypercore_vault_pair) == pytest.approx(Decimal("150"))

    # 3. Rebuild the source with a future lock-up expiry and confirm redemption is blocked.
    # We mock the lock-up boundary here because that historical state is not present in daily metrics.
    locked_source = HypercoreDailyMetricsReplay.from_single_vault_dataframe(
        hypercore_vault_pair.other_data["hypercore_vault_address"],
        hypercore_daily_metrics_frame,
        lockup_expired_after=datetime.datetime(2026, 3, 1),
    )
    # The same minimal mocks are enough here because we only want to flip the replay lock-up rule.
    locked_pricing = HypercoreVaultPricing(
        value_func=lambda pair: Decimal("150"),
        safe_address_resolver=lambda pair: "0x0000000000000000000000000000000000000001",
        session_factory=lambda pair: None,
        simulate=False,
        market_data_source=locked_source,
    )

    assert locked_pricing.get_max_redemption(timestamp, hypercore_vault_pair) == pytest.approx(Decimal(0))
    assert locked_pricing.can_redeem(timestamp, hypercore_vault_pair) is False


def test_hyper_ai_strategy_import_is_lazy_for_vault_universe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the test-only strategy import does not fetch the vault universe.

    1. Replace the Hypercore vault builder with a sentinel failure before importing the strategy.
    2. Import the test-only Hyper-ai strategy module from disk.
    3. Confirm the import succeeds without building the universe and leaves the lazy cache empty.
    """
    # 1. Replace the Hypercore vault builder with a sentinel failure before importing the strategy.
    # We mock the builder here because this regression test is specifically checking that import
    # time does not depend on live curator network access or a warm cache.
    def _unexpected_builder(*args, **kwargs):
        raise AssertionError("Vault universe builder must not run during strategy import")

    monkeypatch.setattr(vault_universe_module, "build_hyperliquid_vault_universe", _unexpected_builder)

    # 2. Import the test-only Hyper-ai strategy module from disk.
    strategy_path = Path(__file__).resolve().parents[2] / "strategies" / "test_only" / "hyper-ai-test.py"
    spec = importlib.util.spec_from_file_location("hyper_ai_strategy_lazy_import", strategy_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)

    # 3. Confirm the import succeeds without building the universe and leaves the lazy cache empty.
    assert module.VAULTS is None


def test_hypercore_cache_key_includes_selection_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the Hypercore universe cache key covers the selection policy constants.

    1. Build one baseline cache key for the current Hypercore selection inputs.
    2. Change each selection-policy constant that previously went missing from the cache fingerprint.
    3. Confirm every policy change produces a different cache key.
    """
    # 1. Build one baseline cache key for the current Hypercore selection inputs.
    baseline_key = vault_universe_module._make_cache_key(
        min_tvl=10_000,
        top_n=None,
        min_age=0.15,
        sort_period="1Y",
        include_closed_vaults=False,
    )

    # 2. Change each selection-policy constant that previously went missing from the cache fingerprint.
    # We mock these module constants because the regression we want to catch is stale cache reuse
    # after changing the vault-selection policy.
    with monkeypatch.context() as policy_patch:
        policy_patch.setattr(
            vault_universe_module,
            "ALLOWED_DENOMINATIONS",
            set(vault_universe_module.ALLOWED_DENOMINATIONS) | {"DAI"},
        )
        allowed_key = vault_universe_module._make_cache_key(10_000, None, 0.15, "1Y", False)

    with monkeypatch.context() as policy_patch:
        policy_patch.setattr(
            vault_universe_module,
            "EXCLUDED_RISKS",
            set(vault_universe_module.EXCLUDED_RISKS) | {"Experimental"},
        )
        risk_key = vault_universe_module._make_cache_key(10_000, None, 0.15, "1Y", False)

    with monkeypatch.context() as policy_patch:
        policy_patch.setattr(
            vault_universe_module,
            "EXCLUDED_FLAGS",
            set(vault_universe_module.EXCLUDED_FLAGS) | {"needs-review"},
        )
        flag_key = vault_universe_module._make_cache_key(10_000, None, 0.15, "1Y", False)

    with monkeypatch.context() as policy_patch:
        policy_patch.setattr(vault_universe_module, "REQUIRE_KNOWN_PROTOCOL", False)
        protocol_key = vault_universe_module._make_cache_key(10_000, None, 0.15, "1Y", False)

    with monkeypatch.context() as policy_patch:
        policy_patch.setattr(vault_universe_module, "SKIP_CAGR_FILTER", False)
        cagr_key = vault_universe_module._make_cache_key(10_000, None, 0.15, "1Y", False)

    with monkeypatch.context() as policy_patch:
        policy_patch.setattr(vault_universe_module, "USE_PEAK_TVL", False)
        peak_tvl_key = vault_universe_module._make_cache_key(10_000, None, 0.15, "1Y", False)

    # 3. Confirm every policy change produces a different cache key.
    assert allowed_key != baseline_key
    assert risk_key != baseline_key
    assert flag_key != baseline_key
    assert protocol_key != baseline_key
    assert cagr_key != baseline_key
    assert peak_tvl_key != baseline_key


def test_hypercore_cache_expires_after_ttl(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify the Hypercore universe cache expires after the configured TTL.

    1. Write one cached Hypercore universe file into a temporary cache directory.
    2. Age the cache file past the TTL and confirm the loader ignores it as stale.
    3. Refresh the file mtime to a recent value and confirm the loader accepts it again.
    """
    # 1. Write one cached Hypercore universe file into a temporary cache directory.
    # We mock the cache directory and current time here because this regression test is only
    # about cache freshness rules, not live API fetching.
    monkeypatch.setattr(vault_universe_module, "CACHE_DIR", tmp_path)
    now = datetime.datetime(2026, 2, 4, 12, 0, 0)
    monkeypatch.setattr(vault_universe_module, "native_datetime_utc_now", lambda: now)

    cache_key = vault_universe_module._make_cache_key(10_000, None, 0.15, "1Y", False)
    expected_vaults = [(ChainId.hypercore, "0x1111111111111111111111111111111111111111")]
    vault_universe_module._save_cache(cache_key, expected_vaults)
    cache_path = vault_universe_module._cache_path(cache_key)

    # 2. Age the cache file past the TTL and confirm the loader ignores it as stale.
    stale_mtime = (now - vault_universe_module.CACHE_TTL - datetime.timedelta(seconds=1)).timestamp()
    os.utime(cache_path, (stale_mtime, stale_mtime))
    assert vault_universe_module._load_cache(cache_key) is None

    # 3. Refresh the file mtime to a recent value and confirm the loader accepts it again.
    fresh_mtime = (now - datetime.timedelta(hours=1)).timestamp()
    os.utime(cache_path, (fresh_mtime, fresh_mtime))
    assert vault_universe_module._load_cache(cache_key) == expected_vaults


def test_hypercore_pair_configurator_accepts_market_data_source_without_execution_model(
    hypercore_strategy_universe: TradingStrategyUniverse,
    hypercore_replay_source: HypercoreDailyMetricsReplay,
    hypercore_vault_pair: TradingPairIdentifier,
) -> None:
    """Verify Hypercore replay-only configuration works without a live execution model.

    1. Build an EthereumPairConfigurator with a Hypercore replay source but no execution model.
    2. Ask it to create the Hypercore routing configuration for the replay-backed vault pair.
    3. Confirm pricing and valuation use the replay source without requiring a live value function.
    """
    # 1. Build an EthereumPairConfigurator with a Hypercore replay source but no execution model.
    # We use a replay source here because this regression is specifically about the market-data-only path.
    configurator = EthereumPairConfigurator(
        web3=Web3(),
        strategy_universe=hypercore_strategy_universe,
        hypercore_market_data_source=hypercore_replay_source,
        execution_model=None,
    )
    pair = hypercore_strategy_universe.get_pair_by_id(hypercore_vault_pair.internal_id)

    # 2. Ask it to create the Hypercore routing configuration for the replay-backed vault pair.
    routing_id = configurator.match_router(pair)
    config = configurator.create_config(routing_id)

    # 3. Confirm pricing and valuation use the replay source without requiring a live value function.
    assert getattr(config.pricing_model, "market_data_source", None) is hypercore_replay_source
    assert getattr(config.valuation_model, "market_data_source", None) is hypercore_replay_source
    assert config.routing_model is None
