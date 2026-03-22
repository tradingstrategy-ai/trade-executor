"""Test hyper-ai-test strategy indicators work correctly with stale/delayed data.

Vault price data can be up to 24h stale due to the local parquet cache
and the upstream CDN pipeline refresh frequency. This test verifies that
the indicator pipeline still produces valid, non-NaN results that the
alpha model can consume when the data has gaps or is delayed relative to
the decision cycle timestamp.

The production code path uses ``create_from_dataset(forward_fill=True,
forward_fill_until=timestamp)`` which extends every pair's candle and
liquidity data to the decision timestamp. These tests replicate that
behaviour so they exercise the same forward-fill pipeline.
"""

import datetime
import importlib.util
import logging
from pathlib import Path

import pandas as pd
import pytest

from tradeexecutor.ethereum.vault.hypercore_valuation import HypercoreVaultPricing
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.strategy.execution_context import (
    ExecutionContext,
    ExecutionMode,
)
from tradeexecutor.strategy.pandas_trader.indicator import (
    MemoryIndicatorStorage,
    calculate_and_load_indicators_inline,
)
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.trading_strategy_universe import (
    TradingStrategyUniverse,
    create_pair_universe_from_code,
)
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.liquidity import GroupedLiquidityUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe


_backtest_execution_context = ExecutionContext(
    mode=ExecutionMode.backtesting,
    engine_version="0.5",
)


def _build_daily_ohlcv(
    pair_id: int,
    dates: list[datetime.datetime],
    close_prices: list[float],
) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame for a single pair from explicit dates and prices."""
    rows = []
    for dt, price in zip(dates, close_prices):
        rows.append({
            "open": price,
            "high": price * 1.01,
            "low": price * 0.99,
            "close": price,
            "volume": 1000.0,
            "pair_id": pair_id,
            "timestamp": pd.Timestamp(dt),
        })
    return pd.DataFrame(rows)


def _build_forward_filled_universe(
    candle_frames: list[pd.DataFrame],
    liquidity_frames: list[pd.DataFrame],
    forward_fill_until: datetime.datetime,
) -> tuple[GroupedCandleUniverse, GroupedLiquidityUniverse]:
    """Build candle and liquidity universes with production-style forward-fill.

    Replicates the production code path where ``create_from_dataset()``
    passes ``forward_fill=True`` and ``forward_fill_until=timestamp`` to
    ``GroupedCandleUniverse`` and ``GroupedLiquidityUniverse``. The
    constructors then call ``fix_dex_price_data()`` which extends each
    pair's data to ``forward_fill_until`` via ``pad_dataframe_to_frequency()``
    and forward-fills close values.
    """
    candles_df = pd.concat(candle_frames, ignore_index=True)
    liquidity_df = pd.concat(liquidity_frames, ignore_index=True)

    candle_universe = GroupedCandleUniverse(
        candles_df,
        time_bucket=TimeBucket.d1,
        forward_fill=True,
        forward_fill_until=forward_fill_until,
    )

    liquidity_universe = GroupedLiquidityUniverse(
        liquidity_df,
        time_bucket=TimeBucket.d1,
        forward_fill=True,
        forward_fill_until=forward_fill_until,
    )

    return candle_universe, liquidity_universe


@pytest.fixture()
def mock_exchange():
    return generate_exchange(
        exchange_id=1,
        chain_id=ChainId.ethereum,
        address=generate_random_ethereum_address(),
        exchange_slug="test-dex",
    )


@pytest.fixture()
def usdc():
    return AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "USDC", 6, 1)


@pytest.fixture()
def vault_a_pair(usdc, mock_exchange):
    """Vault A — has data up to the decision timestamp."""
    return TradingPairIdentifier(
        AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "VAULT_A", 18, 10),
        usdc,
        generate_random_ethereum_address(),
        mock_exchange.address,
        internal_id=10,
        internal_exchange_id=mock_exchange.exchange_id,
        fee=0.0,
    )


@pytest.fixture()
def vault_b_pair(usdc, mock_exchange):
    """Vault B — data stops 2 days before the decision timestamp (stale)."""
    return TradingPairIdentifier(
        AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "VAULT_B", 18, 20),
        usdc,
        generate_random_ethereum_address(),
        mock_exchange.address,
        internal_id=20,
        internal_exchange_id=mock_exchange.exchange_id,
        fee=0.0,
    )


@pytest.fixture()
def benchmark_pair(usdc, mock_exchange):
    """Supporting/benchmark pair (WETH-USDC)."""
    return TradingPairIdentifier(
        AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "WETH", 18, 30),
        usdc,
        generate_random_ethereum_address(),
        mock_exchange.address,
        internal_id=30,
        internal_exchange_id=mock_exchange.exchange_id,
        fee=0.003,
    )


@pytest.fixture()
def decision_timestamp() -> datetime.datetime:
    """The moment the strategy runs its decision cycle."""
    return datetime.datetime(2026, 3, 20)


@pytest.fixture()
def stale_universe(
    mock_exchange,
    usdc,
    vault_a_pair,
    vault_b_pair,
    benchmark_pair,
    decision_timestamp,
) -> TradingStrategyUniverse:
    """Build a universe where vault B has a 2-day data gap before the decision timestamp.

    Uses the production forward-fill path so stale pairs get extended to
    the decision timestamp, matching ``create_from_dataset(forward_fill=True)``.

    1. Vault A has continuous daily candles from 2025-06-01 to 2026-03-20 (decision day).
    2. Vault B has daily candles from 2025-06-01 but stops at 2026-03-18 (2 days stale).
    3. Benchmark pair has full candle coverage.
    4. TVL (liquidity) data mirrors the candle timestamps.
    5. Framework forward-fill extends vault B's data to the decision timestamp.
    """
    start = datetime.datetime(2025, 6, 1)
    full_dates = pd.date_range(start, decision_timestamp, freq="D").to_pydatetime().tolist()
    stale_end = decision_timestamp - datetime.timedelta(days=2)
    stale_dates = pd.date_range(start, stale_end, freq="D").to_pydatetime().tolist()

    candle_a = _build_daily_ohlcv(vault_a_pair.internal_id, full_dates, [1.05] * len(full_dates))
    candle_b = _build_daily_ohlcv(vault_b_pair.internal_id, stale_dates, [1.10] * len(stale_dates))
    candle_bench = _build_daily_ohlcv(benchmark_pair.internal_id, full_dates, [3000.0] * len(full_dates))

    liq_a = _build_daily_ohlcv(vault_a_pair.internal_id, full_dates, [500_000.0] * len(full_dates))
    liq_b = _build_daily_ohlcv(vault_b_pair.internal_id, stale_dates, [100_000.0] * len(stale_dates))
    liq_bench = _build_daily_ohlcv(benchmark_pair.internal_id, full_dates, [1_000_000.0] * len(full_dates))

    candle_universe, liquidity_universe = _build_forward_filled_universe(
        [candle_a, candle_b, candle_bench],
        [liq_a, liq_b, liq_bench],
        forward_fill_until=decision_timestamp,
    )

    pair_universe = create_pair_universe_from_code(
        ChainId.ethereum,
        [vault_a_pair, vault_b_pair, benchmark_pair],
    )

    universe = Universe(
        time_bucket=TimeBucket.d1,
        chains={ChainId.ethereum},
        exchanges={mock_exchange},
        pairs=pair_universe,
        candles=candle_universe,
        liquidity=liquidity_universe,
    )

    return TradingStrategyUniverse(data_universe=universe, reserve_assets=[usdc])


@pytest.fixture()
def hyper_ai_module():
    """Load the hyper-ai-test strategy module with SUPPORTING_PAIRS cleared.

    The strategy's ``inclusion_criteria`` and ``trading_pair_count`` indicators
    look up SUPPORTING_PAIRS (Arbitrum/Ethereum Uniswap V3 pairs) in the
    universe to exclude them from allocation decisions. Our synthetic test
    universe does not contain those pairs, so we patch them to an empty list
    to avoid lookup failures.
    """
    strategy_path = Path(__file__).resolve().parents[2] / "strategies" / "test_only" / "hyper-ai-test.py"
    spec = importlib.util.spec_from_file_location("hyper_ai_strategy", strategy_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SUPPORTING_PAIRS = []
    return module


def test_indicators_with_stale_vault_data(
    stale_universe: TradingStrategyUniverse,
    hyper_ai_module,
    decision_timestamp: datetime.datetime,
    vault_a_pair: TradingPairIdentifier,
    vault_b_pair: TradingPairIdentifier,
):
    """Verify all alpha-model indicators produce valid values when vault data is stale.

    Framework forward-fill extends vault B's data to the decision timestamp,
    so all indicators should return non-None values for both vaults.

    1. Build a universe where vault B stops 2 days before the decision timestamp.
    2. Calculate all indicators (framework forward-fill bridges the gap).
    3. Assert inclusion_criteria, tvl_included_pair_count, and age_ramp_weight
       are all valid for both fresh and stale vaults.
    """
    parameters = StrategyParameters.from_class(hyper_ai_module.Parameters)

    # 2. Calculate all indicators (framework forward-fill bridges the gap).
    indicator_results = calculate_and_load_indicators_inline(
        strategy_universe=stale_universe,
        parameters=parameters,
        create_indicators=hyper_ai_module.indicators.create_indicators,
        execution_context=_backtest_execution_context,
        storage=MemoryIndicatorStorage(stale_universe.get_cache_key()),
        strategy_cycle_timestamp=decision_timestamp,
    )

    indicator_results.strategy_universe = stale_universe
    indicator_results.timestamp = pd.Timestamp(decision_timestamp)

    # 3. Assert all alpha-model indicators are valid.
    inclusion = indicator_results.get_indicator_value("inclusion_criteria", na_conversion=False)
    tvl_count = indicator_results.get_indicator_value("tvl_included_pair_count")
    age_ramp_a = indicator_results.get_indicator_value("age_ramp_weight", pair=vault_a_pair)
    age_ramp_b = indicator_results.get_indicator_value("age_ramp_weight", pair=vault_b_pair)

    assert inclusion is not None, "inclusion_criteria must not be None at the decision timestamp"
    assert isinstance(inclusion, (set, list, frozenset)), f"inclusion_criteria should be a collection, got {type(inclusion)}"
    assert tvl_count is not None, "tvl_included_pair_count must not be None"
    assert tvl_count >= 0

    # Both vaults should have valid age_ramp_weight because framework
    # forward-fill extends vault B's candle data to the decision timestamp.
    assert age_ramp_a is not None, "age_ramp_weight for vault A (fresh data) must not be None"
    assert age_ramp_a > 0
    assert age_ramp_b is not None, "age_ramp_weight for vault B (stale but forward-filled) must not be None"
    assert age_ramp_b > 0


def test_stale_vault_included_via_forward_fill(
    stale_universe: TradingStrategyUniverse,
    hyper_ai_module,
    decision_timestamp: datetime.datetime,
    vault_a_pair: TradingPairIdentifier,
    vault_b_pair: TradingPairIdentifier,
):
    """Verify a vault with 2-day-stale data stays in inclusion criteria via forward-fill.

    With production-style forward-fill, vault B's candle and TVL data are
    extended to the decision timestamp. This means trading_availability_criteria
    sees vault B at every timestamp, keeping it in the inclusion set.

    1. Build a universe where vault B stops 2 days before the decision timestamp.
    2. Calculate indicators with framework forward-fill active.
    3. Check vault B is included at its last real data point.
    4. Check vault B is still included at the decision timestamp (forward-filled data).
    """
    parameters = StrategyParameters.from_class(hyper_ai_module.Parameters)
    stale_end = decision_timestamp - datetime.timedelta(days=2)

    # 2. Calculate indicators with framework forward-fill active.
    indicator_results = calculate_and_load_indicators_inline(
        strategy_universe=stale_universe,
        parameters=parameters,
        create_indicators=hyper_ai_module.indicators.create_indicators,
        execution_context=_backtest_execution_context,
        storage=MemoryIndicatorStorage(stale_universe.get_cache_key()),
        strategy_cycle_timestamp=decision_timestamp,
    )

    indicator_results.strategy_universe = stale_universe

    # 3. Check vault B is included at its last real data point.
    indicator_results.timestamp = pd.Timestamp(stale_end)
    inclusion_at_stale_end = indicator_results.get_indicator_value("inclusion_criteria", na_conversion=False)
    assert inclusion_at_stale_end is not None, "inclusion_criteria must exist at vault B's last data point"
    assert vault_b_pair.internal_id in inclusion_at_stale_end, (
        f"Vault B (id={vault_b_pair.internal_id}) should be included at its last real data point {stale_end}"
    )

    # 4. Check vault B is still included at the decision timestamp (forward-filled).
    indicator_results.timestamp = pd.Timestamp(decision_timestamp)
    inclusion_at_decision = indicator_results.get_indicator_value("inclusion_criteria", na_conversion=False)
    assert inclusion_at_decision is not None, "inclusion_criteria must not be None at decision timestamp"
    assert vault_a_pair.internal_id in inclusion_at_decision, (
        f"Vault A (id={vault_a_pair.internal_id}) with fresh data should always be included"
    )
    assert vault_b_pair.internal_id in inclusion_at_decision, (
        f"Vault B (id={vault_b_pair.internal_id}) should be included at decision timestamp via forward-fill"
    )


def test_heavily_stale_vault_still_included_with_forward_fill(
    mock_exchange,
    usdc,
    vault_a_pair: TradingPairIdentifier,
    vault_b_pair: TradingPairIdentifier,
    benchmark_pair: TradingPairIdentifier,
    hyper_ai_module,
):
    """Verify even a vault with a 30-day data gap remains included when forward-fill is active.

    With production-style forward-fill, even heavily stale vault data gets
    extended to the decision timestamp. The vault stays in
    trading_availability_criteria because forward-filled rows exist at every
    timestamp.

    1. Build a universe where vault B has no data for the last 30 days.
    2. Apply framework forward-fill to extend vault B's data.
    3. Calculate indicators.
    4. Assert vault B IS included (forward-fill bridges the 30-day gap).
    5. Assert vault A IS included.
    """
    decision_ts = datetime.datetime(2026, 3, 20)
    start = datetime.datetime(2025, 6, 1)
    full_dates = pd.date_range(start, decision_ts, freq="D").to_pydatetime().tolist()
    very_stale_end = decision_ts - datetime.timedelta(days=30)
    stale_dates = pd.date_range(start, very_stale_end, freq="D").to_pydatetime().tolist()

    candle_a = _build_daily_ohlcv(vault_a_pair.internal_id, full_dates, [1.05] * len(full_dates))
    candle_b = _build_daily_ohlcv(vault_b_pair.internal_id, stale_dates, [1.10] * len(stale_dates))
    candle_bench = _build_daily_ohlcv(benchmark_pair.internal_id, full_dates, [3000.0] * len(full_dates))

    liq_a = _build_daily_ohlcv(vault_a_pair.internal_id, full_dates, [500_000.0] * len(full_dates))
    liq_b = _build_daily_ohlcv(vault_b_pair.internal_id, stale_dates, [100_000.0] * len(stale_dates))
    liq_bench = _build_daily_ohlcv(benchmark_pair.internal_id, full_dates, [1_000_000.0] * len(full_dates))

    candle_universe, liquidity_universe = _build_forward_filled_universe(
        [candle_a, candle_b, candle_bench],
        [liq_a, liq_b, liq_bench],
        forward_fill_until=decision_ts,
    )

    pair_universe = create_pair_universe_from_code(
        ChainId.ethereum,
        [vault_a_pair, vault_b_pair, benchmark_pair],
    )

    universe = Universe(
        time_bucket=TimeBucket.d1,
        chains={ChainId.ethereum},
        exchanges={mock_exchange},
        pairs=pair_universe,
        candles=candle_universe,
        liquidity=liquidity_universe,
    )

    stale_universe = TradingStrategyUniverse(data_universe=universe, reserve_assets=[usdc])
    parameters = StrategyParameters.from_class(hyper_ai_module.Parameters)

    # 3. Calculate indicators.
    indicator_results = calculate_and_load_indicators_inline(
        strategy_universe=stale_universe,
        parameters=parameters,
        create_indicators=hyper_ai_module.indicators.create_indicators,
        execution_context=_backtest_execution_context,
        storage=MemoryIndicatorStorage(stale_universe.get_cache_key()),
        strategy_cycle_timestamp=decision_ts,
    )

    indicator_results.strategy_universe = stale_universe
    indicator_results.timestamp = pd.Timestamp(decision_ts)

    # 4. Assert vault B IS included (forward-fill bridges the 30-day gap).
    inclusion = indicator_results.get_indicator_value("inclusion_criteria", na_conversion=False)
    assert inclusion is not None

    assert vault_a_pair.internal_id in inclusion, (
        f"Vault A (id={vault_a_pair.internal_id}) with fresh data should be included"
    )
    assert vault_b_pair.internal_id in inclusion, (
        f"Vault B (id={vault_b_pair.internal_id}) should be included via forward-fill despite 30-day gap"
    )

    # 5. Assert age_ramp_weight works for both vaults.
    age_ramp_a = indicator_results.get_indicator_value("age_ramp_weight", pair=vault_a_pair)
    age_ramp_b = indicator_results.get_indicator_value("age_ramp_weight", pair=vault_b_pair)
    assert age_ramp_a is not None, "age_ramp_weight for vault A must not be None"
    assert age_ramp_b is not None, "age_ramp_weight for vault B must not be None (forward-filled)"


def test_stale_share_price_logs_warning(
    vault_a_pair: TradingPairIdentifier,
    vault_b_pair: TradingPairIdentifier,
    caplog,
):
    """Verify _get_share_price_from_candles logs a warning when vault data is over 24h stale.

    HypercoreVaultPricing.get_mid_price() looks up share prices from the
    data pipeline's candle universe. When the nearest candle is more than
    24 hours old, the method should emit a warning so operators can see
    the data is stale.

    1. Build a candle universe where vault B's last candle is 3 days before the query.
    2. Create a HypercoreVaultPricing with that candle universe.
    3. Call get_mid_price for vault B at the query timestamp.
    4. Assert a warning about stale data was logged.
    5. Assert the price is still returned (not silently dropped).
    """
    query_ts = datetime.datetime(2026, 3, 20)
    start = datetime.datetime(2025, 6, 1)

    # Vault A has fresh data, vault B stops 3 days before query
    full_dates = pd.date_range(start, query_ts, freq="D").to_pydatetime().tolist()
    stale_end = query_ts - datetime.timedelta(days=3)
    stale_dates = pd.date_range(start, stale_end, freq="D").to_pydatetime().tolist()

    candle_a = _build_daily_ohlcv(vault_a_pair.internal_id, full_dates, [1.05] * len(full_dates))
    candle_b = _build_daily_ohlcv(vault_b_pair.internal_id, stale_dates, [1.10] * len(stale_dates))
    candles_df = pd.concat([candle_a, candle_b], ignore_index=True)

    # 1. Build candle universe — no forward-fill so the gap is visible to the pricing model
    candle_universe = GroupedCandleUniverse(candles_df, time_bucket=TimeBucket.d1)

    # 2. Create HypercoreVaultPricing with candle universe for display prices
    pricing = HypercoreVaultPricing(
        value_func=None,
        candle_universe=candle_universe,
    )

    # 3. Call get_mid_price for vault B — should find a candle 3 days old
    with caplog.at_level(logging.WARNING, logger="tradeexecutor.ethereum.vault.hypercore_valuation"):
        price_b = pricing.get_mid_price(query_ts, vault_b_pair)

    # 4. Assert a warning about stale data was logged
    stale_warnings = [r for r in caplog.records if "stale" in r.message.lower() and "VAULT_B" in r.message]
    assert len(stale_warnings) >= 1, (
        f"Expected a staleness warning for VAULT_B, got log records: {[r.message for r in caplog.records]}"
    )

    # 5. Assert the price is still returned (stale data is used, not dropped)
    assert price_b == pytest.approx(1.10, abs=0.02)

    # Vault A should NOT trigger a warning (fresh data)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="tradeexecutor.ethereum.vault.hypercore_valuation"):
        price_a = pricing.get_mid_price(query_ts, vault_a_pair)

    stale_warnings_a = [r for r in caplog.records if "stale" in r.message.lower() and "VAULT_A" in r.message]
    assert len(stale_warnings_a) == 0, "Vault A with fresh data should NOT trigger a staleness warning"
    assert price_a == pytest.approx(1.05, abs=0.02)
