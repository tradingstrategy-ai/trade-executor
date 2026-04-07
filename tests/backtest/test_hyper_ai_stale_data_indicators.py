"""Test hyper-ai-test strategy indicators work correctly with stale/delayed vault data.

Hypercore vault data pipeline and data healing
===============================================

Raw vault share prices arrive from the on-chain scanner via the
``eth_defi`` data pipeline as hourly snapshots with arbitrary block-level
timestamps. They are cleaned, resampled, and published to R2 CDN as a
parquet file (``cleaned-vault-prices-1h.parquet``).

When the trade executor loads this data, it can be stale for two reasons:

1. **CDN pipeline lag** — the batch pipeline (``scan-prices.py`` ->
   ``clean-prices.py`` -> ``export-data-files.py``) runs asynchronously.
2. **Local 24h cache** — ``fetch_vault_price_history()`` in
   ``tradingstrategy/transport/cache.py:625`` caches the parquet for 24h
   in ``~/.tradingstrategy/vaults/downloads/``.  ``client.clear_caches()``
   only purges ``~/.cache/tradingstrategy/``, so vault data is unaffected.

The data goes through several healing stages before indicators see it:

Stage 1 — ``load_partial_data()`` fetches and converts vault data
    ``client.fetch_vault_price_history()`` downloads the parquet from CDN
    (or serves cached). ``convert_vault_prices_to_candles()`` resamples
    raw share prices and TVL into OHLCV candle format (daily). Vault
    candles are concatenated with any DEX pair candles into a single
    ``candles_df`` (and ``liquidity_df`` for TVL).

    **File**: ``tradeexecutor/strategy/trading_strategy_universe.py:2817-2836``

Stage 2 — ``TradingStrategyUniverse.create_from_dataset(forward_fill=True)``
    Passes ``forward_fill=True`` and ``forward_fill_until=timestamp`` to
    both ``GroupedCandleUniverse`` and ``GroupedLiquidityUniverse``
    constructors.

    **File**: ``tradeexecutor/strategy/trading_strategy_universe.py:1376-1423``

Stage 3 — ``GroupedCandleUniverse.__init__()`` forward-fill pipeline
    The constructor calls ``fix_dex_price_data()`` which routes to
    ``forward_fill()`` -> ``resample_candles_multiple_pairs()`` ->
    ``forward_fill_ohlcv_single_pair()`` per pair. Each pair's data is:

    a) Resampled to the target frequency (daily).
    b) Extended to ``forward_fill_until`` via ``pad_dataframe_to_frequency()``
       which appends NaN rows from the pair's last real timestamp to the
       decision timestamp.
    c) Forward-filled: ``close`` via ``.ffill()``, ``open``/``high``/``low``
       filled from close, ``volume`` set to 0.
    d) A ``forward_filled=True`` boolean column marks synthetic rows.

    After this stage the DataFrame has a ``MultiIndex(pair_id, timestamp)``
    and every pair has rows at every timestamp up to ``forward_fill_until``.

    **Files**: ``tradingstrategy/utils/groupeduniverse.py:174-205``,
    ``tradingstrategy/utils/wrangle.py:258-457``,
    ``tradingstrategy/utils/forward_fill.py:752-894``

Stage 4 — ``GroupedLiquidityUniverse.__init__()`` (same pipeline for TVL)
    Same forward-fill chain as Stage 3, applied to the liquidity/TVL data.

Stage 5 — Indicator framework reads healed data
    ``_calculate_and_save_indicator_result()`` calls
    ``data_universe.candles.get_samples_by_pair(pair_id)`` (for candle
    indicators) or ``data_universe.liquidity.get_samples_by_pair(pair_id)``
    (for TVL indicators). Both return the fully forward-filled DataFrames
    from Stages 3-4, so indicators see complete time series with no gaps.

    **File**: ``tradeexecutor/strategy/pandas_trader/indicator.py:1396-1453``

Result: even if a vault's real data stopped days or weeks ago, the
indicator functions receive a complete daily time series from the vault's
inception to the decision timestamp. Stale rows are distinguishable via
the ``forward_filled`` column but are otherwise valid OHLCV rows.

Edge case: ``autoheal_pair_limit``
    ``GroupedCandleUniverse`` skips forward-fill entirely if the pair count
    exceeds ``autoheal_pair_limit`` (default 1500). Not a risk for hyper-ai
    (~120 vaults) but would silently break for larger universes.
    **File**: ``tradingstrategy/utils/groupeduniverse.py:174``

Limitations
-----------

These tests may not fully capture all indicator and data gap scenarios.
In particular:

- Only daily (``TimeBucket.d1``) candle frequency is tested. Strategies
  using hourly candles with daily TVL have a frequency mismatch that
  requires the indicator-level ``resample().ffill()`` pattern.
- The synthetic data uses uniform prices; real vault data has price
  movements and outlier spikes that interact with the cleaning pipeline.
- Only the backtest execution context is exercised. The live trading path
  shares the same forward-fill pipeline but has additional staleness
  concerns (24h parquet cache, CDN lag) that are not mocked here.
- The ``autoheal_pair_limit`` skip path is not tested.
- Vault data that arrives with non-midnight-aligned timestamps (common in
  production) is not covered; ``convert_vault_prices_to_candles()``
  resamples these but the interaction with forward-fill is untested.
"""

import datetime
import importlib.util
import logging
from pathlib import Path

import pandas as pd
import pytest

from tradeexecutor.ethereum.vault.checks import StaleVaultData, check_stale_vault_data, get_vault_data_freshness
from tradeexecutor.ethereum.vault.hypercore_valuation import HypercoreVaultPricing
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind
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
    constructors call ``fix_dex_price_data()`` -> ``forward_fill()`` ->
    ``forward_fill_ohlcv_single_pair()`` -> ``pad_dataframe_to_frequency()``
    which extends each pair's data to ``forward_fill_until`` and
    forward-fills close values (see module docstring Stages 3-4).
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
        kind=TradingPairKind.vault,
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
        kind=TradingPairKind.vault,
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

    Uses the production forward-fill path (Stages 2-4 in the module docstring)
    so stale pairs get extended to the decision timestamp, matching
    ``create_from_dataset(forward_fill=True)``.

    1. Vault A has continuous daily candles from 2025-06-01 to 2026-03-20 (decision day).
    2. Vault B has daily candles from 2025-06-01 but stops at 2026-03-18 (2 days stale).
    3. Benchmark pair has full candle coverage.
    4. TVL (liquidity) data mirrors the candle timestamps.
    5. Framework forward-fill (Stage 3) extends vault B's data to the decision timestamp.
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

    We mock SUPPORTING_PAIRS to an empty list because the strategy's
    ``inclusion_criteria`` and ``trading_pair_count`` indicators look up
    Arbitrum/Ethereum Uniswap V3 pairs in the universe to exclude them.
    Our synthetic test universe does not contain those pairs.
    """
    strategy_path = Path(__file__).resolve().parents[2] / "strategies" / "test_only" / "hyper-ai-test.py"
    spec = importlib.util.spec_from_file_location("hyper_ai_strategy", strategy_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SUPPORTING_PAIRS = []
    return module


def _calculate_indicators(
    hyper_ai_module,
    stale_universe: TradingStrategyUniverse,
    decision_timestamp: datetime.datetime,
):
    """Run create_indicators() and calculate_and_load_indicators_inline().

    Replicates the production flow: the strategy's ``create_indicators``
    callback builds the ``IndicatorSet`` from the decorator registry, then
    ``calculate_and_load_indicators_inline`` computes every indicator using
    the forward-filled universe data (Stage 5 in the module docstring).
    """
    parameters = StrategyParameters.from_class(hyper_ai_module.Parameters)
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
    return indicator_results


def test_all_alpha_model_indicators_valid_with_stale_data(
    stale_universe: TradingStrategyUniverse,
    hyper_ai_module,
    decision_timestamp: datetime.datetime,
    vault_a_pair: TradingPairIdentifier,
    vault_b_pair: TradingPairIdentifier,
):
    """Verify every indicator consumed by decide_trades() resolves to a usable value.

    The alpha model in decide_trades() reads these indicators:
    - ``inclusion_criteria`` (list of pair ids passing all filters)
    - ``tvl_included_pair_count`` (int count for diagnostics)
    - ``age_ramp_weight`` per pair (float signal weight, falls back to 1.0 if None)

    With framework forward-fill active (Stages 2-4), all indicators must
    produce non-None values for both fresh and stale vaults.

    1. Build a forward-filled universe where vault B is 2 days stale.
    2. Calculate all indicators via the strategy's create_indicators callback.
    3. Assert inclusion_criteria is a non-empty collection containing both vaults.
    4. Assert tvl_included_pair_count is a non-negative integer.
    5. Assert age_ramp_weight is a positive float for both vaults.
    """
    # 2. Calculate all indicators via the strategy's create_indicators callback.
    indicator_results = _calculate_indicators(hyper_ai_module, stale_universe, decision_timestamp)

    # 3. Assert inclusion_criteria is a non-empty collection containing both vaults.
    inclusion = indicator_results.get_indicator_value("inclusion_criteria", na_conversion=False)
    assert inclusion is not None, "inclusion_criteria must not be None at the decision timestamp"
    assert isinstance(inclusion, (set, list, frozenset)), f"Expected collection, got {type(inclusion)}"
    assert vault_a_pair.internal_id in inclusion, "Fresh vault A must be in inclusion_criteria"
    assert vault_b_pair.internal_id in inclusion, "Stale vault B must be in inclusion_criteria (forward-filled)"

    # 4. Assert tvl_included_pair_count is a non-negative integer.
    tvl_count = indicator_results.get_indicator_value("tvl_included_pair_count")
    assert tvl_count is not None, "tvl_included_pair_count must not be None"
    assert tvl_count >= 2, f"At least both vaults should pass TVL filter, got {tvl_count}"

    # 5. Assert age_ramp_weight is a positive float for both vaults.
    age_ramp_a = indicator_results.get_indicator_value("age_ramp_weight", pair=vault_a_pair)
    age_ramp_b = indicator_results.get_indicator_value("age_ramp_weight", pair=vault_b_pair)
    assert age_ramp_a is not None, "age_ramp_weight for vault A (fresh) must not be None"
    assert isinstance(age_ramp_a, float), f"Expected float, got {type(age_ramp_a)}"
    assert age_ramp_a > 0
    assert age_ramp_b is not None, "age_ramp_weight for vault B (stale, forward-filled) must not be None"
    assert isinstance(age_ramp_b, float), f"Expected float, got {type(age_ramp_b)}"
    assert age_ramp_b > 0


def test_stale_vault_stays_in_inclusion_criteria(
    stale_universe: TradingStrategyUniverse,
    hyper_ai_module,
    decision_timestamp: datetime.datetime,
    vault_a_pair: TradingPairIdentifier,
    vault_b_pair: TradingPairIdentifier,
):
    """Verify a 2-day-stale vault remains in inclusion criteria at every relevant timestamp.

    The ``inclusion_criteria`` indicator is the intersection of three sub-criteria:
    - ``tvl_inclusion_criteria``: TVL >= min_tvl (forward-filled TVL keeps vault visible)
    - ``trading_availability_criteria``: candle data exists (forward-filled candles keep vault visible)
    - ``age_inclusion_criteria``: age >= min_age (forward-filled close prices keep age growing)

    With framework forward-fill, vault B stays in all three sub-criteria
    even after its real data stops.

    1. Build a forward-filled universe where vault B is 2 days stale.
    2. Calculate indicators.
    3. Check vault B is in inclusion_criteria at its last real data point.
    4. Check vault B is still in inclusion_criteria at the decision timestamp.
    """
    # 2. Calculate indicators.
    indicator_results = _calculate_indicators(hyper_ai_module, stale_universe, decision_timestamp)
    stale_end = decision_timestamp - datetime.timedelta(days=2)

    # 3. Check vault B is in inclusion_criteria at its last real data point.
    indicator_results.timestamp = pd.Timestamp(stale_end)
    inclusion_at_stale_end = indicator_results.get_indicator_value("inclusion_criteria", na_conversion=False)
    assert inclusion_at_stale_end is not None
    assert vault_b_pair.internal_id in inclusion_at_stale_end, (
        f"Vault B must be included at its last real data point {stale_end}"
    )

    # 4. Check vault B is still in inclusion_criteria at the decision timestamp.
    indicator_results.timestamp = pd.Timestamp(decision_timestamp)
    inclusion_at_decision = indicator_results.get_indicator_value("inclusion_criteria", na_conversion=False)
    assert inclusion_at_decision is not None
    assert vault_a_pair.internal_id in inclusion_at_decision, "Fresh vault A must always be included"
    assert vault_b_pair.internal_id in inclusion_at_decision, (
        "Stale vault B must be included at decision timestamp (forward-filled data)"
    )


def test_30_day_stale_vault_still_works(
    mock_exchange,
    usdc,
    vault_a_pair: TradingPairIdentifier,
    vault_b_pair: TradingPairIdentifier,
    benchmark_pair: TradingPairIdentifier,
    hyper_ai_module,
):
    """Verify a vault with a 30-day data gap still produces valid indicators.

    A vault might stop reporting data because the upstream scanner
    crashes, the CDN export stalls, or the vault becomes inactive. The
    framework forward-fill must bridge even large gaps so the indicator
    pipeline does not crash or return NaN.

    1. Build a universe where vault B has no data for the last 30 days.
    2. Apply framework forward-fill to extend vault B's data to the decision timestamp.
    3. Calculate indicators and assert all alpha-model indicators are valid.
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

    # 3. Calculate indicators and assert all alpha-model indicators are valid.
    indicator_results = _calculate_indicators(hyper_ai_module, stale_universe, decision_ts)

    inclusion = indicator_results.get_indicator_value("inclusion_criteria", na_conversion=False)
    assert inclusion is not None
    assert vault_a_pair.internal_id in inclusion, "Fresh vault A must be included"
    assert vault_b_pair.internal_id in inclusion, "30-day-stale vault B must be included (forward-filled)"

    age_ramp_a = indicator_results.get_indicator_value("age_ramp_weight", pair=vault_a_pair)
    age_ramp_b = indicator_results.get_indicator_value("age_ramp_weight", pair=vault_b_pair)
    assert age_ramp_a is not None, "age_ramp_weight for vault A must not be None"
    assert age_ramp_b is not None, "age_ramp_weight for vault B must not be None (30-day gap, forward-filled)"
    assert age_ramp_b > 0


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
    6. Assert vault A (fresh data) does NOT trigger a warning.
    """
    query_ts = datetime.datetime(2026, 3, 20)
    start = datetime.datetime(2025, 6, 1)

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

    # 6. Assert vault A (fresh data) does NOT trigger a warning
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="tradeexecutor.ethereum.vault.hypercore_valuation"):
        price_a = pricing.get_mid_price(query_ts, vault_a_pair)

    stale_warnings_a = [r for r in caplog.records if "stale" in r.message.lower() and "VAULT_A" in r.message]
    assert len(stale_warnings_a) == 0, "Vault A with fresh data should NOT trigger a staleness warning"
    assert price_a == pytest.approx(1.05, abs=0.02)


def test_check_stale_vault_data_raises_on_stale(
    stale_universe: TradingStrategyUniverse,
    decision_timestamp: datetime.datetime,
):
    """Verify check_stale_vault_data raises when vault data exceeds the tolerance.

    With a tight tolerance (e.g. 1 hour), vault B's real data (which
    stopped 2 days ago) should trigger StaleVaultData. The exception
    must list the stale vault's name and last real timestamp.

    1. Build a forward-filled universe where vault B is 2 days stale.
    2. Call check_stale_vault_data with a 1-hour tolerance.
    3. Assert StaleVaultData is raised and mentions VAULT_B.
    """
    # 2. Use a tight tolerance so the 2-day gap triggers the check.
    with pytest.raises(StaleVaultData, match="VAULT_B"):
        check_stale_vault_data(
            stale_universe,
            decision_timestamp,
            execution_mode=ExecutionMode.real_trading,
            tolerance=datetime.timedelta(hours=1),
        )


def test_check_stale_vault_data_passes_when_fresh(
    stale_universe: TradingStrategyUniverse,
    decision_timestamp: datetime.datetime,
):
    """Verify check_stale_vault_data passes when all vaults are within tolerance.

    With the default 36-hour tolerance, vault B's 2-day-old data should
    trigger the check. But with a generous 30-day tolerance, even the
    stale vault is accepted.

    1. Build a forward-filled universe where vault B is 2 days stale.
    2. Call check_stale_vault_data with a 30-day tolerance.
    3. Assert no exception is raised.
    """
    # 2. Generous tolerance — 2-day gap is acceptable.
    check_stale_vault_data(
        stale_universe,
        decision_timestamp,
        execution_mode=ExecutionMode.real_trading,
        tolerance=datetime.timedelta(days=30),
    )


def test_check_stale_vault_data_skipped_in_backtest(
    stale_universe: TradingStrategyUniverse,
    decision_timestamp: datetime.datetime,
):
    """Verify check_stale_vault_data silently returns for non-live execution modes.

    Backtest data is complete by construction, so the staleness check
    should not run. Even with a very tight tolerance, no exception is raised.

    1. Build a forward-filled universe where vault B is 2 days stale.
    2. Call check_stale_vault_data with backtesting mode and a 1-hour tolerance.
    3. Assert no exception is raised.
    """
    # Should not raise despite the tight tolerance — backtest mode is skipped.
    check_stale_vault_data(
        stale_universe,
        decision_timestamp,
        execution_mode=ExecutionMode.backtesting,
        tolerance=datetime.timedelta(hours=1),
    )


def test_get_vault_data_freshness(
    stale_universe: TradingStrategyUniverse,
    vault_a_pair: TradingPairIdentifier,
    vault_b_pair: TradingPairIdentifier,
):
    """Verify get_vault_data_freshness returns correct freshness info for all vault pairs.

    The stale_universe fixture has vault A (fresh, data up to decision timestamp)
    and vault B (2 days stale). The benchmark pair should be excluded.

    1. Call get_vault_data_freshness on the stale universe.
    2. Assert the DataFrame has exactly 2 rows (vault pairs only, no benchmark).
    3. Assert vault B (stalest) appears first due to descending sort.
    4. Assert vault B has ~2 day candle and TVL data age with 2 trailing stale candles.
    5. Assert vault A has ~0 data age and 0 trailing stale candles.
    6. Assert Address column is present and populated.
    7. Assert Latest TVL (USD) is populated for both vaults.
    """
    # 1. Call get_vault_data_freshness on the stale universe.
    df = get_vault_data_freshness(stale_universe)

    # 2. Assert the DataFrame has exactly 2 rows (vault pairs only, no benchmark).
    assert len(df) == 2, f"Expected 2 vault rows, got {len(df)}. Columns: {df.columns.tolist()}"

    # 3. Assert vault B (stalest) appears first due to descending sort.
    row_b = df.iloc[0]
    row_a = df.iloc[1]
    assert "VAULT_B" in row_b["Vault"], f"Expected VAULT_B first (stalest), got {row_b['Vault']}"
    assert "VAULT_A" in row_a["Vault"], f"Expected VAULT_A second (fresh), got {row_a['Vault']}"

    # 4. Assert vault B has ~2 day candle and TVL data age with 2 trailing stale candles.
    assert row_b["Candle data age"] == pd.Timedelta(days=2), f"Expected 2d candle age, got {row_b['Candle data age']}"
    assert row_b["TVL data age"] == pd.Timedelta(days=2), f"Expected 2d TVL age, got {row_b['TVL data age']}"
    assert row_b["Trailing stale candles"] == 2, f"Expected 2 trailing stale candles, got {row_b['Trailing stale candles']}"

    # 5. Assert vault A has ~0 data age and 0 trailing stale candles.
    assert row_a["Candle data age"] == pd.Timedelta(0), f"Expected 0 candle age for fresh vault, got {row_a['Candle data age']}"
    assert row_a["Trailing stale candles"] == 0, f"Expected 0 trailing stale candles, got {row_a['Trailing stale candles']}"

    # 6. Assert Address column is present and populated.
    assert "Address" in df.columns
    assert pd.notna(row_a["Address"])
    assert pd.notna(row_b["Address"])

    # 7. Assert Latest TVL (USD) is populated for both vaults.
    assert row_a["Latest TVL (USD)"] == pytest.approx(500_000.0, abs=1.0)
    assert row_b["Latest TVL (USD)"] == pytest.approx(100_000.0, abs=1.0)

    # Verify timestamp columns are present and valid
    assert pd.notna(row_a["Last real candle"])
    assert pd.notna(row_a["Latest candle"])
    assert pd.notna(row_b["Last real candle"])
    assert pd.notna(row_b["Latest candle"])


def test_vault_data_freshness_chart_renders(
    stale_universe: TradingStrategyUniverse,
    vault_a_pair: TradingPairIdentifier,
    vault_b_pair: TradingPairIdentifier,
):
    """Verify the vault_data_freshness chart wrapper renders through the chart pipeline.

    Tests the end-to-end path: chart registration, ChartInput construction,
    and render_chart() call producing HTML output. This catches broken imports,
    missing registration, or render-path issues.

    1. Create a ChartRegistry and register vault_data_freshness.
    2. Build a minimal StrategyInputIndicators wrapping the stale universe.
    3. Create a ChartInput with backtest execution context.
    4. Call render_chart and assert the result is HTML with vault data.
    """
    from tradeexecutor.strategy.chart.definition import ChartRegistry, ChartKind, ChartInput
    from tradeexecutor.strategy.chart.renderer import render_chart, ChartParameters
    from tradeexecutor.strategy.chart.standard.vault import vault_data_freshness
    from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet
    from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInputIndicators

    # 1. Create a ChartRegistry and register vault_data_freshness.
    registry = ChartRegistry()
    registry.register(vault_data_freshness, ChartKind.indicator_all_pairs)

    # 2. Build a minimal StrategyInputIndicators wrapping the stale universe.
    indicators = StrategyInputIndicators(
        strategy_universe=stale_universe,
        available_indicators=IndicatorSet(),
        indicator_results={},
    )

    # 3. Create a ChartInput with backtest execution context.
    chart_input = ChartInput(
        execution_context=_backtest_execution_context,
        strategy_input_indicators=indicators,
    )

    # 4. Call render_chart and assert the result is HTML with vault data.
    params = ChartParameters()
    result = render_chart(registry, "vault_data_freshness", params, chart_input)
    assert result.error is None, f"Chart render failed: {result.error}"
    assert result.data is not None, "Chart render returned no data"
    assert result.content_type == "text/html", f"Expected HTML output, got {result.content_type}"

    # Verify the HTML contains vault names from our test universe
    html = result.data.decode("utf-8")
    assert "VAULT_A" in html, "HTML output should contain VAULT_A"
    assert "VAULT_B" in html, "HTML output should contain VAULT_B"


def test_vault_data_freshness_without_forward_fill(
    mock_exchange,
    usdc,
    vault_a_pair: TradingPairIdentifier,
    vault_b_pair: TradingPairIdentifier,
    benchmark_pair: TradingPairIdentifier,
):
    """Verify freshness is correctly reported when forward-fill is absent.

    When the universe is built without ``forward_fill=True``, the
    ``forward_filled`` column does not exist. Data age must still be
    computed correctly relative to the reference timestamp (the global
    max candle timestamp), not as ``latest - last_real`` which would
    always be zero.

    1. Build a universe without forward-fill where vault B stops 2 days early.
    2. Call get_vault_data_freshness — reference timestamp is auto-derived.
    3. Assert vault B shows ~2 day candle and TVL data age.
    4. Assert vault A shows ~0 data age.
    """
    decision_ts = datetime.datetime(2026, 3, 20)
    start = datetime.datetime(2025, 6, 1)
    full_dates = pd.date_range(start, decision_ts, freq="D").to_pydatetime().tolist()
    stale_end = decision_ts - datetime.timedelta(days=2)
    stale_dates = pd.date_range(start, stale_end, freq="D").to_pydatetime().tolist()

    candle_a = _build_daily_ohlcv(vault_a_pair.internal_id, full_dates, [1.05] * len(full_dates))
    candle_b = _build_daily_ohlcv(vault_b_pair.internal_id, stale_dates, [1.10] * len(stale_dates))
    candle_bench = _build_daily_ohlcv(benchmark_pair.internal_id, full_dates, [3000.0] * len(full_dates))

    liq_a = _build_daily_ohlcv(vault_a_pair.internal_id, full_dates, [500_000.0] * len(full_dates))
    liq_b = _build_daily_ohlcv(vault_b_pair.internal_id, stale_dates, [100_000.0] * len(stale_dates))
    liq_bench = _build_daily_ohlcv(benchmark_pair.internal_id, full_dates, [1_000_000.0] * len(full_dates))

    # Build WITHOUT forward-fill — no forward_filled column exists
    candles_df = pd.concat([candle_a, candle_b, candle_bench], ignore_index=True)
    liquidity_df = pd.concat([liq_a, liq_b, liq_bench], ignore_index=True)

    candle_universe = GroupedCandleUniverse(candles_df, time_bucket=TimeBucket.d1)
    liquidity_universe = GroupedLiquidityUniverse(liquidity_df, time_bucket=TimeBucket.d1)

    pair_universe = create_pair_universe_from_code(
        ChainId.ethereum, [vault_a_pair, vault_b_pair, benchmark_pair],
    )
    universe = Universe(
        time_bucket=TimeBucket.d1,
        chains={ChainId.ethereum},
        exchanges={mock_exchange},
        pairs=pair_universe,
        candles=candle_universe,
        liquidity=liquidity_universe,
    )
    no_ff_universe = TradingStrategyUniverse(data_universe=universe, reserve_assets=[usdc])

    # 2. Call get_vault_data_freshness — reference timestamp auto-derived from global max.
    df = get_vault_data_freshness(no_ff_universe)

    assert len(df) == 2

    # 3. Assert vault B shows ~2 day candle and TVL data age.
    row_b = df[df["Vault"].str.contains("VAULT_B")].iloc[0]
    assert row_b["Candle data age"] == pd.Timedelta(days=2), f"Expected 2d, got {row_b['Candle data age']}"
    assert row_b["TVL data age"] == pd.Timedelta(days=2), f"Expected 2d, got {row_b['TVL data age']}"
    assert row_b["Trailing stale candles"] == 0, "No forward_filled column, so trailing count should be 0"

    # 4. Assert vault A shows ~0 data age.
    row_a = df[df["Vault"].str.contains("VAULT_A")].iloc[0]
    assert row_a["Candle data age"] == pd.Timedelta(0), f"Expected 0, got {row_a['Candle data age']}"
    assert row_a["TVL data age"] == pd.Timedelta(0), f"Expected 0, got {row_a['TVL data age']}"


def test_vault_data_freshness_asymmetric_candle_tvl_staleness(
    mock_exchange,
    usdc,
    vault_a_pair: TradingPairIdentifier,
    vault_b_pair: TradingPairIdentifier,
    benchmark_pair: TradingPairIdentifier,
):
    """Verify sort considers TVL staleness when candles are fresh.

    A vault with fresh candles but stale TVL should sort above a fully
    fresh vault, because the composite sort uses the worst of candle
    and TVL data age.

    1. Build vault A with fresh candles and fresh TVL.
    2. Build vault B with fresh candles but TVL stopping 3 days early.
    3. Assert vault B sorts above vault A due to stale TVL.
    """
    decision_ts = datetime.datetime(2026, 3, 20)
    start = datetime.datetime(2025, 6, 1)
    full_dates = pd.date_range(start, decision_ts, freq="D").to_pydatetime().tolist()
    tvl_stale_end = decision_ts - datetime.timedelta(days=3)
    tvl_stale_dates = pd.date_range(start, tvl_stale_end, freq="D").to_pydatetime().tolist()

    # Both vaults have full candle coverage
    candle_a = _build_daily_ohlcv(vault_a_pair.internal_id, full_dates, [1.05] * len(full_dates))
    candle_b = _build_daily_ohlcv(vault_b_pair.internal_id, full_dates, [1.10] * len(full_dates))
    candle_bench = _build_daily_ohlcv(benchmark_pair.internal_id, full_dates, [3000.0] * len(full_dates))

    # Vault A has full TVL, vault B has TVL stopping 3 days early
    liq_a = _build_daily_ohlcv(vault_a_pair.internal_id, full_dates, [500_000.0] * len(full_dates))
    liq_b = _build_daily_ohlcv(vault_b_pair.internal_id, tvl_stale_dates, [100_000.0] * len(tvl_stale_dates))
    liq_bench = _build_daily_ohlcv(benchmark_pair.internal_id, full_dates, [1_000_000.0] * len(full_dates))

    candle_universe, liquidity_universe = _build_forward_filled_universe(
        [candle_a, candle_b, candle_bench],
        [liq_a, liq_b, liq_bench],
        forward_fill_until=decision_ts,
    )

    pair_universe = create_pair_universe_from_code(
        ChainId.ethereum, [vault_a_pair, vault_b_pair, benchmark_pair],
    )
    universe = Universe(
        time_bucket=TimeBucket.d1,
        chains={ChainId.ethereum},
        exchanges={mock_exchange},
        pairs=pair_universe,
        candles=candle_universe,
        liquidity=liquidity_universe,
    )
    asym_universe = TradingStrategyUniverse(data_universe=universe, reserve_assets=[usdc])

    df = get_vault_data_freshness(asym_universe)
    assert len(df) == 2

    # 3. Assert vault B sorts above vault A due to stale TVL.
    assert "VAULT_B" in df.iloc[0]["Vault"], f"Expected VAULT_B first (stale TVL), got {df.iloc[0]['Vault']}"
    assert "VAULT_A" in df.iloc[1]["Vault"], f"Expected VAULT_A second (fresh), got {df.iloc[1]['Vault']}"

    row_b = df.iloc[0]
    # Candles are fresh (0 age), but TVL is 3 days stale
    assert row_b["Candle data age"] == pd.Timedelta(0), f"Expected 0 candle age, got {row_b['Candle data age']}"
    assert row_b["TVL data age"] == pd.Timedelta(days=3), f"Expected 3d TVL age, got {row_b['TVL data age']}"


def test_vault_data_freshness_chart_via_create_charts(
    stale_universe: TradingStrategyUniverse,
    hyper_ai_module,
    decision_timestamp: datetime.datetime,
):
    """Verify vault_data_freshness is registered and renderable via create_charts().

    Uses the strategy module's create_charts() to build the registry,
    then renders the chart through the full pipeline. This catches
    broken imports or missing registration in the actual strategy.

    1. Call create_charts() from the strategy module.
    2. Assert vault_data_freshness is in the registry.
    3. Render the chart and assert it produces HTML output.
    """
    from tradeexecutor.strategy.chart.definition import ChartInput
    from tradeexecutor.strategy.chart.renderer import render_chart, ChartParameters
    from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet
    from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInputIndicators

    # 1. Call create_charts() from the strategy module.
    parameters = StrategyParameters.from_class(hyper_ai_module.Parameters)
    registry = hyper_ai_module.create_charts(
        timestamp=decision_timestamp,
        parameters=parameters,
        strategy_universe=stale_universe,
        execution_context=_backtest_execution_context,
    )

    # 2. Assert vault_data_freshness is in the registry.
    chart_entry = registry.get_chart_function("vault_data_freshness")
    assert chart_entry is not None, (
        f"vault_data_freshness not found in registry. "
        f"Available: {list(registry.registry.keys())}"
    )

    # 3. Render the chart and assert it produces HTML output.
    indicators = StrategyInputIndicators(
        strategy_universe=stale_universe,
        available_indicators=IndicatorSet(),
        indicator_results={},
    )
    chart_input = ChartInput(
        execution_context=_backtest_execution_context,
        strategy_input_indicators=indicators,
    )
    result = render_chart(registry, "vault_data_freshness", ChartParameters(), chart_input)
    assert result.error is None, f"Chart render failed: {result.error}"
    assert result.content_type == "text/html"
    html = result.data.decode("utf-8")
    assert "VAULT_A" in html
    assert "VAULT_B" in html
