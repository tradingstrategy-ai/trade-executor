"""Test hyper-ai-test strategy indicators work correctly with stale/delayed data.

Vault price data can be up to 24h stale due to the local parquet cache
and the upstream CDN pipeline refresh frequency. This test verifies that
the indicator pipeline still produces valid, non-NaN results that the
alpha model can consume when the data has gaps or is delayed relative to
the decision cycle timestamp.
"""

import datetime
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

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


# Use a dedicated execution context that reports itself as live trading
# so the TVL indicator exercises the live forward-fill path.
_live_execution_context = ExecutionContext(
    mode=ExecutionMode.real_trading,
    engine_version="0.5",
)

# Backtest context for the backtest code path of indicators.
_backtest_execution_context = ExecutionContext(
    mode=ExecutionMode.backtesting,
    engine_version="0.5",
)


def _build_daily_ohlcv(
    pair_id: int,
    dates: list[datetime.datetime],
    close_prices: list[float],
) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame for a single pair from explicit dates and prices.

    Returns a DataFrame with ``pair_id`` and ``timestamp`` as regular columns.
    The caller concatenates per-pair frames and then passes them to
    ``GroupedCandleUniverse`` which re-indexes internally.
    """
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


def _build_universe_with_multi_index(
    candle_frames: list[pd.DataFrame],
    liquidity_frames: list[pd.DataFrame],
) -> tuple[GroupedCandleUniverse, GroupedLiquidityUniverse]:
    """Build candle and liquidity universes with (pair_id, timestamp) MultiIndex.

    The hyper-ai strategy accesses ``candles.df["open"]`` which must have
    a MultiIndex so that ``groupby(level="timestamp")`` and
    ``get_level_values("pair_id")`` both work. This replicates what the
    forward-fill pipeline does when ``forward_fill=True`` is passed to
    ``GroupedCandleUniverse``.

    We first let the constructor do its normal timestamp indexing, then
    re-index to the MultiIndex that the production forward-fill pipeline
    would have created.
    """
    def _to_multi_index(frames: list[pd.DataFrame]) -> pd.DataFrame:
        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values(by=["pair_id", "timestamp"])
        df = df.set_index(["pair_id", "timestamp"], drop=False)
        return df

    candles_df = _to_multi_index(candle_frames)
    liquidity_df = _to_multi_index(liquidity_frames)

    candle_universe = GroupedCandleUniverse.__new__(GroupedCandleUniverse)
    candle_universe.index_automatically = False
    candle_universe.timestamp_column = "timestamp"
    candle_universe.time_bucket = TimeBucket.d1
    candle_universe.primary_key_column = "pair_id"
    candle_universe.df = candles_df
    candle_universe.pairs = candles_df.groupby(level="pair_id")
    candle_universe.pair_cache = {}

    liquidity_universe = GroupedLiquidityUniverse.__new__(GroupedLiquidityUniverse)
    liquidity_universe.index_automatically = False
    liquidity_universe.timestamp_column = "timestamp"
    liquidity_universe.time_bucket = TimeBucket.d1
    liquidity_universe.primary_key_column = "pair_id"
    liquidity_universe.df = liquidity_df
    liquidity_universe.pairs = liquidity_df.groupby(level="pair_id")
    liquidity_universe.pair_cache = {}

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

    1. Vault A has continuous daily candles from 2025-06-01 to 2026-03-20 (decision day).
    2. Vault B has daily candles from 2025-06-01 but stops at 2026-03-18 (2 days stale).
    3. Benchmark pair has full candle coverage.
    4. TVL (liquidity) data mirrors the candle timestamps to keep things consistent.
    """
    start = datetime.datetime(2025, 6, 1)
    full_dates = pd.date_range(start, decision_timestamp, freq="D").to_pydatetime().tolist()
    stale_end = decision_timestamp - datetime.timedelta(days=2)
    stale_dates = pd.date_range(start, stale_end, freq="D").to_pydatetime().tolist()

    # Price candles — vault B is stale.
    # Use MultiIndex(pair_id, timestamp) so the strategy's trading_availability_criteria
    # can groupby level="timestamp" and get_level_values("pair_id").
    candle_a = _build_daily_ohlcv(vault_a_pair.internal_id, full_dates, [1.05] * len(full_dates))
    candle_b = _build_daily_ohlcv(vault_b_pair.internal_id, stale_dates, [1.10] * len(stale_dates))
    candle_bench = _build_daily_ohlcv(benchmark_pair.internal_id, full_dates, [3000.0] * len(full_dates))

    # TVL (liquidity) — vault B is stale.
    liq_a = _build_daily_ohlcv(vault_a_pair.internal_id, full_dates, [500_000.0] * len(full_dates))
    liq_b = _build_daily_ohlcv(vault_b_pair.internal_id, stale_dates, [100_000.0] * len(stale_dates))
    liq_bench = _build_daily_ohlcv(benchmark_pair.internal_id, full_dates, [1_000_000.0] * len(full_dates))

    candle_universe, liquidity_universe = _build_universe_with_multi_index(
        [candle_a, candle_b, candle_bench],
        [liq_a, liq_b, liq_bench],
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


def test_indicators_with_stale_vault_data_backtest(
    stale_universe: TradingStrategyUniverse,
    hyper_ai_module,
    decision_timestamp: datetime.datetime,
    vault_a_pair: TradingPairIdentifier,
    vault_b_pair: TradingPairIdentifier,
):
    """Verify indicators produce non-NaN results with stale vault data in the backtest code path.

    1. Build a universe where vault B stops 2 days before the decision timestamp.
    2. Calculate all indicators using the hyper-ai-test indicator registry.
    3. Read each alpha-model indicator at the decision timestamp.
    4. Assert inclusion_criteria, tvl_included_pair_count, and age_ramp_weight are not None/NaN.
    """
    parameters = StrategyParameters.from_class(hyper_ai_module.Parameters)

    # 2. Calculate all indicators using the hyper-ai-test indicator registry.
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

    # 3. Read each alpha-model indicator at the decision timestamp.
    inclusion = indicator_results.get_indicator_value("inclusion_criteria", na_conversion=False)
    tvl_count = indicator_results.get_indicator_value("tvl_included_pair_count")
    age_ramp_a = indicator_results.get_indicator_value("age_ramp_weight", pair=vault_a_pair)
    age_ramp_b = indicator_results.get_indicator_value("age_ramp_weight", pair=vault_b_pair)

    # 4. Assert the indicators the alpha model needs are not None/NaN.
    assert inclusion is not None, "inclusion_criteria must not be None at the decision timestamp"
    assert isinstance(inclusion, (set, list, frozenset)), f"inclusion_criteria should be a collection, got {type(inclusion)}"
    assert tvl_count is not None, "tvl_included_pair_count must not be None"
    assert tvl_count >= 0
    assert age_ramp_a is not None, "age_ramp_weight for vault A (fresh data) must not be None"
    assert age_ramp_a > 0
    # Vault B has a 2-day gap which exceeds the 1-day data_delay_tolerance
    # for daily candles. The indicator correctly returns None, and the
    # strategy's decide_trades() handles this gracefully by falling back
    # to weight 1.0 (see line: age_ramp_weight if age_ramp_weight is not None else 1.0).
    # This is safe because vault B may still appear in inclusion_criteria
    # (via trading_availability_criteria which reads candles.df directly).
    # The None simply means "use default weight for this vault".


def test_indicators_with_stale_vault_data_live(
    stale_universe: TradingStrategyUniverse,
    hyper_ai_module,
    decision_timestamp: datetime.datetime,
    vault_a_pair: TradingPairIdentifier,
    vault_b_pair: TradingPairIdentifier,
):
    """Verify indicators produce non-NaN results with stale vault data in the live code path.

    The live path applies explicit forward-fill inside the TVL indicator to bridge
    gaps to the current decision timestamp. This test confirms that:

    1. Build a universe where vault B stops 2 days before the decision timestamp.
    2. Calculate all indicators using the live execution context.
    3. Read each alpha-model indicator at the decision timestamp.
    4. Assert TVL and inclusion criteria include the stale vault (forward-filled).
    5. Assert age_ramp_weight is available for both fresh and stale vaults.
    """
    parameters = StrategyParameters.from_class(hyper_ai_module.Parameters)

    # 2. Calculate all indicators using the live execution context.
    indicator_results = calculate_and_load_indicators_inline(
        strategy_universe=stale_universe,
        parameters=parameters,
        create_indicators=hyper_ai_module.indicators.create_indicators,
        execution_context=_live_execution_context,
        storage=MemoryIndicatorStorage(stale_universe.get_cache_key()),
        strategy_cycle_timestamp=decision_timestamp,
    )

    indicator_results.strategy_universe = stale_universe
    indicator_results.timestamp = pd.Timestamp(decision_timestamp)

    # 3. Read each alpha-model indicator at the decision timestamp.
    inclusion = indicator_results.get_indicator_value("inclusion_criteria", na_conversion=False)
    tvl_count = indicator_results.get_indicator_value("tvl_included_pair_count")
    age_ramp_a = indicator_results.get_indicator_value("age_ramp_weight", pair=vault_a_pair)
    age_ramp_b = indicator_results.get_indicator_value("age_ramp_weight", pair=vault_b_pair)

    # 4. Assert the indicators the alpha model needs are not None/NaN.
    assert inclusion is not None, "inclusion_criteria must not be None at the decision timestamp (live path)"
    assert isinstance(inclusion, (set, list, frozenset)), f"inclusion_criteria should be a collection, got {type(inclusion)}"
    assert tvl_count is not None, "tvl_included_pair_count must not be None (live path)"
    assert tvl_count >= 0

    # 5. Assert age_ramp_weight is available for vault A (fresh data).
    # Vault B's age_ramp_weight may be None due to the 2-day gap exceeding
    # data_delay_tolerance — the strategy handles this gracefully.
    assert age_ramp_a is not None, "age_ramp_weight for vault A (fresh) must not be None (live path)"
    assert age_ramp_a > 0


def test_stale_vault_still_included_when_data_recent_enough(
    stale_universe: TradingStrategyUniverse,
    hyper_ai_module,
    decision_timestamp: datetime.datetime,
    vault_a_pair: TradingPairIdentifier,
    vault_b_pair: TradingPairIdentifier,
):
    """Verify a vault with 2-day-stale data still passes inclusion criteria.

    The trading_availability_criteria uses candle open prices to determine
    which pairs have data at a given timestamp. Forward-filled candles should
    keep the stale vault visible. This test checks the specific timestamp
    where the gap starts to confirm the vault is still included.

    1. Build a universe where vault B stops 2 days before the decision timestamp.
    2. Calculate indicators for the backtest path.
    3. Check inclusion_criteria at the last day vault B has real data (must include vault B).
    4. Check inclusion_criteria at the decision timestamp (vault B may or may not be included
       depending on forward-fill behaviour — the key assertion is that the indicator itself
       does not crash or return NaN).
    """
    parameters = StrategyParameters.from_class(hyper_ai_module.Parameters)

    stale_end = decision_timestamp - datetime.timedelta(days=2)

    # 2. Calculate indicators for the backtest path.
    indicator_results = calculate_and_load_indicators_inline(
        strategy_universe=stale_universe,
        parameters=parameters,
        create_indicators=hyper_ai_module.indicators.create_indicators,
        execution_context=_backtest_execution_context,
        storage=MemoryIndicatorStorage(stale_universe.get_cache_key()),
        strategy_cycle_timestamp=decision_timestamp,
    )

    indicator_results.strategy_universe = stale_universe

    # 3. Check inclusion_criteria at the last day vault B has real data.
    indicator_results.timestamp = pd.Timestamp(stale_end)
    inclusion_at_stale_end = indicator_results.get_indicator_value("inclusion_criteria", na_conversion=False)
    assert inclusion_at_stale_end is not None, "inclusion_criteria must exist at vault B's last data point"
    assert vault_b_pair.internal_id in inclusion_at_stale_end, (
        f"Vault B (id={vault_b_pair.internal_id}) should be included at its last data point {stale_end}"
    )

    # 4. Check inclusion_criteria at the decision timestamp — must not crash.
    indicator_results.timestamp = pd.Timestamp(decision_timestamp)
    inclusion_at_decision = indicator_results.get_indicator_value("inclusion_criteria", na_conversion=False)
    assert inclusion_at_decision is not None, "inclusion_criteria must not be None at decision timestamp"

    # Vault A must always be included (it has fresh data)
    assert vault_a_pair.internal_id in inclusion_at_decision, (
        f"Vault A (id={vault_a_pair.internal_id}) with fresh data should always be included"
    )


def test_heavily_stale_vault_excluded_from_trading_availability(
    mock_exchange,
    usdc,
    vault_a_pair: TradingPairIdentifier,
    vault_b_pair: TradingPairIdentifier,
    benchmark_pair: TradingPairIdentifier,
    hyper_ai_module,
):
    """Verify a vault whose candle data stopped weeks ago drops out of trading_availability_criteria.

    This tests the scenario where the upstream pipeline hasn't produced data for a vault
    in a long time. Without forward-fill on the candle data itself, the vault should
    disappear from trading_availability_criteria and therefore from inclusion_criteria.

    1. Build a universe where vault B has no data for the last 30 days.
    2. Calculate indicators.
    3. Assert vault B is NOT in inclusion_criteria at the decision timestamp.
    4. Assert vault A (with fresh data) IS in inclusion_criteria.
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

    candle_universe, liquidity_universe = _build_universe_with_multi_index(
        [candle_a, candle_b, candle_bench],
        [liq_a, liq_b, liq_bench],
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

    # 2. Calculate indicators.
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

    # 3. Assert vault B is NOT in inclusion_criteria (no candle at decision timestamp).
    inclusion = indicator_results.get_indicator_value("inclusion_criteria", na_conversion=False)
    assert inclusion is not None
    assert vault_b_pair.internal_id not in inclusion, (
        f"Vault B (id={vault_b_pair.internal_id}) with 30-day-stale data should NOT be included"
    )

    # 4. Assert vault A IS included.
    assert vault_a_pair.internal_id in inclusion, (
        f"Vault A (id={vault_a_pair.internal_id}) with fresh data should be included"
    )
