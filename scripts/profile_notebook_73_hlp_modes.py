"""Profile each cell of the 73-hyperliquid-hlp-cash-sweep-modes notebook.

Profiles every code cell to find bottlenecks in the notebook execution.
Saves per-cell cProfile results (binary .prof + human-readable .txt) and
a summary to logs/notebook-profile-run/.

Usage:
    # Clear caches first for cold-cache profiling
    rm -rf ~/.cache/indicators/

    poetry run python scripts/profile_notebook_73_hlp_modes.py

Requirements:
    - .env file in working directory or getting-started repo with API keys
    - Vault database at ~/.tradingstrategy/vaults/
"""

import builtins
import cProfile
import datetime
import io
import logging
import os
import pstats
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.WARNING)

# Add getting-started to path for helper imports
sys.path.insert(0, "/Users/moo/code/getting-started")

# Shim display() for non-notebook context
def _noop_display(*args, **kwargs):
    pass

builtins.display = _noop_display

OUTPUT_DIR = Path("logs/notebook-profile-run")


def profile_cell(name, func):
    """Profile a cell, save results to disk, print summary."""
    safe_name = name.replace(" ", "_").replace(":", "").replace("+", "and").replace("/", "_").lower()

    profiler = cProfile.Profile()
    wall_start = time.perf_counter()
    profiler.enable()
    try:
        result = func()
    except Exception as e:
        profiler.disable()
        wall_elapsed = time.perf_counter() - wall_start
        print(f"  {name}: ERROR after {wall_elapsed:.3f}s - {e}")
        # Save error info
        txt_path = OUTPUT_DIR / f"{safe_name}.txt"
        with open(txt_path, "w") as f:
            f.write(f"CELL: {name}\n")
            f.write(f"Wall time: {wall_elapsed:.3f}s\n")
            f.write(f"ERROR: {e}\n")
        raise
    profiler.disable()
    wall_elapsed = time.perf_counter() - wall_start

    # Save binary .prof file
    prof_path = OUTPUT_DIR / f"{safe_name}.prof"
    profiler.dump_stats(str(prof_path))

    # Save human-readable .txt file
    txt_path = OUTPUT_DIR / f"{safe_name}.txt"
    with open(txt_path, "w") as f:
        f.write(f"CELL: {name}\n")
        f.write(f"Wall time: {wall_elapsed:.3f}s\n\n")

        # Cumulative time (top 50)
        stream = io.StringIO()
        ps = pstats.Stats(profiler, stream=stream)
        ps.sort_stats("cumulative")
        ps.print_stats(50)
        f.write("=== TOP 50 BY CUMULATIVE TIME ===\n")
        f.write(stream.getvalue())

        # Self time (top 30)
        stream2 = io.StringIO()
        ps2 = pstats.Stats(profiler, stream=stream2)
        ps2.sort_stats("tottime")
        ps2.print_stats(30)
        f.write("\n=== TOP 30 BY SELF TIME ===\n")
        f.write(stream2.getvalue())

    print(f"  {name}: {wall_elapsed:.3f}s -> {prof_path}")
    return result, wall_elapsed


# =============================================================================
# Cell 3: Client + charting setup
# =============================================================================
def cell_03_client_setup(ctx):
    from tradingstrategy.client import Client
    from tradeexecutor.utils.notebook import setup_charting_and_output, OutputMode

    client = Client.create_jupyter_client()
    setup_charting_and_output(OutputMode.static, image_format="png", width=1500, height=1000)
    ctx["client"] = client


# =============================================================================
# Cell 5: Chain configuration + vault universe building
# =============================================================================
def cell_05_chain_config(ctx):
    from eth_defi.token import USDC_NATIVE_TOKEN
    from tradingstrategy.chain import ChainId

    CHAIN_ID = ChainId.cross_chain
    PRIMARY_CHAIN_ID = ChainId.ethereum

    EXCHANGES = ("uniswap-v2", "uniswap-v3")
    SUPPORTING_PAIRS = [
        (ChainId.arbitrum, "uniswap-v3", "WETH", "USDC", 0.0005),
        (ChainId.ethereum, "uniswap-v3", "WETH", "USDC", 0.0005),
        (ChainId.ethereum, "uniswap-v3", "WBTC", "USDC", 0.003),
    ]
    LENDING_RESERVES = None
    PREFERRED_STABLECOIN = USDC_NATIVE_TOKEN[PRIMARY_CHAIN_ID].lower()

    from getting_started.hyperliquid_vault_universe import build_hyperliquid_vault_universe
    VAULTS = build_hyperliquid_vault_universe(
        min_tvl=10_000,
        min_age=0.15,
    )

    ALLOWED_VAULT_DENOMINATION_TOKENS = {"USDC", "USDT", "USDC.e", "crvUSD", "USD₮0", "USDt", "USDT0", "USDS"}

    BENCHMARK_PAIRS = [
        (ChainId.ethereum, "uniswap-v3", "WBTC", "USDC", 0.003),
    ]

    print(f"Chain universe: {CHAIN_ID}")
    print(f"Vaults count: {len(VAULTS)}")

    ctx["CHAIN_ID"] = CHAIN_ID
    ctx["PRIMARY_CHAIN_ID"] = PRIMARY_CHAIN_ID
    ctx["EXCHANGES"] = EXCHANGES
    ctx["SUPPORTING_PAIRS"] = SUPPORTING_PAIRS
    ctx["LENDING_RESERVES"] = LENDING_RESERVES
    ctx["PREFERRED_STABLECOIN"] = PREFERRED_STABLECOIN
    ctx["VAULTS"] = VAULTS
    ctx["ALLOWED_VAULT_DENOMINATION_TOKENS"] = ALLOWED_VAULT_DENOMINATION_TOKENS
    ctx["BENCHMARK_PAIRS"] = BENCHMARK_PAIRS


# =============================================================================
# Cell 7: Strategy parameters
# =============================================================================
def cell_07_parameters(ctx):
    import datetime
    import pandas as pd
    from tradingstrategy.timebucket import TimeBucket
    from tradeexecutor.strategy.cycle import CycleDuration
    from tradeexecutor.strategy.parameters import StrategyParameters
    from skopt.space import Categorical
    from tradeexecutor.strategy.default_routing_options import TradeRouting

    CHAIN_ID = ctx["CHAIN_ID"]
    PRIMARY_CHAIN_ID = ctx["PRIMARY_CHAIN_ID"]
    EXCHANGES = ctx["EXCHANGES"]

    class Parameters:
        id = "73-hyperliquid-hlp-cash-sweep-modes"
        candle_time_bucket = TimeBucket.d1
        cycle_duration = CycleDuration.cycle_1d
        chain_id = CHAIN_ID
        primary_chain_id = PRIMARY_CHAIN_ID
        exchanges = EXCHANGES
        max_assets_in_portfolio = 30
        allocation = 0.98
        individual_rebalance_min_threshold_usd = 50.0
        sell_rebalance_min_threshold = 10.0
        per_position_cap_of_pool = 0.20
        max_concentration = 0.20
        min_portfolio_weight = 0.0050
        hlp_mode = Categorical(["none", "weight_boost", "cash_sweep", "two_tier"])
        hlp_target_pct = Categorical([0.15, 0.25, 0.35])
        sterling_constant = 0.10
        calmar_signal_transform = "bayesian_credibility"
        bayesian_halflife = 60
        weight_signal = "rolling_sharpe"
        blend_alpha = 0.6
        waterfall = False
        volatility_window = 90
        min_tvl = 25_000
        min_age = 0.3
        backtest_start = datetime.datetime(2025, 8, 1)
        backtest_end = datetime.datetime(2026, 3, 11)
        initial_cash = 100_000
        routing = TradeRouting.default
        required_history_period = datetime.timedelta(days=60 * 2)
        slippage_tolerance = 0.0060
        assummed_liquidity_when_data_missings = 10_000

    parameters = StrategyParameters.from_class(Parameters)
    ctx["Parameters"] = Parameters
    ctx["parameters"] = parameters


# =============================================================================
# Cell 9: Universe creation
# =============================================================================
def cell_09_universe(ctx):
    from pathlib import Path
    from tradingstrategy.pair import PandasPairUniverse
    from tradingstrategy.timebucket import TimeBucket
    from eth_defi.vault.vaultdb import DEFAULT_VAULT_DATABASE, DEFAULT_RAW_PRICE_DATABASE
    from tradingstrategy.utils.token_filter import filter_for_selected_pairs
    from tradingstrategy.alternative_data.vault import load_vault_database
    from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, load_partial_data
    from tradeexecutor.strategy.execution_context import notebook_execution_context
    from tradeexecutor.strategy.universe_model import UniverseOptions
    from tradeexecutor.strategy.pandas_trader.trading_universe_input import CreateTradingUniverseInput

    from dotenv import load_dotenv
    load_dotenv(override=True)

    client = ctx["client"]
    Parameters = ctx["Parameters"]
    parameters = ctx["parameters"]
    SUPPORTING_PAIRS = ctx["SUPPORTING_PAIRS"]
    VAULTS = ctx["VAULTS"]
    ALLOWED_VAULT_DENOMINATION_TOKENS = ctx["ALLOWED_VAULT_DENOMINATION_TOKENS"]
    LENDING_RESERVES = ctx["LENDING_RESERVES"]
    PREFERRED_STABLECOIN = ctx["PREFERRED_STABLECOIN"]

    all_pairs_df = client.fetch_pair_universe().to_pandas()
    pairs_df = filter_for_selected_pairs(all_pairs_df, SUPPORTING_PAIRS)
    print(f"Total {len(all_pairs_df)} pairs, using {len(pairs_df)} for strategy")

    vault_universe = load_vault_database(path=DEFAULT_VAULT_DATABASE)
    total_vaults = vault_universe.get_vault_count()
    vault_universe = vault_universe.limit_to_vaults(VAULTS, check_all_vaults_found=False)
    vault_universe = vault_universe.limit_to_denomination(ALLOWED_VAULT_DENOMINATION_TOKENS, check_all_vaults_found=True)
    print(f"Loaded {vault_universe.get_vault_count()} vaults from total {total_vaults}")

    vault_bundled_price_data = DEFAULT_RAW_PRICE_DATABASE

    execution_context = notebook_execution_context
    universe_options = UniverseOptions.from_strategy_parameters_class(Parameters, execution_context)

    dataset = load_partial_data(
        client=client,
        time_bucket=parameters.candle_time_bucket,
        pairs=pairs_df,
        execution_context=execution_context,
        universe_options=universe_options,
        liquidity=True,
        liquidity_time_bucket=TimeBucket.d1,
        lending_reserves=LENDING_RESERVES,
        vaults=vault_universe,
        vault_bundled_price_data=vault_bundled_price_data,
        check_all_vaults_found=True,
    )

    strategy_universe = TradingStrategyUniverse.create_from_dataset(
        dataset,
        reserve_asset=PREFERRED_STABLECOIN,
        forward_fill=True,
        forward_fill_until=None,
        primary_chain=parameters.primary_chain_id,
    )
    print("Universe creation done")
    ctx["strategy_universe"] = strategy_universe


# =============================================================================
# Cell 11: Indicators definition + calculation
# =============================================================================
def cell_11_indicators(ctx):
    import pandas as pd
    import pandas_ta
    import numpy as np
    from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSource, calculate_and_load_indicators_inline
    from tradeexecutor.strategy.pandas_trader.indicator import IndicatorDependencyResolver
    from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
    from tradeexecutor.state.types import USDollarAmount
    from tradeexecutor.strategy.pandas_trader.indicator_decorator import IndicatorRegistry
    from tradeexecutor.state.identifier import TradingPairIdentifier
    from tradeexecutor.strategy.execution_context import ExecutionContext

    strategy_universe = ctx["strategy_universe"]
    Parameters = ctx["Parameters"]
    parameters = ctx["parameters"]
    SUPPORTING_PAIRS = ctx["SUPPORTING_PAIRS"]

    indicators = IndicatorRegistry()
    empty_series = pd.Series([], index=pd.DatetimeIndex([]))

    def _pad_daily_returns(close: pd.Series, volatility_window: int) -> pd.Series:
        daily_returns = close.pct_change().fillna(0)
        pad_index = pd.date_range(
            end=daily_returns.index[0] - pd.Timedelta(days=1),
            periods=volatility_window,
            freq=daily_returns.index.inferred_freq or "D",
        )
        pad = pd.Series(0.0, index=pad_index)
        return pd.concat([pad, daily_returns])

    @indicators.define()
    def rolling_returns(close: pd.Series, volatility_window: int = 180) -> pd.Series:
        min_periods = 7
        first_price = close.rolling(window=volatility_window, min_periods=min_periods).apply(lambda x: x[0], raw=True)
        actual_days = close.rolling(window=volatility_window, min_periods=min_periods).apply(lambda x: len(x), raw=True)
        period_return = close / first_price - 1
        annualised = (1 + period_return) ** (365 / actual_days) - 1
        return annualised

    @indicators.define()
    def rolling_sharpe(close: pd.Series, volatility_window: int = 180) -> pd.Series:
        min_periods = 14
        daily_returns = close.pct_change()
        rolling_mean = daily_returns.rolling(window=volatility_window, min_periods=min_periods).mean()
        rolling_std = daily_returns.rolling(window=volatility_window, min_periods=min_periods).std()
        sharpe = (rolling_mean * 365) / (rolling_std * (365 ** 0.5))
        return sharpe

    @indicators.define()
    def rolling_calmar(close: pd.Series, volatility_window: int = 180) -> pd.Series:
        daily_returns = _pad_daily_returns(close, volatility_window)
        rolling_mean = daily_returns.rolling(window=volatility_window, min_periods=volatility_window).mean()

        def _max_drawdown(window):
            cumulative = (1 + window).cumprod()
            peak = cumulative.cummax()
            dd = (cumulative / peak - 1).min()
            return max(abs(dd), 1e-4)

        rolling_mdd = daily_returns.rolling(window=volatility_window, min_periods=volatility_window).apply(_max_drawdown, raw=False)
        calmar = (rolling_mean * 365) / rolling_mdd
        calmar = calmar.reindex(close.index)
        return calmar

    @indicators.define()
    def rolling_sterling_floor(close: pd.Series, volatility_window: int = 180, sterling_constant: float = 0.10) -> pd.Series:
        daily_returns = _pad_daily_returns(close, volatility_window)
        rolling_mean = daily_returns.rolling(window=volatility_window, min_periods=volatility_window).mean()

        def _max_drawdown(window):
            cumulative = (1 + window).cumprod()
            peak = cumulative.cummax()
            dd = (cumulative / peak - 1).min()
            return abs(dd)

        rolling_mdd = daily_returns.rolling(window=volatility_window, min_periods=volatility_window).apply(_max_drawdown, raw=False)
        sterling = (rolling_mean * 365) / (rolling_mdd + sterling_constant)
        sterling = sterling.reindex(close.index)
        return sterling

    @indicators.define()
    def drawdown_from_peak(close: pd.Series) -> pd.Series:
        cummax = close.cummax()
        drawdown = close / cummax - 1
        return drawdown

    @indicators.define(
        dependencies=(rolling_calmar, rolling_sterling_floor, drawdown_from_peak),
        source=IndicatorSource.dependencies_only_per_pair,
    )
    def rolling_calmar_transformed(
        pair: TradingPairIdentifier,
        dependency_resolver: IndicatorDependencyResolver,
        volatility_window: int = 180,
        calmar_signal_transform: str = "raw",
        bayesian_halflife: int = 30,
        sterling_constant: float = 0.10,
    ) -> pd.Series:
        calmar = dependency_resolver.get_indicator_data(
            "rolling_calmar", pair=pair,
            parameters={"volatility_window": volatility_window},
        )

        if calmar_signal_transform == "bayesian_credibility":
            BAYESIAN_HALFLIFE = bayesian_halflife
            sterling = dependency_resolver.get_indicator_data(
                "rolling_sterling_floor", pair=pair,
                parameters={"volatility_window": volatility_window, "sterling_constant": sterling_constant},
            )
            base = np.log1p(sterling.clip(lower=0))
            all_sterling = dependency_resolver.get_indicator_data_pairs_combined(
                "rolling_sterling_floor",
                parameters={"volatility_window": volatility_window, "sterling_constant": sterling_constant},
            )
            sterling_df = all_sterling.unstack(level="pair_id")
            log_sterling_df = np.log1p(sterling_df.clip(lower=0))
            prior = log_sterling_df.mean(axis=1)
            real_days = (~calmar.isna()).cumsum().astype(float)
            w_t = real_days / (real_days + BAYESIAN_HALFLIFE)
            return w_t * base + (1 - w_t) * prior

        return calmar

    @indicators.define(source=IndicatorSource.tvl)
    def tvl(close: pd.Series, execution_context: ExecutionContext, timestamp: pd.Timestamp) -> pd.Series:
        if execution_context.live_trading:
            assert isinstance(timestamp, pd.Timestamp)
            from tradingstrategy.utils.forward_fill import forward_fill
            df = pd.DataFrame({"close": close})
            df_ff = forward_fill(df, Parameters.candle_time_bucket.to_frequency(), columns=("close",), forward_fill_until=timestamp)
            return df_ff["close"]
        else:
            return close.resample("1h").ffill()

    @indicators.define()
    def age(close: pd.Series) -> pd.Series:
        inception = close.index[0]
        age_years = (close.index - inception) / pd.Timedelta(days=365.25)
        return pd.Series(age_years, index=close.index)

    @indicators.define(dependencies=(tvl,), source=IndicatorSource.dependencies_only_universe)
    def tvl_inclusion_criteria(min_tvl: USDollarAmount, dependency_resolver: IndicatorDependencyResolver) -> pd.Series:
        series = dependency_resolver.get_indicator_data_pairs_combined(tvl)
        mask = series >= min_tvl
        mask_true_values_only = mask[mask == True]
        series = mask_true_values_only.groupby(level='timestamp').apply(lambda x: x.index.get_level_values('pair_id').tolist())
        return series

    @indicators.define(dependencies=(age,), source=IndicatorSource.dependencies_only_universe)
    def age_inclusion_criteria(min_age: float, dependency_resolver: IndicatorDependencyResolver) -> pd.Series:
        series = dependency_resolver.get_indicator_data_pairs_combined(age)
        mask = series >= min_age
        mask_true_values_only = mask[mask == True]
        series = mask_true_values_only.groupby(level='timestamp').apply(lambda x: x.index.get_level_values('pair_id').tolist())
        return series

    @indicators.define(source=IndicatorSource.strategy_universe)
    def trading_availability_criteria(strategy_universe: TradingStrategyUniverse) -> pd.Series:
        candle_series = strategy_universe.data_universe.candles.df["open"]
        pairs_per_timestamp = candle_series.groupby(level='timestamp').apply(lambda x: x.index.get_level_values('pair_id').tolist())
        return pairs_per_timestamp

    @indicators.define(
        dependencies=[tvl_inclusion_criteria, trading_availability_criteria, age_inclusion_criteria],
        source=IndicatorSource.strategy_universe,
    )
    def inclusion_criteria(
        strategy_universe: TradingStrategyUniverse,
        min_tvl: USDollarAmount,
        min_age: float,
        dependency_resolver: IndicatorDependencyResolver,
    ) -> pd.Series:
        benchmark_pair_ids = set(strategy_universe.get_pair_by_human_description(desc).internal_id for desc in SUPPORTING_PAIRS)
        tvl_series = dependency_resolver.get_indicator_data(tvl_inclusion_criteria, parameters={"min_tvl": min_tvl})
        trading_availability_series = dependency_resolver.get_indicator_data(trading_availability_criteria)
        age_series = dependency_resolver.get_indicator_data(age_inclusion_criteria, parameters={"min_age": min_age})

        df = pd.DataFrame({
            "tvl_pair_ids": tvl_series,
            "trading_availability_pair_ids": trading_availability_series,
            "age_pair_ids": age_series,
        })
        df = df.fillna("").apply(list)

        def _combine_criteria(row):
            return set(row["tvl_pair_ids"]) & set(row["trading_availability_pair_ids"]) & set(row["age_pair_ids"]) - benchmark_pair_ids

        union_criteria = df.apply(_combine_criteria, axis=1)
        full_index = pd.date_range(
            start=union_criteria.index.min(),
            end=union_criteria.index.max(),
            freq=Parameters.candle_time_bucket.to_frequency(),
        )
        return union_criteria.reindex(full_index, fill_value=[])

    @indicators.define(dependencies=(tvl_inclusion_criteria,), source=IndicatorSource.dependencies_only_universe)
    def tvl_included_pair_count(min_tvl: USDollarAmount, dependency_resolver: IndicatorDependencyResolver) -> pd.Series:
        series = dependency_resolver.get_indicator_data(tvl_inclusion_criteria, parameters={"min_tvl": min_tvl})
        series = series.apply(len)
        full_index = pd.date_range(start=series.index.min(), end=series.index.max(), freq=Parameters.candle_time_bucket.to_frequency())
        return series.reindex(full_index, fill_value=0)

    @indicators.define(dependencies=(age_inclusion_criteria,), source=IndicatorSource.dependencies_only_universe)
    def age_included_pair_count(min_age: float, dependency_resolver: IndicatorDependencyResolver) -> pd.Series:
        series = dependency_resolver.get_indicator_data(age_inclusion_criteria, parameters={"min_age": min_age})
        series = series.apply(len)
        full_index = pd.date_range(start=series.index.min(), end=series.index.max(), freq=Parameters.candle_time_bucket.to_frequency())
        return series.reindex(full_index, fill_value=0)

    @indicators.define(dependencies=(inclusion_criteria,), source=IndicatorSource.dependencies_only_universe)
    def all_criteria_included_pair_count(min_tvl: USDollarAmount, min_age: float, dependency_resolver: IndicatorDependencyResolver) -> pd.Series:
        series = dependency_resolver.get_indicator_data("inclusion_criteria", parameters={"min_tvl": min_tvl, "min_age": min_age})
        return series.apply(len)

    @indicators.define(source=IndicatorSource.strategy_universe)
    def trading_pair_count(strategy_universe: TradingStrategyUniverse) -> pd.Series:
        benchmark_pair_ids = {strategy_universe.get_pair_by_human_description(desc).internal_id for desc in SUPPORTING_PAIRS}
        series = strategy_universe.data_universe.candles.df["open"]
        swap_index = series.index.swaplevel(0, 1)
        seen_pairs = set()
        seen_data = {}
        for timestamp, pair_id in swap_index:
            if pair_id in benchmark_pair_ids:
                continue
            seen_pairs.add(pair_id)
            seen_data[timestamp] = len(seen_pairs)
        return pd.Series(seen_data.values(), index=list(seen_data.keys()))

    indicator_data = calculate_and_load_indicators_inline(
        strategy_universe=strategy_universe,
        create_indicators=indicators.create_indicators,
        parameters=parameters,
        max_workers=1,
    )
    ctx["indicator_data"] = indicator_data
    ctx["indicators"] = indicators
    ctx["create_indicators"] = indicators.create_indicators


# =============================================================================
# Cell 13: Time range extraction
# =============================================================================
def cell_13_time_range(ctx):
    Parameters = ctx["Parameters"]
    ctx["backtest_start"] = Parameters.backtest_start
    ctx["backtest_end"] = Parameters.backtest_end
    print(f"Time range is {ctx['backtest_start']} - {ctx['backtest_end']}")


# =============================================================================
# Cell 15: decide_trades function definition
# =============================================================================
def cell_15_decide_trades(ctx):
    import numpy as np
    from tradeexecutor.strategy.alpha_model import AlphaModel
    from tradeexecutor.state.trade import TradeExecution
    from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput, IndicatorDataNotFoundWithinDataTolerance
    from tradeexecutor.strategy.tvl_size_risk import USDTVLSizeRiskModel
    from tradeexecutor.strategy.weighting import weight_by_blend
    from tradeexecutor.utils.dedent import dedent_any
    from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
    from getting_started.curator import is_quarantined

    SUPPORTING_PAIRS = ctx["SUPPORTING_PAIRS"]

    HLP_ADDRESS = "0xdfc24b077bc1425ad1dea75bcb6f8158e10df303"

    def decide_trades(input: StrategyInput) -> list[TradeExecution]:
        parameters = input.parameters
        position_manager = input.get_position_manager()
        state = input.state
        timestamp = input.timestamp
        indicators = input.indicators
        strategy_universe = input.strategy_universe

        portfolio = position_manager.get_current_portfolio()
        equity = portfolio.get_total_equity()

        if input.execution_context.mode == ExecutionMode.backtesting:
            if equity < parameters.initial_cash * 0.10:
                return []

        hlp_mode = str(parameters.hlp_mode)
        hlp_target_pct = float(parameters.hlp_target_pct)

        alpha_model = AlphaModel(
            timestamp,
            close_position_weight_epsilon=parameters.min_portfolio_weight,
        )

        tvl_included_pair_count = indicators.get_indicator_value("tvl_included_pair_count")

        included_pairs = indicators.get_indicator_value("inclusion_criteria", na_conversion=False)
        if included_pairs is None:
            included_pairs = []

        weight_indicator_name = "rolling_sharpe"
        gate_indicator_name = "rolling_calmar_transformed"

        hlp_pair = None
        hlp_pair_id = None

        signal_count = 0
        gate_filtered = 0

        for pair_id in included_pairs:
            pair = strategy_universe.get_pair_by_id(pair_id)

            if not state.is_good_pair(pair):
                continue

            if is_quarantined(pair.pool_address, timestamp):
                continue

            if pair.pool_address and pair.pool_address.lower() == HLP_ADDRESS:
                hlp_pair = pair
                hlp_pair_id = pair_id

            gate_signal = indicators.get_indicator_value(gate_indicator_name, pair=pair)
            if gate_signal is None or gate_signal < 0:
                gate_filtered += 1
                continue

            weight_signal_value = indicators.get_indicator_value(weight_indicator_name, pair=pair)
            if weight_signal_value is None or weight_signal_value < 0:
                weight_signal_value = 1.0

            alpha_model.set_signal(pair, weight_signal_value)
            signal_count += 1

        portfolio = position_manager.get_current_portfolio()
        equity = portfolio.get_total_equity()

        if hlp_mode == "two_tier" and hlp_pair_id is not None and hlp_pair_id in alpha_model.signals:
            hlp_reserve = equity * float(parameters.allocation) * hlp_target_pct
            portfolio_target_value = equity * float(parameters.allocation) * (1.0 - hlp_target_pct)
        else:
            hlp_reserve = 0.0
            portfolio_target_value = equity * float(parameters.allocation)

        alpha_model.select_top_signals(count=999)
        alpha_model.assign_weights(method=lambda s: weight_by_blend(s, blend_alpha=float(parameters.blend_alpha)))

        if hlp_mode == "weight_boost" and hlp_pair_id is not None and hlp_pair_id in alpha_model.signals:
            hlp_signal = alpha_model.signals[hlp_pair_id]
            if hlp_signal.raw_weight and hlp_signal.raw_weight > 0:
                other_weight_sum = sum(
                    s.raw_weight for pid, s in alpha_model.signals.items()
                    if pid != hlp_pair_id and s.raw_weight and s.raw_weight > 0
                )
                if other_weight_sum > 0:
                    target_hlp_weight = (hlp_target_pct / (1.0 - hlp_target_pct)) * other_weight_sum
                    hlp_signal.raw_weight = target_hlp_weight

        zero_weight_pairs = [
            pair_id for pair_id, s in alpha_model.signals.items()
            if s.raw_weight == 0.0
        ]
        for pair_id in zero_weight_pairs:
            del alpha_model.signals[pair_id]

        size_risk_model = USDTVLSizeRiskModel(
            pricing_model=input.pricing_model,
            per_position_cap=float(parameters.per_position_cap_of_pool),
            missing_tvl_placeholder_usd=0.0,
        )

        if hlp_mode in ("weight_boost", "two_tier") and hlp_pair_id is not None:
            def hlp_max_weight_fn(signal):
                if signal.pair.pool_address and signal.pair.pool_address.lower() == HLP_ADDRESS:
                    return min(hlp_target_pct + 0.10, 0.60)
                return None
        else:
            hlp_max_weight_fn = None

        alpha_model.normalise_weights(
            investable_equity=portfolio_target_value,
            size_risk_model=size_risk_model,
            max_weight=float(parameters.max_concentration),
            max_weight_function=hlp_max_weight_fn,
            max_positions=parameters.max_assets_in_portfolio,
            waterfall=parameters.waterfall,
        )

        if hlp_mode == "two_tier" and hlp_pair_id is not None and hlp_pair_id in alpha_model.signals:
            hlp_signal = alpha_model.signals[hlp_pair_id]
            if hlp_signal.position_target is not None:
                hlp_signal.position_target += hlp_reserve
            else:
                hlp_signal.position_target = hlp_reserve

        if hlp_mode == "cash_sweep" and hlp_pair_id is not None and hlp_pair_id in alpha_model.signals:
            leftover = alpha_model.investable_equity - alpha_model.accepted_investable_equity
            if leftover > 0:
                max_hlp_total = equity * float(parameters.allocation) * hlp_target_pct
                hlp_signal = alpha_model.signals[hlp_pair_id]
                current_hlp = hlp_signal.position_target or 0.0
                sweep_amount = min(leftover, max_hlp_total - current_hlp)
                if sweep_amount > 0:
                    hlp_signal.position_target = current_hlp + sweep_amount

        alpha_model.update_old_weights(state.portfolio, ignore_credit=False)
        alpha_model.calculate_target_positions(position_manager)

        rebalance_threshold_usd = parameters.individual_rebalance_min_threshold_usd
        assert rebalance_threshold_usd > 0.1

        trades = alpha_model.generate_rebalance_trades_and_triggers(
            position_manager,
            min_trade_threshold=rebalance_threshold_usd,
            invidiual_rebalance_min_threshold=parameters.individual_rebalance_min_threshold_usd,
            sell_rebalance_min_threshold=parameters.sell_rebalance_min_threshold,
            execution_context=input.execution_context,
        )

        if input.is_visualisation_enabled():
            try:
                top_signal = next(iter(alpha_model.get_signals_sorted_by_weight()))
                if top_signal.normalised_weight == 0:
                    top_signal = None
            except StopIteration:
                top_signal = None

            rebalance_volume = sum(t.get_value() for t in trades)
            hlp_pos_target = 0
            if hlp_pair_id and hlp_pair_id in alpha_model.signals:
                hlp_pos_target = alpha_model.signals[hlp_pair_id].position_target or 0

            report = dedent_any(f"""
            Cycle: #{input.cycle}
            Rebalanced: {'yes' if alpha_model.is_rebalance_triggered() else 'no'}
            Open/about to open positions: {len(state.portfolio.open_positions)}
            Max position value change: {alpha_model.max_position_adjust_usd:,.2f} USD
            Rebalance threshold: {alpha_model.position_adjust_threshold_usd:,.2f} USD
            Trades decided: {len(trades)}
            Pairs total: {strategy_universe.data_universe.pairs.get_count()}
            Pairs meeting inclusion criteria: {len(included_pairs)}
            Pairs meeting TVL inclusion criteria: {tvl_included_pair_count}
            Signals created: {signal_count}
            Filtered by Bayesian gate: {gate_filtered}
            HLP mode: {hlp_mode}, target: {hlp_target_pct:.0%}
            HLP position target: {hlp_pos_target:,.2f} USD
            Total equity: {portfolio.get_total_equity():,.2f} USD
            Cash: {position_manager.get_current_cash():,.2f} USD
            Investable equity: {alpha_model.investable_equity:,.2f} USD
            Accepted investable equity: {alpha_model.accepted_investable_equity:,.2f} USD
            Allocated to signals: {alpha_model.get_allocated_value():,.2f} USD
            Discarted allocation because of lack of lit liquidity: {alpha_model.size_risk_discarded_value:,.2f} USD
            Rebalance volume: {rebalance_volume:,.2f} USD
            """)

            if top_signal:
                assert top_signal.position_size_risk
                report += dedent_any(f"""
                Top signal pair: {top_signal.pair.get_ticker()}
                Top signal value: {top_signal.signal}
                Top signal weight: {top_signal.raw_weight}
                Top signal weight (normalised): {top_signal.normalised_weight * 100:.2f} % (got {top_signal.position_size_risk.get_relative_capped_amount() * 100:.2f} % of asked size)
                """)

            for flag, count in alpha_model.get_flag_diagnostics_data().items():
                report += f"Signals with flag {flag.name}: {count}\n"

            state.visualisation.add_message(timestamp, report)
            state.visualisation.set_discardable_data("alpha_model", alpha_model)

        return trades

    ctx["decide_trades"] = decide_trades
    print("decide_trades() function defined")


# =============================================================================
# Cell 17: Optimiser
# =============================================================================
def cell_17_optimiser(ctx):
    from tradeexecutor.backtest.optimiser import perform_optimisation, prepare_optimiser_parameters
    from tradeexecutor.backtest.optimiser import MinTradeCountFilter
    from tradeexecutor.backtest.optimiser_functions import optimise_calmar

    search_func = optimise_calmar

    # Reduced iterations for profiling (3 x 2 = 6 backtests instead of 10 x 6 = 60)
    iterations = 3
    batch_size = 2

    optimiser_result = perform_optimisation(
        iterations=iterations,
        search_func=search_func,
        decide_trades=ctx["decide_trades"],
        strategy_universe=ctx["strategy_universe"],
        parameters=prepare_optimiser_parameters(ctx["Parameters"]),
        create_indicators=ctx["create_indicators"],
        result_filter=MinTradeCountFilter(50),
        timeout=70 * 60,
        batch_size=batch_size,
        max_workers=1,  # Single process for accurate cProfile
        ignore_wallet_errors=True,
        draw_visualisation=False,
    )

    print(f"Optimise completed, {optimiser_result.get_combination_count()} combinations, "
          f"{optimiser_result.get_cached_count()} cached, {optimiser_result.get_filtered_count()} filtered")
    print(f"Backtests failed with exception: {optimiser_result.get_failed_count()}")

    ctx["optimiser_result"] = optimiser_result
    ctx["search_func"] = search_func


# =============================================================================
# Cell 20: Analyse optimiser results
# =============================================================================
def cell_20_results_table(ctx):
    from tradeexecutor.analysis.optimiser import analyse_optimiser_result
    from tradeexecutor.analysis.grid_search import render_grid_search_result_table

    optimiser_result = ctx["optimiser_result"]
    filtered = [r for r in optimiser_result.results if r.filtered]
    print(f"Filtering out {len(filtered)} results")

    df = analyse_optimiser_result(optimiser_result, max_search_results=300, drop_duplicates=False)
    print(f"Showing the best {len(df)} results")

    render_grid_search_result_table(df, calmar=True, sortino=False)
    ctx["results_df"] = df


# =============================================================================
# Cell 23: Decision tree
# =============================================================================
def cell_23_decision_tree(ctx):
    from tradeexecutor.analysis.grid_search_parameters import analyse_decision_tree

    df = ctx["results_df"]
    optimiser_result = ctx["optimiser_result"]
    df_clean = df.dropna(subset=[optimiser_result.target_metric_name])
    fig, tree = analyse_decision_tree(df_clean, analysis_metric=optimiser_result.target_metric_name)
    ctx["df_clean"] = df_clean


# =============================================================================
# Cell 25: Feature importance
# =============================================================================
def cell_25_feature_importance(ctx):
    from tradeexecutor.analysis.grid_search_parameters import analyse_feature_importance

    fig, importances = analyse_feature_importance(
        ctx["df_clean"],
        analysis_metric=ctx["optimiser_result"].target_metric_name,
    )


# =============================================================================
# Cell 27: Parameter pair heatmaps
# =============================================================================
def cell_27_heatmaps(ctx):
    from tradeexecutor.analysis.grid_search_parameters import analyse_parameter_pair_heatmaps
    import plotly.graph_objects as go

    # Pre-warm Kaleido
    go.Figure().to_image(format="png")

    figs = analyse_parameter_pair_heatmaps(
        ctx["df_clean"],
        analysis_metric=ctx["optimiser_result"].target_metric_name,
    )


# =============================================================================
# Cell 29: Grid search equity curves
# =============================================================================
def cell_29_equity_curves(ctx):
    from tradeexecutor.visual.grid_search_basic import visualise_grid_search_equity_curves
    from tradeexecutor.analysis.multi_asset_benchmark import get_benchmark_data

    Parameters = ctx["Parameters"]
    optimiser_result = ctx["optimiser_result"]

    benchmark_indexes = get_benchmark_data(
        ctx["strategy_universe"],
        cumulative_with_initial_cash=Parameters.initial_cash,
    )

    grid_search_results = [r.result for r in optimiser_result.results if not r.filtered]

    fig = visualise_grid_search_equity_curves(
        grid_search_results,
        benchmark_indexes=benchmark_indexes,
        log_y=False,
        group_by="hlp_mode",
        color_mode="discrete",
    )


# =============================================================================
# Cell 31: Best pick equity curve
# =============================================================================
def cell_31_best_pick(ctx):
    from tradeexecutor.visual.grid_search import visualise_single_grid_search_result_benchmark

    optimiser_result = ctx["optimiser_result"]
    Parameters = ctx["Parameters"]

    best_pick = optimiser_result.results[0].result
    state = best_pick.hydrate_state()

    print(f"Best result: {best_pick}")

    fig = visualise_single_grid_search_result_benchmark(
        best_pick,
        ctx["strategy_universe"],
        initial_cash=Parameters.initial_cash,
        log_y=True,
    )

    ctx["best_pick"] = best_pick
    ctx["state"] = state


# =============================================================================
# Cell 33: Multi-asset comparison
# =============================================================================
def cell_33_multi_asset(ctx):
    from tradeexecutor.analysis.multi_asset_benchmark import compare_strategy_backtest_to_multiple_assets

    best_pick = ctx["best_pick"]
    returns = best_pick.returns

    df = compare_strategy_backtest_to_multiple_assets(
        state=ctx["state"],
        strategy_universe=ctx["strategy_universe"],
        returns=returns,
        display=True,
        asset_count=3,
    )
    ctx["returns"] = returns


# =============================================================================
# Cell 35: Trade summary
# =============================================================================
def cell_35_trade_summary(ctx):
    summary = ctx["best_pick"].summary
    summary.to_dataframe()


# =============================================================================
# Cell 37: Multipair analysis
# =============================================================================
def cell_37_multipair(ctx):
    from tradeexecutor.analysis.multipair import analyse_multipair, format_multipair_summary

    multipair_summary = analyse_multipair(ctx["state"], show_address=True)
    format_multipair_summary(multipair_summary)


# =============================================================================
# Cell 39: Rolling Sharpe plot
# =============================================================================
def cell_39_rolling_sharpe(ctx):
    import plotly.express as px
    from tradeexecutor.visual.equity_curve import calculate_rolling_sharpe

    rolling_sharpe = calculate_rolling_sharpe(ctx["returns"], freq="D", periods=90)
    fig = px.line(rolling_sharpe, title='Strategy rolling Sharpe (6 months)')


# =============================================================================
# Cell 41: Chart registry setup
# =============================================================================
def cell_41_chart_registry(ctx):
    import pandas as pd
    from tradeexecutor.strategy.chart.definition import ChartRegistry, ChartKind, ChartInput
    from tradeexecutor.strategy.chart.renderer import ChartBacktestRenderingSetup
    from tradeexecutor.strategy.chart.standard.weight import equity_curve_by_asset
    from tradeexecutor.strategy.chart.standard.weight import equity_curve_by_chain
    from tradeexecutor.strategy.chart.standard.weight import weight_allocation_statistics
    from tradeexecutor.strategy.chart.standard.weight import volatile_weights_by_percent
    from tradeexecutor.strategy.chart.standard.weight import volatile_and_non_volatile_percent
    from tradeexecutor.strategy.chart.standard.profit_breakdown import trading_pair_breakdown
    from tradeexecutor.strategy.chart.standard.interest import vault_statistics
    from tradeexecutor.strategy.chart.standard.vault import all_vault_positions
    from tradeexecutor.strategy.chart.standard.vault import all_vault_daily_gains_losses
    from tradeexecutor.strategy.chart.standard.vault import vault_position_timeline
    from tradeexecutor.strategy.chart.standard.trading_universe import available_trading_pairs
    from tradeexecutor.strategy.pandas_trader.indicator import load_indicators_inline
    from tradeexecutor.strategy.parameters import StrategyParameters

    Parameters = ctx["Parameters"]
    best_pick = ctx["best_pick"]
    BENCHMARK_PAIRS = ctx["BENCHMARK_PAIRS"]

    merged_parameters = StrategyParameters.from_class(Parameters)
    merged_parameters.update(best_pick.combination.as_dict())

    indicator_data = load_indicators_inline(
        strategy_universe=ctx["strategy_universe"],
        create_indicators=ctx["create_indicators"],
        parameters=merged_parameters,
    )

    def trading_pair_breakdown_with_chain(input: ChartInput) -> pd.DataFrame:
        """Trading pair breakdown with chain column and address."""
        return trading_pair_breakdown(input, show_chain=True, show_address=True)

    def all_vault_positions_by_profit(input: ChartInput) -> pd.DataFrame:
        """Top 10 winning and bottom 10 losing vault positions."""
        return all_vault_positions(input, sort_by="Profit USD", sort_ascending=False, show_address=True, top_n=10, bottom_n=10)

    def biggest_daily_gains_losses(input: ChartInput) -> pd.DataFrame:
        """Top 10 biggest daily gains and bottom 10 biggest daily losses."""
        return all_vault_daily_gains_losses(input, top_n=10, bottom_n=10)

    charts = ChartRegistry(default_benchmark_pairs=BENCHMARK_PAIRS)
    charts.register(equity_curve_by_asset, ChartKind.state_all_pairs)
    charts.register(equity_curve_by_chain, ChartKind.state_all_pairs)
    charts.register(weight_allocation_statistics, ChartKind.state_all_pairs)
    charts.register(volatile_weights_by_percent, ChartKind.state_all_pairs)
    charts.register(volatile_and_non_volatile_percent, ChartKind.state_all_pairs)
    charts.register(trading_pair_breakdown_with_chain, ChartKind.state_all_pairs)
    charts.register(vault_statistics, ChartKind.state_all_pairs)
    charts.register(all_vault_positions_by_profit, ChartKind.state_single_vault_pair)
    charts.register(biggest_daily_gains_losses, ChartKind.state_single_vault_pair)
    charts.register(vault_position_timeline, ChartKind.state_single_vault_pair)
    charts.register(available_trading_pairs, ChartKind.indicator_all_pairs)

    chart_renderer = ChartBacktestRenderingSetup(
        registry=charts,
        state=ctx["state"],
        backtest_start_at=ctx["backtest_start"],
        backtest_end_at=ctx["backtest_end"],
        strategy_input_indicators=indicator_data,
    )

    ctx["chart_renderer"] = chart_renderer
    ctx["charts"] = charts
    ctx["chart_indicator_data"] = indicator_data
    # Store chart functions for later cells
    ctx["available_trading_pairs"] = available_trading_pairs
    ctx["trading_pair_breakdown_with_chain"] = trading_pair_breakdown_with_chain
    ctx["vault_statistics"] = vault_statistics
    ctx["equity_curve_by_asset"] = equity_curve_by_asset
    ctx["weight_allocation_statistics"] = weight_allocation_statistics
    ctx["all_vault_positions_by_profit"] = all_vault_positions_by_profit
    ctx["biggest_daily_gains_losses"] = biggest_daily_gains_losses


# =============================================================================
# Cell 43: Available trading pairs
# =============================================================================
def cell_43_available_pairs(ctx):
    fig, df = ctx["chart_renderer"].render(ctx["available_trading_pairs"], with_dataframe=True)


# =============================================================================
# Cell 45: Trading pair breakdown
# =============================================================================
def cell_45_pair_breakdown(ctx):
    df = ctx["chart_renderer"].render(ctx["trading_pair_breakdown_with_chain"])


# =============================================================================
# Cell 48: Vault statistics
# =============================================================================
def cell_48_vault_stats(ctx):
    df = ctx["chart_renderer"].render(ctx["vault_statistics"])


# =============================================================================
# Cell 50: Rolling Sharpe (repeated)
# =============================================================================
def cell_50_rolling_sharpe(ctx):
    import plotly.express as px
    from tradeexecutor.visual.equity_curve import calculate_equity_curve, calculate_returns, calculate_rolling_sharpe

    equity_curve = calculate_equity_curve(ctx["state"])
    returns = calculate_returns(equity_curve)
    rolling_sharpe = calculate_rolling_sharpe(returns, freq="D", periods=180)
    fig = px.line(rolling_sharpe, title='Strategy rolling Sharpe (6 months)')


# =============================================================================
# Cell 53: Equity curve by asset
# =============================================================================
def cell_53_equity_by_asset(ctx):
    fig = ctx["chart_renderer"].render(ctx["equity_curve_by_asset"])


# =============================================================================
# Cell 55: Weight allocation statistics
# =============================================================================
def cell_55_weight_stats(ctx):
    stats = ctx["chart_renderer"].render(ctx["weight_allocation_statistics"])


# =============================================================================
# Cell 57: Vault positions
# =============================================================================
def cell_57_vault_positions(ctx):
    df = ctx["chart_renderer"].render(ctx["all_vault_positions_by_profit"])


# =============================================================================
# Cell 58: Daily gains/losses
# =============================================================================
def cell_58_daily_gains(ctx):
    df = ctx["chart_renderer"].render(ctx["biggest_daily_gains_losses"])


# =============================================================================
# Main
# =============================================================================
def write_summary(timings, output_dir):
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("Notebook 73 profiling summary\n")
        f.write(f"Date: {datetime.datetime.utcnow().isoformat()}\n")
        f.write("=" * 70 + "\n\n")

        total = 0
        for name, elapsed in timings.items():
            if isinstance(elapsed, float):
                f.write(f"{name:55s} {elapsed:10.3f}s\n")
                total += elapsed
            else:
                f.write(f"{name:55s} {elapsed}\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write(f"{'TOTAL':55s} {total:10.3f}s\n")

    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    print("Profiling notebook 73: hyperliquid-hlp-cash-sweep-modes")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ctx = {}
    timings = {}

    cells = [
        ("cell_03: Client + charting setup", lambda: cell_03_client_setup(ctx)),
        ("cell_05: Chain config + vault universe", lambda: cell_05_chain_config(ctx)),
        ("cell_07: Strategy parameters", lambda: cell_07_parameters(ctx)),
        ("cell_09: Universe creation", lambda: cell_09_universe(ctx)),
        ("cell_11: Indicators + calculation", lambda: cell_11_indicators(ctx)),
        ("cell_13: Time range", lambda: cell_13_time_range(ctx)),
        ("cell_15: decide_trades definition", lambda: cell_15_decide_trades(ctx)),
        ("cell_17: Optimiser (max_workers=1)", lambda: cell_17_optimiser(ctx)),
        ("cell_20: Results table", lambda: cell_20_results_table(ctx)),
        ("cell_23: Decision tree", lambda: cell_23_decision_tree(ctx)),
        ("cell_25: Feature importance", lambda: cell_25_feature_importance(ctx)),
        ("cell_27: Heatmaps", lambda: cell_27_heatmaps(ctx)),
        ("cell_29: Equity curves", lambda: cell_29_equity_curves(ctx)),
        ("cell_31: Best pick", lambda: cell_31_best_pick(ctx)),
        ("cell_33: Multi-asset comparison", lambda: cell_33_multi_asset(ctx)),
        ("cell_35: Trade summary", lambda: cell_35_trade_summary(ctx)),
        ("cell_37: Multipair analysis", lambda: cell_37_multipair(ctx)),
        ("cell_39: Rolling Sharpe", lambda: cell_39_rolling_sharpe(ctx)),
        ("cell_41: Chart registry setup", lambda: cell_41_chart_registry(ctx)),
        ("cell_43: Available trading pairs", lambda: cell_43_available_pairs(ctx)),
        ("cell_45: Trading pair breakdown", lambda: cell_45_pair_breakdown(ctx)),
        ("cell_48: Vault statistics", lambda: cell_48_vault_stats(ctx)),
        ("cell_50: Rolling Sharpe (repeated)", lambda: cell_50_rolling_sharpe(ctx)),
        ("cell_53: Equity curve by asset", lambda: cell_53_equity_by_asset(ctx)),
        ("cell_55: Weight allocation stats", lambda: cell_55_weight_stats(ctx)),
        ("cell_57: Vault positions", lambda: cell_57_vault_positions(ctx)),
        ("cell_58: Daily gains/losses", lambda: cell_58_daily_gains(ctx)),
    ]

    for name, func in cells:
        try:
            _, elapsed = profile_cell(name, func)
            timings[name] = elapsed
        except Exception as e:
            timings[name] = f"ERROR: {e}"
            print(f"  Stopping due to error in {name}")
            break

    write_summary(timings, OUTPUT_DIR)

    print(f"\n{'='*80}")
    print("DONE")
    print("=" * 80)
