"""Profile each cell of the 40-hyperliquid-august-start vault-of-vaults notebook.

Profiles every code cell to find bottlenecks in the notebook execution.
Skips commented-out cells and groups chart-render cells together.

Usage:
    poetry run python scripts/profile_vault_of_vaults_notebook.py

Requirements:
    - .env file in working directory or getting-started repo with API keys
    - Vault database at ~/.tradingstrategy/vaults/
"""

import cProfile
import pstats
import io
import sys
import time
import datetime
import logging

logging.basicConfig(level=logging.WARNING)

# Add getting-started to path for helper imports
sys.path.insert(0, "/Users/moo/code/getting-started")


def profile_cell(name, func):
    """Profile a cell and print results."""
    print(f"\n{'='*80}")
    print(f"CELL: {name}")
    print(f"{'='*80}")

    profiler = cProfile.Profile()
    wall_start = time.perf_counter()
    profiler.enable()
    result = func()
    profiler.disable()
    wall_elapsed = time.perf_counter() - wall_start

    stream = io.StringIO()
    ps = pstats.Stats(profiler, stream=stream)

    # Cumulative time
    ps.sort_stats("cumulative")
    ps.print_stats(30)
    print(stream.getvalue())

    # Self time
    stream2 = io.StringIO()
    ps2 = pstats.Stats(profiler, stream=stream2)
    ps2.sort_stats("tottime")
    ps2.print_stats(20)
    print("--- Top by self time ---")
    print(stream2.getvalue())

    print(f"Wall time: {wall_elapsed:.3f}s")
    return result


# =============================================================================
# Cell 3: Client + charting setup
# =============================================================================
def cell_3_setup():
    from tradingstrategy.client import Client
    from tradeexecutor.utils.notebook import setup_charting_and_output, OutputMode

    client = Client.create_jupyter_client()
    setup_charting_and_output(OutputMode.static, image_format="png", width=1500, height=1000)
    return client


# =============================================================================
# Cell 5: Chain configuration + vault universe building
# =============================================================================
def cell_5_chain_config():
    from eth_defi.token import USDC_NATIVE_TOKEN
    from tradingstrategy.chain import ChainId

    CHAIN_ID = ChainId.cross_chain
    PRIMARY_CHAIN_ID = ChainId.ethereum
    HYPERCORE_CHAIN_ID = ChainId.hypercore

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
        top_n=120,
        min_age=0.15,
    )

    ALLOWED_VAULT_DENOMINATION_TOKENS = {"USDC", "USDT", "USDC.e", "crvUSD", "USD₮0", "USDt", "USDT0", "USDS"}

    BENCHMARK_PAIRS = [
        (ChainId.ethereum, "uniswap-v3", "WBTC", "USDC", 0.003),
    ]

    print(f"Chain universe: {CHAIN_ID}")
    print(f"Vaults count: {len(VAULTS)}")

    return {
        "CHAIN_ID": CHAIN_ID,
        "PRIMARY_CHAIN_ID": PRIMARY_CHAIN_ID,
        "EXCHANGES": EXCHANGES,
        "SUPPORTING_PAIRS": SUPPORTING_PAIRS,
        "LENDING_RESERVES": LENDING_RESERVES,
        "PREFERRED_STABLECOIN": PREFERRED_STABLECOIN,
        "VAULTS": VAULTS,
        "ALLOWED_VAULT_DENOMINATION_TOKENS": ALLOWED_VAULT_DENOMINATION_TOKENS,
        "BENCHMARK_PAIRS": BENCHMARK_PAIRS,
    }


# =============================================================================
# Cell 7: Parameters
# =============================================================================
def cell_7_parameters(config):
    import datetime
    import pandas as pd
    from tradingstrategy.timebucket import TimeBucket
    from tradeexecutor.strategy.cycle import CycleDuration
    from tradeexecutor.strategy.parameters import StrategyParameters
    from tradeexecutor.strategy.default_routing_options import TradeRouting

    class Parameters:
        id = "40-hyperliquid-august-start"
        candle_time_bucket = TimeBucket.d1
        cycle_duration = CycleDuration.cycle_4d
        chain_id = config["CHAIN_ID"]
        primary_chain_id = config["PRIMARY_CHAIN_ID"]
        exchanges = config["EXCHANGES"]
        min_asset_universe = 5
        max_assets_in_portfolio = 45
        allocation = 0.95
        individual_rebalance_min_threshold_usd = 50.0
        sell_rebalance_min_threshold = 10.0
        sell_threshold = 0.05
        per_position_cap_of_pool = 0.10
        max_concentration = 0.33
        min_portfolio_weight = 0.0050
        rolling_returns_bars = 60
        weighting_method = "rolling_returns"
        weight_function = "weight_1_slash_n"
        waterfall = True
        volatility_window = 60
        min_tvl = 10_000
        max_rolling_volatility = 1.50
        backtest_start = datetime.datetime(2025, 8, 1)
        backtest_end = datetime.datetime(2026, 3, 11)
        initial_cash = 100_000
        routing = TradeRouting.default
        required_history_period = datetime.timedelta(days=60*2)
        slippage_tolerance = 0.0060
        assummed_liquidity_when_data_missings = 10_000

    parameters = StrategyParameters.from_class(Parameters)
    return Parameters, parameters


# =============================================================================
# Cell 9: Universe creation (the heavy one)
# =============================================================================
def cell_9_universe(client, config, Parameters, parameters):
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

    SUPPORTING_PAIRS = config["SUPPORTING_PAIRS"]
    VAULTS = config["VAULTS"]
    ALLOWED_VAULT_DENOMINATION_TOKENS = config["ALLOWED_VAULT_DENOMINATION_TOKENS"]
    LENDING_RESERVES = config["LENDING_RESERVES"]
    PREFERRED_STABLECOIN = config["PREFERRED_STABLECOIN"]

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
    return strategy_universe


# =============================================================================
# Cell 11: Indicators definition + calculation
# =============================================================================
def cell_11_indicators(strategy_universe, Parameters, parameters, config):
    import pandas as pd
    import pandas_ta
    from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSource, calculate_and_load_indicators_inline
    from tradeexecutor.strategy.pandas_trader.indicator import IndicatorDependencyResolver
    from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
    from tradeexecutor.state.types import USDollarAmount
    from tradeexecutor.strategy.pandas_trader.indicator_decorator import IndicatorRegistry
    from tradeexecutor.state.identifier import TradingPairIdentifier
    from tradeexecutor.strategy.execution_context import ExecutionContext
    from tradingstrategy.timebucket import TimeBucket

    SUPPORTING_PAIRS = config["SUPPORTING_PAIRS"]

    indicators = IndicatorRegistry()
    empty_series = pd.Series([], index=pd.DatetimeIndex([]))

    @indicators.define()
    def rolling_returns(close: pd.Series, rolling_returns_bars: int = 60) -> pd.Series:
        min_periods = 7
        first_price = close.rolling(window=rolling_returns_bars, min_periods=min_periods).apply(lambda x: x[0], raw=True)
        actual_days = close.rolling(window=rolling_returns_bars, min_periods=min_periods).apply(lambda x: len(x), raw=True)
        period_return = close / first_price - 1
        annualised = (1 + period_return) ** (365 / actual_days) - 1
        return annualised

    @indicators.define()
    def rolling_volatility(close: pd.Series, volatility_window: int = 180) -> pd.Series:
        min_periods = 14
        daily_returns = close.pct_change()
        rolling_std = daily_returns.rolling(window=volatility_window, min_periods=min_periods).std()
        return rolling_std * (365 ** 0.5)

    @indicators.define()
    def rolling_sharpe(close: pd.Series, volatility_window: int = 180) -> pd.Series:
        min_periods = 14
        daily_returns = close.pct_change()
        rolling_mean = daily_returns.rolling(window=volatility_window, min_periods=min_periods).mean()
        rolling_std = daily_returns.rolling(window=volatility_window, min_periods=min_periods).std()
        return (rolling_mean * 365) / (rolling_std * (365 ** 0.5))

    @indicators.define()
    def rolling_sortino(close: pd.Series, volatility_window: int = 180) -> pd.Series:
        min_periods = 14
        daily_returns = close.pct_change()
        rolling_mean = daily_returns.rolling(window=volatility_window, min_periods=min_periods).mean()
        downside_returns = daily_returns.clip(upper=0)
        rolling_downside_std = downside_returns.rolling(window=volatility_window, min_periods=min_periods).std()
        return (rolling_mean * 365) / (rolling_downside_std * (365 ** 0.5))

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

    @indicators.define(dependencies=(tvl,), source=IndicatorSource.dependencies_only_universe)
    def tvl_inclusion_criteria(min_tvl: USDollarAmount, dependency_resolver: IndicatorDependencyResolver) -> pd.Series:
        series = dependency_resolver.get_indicator_data_pairs_combined(tvl)
        mask = series >= min_tvl
        mask_true_values_only = mask[mask == True]
        series = mask_true_values_only.groupby(level='timestamp').apply(lambda x: x.index.get_level_values('pair_id').tolist())
        return series

    @indicators.define(source=IndicatorSource.strategy_universe)
    def trading_availability_criteria(strategy_universe: TradingStrategyUniverse) -> pd.Series:
        candle_series = strategy_universe.data_universe.candles.df["open"]
        pairs_per_timestamp = candle_series.groupby(level='timestamp').apply(lambda x: x.index.get_level_values('pair_id').tolist())
        return pairs_per_timestamp

    @indicators.define(
        dependencies=[tvl_inclusion_criteria, trading_availability_criteria],
        source=IndicatorSource.strategy_universe,
    )
    def inclusion_criteria(
        strategy_universe: TradingStrategyUniverse,
        min_tvl: USDollarAmount,
        dependency_resolver: IndicatorDependencyResolver,
    ) -> pd.Series:
        benchmark_pair_ids = set(strategy_universe.get_pair_by_human_description(desc).internal_id for desc in SUPPORTING_PAIRS)
        tvl_series = dependency_resolver.get_indicator_data(tvl_inclusion_criteria, parameters={"min_tvl": min_tvl})
        trading_availability_series = dependency_resolver.get_indicator_data(trading_availability_criteria)
        df = pd.DataFrame({"tvl_pair_ids": tvl_series, "trading_availability_pair_ids": trading_availability_series})
        df = df.fillna("").apply(list)
        def _combine_criteria(row):
            return set(row["tvl_pair_ids"]) & set(row["trading_availability_pair_ids"]) - benchmark_pair_ids
        union_criteria = df.apply(_combine_criteria, axis=1)
        full_index = pd.date_range(start=union_criteria.index.min(), end=union_criteria.index.max(), freq=Parameters.candle_time_bucket.to_frequency())
        return union_criteria.reindex(full_index, fill_value=[])

    @indicators.define(dependencies=(tvl_inclusion_criteria,), source=IndicatorSource.dependencies_only_universe)
    def tvl_included_pair_count(min_tvl: USDollarAmount, dependency_resolver: IndicatorDependencyResolver) -> pd.Series:
        series = dependency_resolver.get_indicator_data(tvl_inclusion_criteria, parameters={"min_tvl": min_tvl})
        series = series.apply(len)
        full_index = pd.date_range(start=series.index.min(), end=series.index.max(), freq=Parameters.candle_time_bucket.to_frequency())
        return series.reindex(full_index, fill_value=0)

    @indicators.define(dependencies=(inclusion_criteria,), source=IndicatorSource.dependencies_only_universe)
    def all_criteria_included_pair_count(min_tvl: USDollarAmount, dependency_resolver: IndicatorDependencyResolver) -> pd.Series:
        series = dependency_resolver.get_indicator_data("inclusion_criteria", parameters={"min_tvl": min_tvl})
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
    )
    return indicator_data, indicators.create_indicators


# =============================================================================
# Cell 13: Chart registry setup
# =============================================================================
def cell_13_chart_registry(indicator_data, Parameters):
    import pandas as pd
    from plotly.graph_objects import Figure
    from tradeexecutor.strategy.chart.definition import ChartRegistry, ChartKind, ChartInput
    from tradeexecutor.strategy.chart.renderer import ChartBacktestRenderingSetup
    from tradeexecutor.strategy.chart.standard.trading_universe import available_trading_pairs
    from tradeexecutor.strategy.chart.standard.trading_universe import inclusion_criteria_check
    from tradeexecutor.strategy.chart.standard.volatility import volatility_benchmark
    from tradeexecutor.strategy.chart.standard.signal import signal_comparison
    from tradeexecutor.strategy.chart.standard.signal import price_vs_signal
    from tradeexecutor.strategy.chart.standard.vault import all_vaults_share_price_and_tvl as _all_vaults_share_price_and_tvl
    from tradeexecutor.strategy.chart.standard.vault import vault_position_timeline
    from tradeexecutor.strategy.chart.standard.vault import all_vault_positions
    from tradeexecutor.strategy.chart.standard.equity_curve import equity_curve
    from tradeexecutor.strategy.chart.standard.equity_curve import equity_curve_with_drawdown
    from tradeexecutor.strategy.chart.standard.performance_metrics import performance_metrics
    from tradeexecutor.strategy.chart.standard.weight import volatile_weights_by_percent
    from tradeexecutor.strategy.chart.standard.weight import volatile_and_non_volatile_percent
    from tradeexecutor.strategy.chart.standard.weight import equity_curve_by_asset
    from tradeexecutor.strategy.chart.standard.weight import equity_curve_by_chain
    from tradeexecutor.strategy.chart.standard.weight import weight_allocation_statistics
    from tradeexecutor.strategy.chart.standard.position import positions_at_end
    from tradeexecutor.strategy.chart.standard.thinking import last_messages
    from tradeexecutor.strategy.chart.standard.alpha_model import alpha_model_diagnostics
    from tradeexecutor.strategy.chart.standard.profit_breakdown import trading_pair_breakdown
    from tradeexecutor.strategy.chart.standard.trading_metrics import trading_metrics
    from tradeexecutor.strategy.chart.standard.interest import lending_pool_interest_accrued
    from tradeexecutor.strategy.chart.standard.interest import vault_statistics
    from tradeexecutor.strategy.chart.standard.single_pair import trading_pair_positions
    from tradeexecutor.strategy.chart.standard.single_pair import trading_pair_price_and_trades

    from tradingstrategy.chain import ChainId

    BENCHMARK_PAIRS = [
        (ChainId.ethereum, "uniswap-v3", "WBTC", "USDC", 0.003),
    ]

    def equity_curve_with_benchmark(input: ChartInput) -> list[Figure]:
        """Equity curve with ETH benchmark."""
        return equity_curve(input, benchmark_token_symbols=["ETH"])

    def all_vaults_share_price_and_tvl(input: ChartInput) -> list[Figure]:
        """All vaults share price and TVL."""
        return _all_vaults_share_price_and_tvl(input, max_count=2)

    def inclusion_criteria_check_with_chain(input: ChartInput) -> pd.DataFrame:
        """Inclusion criteria check with chain."""
        return inclusion_criteria_check(input, show_chain=True)

    def trading_pair_breakdown_with_chain(input: ChartInput) -> pd.DataFrame:
        """Trading pair breakdown with chain."""
        return trading_pair_breakdown(input, show_chain=True, show_address=True)

    def all_vault_positions_by_profit(input: ChartInput) -> pd.DataFrame:
        """Vault positions by profit."""
        return all_vault_positions(input, sort_by="Profit USD", sort_ascending=False, show_address=True)

    charts = ChartRegistry(default_benchmark_pairs=BENCHMARK_PAIRS)
    charts.register(available_trading_pairs, ChartKind.indicator_all_pairs)
    charts.register(inclusion_criteria_check, ChartKind.indicator_all_pairs)
    charts.register(volatility_benchmark, ChartKind.indicator_multi_pair)
    charts.register(signal_comparison, ChartKind.indicator_multi_pair)
    charts.register(price_vs_signal, ChartKind.indicator_multi_pair)
    charts.register(all_vaults_share_price_and_tvl, ChartKind.indicator_all_pairs)
    charts.register(equity_curve_with_benchmark, ChartKind.state_all_pairs)
    charts.register(equity_curve_with_drawdown, ChartKind.state_all_pairs)
    charts.register(performance_metrics, ChartKind.state_all_pairs)
    charts.register(volatile_weights_by_percent, ChartKind.state_all_pairs)
    charts.register(volatile_and_non_volatile_percent, ChartKind.state_all_pairs)
    charts.register(equity_curve_by_asset, ChartKind.state_all_pairs)
    charts.register(equity_curve_by_chain, ChartKind.state_all_pairs)
    charts.register(weight_allocation_statistics, ChartKind.state_all_pairs)
    charts.register(positions_at_end, ChartKind.state_all_pairs)
    charts.register(last_messages, ChartKind.state_all_pairs)
    charts.register(alpha_model_diagnostics, ChartKind.state_all_pairs)
    charts.register(trading_pair_breakdown_with_chain, ChartKind.state_all_pairs)
    charts.register(trading_metrics, ChartKind.state_all_pairs)
    charts.register(lending_pool_interest_accrued, ChartKind.state_all_pairs)
    charts.register(vault_statistics, ChartKind.state_all_pairs)
    charts.register(vault_position_timeline, ChartKind.state_single_vault_pair)
    charts.register(all_vault_positions_by_profit, ChartKind.state_single_vault_pair)
    charts.register(trading_pair_positions, ChartKind.state_single_vault_pair)
    charts.register(trading_pair_price_and_trades, ChartKind.state_single_vault_pair)
    charts.register(inclusion_criteria_check_with_chain, ChartKind.indicator_all_pairs)

    chart_renderer = ChartBacktestRenderingSetup(
        registry=charts,
        strategy_input_indicators=indicator_data,
        backtest_start_at=Parameters.backtest_start,
        backtest_end_at=Parameters.backtest_end,
    )

    return chart_renderer, charts


# =============================================================================
# Cell 26: Backtest
# =============================================================================
def cell_26_backtest(strategy_universe, indicator_data, create_indicators, Parameters, parameters, config, client):
    import numpy as np
    import pandas as pd
    from tradeexecutor.backtest.backtest_runner import run_backtest_inline
    from tradeexecutor.strategy.alpha_model import AlphaModel
    from tradeexecutor.state.trade import TradeExecution
    from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput, IndicatorDataNotFoundWithinDataTolerance
    from tradeexecutor.state.visualisation import PlotKind
    from tradeexecutor.strategy.tvl_size_risk import USDTVLSizeRiskModel
    from tradeexecutor.strategy.weighting import weight_by_1_slash_n, weight_passthrouh
    from tradeexecutor.utils.dedent import dedent_any
    from tradeexecutor.strategy.pandas_trader.yield_manager import YieldManager, YieldRuleset, YieldWeightingRule, YieldDecisionInput
    from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
    from tradeexecutor.strategy.cycle import CycleDuration
    from tradeexecutor.strategy.chart.definition import ChartRegistry

    SUPPORTING_PAIRS = config["SUPPORTING_PAIRS"]

    def decide_trades(input: StrategyInput) -> list[TradeExecution]:
        parameters = input.parameters
        position_manager = input.get_position_manager()
        state = input.state
        timestamp = input.timestamp
        indicators = input.indicators
        strategy_universe = input.strategy_universe
        execution_context = input.execution_context

        max_assets_in_portfolio = parameters.max_assets_in_portfolio
        allocation = parameters.allocation
        min_portfolio_weight = parameters.min_portfolio_weight
        max_concentration = parameters.max_concentration
        weighting_method = parameters.weighting_method
        rolling_returns_bars = parameters.rolling_returns_bars
        waterfall = parameters.waterfall

        inclusion_criteria = indicators.get_indicator_value("inclusion_criteria", na_conversion=False)

        if inclusion_criteria is None or (isinstance(inclusion_criteria, float) and np.isnan(inclusion_criteria)):
            return []

        included_pair_ids = inclusion_criteria
        if not isinstance(included_pair_ids, (set, list)):
            return []
        if len(included_pair_ids) == 0:
            return []

        alpha_model = AlphaModel(timestamp)

        for pair_id in included_pair_ids:
            pair = strategy_universe.get_pair_by_id(pair_id)
            signal = indicators.get_indicator_value("rolling_returns", pair=pair)

            if signal is None or (isinstance(signal, float) and np.isnan(signal)):
                continue
            if signal <= 0:
                continue

            alpha_model.set_signal(pair, signal)

        alpha_model.select_top_signals(max_assets_in_portfolio)
        alpha_model.assign_weights(method=weight_by_1_slash_n)
        alpha_model.normalise_weights(
            max_weight=max_concentration,
            investable_equity_ratio=allocation,
        )
        alpha_model.update_old_weights(
            state.portfolio,
            ignore_credit=True,
        )
        alpha_model.calculate_target_positions(position_manager)

        trades = alpha_model.generate_rebalance_trades_and_triggers(
            position_manager,
            min_trade_threshold=parameters.individual_rebalance_min_threshold_usd,
            individual_rebalance_min_threshold=parameters.sell_rebalance_min_threshold,
        )

        return trades

    state, strategy_universe_out, debug_dump = run_backtest_inline(
        client=client,
        decide_trades=decide_trades,
        create_indicators=create_indicators,
        universe=strategy_universe,
        cycle_duration=Parameters.cycle_duration,
        initial_deposit=Parameters.initial_cash,
        parameters=parameters,
        start_at=Parameters.backtest_start,
        end_at=Parameters.backtest_end,
        name="40-hyperliquid-august-start-profile",
        log_level=logging.WARNING,
        engine_version="0.5",
    )

    print(f"Backtest done, positions: {sum(1 for _ in state.portfolio.get_all_positions())}")
    return state


# =============================================================================
# Cell 28+: Chart rendering cells (grouped)
# =============================================================================
def cell_28_charts(chart_renderer, state, strategy_universe, indicator_data, charts, config, Parameters):
    """Run all chart render cells (28-65) and time each one."""
    import pandas as pd
    from tradeexecutor.strategy.chart.renderer import ChartBacktestRenderingSetup
    from tradeexecutor.analysis.multi_asset_benchmark import compare_strategy_backtest_to_multiple_assets

    # Inject backtest state into chart renderer
    chart_renderer = ChartBacktestRenderingSetup(
        registry=charts,
        strategy_input_indicators=indicator_data,
        state=state,
        backtest_start_at=Parameters.backtest_start,
        backtest_end_at=Parameters.backtest_end,
    )

    results = {}

    # Cell 28: performance metrics
    t0 = time.perf_counter()
    try:
        df = chart_renderer.render("performance_metrics")
        results["28: performance_metrics"] = time.perf_counter() - t0
    except Exception as e:
        results["28: performance_metrics"] = f"ERROR: {e}"

    # Cell 30: equity curve with benchmark
    t0 = time.perf_counter()
    try:
        fig = chart_renderer.render("equity_curve_with_benchmark")
        results["30: equity_curve_with_benchmark"] = time.perf_counter() - t0
    except Exception as e:
        results["30: equity_curve_with_benchmark"] = f"ERROR: {e}"

    # Cell 32: equity curve with drawdown
    t0 = time.perf_counter()
    try:
        fig = chart_renderer.render("equity_curve_with_drawdown")
        results["32: equity_curve_with_drawdown"] = time.perf_counter() - t0
    except Exception as e:
        results["32: equity_curve_with_drawdown"] = f"ERROR: {e}"

    # Cell 39: equity curve by asset
    t0 = time.perf_counter()
    try:
        fig = chart_renderer.render("equity_curve_by_asset")
        results["39: equity_curve_by_asset"] = time.perf_counter() - t0
    except Exception as e:
        results["39: equity_curve_by_asset"] = f"ERROR: {e}"

    # Cell 41: equity curve by chain
    t0 = time.perf_counter()
    try:
        fig = chart_renderer.render("equity_curve_by_chain")
        results["41: equity_curve_by_chain"] = time.perf_counter() - t0
    except Exception as e:
        results["41: equity_curve_by_chain"] = f"ERROR: {e}"

    # Cell 43: weight allocation stats
    t0 = time.perf_counter()
    try:
        stats = chart_renderer.render("weight_allocation_statistics")
        results["43: weight_allocation_statistics"] = time.perf_counter() - t0
    except Exception as e:
        results["43: weight_allocation_statistics"] = f"ERROR: {e}"

    # Cell 45: rolling sharpe
    t0 = time.perf_counter()
    try:
        from tradeexecutor.visual.equity_curve import calculate_equity_curve, calculate_returns, calculate_rolling_sharpe
        equity_curve_data = calculate_equity_curve(state)
        returns = calculate_returns(equity_curve_data)
        rolling_sharpe_data = calculate_rolling_sharpe(returns, freq="D", periods=180)
        results["45: rolling_sharpe"] = time.perf_counter() - t0
    except Exception as e:
        results["45: rolling_sharpe"] = f"ERROR: {e}"

    # Cell 47: positions at end
    t0 = time.perf_counter()
    try:
        stats = chart_renderer.render("positions_at_end")
        results["47: positions_at_end"] = time.perf_counter() - t0
    except Exception as e:
        results["47: positions_at_end"] = f"ERROR: {e}"

    # Cell 49: last messages
    t0 = time.perf_counter()
    try:
        df = chart_renderer.render("last_messages")
        results["49: last_messages"] = time.perf_counter() - t0
    except Exception as e:
        results["49: last_messages"] = f"ERROR: {e}"

    # Cell 51: alpha model diagnostics
    t0 = time.perf_counter()
    try:
        df = chart_renderer.render("alpha_model_diagnostics")
        results["51: alpha_model_diagnostics"] = time.perf_counter() - t0
    except Exception as e:
        results["51: alpha_model_diagnostics"] = f"ERROR: {e}"

    # Cell 53: trading pair breakdown
    t0 = time.perf_counter()
    try:
        df = chart_renderer.render("trading_pair_breakdown_with_chain")
        results["53: trading_pair_breakdown"] = time.perf_counter() - t0
    except Exception as e:
        results["53: trading_pair_breakdown"] = f"ERROR: {e}"

    # Cell 55: trading metrics
    t0 = time.perf_counter()
    try:
        df = chart_renderer.render("trading_metrics")
        results["55: trading_metrics"] = time.perf_counter() - t0
    except Exception as e:
        results["55: trading_metrics"] = f"ERROR: {e}"

    # Cell 58: lending pool interest
    t0 = time.perf_counter()
    try:
        df = chart_renderer.render("lending_pool_interest_accrued")
        results["58: lending_pool_interest"] = time.perf_counter() - t0
    except Exception as e:
        results["58: lending_pool_interest"] = f"ERROR: {e}"

    # Cell 61: vault statistics
    t0 = time.perf_counter()
    try:
        df = chart_renderer.render("vault_statistics")
        results["61: vault_statistics"] = time.perf_counter() - t0
    except Exception as e:
        results["61: vault_statistics"] = f"ERROR: {e}"

    # Cell 63: vault positions
    t0 = time.perf_counter()
    try:
        df = chart_renderer.render("all_vault_positions_by_profit")
        results["63: vault_positions"] = time.perf_counter() - t0
    except Exception as e:
        results["63: vault_positions"] = f"ERROR: {e}"

    return results


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("Profiling vault-of-vaults notebook: 40-hyperliquid-august-start")
    print("=" * 80)

    timings = {}

    # Cell 3
    client = profile_cell("3: Client + charting setup", cell_3_setup)
    timings["3: Client + charting setup"] = None  # Will be filled by profile_cell output

    # Cell 5
    config = profile_cell("5: Chain config + vault universe", cell_5_chain_config)

    # Cell 7
    Parameters, parameters = profile_cell("7: Parameters", lambda: cell_7_parameters(config))

    # Cell 9: Universe creation
    strategy_universe = profile_cell(
        "9: Universe creation",
        lambda: cell_9_universe(client, config, Parameters, parameters),
    )

    # Cell 11: Indicators
    indicator_data, create_indicators = profile_cell(
        "11: Indicators + calculation",
        lambda: cell_11_indicators(strategy_universe, Parameters, parameters, config),
    )

    # Cell 13: Chart registry
    chart_renderer, charts = profile_cell(
        "13: Chart registry setup",
        lambda: cell_13_chart_registry(indicator_data, Parameters),
    )

    # Cell 26: Backtest
    state = profile_cell(
        "26: Backtest",
        lambda: cell_26_backtest(strategy_universe, indicator_data, create_indicators, Parameters, parameters, config, client),
    )

    # Cells 28-65: Chart rendering
    print(f"\n{'='*80}")
    print("CELLS 28-65: Chart rendering (individual timings)")
    print(f"{'='*80}")

    wall_start = time.perf_counter()
    chart_results = cell_28_charts(chart_renderer, state, strategy_universe, indicator_data, charts, config, Parameters)
    wall_elapsed = time.perf_counter() - wall_start

    for name, elapsed in chart_results.items():
        if isinstance(elapsed, float):
            print(f"  {name}: {elapsed:.3f}s")
        else:
            print(f"  {name}: {elapsed}")

    print(f"\n  Total chart rendering: {wall_elapsed:.3f}s")

    print(f"\n{'='*80}")
    print("DONE")
    print("=" * 80)
