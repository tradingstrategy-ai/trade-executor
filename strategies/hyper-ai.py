"""Hyperliquid vault of vaults - minimum-hold cleanup.

Based on notebook ``153-research-hyperliquid-equal-weight-recycle-minimum-hold-cleanup.ipynb``.

Uses ``age_ramp`` for survivor selection, equal weights, survivor-first recycling,
and a minimum 3-day hold rule to prevent rapid position churning.

Backtest results (2025-08-02 to 2026-03-10)
=============================================

Last backtest run: 2026-03-19

====================  =========  ==========  ==========
Metric                Strategy   BTC         ETH
====================  =========  ==========  ==========
Start period          2025-08-02 2025-08-02  2025-08-02
End period            2026-03-10 2026-03-10  2026-03-10
Risk-free rate        0.0%       0.0%        0.0%
Time in market        35.0%      83.0%       82.0%
Cumulative return     41.18%     -37.54%     -52.75%
CAGR﹪                 77.2%      -54.19%     -71.17%
Sharpe                2.15       -1.49       -1.56
Prob. Sharpe ratio    99.25%     12.27%      11.26%
Smart Sharpe          1.6        -1.11       -1.16
Sortino               6.11       -1.95       -2.06
Smart Sortino         4.55       -1.45       -1.53
Sortino/√2            4.32       -1.38       -1.46
Smart Sortino/√2      3.22       -1.03       -1.09
Omega                 2.17       2.17        2.17
Max drawdown          -6.12%     -49.82%     -61.35%
====================  =========  ==========  ==========
"""

#
# Imports
#

import datetime
import logging

import pandas as pd
from eth_defi.token import USDC_NATIVE_TOKEN
from plotly.graph_objects import Figure
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.utils.token_filter import filter_for_selected_pairs

from tradeexecutor.curator.curator import is_quarantined
from tradeexecutor.curator.hyperliquid_vault_universe import build_hyperliquid_vault_universe
from tradeexecutor.exchange_account.allocation import (
    calculate_portfolio_target_value,
    get_redeemable_portfolio_capital,
)

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.types import USDollarAmount
from tradeexecutor.strategy.alpha_model import AlphaModel
from tradeexecutor.strategy.chart.definition import ChartInput, ChartKind, ChartRegistry
from tradeexecutor.strategy.chart.standard.alpha_model import alpha_model_diagnostics
from tradeexecutor.strategy.chart.standard.equity_curve import (
    equity_curve, equity_curve_with_drawdown,
)
from tradeexecutor.strategy.chart.standard.interest import vault_statistics
from tradeexecutor.strategy.chart.standard.performance_metrics import performance_metrics
from tradeexecutor.strategy.chart.standard.position import positions_at_end
from tradeexecutor.strategy.chart.standard.profit_breakdown import trading_pair_breakdown
from tradeexecutor.strategy.chart.standard.thinking import last_messages
from tradeexecutor.strategy.chart.standard.trading_metrics import trading_metrics
from tradeexecutor.strategy.chart.standard.trading_universe import (
    available_trading_pairs, inclusion_criteria_check,
)
from tradeexecutor.strategy.chart.standard.vault import (
    all_vault_daily_gains_losses, all_vault_positions,
)
from tradeexecutor.strategy.chart.standard.weight import (
    equity_curve_by_asset, equity_curve_by_chain, weight_allocation_statistics,
)
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.pandas_trader.indicator import (
    IndicatorDependencyResolver, IndicatorSource,
)
from tradeexecutor.strategy.pandas_trader.indicator_decorator import IndicatorRegistry
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.pandas_trader.trading_universe_input import CreateTradingUniverseInput
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.tag import StrategyTag
from tradeexecutor.strategy.trading_strategy_universe import (
    TradingStrategyUniverse, load_partial_data, load_vault_universe_with_metadata,
)
from tradeexecutor.strategy.tvl_size_risk import USDTVLSizeRiskModel
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradeexecutor.strategy.weighting import weight_equal
from tradeexecutor.utils.dedent import dedent_any

logger = logging.getLogger(__name__)

#
# Trading universe constants
#

trading_strategy_engine_version = "0.5"

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

VAULTS: list[tuple[ChainId, str]] | None = None

ALLOWED_VAULT_DENOMINATION_TOKENS = {"USDC", "USDT", "USDC.e", "crvUSD", "USD₮0", "USDt", "USDT0", "USDS"}

BENCHMARK_PAIRS = [
    (ChainId.ethereum, "uniswap-v3", "WBTC", "USDC", 0.003),
]

#
# Strategy parameters
#


class Parameters:

    id = "hyper-ai"

    candle_time_bucket = TimeBucket.d1
    cycle_duration = CycleDuration.cycle_1d

    chain_id = CHAIN_ID
    primary_chain_id = PRIMARY_CHAIN_ID
    exchanges = EXCHANGES

    # Portfolio construction.
    max_assets_in_portfolio = 20
    allocation = 0.98
    max_concentration = 0.12
    per_position_cap_of_pool = 0.2
    min_portfolio_weight = 0.005

    # Trade generation.
    individual_rebalance_min_threshold_usd = 50.0
    sell_rebalance_min_threshold = 10.0

    # Final fixed signal configuration.
    min_tvl = 7_500
    min_age = 0.075
    weight_signal = "age_ramp"
    age_ramp_period = 0.75

    # Minimum hold: protect positions younger than this from forced sells.
    minimum_hold_days = 3

    # Backtesting only.
    backtest_start = datetime.datetime(2025, 8, 1)
    backtest_end = datetime.datetime(2026, 3, 11)
    initial_cash = 100000

    # Live only.
    routing = TradeRouting.default
    required_history_period = datetime.timedelta(days=60 * 2)
    slippage_tolerance = 0.0060


#
# Universe creation
#


def create_trading_universe(
    input: CreateTradingUniverseInput,
) -> TradingStrategyUniverse:
    """Create the trading universe.

    - Load Trading Strategy full pairs dataset
    - Load vault universe from the remote metadata blob
    - Load OHLCV data for supporting pairs
    - Load also BTC and ETH price data to be used as a benchmark
    """

    execution_context = input.execution_context
    client = input.client
    timestamp = input.timestamp
    parameters = input.parameters or Parameters
    universe_options = input.universe_options

    if execution_context.live_trading:
        debug_printer = logger.info
    else:
        debug_printer = print

    chain_id = parameters.primary_chain_id

    debug_printer(f"Preparing trading universe on chain {chain_id.get_name()}")

    all_pairs_df = client.fetch_pair_universe().to_pandas()
    pairs_df = filter_for_selected_pairs(
        all_pairs_df,
        SUPPORTING_PAIRS,
    )

    debug_printer(f"We have total {len(all_pairs_df)} pairs in dataset and going to use {len(pairs_df)} pairs for the strategy")

    def get_vaults() -> list[tuple[ChainId, str]]:
        """Load the curated Hypercore vault list lazily for universe creation.

        The strategy module is imported in unit and integration tests
        that only exercise the decision logic. We therefore avoid any live
        curator fetch during module import and only build the universe when the
        trading universe itself is requested.

        The resulting list is still cached for the process lifetime on purpose.
        This preserves the current intentional behaviour of using one curator
        view per process run.
        """
        global VAULTS
        if VAULTS is None:
            VAULTS = build_hyperliquid_vault_universe(
                min_tvl=5_000,
                min_age=0.0,
            )
        return VAULTS

    vaults = get_vaults()

    vault_universe = load_vault_universe_with_metadata(client, vaults=vaults)
    vault_universe = vault_universe.limit_to_denomination(ALLOWED_VAULT_DENOMINATION_TOKENS, check_all_vaults_found=True)
    debug_printer(f"Loaded {vault_universe.get_vault_count()} vaults from remote vault metadata, source vaults count: {len(vaults)}")

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
        vault_history_source="trading-strategy-website",
        check_all_vaults_found=True,
    )

    debug_printer("Creating strategy universe with price feeds and vaults")
    strategy_universe = TradingStrategyUniverse.create_from_dataset(
        dataset,
        reserve_asset=PREFERRED_STABLECOIN,
        forward_fill=True,
        forward_fill_until=timestamp,
        primary_chain=parameters.primary_chain_id,
    )
    return strategy_universe


#
# Strategy logic
#


def _get_minimum_hold_pair_ids(
    portfolio,
    strategy_universe: TradingStrategyUniverse,
    state,
    timestamp,
    minimum_hold_days: int,
) -> set[int]:
    """Return pair ids for positions that are still within the minimum hold window.

    - Positions opened within the last `minimum_hold_days` days are protected
      from being sold even when they fall out of the inclusion criteria.
    - Quarantined and bad pairs are excluded.
    """
    if minimum_hold_days <= 0:
        return set()

    threshold = datetime.timedelta(days=minimum_hold_days)
    current_dt = timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else timestamp
    pair_ids = set()

    for position in portfolio.get_open_positions():
        pair = position.pair
        if pair.is_credit_supply():
            continue
        if current_dt - position.opened_at >= threshold:
            continue
        quarantine_address = pair.pool_address
        if quarantine_address and is_quarantined(quarantine_address, timestamp):
            continue
        if not state.is_good_pair(pair):
            continue
        if strategy_universe.get_pair_by_id(pair.internal_id) is None:
            continue
        pair_ids.add(pair.internal_id)

    return pair_ids


def decide_trades(
    input: StrategyInput,
) -> list[TradeExecution]:
    """Run survivor-first equal-weight sizing with minimum-hold protection."""
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

    included_pairs = indicators.get_indicator_value(
        "inclusion_criteria",
        na_conversion=False,
    )
    if included_pairs is None:
        included_pairs = []

    tvl_included_pair_count = indicators.get_indicator_value("tvl_included_pair_count")

    # Build candidate signals from inclusion criteria pairs.
    candidate_signals = []
    candidate_signal_ids = set()
    for pair_id in included_pairs:
        pair = strategy_universe.get_pair_by_id(pair_id)
        if not state.is_good_pair(pair):
            continue
        quarantine_address = pair.pool_address
        if quarantine_address and is_quarantined(quarantine_address, timestamp):
            continue
        age_ramp = indicators.get_indicator_value("age_ramp_weight", pair=pair)
        candidate_signals.append({"pair": pair, "signal": age_ramp if age_ramp is not None else 1.0})
        candidate_signal_ids.add(pair.internal_id)

    # Add positions still under minimum hold that are not already candidates.
    minimum_hold_pair_ids = _get_minimum_hold_pair_ids(
        portfolio,
        strategy_universe,
        state,
        timestamp,
        parameters.minimum_hold_days,
    )
    for pair_id in minimum_hold_pair_ids:
        if pair_id in candidate_signal_ids:
            continue
        pair = strategy_universe.get_pair_by_id(pair_id)
        age_ramp = indicators.get_indicator_value("age_ramp_weight", pair=pair)
        candidate_signals.append({"pair": pair, "signal": age_ramp if age_ramp is not None else 1.0})
        candidate_signal_ids.add(pair_id)

    # Sort candidates: highest signal first, then by pair id for stability.
    ordered_candidates = sorted(
        candidate_signals,
        key=lambda item: (-item["signal"], item["pair"].internal_id),
    )

    # Select survivors: minimum-hold positions fill first, then top signals fill remaining slots.
    selected = []
    selected_ids = set()

    for item in ordered_candidates:
        pair_id = item["pair"].internal_id
        if pair_id in minimum_hold_pair_ids and pair_id not in selected_ids:
            selected.append(item)
            selected_ids.add(pair_id)

    remaining_slots = max(parameters.max_assets_in_portfolio - len(selected), 0)
    for item in ordered_candidates:
        if len(selected) >= len(minimum_hold_pair_ids) + remaining_slots:
            break
        pair_id = item["pair"].internal_id
        if pair_id in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(pair_id)

    alpha_model = AlphaModel(
        timestamp,
        close_position_weight_epsilon=parameters.min_portfolio_weight,
    )

    for item in selected:
        alpha_model.set_signal(item["pair"], item["signal"])

    redeemable_capital = get_redeemable_portfolio_capital(position_manager)
    portfolio_target_value = calculate_portfolio_target_value(
        position_manager,
        parameters.allocation,
    )

    alpha_model.select_top_signals(count=len(selected))
    alpha_model.assign_weights(method=weight_equal)

    # Hyperliquid vaults are expected to have complete TVL data in live trading.
    # Fail the cycle loudly if TVL is missing instead of silently sizing with a placeholder.
    size_risk_model = USDTVLSizeRiskModel(
        pricing_model=input.pricing_model,
        per_position_cap=parameters.per_position_cap_of_pool,
    )

    alpha_model.normalise_weights(
        investable_equity=portfolio_target_value,
        size_risk_model=size_risk_model,
        max_weight=parameters.max_concentration,
        max_positions=parameters.max_assets_in_portfolio,
        waterfall=False,
    )

    alpha_model.update_old_weights(
        state.portfolio,
        ignore_credit=False,
    )
    alpha_model.calculate_target_positions(position_manager)

    rebalance_threshold_usd = parameters.individual_rebalance_min_threshold_usd
    assert rebalance_threshold_usd > 0.1, "Safety check tripped"

    trades = alpha_model.generate_rebalance_trades_and_triggers(
        position_manager,
        min_trade_threshold=rebalance_threshold_usd,
        individual_rebalance_min_threshold=parameters.individual_rebalance_min_threshold_usd,
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

        rebalance_volume = sum(trade.get_value() for trade in trades)

        unheld_selected = {
            item["pair"].internal_id
            for item in ordered_candidates[:parameters.max_assets_in_portfolio]
        }
        blocked_sell_count = len(minimum_hold_pair_ids - unheld_selected)

        report = dedent_any(f"""
        Cycle: #{input.cycle}
        Rebalanced: {'👍' if alpha_model.is_rebalance_triggered() else '👎'}
        Open/about to open positions: {len(state.portfolio.open_positions)}
        Candidate signals created: {len(candidate_signals)}
        Selected survivor signals: {len(selected)}
        Signals blocked by minimum hold: {blocked_sell_count}
        Max position value change: {alpha_model.max_position_adjust_usd:,.2f} USD
        Rebalance threshold: {alpha_model.position_adjust_threshold_usd:,.2f} USD
        Trades decided: {len(trades)}
        Pairs meeting inclusion criteria: {len(included_pairs)}
        Pairs meeting TVL inclusion criteria: {tvl_included_pair_count}
        Weight signal: {parameters.weight_signal}
        Age ramp period: {parameters.age_ramp_period}
        Total equity: {portfolio.get_total_equity():,.2f} USD
        Cash: {position_manager.get_current_cash():,.2f} USD
        Redeemable capital: {redeemable_capital:,.2f} USD
        Pending redemptions: {position_manager.get_pending_redemptions():,.2f} USD
        Investable equity: {alpha_model.investable_equity:,.2f} USD
        Accepted investable equity: {alpha_model.accepted_investable_equity:,.2f} USD
        Allocated to signals: {alpha_model.get_allocated_value():,.2f} USD
        Discarded allocation because of lack of lit liquidity: {alpha_model.size_risk_discarded_value:,.2f} USD
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

        state.visualisation.add_message(
            timestamp,
            report,
        )
        state.visualisation.set_discardable_data("alpha_model", alpha_model)

    return trades


#
# Indicators
#

indicators = IndicatorRegistry()


@indicators.define(source=IndicatorSource.tvl)
def tvl(
    close: pd.Series,
    execution_context: ExecutionContext,
    timestamp: pd.Timestamp,
) -> pd.Series:
    """Get a forward-filled TVL series for a pair."""
    if execution_context.live_trading:
        assert isinstance(timestamp, pd.Timestamp), (
            f"Live trading needs forward-fill end time, got {timestamp}"
        )
        from tradingstrategy.utils.forward_fill import forward_fill

        df = pd.DataFrame({"close": close})
        df_ff = forward_fill(
            df,
            Parameters.candle_time_bucket.to_frequency(),
            columns=("close",),
            forward_fill_until=timestamp,
        )
        return df_ff["close"]

    return close.resample("1h").ffill()


@indicators.define()
def age(
    close: pd.Series,
) -> pd.Series:
    """Calculate vault age in years from the first available price point."""
    inception = close.index[0]
    age_years = (close.index - inception) / pd.Timedelta(days=365.25)
    return pd.Series(age_years, index=close.index)


@indicators.define(dependencies=(tvl,), source=IndicatorSource.dependencies_only_universe)
def tvl_inclusion_criteria(
    min_tvl: USDollarAmount,
    dependency_resolver: IndicatorDependencyResolver,
) -> pd.Series:
    """Return pair ids whose TVL is above the configured minimum."""
    series = dependency_resolver.get_indicator_data_pairs_combined(tvl)
    mask = series >= min_tvl
    mask_true_values_only = mask[mask == True]
    return mask_true_values_only.groupby(level="timestamp").apply(
        lambda x: x.index.get_level_values("pair_id").tolist()
    )


@indicators.define(dependencies=(age,), source=IndicatorSource.dependencies_only_universe)
def age_inclusion_criteria(
    min_age: float,
    dependency_resolver: IndicatorDependencyResolver,
) -> pd.Series:
    """Return pair ids whose vault age is above the configured minimum."""
    series = dependency_resolver.get_indicator_data_pairs_combined(age)
    mask = series >= min_age
    mask_true_values_only = mask[mask == True]
    return mask_true_values_only.groupby(level="timestamp").apply(
        lambda x: x.index.get_level_values("pair_id").tolist()
    )


@indicators.define(source=IndicatorSource.strategy_universe)
def trading_availability_criteria(
    strategy_universe: TradingStrategyUniverse,
) -> pd.Series:
    """Return pair ids that have price data at each timestamp."""
    candle_series = strategy_universe.data_universe.candles.df["open"]
    return candle_series.groupby(level="timestamp").apply(
        lambda x: x.index.get_level_values("pair_id").tolist()
    )


@indicators.define(
    dependencies=[
        tvl_inclusion_criteria,
        trading_availability_criteria,
        age_inclusion_criteria,
    ],
    source=IndicatorSource.strategy_universe,
)
def inclusion_criteria(
    strategy_universe: TradingStrategyUniverse,
    min_tvl: USDollarAmount,
    min_age: float,
    dependency_resolver: IndicatorDependencyResolver,
) -> pd.Series:
    """Pairs meeting the full live inclusion rule set."""
    benchmark_pair_ids = {
        strategy_universe.get_pair_by_human_description(desc).internal_id
        for desc in SUPPORTING_PAIRS
    }

    tvl_series = dependency_resolver.get_indicator_data(
        tvl_inclusion_criteria,
        parameters={"min_tvl": min_tvl},
    )
    trading_availability_series = dependency_resolver.get_indicator_data(
        trading_availability_criteria,
    )
    age_series = dependency_resolver.get_indicator_data(
        age_inclusion_criteria,
        parameters={"min_age": min_age},
    )

    df = pd.DataFrame(
        {
            "tvl_pair_ids": tvl_series,
            "trading_availability_pair_ids": trading_availability_series,
            "age_pair_ids": age_series,
        }
    )
    df = df.fillna("").apply(list)

    def _combine_criteria(row):
        final_set = (
            set(row["tvl_pair_ids"])
            & set(row["trading_availability_pair_ids"])
            & set(row["age_pair_ids"])
        )
        return final_set - benchmark_pair_ids

    union_criteria = df.apply(_combine_criteria, axis=1)
    full_index = pd.date_range(
        start=union_criteria.index.min(),
        end=union_criteria.index.max(),
        freq=Parameters.candle_time_bucket.to_frequency(),
    )
    return union_criteria.reindex(full_index, fill_value=[])


@indicators.define(
    dependencies=(tvl_inclusion_criteria,),
    source=IndicatorSource.dependencies_only_universe,
)
def tvl_included_pair_count(
    min_tvl: USDollarAmount,
    dependency_resolver: IndicatorDependencyResolver,
) -> pd.Series:
    """Count pairs passing the TVL filter at each timestamp."""
    series = dependency_resolver.get_indicator_data(
        tvl_inclusion_criteria,
        parameters={"min_tvl": min_tvl},
    ).apply(len)

    full_index = pd.date_range(
        start=series.index.min(),
        end=series.index.max(),
        freq=Parameters.candle_time_bucket.to_frequency(),
    )
    return series.reindex(full_index, fill_value=0)


@indicators.define(
    dependencies=(age_inclusion_criteria,),
    source=IndicatorSource.dependencies_only_universe,
)
def age_included_pair_count(
    min_age: float,
    dependency_resolver: IndicatorDependencyResolver,
) -> pd.Series:
    """Count pairs passing the age filter at each timestamp."""
    series = dependency_resolver.get_indicator_data(
        age_inclusion_criteria,
        parameters={"min_age": min_age},
    ).apply(len)

    full_index = pd.date_range(
        start=series.index.min(),
        end=series.index.max(),
        freq=Parameters.candle_time_bucket.to_frequency(),
    )
    return series.reindex(full_index, fill_value=0)


@indicators.define(
    dependencies=(inclusion_criteria,),
    source=IndicatorSource.dependencies_only_universe,
)
def all_criteria_included_pair_count(
    min_tvl: USDollarAmount,
    min_age: float,
    dependency_resolver: IndicatorDependencyResolver,
) -> pd.Series:
    """Count pairs meeting the full inclusion rule set."""
    series = dependency_resolver.get_indicator_data(
        "inclusion_criteria",
        parameters={
            "min_tvl": min_tvl,
            "min_age": min_age,
        },
    )
    return series.apply(len)


@indicators.define(source=IndicatorSource.strategy_universe)
def trading_pair_count(
    strategy_universe: TradingStrategyUniverse,
) -> pd.Series:
    """Get the cumulative number of tradeable non-benchmark pairs."""
    benchmark_pair_ids = {
        strategy_universe.get_pair_by_human_description(desc).internal_id
        for desc in SUPPORTING_PAIRS
    }

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


@indicators.define(
    dependencies=(age,),
    source=IndicatorSource.dependencies_only_per_pair,
)
def age_ramp_weight(
    pair: TradingPairIdentifier,
    dependency_resolver: IndicatorDependencyResolver,
    age_ramp_period: float = 1.0,
) -> pd.Series:
    """Ramp up weight linearly with vault age."""
    vault_age = dependency_resolver.get_indicator_data("age", pair=pair)
    return (vault_age / age_ramp_period).clip(upper=1.0).clip(lower=0.05)


def create_indicators(
    timestamp: datetime.datetime | None,
    parameters: StrategyParameters,
    strategy_universe: TradingStrategyUniverse,
    execution_context: ExecutionContext,
):
    """Create indicators for the strategy."""
    return indicators.create_indicators(
        timestamp=timestamp,
        parameters=parameters,
        strategy_universe=strategy_universe,
        execution_context=execution_context,
    )


#
# Charts
#


def equity_curve_with_benchmark(input: ChartInput) -> list[Figure]:
    """Add ETH as the benchmark token."""
    return equity_curve(
        input,
        benchmark_token_symbols=["ETH"],
    )


def inclusion_criteria_check_with_chain(input: ChartInput) -> pd.DataFrame:
    """Inclusion criteria table with chain shown."""
    return inclusion_criteria_check(
        input,
        show_chain=True,
    )


def trading_pair_breakdown_with_chain(input: ChartInput) -> pd.DataFrame:
    """Show the trading pair breakdown with chain and address columns."""
    return trading_pair_breakdown(
        input,
        show_chain=True,
        show_address=True,
    )


def all_vault_positions_by_profit(input: ChartInput) -> pd.DataFrame:
    """Show the most profitable and least profitable vault positions."""
    return all_vault_positions(
        input,
        sort_by="Profit USD",
        sort_ascending=False,
        show_address=True,
        top_n=10,
        bottom_n=10,
    )


def biggest_daily_gains_losses(input: ChartInput) -> pd.DataFrame:
    """Show the largest positive and negative single-day vault contributions."""
    return all_vault_daily_gains_losses(input, top_n=10, bottom_n=10)


def create_charts(
    timestamp: datetime.datetime | None,
    parameters: StrategyParameters,
    strategy_universe: TradingStrategyUniverse,
    execution_context: ExecutionContext,
) -> ChartRegistry:
    """Define charts we use in backtesting and live trading."""
    charts = ChartRegistry(default_benchmark_pairs=BENCHMARK_PAIRS)
    charts.register(available_trading_pairs, ChartKind.indicator_all_pairs)
    charts.register(inclusion_criteria_check_with_chain, ChartKind.indicator_all_pairs)
    charts.register(equity_curve_with_benchmark, ChartKind.state_all_pairs)
    charts.register(equity_curve_with_drawdown, ChartKind.state_all_pairs)
    charts.register(performance_metrics, ChartKind.state_all_pairs)
    charts.register(equity_curve_by_asset, ChartKind.state_all_pairs)
    charts.register(equity_curve_by_chain, ChartKind.state_all_pairs)
    charts.register(weight_allocation_statistics, ChartKind.state_all_pairs)
    charts.register(positions_at_end, ChartKind.state_all_pairs)
    charts.register(last_messages, ChartKind.state_all_pairs)
    charts.register(alpha_model_diagnostics, ChartKind.state_all_pairs)
    charts.register(trading_pair_breakdown_with_chain, ChartKind.state_all_pairs)
    charts.register(trading_metrics, ChartKind.state_all_pairs)
    charts.register(vault_statistics, ChartKind.state_all_pairs)
    charts.register(all_vault_positions_by_profit, ChartKind.state_single_vault_pair)
    charts.register(biggest_daily_gains_losses, ChartKind.state_single_vault_pair)
    return charts


#
# Metadata
#

tags = {StrategyTag.beta, StrategyTag.deposits_disabled}

name = "Hyperliquid vault of vaults - minimum hold"

short_description = "Cross-chain vault of vaults with age-ramped survivor selection, equal-weight recycling, and minimum 3-day hold"

icon = ""

long_description = """
# Hyperliquid vault of vaults

A cross-chain strategy that selects from Hyperliquid-ecosystem DeFi vaults using
age-ramped survivor selection, equal-weight allocation with capital recycling, and
a minimum 3-day hold rule to prevent rapid position churning.

## Strategy features

- **Dynamic vault universe**: Fetches vaults from the Trading Strategy vault database
- **Survivor-first selection**: Vaults must pass TVL, age, and trading availability criteria
- **Age ramp weighting**: New vaults ramp up linearly over a configurable period
- **Equal-weight recycling**: Accepted survivors are equal-weighted; leftover capital is recycled
- **Minimum hold**: Positions younger than 3 days are protected from forced sells
- **Daily rebalancing**: Adjusts positions based on inclusion criteria changes
- **Risk management**: Caps individual positions at 12% of portfolio and 20% of pool TVL
"""
