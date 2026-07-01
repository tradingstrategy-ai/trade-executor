"""Cross-chain master vault CCTP strategy.

Based on ``01-initial.ipynb`` notebook.

Cross-chain vault allocation across Ethereum, Base, and Arbitrum with CCTP
bridge pairs, age-ramp weighting, daily rebalancing, vol-of-vol veto,
stale-data protection, and redemption-aware target sizing.

Backtest results (2025-01-02 to 2026-03-10)
=============================================

Last backtest run: 2026-06-12

Run with the two-stage async vault settlement simulation: ERC-7540/Ostium
style vaults request on one cycle and settle two days later at the
settlement-time price (``DEFAULT_VAULT_SETTLEMENT_DELAY``). The lower
return versus older instant-settlement runs reflects capital waiting in
settlement queues.

Note: CLI summary inflates return to 51% because bridge positions
(get_value) double-count capital already held in satellite vault
positions. The equity curve uses calculate_total_equity() which
correctly returns get_equity()=0 for bridge positions.

=================================  ===========
Metric                             Strategy
=================================  ===========
Start period                       2025-01-02
End period                         2026-03-10
Trading period length              433 days
Cumulative return                  13.99%
CAGR                               11.70%
Cash at start                      $100,000.00
Value at end                       $113,899.50
Max drawdown                       -0.07%
Longest DD days                    4
Volatility (ann.)                  0.40%
Sharpe                             27.47
Sortino                            110.52
Calmar                             169.99
Total positions                    82
Won positions                      79
Lost positions                     0
=================================  ===========
"""

#
# Imports
#

import datetime
import logging

import numpy as np
import pandas as pd
from eth_defi.token import USDC_NATIVE_TOKEN
from plotly.graph_objects import Figure
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.utils.token_filter import filter_for_selected_pairs

from tradeexecutor.curator import is_quarantined
from tradeexecutor.ethereum.vault.checks import check_stale_vault_data
from tradeexecutor.exchange_account.allocation import (
    calculate_portfolio_target_value,
    get_redeemable_portfolio_capital,
)
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.types import USDollarAmount
from tradeexecutor.strategy.alpha_model import AlphaModel
from tradeexecutor.strategy.chart.definition import ChartInput, ChartKind, ChartRegistry
from tradeexecutor.strategy.chart.standard.alpha_model import (
    alpha_model_diagnostics,
    missed_vault_deposit_redemption_events,
    missed_vault_deposit_redemption_timeline,
)
from tradeexecutor.strategy.chart.standard.cycle_snapshot import latest_cycle_snapshot
from tradeexecutor.strategy.chart.standard.equity_curve import (
    equity_curve as equity_curve_chart,
    equity_curve_with_drawdown,
)
from tradeexecutor.strategy.chart.standard.interest import vault_statistics
from tradeexecutor.strategy.chart.standard.performance_metrics import performance_metrics
from tradeexecutor.strategy.chart.standard.position import positions_at_end
from tradeexecutor.strategy.chart.standard.profit_breakdown import trading_pair_breakdown
from tradeexecutor.strategy.chart.standard.thinking import last_messages
from tradeexecutor.strategy.chart.standard.trading_metrics import trading_metrics
from tradeexecutor.strategy.chart.standard.trading_universe import (
    available_trading_pairs,
    inclusion_criteria_check,
)
from tradeexecutor.strategy.chart.standard.vault import all_vault_positions, pending_vault_settlements
from tradeexecutor.strategy.chart.standard.weight import (
    equity_curve_by_asset,
    equity_curve_by_chain,
    weight_allocation_statistics,
)
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.pandas_trader.indicator import (
    IndicatorDependencyResolver,
    IndicatorSource,
)
from tradeexecutor.strategy.pandas_trader.indicator_decorator import IndicatorRegistry
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.pandas_trader.trading_universe_input import CreateTradingUniverseInput
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.tag import StrategyTag
from tradeexecutor.strategy.trading_strategy_universe import (
    TradingStrategyUniverse,
    load_partial_data,
    load_vault_universe_with_metadata,
)
from tradeexecutor.strategy.tvl_size_risk import USDTVLSizeRiskModel
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradeexecutor.strategy.weighting import weight_passthrouh
from tradeexecutor.utils.dedent import dedent_any

logger = logging.getLogger(__name__)

#
# Trading universe constants
#

trading_strategy_engine_version = "0.5"

LENDING_RESERVES = None

SUPPORTING_PAIRS = [
    (ChainId.ethereum, "uniswap-v3", "WETH", "USDC", 0.0005),
    (ChainId.ethereum, "uniswap-v3", "WBTC", "USDC", 0.003),
    (ChainId.arbitrum, "uniswap-v3", "WETH", "USDC", 0.0005),
    (ChainId.base, "uniswap-v3", "WETH", "USDC", 0.0005),
]

SOURCE_VAULTS = [
    # Ethereum native-USDC vaults
    (ChainId.ethereum, "0x09c4c7b1d2e9aa7506db8b76f1dbbd61c08c114b"),
    (ChainId.ethereum, "0x438982ea288763370946625fd76c2508ee1fb229"),
    (ChainId.ethereum, "0x8df3deba711ae4a9af16cbca5e4fbb1402f036d5"),
    (ChainId.ethereum, "0xca790385506b790554571cbc9da73f0130cdcfd5"),
    (ChainId.ethereum, "0xe9d33286f0e37f517b1204aa6da085564414996d"),
    # Base native-USDC vaults
    (ChainId.base, "0xf7e26fa48a568b8b0038e104dfd8abdf0f99074f"),
    (ChainId.base, "0x3094b241aade60f91f1c82b0628a10d9501462f9"),
    (ChainId.base, "0x70fffbacb53ef74903ac074aae769414a70970d1"),
    (ChainId.base, "0x3ec4a293fb906dd2cd440c20decb250def141df1"),
    (ChainId.base, "0x8092ca384d44260ea4feaf7457b629b8dc6f88f0"),
    (ChainId.base, "0xc777031d50f632083be7080e51e390709062263e"),
    (ChainId.base, "0xad20523a7dc37babc1cc74897e4977232b3d02e5"),
    (ChainId.base, "0xbc10718571fcb3c3f67800e7c0887e450d2ff398"),
    (ChainId.base, "0xefe32813dba3a783059d50e5358b9e3661218dad"),
    (ChainId.base, "0xd5c22fa3f7ee979ed7c28e36669b29797ab277e4"),
    # Arbitrum native-USDC vaults
    (ChainId.arbitrum, "0x75288264fdfea8ce68e6d852696ab1ce2f3e5004"),
    (ChainId.arbitrum, "0x58bfc95a864e18e8f3041d2fcd3418f48393fe6a"),
    (ChainId.arbitrum, "0xf63b7f49b4f5dc5d0e7e583cfd79dc64e646320c"),
    (ChainId.arbitrum, "0x1723cb57af58efb35a013870c90fcc3d60174a4e"),
    (ChainId.arbitrum, "0x20d419a8e12c45f88fda7c5760bb6923cee27f98"),
    (ChainId.arbitrum, "0xd3443ee1e91af28e5fb858fbd0d72a63ba8046e0"),
    (ChainId.arbitrum, "0x0df2e3a0b5997adc69f8768e495fd98a4d00f134"),
]

BENCHMARK_PAIRS = SUPPORTING_PAIRS

#
# Strategy parameters
#


class Parameters:

    id = "xchain-master-vault"

    candle_time_bucket = TimeBucket.d1
    cycle_duration = CycleDuration.cycle_1d
    chain_id = ChainId.cross_chain
    primary_chain_id = ChainId.arbitrum
    supporting_pairs = SUPPORTING_PAIRS
    source_vaults = SOURCE_VAULTS
    preferred_stablecoin = USDC_NATIVE_TOKEN[ChainId.ethereum].lower()
    auto_generate_cctp_bridges = True

    backtest_start = datetime.datetime(2025, 1, 1)
    backtest_end = datetime.datetime(2026, 3, 11)
    initial_cash = 100_000

    max_assets_in_portfolio = 12
    allocation_pct = 0.98
    max_concentration_pct = 0.12
    per_position_cap_of_pool_pct = 0.2
    min_portfolio_weight_pct = 0.005

    absolute_min_vault_deposit_usd = 5.0
    individual_rebalance_min_threshold_usd = 50.0
    sell_rebalance_min_threshold_usd = 10.0

    #: Same-cycle cash headroom withheld when async redemptions force this cycle's
    #: buys to be funded from cash + synchronous sell proceeds only. The sync sell
    #: proceeds are mark-to-market; execution realises slightly less (fees, price,
    #: raw-unit rounding, CCTP bridge floors), so this buffer keeps buys below the
    #: realisable same-cycle cash and prevents a cross-chain plan coming up a few
    #: dollars short (NotEnoughMoney from the CCTP planner). Tunable per portfolio.
    cross_chain_cash_buffer_usd = 1_000.0

    min_tvl_usd = 7_500
    min_age = 0.075
    weight_signal = "age_ramp"
    age_ramp_period = 0.75

    vol_window = 60
    volvol_veto_percentile = 0.75

    required_history_period = datetime.timedelta(days=365)
    routing = TradeRouting.default
    slippage_tolerance_pct = 0.0060
    assummed_liquidity_when_data_missings_usd = 0.01


#
# Universe creation
#


def _get_available_supporting_pair_ids(
    strategy_universe: TradingStrategyUniverse,
) -> set[int]:
    """Return supporting pair ids that are actually present in the universe."""
    pair_ids = set()
    for desc in SUPPORTING_PAIRS:
        try:
            pair_ids.add(strategy_universe.get_pair_by_human_description(desc).internal_id)
        except KeyError:
            continue
    return pair_ids


def create_trading_universe(
    input: CreateTradingUniverseInput,
) -> TradingStrategyUniverse:
    """Create the cross-chain trading universe."""
    execution_context = input.execution_context
    client = input.client
    timestamp = input.timestamp
    params = input.parameters or Parameters
    universe_options = input.universe_options

    debug_printer = logger.info if execution_context.live_trading else print
    chain_id = params.primary_chain_id

    supporting_pairs = [] if execution_context.live_trading else params.supporting_pairs

    debug_printer(f"Preparing trading universe on chain {chain_id.get_name()}")

    all_pairs_df = client.fetch_pair_universe().to_pandas()
    pairs_df = filter_for_selected_pairs(all_pairs_df, supporting_pairs)
    debug_printer(f"We have total {len(all_pairs_df)} pairs in dataset and going to use {len(pairs_df)} pairs for the strategy")

    vault_universe = load_vault_universe_with_metadata(client, vaults=params.source_vaults)
    if params.auto_generate_cctp_bridges:
        vault_universe = vault_universe.limit_to_native_usdc()
    debug_printer(
        f"Loaded {vault_universe.get_vault_count()} vaults from remote vault metadata, "
        f"source vaults count: {len(params.source_vaults)}"
    )

    dataset = load_partial_data(
        client=client,
        time_bucket=params.candle_time_bucket,
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

    return TradingStrategyUniverse.create_from_dataset(
        dataset,
        reserve_asset=params.preferred_stablecoin,
        forward_fill=True,
        forward_fill_until=timestamp,
        primary_chain=params.primary_chain_id,
        auto_generate_cctp_bridges=params.auto_generate_cctp_bridges,
    )


#
# Strategy logic
#


def decide_trades(input: StrategyInput) -> list[TradeExecution]:
    """For each strategy tick, generate the list of trades."""
    parameters = input.parameters
    position_manager = input.get_position_manager()
    state = input.state
    timestamp = input.timestamp
    indicators = input.indicators
    strategy_universe = input.strategy_universe

    check_stale_vault_data(strategy_universe, timestamp, input.execution_context.mode)

    portfolio = position_manager.get_current_portfolio()
    equity = portfolio.get_total_equity()

    if input.execution_context.mode == ExecutionMode.backtesting and equity < parameters.initial_cash * 0.10:
        return []

    alpha_model = AlphaModel(timestamp, close_position_weight_epsilon=parameters.min_portfolio_weight_pct)

    tvl_included_pair_count = indicators.get_indicator_value("tvl_included_pair_count")
    included_pairs = indicators.get_indicator_value("inclusion_criteria", na_conversion=False)
    if included_pairs is None:
        included_pairs = []

    volvol_values = {}
    for pair_id in included_pairs:
        pair = strategy_universe.get_pair_by_id(pair_id)
        vol_of_vol_value = indicators.get_indicator_value("vol_of_vol", pair=pair)
        if vol_of_vol_value is not None and not pd.isna(vol_of_vol_value):
            volvol_values[pair_id] = vol_of_vol_value

    volvol_threshold = np.percentile(list(volvol_values.values()), parameters.volvol_veto_percentile * 100) if volvol_values else float("inf")

    vetoed_count = 0
    signal_count = 0
    for pair_id in included_pairs:
        pair = strategy_universe.get_pair_by_id(pair_id)
        if not state.is_good_pair(pair):
            continue
        if is_quarantined(pair.pool_address, timestamp):
            continue
        vol_of_vol_value = volvol_values.get(pair_id)
        if vol_of_vol_value is not None and vol_of_vol_value > volvol_threshold:
            vetoed_count += 1
            continue
        age_ramp_weight_value = indicators.get_indicator_value("age_ramp_weight", pair=pair)
        weight_signal_value = age_ramp_weight_value if age_ramp_weight_value is not None else 1.0
        alpha_model.set_signal(pair, weight_signal_value)
        signal_count += 1

    locked_position_value = alpha_model.carry_forward_non_redeemable_positions(position_manager)
    redeemable_capital = get_redeemable_portfolio_capital(position_manager)
    portfolio_target_value = calculate_portfolio_target_value(position_manager, parameters.allocation_pct)
    deployable_target_value = max(portfolio_target_value - locked_position_value, 0.0)

    alpha_model.select_top_signals(count=parameters.max_assets_in_portfolio)
    alpha_model.assign_weights(method=weight_passthrouh)

    size_risk_model = USDTVLSizeRiskModel(
        pricing_model=input.pricing_model,
        per_position_cap=parameters.per_position_cap_of_pool_pct,
    )

    alpha_model.normalise_weights(
        investable_equity=deployable_target_value,
        size_risk_model=size_risk_model,
        max_weight=parameters.max_concentration_pct,
        max_positions=parameters.max_assets_in_portfolio,
        waterfall=True,
    )
    alpha_model.update_old_weights(state.portfolio, ignore_credit=False)
    alpha_model.calculate_target_positions(position_manager)

    trades = alpha_model.generate_rebalance_trades_and_triggers(
        position_manager,
        min_trade_threshold=parameters.individual_rebalance_min_threshold_usd,
        individual_rebalance_min_threshold=parameters.individual_rebalance_min_threshold_usd,
        sell_rebalance_min_threshold=parameters.sell_rebalance_min_threshold_usd,
        execution_context=input.execution_context,
        same_cycle_cash_buffer_usd=parameters.cross_chain_cash_buffer_usd,
    )
    missed_vault_events = alpha_model.get_missed_vault_events()
    if missed_vault_events:
        state.visualisation.add_calculations(timestamp, {"missed_vault_events": missed_vault_events})

    if input.is_visualisation_enabled():
        try:
            top_signal = next(iter(alpha_model.get_signals_sorted_by_weight()))
            if top_signal.normalised_weight == 0:
                top_signal = None
        except StopIteration:
            top_signal = None

        rebalance_volume = sum(trade.get_value() for trade in trades)
        report = dedent_any(f"""
            Cycle: #{input.cycle}
            Rebalanced: {'yes' if alpha_model.is_rebalance_triggered() else 'no'}
            Open/about to open positions: {len(state.portfolio.open_positions)}
            Trades decided: {len(trades)}
            Pairs meeting inclusion criteria: {len(included_pairs)}
            Vol-of-vol threshold: {volvol_threshold:.6f}
            Vaults vetoed by vol-of-vol: {vetoed_count}
            Candidate signals: {signal_count}
            Total equity: {portfolio.get_total_equity():,.2f} USD
            Cash: {position_manager.get_current_cash():,.2f} USD
            Redeemable capital: {redeemable_capital:,.2f} USD
            Locked capital carried forward: {locked_position_value:,.2f} USD
            Pending redemptions: {position_manager.get_pending_redemptions():,.2f} USD
            Investable equity: {alpha_model.investable_equity:,.2f} USD
            Accepted investable equity: {alpha_model.accepted_investable_equity:,.2f} USD
            Allocated to signals: {alpha_model.get_allocated_value():,.2f} USD
            Discarded allocation because of lack of lit liquidity: {alpha_model.size_risk_discarded_value:,.2f} USD
            Rebalance volume: {rebalance_volume:,.2f} USD
        """)

        if top_signal:
            report += dedent_any(f"""
                Top signal: {top_signal.pair.get_ticker()} weight={top_signal.normalised_weight * 100:.2f}%
            """)

        state.visualisation.add_message(timestamp, report)
        state.visualisation.set_discardable_data("alpha_model", alpha_model)

    return trades


#
# Indicators
#

indicators = IndicatorRegistry()


@indicators.define(source=IndicatorSource.tvl)
def tvl(close: pd.Series) -> pd.Series:
    """TVL series for a pair."""
    return close


@indicators.define()
def age(close: pd.Series) -> pd.Series:
    """Age of a vault in years since first candle."""
    inception = close.index[0]
    age_years = (close.index - inception) / pd.Timedelta(days=365.25)
    return pd.Series(age_years, index=close.index)


@indicators.define(dependencies=(tvl,), source=IndicatorSource.dependencies_only_universe)
def tvl_inclusion_criteria(
    min_tvl_usd: USDollarAmount,
    dependency_resolver: IndicatorDependencyResolver,
) -> pd.Series:
    """The pair must have min TVL to be included."""
    series = dependency_resolver.get_indicator_data_pairs_combined(tvl)
    mask = series >= min_tvl_usd
    mask_true_values_only = mask[mask]
    return mask_true_values_only.groupby(level="timestamp").apply(
        lambda x: x.index.get_level_values("pair_id").tolist()
    )


@indicators.define(dependencies=(age,), source=IndicatorSource.dependencies_only_universe)
def age_inclusion_criteria(
    min_age: float,
    dependency_resolver: IndicatorDependencyResolver,
) -> pd.Series:
    """The pair must be at least min_age years old."""
    series = dependency_resolver.get_indicator_data_pairs_combined(age)
    mask = series >= min_age
    mask_true_values_only = mask[mask]
    return mask_true_values_only.groupby(level="timestamp").apply(
        lambda x: x.index.get_level_values("pair_id").tolist()
    )


@indicators.define(source=IndicatorSource.strategy_universe)
def trading_availability_criteria(
    strategy_universe: TradingStrategyUniverse,
) -> pd.Series:
    """Is pair tradeable at each timestamp."""
    candle_series = strategy_universe.data_universe.candles.df["open"]
    return candle_series.groupby(level="timestamp").apply(
        lambda x: x.index.get_level_values("pair_id").tolist()
    )


@indicators.define(
    dependencies=[tvl_inclusion_criteria, trading_availability_criteria, age_inclusion_criteria],
    source=IndicatorSource.strategy_universe,
)
def inclusion_criteria(
    strategy_universe: TradingStrategyUniverse,
    min_tvl_usd: USDollarAmount,
    min_age: float,
    dependency_resolver: IndicatorDependencyResolver,
) -> pd.Series:
    """Pairs meeting all of our inclusion criteria."""
    benchmark_pair_ids = _get_available_supporting_pair_ids(strategy_universe)
    tvl_series = dependency_resolver.get_indicator_data(tvl_inclusion_criteria, parameters={"min_tvl_usd": min_tvl_usd})
    trading_availability_series = dependency_resolver.get_indicator_data(trading_availability_criteria)
    age_series = dependency_resolver.get_indicator_data(age_inclusion_criteria, parameters={"min_age": min_age})

    df = pd.DataFrame({"tvl_pair_ids": tvl_series, "trading_availability_pair_ids": trading_availability_series, "age_pair_ids": age_series})
    df = df.fillna("").apply(list)

    def _combine(row):
        return set(row["tvl_pair_ids"]) & set(row["trading_availability_pair_ids"]) & set(row["age_pair_ids"]) - benchmark_pair_ids

    union_criteria = df.apply(_combine, axis=1)
    full_index = pd.date_range(start=union_criteria.index.min(), end=union_criteria.index.max(), freq=Parameters.candle_time_bucket.to_frequency())
    return union_criteria.reindex(full_index, fill_value=[])


@indicators.define(dependencies=(age,), source=IndicatorSource.dependencies_only_per_pair)
def age_ramp_weight(
    pair: TradingPairIdentifier,
    dependency_resolver: IndicatorDependencyResolver,
    age_ramp_period: float = 1.0,
) -> pd.Series:
    """Younger vaults receive lower weights, ramping up over age_ramp_period years."""
    vault_age = dependency_resolver.get_indicator_data("age", pair=pair)
    return (vault_age / age_ramp_period).clip(upper=1.0).clip(lower=0.05)


@indicators.define()
def realised_vol(
    close: pd.Series,
    vol_window: int = 60,
) -> pd.Series:
    """Rolling annualised standard deviation of daily returns."""
    daily_returns = close.pct_change().fillna(0)
    rolling_std = daily_returns.rolling(window=vol_window, min_periods=7).std()
    return rolling_std * (365 ** 0.5)


@indicators.define(
    dependencies=(realised_vol,),
    source=IndicatorSource.dependencies_only_per_pair,
)
def vol_of_vol(
    pair: TradingPairIdentifier,
    dependency_resolver: IndicatorDependencyResolver,
    vol_window: int = 60,
) -> pd.Series:
    """Rolling standard deviation of rolling volatility."""
    realised_vol_series = dependency_resolver.get_indicator_data(
        "realised_vol",
        pair=pair,
        parameters={"vol_window": vol_window},
    )
    return realised_vol_series.rolling(window=vol_window, min_periods=7).std()


@indicators.define(dependencies=(inclusion_criteria,), source=IndicatorSource.dependencies_only_universe)
def all_criteria_included_pair_count(
    min_tvl_usd: USDollarAmount,
    min_age: float,
    dependency_resolver: IndicatorDependencyResolver,
) -> pd.Series:
    """Count of pairs meeting all inclusion criteria per timestamp."""
    series = dependency_resolver.get_indicator_data("inclusion_criteria", parameters={"min_tvl_usd": min_tvl_usd, "min_age": min_age})
    return series.apply(len)


@indicators.define(dependencies=(tvl_inclusion_criteria,), source=IndicatorSource.dependencies_only_universe)
def tvl_included_pair_count(
    min_tvl_usd: USDollarAmount,
    dependency_resolver: IndicatorDependencyResolver,
) -> pd.Series:
    """Count of pairs meeting TVL inclusion criteria per timestamp."""
    series = dependency_resolver.get_indicator_data("tvl_inclusion_criteria", parameters={"min_tvl_usd": min_tvl_usd})
    return series.apply(len)


@indicators.define(dependencies=(age_inclusion_criteria,), source=IndicatorSource.dependencies_only_universe)
def age_included_pair_count(
    min_age: float,
    dependency_resolver: IndicatorDependencyResolver,
) -> pd.Series:
    """Count of pairs meeting age inclusion criteria per timestamp."""
    series = dependency_resolver.get_indicator_data("age_inclusion_criteria", parameters={"min_age": min_age})
    return series.apply(len)


@indicators.define(source=IndicatorSource.strategy_universe)
def trading_pair_count(
    strategy_universe: TradingStrategyUniverse,
) -> pd.Series:
    """Get number of pairs that trade at each timestamp."""
    benchmark_pair_ids = _get_available_supporting_pair_ids(strategy_universe)
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
    """Equity curve with ETH benchmark."""
    return equity_curve_chart(input, benchmark_token_symbols=["ETH"])


def inclusion_criteria_check_with_chain(input: ChartInput) -> pd.DataFrame:
    """Inclusion criteria table with chain shown."""
    return inclusion_criteria_check(input, show_chain=True)


def trading_pair_breakdown_with_chain(input: ChartInput) -> pd.DataFrame:
    """Trading pair breakdown with chain and address."""
    return trading_pair_breakdown(input, show_chain=True, show_address=True)


def all_vault_positions_by_profit(input: ChartInput) -> pd.DataFrame:
    """Vault positions sorted by profit."""
    return all_vault_positions(input, sort_by="Profit USD", sort_ascending=False, show_address=True)


def create_charts(
    timestamp: datetime.datetime | None,
    parameters: StrategyParameters,
    strategy_universe: TradingStrategyUniverse,
    execution_context: ExecutionContext,
) -> ChartRegistry:
    """Define charts we use in backtesting and live trading."""
    default_benchmark_pairs = [] if execution_context.live_trading else BENCHMARK_PAIRS
    charts = ChartRegistry(default_benchmark_pairs=default_benchmark_pairs)
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
    charts.register(missed_vault_deposit_redemption_events, ChartKind.state_all_pairs)
    charts.register(missed_vault_deposit_redemption_timeline, ChartKind.state_all_pairs)
    charts.register(pending_vault_settlements, ChartKind.state_all_pairs)
    charts.register(trading_pair_breakdown_with_chain, ChartKind.state_all_pairs)
    charts.register(trading_metrics, ChartKind.state_all_pairs)
    charts.register(vault_statistics, ChartKind.state_all_pairs)
    charts.register(all_vault_positions_by_profit, ChartKind.state_all_pairs)
    charts.register(latest_cycle_snapshot, ChartKind.state_all_pairs)
    return charts


#
# Metadata
#

tags = {StrategyTag.beta, StrategyTag.deposits_disabled}

name = "Xchain master vault strategy"

short_description = "Cross-chain vault allocation strategy with age-ramp weighting, vol-of-vol veto, and CCTP bridges"

icon = ""

long_description = """
# Cross-chain master vault strategy

A diversified yield strategy that allocates across native-USDC vaults on Ethereum, Base, and Arbitrum
using CCTP bridge pairs for cross-chain capital movement.

## Strategy features

- **Cross-chain allocation**: Invests across Ethereum, Base, and Arbitrum via CCTP bridges
- **Age-ramp weighting**: Younger vaults receive lower weights, ramping up over 0.75 years
- **Vol-of-vol veto**: Excludes vaults whose realised volatility is too unstable versus peers
- **Daily rebalancing**: Adjusts positions every day using survivor-first capped waterfall sizing
- **Stale-data guard**: Refuses to rebalance on forward-filled-but-stale vault data
- **Redemption-aware**: Locked vault capital is carried forward and excluded from fresh allocation

## Risk parameters

- Maximum 12 positions at any time
- 98% allocation target
- 12% maximum concentration per asset
- 20% per-position cap of pool TVL
- 5 USD minimum vault deposit floor
"""
