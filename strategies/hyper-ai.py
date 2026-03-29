"""Hyperliquid vault of vaults strategy.

Based on ``160-backtest-hyperliquid-low-value-treasury-test.ipynb`` notebook.

Survivor-first capped waterfall allocation across Hyperliquid native vaults,
with age-ramp weighting, daily rebalancing, and redemption-aware target sizing.

Backtest results (2025-08-02 to 2026-03-10)
=============================================

Last backtest run: 2026-03-21

================================  =========  ======  ======
Metric                            Strategy   BTC     ETH
================================  =========  ======  ======
Start period                      2025-08-02 2025-08-02 2025-08-02
End period                        2026-03-10 2026-03-10 2026-03-10
Risk-free rate                    0.0%       0.0%    0.0%
Time in market                    37.0%      99.0%   98.0%
Cumulative return                 126.74%    -38.86% -43.6%
CAGR                              288.93%    -55.79% -61.34%
Sharpe                            2.02       -1.49   -0.93
Prob. Sharpe ratio                96.77%     12.27%  23.72%
Sortino                           4.28       -1.96   -1.29
Max drawdown                      -24.86%    -49.82% -62.29%
Longest DD days                   70         155     200
Volatility (ann.)                 82.28%     47.06%  73.2%
Calmar                            11.62      -1.12   -0.98
================================  =========  ======  ======
"""

#
# Imports
#

import datetime
import logging
import re

import pandas as pd
from eth_defi.token import USDC_NATIVE_TOKEN
from plotly.graph_objects import Figure
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.utils.token_filter import filter_for_selected_pairs

from tradeexecutor.curator import (build_hyperliquid_vault_universe,
                                   is_quarantined)
from tradeexecutor.ethereum.vault.checks import check_stale_vault_data
from tradeexecutor.exchange_account.allocation import (
    calculate_portfolio_target_value, get_redeemable_portfolio_capital)
from tradeexecutor.state.identifier import (AssetIdentifier,
                                            TradingPairIdentifier)
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.types import USDollarAmount
from tradeexecutor.strategy.alpha_model import AlphaModel
from tradeexecutor.strategy.chart.definition import (ChartInput, ChartKind,
                                                     ChartRegistry)
from tradeexecutor.strategy.chart.standard.alpha_model import \
    alpha_model_diagnostics
from tradeexecutor.strategy.chart.standard.equity_curve import \
    equity_curve as equity_curve_chart
from tradeexecutor.strategy.chart.standard.equity_curve import \
    equity_curve_with_drawdown
from tradeexecutor.strategy.chart.standard.interest import vault_statistics
from tradeexecutor.strategy.chart.standard.performance_metrics import \
    performance_metrics
from tradeexecutor.strategy.chart.standard.position import positions_at_end
from tradeexecutor.strategy.chart.standard.profit_breakdown import \
    trading_pair_breakdown
from tradeexecutor.strategy.chart.standard.thinking import last_messages
from tradeexecutor.strategy.chart.standard.trading_metrics import \
    trading_metrics
from tradeexecutor.strategy.chart.standard.trading_universe import (
    available_trading_pairs, inclusion_criteria_check)
from tradeexecutor.strategy.chart.standard.vault import all_vault_positions
from tradeexecutor.strategy.chart.standard.weight import (
    equity_curve_by_asset, equity_curve_by_chain, weight_allocation_statistics)
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import (ExecutionContext,
                                                      ExecutionMode)
from tradeexecutor.strategy.pandas_trader.indicator import (
    IndicatorDependencyResolver, IndicatorSource)
from tradeexecutor.strategy.pandas_trader.indicator_decorator import \
    IndicatorRegistry
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.pandas_trader.trading_universe_input import \
    CreateTradingUniverseInput
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.tag import StrategyTag
from tradeexecutor.strategy.trading_strategy_universe import (
    TradingStrategyUniverse, load_partial_data,
    load_vault_universe_with_metadata)
from tradeexecutor.strategy.tvl_size_risk import USDTVLSizeRiskModel
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradeexecutor.strategy.weighting import weight_passthrouh
from tradeexecutor.utils.dedent import dedent_any

logger = logging.getLogger(__name__)

#
# Trading universe constants
#

trading_strategy_engine_version = "0.5"

CHAIN_ID = ChainId.hyperliquid

EXCHANGES = ("uniswap-v2", "uniswap-v3")

SUPPORTING_PAIRS = [
    (ChainId.arbitrum, "uniswap-v3", "WETH", "USDC", 0.0005),
    (ChainId.ethereum, "uniswap-v3", "WETH", "USDC", 0.0005),
    (ChainId.ethereum, "uniswap-v3", "WBTC", "USDC", 0.003),
]

LENDING_RESERVES = None

PREFERRED_STABLECOIN = AssetIdentifier(
    chain_id=ChainId.hyperliquid.value,
    address=USDC_NATIVE_TOKEN[ChainId.hyperliquid].lower(),
    token_symbol="USDC",
    decimals=6,
)

ALLOWED_VAULT_DENOMINATION_TOKENS = {"USDC", "USDT", "USDC.e", "crvUSD", "USDT0", "USD₮0", "USDt", "USDS"}

BENCHMARK_PAIRS = SUPPORTING_PAIRS

#
# Strategy parameters
#


class Parameters:

    id = "hyper-ai"

    #: Daily candles match the whole Hyperliquid survivor-first research chain.
    candle_time_bucket = TimeBucket.d1
    #: Daily rebalance cadence is the validated schedule used in NB143 and NB154-NB157.
    cycle_duration = CycleDuration.cycle_1d
    #: HyperEVM is the primary chain for vaults.
    chain_id = CHAIN_ID
    #: HyperEVM is the primary execution chain for the survivor-first release branch.
    primary_chain_id = CHAIN_ID
    #: Keep the same exchange set as NB143, NB153, and the live-style test strategy.
    exchanges = EXCHANGES

    #: NB141-NB157 validated the survivor-first allocator family with a 20-vault basket.
    #: Keep this fixed so NB160 is directly comparable to NB158, NB143, and the NB154-NB157 robustness reruns.
    max_assets_in_portfolio = 20
    #: NB143 and NB154-NB157 used 98% target deployment and the corrected waterfall branch held up there.
    #: NB156 showed lower allocation is viable, but not better enough to replace the release default.
    allocation_pct = 0.98
    #: The 12% cap is the survivor-first concentration limit carried through NB143-NB157.
    #: Keep it unchanged because the corrected waterfall case was validated with this exact cap.
    max_concentration_pct = 0.12
    #: NB156 showed tighter per-pool caps reduce both return and deployment.
    #: Keep the 20% pool-cap ceiling from NB143 so we do not reintroduce unnecessary cash drag.
    per_position_cap_of_pool_pct = 0.2
    #: Engine hygiene threshold used across the survivor-first notebooks.
    #: Keep it because the alpha model uses it when cleaning up tiny residual positions.
    min_portfolio_weight_pct = 0.005

    #: Hyperliquid has a hard 5 USD minimum deposit.
    absolute_min_vault_deposit_usd = 5.0
    #: Preserve the old 50 USD buy threshold at 100k initial cash.
    individual_rebalance_min_threshold_of_initial_cash_pct = 0.0005
    #: Preserve the old 10 USD sell threshold at 100k initial cash.
    sell_rebalance_min_threshold_of_initial_cash_pct = 0.0001

    #: NB141 selected this wider survivor-first TVL floor and NB143-NB157 validated the allocator on it.
    #: This is intentionally the survivor-first release setting, not the older NB124/NB126 production threshold.
    min_tvl_usd = 10_000
    #: NB141 also selected this young-vault-inclusive age floor for the survivor-first branch.
    #: Keep it fixed so NB158 stays comparable with the corrected waterfall validation chain.
    min_age = 0.075
    #: The corrected reruns kept `age_ramp` as the surviving signal family and NB154-NB157 validated waterfall on top of it.
    weight_signal = "age_ramp"
    #: NB143 and the corrected robustness notebooks all used a 0.75-year ramp.
    #: Keep the signal definition unchanged so NB160 isolates the treasury-size change, not signal drift.
    age_ramp_period = 0.75

    #: August 2025 remains the canonical mature-universe start used by the release branch.
    backtest_start = datetime.datetime(2025, 8, 1)
    #: Keep the same end date as NB143, NB153, and NB154-NB157 for an apples-to-apples comparison.
    backtest_end = datetime.datetime(2026, 3, 11)
    #: Low-value treasury bankroll for small-cap forward-testing validation.
    initial_cash = 1000
    #: Derived at class creation time from the configured initial cash and the 5 USD hard floor.
    individual_rebalance_min_threshold_usd = max(
        absolute_min_vault_deposit_usd,
        initial_cash * individual_rebalance_min_threshold_of_initial_cash_pct,
    )
    #: Derived at class creation time from the configured initial cash and the 5 USD hard floor.
    sell_rebalance_min_threshold_usd = max(
        absolute_min_vault_deposit_usd,
        initial_cash * sell_rebalance_min_threshold_of_initial_cash_pct,
    )

    #: Default routing is still required by the strategy runtime even though it is not the alpha source.
    routing = TradeRouting.default
    #: Set deliberately high so live and notebook indicator calculations use effectively all available history.
    #: This is an operational correction to the old 120-day default because `age()` and other history-derived indicators
    #: depend on the first available data point and would be silently biased by a truncated lookback window.
    required_history_period = datetime.timedelta(days=365 * 20)
    #: Keep the same live-style slippage assumption as NB153 and `hyper-ai-test.py`.
    slippage_tolerance_pct = 0.0060
    #: Assume no liquidity if there is a gap in TVL data.
    assummed_liquidity_when_data_missings_usd = 0.01


#
# Universe creation
#


def create_trading_universe(
    input: CreateTradingUniverseInput,
) -> TradingStrategyUniverse:
    """Create the trading universe.

    Keep the backtest trading window fixed to ``Parameters.backtest_start`` /
    ``Parameters.backtest_end``, but let ``required_history_period`` extend the
    data-loading window backwards so age and other history-derived indicators
    can see the full pre-backtest history for the selected vaults.
    """
    execution_context = input.execution_context
    client = input.client
    timestamp = input.timestamp
    parameters = input.parameters or Parameters
    universe_options = input.universe_options

    debug_printer = logger.info if execution_context.live_trading else print

    chain_id = parameters.primary_chain_id

    # Supporting benchmark pairs live on Uniswap on Ethereum and Arbitrum.
    # We only need them for backtest benchmarking and research visualisations.
    # In live-style runs such as trade execution, one-off diagnostics, and
    # Lagoon deployment, we do not trade these pairs at all.
    # Loading them anyway makes the universe look multichain to later routing
    # and deployment code, even though the executable strategy is Hyperliquid
    # vault-only on HyperEVM.
    # That false multichain signal is what caused Lagoon deployment to demand
    # an Ethereum Web3 connection for a Hyperliquid-only deployment.
    if execution_context.live_trading:
        supporting_pairs = []
    else:
        supporting_pairs = SUPPORTING_PAIRS

    debug_printer(f"Preparing trading universe on chain {chain_id.get_name()}")

    all_pairs_df = client.fetch_pair_universe().to_pandas()
    # Filter the benchmark pairs only when we are in backtesting / research.
    # In live paths this intentionally becomes an empty frame, because the
    # strategy obtains its real tradeable instruments from the vault universe
    # loaded below, not from spot benchmark pairs.
    pairs_df = filter_for_selected_pairs(all_pairs_df, supporting_pairs)
    debug_printer(f"We have total {len(all_pairs_df)} pairs in dataset and going to use {len(pairs_df)} pairs for the strategy")

    source_vaults = build_hyperliquid_vault_universe(
        min_tvl=parameters.min_tvl_usd,
        min_age=0.0,
    )
    vault_universe = load_vault_universe_with_metadata(client, vaults=source_vaults)
    vault_universe = vault_universe.limit_to_denomination(ALLOWED_VAULT_DENOMINATION_TOKENS, check_all_vaults_found=True)
    debug_printer(f"Loaded {vault_universe.get_vault_count()} vaults from remote vault metadata, source vaults count: {len(source_vaults)}")

    # `load_partial_data()` now honours `required_history_period` in backtests
    # as a loader-window extension instead of clipping history to the trading window.
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

    return TradingStrategyUniverse.create_from_dataset(
        dataset,
        reserve_asset=PREFERRED_STABLECOIN,
        forward_fill=True,
        forward_fill_until=timestamp,
        primary_chain=parameters.primary_chain_id,
    )


def _get_available_supporting_pair_ids(
    strategy_universe: TradingStrategyUniverse,
) -> set[int]:
    """Return supporting pair ids that are actually present in the universe."""
    pair_ids = set()

    # Live universes intentionally skip SUPPORTING_PAIRS above.
    # Because of that, any later benchmark lookup must tolerate the pairs being
    # absent instead of crashing.
    # We resolve only the pairs that are really present, so both backtest and
    # live code paths can share the same indicator helpers safely.
    #
    # get_pair_by_human_description() raises:
    # - KeyError when the pair itself is missing from the universe
    # - RuntimeError when the exchange (e.g. uniswap-v3) is not in the universe at all
    for desc in SUPPORTING_PAIRS:
        try:
            pair_ids.add(strategy_universe.get_pair_by_human_description(desc).internal_id)
        except (KeyError, RuntimeError):
            continue
    return pair_ids


#
# Strategy logic
#


def decide_trades(input: StrategyInput) -> list[TradeExecution]:
    """Run survivor-first capped waterfall sizing with a redemption-aware target value."""
    parameters = input.parameters
    position_manager = input.get_position_manager()
    state = input.state
    timestamp = input.timestamp
    indicators = input.indicators
    strategy_universe = input.strategy_universe

    # Guard against allocating based on stale forward-filled vault data.
    # The framework forward-fill keeps indicators from crashing, but it
    # also masks stale data — tvl() sees the last real TVL repeated,
    # age() keeps growing on synthetic rows, and age_ramp_weight()
    # increases. Bail out before the alpha model uses those values.
    # check_stale_vault_data(
    #    strategy_universe, 
    #    timestamp, 
    #    input.execution_context.mode,
    #    min_tvl=parameters.min_tvl_usd,
    #)

    portfolio = position_manager.get_current_portfolio()
    equity = portfolio.get_total_equity()

    if input.execution_context.mode == ExecutionMode.backtesting and equity < parameters.initial_cash * 0.10:
        return []

    alpha_model = AlphaModel(
        timestamp,
        close_position_weight_epsilon=parameters.min_portfolio_weight_pct,
    )

    tvl_included_pair_count = indicators.get_indicator_value("tvl_included_pair_count")
    included_pairs = indicators.get_indicator_value("inclusion_criteria", na_conversion=False)
    if included_pairs is None:
        included_pairs = []

    signal_count = 0
    for pair_id in included_pairs:
        pair = strategy_universe.get_pair_by_id(pair_id)
        if not state.is_good_pair(pair):
            continue
        if is_quarantined(pair.pool_address, timestamp):
            continue
        age_ramp_weight = indicators.get_indicator_value("age_ramp_weight", pair=pair)
        weight_signal_value = age_ramp_weight if age_ramp_weight is not None else 1.0
        alpha_model.set_signal(pair, weight_signal_value)
        signal_count += 1

    locked_position_value = alpha_model.carry_forward_non_redeemable_positions(position_manager)
    redeemable_capital = get_redeemable_portfolio_capital(position_manager)
    portfolio_target_value = calculate_portfolio_target_value(
        position_manager,
        parameters.allocation_pct,
    )
    deployable_target_value = max(portfolio_target_value - locked_position_value, 0.0)

    alpha_model.select_top_signals(count=parameters.max_assets_in_portfolio)
    alpha_model.assign_weights(method=weight_passthrouh)

    # Hyperliquid vaults are expected to have complete TVL data in live trading.
    # Fail the cycle loudly if TVL is missing instead of silently sizing with a placeholder.
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
    )

    if input.is_visualisation_enabled():
        try:
            top_signal = next(iter(alpha_model.get_signals_sorted_by_weight()))
            if top_signal.normalised_weight == 0:
                top_signal = None
        except StopIteration:
            top_signal = None

        rebalance_volume = sum(trade.get_value() for trade in trades)
        report = dedent_any(
            f"""
            Cycle: #{input.cycle}
            Rebalanced: {'👍' if alpha_model.is_rebalance_triggered() else '👎'}
            Open/about to open positions: {len(state.portfolio.open_positions)}
            Max position value change: {alpha_model.max_position_adjust_usd:,.2f} USD
            Rebalance threshold: {alpha_model.position_adjust_threshold_usd:,.2f} USD
            Trades decided: {len(trades)}
            Pairs meeting inclusion criteria: {len(included_pairs)}
            Pairs meeting TVL inclusion criteria: {tvl_included_pair_count}
            Candidate signals created: {signal_count}
            Selected survivor signals: {len(alpha_model.signals)}
            Weight signal: {parameters.weight_signal}
            Age ramp period: {parameters.age_ramp_period}
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
            """
        )

        if top_signal:
            assert top_signal.position_size_risk
            report += dedent_any(
                f"""
                Top signal pair: {top_signal.pair.get_ticker()}
                Top signal value: {top_signal.signal}
                Top signal weight: {top_signal.raw_weight}
                Top signal weight (normalised): {top_signal.normalised_weight * 100:.2f} % (got {top_signal.position_size_risk.get_relative_capped_amount() * 100:.2f} % of asked size)
                """
            )

        for flag, count in alpha_model.get_flag_diagnostics_data().items():
            report += f"Signals with flag {flag.name}: {count}" + "\n"

        state.visualisation.add_message(timestamp, report)
        state.visualisation.set_discardable_data("alpha_model", alpha_model)

    return trades


#
# Indicators
#

indicators = IndicatorRegistry()


@indicators.define(source=IndicatorSource.tvl)
def tvl(close: pd.Series) -> pd.Series:
    """TVL series for a pair.

    Framework forward-fill (via ``create_from_dataset(forward_fill=True,
    forward_fill_until=timestamp)``) already extends each pair's liquidity
    data to the decision timestamp. No manual forward-fill needed here
    because this strategy uses daily candles with daily TVL data.
    """
    return close


@indicators.define()
def age(close: pd.Series) -> pd.Series:
    inception = close.index[0]
    age_years = (close.index - inception) / pd.Timedelta(days=365.25)
    return pd.Series(age_years, index=close.index)


@indicators.define(dependencies=(tvl,), source=IndicatorSource.dependencies_only_universe)
def tvl_inclusion_criteria(
    min_tvl_usd: USDollarAmount,
    dependency_resolver: IndicatorDependencyResolver,
) -> pd.Series:
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
    min_tvl_usd: USDollarAmount,
    min_age: float,
    dependency_resolver: IndicatorDependencyResolver,
) -> pd.Series:
    # Supporting benchmark pairs are for comparison charts only.
    # They must never compete with real vaults for allocation decisions.
    # In live mode they are not loaded at all, so we resolve them defensively.
    benchmark_pair_ids = _get_available_supporting_pair_ids(strategy_universe)

    tvl_series = dependency_resolver.get_indicator_data(
        tvl_inclusion_criteria,
        parameters={"min_tvl_usd": min_tvl_usd},
    )
    trading_availability_series = dependency_resolver.get_indicator_data(trading_availability_criteria)
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

    def _combine(row):
        final_set = (
            set(row["tvl_pair_ids"])
            & set(row["trading_availability_pair_ids"])
            & set(row["age_pair_ids"])
        )
        return final_set - benchmark_pair_ids

    union_criteria = df.apply(_combine, axis=1)
    full_index = pd.date_range(
        start=union_criteria.index.min(),
        end=union_criteria.index.max(),
        freq=Parameters.candle_time_bucket.to_frequency(),
    )
    return union_criteria.reindex(full_index, fill_value=[])


@indicators.define(dependencies=(age,), source=IndicatorSource.dependencies_only_per_pair)
def age_ramp_weight(
    pair: TradingPairIdentifier,
    dependency_resolver: IndicatorDependencyResolver,
    age_ramp_period: float = 1.0,
) -> pd.Series:
    vault_age = dependency_resolver.get_indicator_data("age", pair=pair)
    return (vault_age / age_ramp_period).clip(upper=1.0).clip(lower=0.05)


@indicators.define(dependencies=(inclusion_criteria,), source=IndicatorSource.dependencies_only_universe)
def all_criteria_included_pair_count(
    min_tvl_usd: USDollarAmount,
    min_age: float,
    dependency_resolver: IndicatorDependencyResolver,
) -> pd.Series:
    series = dependency_resolver.get_indicator_data(
        "inclusion_criteria",
        parameters={
            "min_tvl_usd": min_tvl_usd,
            "min_age": min_age,
        },
    )
    return series.apply(len)


@indicators.define(dependencies=(tvl_inclusion_criteria,), source=IndicatorSource.dependencies_only_universe)
def tvl_included_pair_count(
    min_tvl_usd: USDollarAmount,
    dependency_resolver: IndicatorDependencyResolver,
) -> pd.Series:
    series = dependency_resolver.get_indicator_data(
        "tvl_inclusion_criteria",
        parameters={
            "min_tvl_usd": min_tvl_usd,
        },
    )
    return series.apply(len)


@indicators.define(dependencies=(age_inclusion_criteria,), source=IndicatorSource.dependencies_only_universe)
def age_included_pair_count(
    min_age: float,
    dependency_resolver: IndicatorDependencyResolver,
) -> pd.Series:
    series = dependency_resolver.get_indicator_data(
        "age_inclusion_criteria",
        parameters={
            "min_age": min_age,
        },
    )
    return series.apply(len)


@indicators.define(source=IndicatorSource.strategy_universe)
def trading_pair_count(
    strategy_universe: TradingStrategyUniverse,
) -> pd.Series:
    # Exclude benchmark-only pairs from the cumulative tradeable pair count.
    # This lookup must also work in live mode where the supporting pairs are
    # intentionally absent from the universe.
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
    return equity_curve_chart(
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
    """Trading pair breakdown with chain and address."""
    return trading_pair_breakdown(
        input,
        show_chain=True,
        show_address=True,
    )


def all_vault_positions_by_profit(input: ChartInput) -> pd.DataFrame:
    """Vault positions sorted by profit."""
    return all_vault_positions(
        input,
        sort_by="Profit USD",
        sort_ascending=False,
        show_address=True,
    )


def create_charts(
    timestamp: datetime.datetime | None,
    parameters: StrategyParameters,
    strategy_universe: TradingStrategyUniverse,
    execution_context: ExecutionContext,
) -> ChartRegistry:
    """Define charts we use in backtesting and live trading."""
    # Live universes intentionally omit SUPPORTING_PAIRS above.
    # Keeping them as default chart benchmark lookups would make the webhook
    # chart API try to resolve pairs that are not present in the live universe.
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
    charts.register(trading_pair_breakdown_with_chain, ChartKind.state_all_pairs)
    charts.register(trading_metrics, ChartKind.state_all_pairs)
    charts.register(vault_statistics, ChartKind.state_all_pairs)
    charts.register(all_vault_positions_by_profit, ChartKind.state_all_pairs)
    return charts


#
# Metadata
#

tags = {StrategyTag.beta, StrategyTag.live}

name = "Hyper AI"

short_description = "Survivor-first vault-of-vaults strategy on Hyperliquid"

icon = ""

long_description = """
# Hyper AI strategy

A diversified yield strategy that allocates across Hyperliquid native vaults using
survivor-first selection and age-ramp weighting.

## Strategy features

- **Survivor-first selection**: Vaults must pass TVL, age, and trading availability filters
- **Age-ramp weighting**: Younger vaults receive lower weights, ramping up over 0.75 years
- **Daily rebalancing**: Adjusts positions daily based on inclusion criteria and signal weights
- **Waterfall sizing**: Capped waterfall normalisation prevents over-concentration
- **Redemption-aware**: Target value accounts for pending redemptions

## Risk parameters

- Maximum 20 positions at any time
- 98% allocation target
- 12% maximum concentration per asset
- 20% per-position cap of pool TVL
- 5 USD minimum vault deposit (Hyperliquid hard floor)
"""
