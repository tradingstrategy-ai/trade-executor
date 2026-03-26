"""Cross-chain master vault strategy.

Based on ``16-backtest-volvol-veto-best-cagr-jan-start.ipynb`` notebook.

Cross-chain vault allocation with age-ramp weighting, daily rebalancing,
vol-of-vol veto, stale-data protection, and redemption-aware target sizing.
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
from tradeexecutor.strategy.chart.standard.alpha_model import alpha_model_diagnostics
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
from tradeexecutor.strategy.chart.standard.vault import all_vault_positions
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

ALLOWED_VAULT_DENOMINATION_TOKENS = {
    "USDC",
    "USDT",
    "USDC.e",
    "crvUSD",
    "USDT0",
    "USD₮0",
    "USDt",
    "USDS",
}

HAND_CURATED_VAULTS = [
    # Ethereum
    (ChainId.ethereum, "0x2e87d6bfa3f2a932e0c70a32607c0b839404984d"),
    (ChainId.ethereum, "0x438982ea288763370946625fd76c2508ee1fb229"),
    (ChainId.ethereum, "0x786977528b0265c5c5bc9544ac56c863c03e34d1"),
    (ChainId.ethereum, "0x09c4c7b1d2e9aa7506db8b76f1dbbd61c08c114b"),
    (ChainId.ethereum, "0x01f461a0bbb218bc1943aa027c5bbc424391e541"),
    (ChainId.ethereum, "0xedc72b49542e4362c677b8369bc23882ed635a75"),
    (ChainId.ethereum, "0xca790385506b790554571cbc9da73f0130cdcfd5"),
    (ChainId.ethereum, "0xb250c9e0f7be4cff13f94374c993ac445a1385fe"),
    (ChainId.ethereum, "0x8df3deba711ae4a9af16cbca5e4fbb1402f036d5"),
    (ChainId.ethereum, "0xe9d33286f0e37f517b1204aa6da085564414996d"),
    # Base
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
    # Arbitrum
    (ChainId.arbitrum, "0x75288264fdfea8ce68e6d852696ab1ce2f3e5004"),
    (ChainId.arbitrum, "0x58bfc95a864e18e8f3041d2fcd3418f48393fe6a"),
    (ChainId.arbitrum, "0xf63b7f49b4f5dc5d0e7e583cfd79dc64e646320c"),
    (ChainId.arbitrum, "0x1723cb57af58efb35a013870c90fcc3d60174a4e"),
    (ChainId.arbitrum, "0x20d419a8e12c45f88fda7c5760bb6923cee27f98"),
    (ChainId.arbitrum, "0xc8248953429d707c6a2815653eca89846ffaa63b"),
    (ChainId.arbitrum, "0x4739e2c293bdcd835829aa7c5d7fbdee93565d1a"),
    (ChainId.arbitrum, "0xd3443ee1e91af28e5fb858fbd0d72a63ba8046e0"),
    (ChainId.arbitrum, "0x0df2e3a0b5997adc69f8768e495fd98a4d00f134"),
    (ChainId.arbitrum, "0x9fa306b1f4a6a83fec98d8ebbabedff78c407f6b"),
    # Avalanche
    (ChainId.avalanche, "0x606fe9a70338e798a292ca22c1f28c829f24048e"),
    (ChainId.avalanche, "0x4af3abe954259fb70b97c57ebd7ac1eb822028ef"),
    (ChainId.avalanche, "0x37ca03ad51b8ff79aad35fadacba4cedf0c3e74e"),
    (ChainId.avalanche, "0x39de0f00189306062d79edec6dca5bb6bfd108f9"),
    (ChainId.avalanche, "0xeaf77df5d03306bca4ee8b58b6821e6aca76309d"),
    (ChainId.avalanche, "0x8fc260cd0a00cac30eb1f444b8f1511d71420af9"),
    (ChainId.avalanche, "0x8f23da78e3f31ab5deb75dc3282198bed630ffde"),
    (ChainId.avalanche, "0x39288474bc5931d3c4705e866b6e21cc2e47617d"),
    # HyperEVM
    (ChainId.hyperliquid, "0x1c5164a764844356d57654ea83f9f1b72cd10db5"),
    (ChainId.hyperliquid, "0x195eb4d088f222c982282b5dd495e76dba4bc7d1"),
    (ChainId.hyperliquid, "0x2c910f67dbf81099e6f8e126e7265d7595dc20ad"),
    (ChainId.hyperliquid, "0xe5add96840f0b908ddeb3bd144c0283ac5ca7ca0"),
    (ChainId.hyperliquid, "0x9896a8605763106e57a51aa0a97fe8099e806bb3"),
    (ChainId.hyperliquid, "0x08c00f8279dff5b0cb5a04d349e7d79708ceadf3"),
    (ChainId.hyperliquid, "0xfc5126377f0efc0041c0969ef9ba903ce67d151e"),
    (ChainId.hyperliquid, "0x8a862fd6c12f9ad34c9c2ff45ab2b6712e8cea27"),
    (ChainId.hyperliquid, "0x53a333e51e96fe288bc9add7cdc4b1ead2cd2ffa"),
    (ChainId.hyperliquid, "0xdc6f4239c1d8d3b955c06cb8f1a6cf18effc5bfe"),
    # Monad
    (ChainId.monad, "0xa8665084d8cd6276c00ca97cbc0bf4bc9ae94c79"),
    (ChainId.monad, "0x8ee9fc28b8da872c38a496e9ddb9700bb7261774"),
    (ChainId.monad, "0x0da39b740834090c146dc48357f6a435a1bb33b3"),
    (ChainId.monad, "0x802c91d807a8daca257c4708ab264b6520964e44"),
    (ChainId.monad, "0x6b343f7b797f1488aa48c49d540690f2b2c89751"),
    (ChainId.monad, "0x961a59fe249b9795fae7fa35f9e89629689d5278"),
    (ChainId.monad, "0xf19e8ddc541dee2f4d6796a79b1c1e10a415a0da"),
    (ChainId.monad, "0x78999cc96d2ba0341588c60ccb0e91c6c33cf371"),
    (ChainId.monad, "0xbeeff443c3cba3e369da795002243beac311ab83"),
    (ChainId.monad, "0xbeeff300e9a9caec7beea740ab8758d33b777509"),
]

SOURCE_VAULTS = HAND_CURATED_VAULTS
BENCHMARK_PAIRS = SUPPORTING_PAIRS

#
# Strategy parameters
#


class Parameters:

    id = "xchain-master-vault"

    #: Daily candles match the validated survivor-first research branch.
    candle_time_bucket = TimeBucket.d1
    #: Keep daily rebalance cadence from the notebook.
    cycle_duration = CycleDuration.cycle_1d
    #: Run this strategy as a cross-chain vault universe.
    chain_id = CHAIN_ID
    #: Use Ethereum as the primary reserve chain for the cross-chain universe.
    primary_chain_id = PRIMARY_CHAIN_ID
    #: Keep the same exchange set as the notebook.
    exchanges = EXCHANGES
    #: Allow test-only variants reuse the production universe builder.
    supporting_pairs = SUPPORTING_PAIRS
    #: Allow test-only variants narrow the live vault universe.
    source_vaults = SOURCE_VAULTS
    #: Allow test-only variants change the reserve chain cleanly.
    preferred_stablecoin = PREFERRED_STABLECOIN
    #: Enable shared synthetic forward CCTP bridge generation for satellite chains.
    auto_generate_cctp_bridges = True

    #: Keep the validated 20-vault basket from the notebook.
    max_assets_in_portfolio = 20
    #: Keep the validated deployment target from the notebook.
    allocation_pct = 0.98
    #: Keep the validated concentration cap from the notebook.
    max_concentration_pct = 0.12
    #: Keep the pool-cap ceiling from the notebook.
    per_position_cap_of_pool_pct = 0.2
    #: Engine hygiene threshold used across the survivor-first strategies.
    min_portfolio_weight_pct = 0.005

    #: Keep the absolute minimum vault deposit floor from the notebook.
    absolute_min_vault_deposit_usd = 5.0
    #: Preserve the old 50 USD buy threshold at 100k initial cash.
    individual_rebalance_min_threshold_of_initial_cash_pct = 0.0005
    #: Preserve the old 10 USD sell threshold at 100k initial cash.
    sell_rebalance_min_threshold_of_initial_cash_pct = 0.0001

    #: Keep the survivor-first release TVL floor for the initial cross-chain read.
    min_tvl_usd = 7_500
    #: Keep the same young-vault-inclusive age floor from the notebook.
    min_age = 0.075
    #: Keep the surviving signal family unchanged.
    weight_signal = "age_ramp"
    #: Keep the validated age-ramp period unchanged.
    age_ramp_period = 0.75

    #: Rolling window in days for volatility and vol-of-vol calculations.
    vol_window = 60
    #: Vol-of-vol percentile threshold for veto.
    volvol_veto_percentile = 0.75

    #: Keep the same mature-universe start as the Jan-start notebook.
    backtest_start = datetime.datetime(2025, 1, 1)
    #: Keep the same end date so the result stays comparable.
    backtest_end = datetime.datetime(2026, 3, 11)
    #: Use a standard treasury size.
    initial_cash = 100_000
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

    #: Keep the same data loading window so age and TVL indicators can warm up.
    required_history_period = datetime.timedelta(days=365)
    #: Route through the default DEX router stack.
    routing = TradeRouting.default
    #: Keep a live-style slippage assumption aligned with other vault strategies.
    slippage_tolerance_pct = 0.0060
    #: Assume no liquidity if there is a gap in TVL data.
    assummed_liquidity_when_data_missings_usd = 0.01


#
# Universe creation
#


def create_trading_universe(
    input: CreateTradingUniverseInput,
) -> TradingStrategyUniverse:
    """Create the cross-chain trading universe.

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

    if execution_context.live_trading:
        supporting_pairs = []
    else:
        supporting_pairs = parameters.supporting_pairs

    debug_printer(f"Preparing trading universe on chain {chain_id.get_name()}")

    all_pairs_df = client.fetch_pair_universe().to_pandas()
    pairs_df = filter_for_selected_pairs(all_pairs_df, supporting_pairs)
    debug_printer(f"We have total {len(all_pairs_df)} pairs in dataset and going to use {len(pairs_df)} pairs for the strategy")

    vault_universe = load_vault_universe_with_metadata(client, vaults=parameters.source_vaults)
    vault_universe = vault_universe.limit_to_denomination(
        ALLOWED_VAULT_DENOMINATION_TOKENS,
        check_all_vaults_found=True,
    )
    debug_printer(
        f"Loaded {vault_universe.get_vault_count()} vaults from remote vault metadata, "
        f"source vaults count: {len(parameters.source_vaults)}"
    )

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
        reserve_asset=parameters.preferred_stablecoin,
        forward_fill=True,
        forward_fill_until=timestamp,
        primary_chain=parameters.primary_chain_id,
        auto_generate_cctp_bridges=parameters.auto_generate_cctp_bridges,
    )


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
    # also masks stale data: tvl() sees the last real TVL repeated,
    # age() keeps growing on synthetic rows, and age_ramp_weight()
    # increases. Bail out before the alpha model uses those values.
    check_stale_vault_data(strategy_universe, timestamp, input.execution_context.mode)

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

    volvol_values = {}
    for pair_id in included_pairs:
        pair = strategy_universe.get_pair_by_id(pair_id)
        vol_of_vol_value = indicators.get_indicator_value("vol_of_vol", pair=pair)
        if vol_of_vol_value is not None and not pd.isna(vol_of_vol_value):
            volvol_values[pair_id] = vol_of_vol_value

    if volvol_values:
        volvol_threshold = np.percentile(
            list(volvol_values.values()),
            parameters.volvol_veto_percentile * 100,
        )
    else:
        volvol_threshold = float("inf")

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
    portfolio_target_value = calculate_portfolio_target_value(
        position_manager,
        parameters.allocation_pct,
    )
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
            Vol-of-vol veto percentile: {parameters.volvol_veto_percentile}
            Vol-of-vol threshold: {volvol_threshold:.6f}
            Vaults vetoed by vol-of-vol: {vetoed_count}
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
            report += f"Signals with flag {flag.name}: {count}\n"

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
        parameters={"min_tvl_usd": min_tvl_usd},
    )
    return series.apply(len)


@indicators.define(dependencies=(age_inclusion_criteria,), source=IndicatorSource.dependencies_only_universe)
def age_included_pair_count(
    min_age: float,
    dependency_resolver: IndicatorDependencyResolver,
) -> pd.Series:
    series = dependency_resolver.get_indicator_data(
        "age_inclusion_criteria",
        parameters={"min_age": min_age},
    )
    return series.apply(len)


@indicators.define(source=IndicatorSource.strategy_universe)
def trading_pair_count(
    strategy_universe: TradingStrategyUniverse,
) -> pd.Series:
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

tags = {StrategyTag.beta, StrategyTag.deposits_disabled}

name = "Xchain master vault strategy"

short_description = "Cross-chain vault allocation strategy with age-ramp weighting, vol-of-vol veto, and redemption-aware sizing"

icon = ""

long_description = """
# Cross-chain master vault strategy

A diversified yield strategy that allocates across a hand-curated cross-chain vault universe.

## Strategy features

- **Cross-chain allocation**: Invests across Ethereum, Base, Arbitrum, Avalanche, HyperEVM, and Monad
- **Age-ramp weighting**: Younger vaults receive lower weights, ramping up over 0.75 years
- **Vol-of-vol veto**: Excludes vaults whose realised volatility is too unstable versus peers
- **Daily rebalancing**: Adjusts positions every day using survivor-first capped waterfall sizing
- **Stale-data guard**: Refuses to rebalance on forward-filled-but-stale vault data
- **Redemption-aware**: Locked vault capital is carried forward and excluded from fresh allocation

## Risk parameters

- Maximum 20 positions at any time
- 98% allocation target
- 12% maximum concentration per asset
- 20% per-position cap of pool TVL
- 5 USD minimum vault deposit floor
"""
