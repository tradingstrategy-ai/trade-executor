"""Create a trading universe and a simple vault rebalance backtest.
"""
import datetime
import os
from pathlib import Path

import pandas as pd
import pytest

from plotly.graph_objs import Figure

from eth_defi.erc_4626.vault import ERC4626Vault
from eth_defi.provider.multi_provider import create_multi_provider_web3

from tradeexecutor.analysis.vault import visualise_vaults
from tradeexecutor.ethereum.vault.vault_utils import get_vault_from_trading_pair
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.exchange import ExchangeUniverse
from tradingstrategy.liquidity import GroupedLiquidityUniverse
from tradingstrategy.pair import PandasPairUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe


from tradeexecutor.backtest.backtest_runner import run_backtest_inline
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.trade import TradeExecution, TradeStatus
from tradeexecutor.strategy.alpha_model import AlphaModel
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.execution_context import unit_test_execution_context, ExecutionContext, ExecutionMode
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSource, calculate_and_load_indicators_inline, IndicatorDependencyResolver, DiskIndicatorStorage
from tradeexecutor.strategy.pandas_trader.indicator_decorator import IndicatorRegistry
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput, StrategyInputIndicators
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, translate_token
from tradeexecutor.strategy.tvl_size_risk import USDTVLSizeRiskModel
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradeexecutor.strategy.weighting import weight_passthrouh
from tradingstrategy.alternative_data.vault import load_multiple_vaults, load_vault_price_data, convert_vault_prices_to_candles



JSON_RPC_BASE = os.environ.get("JSON_RPC_BASE")


class Parameters:
    id = "vault-optimiser"
    candle_time_bucket = TimeBucket.d1
    cycle_duration = CycleDuration.cycle_1d
    chain_id = ChainId.base

    # Backtest duration
    backtest_start = datetime.datetime(2025, 1, 1)
    backtest_end = datetime.datetime(2025, 5, 10)
    initial_cash = 100_000

    # Signal parameters
    rolling_returns_bars = 7

    # Basket parameters
    allocation = 0.99  # Remaining % always in cash
    min_portfolio_weight = 0.005  # Don't open too small positions
    max_assets_in_portfolio = 5  # N vaults at a time
    max_concentration = 0.40  # Max % of portfolio per vault
    per_position_cap_of_pool = 0.01  # 1% of the vault TVL
    assumed_liquidity_when_data_missing = 0.0  # In data gaps, assume
    individual_rebalance_min_threshold_usd = 150.00
    sell_rebalance_min_threshold = 5.0



#
VAULTS = [
    (ChainId.base, "0x45aa96f0b3188d47a1dafdbefce1db6b37f58216"),  # Ipor Base
    (ChainId.base, "0xad20523a7dc37babc1cc74897e4977232b3d02e5"),  # Gains Network
    (ChainId.base, "0xcddcdd18a16ed441f6cb10c3909e5e7ec2b9e8f3"),  # Apostro Resolv USDC
    (ChainId.base, "0xc0c5689e6f4d256e861f65465b691aeecc0deb12"),  # Gauntled USDC core
    (ChainId.base, "0xb99b6df96d4d5448cc0a5b3e0ef7896df9507cf5"),  # 40 acres
    # https://summer.fi/earn/base/position/0x98c49e13bf99d7cad8069faa2a370933ec9ecf17
    (ChainId.base, "0x98c49e13bf99d7cad8069faa2a370933ec9ecf17"),  # Summer.fi lazy vault
    # https://app.morpho.org/base/vault/0x50b5b81Fc8B1f1873Ec7F31B0E98186ba008814D/indefi-usdc
    (ChainId.base, "0x50b5b81fc8b1f1873ec7f31b0e98186ba008814d"),  # InDefi USDc on Morpho
]


@pytest.fixture(scope="module")
def web3():
    return create_multi_provider_web3(JSON_RPC_BASE)


@pytest.fixture(scope="module")
def strategy_universe(persistent_test_client: Client):
    client = persistent_test_client
    strategy_universe = create_trading_universe(
        None,
        client=client,
        execution_context=unit_test_execution_context,
        universe_options=UniverseOptions.from_strategy_parameters_class(Parameters, unit_test_execution_context)
    )
    return strategy_universe


def test_create_vault_universe(
    strategy_universe,
):
    """Check we can read vault data from the universe."""

    # We have liquidity data correctly loaded
    pair = strategy_universe.get_pair_by_address("0x50b5b81fc8b1f1873ec7f31b0e98186ba008814d")
    assert pair.base.token_symbol == "indeUSDC"
    assert pair.get_vault_name() == "IndeFi USDC"


def test_visualise_vault_analysis_chart(
    strategy_universe,
):
    """Visualise vault data."""

    figures = visualise_vaults(strategy_universe)
    for fig in figures:
        assert isinstance(fig, Figure), f"Expected figure, got {type(fig)}"


@pytest.mark.skipif(not JSON_RPC_BASE, reason="Skip if JSON_RPC_BASE is not set")
def test_reverse_translate_vault(
    web3,
    strategy_universe,
):
    """Check we can construct vault instance from trading pair."""
    pair = strategy_universe.get_pair_by_address("0x50b5b81fc8b1f1873ec7f31b0e98186ba008814d")
    vault = get_vault_from_trading_pair(web3, pair)
    assert isinstance(vault, ERC4626Vault)
    assert vault.denomination_token.symbol == "USDC"
    assert vault.share_token.symbol == "indeUSDC"
    assert vault.name == "IndeFi USDC"



def test_vault_rebalance_strategy(
    strategy_universe: TradingStrategyUniverse,
    tmp_path: Path,
):
    """Simple vault rebalacne strategy."""
    # Calculate all indicators where parameters have changed and store the result on disk
    parameters = StrategyParameters.from_class(Parameters)
    indicator_storage = DiskIndicatorStorage(
        tmp_path,
        universe_key=strategy_universe.get_cache_key(),
    )
    indicator_data: StrategyInputIndicators = calculate_and_load_indicators_inline(
        strategy_universe=strategy_universe,
        create_indicators=indicators.create_indicators,
        parameters=parameters,
        storage=indicator_storage,
        max_workers=4,
    )

    result = run_backtest_inline(
        client=None,
        decide_trades=decide_trades,
        universe=strategy_universe,
        reserve_currency=ReserveCurrency.usdc,
        engine_version="0.5",
        parameters=parameters,
        mode=ExecutionMode.unit_testing,
        indicator_storage=indicator_storage,
        indicator_combinations=indicator_data.indicator_combinations,
    )

    state = result.state
    assert len(state.portfolio.closed_positions) >= 1

    trades = list(state.portfolio.get_all_trades())
    assert len(trades) == 506
    for t in trades:
        assert t.get_status() == TradeStatus.success
        assert t.is_success(), f"Trade {t.id} is not successful: {t}"
        assert t.executed_at, f"Trade executed at is not set: {t}"
        assert t.executed_price
        assert t.planned_price
        assert t.planned_reserve


def create_trading_universe(
    timestamp: datetime.datetime,
    client: Client,
    execution_context: ExecutionContext,
    universe_options: UniverseOptions,
) -> TradingStrategyUniverse:
    """Create a trading universe with named vaults on Base."""
    chain_id = Parameters.chain_id
    time_bucket = Parameters.candle_time_bucket

    exchanges, pairs_df = load_multiple_vaults(VAULTS)
    vault_prices_df = load_vault_price_data(pairs_df)

    # Create pair universe based on the vault data
    exchange_universe = ExchangeUniverse({e.exchange_id: e for e in exchanges})
    pair_universe = PandasPairUniverse(pairs_df, exchange_universe=exchange_universe)

    # Create price candles from vault share price scrape
    candle_df, liquidity_df = convert_vault_prices_to_candles(vault_prices_df, "1h")
    candle_universe = GroupedCandleUniverse(candle_df, time_bucket=TimeBucket.h1)
    liquidity_universe = GroupedLiquidityUniverse(liquidity_df, time_bucket=TimeBucket.h1)

    data_universe = Universe(
        time_bucket=time_bucket,
        chains={chain_id},
        exchange_universe=exchange_universe,
        pairs=pair_universe,
        candles=candle_universe,
        liquidity=liquidity_universe,
    )

    usdc_token = pair_universe.get_token("0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913".lower(), chain_id)
    assert usdc_token is not None
    usdc = translate_token(usdc_token)

    strategy_universe = TradingStrategyUniverse(
        data_universe=data_universe,
        reserve_assets=[usdc],
    )

    return strategy_universe


indicators = IndicatorRegistry()

# Calculate cleaned rolling returns
@indicators.define()
def rolling_returns(close: pd.Series, rolling_returns_bars: int) -> pd.Series:
    returns = close.pct_change()
    cumulative_rolling_returns = (1 + returns).rolling(window=rolling_returns_bars).apply(lambda x: x.prod() - 1)
    return cumulative_rolling_returns


@indicators.define(source=IndicatorSource.tvl)
def tvl(
    close: pd.Series,
    execution_context: ExecutionContext,
    timestamp: pd.Timestamp,
) -> pd.Series:
    if execution_context.live_trading:
        # TVL is daily data.
        # We need to forward fill until the current hour.
        # Use our special ff function.
        assert isinstance(timestamp, pd.Timestamp), f"Live trading needs forward-fill end time, we got {timestamp}"
        from tradingstrategy.utils.forward_fill import forward_fill
        df = pd.DataFrame({"close": close})
        df_ff = forward_fill(
            df,
            Parameters.candle_time_bucket.to_frequency(),
            columns=("close",),
            forward_fill_until=timestamp,
        )
        series = df_ff["close"]
        return series
    else:
        return close.resample("1h").ffill()


@indicators.define(
    dependencies=(rolling_returns,),
    source=IndicatorSource.dependencies_only_per_pair
)
def signal(
    rolling_returns_bars: int,
    dependency_resolver: IndicatorDependencyResolver,
    pair: TradingPairIdentifier,
) -> pd.Series:
    """Momentum signal: signal = 7 days returns"""
    series = dependency_resolver.get_indicator_data(
        name=rolling_returns,
        parameters={
            "rolling_returns_bars": rolling_returns_bars,
        },
        pair=pair,
    )
    return series


def decide_trades(
    input: StrategyInput
) -> list[TradeExecution]:
    """For each strategy tick, generate the list of trades."""
    parameters = input.parameters
    position_manager = input.get_position_manager()
    state = input.state
    timestamp = input.timestamp
    indicators = input.indicators
    strategy_universe = input.strategy_universe

    portfolio = state.portfolio
    assert portfolio.get_total_equity() > 0
    assert portfolio.get_position_equity_and_loan_nav() >= 0

    # Set up alpha model
    alpha_model = AlphaModel(
        timestamp,
        close_position_weight_epsilon=parameters.min_portfolio_weight,  # 10 BPS is our min portfolio weight
    )

    # Generate new weights
    for pair in strategy_universe.iterate_pairs():
        weight = indicators.get_indicator_value("signal", pair=pair)
        if weight is None or weight <= 0:
            # The vault does not yet exist in this timestamp,
            # or has negative rolling returns
            continue
        alpha_model.set_signal(
            pair,
            weight,
        )

    portfolio_target_value = portfolio.get_total_equity() * parameters.allocation

    alpha_model.select_top_signals(count=parameters.max_assets_in_portfolio)
    alpha_model.assign_weights(method=weight_passthrouh)

    alpha_model.update_old_weights(
        state.portfolio,
        ignore_credit=False,
    )

    size_risk_model = USDTVLSizeRiskModel(
        pricing_model=input.pricing_model,
        per_position_cap=parameters.per_position_cap_of_pool,  # This is how much % by all pool TVL we can allocate for a position
        missing_tvl_placeholder_usd=parameters.assumed_liquidity_when_data_missing,  # Placeholder for missing TVL data until we get the data off the chain
    )

    alpha_model.normalise_weights(
        investable_equity=portfolio_target_value,
        size_risk_model=size_risk_model,
        max_weight=parameters.max_concentration,
    )
    alpha_model.calculate_target_positions(position_manager)

    trades = alpha_model.generate_rebalance_trades_and_triggers(
        position_manager,
        min_trade_threshold=parameters.individual_rebalance_min_threshold_usd,
        invidiual_rebalance_min_threshold=parameters.individual_rebalance_min_threshold_usd,
        sell_rebalance_min_threshold=parameters.sell_rebalance_min_threshold,
        execution_context=input.execution_context,
    )

    try:
        position_manager.check_enough_cash(trades)
    except Exception as e:
        # Dump alpha model
        raise RuntimeError(f"Alpha model flow calculations failed: {alpha_model.get_debug_print()}") from e

    return trades
