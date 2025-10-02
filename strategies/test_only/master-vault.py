"""Master vault strategy stub.

- Currently a place holder

Check universe and indicators:

    trade-executor \
        check-universe \
        --strategy-file=strategy/master-vault.py \
        --trading-strategy-api-key=$TRADING_STRATEGY_API_KEY

Run backtest:

    trade-executor \
        backtest \
        --strategy-file=strategy/master-vault.py \
        --trading-strategy-api-key=$TRADING_STRATEGY_API_KEY

Perform test trade:

    docker compose run \
        bnb-local-high \
        perform-test-trade \
        --pair "(binance, pancakeswap-v2, Cake, WBNB, 0.0025)"  \
        --simulate \
        --amount=1.0
"""
import datetime
import logging
import os
from pathlib import Path
from pprint import pformat
from typing import cast

import pandas as pd

from eth_defi.token import USDC_NATIVE_TOKEN, USDT_NATIVE_TOKEN
from eth_defi.token import WRAPPED_NATIVE_TOKEN
from eth_defi.token_analysis.tokenrisk import CachedTokenRisk
from tradeexecutor.ethereum.lagoon.execution import LagoonExecution
from tradeexecutor.ethereum.lagoon.token_compat_check import check_tokens_for_lagoon

from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.types import USDollarAmount
from tradeexecutor.strategy.alpha_model import AlphaModel
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorDependencyResolver
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSource
from tradeexecutor.strategy.pandas_trader.indicator_decorator import IndicatorRegistry
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.pandas_trader.trading_universe_input import CreateTradingUniverseInput
from tradeexecutor.strategy.pandas_trader.yield_manager import YieldRuleset, YieldWeightingRule, YieldManager, YieldDecisionInput
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.tag import StrategyTag
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.strategy.trading_strategy_universe import (
    load_partial_data)
from tradeexecutor.strategy.tvl_size_risk import USDTVLSizeRiskModel
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradeexecutor.strategy.weighting import weight_passthrouh
from tradingstrategy.alternative_data.vault import load_single_vault, DEFAULT_VAULT_PRICE_BUNDLE
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.pair import PandasPairUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.utils.forward_fill import forward_fill
from tradingstrategy.utils.groupeduniverse import resample_candles

from tradingstrategy.utils.token_filter import add_base_quote_address_columns
from tradingstrategy.utils.token_filter import filter_for_exchange_slugs
from tradingstrategy.utils.token_filter import filter_pairs_default
from tradingstrategy.utils.token_extra_data import load_token_metadata
from tradingstrategy.utils.token_filter import filter_by_token_sniffer_score
from tradingstrategy.utils.token_filter import deduplicate_pairs_by_volume


from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.utils.dedent import dedent_any

from tqdm_loggable.auto import tqdm


logger = logging.getLogger(__name__)

#
# Strategy parameters
#


trading_strategy_engine_version = "0.6"

CHAIN_ID = ChainId.arbitrum

EXCHANGES = ("uniswap-v2", "uniswap-v3")

TOKEN_RISK_ENABLED = True

ANVIL_CHECK_ENABLED = True

SUPPORTING_PAIRS = [
    (ChainId.arbitrum, "uniswap-v3", "WETH", "USDC", 0.0005),
]

EXAMINED_PAIRS = [
    (ChainId.arbitrum, "uniswap-v3", "WETH", "USDC", 0.0005),
]


LENDING_RESERVES = None

PREFERRED_STABLECOIN = USDC_NATIVE_TOKEN[CHAIN_ID].lower()

MIN_TVL = 1_000_000
MIN_VOLUME = 25_000

# See deploy script for the vault address.
# Taken from Arbitrum notebook erc-4626-vaults-per-chain.ipynb
VAULT_LIST = "0x4b6f1c9e5d470b97181786b26da0d0945a7cf027, 0x3a87cf9af4d21778dad1ce7d0bf053f4b8f2631f, 0x959f3807f0aa7921e18c78b00b2819ba91e52fef, 0xeba51f6472f4ce1c47668c2474ab8f84b32e1ae7, 0x407d3d942d0911a2fea7e22417f81e27c02d6c6f, 0xb739ae19620f7ecb4fb84727f205453aa5bc1ad2, 0x36b69949d60d06eccc14de0ae63f4e00cc2cd8b9, 0x79f76e343807ea194789d114e61be6676e6bbeda, 0xa53cf822fe93002aeae16d395cd823ece161a6ac, 0xacb7432a4bb15402ce2afe0a7c9d5b738604f6f9, 0xd3443ee1e91af28e5fb858fbd0d72a63ba8046e0, 0x9fa306b1f4a6a83fec98d8ebbabedff78c407f6b, 0xc8248953429d707c6a2815653eca89846ffaa63b, 0xea50f402653c41cadbafd1f788341db7b7f37816, 0x8a1ef3066553275829d1c0f64ee8d5871d5ce9d3, 0xa7781f1d982eb9000bc1733e29ff5ba2824cdbe5, 0x4f63cfea7458221cb3a0eee2f31f7424ad34bb58, 0x6ca200319a0d4127a7a473d6891b86f34e312f42, 0xd691d8e3bc5008708786114481714b9c636f766f, 0x4a3f7dd63077cde8d7eff3c958eb69a3dd7d31a9, 0xd46993f25d298ebbcd31e941156c66f7e628a52a, 0x0df2e3a0b5997adc69f8768e495fd98a4d00f134, 0xcafc0a559a9bf2fc77a7ecfaf04bd929a7d9c5cf, 0x444868b6e8079ac2c55eea115250f92c2b2c4d14, 0xbc404429558292ee2d769e57d57d6e74bbd2792d, 0x4b6f1c9e5d470b97181786b26da0d0945a7cf027"

VAULTS = [
    (ChainId.arbitrum, v.strip()) for v in VAULT_LIST.split(",")
]

class Parameters:
    id = "master-vault"

    # We trade 1h candle
    candle_time_bucket = TimeBucket.h1
    cycle_duration = CycleDuration.cycle_1h

    chain_id = CHAIN_ID
    exchanges = EXCHANGES

    min_tvl_prefilter = 1_000_000
    min_tvl_filter = min_tvl_prefilter

    #
    #
    # Backtesting only
    # Limiting factor: Aave v3 on Base starts at the end of DEC 2023
    #
    backtest_start = datetime.datetime(2024, 6, 1)
    backtest_end = datetime.datetime(2025, 7, 1)
    initial_cash = 100_000

    #
    # Live only
    #
    routing = TradeRouting.default
    required_history_period = datetime.timedelta(days=365 * 3)
    slippage_tolerance = 0.0060  # 0.6%
    assummed_liquidity_when_data_missings = 10_000


def create_trading_universe(
    input: CreateTradingUniverseInput,
) -> TradingStrategyUniverse:
    """Create the trading universe.

    - Load Trading Strategy full pairs dataset

    - Load built-in Coingecko top 1000 dataset

    - Get all DEX tokens for a certain Coigecko category

    - Load OHCLV data for these pairs

    - Load also BTC and ETH price data to be used as a benchmark
    """

    execution_context = input.execution_context
    client = input.client
    timestamp = input.timestamp
    parameters = input.parameters or Parameters  # Some CLI commands do not support yet passing this
    universe_options = input.universe_options

    if execution_context.live_trading:
        # Live trading, send strategy universe formation details
        # to logs
        debug_printer = logger.info

    else:
        # Notebook node
        debug_printer = print

    chain_id = parameters.chain_id

    debug_printer(f"Preparing trading universe on chain {chain_id.get_name()}")

    exchange_universe = client.fetch_exchange_universe()
    targeted_exchanges = [exchange_universe.get_by_chain_and_slug(chain_id, slug) for slug in parameters.exchanges]

    # Pull out our benchmark pairs ids.
    # We need to construct pair universe object for the symbolic lookup.
    # TODO: PandasPairUniverse(buidl_index=True) - speed this up by skipping index building
    all_pairs_df = client.fetch_pair_universe().to_pandas()
    all_pairs_df = filter_for_exchange_slugs(all_pairs_df, parameters.exchanges)
    debug_printer("Creating universe for benchmark pair extraction")
    pair_universe = PandasPairUniverse(
        all_pairs_df,
        exchange_universe=exchange_universe,
        build_index=False,
    )
    debug_printer(f"Exchanges {parameters.exchanges} have total {len(all_pairs_df):,} pairs on chain {parameters.chain_id.get_name()}")

    # Get TVL data for prefilteirng
    if execution_context.live_trading:
        # For live trading, we take TVL data from ~around the start of the strategy until today
        tvl_time_bucket = TimeBucket.d1
        start = datetime.datetime(2024, 2, 1)
        end = tvl_time_bucket.floor(pd.Timestamp(datetime.datetime.utcnow() - tvl_time_bucket.to_timedelta()))
    else:
        start = parameters.backtest_start
        end = parameters.backtest_end

    #
    # Do exchange and TVL prefilter pass for the trading universe
    #
    min_tvl = parameters.min_tvl_prefilter
    # logging.getLogger().setLevel(logging.INFO)
    liquidity_time_bucket = TimeBucket.d1
    tvl_df = client.fetch_tvl(
        mode="min_tvl_low",
        bucket=liquidity_time_bucket,
        start_time=start,
        end_time=end,
        exchange_ids=[exc.exchange_id for exc in targeted_exchanges],
        min_tvl=min_tvl,
    )
    # logging.getLogger().setLevel(logging.WARNING)
    debug_printer(f"Fetch TVL, we got {len(tvl_df['pair_id'].unique())} pairs with TVL data for min TVL criteria {min_tvl}")

    tvl_filtered_pair_ids = tvl_df["pair_id"].unique()
    benchmark_pair_ids = [pair_universe.get_pair_by_human_description(desc).pair_id for desc in SUPPORTING_PAIRS]
    needed_pair_ids = set(benchmark_pair_ids) | set(tvl_filtered_pair_ids)
    pairs_df = all_pairs_df[all_pairs_df["pair_id"].isin(needed_pair_ids)]
    debug_printer(f"After TVL prefilter to {parameters.min_tvl_prefilter:,} in {parameters.backtest_start} - {parameters.backtest_end}, we have {len(pairs_df)} trading pairs")
    pairs_df = add_base_quote_address_columns(pairs_df)

    # Never deduplicate supporting pars
    supporting_pairs_df = pairs_df[pairs_df["pair_id"].isin(benchmark_pair_ids)]

    allowed_quotes = {
        PREFERRED_STABLECOIN,
        WRAPPED_NATIVE_TOKEN[chain_id.value].lower(),
    }
    filtered_pairs_df = filter_pairs_default(
        pairs_df,
        good_quote_token_addresses=allowed_quotes,
        verbose_print=print,
    )

    # Deduplicate trading pairs - Choose the best pair with the best volume
    deduplicated_df = deduplicate_pairs_by_volume(filtered_pairs_df)

    # Get our reference pairs back to the dataset
    pairs_df = pd.concat([deduplicated_df, supporting_pairs_df]).drop_duplicates(subset='pair_id', keep='first')
    debug_printer(f"After deduplication we have {len(pairs_df)} pairs")

    # Add benchmark pairs back to the dataset
    pairs_df = pd.concat([pairs_df, supporting_pairs_df]).drop_duplicates(subset='pair_id', keep='first')

    # Load metadata
    debug_printer("Loading metadata")
    # logging.getLogger().setLevel(logging.INFO)
    pairs_df = load_token_metadata(pairs_df, client, printer=debug_printer)
    # logging.getLogger().setLevel(logging.WARNING)

    risk_filtered_pairs_df = pairs_df

    uni_v2 = pairs_df.loc[pairs_df["exchange_slug"] == "uniswap-v2"]
    uni_v3 = pairs_df.loc[pairs_df["exchange_slug"] == "uniswap-v3"]
    other_dex = pairs_df.loc[~((pairs_df["exchange_slug"] != "uniswap-v3") | (pairs_df["exchange_slug"] != "uniswap-v2"))]
    debug_printer(f"Pairs on Uniswap v2: {len(uni_v2)}, Uniswap v3: {len(uni_v3)}, other DEX: {len(other_dex)}")

    if execution_context.live_trading:
        # Bundled vault price data is only used for backtesting
        vault_bundled_price_data = None
    else:
        if Path("/.dockerenv").exists():
            # Running inside Docker Container (prod).
            # Specially mapped path for Docker image,
            # in docker-compose.yml to get the production backtest done
            vault_path = Path.cwd() / "vaults"
            assert vault_path.exists(), f"Vaults path {vault_path} does not exist, please check your docker-compose.yml file"
            vault_bundled_price_data = vault_path / "cleaned-vault-prices-1h.parquet"
        else:
            # Default vault data bundle path for backtesting
            vault_bundled_price_data = Path.home() / ".tradingstrategy" / "vaults" / "cleaned-vault-prices-1h.parquet"

    dataset = load_partial_data(
        client=client,
        time_bucket=parameters.candle_time_bucket,
        pairs=pairs_df,
        execution_context=execution_context,
        universe_options=universe_options,
        liquidity_time_bucket=liquidity_time_bucket,
        preloaded_tvl_df=tvl_df,
        lending_reserves=LENDING_RESERVES,
        vaults=VAULTS,
        vault_bundled_price_data=vault_bundled_price_data,
        check_all_vaults_found=False,
    )

    reserve_asset = PREFERRED_STABLECOIN

    debug_printer("Creating trading universe")
    strategy_universe = TradingStrategyUniverse.create_from_dataset(
        dataset,
        reserve_asset=reserve_asset,
        forward_fill=True,  # We got very gappy data from low liquid DEX coins
        forward_fill_until=timestamp,
    )

    # Tag benchmark/routing pairs tokens so they can be separated from the rest of the tokens
    # for the index construction.
    strategy_universe.warm_up_data()
    for pair_id in benchmark_pair_ids:
        pair = strategy_universe.get_pair_by_id(pair_id)
        pair.other_data["benchmark"] = False

    return strategy_universe


#
# Strategy logic
#




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
    return []


#
# Indicators
#


empty_series = pd.Series([], index=pd.DatetimeIndex([]))

indicators = IndicatorRegistry()


def create_indicators(
    timestamp: datetime.datetime,
    parameters: StrategyParameters,
    strategy_universe: TradingStrategyUniverse,
    execution_context: ExecutionContext
):
    return indicators.create_indicators(
        timestamp=timestamp,
        parameters=parameters,
        strategy_universe=strategy_universe,
        execution_context=execution_context,
    )


#
# Charts
#

from plotly.graph_objects import Figure
from tradeexecutor.strategy.chart.definition import ChartRegistry, ChartKind, ChartInput

from tradeexecutor.strategy.chart.standard.trading_universe import available_trading_pairs
from tradeexecutor.strategy.chart.standard.trading_universe import inclusion_criteria_check
from tradeexecutor.strategy.chart.standard.volatility import volatility_benchmark
from tradeexecutor.strategy.chart.standard.signal import signal_comparison
from tradeexecutor.strategy.chart.standard.signal import price_vs_signal
from tradeexecutor.strategy.chart.standard.equity_curve import equity_curve
from tradeexecutor.strategy.chart.standard.equity_curve import equity_curve_with_drawdown
from tradeexecutor.strategy.chart.standard.performance_metrics import performance_metrics
from tradeexecutor.strategy.chart.standard.weight import volatile_weights_by_percent
from tradeexecutor.strategy.chart.standard.weight import volatile_and_non_volatile_percent
from tradeexecutor.strategy.chart.standard.weight import equity_curve_by_asset
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


# Some custom indicators for this notebook
def local_high_chart(input: ChartInput) -> list[Figure]:
    """Local high indicator vs. price"""
    return price_vs_signal(input, indicator_name="local_high")


def equity_curve_with_benchmark(input: ChartInput) -> Figure:
    """Equity curve with benchmark comparison"""
    return equity_curve(
        input,
        benchmark_token_symbols=["Cake", "WBNB"],
    )

# Define charts we use in backtesting and live trading
def create_charts(
    timestamp: datetime.datetime | None,
    parameters: StrategyParameters,
    strategy_universe: TradingStrategyUniverse,
    execution_context: ExecutionContext,
) -> ChartRegistry:
    charts = ChartRegistry(default_benchmark_pairs=EXAMINED_PAIRS)
    charts.register(available_trading_pairs, ChartKind.indicator_all_pairs)
    charts.register(inclusion_criteria_check, ChartKind.indicator_all_pairs)
    charts.register(volatility_benchmark, ChartKind.indicator_multi_pair)
    charts.register(signal_comparison, ChartKind.indicator_multi_pair)
    charts.register(price_vs_signal, ChartKind.indicator_multi_pair)
    charts.register(local_high_chart, ChartKind.indicator_multi_pair, "Local high")
    charts.register(equity_curve_with_benchmark, ChartKind.state_all_pairs)
    charts.register(equity_curve_with_drawdown, ChartKind.state_all_pairs)
    charts.register(performance_metrics, ChartKind.state_all_pairs)
    charts.register(volatile_weights_by_percent, ChartKind.state_all_pairs)
    charts.register(volatile_and_non_volatile_percent, ChartKind.state_all_pairs)
    charts.register(equity_curve_by_asset, ChartKind.state_all_pairs)
    charts.register(weight_allocation_statistics, ChartKind.state_all_pairs)
    charts.register(positions_at_end, ChartKind.state_all_pairs)
    charts.register(last_messages, ChartKind.state_all_pairs)
    charts.register(alpha_model_diagnostics, ChartKind.state_all_pairs)
    charts.register(trading_pair_breakdown, ChartKind.state_all_pairs)
    charts.register(trading_metrics, ChartKind.state_all_pairs)
    charts.register(lending_pool_interest_accrued, ChartKind.state_all_pairs)
    charts.register(vault_statistics, ChartKind.state_all_pairs)
    charts.register(trading_pair_positions, ChartKind.state_single_pair)
    charts.register(trading_pair_price_and_trades, ChartKind.state_single_pair)
    return charts


tags = {StrategyTag.beta}

name = "Master Vault"

short_description = "Vault of vaults strategy which invests to other vaults"

icon = ""

long_description = """
xxx
"""
