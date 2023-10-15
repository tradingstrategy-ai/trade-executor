"""Test backtesting where decide_trades and create_universe functions are passed directly.

"""
import logging
import random
import datetime
from typing import List, Dict

import pytest

import pandas as pd
from pandas_ta.overlap import ema

from tradeexecutor.analysis.trade_analyser import build_trade_analysis, expand_timeline, expand_timeline_raw, TimelineRowStylingMode, TradeAnalysis, TradeSummary
from tradeexecutor.backtest.backtest_runner import run_backtest_inline
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.visualisation import PlotKind
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, \
    create_pair_universe_from_code
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange, generate_simple_routing_model
from tradeexecutor.testing.synthetic_price_data import generate_ohlcv_candles
from tradeexecutor.visual.benchmark import visualise_benchmark
from tradingstrategy.candle import GroupedCandleUniverse
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.state.state import State
from tradingstrategy.universe import Universe
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.strategy_module import pregenerated_create_trading_universe
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.strategy_type import StrategyType
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.visual.equity_curve import calculate_equity_curve, calculate_returns
from tradeexecutor.analysis.advanced_metrics import visualise_advanced_metrics, AdvancedMetricsMode
from tradeexecutor.analysis.multipair import analyse_multipair
from tradeexecutor.analysis.multipair import format_multipair_summary

import datetime
import pandas as pd

from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.strategy_module import StrategyType, TradeRouting, ReserveCurrency

from tradeexecutor.strategy.trading_strategy_universe import translate_trading_pair
from typing import List, Dict, Counter

from tradingstrategy.universe import Universe
from tradeexecutor.strategy.weighting import weight_by_1_slash_n
from tradeexecutor.strategy.alpha_model import AlphaModel
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.state.state import State
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse

from tradingstrategy.client import Client

from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.strategy.trading_strategy_universe import load_partial_data, load_all_data
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.execution_context import ExecutionMode
from tradeexecutor.strategy.universe_model import UniverseOptions



# relative tolerance for floating point tests
APPROX_REL = 1e-6


# Tell what trade execution engine version this strategy needs to use
# NOTE: this setting has currently no effect
TRADING_STRATEGY_ENGINE_VERSION = "0.2"
BACKTEST_START=datetime.datetime(2022,1,1)
BACKTEST_END=datetime.datetime(2023,1,1)
INITIAL_CASH=5000


# What kind of strategy we are running.
# This tells we are going to use
trading_strategy_type = StrategyType.managed_positions

# How our trades are routed.
# PancakeSwap basic routing supports two way trades with BUSD
# and three way trades with BUSD-BNB hop.
trade_routing = TradeRouting.generic_routing

# Set cycle to 7 days and look back the momentum of the previous candle
trading_strategy_cycle = CycleDuration.cycle_1h
lookback_data_granularity = TimeBucket.h1
momentum_lookback_period = datetime.timedelta(hours=7)

# Hold top 3 coins for every cycle
max_assets_in_portfolio = 7

# Leave 90% cash buffer
value_allocated_to_positions = 0.05

# Set 33% stop loss over mid price
stop_loss = 0.5

# Set 5% take profit over mid price
take_profit = 1.02

# The weekly price must be up 2.5% for us to take a long position
minimum_signal_threshold = 0.0001

returns_average = 4

# Don't bother with trades that would move position
# less than 300 USD
minimum_rebalance_trade_threshold = 3

# Use hourly candles to trigger the stop loss
stop_loss_data_granularity = TimeBucket.h1

# Strategy keeps its cash in USDC
reserve_currency = ReserveCurrency.usdc

# The duration of the backtesting period
start_at = datetime.datetime(2023, 1, 1)
#start_at = datetime.datetime(2022, 6, 1)

#end_at = datetime.datetime(2023, 1, 1)
end_at = datetime.datetime(2023, 6, 1)

# Start with 10,000 USD
initial_deposit = 10_000

# We trade on Polygon
CHAIN_ID = ChainId.polygon

# List of trading pairs that we consider "DeFi blueschips" for this strategy
# For token ordering, wrappign see https://tradingstrategy.ai/docs/programming/market-data/trading-pairs.html
pairs = (
    (ChainId.polygon, "quickswap", "WMATIC", "USDC"),  # Matic
    #(ChainId.polygon, "quickswap", "KLIMA", "USDC"), 
    #(ChainId.polygon, "quickswap", "CHC", "USDC"), 
    (ChainId.polygon, "uniswap-v3", "WMATIC", "USDC", 0.0005), 
    (ChainId.polygon, "quickswap", "SAND", "WMATIC"), 
    (ChainId.polygon, "quickswap", "WBTC", "WETH"), 
    #(ChainId.polygon, "quickswap", "NASMG", "WMATIC"), 
    (ChainId.polygon, "uniswap-v3", "WETH", "USDC", 0.0005),  
    #(ChainId.polygon, "quickswap", "GHST", "USDC"), 
    #(ChainId.polygon, "sushi", "NCT", "USDC"), 
    #(ChainId.polygon, "quickswap", "QI", "WMATIC"), 
    (ChainId.polygon, "quickswap", "mOCEAN", "WMATIC"), 
    #(ChainId.polygon, "quickswap", "ICE", "USDC"), 
    #(ChainId.polygon, "sushi", "GIDDY", "USDC"), 
    (ChainId.polygon, "quickswap", "DG", "WMATIC"), 
    #(ChainId.polygon, "quickswap", "ORBS", "USDC"), 
)


def decide_trades(
        timestamp: pd.Timestamp,
        universe: Universe,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: Dict) -> List[TradeExecution]:

    # Create a position manager helper class that allows us easily to create
    # opening/closing trades for different positions
    position_manager = PositionManager(timestamp, universe, state, pricing_model)

    alpha_model = AlphaModel(timestamp)

    # Watch out for the inclusive range and include and avoid peeking in the future
    adjusted_timestamp = timestamp - pd.Timedelta(seconds=1)
    start = adjusted_timestamp - momentum_lookback_period - datetime.timedelta(seconds=1)
    end = adjusted_timestamp 

    candle_universe = universe.candles
    pair_universe = universe.pairs

    # Get candle data for all candles, inclusive time range
    candle_data = candle_universe.iterate_samples_by_pair_range(start, end)

    # Iterate over all candles for all pairs in this timestamp (ts)
    for pair_id, pair_df in candle_data:

        # We should have candles for range start - end,
        # where end is the current strategy cycle timestamp 
        # and start is one week before end.
        # Because of sparse data we may have 0, 1 or 2 candles
        last_candle = pair_df.iloc[-1]

        # How many candles we are going to evaluate
        candle_count = len(pair_df)

        #assert last_candle["timestamp"] <= timestamp, "Something wrong with the data - we should not be able to peek the candle of the current timestamp, but always use the previous candle"

        # DEXPair instance contains more data than internal TradingPairIdentifier
        # we use to store this pair across the strategy
        dex_pair = pair_universe.get_pair_by_id(pair_id)
        pair = translate_trading_pair(dex_pair)

        # We define momentum as how many % the trading pair price gained during
        # the momentum window
        signals = [-1]

        for timestamp, day in pair_df.iterrows():
            open = day["open"]
            close = day["close"]
            dayreturn = (close - open) / open
            signals.append(dayreturn)

        signal = sorted(signals, reverse=True)[:returns_average]
        signal = sum(signal) / len(signal)

        # This pair has not positive momentum,
        # we only buy when stuff goes up
        if signal <= minimum_signal_threshold:
            continue

        alpha_model.set_signal(
            pair,
            signal,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    # Select max_assets_in_portfolio assets in which we are going to invest
    # Calculate a weight for ecah asset in the portfolio using 1/N method based on the raw signal
    alpha_model.select_top_signals(max_assets_in_portfolio)

    alpha_model.assign_weights(method=weight_by_1_slash_n)
    alpha_model.normalise_weights()

    # Load in old weight for each trading pair signal,
    # so we can calculate the adjustment trade size
    alpha_model.update_old_weights(state.portfolio)

    # Calculate how much dollar value we want each individual position to be on this strategy cycle,
    # based on our total available equity
    portfolio = position_manager.get_current_portfolio()
    portfolio_target_value = portfolio.get_total_equity() * value_allocated_to_positions
    alpha_model.calculate_target_positions(position_manager, portfolio_target_value)

    # Shift portfolio from current positions to target positions
    # determined by the alpha signals (momentum)
    trades = alpha_model.generate_rebalance_trades_and_triggers(
        position_manager,
        min_trade_threshold=minimum_rebalance_trade_threshold,  # Don't bother with trades under 300 USD
    )

    # Record alpha model state so we can later visualise our alpha model thinking better
    state.visualisation.add_calculations(timestamp, alpha_model.to_dict())

    return trades

def create_trading_universe(
        ts: datetime.datetime,
        client: Client,
        execution_context: ExecutionContext,
        universe_options: UniverseOptions,
) -> TradingStrategyUniverse:

    assert not execution_context.mode.is_live_trading(), \
        f"Only strategy backtesting supported, got {execution_context.mode}"


    # Load data for our trading pair whitelist
    dataset = load_partial_data(
        client=client,
        time_bucket=lookback_data_granularity,
        pairs=pairs,
        execution_context=execution_context,
        universe_options=universe_options,
        stop_loss_time_bucket=stop_loss_data_granularity,
        start_at=start_at,
        end_at=end_at,
    )

    # Filter down the dataset to the pairs we specified
    universe = TradingStrategyUniverse.create_multichain_universe_by_pair_descriptions(
        dataset,
        pairs,
        reserve_token_symbol="USDC",  # Pick any USDC - does not matter as we do not route
        
    )

    return universe


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request, mute_requests=False)


@pytest.fixture(scope = "module")
def client(
    
):
    pass


@pytest.fixture(scope="module")
def backtest_result(
    persistent_test_client
) -> tuple[State, TradingStrategyUniverse, dict]:


    # Run the test
    state, universe, debug_dump = run_backtest_inline(
        start_at=start_at,
        end_at=end_at,
        client=persistent_test_client,  # None of downloads needed, because we are using synthetic data
        cycle_duration=CycleDuration.cycle_1d,  # Override to use 24h cycles despite what strategy file says
        decide_trades=decide_trades,
        create_trading_universe=create_trading_universe,
        initial_deposit=10_000,
        reserve_currency=reserve_currency,
        trade_routing=trade_routing,
        log_level=logging.WARNING,
    )

    return state, universe, debug_dump


def test_run_inline_backtest(
        logger: logging.Logger,
        backtest_result: tuple[State, TradingStrategyUniverse, dict],
    ):
    """Run the strategy backtest using inline decide_trades function.
    """

    state, universe, debug_dump = backtest_result

    assert len(debug_dump) == 152


@pytest.fixture(scope = "module")
def analysis(
    backtest_result: tuple[State, TradingStrategyUniverse, dict]
) -> TradeAnalysis:
    state, universe, debug_dump = backtest_result
    analysis = build_trade_analysis(state.portfolio)

    return analysis


def test_summary(
    backtest_result: tuple[State, TradingStrategyUniverse, dict],
    analysis: TradeAnalysis
) -> TradeSummary:

    state, universe, debug_dump = backtest_result

    summary = analysis.calculate_summary_statistics(state = state)

    # Should not cause exception
    summary.to_dataframe()
    
    
    # should not cause an exception
    multipair_summary = analyse_multipair(state)
    format_multipair_summary(multipair_summary)
    
    # should have data for 7 pairs
    assert len(multipair_summary) == 7
    
    return summary
