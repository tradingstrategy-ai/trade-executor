import datetime
from typing import List, Dict
import logging
import pandas as pd

from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.alpha_model import AlphaModel, format_signals
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, load_trading_and_lending_data
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradeexecutor.strategy.weighting import weight_by_1_slash_n
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.timebucket import TimeBucket
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.strategy_module import StrategyType, TradeRouting, ReserveCurrency

logger = logging.getLogger(__name__)

# Tell what trade execution engine version this strategy needs to use
trading_strategy_engine_version = "0.3"

# What kind of strategy we are running.
# This tells we are going to use
trading_strategy_type = StrategyType.managed_positions

# How our trades are routed.
# PancakeSwap basic routing supports two way trades with BUSD
# and three way trades with BUSD-BNB hop.
trade_routing = TradeRouting.ignore

# Set cycle to 7 days and look back the momentum of the previous candle
trading_strategy_cycle = CycleDuration.cycle_1d
momentum_lookback_period = datetime.timedelta(days=7)

# Hold top 3 coins for every cycle
max_assets_in_portfolio = 3

# Leave 20% cash buffer
value_allocated_to_positions = 0.80

# Set 33% stop loss over mid price
stop_loss = None

# Set 5% take profit over mid price
take_profit = 1.07
#take_profit = None

# The weekly price must be up 2.5% for us to take a long position
positive_mometum_threshold = 0.001

negative_mometum_threshold = -0.06

# Don't bother with trades that would move position
# less than 300 USD
minimum_rebalance_trade_threshold = 300

# Strategy keeps its cash in USDC
reserve_currency = ReserveCurrency.usdc

# The duration of the backtesting period
backtest_start = datetime.datetime(2022, 1, 1)
backtest_end = datetime.datetime(2023, 11, 1)

# Start with 10,000 USD
initial_cash = 10_000




def decide_trades(
        timestamp: pd.Timestamp,
        strategy_universe: TradingStrategyUniverse,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: Dict
) -> List[TradeExecution]:

    cycle = cycle_debug_data["cycle"]
    if (cycle + 3) % 7 != 0:
        return []

    # Create a position manager helper class that allows us easily to create
    # opening/closing trades for different positions
    position_manager = PositionManager(timestamp, strategy_universe, state, pricing_model)

    alpha_model = AlphaModel(timestamp)

    # Watch out for the inclusive range and include and avoid peeking in the future
    adjusted_timestamp = timestamp - pd.Timedelta(seconds=1)
    start = adjusted_timestamp - momentum_lookback_period - datetime.timedelta(seconds=1)
    end = adjusted_timestamp

    data_universe = strategy_universe.data_universe

    # Get candle data for all candles, inclusive time range
    candle_data = data_universe.candles.iterate_samples_by_pair_range(start, end)

    # Iterate over all candles for all pairs in this timestamp (ts)
    for pair_id, pair_df in candle_data:

        # Workaround LINK issues
        # Need at least 4 days worth of data
        if len(pair_df) < 4:
            continue

        first_candle = pair_df.iloc[0]
        last_candle = pair_df.iloc[-1]
        open = first_candle["open"]
        close = last_candle["close"]

        # DEXPair instance contains more data than internal TradingPairIdentifier
        # we use to store this pair across the strategy
        pair = strategy_universe.get_trading_pair(pair_id)

        # We define momentum as how many % the trading pair price gained during
        # the momentum window
        momentum = (close - open) / open

        # This pair has not positive momentum,
        # we only buy when stuff goes up
        if momentum >= positive_mometum_threshold:
            alpha_model.set_signal(
                pair,
                momentum,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
        elif momentum <= negative_mometum_threshold:
            pass
            # if strategy_universe.can_open_short(timestamp, pair):
            #    # Only open a short if we have lending markets available at this point
            #    alpha_model.set_signal(
            #        pair,
            #        momentum,
            #        stop_loss=stop_loss,
            #        take_profit=take_profit,
            #        leverage=1.0,
            #    )
        else:
            # Momentum is ~0,
            # not worth of a signal
            pass

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
    portfolio_target_value = portfolio.calculate_total_equity() * value_allocated_to_positions
    alpha_model.calculate_target_positions(position_manager, portfolio_target_value)

    signal_df = format_signals(alpha_model)
    logger.info("Cycle %s signals:\n%s", timestamp, signal_df)

    # Shift portfolio from current positions to target positions
    # determined by the alpha signals (momentum)
    trades = alpha_model.generate_rebalance_trades_and_triggers(
        position_manager,
        min_trade_threshold=minimum_rebalance_trade_threshold,  # Don't bother with trades under XXX USD
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
    # We limit ourselves to price feeds on Uniswap v3 and Quickswap on Polygon,
    # as there are multiple small or dead DEXes on Polygon
    # which also have price feeds but not interesting liquidity
    dataset = load_trading_and_lending_data(
        client,
        execution_context=execution_context,
        universe_options=universe_options,
        # Ask for all Polygon data
        chain_id=ChainId.polygon,
        exchange_slugs={"uniswap-v3"},
        reserve_assets={"USDC"},
        asset_ids={"LINK", "WMATIC", "WETH"},
        trading_fee=0.0005,
        time_bucket=TimeBucket.d4,
        stop_loss_time_bucket=TimeBucket.h1,
    )

    # Filter down the dataset to the pairs we specified
    universe = TradingStrategyUniverse.create_from_dataset(dataset)

    return universe



