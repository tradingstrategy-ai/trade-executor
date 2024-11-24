import datetime
from collections import Counter
from typing import Dict, List
import logging

import pandas as pd

from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.alpha_model import AlphaModel
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.strategy_module import StrategyType, TradeRouting, ReserveCurrency
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, translate_trading_pair, \
    load_partial_data
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradeexecutor.strategy.weighting import weight_by_1_slash_n


logger = logging.getLogger(__name__)

# Tell what trade execution engine version this strategy needs to use
trading_strategy_engine_version = "0.1"

# What kind of strategy we are running.
# This tells we are going to use
trading_strategy_type = StrategyType.managed_positions

# We ignore trading routing, price impact, etc. in this backtesting
trade_routing = TradeRouting.ignore

# Rebalance weekly
trading_strategy_cycle = CycleDuration.cycle_7d

momentum_lookback_period = datetime.timedelta(days=7)

# Hold top 3 coins for every cycle
max_assets_in_portfolio = 3

# Leave 20% cash buffer
value_allocated_to_positions = 0.80

# Set 5% midprice stop loss
stop_loss = 0.95

# Set 5% mid price take profit
take_profit = 1.05

# Which candle time frame we use for backtesting stop loss triggers
stop_loss_data_granularity = TimeBucket.h1

# Strategy keeps its cash in USDC
reserve_currency = ReserveCurrency.usdc

# The duration of the backtesting period
start_at = datetime.datetime(2021, 6, 1)
end_at = datetime.datetime(2023, 1, 1)

# Start with 10,000 USD
initial_deposit = 10_000

# List of trading pairs that we consider "DeFi blueschips" for this strategy
# For token ordering, wrappign see https://tradingstrategy.ai/docs/programming/market-data/trading-pairs.html
pairs = (
    (ChainId.ethereum, "uniswap-v2", "WETH", "USDC"),  # ETH
    (ChainId.ethereum, "uniswap-v2", "AAVE", "WETH"),  # AAVE
    (ChainId.ethereum, "uniswap-v2", "UNI", "WETH"),  # UNI
    (ChainId.ethereum, "uniswap-v2", "CRV", "WETH"),  # Curve
    (ChainId.ethereum, "sushi", "SUSHI", "WETH"),  # Sushi
    (ChainId.bsc, "pancakeswap-v2", "WBNB", "BUSD"),  # BNB
    (ChainId.bsc, "pancakeswap-v2", "Cake", "BUSD"),  # Cake
    (ChainId.polygon, "quickswap", "WMATIC", "USDC"),  # Matic
    (ChainId.avalanche, "trader-joe", "WAVAX", "USDC"),  # Avax
    # Price adjust problems
    #  (ChainId.avalanche, "trader-joe", "JOE", "WAVAX"),  # TraderJoe
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
        first_candle = pair_df.iloc[0]
        last_candle = pair_df.iloc[-1]

        # How many candles we are going to evaluate
        candle_count = len(pair_df)

        assert last_candle["timestamp"] < timestamp, "Something wrong with the data - we should not be able to peek the candle of the current timestamp, but always use the previous candle"

        open = last_candle["open"]
        close = last_candle["close"]

        # DEXPair instance contains more data than internal TradingPairIdentifier
        # we use to store this pair across the strategy
        dex_pair = pair_universe.get_pair_by_id(pair_id)
        pair = translate_trading_pair(dex_pair)

        # We define momentum as how many % the trading pair price gained during
        # the momentum window
        momentum = (close - open) / open
        momentum = max(0, momentum)

        # This pair has not positive momentum,
        # we only buy when stuff goes up
        if momentum <= 0:
            continue

        alpha_model.set_signal(
            pair,
            momentum,
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

    # Added for tests
    diagnostics = alpha_model.get_flag_diagnostics_data()
    assert isinstance(diagnostics, dict)

    # Shift portfolio from current positions to target positions
    # determined by the alpha signals (momentum)
    trades = alpha_model.generate_rebalance_trades_and_triggers(position_manager)

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
        time_bucket=trading_strategy_cycle.to_timebucket(),
        pairs=pairs,
        execution_context=execution_context,
        universe_options=universe_options,
        liquidity=False,
        stop_loss_time_bucket=stop_loss_data_granularity,
        start_at=start_at,
        end_at=end_at,
    )

    # Filter down the dataset to the pairs we specified
    universe = TradingStrategyUniverse.create_multichain_universe_by_pair_descriptions(
        dataset,
        pairs,
        reserve_token_symbol="USDC"  # Pick any USDC - does not matter as we do not route
    )

    return universe
