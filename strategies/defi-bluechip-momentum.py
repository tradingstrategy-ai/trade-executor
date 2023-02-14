import datetime
from collections import Counter
from typing import Dict, List

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
    load_all_data
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradeexecutor.strategy.weighting import weight_by_1_slash_n

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
    (ChainId.ethereum, "uniswap-v2", "WETH", "USD"),  # ETH
    (ChainId.ethereum, "uniswap-v2", "AAVE", "ETH"),  # AAVE
    (ChainId.ethereum, "uniswap-v2", "UNI", "ETH"),  # UNI
    (ChainId.ethereum, "uniswap-v2", "CRV", "ETH"),  # Curve
    (ChainId.ethereum, "sushiswap", "SUSHI", "ETH"),  # Sushi
    (ChainId.bsc, "pancakeswap-v2", "WBNB", "BUSD"),  # BNB
    (ChainId.bsc, "pancakeswap-v2", "Cake", "BUSD"),  # Cake
    (ChainId.polygon, "quickswap", "WMATIC", "USDC"),  # Matic
    (ChainId.avalanche, "trader-joe", "WAVAX", "USDC"),  # Avax
    (ChainId.avalanche, "trader-joe", "JOE", "WAVAX"),  # TraderJoe
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

    # The time range end is the current candle
    # The time range start is 2 * 4 hours back, and turn the range
    # exclusive instead of inclusive
    start = timestamp - momentum_lookback_period - datetime.timedelta(minutes=59)
    end = timestamp

    candle_universe = universe.candles
    pair_universe = universe.pairs
    candle_data = candle_universe.iterate_samples_by_pair_range(start, end)

    # Track the data issues so we can later use this information
    # in diagnosing strategy issues
    issue_tracker = Counter({
        "lacks_open_and_close_in_momentum_window": 0,
        "non_positive_momentum": 0,
        "accepted_alpha_candidates": 0,
    })

    # Iterate over all candles for all pairs in this timestamp (ts)
    for pair_id, pair_df in candle_data:

        first_candle = pair_df.iloc[0]
        last_candle = pair_df.iloc[-1]

        # How many candles we are going to evaluate
        candle_count = len(pair_df)

        pair_momentum_window = last_candle["timestamp"] - first_candle["timestamp"]
        if pair_momentum_window < momentum_lookback_period:
            # This trading pair does not have data for this window,
            # ignore the pair and mark it as a problem
            issue_tracker["lacks_open_and_close_in_momentum_window"] += 1
            continue

        open = first_candle["open"]
        close = last_candle["close"]

        # DEXPair instance contains more data than internal TradingPairIdentifier
        # we use to store this pair across the strategy
        dex_pair = pair_universe.get_pair_by_id(pair_id)
        pair = translate_trading_pair(dex_pair)

        # We define momentum as how many % the trading pair price gained during
        # the momentum window
        momentum = (close - open) / open
        momentum = max(0, momentum)

        # This pair has positive momentum, check if it has enough available liquidity
        if momentum <= 0:
            issue_tracker["non_positive_momentum"] += 1
            continue

        alpha_model.set_signal(
            pair,
            momentum,
            stop_loss=0,
        )

        issue_tracker["accepted_alpha_candidates"] += 1

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
    alpha_model.calculate_target_positions(portfolio_target_value)

    # Shift portfolio from current positions to target positions
    # determined by the alpha signals (momentum)
    trades = alpha_model.generate_adjustment_trades_and_update_stop_losses(position_manager)

    # Record alpha model state so we can later visualise our alpha model thinking better
    state.visualisation.add_calculations(timestamp, alpha_model.to_dict())

    return trades


def create_trading_universe(
        ts: datetime.datetime,
        client: Client,
        execution_context: ExecutionContext,
        universe_options: UniverseOptions,
) -> TradingStrategyUniverse:
    """Creates the trading universe where the strategy trades.

    We reload candle data for each cycle.
    """

    assert execution_context.mode == ExecutionMode.backtesting, f"Live trading not supported"

    # Load all datas we can get for our candle time bucket
    dataset = load_all_data(
        client,
        trading_strategy_cycle.to_timebucket(),
        execution_context,
        universe_options,
    )

    # Filter down the dataset to the pairs we specified
    universe = TradingStrategyUniverse.create_multipair_universe(
        dataset,
        [chain_id],
        [exchange_slug],
    )

    return universe
