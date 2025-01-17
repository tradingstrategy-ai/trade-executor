"""PancakeSwap v2 momentum strategy build on the top of the new trading framework.

This is "alpha model" strategy that predicts the alpha (price increase)
of multiple tokens based on their past behavior.
"""

import datetime
import enum
from collections import Counter
from typing import Dict, List, Optional

import pandas as pd


from tradeexecutor.ethereum.routing_data import get_pancake_default_routing_parameters

from tradeexecutor.strategy.pandas_trader.rebalance import rebalance_portfolio_old
from tradeexecutor.strategy.weights import normalise_weights, weight_by_1_slash_n
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.utils.price import is_legit_price_value
from tradingstrategy.chain import ChainId
from tradingstrategy.liquidity import LiquidityDataUnavailable
from tradingstrategy.pair import DEXPair
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.strategy_type import StrategyType
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, load_all_data, \
    translate_trading_pair
from tradingstrategy.client import Client


trading_strategy_name = "PancakeSwap momentum v2"

# Tell what trade execution engine version this strategy needs to use
trading_strategy_engine_version = "0.1"

# What kind of strategy we are running.
# This tells we are going to use
trading_strategy_type = StrategyType.alpha_model

# How our trades are routed.
# PancakeSwap basic routing supports two way trades with BUSD
# and three way trades with BUSD-BNB hop.
trade_routing = TradeRouting.pancakeswap_busd

# How often the strategy performs the decide_trades cycle.
# We do it for every 16h.
trading_strategy_cycle = CycleDuration.cycle_16h

# Strategy keeps its cash in BUSD
reserve_currency = ReserveCurrency.busd

# Time bucket for our candles
candle_time_bucket = TimeBucket.h4

# Which chain we are trading
chain_id = ChainId.bsc

# Which exchange we are trading on.
exchange_slug = "pancakeswap-v2"

# Use 4h candles for trading
candle_time_frame = TimeBucket.h4

# How much of portfolio's total value is allocated
# to positions (rest is kept in cash)
value_allocated_to_positions = 0.50

# How many assets we fit to our portfolio once
max_assets_in_portfolio = 5

# How far back we look the momentum.
momentum_lookback_period = pd.Timedelta(hours=32)

# What is the liquidity risk we are willing to accept (USD)
risk_min_liquidity_threshold = 100_000

# If the trade would shift the position value less
# than this USD, then don't do an unnecessary trade
min_position_update_threshold = 5.0


class RiskAssessment(enum.Enum):
    """Potential risk flags for a trading pair."""
    accepted_risk = "risk_accepted"
    pair_info_missing = "risk_pair_info_missing"
    blacklisted = "risk_blacklisted"
    token_tax = "risk_token_tax"
    lack_of_liquidity = "risk_lack_of_liquidity"
    bad_price_units = "risk_bad_price_units"


def assess_risk(
        state: State,
        pair: DEXPair,
        price: float,
        liquidity: float) -> RiskAssessment:
    """Do the risk check for the trading pair if it accepted to our alpha model.

    - There needs to be enough liquidity

    - The price unit must look sensible
    """

    executor_pair = translate_trading_pair(pair)

    # Skip any trading pair with machine generated tokens
    # or otherwise partial looking info
    if not executor_pair.has_complete_info():
        return RiskAssessment.pair_info_missing

    # Wast this pair blacklisted earlier by the strategy itself
    if not state.is_good_pair(executor_pair):
        return RiskAssessment.blacklisted

    # This token is marked as not tradeable, so we don't touch it
    if (pair.buy_tax != 0) or (pair.sell_tax != 0) or (pair.transfer_tax != 0):
        return RiskAssessment.token_tax

    # The pair does not have enough liquidity for us to enter
    if liquidity < risk_min_liquidity_threshold:
        return RiskAssessment.lack_of_liquidity

    # The price value does not seem legit
    # and might have floating point issues
    if is_legit_price_value(price):
        return RiskAssessment.bad_price_units

    return RiskAssessment.accepted_risk


def filter_duplicate_base_tokens(self, alpha_signals: Counter, debug_data: dict) -> Counter:
    """Filter duplicate alpha signals for trading pairs sharing the same base token.

    This is because on DEX markets, the base token may trade with several quote tokens.

    For example, we can end up holding BNB by buying
    - WBNB-BUSD
    - WBNB-USDT

    For the portfolio, it does not matter which route we ended up buying the token
    (although this matters for the price impact).

    We use the already resolved pair data to check for the duplicates.
    We'll just filter out the second entry (worse alpha).
    """
    accumulated_quote_tokens = set()
    filtered_alpha = Counter()
    for pair_id, alpha in alpha_signals.most_common():
        pair: DEXPair = debug_data[pair_id]["pair"]
        base_token = pair.base_token_symbol
        if base_token in accumulated_quote_tokens:
            continue
        filtered_alpha[pair_id] = alpha
        accumulated_quote_tokens.add(base_token)
    return filtered_alpha



def decide_trades(
        timestamp: pd.Timestamp,
        universe: Universe,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: Dict) -> List[TradeExecution]:

    # Create a position manager helper class that allows us easily to create
    # opening/closing trades for different positions
    position_manager = PositionManager(timestamp, universe, state, pricing_model)

    # pair id -> how much alpha it has
    alpha_signals = Counter()

    # The time range end is the current candle
    # The time range start is 2 * 4 hours back, and turn the range
    # exclusive instead of inclusive
    start = timestamp - momentum_lookback_period - datetime.timedelta(minutes=59)
    end = timestamp

    candle_universe = universe.candles
    pair_universe = universe.pairs
    liquidity_universe = universe.liquidity
    candle_data = candle_universe.iterate_samples_by_pair_range(start, end)

    # Track number of problematic trading pairs
    # for this trade cycle
    issue_tracker = Counter({
        "lacks_open_and_close_in_momentum_window": 0,
        "liquidity_information_missing": 0,
        "non_positive_momentum": 0,
        "accepted_alpha_candidates": 0,
    })

    # Expose pair specific debug data to the
    # research
    pair_debug_data = {}

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

        open = first_candle["open"]  # QStrader data frames are using capitalised version of OHLCV core variables
        close = last_candle["close"]  # QStrader data frames are using capitalised version of OHLCV core variables

        pair = pair_universe.get_pair_by_id(pair_id)

        if is_legit_price_value(close):
            # This trading pair is too funny that we do not want to play with it
            issue_tracker["price_unit_problematic"] += 1
            continue

        # We define momentum as how many % the trading pair price gained during
        # the momentum window
        momentum = (close - open) / open
        momentum = max(0, momentum)

        # This pair has positive momentum, check if it has enough available liquidity
        available_liquidity_for_pair = 0
        if momentum > 0:

            # Check for the liquidity requirement
            try:
                available_liquidity_for_pair = liquidity_universe.get_closest_liquidity(pair_id, timestamp)
            except LiquidityDataUnavailable as e:
                # There might be holes in the data, because BSC network not syncing properly,
                # BSC blockchain was halted or because BSC nodes themselves had crashed.
                # In this case, we just assume the liquidity was zero and don't backtest.
                issue_tracker["liquidity_information_missing"] += 1

        else:
            issue_tracker["non_positive_momentum"] += 1
            continue

        risk = assess_risk(
            state,
            pair,
            close,
            available_liquidity_for_pair)

        if risk == RiskAssessment.accepted_risk:
            # Do candle check only after we know the pair is "good" liquidity wise
            # and should have candles
            candle_count = len(pair_df)
            alpha_signals[pair_id] = momentum

            issue_tracker["accepted_alpha_candidates"] += 1
        else:
            # Delette pair from the alpha set because of observed risk
            del alpha_signals[pair_id]

            # Track the number of risk issues we have detectd in this cycle
            issue_tracker[str(risk.value)] += 1

        # Extra debug details are available for pairs for which a buy decision can be made
        pair_debug_data[pair_id] = {
            "pair": pair,
            "open": open,
            "close": close,
            "momentum": momentum,
            "liquidity": available_liquidity_for_pair,
            "candle_count": candle_count,
        }

    alpha_signals = alpha_signals.most_common(max_assets_in_portfolio)

    weights = normalise_weights(weight_by_1_slash_n(alpha_signals))

    portfolio = position_manager.get_current_portfolio()
    portfolio_target_value = portfolio.calculate_total_equity() * value_allocated_to_positions

    # Shift portfolio from current positions to target positions
    # determined by the alpha signals (momentum)
    trades = rebalance_portfolio_old(
        position_manager,
        weights,
        portfolio_target_value,
        min_position_update_threshold
    )

    return trades


def create_trading_universe(
        ts: datetime.datetime,
        client: Client,
        execution_context: ExecutionContext,
        candle_time_frame_override: Optional[TimeBucket] = None,
) -> TradingStrategyUniverse:
    """Creates the trading universe where the strategy trades.

    We reload candle data for each cycle.
    """

    # Load all datas we can get for our candle time bucket
    dataset = load_all_data(
        client,
        candle_time_frame_override or candle_time_bucket,
        execution_context)

    routing_parameters = get_pancake_default_routing_parameters(ReserveCurrency.busd)

    universe = TradingStrategyUniverse.create_multipair_universe(
        dataset,
        [chain_id],
        [exchange_slug],
        quote_tokens=routing_parameters["quote_token_addresses"],
        reserve_token=routing_parameters["reserve_token_address"],
        factory_router_map=routing_parameters["factory_router_map"],
    )

    return universe
