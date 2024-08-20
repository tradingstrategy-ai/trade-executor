"""Polygon strategy based on rolling ratio indicator. Long only.

To backtest this strategy module locally:

.. code-block:: console

    source scripts/set-latest-tag-gcp.sh
    docker-compose run enzyme-polygon-eth-rolling-ratio backtest

Or:

.. code-block:: console

    trade-executor \
        backtest \
        --strategy-file=strategy/enzyme-polygon-eth-rolling-ratio.py \
        --trading-strategy-api-key=$TRADING_STRATEGY_API_KEY

"""

import datetime

from tradingstrategy.chain import ChainId
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradingstrategy.pair import HumanReadableTradingPairDescription
from tradingstrategy.timebucket import TimeBucket
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.trading_strategy_universe import load_partial_data
from tradingstrategy.client import Client
from tradeexecutor.utils.binance import create_binance_universe
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradingstrategy.lending import LendingProtocolType, LendingReserveDescription
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet, IndicatorSource
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse

from tradeexecutor.state.visualisation import PlotKind
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.utils.crossover import contains_cross_over, contains_cross_under
from tradeexecutor.strategy.tag import StrategyTag


trading_strategy_engine_version = "0.5"


class Parameters:

    id = "enzyme-polygon-eth-rolling-ratio" # Used in cache paths

    cycle_duration = CycleDuration.cycle_1d
    candle_time_bucket = TimeBucket.d1
    allocation = 0.98
    credit_allocation = 0.99
    
    # rolling mean and standard deviation lengths, used to calculate z-score
    rolling_short_mean = 7
    rolling_long_mean = 65
    rolling_std = 18
    upper_threshold = 1.32 # number standard deviations where z-score is higher than zero to enter a position
    max_upper_threshold = 1.92 # number maximum standard deviations allowed to enter a position

    stop_loss_pct = 0.88
    take_profit_pct = 1.14

    #
    # Live trading only
    #
    chain_id = ChainId.polygon
    routing = TradeRouting.default  
    required_history_period = datetime.timedelta(hours=70)
    
    #
    # Backtesting only
    #

    # Use Binance data in backtesting,
    # We get a longer, more meaningful, history but no credit simulation.
    binance_data = True

    if binance_data:
        backtest_start = datetime.datetime(2020, 1, 1)
        backtest_end = datetime.datetime(2024, 7, 15)

    else:
        # dex dates
        backtest_start = datetime.datetime(2022, 10, 1)
        backtest_end = datetime.datetime(2024, 7, 15)
    
    stop_loss_time_bucket = TimeBucket.d1
    backtest_trading_fee = 0.0005
    initial_cash = 10_000



def get_strategy_trading_pairs(mode: ExecutionMode) -> list[HumanReadableTradingPairDescription]:
    """Get trading pairs the strategy uses
    
    - Different options for backtesting
    """
    use_binance = mode.is_backtesting() and Parameters.binance_data 

    if use_binance:
        trading_pairs = [
            (ChainId.centralised_exchange, "binance", "BTC", "USDT"),
            (ChainId.centralised_exchange, "binance", "ETH", "USDT"),
        ]
    else:
        trading_pairs = [
            (ChainId.polygon, "quickswap", "WBTC", "USDC", 0.0030),  # Deep liquidity
            (ChainId.polygon, "uniswap-v3", "WETH", "USDC", 0.0005),  # Deep liquidity
        ]

    return trading_pairs


def get_lending_reserves(mode: ExecutionMode) -> list[LendingReserveDescription]:
    """Get lending reserves the strategy needs."""
    
    use_binance = mode.is_backtesting() and Parameters.binance_data 

    if use_binance:
        # Credit interest is not available on Binance
       return []
    else:
        # and Aave v3 in live execution (more liquid market)
        lending_reserves = [
            (ChainId.polygon, LendingProtocolType.aave_v3, "USDC.e"),
        ]

    return lending_reserves


def create_trading_universe(
    timestamp: datetime.datetime,
    client: Client,
    execution_context: ExecutionContext,
    universe_options: UniverseOptions,
) -> TradingStrategyUniverse:
    """Create the trading universe.

    - In this example, we load all Binance spot data based on our Binance trading pair list.
    """
    trading_pairs = get_strategy_trading_pairs(execution_context.mode)
    lending_reserves = get_lending_reserves(execution_context.mode)

    use_binance = trading_pairs[0][0] == ChainId.centralised_exchange

    if use_binance:
        # Backtesting - load Binance data
        strategy_universe = create_binance_universe(
            [f"{p[2]}{p[3]}" for p in trading_pairs],
            candle_time_bucket=Parameters.candle_time_bucket,
            stop_loss_time_bucket=Parameters.stop_loss_time_bucket,
            start_at=universe_options.start_at,
            end_at=universe_options.end_at,
            trading_fee_override=Parameters.backtest_trading_fee,
            include_lending=False,
            forward_fill=True,
        )
    else:

        if execution_context.live_trading or execution_context.mode == ExecutionMode.preflight_check:
            start_at, end_at = None, None
            required_history_period=Parameters.required_history_period
        else:
            required_history_period = None
            start_at=universe_options.start_at
            end_at=universe_options.end_at

        dataset = load_partial_data(
            client,
            execution_context=execution_context,
            time_bucket=Parameters.candle_time_bucket,
            pairs=trading_pairs,
            universe_options=universe_options,
            start_at=start_at,
            end_at=end_at,
            stop_loss_time_bucket=Parameters.stop_loss_time_bucket,
            lending_reserves=lending_reserves,
            required_history_period=required_history_period
        )

        # Filter down to the single pair we are interested in
        strategy_universe = TradingStrategyUniverse.create_from_dataset(
            dataset,
            forward_fill=True,
        )
    return strategy_universe


# calculating the z-score of target token and other token ratio
def calculate_rolling_ratio(
    strategy_universe: TradingStrategyUniverse,
    short_ma: int,
    long_ma: int,
    std: int,
    execution_mode: ExecutionMode,
):
    trading_pairs = get_strategy_trading_pairs(execution_mode)
    pairs = strategy_universe.data_universe.pairs
    candles = strategy_universe.data_universe.candles

    target_pair = pairs.get_pair_by_human_description(trading_pairs[0])
    target_data = candles.get_candles_by_pair(target_pair)["close"]
    other_pair = pairs.get_pair_by_human_description(trading_pairs[1])
    other_data = candles.get_candles_by_pair(other_pair)["close"]

    ratios = target_data/other_data

    ratios_mavg_short = ratios.rolling(window=short_ma, center=False).mean()
    ratios_mavg_long = ratios.rolling(window=long_ma, center=False).mean()
    ratios_std = ratios.rolling(window=std, center=False).std()
    zscore = (ratios_mavg_short - ratios_mavg_long)/ratios_std
    
    return zscore

def create_indicators(
    timestamp: datetime.datetime | None,
    parameters: StrategyParameters,
    strategy_universe: TradingStrategyUniverse,
    execution_context: ExecutionContext
):
    indicators = IndicatorSet()

    indicators.add(
        "rolling_ratio",
        calculate_rolling_ratio,
        {
            "short_ma": parameters.rolling_short_mean,
            "long_ma": parameters.rolling_long_mean,
            "std": parameters.rolling_std,
            "execution_mode": execution_context.mode,
        },
        source=IndicatorSource.strategy_universe,
    )
    return indicators


def decide_trades(
    input: StrategyInput,
) -> list[TradeExecution]:
    # 
    # Decidion cycle setup.
    # Read all variables we are going to use for the decisions.
    #
    parameters: Parameters = input.parameters
    position_manager = input.get_position_manager()
    state = input.state
    timestamp = input.timestamp
    indicators = input.indicators
    strategy_universe = input.strategy_universe
    cash = position_manager.get_current_cash()
    trading_pairs = get_strategy_trading_pairs(input.execution_context.mode)
    lending_reserves = get_lending_reserves(input.execution_context.mode)

    pair = strategy_universe.get_pair_by_human_description(trading_pairs[1])

    target_price = indicators.get_price(trading_pairs[0])
    close_price = indicators.get_price(trading_pairs[1])

    trades = []

    # Setup asset allocation parameters
    use_credit = len(lending_reserves) > 0

    # If any of trading pairs enters to long position,
    # close our credit position
    credit_closed = False
    traded_this_cycle = False
    available_cash = cash
    ready = False

    z_score = indicators.get_indicator_value("rolling_ratio")

    if None in (z_score, target_price, close_price):
    # if not z_score or not target_price or not other_price:
        # Not enough historic data,
        # cannot make decisions yet
        return []

    ready = True

    # Check for open condition
    if state.portfolio.get_open_position_for_pair(pair) is None:        
        if parameters.upper_threshold < z_score < parameters.max_upper_threshold:
            # close credit supply position before opening a new long position
            if position_manager.is_any_credit_supply_position_open():
                current_pos = position_manager.get_current_credit_supply_position()
                new_trades = position_manager.close_credit_supply_position(current_pos)
                trades.extend(new_trades)
                # Est. available cash after all credit positions are closed
                available_cash += float(current_pos.get_quantity()) 

            trades += position_manager.open_spot(
                pair,
                value=available_cash * parameters.allocation,
                stop_loss_pct=parameters.stop_loss_pct,
                take_profit_pct=parameters.take_profit_pct,
            )

            traded_this_cycle = True

    if ready:

        # We have accumulatd enough data to make the first real (non credit) trading decision.
        # This allows us to have fair buy-and-hold vs backtest period comparison
        state.mark_ready(timestamp)

        # If we have any access cash or new deposit, move them to Aave
        if use_credit and not traded_this_cycle:
            cash_to_deposit = available_cash * parameters.credit_allocation
            new_trades = position_manager.add_cash_to_credit_supply(cash_to_deposit)
            trades += new_trades

    # Visualisations
    if input.is_visualisation_enabled():
        visualisation = state.visualisation
        visualisation.plot_indicator(timestamp, "rolling_ratio", PlotKind.technical_indicator_on_price, z_score)
        
    return trades


#
# Strategy metadata.
#
# Displayed in the user interface.
#

tags = {StrategyTag.live, StrategyTag.beta}

name = "ETH/BTC rolling ratio"

short_description = "A pair trading strategy for ETH/BTC"

icon = ""

long_description = """
# Strategy description

**Past performance is not indicative of future results**.

This is a statistical arbitrage strategy.

The strategy uses the normalized ratio of BTC price divided by ETH price. The fundamental assumption is that ETH price follows BTC price (high correlation), and thus when the normalized ratio value is far from equilibrium (zero), the ratio will revert to equilibrium.

- The strategy only takes long positions in ETH, as ETH has been observed to have more volatility and thus will be the asset to have higher price action in times where ratio is far from equilibrium.
- Short positions have been excluded in this version of the strategy as negative price actions happen quicker. This makes prediction of negative trends more difficult with the rolling ratio based model.
- The strategy calculates rolling Z-score with long term and short term moving averages, as well as rolling standard deviation of BTC/ETH price ratio
- Enters a long ETH position when rolling Z-score is in acceptable boundaries
- Exits positions when the profit threshold or a stop loss limit is reached

The strategy enables

- Capturing gains in bull markets (like July 2021 - December 2022, and October 2023 - February 2024)
- Capturing gains in neutral markets (like January 2023 - July 2023)
- Reducing max drawdowns in bear markets compared to pure buy&hold strategies with ETH or BTC

Furthermore

- The strategy deposits excess cash to Aave V3 USDC pool to gain interest on cash

## Assets and trading venues

- The strategy trades only spot market
- Trade only single asset: ETH
- The strategy keeps reserves in USDC stablecoin
- Trading takes place on Uniswap on Polygon blockchain
- The strategy decision cycle is one day

## Backtesting

The strategy parameters (length for calculating rolling Z-score, and take profit and stop loss limits) have been optimized using Binance data from 1 January 2021 to 31 March 2024. Backtests have been carried out for subsets of this timeframe.

## Profit

The backtested historical results indicate 104.5% estimated yearly profit (CAGR).

This is above the historical profit you would have gotten by buying and holding BTC or ETH.

## Risk

This strategy has produced a maximum -22.8% backtested drawdown. This is much less severe compared to buy and hold, making the strategy less risky than buy and hold historically.

For further understanding the key aspects of risks

- The strategy does not use any leverage
- The strategy trades only the established, highly liquid, trading pair ETH-USDC which is unlikely to go zero based on historical data

"""

# Fees
management_fee = "0%"
trading_strategy_protocol_fee = "2%"
strategy_developer_fee = "5%"
enzyme_protocol_fee = "0.25%"