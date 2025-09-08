"""Multipair arbitrum strategy based on stoch rsi indicator. Long only.

To backtest this strategy module locally:

.. code-block:: console

    source scripts/set-latest-tag-gcp.sh
    docker-compose run arbitrum-btc-eth-stoch-rsi backtest

    trade-executor \
        backtest \
        --strategy-file=strategy/enzyme-ethereum-btc-eth-stoch-rsi.py \
        --trading-strategy-api-key=$TRADING_STRATEGY_API_KEY

"""

import datetime
import pandas_ta_classic as pandas_ta


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

    id = "Weekly Stochastic Crossover 3" # Used in cache paths

    cycle_duration = CycleDuration.cycle_7d
    candle_time_bucket = TimeBucket.d7
    credit_allocation = 1.0

    rsi_length = 26

    stoch_rsi_low = 20
    stoch_rsi_high = 40 
    stoch_rsi_length = 19

    # stop_loss_pct = Real(0.7, 0.99)
    stop_loss_pct = 0.9
    trailing_stop_loss_pct = 0.80 
    trailing_stop_loss_activation_level = 1.0 

    #
    # Live trading only
    #
    chain_id = ChainId.ethereum
    routing = TradeRouting.default  
    required_history_period = datetime.timedelta(weeks=max(rsi_length, stoch_rsi_length) + 2)
    trading_strategy_engine_version = "0.5"
    
    #
    # Backtesting only
    #

    # Use Binance data in backtesting,
    # We get a longer, more meaningful, history but no credit simulation.
    binance_data = True

    if binance_data:
        backtest_start = datetime.datetime(2020, 1, 1)
        # backtest_end = datetime.datetime(2024, 4, 20)
        # backtest_start = datetime.datetime(2022, 6, 1)
        backtest_end = datetime.datetime(2024, 7, 15)

    else:
        # dex dates
        backtest_start = datetime.datetime(2021, 4, 1)
        backtest_end = datetime.datetime(2024, 5, 15)
    
    stop_loss_time_bucket = TimeBucket.h4
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
            (ChainId.ethereum, "uniswap-v3", "WBTC", "USDC", 0.0030),  # Deep liquidity
            (ChainId.ethereum, "uniswap-v3", "WETH", "USDC", 0.0005),  # Deep liquidity
        ]

    return trading_pairs


def get_lending_reserves(mode: ExecutionMode) -> list[LendingReserveDescription]:
    """Get lending reserves the strategy needs."""
    
    use_binance = mode.is_backtesting() and Parameters.binance_data 

    if use_binance:
        # Credit interest is not available on Binance
       return []
    else:
        # We use Aave v2 in backtesting (longer history)
        # and Aave v3 in live execution (more liquid market)
        if mode.is_backtesting():
            lending_reserves = [
               (ChainId.ethereum, LendingProtocolType.aave_v2, "USDC"),
            ]
        else:
            lending_reserves = [
               (ChainId.ethereum, LendingProtocolType.aave_v3, "USDC"),
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


def create_indicators(
    timestamp: datetime.datetime | None,
    parameters: StrategyParameters,
    strategy_universe: TradingStrategyUniverse,
    execution_context: ExecutionContext
):
    indicators = IndicatorSet()

    indicators.add(
        "stoch_rsi",
        pandas_ta.stochrsi,
        {"length": parameters.stoch_rsi_length, 'rsi_length': parameters.stoch_rsi_length, 'k': 3, 'd': 3},  # No parameters needed for this custom function
        IndicatorSource.close_price,
    )

    indicators.add(
        "rsi",
        pandas_ta.rsi,
        {"length": parameters.rsi_length},
        IndicatorSource.close_price,
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

    trades = []

    # Enable trailing stop loss after we reach the profit taking level
    #
    for position in state.portfolio.open_positions.values():
        if not position.is_credit_supply():
            if position.trailing_stop_loss_pct is None:
                close_price = indicators.get_price(position.pair)
                if close_price >= position.get_opening_price() * parameters.trailing_stop_loss_activation_level:
                    position.trailing_stop_loss_pct = parameters.trailing_stop_loss_pct 

    # Setup asset allocation parameters
    max_assets = len(trading_pairs)
    allocation = round(1/max_assets - 0.01, 2)
    use_credit = len(lending_reserves) > 0

    # If any of trading pairs enters to long position,
    # close our credit position
    credit_closed = False
    traded_this_cycle = False
    available_cash = cash
    ready = False

    for pair_desc in trading_pairs:
        
        #
        # Indicators
        #
        pair = strategy_universe.get_pair_by_human_description(pair_desc)

        close_price = indicators.get_price(pair=pair)  # Price the previous 15m candle closed for this decision cycle timestamp
        rsi_k = indicators.get_indicator_value("stoch_rsi", pair=pair, column=f'STOCHRSIk_{parameters.stoch_rsi_length}_{parameters.stoch_rsi_length}_3_3')  
        rsi_d = indicators.get_indicator_value("stoch_rsi", pair=pair, column=f'STOCHRSId_{parameters.stoch_rsi_length}_{parameters.stoch_rsi_length}_3_3')  

        # Visualisations
        #
        if input.is_visualisation_enabled():
            visualisation = state.visualisation
            visualisation.plot_indicator(timestamp, f"RSI Stochastic {pair.base}", PlotKind.technical_indicator_detached, rsi_d, pair=pair)
            visualisation.plot_indicator(timestamp,f"rsi_k {pair}", PlotKind.technical_indicator_overlay_on_detached, rsi_k, pair=pair, detached_overlay_name=f"RSI Stochastic {pair.base}")
            visualisation.plot_indicator(timestamp,f"Rsi Stochastic Low {pair}", PlotKind.technical_indicator_overlay_on_detached, parameters.stoch_rsi_low, pair=pair, detached_overlay_name=f"RSI Stochastic {pair.base}")

        if None in (rsi_k, rsi_d, close_price):
            # Not enough historic data,
            # cannot make decisions yet
            continue

        ready = True

        rsi_k_series = indicators.get_indicator_series("stoch_rsi", pair=pair, column=f'STOCHRSIk_{parameters.stoch_rsi_length}_{parameters.stoch_rsi_length}_3_3')  
        rsi_d_series = indicators.get_indicator_series("stoch_rsi", pair=pair, column=f'STOCHRSId_{parameters.stoch_rsi_length}_{parameters.stoch_rsi_length}_3_3')  

        crossover, crossover_index = contains_cross_over(
                rsi_k_series,
                rsi_d_series,
                lookback_period=2,
                must_return_index=True
            )

        crossunder, crossunder_index = contains_cross_under(
                rsi_k_series,
                rsi_d_series,
                lookback_period=2,
                must_return_index=True
            )
        #
        # Trading logic
        #
        if crossover and crossover_index == -1 :
            if len(state.portfolio.open_positions) >= max_assets:
                pass
                # print(f"Want to place in a trade but there are already {len(state.portfolio.open_positions)} positions open and max is {parameters.max_assets} for pair {pair.base}")

        # Check for open condition - is the price breaking out
        #
        non_credit_open_positions = [p for p in state.portfolio.open_positions.values() if not p.is_credit_supply()]
        if len(non_credit_open_positions) < max_assets and state.portfolio.get_open_position_for_pair(pair) is None:        
            if  crossover and crossover_index == -1 :
                # close credit supply position before opening a new long position
                if position_manager.is_any_credit_supply_position_open():
                    #print(f"Closing credit supply position on {timestamp}")
                    if not credit_closed:
                        current_pos = position_manager.get_current_credit_supply_position()
                        new_trades = position_manager.close_credit_supply_position(current_pos)
                        trades.extend(new_trades)
                        # Est. available cash after all credit positions are closed
                        available_cash += float(current_pos.get_quantity()) 
                        credit_closed = True

                trades += position_manager.open_spot(
                    pair,
                    value=available_cash * allocation,
                    stop_loss_pct=parameters.stop_loss_pct,             
                )

                traded_this_cycle = True
        else:
            # Check for close condition
            if  crossunder and crossunder_index == -1 and rsi_d > parameters.stoch_rsi_high and state.portfolio.get_open_position_for_pair(pair) is not None:
                position = state.portfolio.get_open_position_for_pair(pair)
                trades += position_manager.close_position(position)
                traded_this_cycle = True

    if ready:

        # We have accumulatd enough data to make the first real (non credit) trading decision.
        # This allows us to have fair buy-and-hold vs backtest period comparison
        state.mark_ready(timestamp)

        if not position_manager.is_any_credit_supply_position_open() and use_credit and not traded_this_cycle:
            pos_size = available_cash * 0.999
            new_trades = position_manager.open_credit_supply_position_for_reserves(pos_size)
            trades += new_trades
        
    return trades  # Return the list of trades we made in this cycle


#
# Strategy metadata.
#
# Displayed in the user interface.
#

tags = {StrategyTag.beta, StrategyTag.live}

name = Parameters.id

short_description = "Stochastic RSI-based strategy for multiple pairs on Arbitrum, focusing on long positions."

icon = "https://tradingstrategy.ai/avatars/arbitrum-stoch-rsi.webp"

long_description = """
# Strategy description

This strategy leverages the Stochastic RSI indicator to identify long-only opportunities on multiple trading pairs within the Arbitrum ecosystem, specifically ETH and BTC.

- Trades on multiple pairs including WBTC/USDC and WETH/USDC on Uniswap V3.
- Designed to capture long-term trends while minimizing drawdowns.
- The strategy focuses on weekly cycles, rebalancing every 7 days.
- It performs well in trending markets and aims to protect capital during downturns with strict stop-loss mechanisms.

**Past performance is not indicative of future results**.

## Assets and trading venues

- The strategy trades on decentralized exchanges (DEX) such as Uniswap V3 on the Arbitrum chain.
- Trading pairs include WBTC/USDC and WETH/USDC with a fee tier of 0.0005.
- Keeps reserves in stablecoins such as USDC.

## Stochastic trading

Stochastic trading is a technical analysis strategy that utilizes the Stochastic Oscillator to identify overbought and oversold conditions in the market. The Stochastic Oscillator, developed by George Lane, compares a security's closing price to its price range over a specified period, typically 14 days. The indicator generates values between 0 and 100, with readings above 80 indicating overbought conditions and readings below 20 indicating oversold conditions. Traders use these signals to make buy or sell decisions, often in conjunction with other technical indicators to confirm trends and improve the accuracy of their trades.


## Backtesting

The backtesting was performed using data from Uniswap V3 on Arbitrum from June 2022 to June 2024.

- [See backtesting results](./backtest)
- [Read more about what is backtesting](https://tradingstrategy.ai/glossary/backtest).

The backtesting period included various market conditions, providing a comprehensive overview of the strategy's performance.

## Profit

The backtested results indicate an estimated yearly profit (CAGR) of **XX%**. The exact figure can be derived from the backtesting results.

## Risk

The strategy has a backtested maximum drawdown of **-XX%**. It employs strict stop-loss and trailing stop mechanisms to mitigate losses.

For further understanding the key aspects of risks:
- The strategy does not use any leverage.
- Trades only highly liquid pairs to ensure minimal slippage and robust trade execution.

## Benchmark

Here are some benchmarks comparing the strategy's performance with other indices:

|                              | CAGR | Maximum drawdown | Sharpe |
|------------------------------|------|------------------|--------|
| This strategy                | 32%  | -10%             | 1.41   |
| SP500 (20 years)             | 11%  | -33%             | 0.72   |
| Bitcoin (backtesting period) | 76%  | -76%             | 1.17   |
| Ether (backtesting period)   | 85%  | -79%             | 1.18   |

Sources:

- [Our strategy](./backtest)
- [Buy and hold BTC](./backtest)
- [Buy and hold ETH](./backtest)
- [SP500 stock index](https://curvo.eu/backtest/en/portfolio/s-p-500--NoIgygZACgBArABgSANMUBJAokgQnXAWQCUEAOAdlQEYBdeoA?config=%7B%22periodStart%22%3A%222004-02%22%7D)

## Trading frequency

The strategy operates on a weekly cycle, rebalancing every 7 days and adjusting positions as necessary based on Stochastic RSI signals.

## Robustness

The strategy has been tested on a longer time frame of a number of years with Binance data and was subsequently optimized for the Arbitrum chain.

## Updates

This strategy is periodically reviewed and updated to incorporate the latest market data and trading techniques. Stay tuned for updates via the [Trading Strategy community](https://tradingstrategy.ai/community).

## Further information

- Any questions are welcome in [the Discord community chat](https://tradingstrategy.ai/community).
- See the blog post [on how this strategy is constructed](https://tradingstrategy.ai/blog/arbitrum-stoch-rsi) for more details.

"""
