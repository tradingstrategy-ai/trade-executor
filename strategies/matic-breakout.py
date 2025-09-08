"""MATIC breakout strategy.

- See https://github.com/tradingstrategy-ai/tradingview-defi-strategy for the strategy development information

To backtest this strategy module locally:

.. code-block:: console

    trade-executor \
        backtest \
        --strategy-file=strategies/matic-breakout.py \
        --trading-strategy-api-key=$TRADING_STRATEGY_API_KEY

To see the backtest for longer history, refer to the notebook doing backtest with Binance data.
"""
import datetime

import pandas_ta_classic as pandas_ta
import pandas as pd

from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.visualisation import PlotKind, PlotLabel, PlotShape
from tradeexecutor.strategy.alpha_model import AlphaModel
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet, IndicatorSource
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.tag import StrategyTag
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, load_partial_data
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradeexecutor.strategy.weighting import weight_passthrouh
from tradeexecutor.utils.binance import create_binance_universe
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.pair import HumanReadableTradingPairDescription
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.utils.groupeduniverse import resample_price_series

trading_strategy_engine_version = "0.5"
management_fee=0.00
trading_strategy_protocol_fee=0.02
strategy_developer_fee=0.1

class Parameters:
    """Parameteres for this strategy.

    - Collect parameters used for this strategy here

    - Both live trading and backtesting parameters
    """

    id = "matic-breakout" # Used in cache paths

    cycle_duration = CycleDuration.cycle_1h  # Run decide_trades() every 8h
    source_time_bucket = TimeBucket.h1  # Use 1h candles as the raw data
    target_time_bucket = TimeBucket.h1  # Create synthetic 8h candles
    clock_shift_bars = 0  # Do not do shifted candles

    allocation = 0.98   # Cash allocation per open position

    rsi_bars = 14  # Number of bars to calculate RSI for each trading bar
    bollinger_bands_ma_length = 19
    std_dev = 2.8

    rsi_entry = 50
    trailing_stop_loss = 0.98  # Trailing stop loss as 1 - x
    trailing_stop_loss_activation_level = 1.05  # How much above opening price we must be before starting to use trailing stop loss
    stop_loss = 0.98  # Hard stop loss when opening a new position

    #
    # Live trading only
    #
    chain_id = ChainId.polygon
    routing = TradeRouting.default  # Pick default routes for trade execution
    required_history_period = datetime.timedelta(hours=bollinger_bands_ma_length) * 2

    #
    # Backtesting only
    #
    # backtest_start = datetime.datetime(2019, 9, 1) #  Include early abnormal MATIC pump
    backtest_start = datetime.datetime(2021, 9, 1)
    backtest_end = datetime.datetime(2024, 4, 15)
    stop_loss_time_bucket = TimeBucket.m15  # use 1h close as the stop loss signal
    backtest_trading_fee = 0.0005  # Switch to QuickSwap 30 BPS free from the default Binance 5 BPS fee
    initial_cash = 10_000




def get_strategy_trading_pairs(execution_mode: ExecutionMode) -> list[HumanReadableTradingPairDescription]:
    """Switch between backtest and live trading pairs.

    Because the live trading DEX venues do not have enough history (< 2 years)
    for meaningful backtesting, we test with Binance CEX data.
    """
    if execution_mode.is_live_trading():
        # Live trade
        return [
            (ChainId.polygon, "uniswap-v3", "WMATIC", "USDC", 0.0005),
        ]
    else:
        # Backtest - Binance fee matched to DEXes with Parameters.backtest_trading_fee
        return [
            (ChainId.centralised_exchange, "binance", "MATIC", "USDT"),
        ]


def create_indicators(
    timestamp: datetime.datetime | None,
    parameters: StrategyParameters,
    strategy_universe: TradingStrategyUniverse,
    execution_context: ExecutionContext
):
    indicators = IndicatorSet()

    indicators.add(
        "rsi",
        pandas_ta.rsi,
        {"length": parameters.rsi_bars}
    )

    indicators.add(
        "bbands",
        pandas_ta.bbands,
        {"length": parameters.bollinger_bands_ma_length, "std": parameters.std_dev},
    )
    return indicators


def decide_trades(
    input: StrategyInput,
) -> list[TradeExecution]:


    # Resolve our pair metadata for our two pair strategy
    parameters = input.parameters
    position_manager = input.get_position_manager()
    state = input.state
    timestamp = input.timestamp
    indicators = input.indicators
    strategy_universe = input.strategy_universe

    position_manager.log("decide_trades() start")

    pair = strategy_universe.get_single_pair()
    cash = position_manager.get_current_cash()

    #
    # Indicators
    #

    # Read the indicator data for the current timestamp,
    # as calculated from the previous close value.
    # pandas_ta.bbands creates 3 series (columns) in its output
    #
    bollinger_bands_ma_length = parameters.bollinger_bands_ma_length
    std_dev = parameters.std_dev
    bb_upper_column = f"BBU_{bollinger_bands_ma_length}_{std_dev:.1f}" # pandas_ta internal column naming
    bb_mid_column = f"BBM_{bollinger_bands_ma_length}_{std_dev:.1f}" # pandas_ta internal column naming
    bb_lower_column = f"BBL_{bollinger_bands_ma_length}_{std_dev:.1f}" # pandas_ta internal column naming

    bb_upper = indicators.get_indicator_value("bbands", column=bb_upper_column)
    bb_mid = indicators.get_indicator_value("bbands", column=bb_mid_column)
    bb_lower = indicators.get_indicator_value("bbands", column=bb_lower_column)
    rsi = indicators.get_indicator_value("rsi")
    last_close_price = indicators.get_price()

    #
    # Trading logic
    #

    trades = []

    # Check if we are too early in the backtesting to have enough data to calculate indicators
    if None in (bb_upper, bb_mid, bb_lower, rsi):
        return []

    if not position_manager.is_any_open():
        # No open positions, decide if BUY in this cycle.
        # We buy if the price on the daily chart closes above the upper Bollinger Band.
        if last_close_price > bb_upper and rsi > parameters.rsi_entry:
            buy_amount = cash * parameters.allocation
            trades += position_manager.open_1x_long(pair, buy_amount, stop_loss_pct=parameters.stop_loss)

    else:
        # We have an open position, decide if SELL in this cycle.
        # We close the position when the price closes below the X bars moving average.
        if last_close_price < bb_mid:
            trades += position_manager.close_all()

        # Check if we have reached out level where we activate trailing stop loss
        current_position = position_manager.get_current_position()
        if last_close_price >= current_position.get_opening_price() * parameters.trailing_stop_loss_activation_level:
            current_position.trailing_stop_loss_pct = parameters.trailing_stop_loss

    #
    # Visualisations
    #

    if input.is_visualisation_enabled():
        visualisation = state.visualisation  # Helper class to visualise strategy output
        visualisation.plot_indicator(timestamp, "BB upper", PlotKind.technical_indicator_on_price, bb_upper, colour="darkblue")
        visualisation.plot_indicator(timestamp, "BB lower", PlotKind.technical_indicator_on_price, bb_lower, colour="darkblue")
        visualisation.plot_indicator(timestamp, "BB mid", PlotKind.technical_indicator_on_price, bb_mid, colour="blue")

        # Draw the RSI indicator on a separate chart pane.
        # Visualise the high RSI threshold we must exceed to take a position.
        visualisation.plot_indicator(timestamp, "RSI", PlotKind.technical_indicator_detached, rsi)
        visualisation.plot_indicator(timestamp, "RSI entry", PlotKind.technical_indicator_overlay_on_detached, parameters.rsi_entry, colour="red", detached_overlay_name="RSI")

    return trades


def create_trading_universe(
    timestamp: datetime.datetime,
    client: Client,
    execution_context: ExecutionContext,
    universe_options: UniverseOptions,
) -> TradingStrategyUniverse:
    """Create the trading universe.

    - For live trading, we load DEX data

    - We backtest with Binance data, as it has more history
    """

    pair_ids = get_strategy_trading_pairs(execution_context.mode)

    if execution_context.mode.is_backtesting():
        # Backtesting - load Binance data
        strategy_universe = create_binance_universe(
            [f"{p[2]}{p[3]}" for p in pair_ids],
            candle_time_bucket=Parameters.source_time_bucket,
            stop_loss_time_bucket=Parameters.stop_loss_time_bucket,
            start_at=universe_options.start_at,
            end_at=universe_options.end_at,
            trading_fee_override=Parameters.backtest_trading_fee,
        )
    else:

        dataset = load_partial_data(
            client=client,
            time_bucket=Parameters.source_time_bucket,
            pairs=pair_ids,
            execution_context=execution_context,
            universe_options=universe_options,
            liquidity=False,
        )
        # Construct a trading universe from the loaded data,
        # and apply any data preprocessing needed before giving it
        # to the strategy and indicators
        strategy_universe = TradingStrategyUniverse.create_from_dataset(
            dataset,
            reserve_asset="USDC",
            forward_fill=True,
        )

    return strategy_universe
#
# Strategy metadata.
#
# Displayed in the user interface.
#

tags = {StrategyTag.beta, StrategyTag.live}

sort_priority = 99

name = "MATIC breakout"

short_description = "MATIC price breakout strategy"

icon = "https://tradingstrategy.ai/avatars/polygon-eth-spot-short.webp"

long_description = """
# Strategy description

This strategy is a breakout strategy.

- Based on [RSI technical indicator](https://tradingstrategy.ai/glossary/relative-strength-index-rsi), the strategy enters to short-lived positions when MATIC price is breaking up 

**Past Performance Is Not Indicative Of Future Results**.

## Assets and trading venues

- The strategy trades only spot market
- We trade a single trading asset: MATIC
- The strategy keeps reserves in USDC stablecoin
- The trading happens on Uniswap on Polygon blockchain

## Backtesting

## Profit

## Risk

## Benchmark

## Trading frequency

## Robustness

## Further information

- [Any questions are welcome in the Discord community chat](https://tradingstrategy.ai/community)
- [See the blog post how the strategy is constructed](https://tradingstrategy.ai/blog/outperfoming-eth) on how this strategy is constructed
 
"""