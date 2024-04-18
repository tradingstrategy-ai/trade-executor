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

import pandas_ta
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
            (ChainId.centralised_exchange, "binance", "WMATIC", "USDC"),
        ]



class Parameters:
    """Parameteres for this strategy.

    - Collect parameters used for this strategy here

    - Both live trading and backtesting parameters
    """

    id = "v02-matic-breakout-single" # Used in cache paths

    cycle_duration = CycleDuration.cycle_1h  # Run decide_trades() every 8h
    source_time_bucket = TimeBucket.h1  # Use 1h candles as the raw data
    target_time_bucket = TimeBucket.h1  # Create synthetic 8h candles
    clock_shift_bars = 0  # Do not do shifted candles

    allocation = 0.98   # Cash allocation per open position

    rsi_bars = 14  # Number of bars to calculate RSI for each trading bar
    bollinger_bands_ma_length = 19
    std_dev = 2.8

    # regime_filter_ma_length = 200  # Bull/bear MA begime filter in days
    # regime_filter_only_btc = 1   # Use BTC or per-pair regime filter

    rsi_entry = 50
    trailing_stop_loss = 0.98  # Trailing stop loss as 1 - x
    trailing_stop_loss_activation_level = 1.05  # How much above opening price we must be before starting to use trailing stop loss
    stop_loss = 0.98  # Hard stop loss when opening a new position

    #
    # Live trading only
    #
    chain_id = ChainId.polygon
    routing = TradeRouting.default  # Pick default routes for trade execution
    required_history_period = datetime.timedelta(hours=25) * 2  # Based on max MA length 2x extra history just in case

    #
    # Backtesting only
    #
    # backtest_start = datetime.datetime(2019, 9, 1) #  Include early abnormal MATIC pump
    backtest_start = datetime.datetime(2021, 9, 1)
    backtest_end = datetime.datetime(2024, 4, 15)
    stop_loss_time_bucket = TimeBucket.m15  # use 1h close as the stop loss signal
    backtest_trading_fee = 0.0005  # Switch to QuickSwap 30 BPS free from the default Binance 5 BPS fee
    initial_cash = 10_000


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
    """Trade logic."""

    # Resolve some variables we are going to use to here
    parameters = input.parameters
    position_manager = input.get_position_manager()
    state = input.state
    timestamp = input.timestamp
    indicators = input.indicators
    shift = parameters.clock_shift_bars
    clock_shift = pd.Timedelta(hours=1) * shift
    upsample = parameters.target_time_bucket
    mode = input.execution_context.mode
    our_pairs = get_strategy_trading_pairs(mode)

    # Execute the daily trade cycle when the clock hour 0..24 is correct for our hourly shift
    assert upsample.to_timedelta() >= parameters.cycle_duration.to_timedelta(), "Upsample period must be longer than cycle period"
    assert shift <= 0  # Shift -1 = do action 1 hour later

    # Override the trading fee to simulate worse DEX fees and price impact vs. Binance
    if mode.is_backtesting():
        if parameters.backtest_trading_fee:
            input.pricing_model.set_trading_fee_override(parameters.backtest_trading_fee)

    # Do the clock shift trick
    if parameters.cycle_duration.to_timedelta() != upsample.to_timedelta():
        if (input.cycle - 1 + shift) % int(upsample.to_hours()) != 0:
            return []

    alpha_model = AlphaModel(input.timestamp)
    btc_pair = position_manager.get_trading_pair(our_pairs[0])
    eth_pair = position_manager.get_trading_pair(our_pairs[1])
    position_manager.log("decide_trades() start")

    #
    # Indicators
    #
    # Calculate indicators for each pair.
    #

    # Per-trading pair calcualted data
    current_rsi_values = {}  # RSI yesterday
    previous_rsi_values = {}  # RSI day before yesterday
    current_price = {}  # Close price yesterday
    momentum = {btc_pair: 0, eth_pair: 0}

    for pair in [btc_pair, eth_pair]:
        current_price[pair] = indicators.get_price(pair)
        current_rsi_values[pair] = indicators.get_indicator_value("rsi", index=-1, pair=pair, clock_shift=clock_shift)
        previous_rsi_values[pair] = indicators.get_indicator_value("rsi", index=-2, pair=pair, clock_shift=clock_shift)

    eth_btc_yesterday = indicators.get_indicator_value("eth_btc", clock_shift=clock_shift)
    eth_btc_rsi_yesterday = indicators.get_indicator_value("eth_btc_rsi", clock_shift=clock_shift)
    if eth_btc_rsi_yesterday is not None:
        eth_momentum = (eth_btc_rsi_yesterday / 100) + 0.5
        btc_momentum = (1 - (eth_btc_rsi_yesterday / 100)) + 0.5
        momentum[eth_pair] = eth_momentum ** parameters.momentum_exponent
        momentum[btc_pair] = btc_momentum ** parameters.momentum_exponent

    #
    # Trading logic
    #

    for pair in [btc_pair, eth_pair]:

        #
        # Regime filter
        #
        # If no indicator data yet, or regime filter disabled,
        # be always bullish
        bullish = True
        if parameters.regime_filter_ma_length:  # Regime filter is not disabled
            regime_filter_pair = btc_pair if parameters.regime_filter_only_btc else pair  # Each pair has its own bullish/bearish regime?
            regime_filter_price = current_price[regime_filter_pair]
            sma = indicators.get_indicator_value("sma", index=-1, pair=regime_filter_pair, clock_shift=clock_shift)
            if sma:
                # We are bearish if close price is beloe SMA
                bullish = regime_filter_price > sma

        if bullish:
            rsi_entry = parameters.bullish_rsi_entry
            rsi_exit = parameters.bullish_rsi_exit
        else:
            rsi_entry = parameters.bearish_rsi_entry
            rsi_exit = parameters.bearish_rsi_exit

        existing_position = position_manager.get_current_position_for_pair(pair)
        pair_open = existing_position is not None
        pair_momentum = momentum.get(pair, 0)
        signal_strength = max(pair_momentum, 0.1)  # Singal strength must be positive, as we do long-only
        if pd.isna(signal_strength):
            signal_strength = 0
        alpha_model.set_signal(pair, 0)

        if pair_open:
            # We have existing open position for this pair,
            # keep it open by default unless we get a trigger condition below
            position_manager.log(f"Pair {pair} already open")
            alpha_model.set_signal(pair, signal_strength, stop_loss=parameters.stop_loss)

        if current_rsi_values[pair] and previous_rsi_values[pair]:

            # Check for RSI crossing our threshold values in this cycle, compared to the previous cycle
            if rsi_entry:
                rsi_cross_above = current_rsi_values[pair] >= rsi_entry and previous_rsi_values[pair] < rsi_entry
            else:
                # bearish_rsi_entry = None -> don't trade in bear market
                rsi_cross_above = False

            rsi_cross_below = current_rsi_values[pair] < rsi_exit and previous_rsi_values[pair] > rsi_exit

            if not pair_open:
                # Check for opening a position if no position is open
                if rsi_cross_above:
                    position_manager.log(f"Pair {pair} crossed above")
                    alpha_model.set_signal(pair, signal_strength, stop_loss=parameters.stop_loss)
            else:
                # We have open position, check for the close condition
                if rsi_cross_below:
                    position_manager.log(f"Pair {pair} crossed below")
                    alpha_model.set_signal(pair, 0)

    # Enable trailing stop loss if we have reached the activation level
    if parameters.trailing_stop_loss_activation_level is not None and parameters.trailing_stop_loss is not None:
       for p in state.portfolio.open_positions.values():
           if p.trailing_stop_loss_pct is None:
               if current_price[p.pair] >= p.get_opening_price() * parameters.trailing_stop_loss_activation_level:
                   p.trailing_stop_loss_pct = parameters.trailing_stop_loss

    # Use alpha model and construct a portfolio of two assets
    alpha_model.select_top_signals(2)
    alpha_model.assign_weights(weight_passthrouh)
    alpha_model.normalise_weights()
    alpha_model.update_old_weights(state.portfolio)
    portfolio = position_manager.get_current_portfolio()
    portfolio_target_value = portfolio.get_total_equity() * parameters.allocation
    alpha_model.calculate_target_positions(position_manager, portfolio_target_value)
    trades = alpha_model.generate_rebalance_trades_and_triggers(
        position_manager,
        min_trade_threshold=parameters.rebalance_threshold * portfolio.get_total_equity(),
    )

    #
    # Visualisations
    #

    if input.is_visualisation_enabled():

        visualisation = state.visualisation  # Helper class to visualise strategy output

        # BTC RSI daily
        if current_rsi_values[btc_pair]:
            visualisation.plot_indicator(
                timestamp,
                f"RSI BTC",
                PlotKind.technical_indicator_detached,
                current_rsi_values[btc_pair],
                colour="orange",
            )

            # RSI exit value
            visualisation.plot_indicator(
                timestamp,
                f"RSI exit trigger",
                PlotKind.technical_indicator_overlay_on_detached,
                rsi_exit,
                detached_overlay_name=f"RSI BTC",
                colour="red",
                label=PlotLabel.hidden,
            )

            # RSI entry value
            visualisation.plot_indicator(
                timestamp,
                f"RSI entry trigger",
                PlotKind.technical_indicator_overlay_on_detached,
                rsi_entry,
                detached_overlay_name=f"RSI BTC",
                colour="red",
                label=PlotLabel.hidden,
            )

        # ETH RSI daily
        if current_rsi_values[eth_pair]:
            visualisation.plot_indicator(
                timestamp,
                f"RSI ETH",
                PlotKind.technical_indicator_overlay_on_detached,
                current_rsi_values[eth_pair],
                colour="blue",
                label=PlotLabel.hidden,
                detached_overlay_name=f"RSI BTC",
            )

        if eth_btc_yesterday is not None:
            visualisation.plot_indicator(
                timestamp,
                f"ETH/BTC",
                PlotKind.technical_indicator_detached,
                eth_btc_yesterday,
                colour="grey",
            )

        if eth_btc_rsi_yesterday is not None:
            visualisation.plot_indicator(
                timestamp,
                f"ETH/BTC RSI",
                PlotKind.technical_indicator_detached,
                eth_btc_rsi_yesterday,
                colour="grey",
            )

        state.visualisation.add_calculations(timestamp, alpha_model.to_dict())  # Record alpha model thinking

    position_manager.log(
        f"BTC RSI: {current_rsi_values[btc_pair]}, BTC RSI yesterday: {previous_rsi_values[btc_pair]}",
    )

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
            stop_loss_time_bucket=Parameters.stop_loss_time_bucket,
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

sort_priorty = 99

name = "ETH-BTC-USDC momentum"

short_description = "A momentum strategy for ETH-USDC and BTC-USDC pairs based on RSI indicators"

icon = "https://tradingstrategy.ai/avatars/polygon-eth-spot-short.webp"

long_description = """
# Strategy description

This strategy is a momentum and breakout strategy.

- Based on [RSI technical indicator](https://tradingstrategy.ai/glossary/relative-strength-index-rsi), the strategy is buying when others are buying, and the strategy is selling when others are selling
- The strategy has [a regime filter](https://tradingstrategy.ai/glossary/regime-filter) to reduce trading in a [bear market](https://tradingstrategy.ai/glossary/bear-market) 

**Past Performance Is Not Indicative Of Future Results**.

## Assets and trading venues

- The strategy trades only spot market
- We trade two trading asset: ETH and BTC
- The strategy keeps reserves in USDC stablecoin
- The trading happens on QuickSwap and Uniswap on Polygon blockchain
- The strategy decision cycle is daily rebalance

## Backtesting

The backtesting was performed with Binance ETH-USDT and BTC-USDT data of 2019-2024.
[Read more about what is backtesting](https://tradingstrategy.ai/glossary/backtest).

The backtesting trading venue (Binance) is different from the live trading venue (Quickswap, Uniswap), because DEX markets
do not have long enough history to result to a meaningful backtest.

The backtesting period saw one bull market rally that is unlikely to repeat in the same magnitude 
for the assets we trade.

Past peformance is no guarantee of future results. Like with manual trading, automated trading is unlikely to be perfect.
There will be variance in the range of 30% - 50% in the results.

## Profit

The backtested results indicate *80%* yearly profit ([CAGR](https://tradingstrategy.ai/glossary/compound-annual-growth-rate-cagr)). 

## Risk

This is medium risk strategy with *-50%* backtested [maximum drawdown](https://tradingstrategy.ai/glossary/maximum-drawdown).
The backtested [Sharpe ratio](https://tradingstrategy.ai/glossary/sharpe) is is *1.00*.

For understanding the risk assessment

- The strategy does not use any leverage
- The strategy has high maximum drawdown, comparable to buying and holding cryptocurrency  
- The strategy trades only highly liquid trading pairs which are unlikely to go to zero

## Benchmark

For the same backtesting period, here are some benchmark of performance of different assets and indices:

- Buy and hold BTC:
- Buy and hold ETH:
- SP500 stock index:
- US treasury notes: 

## Trading frequency

This strategy is estimated to enter a position every *20 days*. 

## Robustness

The strategy wins 50% of its trading positions.
The strategy is not very accurate. The profits 
come for long running bull market positions.

## Further information

- [Any questions are welcome in the Discord community chat](https://tradingstrategy.ai/community)
- [See the blog post how the strategy is constructed](https://tradingstrategy.ai/blog/outperfoming-eth) on how this strategy is constructed
 
"""