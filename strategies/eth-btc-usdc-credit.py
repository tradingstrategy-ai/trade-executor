"""ETH-BTC-USDC 1h rebalance strategy, high fee variant.

= Updated version 2.0 that uses Aave credit positions for extra yield

- See https://tradingstrategy.ai/blog/outperfoming-eth for the strategy development information

To backtest this strategy module locally:

.. code-block:: console

    trade-executor \
        backtest \
        --strategy-file=strategies/eth-btc-usdc.py \
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
from tradingstrategy.lending import LendingReserveDescription, LendingProtocolType
from tradingstrategy.pair import HumanReadableTradingPairDescription
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.utils.groupeduniverse import resample_price_series

trading_strategy_engine_version = "0.5"



class Parameters:
    """Parameteres for this strategy.

    - Collect parameters used for this strategy here

    - Both live trading and backtesting parameters
    """

    cycle_duration = CycleDuration.cycle_1d  # Run decide_trades() every 8h
    source_time_bucket = TimeBucket.d1  # Use 1h candles as the raw data
    target_time_bucket = TimeBucket.d1  # Create synthetic 8h candles
    clock_shift_bars = 0  # Do not do shifted candles

    rsi_bars = 8  # Number of bars to calculate RSI for each tradingbar
    eth_btc_rsi_bars = 5  # Number of bars for the momentum factor

    # RSI parameters for bull and bear market
    bearish_rsi_entry = 65
    bearish_rsi_exit = 70
    bullish_rsi_entry = 80
    bullish_rsi_exit = 65

    regime_filter_ma_length = 200  # Bull/bear MA begime filter in days
    regime_filter_only_btc = 1   # Use BTC or per-pair regime filter

    allocation = 0.98  # How much cash allocate for volatile positions
    rebalance_threshold = 0.275  # How much position mix % must change when we rebalance between two open positions
    initial_cash = 10_000  # Backtesting start cash
    trailing_stop_loss = None  # Trailing stop loss as 1 - x
    trailing_stop_loss_activation_level = None  # How much above opening price we must be before starting to use trailing stop loss
    stop_loss = None  # 0.80  # Hard stop loss when opening a new position
    momentum_exponent = 3.5  # How much momentum we capture when rebalancing between open positions

    #
    # Live trading only
    #
    chain_id = ChainId.polygon
    routing = TradeRouting.default  # Pick default routes for trade execution
    required_history_period = datetime.timedelta(days=regime_filter_ma_length) * 2  # Ask some extra history just in case

    #
    # Backtesting only
    #

    backtest_start = datetime.datetime(2023, 8, 1)
    backtest_end = datetime.datetime(2024, 3, 15)
    stop_loss_time_bucket = TimeBucket.h1  # use 1h close as the stop loss signal
    use_credit = True  # Allow us to flip credit usage on/off in backtesting to more easily test different scenarios
    backtest_trading_fee = 0.0030  # Switch to QuickSwap 30 BPS free from the default Binance 5 BPS fee


def calculate_eth_btc(strategy_universe: TradingStrategyUniverse, mode: ExecutionMode):
    """Calculate ETH/BTC price used as a rebalance factor."""
    pair_ids = get_strategy_trading_pairs(mode)
    eth = strategy_universe.get_pair_by_human_description(pair_ids[1])
    btc = strategy_universe.get_pair_by_human_description(pair_ids[0])
    btc_price = strategy_universe.data_universe.candles.get_candles_by_pair(btc.internal_id)
    eth_price = strategy_universe.data_universe.candles.get_candles_by_pair(eth.internal_id)
    series = eth_price["close"] / btc_price["close"]  # Divide two series
    return series


def calculate_eth_btc_rsi(strategy_universe: TradingStrategyUniverse, mode: ExecutionMode, length: int):
    """Calculate x hours RSI for MATIC/ETH price used as the rebalancing factor."""
    eth_btc_series = calculate_eth_btc(strategy_universe, mode)
    return pandas_ta.rsi(eth_btc_series, length=length)


def calculate_resampled_rsi(pair_close_price_series: pd.Series, length: int, upsample: TimeBucket, shift: int):
    """Calculate x hours RSI for a particular trading pair"""
    resampled_close = resample_price_series(pair_close_price_series, upsample.to_pandas_timedelta(), shift=shift)
    return pandas_ta.rsi(resampled_close, length=length)


def calculate_resampled_eth_btc(strategy_universe: TradingStrategyUniverse, mode: ExecutionMode, upsample: TimeBucket, shift: int):
    """Calculate BTC/ETH price series for x hours."""
    pair_ids = get_strategy_trading_pairs(mode)
    eth = strategy_universe.get_pair_by_human_description(pair_ids[1])
    btc = strategy_universe.get_pair_by_human_description(pair_ids[0])
    eth_price = strategy_universe.data_universe.candles.get_candles_by_pair(eth.internal_id)
    btc_price = strategy_universe.data_universe.candles.get_candles_by_pair(btc.internal_id)
    resampled_eth = resample_price_series(eth_price["close"], upsample.to_pandas_timedelta(), shift=shift)
    resampled_btc = resample_price_series(btc_price["close"], upsample.to_pandas_timedelta(), shift=shift)
    series = resampled_eth / resampled_btc
    return series


def calculate_resampled_eth_btc_rsi(strategy_universe: TradingStrategyUniverse, mode: ExecutionMode, length: int, upsample: TimeBucket, shift: int):
     """Caclulate RSI for MATIC/ETH price series for x hours."""
     etc_btc = calculate_resampled_eth_btc(strategy_universe, mode, upsample, shift)
     return pandas_ta.rsi(etc_btc, length=length)


def calculate_shifted_sma(pair_close_price_series: pd.Series, length: int, upsample: TimeBucket, shift: int):
    resampled_close = resample_price_series(pair_close_price_series, upsample.to_pandas_timedelta(), shift=shift)
    return pandas_ta.sma(resampled_close, length=length)


def create_indicators(
    timestamp: datetime.datetime,
    parameters: StrategyParameters,
    strategy_universe: TradingStrategyUniverse,
    execution_context: ExecutionContext
):
    """Create indicators for this trading strategy.

    - Because we use non-standard candle time bucket, we do upsamplign from 1h candles
    """
    indicators = IndicatorSet()
    mode = execution_context.mode  # Switch between live trading and backtesting pairs
    upsample = parameters.target_time_bucket
    shift = parameters.clock_shift_bars
    indicators.add(
        "rsi", calculate_resampled_rsi,
        {"length": parameters.rsi_bars, "upsample": upsample, "shift": shift}
    )
    indicators.add(
        "eth_btc",
        calculate_resampled_eth_btc,
        {"upsample": parameters.target_time_bucket, "mode": mode, "shift": shift},
        source=IndicatorSource.strategy_universe
    )
    indicators.add(
        "eth_btc_rsi",
        calculate_resampled_eth_btc_rsi,
        {"length": parameters.eth_btc_rsi_bars, "mode": mode, "upsample": upsample, "shift": shift},
        source=IndicatorSource.strategy_universe
    )
    indicators.add(
        "sma",
        calculate_shifted_sma,
        {"length": parameters.regime_filter_ma_length, "upsample": upsample, "shift": shift}
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
        closed_positions = position_manager.get_closed_positions_for_pair(pair)
        has_no_position = not pair_open and len(closed_positions) == 0 and mode.is_live_trading()
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
                if rsi_cross_above or has_no_position:
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

    trades = []

    #
    # If we have cash stashed in Aave, withdraw
    #
    rebalance_trades = alpha_model.generate_rebalance_trades_and_triggers(
        position_manager,
        min_trade_threshold=parameters.rebalance_threshold * portfolio.get_total_equity(),
    )

    if len(rebalance_trades) > 0:
        # We simplify credit handling by always fully closing the credit position,
        # and then opening it in the next cycle when there are no rebalances
        if position_manager.is_any_credit_supply_position_open():
            current_credit_pos = position_manager.get_current_credit_supply_position()
            trades += position_manager.close_credit_supply_position(current_credit_pos)

    trades += rebalance_trades

    #
    # Credit supply
    #
    if len(trades) == 0 and parameters.use_credit:
        # Check if we can push any extra cash to Aave,
        # as this cycle does not see any rebalancing trades
        cash_to_deposit = position_manager.get_current_cash() * 0.99
        trades += position_manager.add_cash_to_credit_supply(cash_to_deposit)
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
                pair=btc_pair,
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
                pair=btc_pair,
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
                pair=btc_pair,
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
                pair=btc_pair,
            )

        state.visualisation.add_calculations(timestamp, alpha_model.to_dict())  # Record alpha model thinking

    position_manager.log(
        f"BTC RSI: {current_rsi_values[btc_pair]}, BTC RSI yesterday: {previous_rsi_values[btc_pair]}",
    )

    return trades


def get_strategy_trading_pairs(execution_mode: ExecutionMode) -> list[HumanReadableTradingPairDescription]:
    """Switch between backtest and live trading pairs.

    Because the live trading DEX venues do not have enough history (< 2 years)
    for meaningful backtesting, we test with Binance CEX data.
    """

    use_binance = Parameters.use_binance

    if use_binance:
        assert not execution_mode.is_live_trading(), "Binance market data can be only used for backtesting"

    if use_binance:
        trading_pairs = [
             (ChainId.centralised_exchange, "binance", "BTC", "USDT"),
             (ChainId.centralised_exchange, "binance", "ETH", "USDT"),
        ]

    else:
        # Live trading, backtesting DEX data

        # Live trade
        trading_pairs = [
            (ChainId.polygon, "quickswap", "WBTC", "WETH", 0.0030),
            (ChainId.polygon, "uniswap-v3", "WETH", "USDC", 0.0005),
            (ChainId.polygon, "quickswap", "WETH", "USDC", 0.0030),  # keep this temporarily here for pricing
        ]

    return trading_pairs


def get_lending_reserves(mode: ExecutionMode) -> list[LendingReserveDescription]:
    """Get lending reserves the strategy needs."""

    use_binance = Parameters.use_binance

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


#
# Strategy metadata.
#
# Displayed in the user interface.
#

tags = {StrategyTag.beta, StrategyTag.live}

name = "ETH-BTC-USDC momentum"

short_description = "ETH and BTC momentum strategy to maximize gains in bull market and avoid losses in bear market, on Polygon"

icon = "https://tradingstrategy.ai/avatars/polygon-eth-spot-short.webp"

long_description = """
# Strategy description

This strategy is a momentum and breakout strategy.

- The strategy trades ETH and BTC over long term time horizon, doing only few trades per a year.
- The strategy delivers similar profits as buying and holding ETH and BTC, but with much less severe drawdowns.
- The strategy performs well in long-term Bitcoin [bull market](https://tradingstrategy.ai/glossary/bull-market).
- In [bear](https://tradingstrategy.ai/glossary/bear-market) and sideways markets the strategy does not perform well.
- It is based on [RSI technical indicator](https://tradingstrategy.ai/glossary/relative-strength-index-rsi), the strategy is buying when others are buying, and the strategy is selling when others are selling.

**Past performance is not indicative of future results**.

## Assets and trading venues

- The strategy trades only spot market
- We trade two trading asset: ETH and BTC
- The strategy keeps reserves in USDC stablecoin
- The trading happens on QuickSwap and Uniswap on Polygon blockchain
- The strategy decision cycle is daily rebalances

## Backtesting

The backtesting was performed with Binance ETH-USDT and BTC-USDT data of 2019-2024.

- [See backtesting results](./backtest)
- [Read more about what is backtesting](https://tradingstrategy.ai/glossary/backtest).

The backtesting trading venue (Binance) is different from the live trading venue (Quickswap, Uniswap), because DEX markets
do not have long enough history to result to a meaningful backtest.

The backtesting period saw one bull market rally that is unlikely to repeat in the same magnitude 
for the assets we trade.

Past peformance is no guarantee of future results. Like with manual trading, automated trading is unlikely to be perfect.
There will be variance in the range of 30% - 50% in the results.

## Profit

The backtested results indicate **80%** estimated yearly profit ([CAGR](https://tradingstrategy.ai/glossary/compound-annual-growth-rate-cagr)). 

This is similar profit as you would get by buying and holding BTC or ETH.

## Risk

This strategy has **-30%** backtested [maximum drawdown](https://tradingstrategy.ai/glossary/maximum-drawdown).
This is much less severe compared to buy and hold, making the strategy less risky than buy and hold.

For further understanding the key aspescts of risks
- The strategy does not use any leverage
- The strategy trades only established, highly liquid, trading pairs which are unlikely to go zero 

## Benchmark

For the same backtesting period, here are some benchmark of performance of different assets and indices:

|                              | CAGR | Maximum drawdown | Sharpe |
|------------------------------|------|------------------|--------|
| This strategy                | 84%  | -34%             | 1.78   |
| SP500 (20 years)             | 11%  | -33%             | 0.72   |
| Bitcoin (backtesting period) | 76%  | -76%             | 1.17   |
| Ether (backtesting period)   | 85%  | -79%             | 1.18   |


Sources:

- [Our strategy](./backtest)
- [Buy and hold BTC](./backtest)
- [Buy and hold ETH](./backtest)
- [SP500 stock index](https://curvo.eu/backtest/en/portfolio/s-p-500--NoIgygZACgBArABgSANMUBJAokgQnXAWQCUEAOAdlQEYBdeoA?config=%7B%22periodStart%22%3A%222004-02%22%7D)

## Trading frequency

The strategy is very slow moving macro-like strategy.

This strategy is estimated to to rebalance every **20 days** and enter/exit positions even less frequently. 

## Robustness

This strategy does not have good robustness tests available.

## Updates

This is one of the early, simple, strategies deployed on Trading Strategy protocol.

It is likely this strategy will be replaced with a newer, more robust, more optimised, version in some point of the future.
[Follow Trading Strategy for updates](https://tradingstrategy.ai/community) as you need to move your balance to a new strategy.

## Further information

- Any questions are welcome in [the Discord community chat](https://tradingstrategy.ai/community)
- See the blog post [on how this strategy is constructed](https://tradingstrategy.ai/blog/outperfoming-eth) 

"""
