"""MATIC-ETH-USDC rebalance strategy.

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
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet, IndicatorSource
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.tag import StrategyTag
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, load_partial_data
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradeexecutor.strategy.weighting import weight_passthrouh
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.utils.groupeduniverse import resample_price_series

trading_strategy_engine_version = "0.5"

tags = {StrategyTag.beta}

name = "ETH-BTC-USDC momentum"

short_description = "A momentum strategy for ETH-USDC and BTC-USDC pairs based on RSI indicators"

long_description = """
- [See the blog post for more details](https://tradingstrategy.ai/blog/outperfoming-eth) on how this strategy is constructed
"""

# The pairs we are rading
pair_ids = [
    (ChainId.polygon, "quickswap", "WBTC", "WETH", 0.0030),
    (ChainId.polygon, "uniswap-v3", "WMATIC", "USDC", 0.0005),
]


# See v37-matic-eth-robustness search for parameters details
class Parameters:
    """Parameteres for this strategy.

    - Collect parameters used for this strategy here

    - Both live trading and backtesting parameters
    """


    cycle_duration = CycleDuration.cycle_8h  # Run decide_trades() every 8h
    source_time_bucket = TimeBucket.h1  # Use 1h candles as the raw data
    target_time_bucket = TimeBucket.h8  # Create synthetic 8h candles
    clock_shift_bars = 0  # Do not do shifted candles

    rsi_bars = 12  # Number of bars to calculate RSI for each tradingbar
    eth_btc_rsi_bars = 5  # Number of bars for the momentum factor
    rsi_entry = 67  # Single pair entry level - when RSI crosses above this value open a position
    rsi_exit = 60  # Single pair exit level - when RSI crosses below this value exit a position
    allocation = 0.98  # How much cash allocate for volatile positions
    rebalance_threshold = 0.10  # How much position mix % must change when we rebalance between two open positions
    initial_cash = 10_000  # Backtesting start cash
    trailing_stop_loss = 0.95  # Trailing stop loss as 1 - x
    trailing_stop_loss_activation_level = 1.07  # How much above opening price we must be before starting to use trailing stop loss
    stop_loss = 0.90  # Hard stop loss when opening a new position
    momentum_exponent = 3.5  # How much momentum we capture when rebalancing between open positions

    #
    # Live trading only
    #
    chain_id = ChainId.polygon
    routing = TradeRouting.default  # Pick default routes for trade execution
    required_history_period = target_time_bucket.to_timedelta() * rsi_bars * 2  # How much data a live trade execution needs to load to be able to calculate indicators

    #
    # Backtesting only
    #

    backtest_start = datetime.datetime(2021, 1, 1)
    backtest_end = datetime.datetime(2024, 3, 15)
    stop_loss_time_bucket = TimeBucket.h1  # use 1h close as the stop loss signal


def calculate_eth_btc(strategy_universe: TradingStrategyUniverse):
    """Calculate ETH/BTC price used as a rebalance factor."""
    eth = strategy_universe.get_pair_by_human_description(pair_ids[1])
    btc = strategy_universe.get_pair_by_human_description(pair_ids[0])
    btc_price = strategy_universe.data_universe.candles.get_candles_by_pair(btc.internal_id)
    eth_price = strategy_universe.data_universe.candles.get_candles_by_pair(eth.internal_id)
    series = eth_price["close"] / btc_price["close"]  # Divide two series
    return series


def calculate_eth_btc_rsi(strategy_universe: TradingStrategyUniverse, length: int):
    """Calculate x hours RSI for MATIC/ETH price used as the rebalancing factor."""
    eth_btc_series = calculate_eth_btc(strategy_universe)
    return pandas_ta.rsi(eth_btc_series, length=length)


def calculate_resampled_rsi(pair_close_price_series: pd.Series, length: int, upsample: TimeBucket, shift: int):
    """Calculate x hours RSI for a particular trading pair"""
    resampled_close = resample_price_series(pair_close_price_series, upsample.to_pandas_timedelta(), shift=shift)
    return pandas_ta.rsi(resampled_close, length=length)


def calculate_resampled_eth_btc(strategy_universe: TradingStrategyUniverse, upsample: TimeBucket, shift: int):
    """Calculate BTC/ETH price series for x hours."""
    eth = strategy_universe.get_pair_by_human_description(pair_ids[1])
    btc = strategy_universe.get_pair_by_human_description(pair_ids[0])
    eth_price = strategy_universe.data_universe.candles.get_candles_by_pair(eth.internal_id)
    btc_price = strategy_universe.data_universe.candles.get_candles_by_pair(btc.internal_id)
    resampled_eth = resample_price_series(eth_price["close"], upsample.to_pandas_timedelta(), shift=shift)
    resampled_btc = resample_price_series(btc_price["close"], upsample.to_pandas_timedelta(), shift=shift)
    series = resampled_eth / resampled_btc
    return series


def calculate_resampled_eth_btc_rsi(strategy_universe: TradingStrategyUniverse, length: int, upsample: TimeBucket, shift: int):
     """Caclulate RSI for MATIC/ETH price series for x hours."""
     etc_btc = calculate_resampled_eth_btc(strategy_universe, upsample, shift)
     return pandas_ta.rsi(etc_btc, length=length)


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
    indicators.add(
        "rsi", calculate_resampled_rsi,
        {"length": parameters.rsi_bars, "upsample": parameters.target_time_bucket, "shift": parameters.clock_shift_bars}
    )
    indicators.add(
        "eth_btc",
        calculate_resampled_eth_btc,
        {"upsample": parameters.target_time_bucket, "shift": parameters.clock_shift_bars},
        source=IndicatorSource.strategy_universe
    )
    indicators.add(
        "eth_btc_rsi",
        calculate_resampled_eth_btc_rsi,
        {"length": parameters.eth_btc_rsi_bars, "upsample": parameters.target_time_bucket, "shift": parameters.clock_shift_bars},
        source=IndicatorSource.strategy_universe
    )
    return indicators


def decide_trades(
    input: StrategyInput,
) -> list[TradeExecution]:
    """Strategy entry/exit logic.

    - Indicator reading for the input]
    - The trade logic
    - Visualisation for the strategy diagnostics
    """

    # Resolve our pair metadata for our two pair strategy
    parameters = input.parameters
    position_manager = input.get_position_manager()
    state = input.state
    timestamp = input.timestamp
    indicators = input.indicators
    clock_shift = parameters.clock_shift_bars * parameters.source_time_bucket.to_pandas_timedelta()

    alpha_model = AlphaModel(input.timestamp)
    btc_pair = position_manager.get_trading_pair(pair_ids[0])
    eth_pair = position_manager.get_trading_pair(pair_ids[1])
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

    # Create a number (0...1) how much momentum the trading pair has,
    # with the effect exaggerated with an exponent.
    #
    # Anything below RSI 50 scales towards 0 and above scales towards infinity.
    #
    # 0.5 = RSI 0
    # 1.5 = RSI 100
    # 1 = RSI 50
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
            rsi_cross_above = current_rsi_values[pair] >= parameters.rsi_entry and previous_rsi_values[pair] < parameters.rsi_entry
            rsi_cross_below = current_rsi_values[pair] < parameters.rsi_exit and previous_rsi_values[pair] > parameters.rsi_exit

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
    if parameters.trailing_stop_loss_activation_level is not None:
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

        if current_rsi_values[btc_pair]:
            visualisation.plot_indicator(
                timestamp,
                f"RSI BTC",
                PlotKind.technical_indicator_detached,
                current_rsi_values[btc_pair],
                colour="orange",
            )

            # Low (vertical line)
            visualisation.plot_indicator(
                timestamp,
                f"RSI low trigger",
                PlotKind.technical_indicator_overlay_on_detached,
                parameters.rsi_exit,
                detached_overlay_name=f"RSI BTC",
                plot_shape=PlotShape.horizontal_vertical,
                colour="red",
                label=PlotLabel.hidden,
            )

            # High (vertical line)
            visualisation.plot_indicator(
                timestamp,
                f"RSI high trigger",
                PlotKind.technical_indicator_overlay_on_detached,
                parameters.rsi_entry,
                detached_overlay_name=f"RSI BTC",
                plot_shape=PlotShape.horizontal_vertical,
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

    # Load data for our trading pair whitelist
    if execution_context.mode.is_backtesting():
        # For backtesting, we use a specific time range from the strategy parameters
        start_at = universe_options.start_at
        end_at = universe_options.end_at
        required_history_period = None
        stop_loss_time_bucket = Parameters.stop_loss_time_bucket
    else:
        start_at = None
        end_at = None
        required_history_period = datetime.timedelta(days=Parameters.required_history_period)
        stop_loss_time_bucket = None

    dataset = load_partial_data(
        client=client,
        time_bucket=Parameters.source_time_bucket,
        pairs=pair_ids,
        execution_context=execution_context,
        universe_options=universe_options,
        liquidity=False,
        stop_loss_time_bucket=stop_loss_time_bucket,
        start_at=start_at,
        end_at=end_at,
        required_history_period=required_history_period,
    )

    # Construct a trading universe from the loaded data,
    # and apply any data preprocessing needed before giving it
    # to the strategy and indicators
    universe = TradingStrategyUniverse.create_from_dataset(
        dataset,
        reserve_asset="USDC",
        forward_fill=True,
    )
    return universe
