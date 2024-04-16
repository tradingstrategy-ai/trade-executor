"""Test of visualising current strategy state for a multipair strategy."""

import logging
import os
import random
import datetime
import webbrowser
from pathlib import Path
from typing import List, Dict

import pytest

import pandas as pd
from pandas_ta import bbands
from pandas_ta.momentum import rsi

from tradingstrategy.universe import Universe
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.exchange import ExchangeUniverse

from tradeexecutor.backtest.backtest_runner import run_backtest_inline
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.visualisation import PlotKind
from tradeexecutor.strategy.execution_context import unit_test_execution_context
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, \
    create_pair_universe_from_code
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange, generate_simple_routing_model
from tradeexecutor.testing.synthetic_price_data import generate_ohlcv_candles
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.state.state import State
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.visual.strategy_state import draw_multi_pair_strategy_state
from tradeexecutor.visual.image_output import open_plotly_figure_in_browser
from tradeexecutor.strategy.trading_strategy_universe import translate_trading_pair


#
# Strategy properties
# 

# How our trades are routed.
TRADE_ROUTING = TradeRouting.uniswap_v3_usdc

# How often the strategy performs the decide_trades cycle.
TRADING_STRATEGY_CYCLE = CycleDuration.cycle_1d

# Time bucket for our candles
CANDLE_TIME_BUCKET = TimeBucket.d1

# Candle time granularity we use to trigger stop loss checks
STOP_LOSS_TIME_BUCKET = TimeBucket.m15

# Strategy keeps its cash in USDC
RESERVE_CURRENCY = ReserveCurrency.usdc

# Which trading pair we are backtesting on.
# We are using historical data feeds with fixed prices
# to test as much as backtesting history as possible
TRADING_PAIRS = [
    (ChainId.polygon, "uniswap-v3", "WETH", "USDC", 0.0005), # Ether-USD Coin (PoS) https://tradingstrategy.ai/trading-view/polygon/uniswap-v3/eth-usdc-fee-5
    (ChainId.polygon, "uniswap-v3", "WMATIC", "USDC", 0.0005), # Wrapped Matic-USD Coin (PoS) https://tradingstrategy.ai/trading-view/polygon/uniswap-v3/matic-usdc-fee-5
    (ChainId.polygon, "uniswap-v3", "XSGD", "USDC", 0.0005), # XSGD-USD Coin (PoS) https://tradingstrategy.ai/trading-view/polygon/uniswap-v3/xsgd-usdc-fee-5
]


# How much % of the cash to put on a single trade
POSITION_SIZE = 0.30

# Start with this amount of USD
INITIAL_DEPOSIT = 50_000

#
# Strategy inputs
#

# How many candles we load in the decide_trades() function for calculating indicators
LOOKBACK_CANDLE_COUNT = 90

# How many candles we use to calculate the Relative Strength Indicator
RSI_LENGTH = 14

# RSI must be above this value to open a new position.
RSI_THRESHOLD = 65

# What's the moving average length in candles for Bollinger bands
MOVING_AVERAGE_LENGTH = 20

# Bollinger band's standard deviation
STDDEV = 2.0

# Backtest range
#
# Note that for this example notebook we deliberately choose a very short period,
# as the backtest completes faster, charts are more readable
# and tables shorter for the demostration.
#
START_AT = datetime.datetime(2022, 7, 1)
START_AT_DATA = datetime.datetime(2022, 3, 1)

# Backtest range
END_AT = datetime.datetime(2023, 6, 6)

# Stop loss relative to the mid price during the time when the position is opened
#
# If the price drops below this level, trigger a stop loss
STOP_LOSS_PCT = 0.95

# What is the trailing stop loss level
TRAILING_STOP_LOSS_PCT = 0.993

# Activate trailing stop loss when this level is reached
TRAILING_STOP_LOSS_ACTIVATION_LEVEL=1.01


def decide_trades(
        timestamp: pd.Timestamp,
        universe: Universe,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: Dict) -> List[TradeExecution]:
    """The brain function to decide the trades on each trading strategy cycle.

    - Reads incoming execution state (positions, past trades)

    - Reads the current universe (candles)

    - Decides what trades to do next, if any, at current timestamp.

    - Outputs strategy thinking for visualisation and debug messages

    :param timestamp:
        The Pandas timestamp object for this cycle. Matches
        TRADING_STRATEGY_CYCLE division.
        Always truncated to the zero seconds and minutes, never a real-time clock.

    :param universe:
        Trading universe that was constructed earlier.

    :param state:
        The current trade execution state.
        Contains current open positions and all previously executed trades, plus output
        for statistics, visualisation and diangnostics of the strategy.

    :param pricing_model:
        Pricing model can tell the buy/sell price of the particular asset at a particular moment.

    :param cycle_debug_data:
        Python dictionary for various debug variables you can read or set, specific to this trade cycle.
        This data is discarded at the end of the trade cycle.

    :return:
        List of trade instructions in the form of :py:class:`TradeExecution` instances.
        The trades can be generated using `position_manager` but strategy could also hand craft its trades.
    """

  # Create a position manager helper class that allows us easily to create
    # opening/closing trades for different positions
    position_manager = PositionManager(timestamp, universe, state, pricing_model)

    # The array of trades we are going to perform in this cycle.
    trades = []

    # How much cash we have in a hand
    cash = state.portfolio.get_cash()

    # Set up the 
    # Load candle data for this decision frame,
    # We look back LOOKBACK_WINDOW candles.
    # Timestamp is the current open time, always make decision based on the last 
    # candle close, so adjust the end time minus one candle.
    start = timestamp - (LOOKBACK_CANDLE_COUNT * CANDLE_TIME_BUCKET.to_pandas_timedelta())
    end = timestamp - CANDLE_TIME_BUCKET.to_pandas_timedelta()  

    # Fetch candle data for all pairs in a single go
    candle_data = universe.candles.iterate_samples_by_pair_range(start, end)

    visualisation = state.visualisation

    for pair_id, candles in candle_data:

        # Convert raw trading pair data to strategy execution format
        pair_data = universe.pairs.get_pair_by_id(pair_id)
        pair = translate_trading_pair(pair_data)

        # Here we manipulate the pair trading fee.
        # A live trading would happen on Polygon Uniswap v3 ETH-USDC pool with 0.05% LP fee.
        # But this pool was deployed only couple of weeks back, so we do not have backtesting history for it.
        # Thus, we are backtesting with QuickSwap ETH-USDC pair that has 0.30% LP fee tier, which
        # we need to bump down to reflect the live trading situation.
        # Drop the fee to 5 BPSs.
        
        #pair.fee = 0.0005

        # We have data for open, high, close, etc.
        # We only operate using candle close values in this strategy.
        close_prices = candles["close"]
        

       
        # Calculate RSI for candle close
        # https://tradingstrategy.ai/docs/programming/api/technical-analysis/momentum/help/pandas_ta.momentum.rsi.html#rsi
        rsi_bars = rsi(close_prices, length=RSI_LENGTH)

        if rsi_bars is None:
            # Lookback buffer does not have enough candles yet
            continue

        current_rsi = rsi_bars[-1]

        price_latest = close_prices.iloc[-1]

        # Calculate Bollinger Bands with a 20-day SMA and 2 standard deviations using pandas_ta
        # See documentation here https://tradingstrategy.ai/docs/programming/api/technical-analysis/volatility/help/pandas_ta.volatility.bbands.html#bbands
        bollinger_bands = bbands(close_prices, length=20, std=2)

        if bollinger_bands is None:
            # Lookback buffer does not have enough candles yet
            continue

        bb_upper = bollinger_bands["BBU_20_2.0"]  # Upper deviation
        bb_lower = bollinger_bands["BBL_20_2.0"]  # Lower deviation
        bb_mid = bollinger_bands["BBM_20_2.0"]  # Same as moving average 

        position_for_pair = position_manager.get_current_position_for_pair(pair)

        if not position_for_pair:
            # No open positions for the current pair, decide if BUY in this cycle.
            # We buy if the price on the daily chart closes above the upper Bollinger Band.
            if price_latest > bb_upper.iloc[-1] and current_rsi >= RSI_THRESHOLD:
                # We are dividing our bets 1/3 equally among all three pairs
                buy_amount = cash * POSITION_SIZE #* 0.33 
                trades += position_manager.open_spot(pair, buy_amount, stop_loss_pct=STOP_LOSS_PCT)

        else:
        # We have an open position, decide if SELL in this cycle.
        # We close the position when the price closes below the 20-day moving average.        
            if price_latest < bb_mid.iloc[-1]:
                trades += position_manager.close_position(position_for_pair)
        
            # Check if we have reached out level where we activate trailing stop loss
            
            if price_latest >= position_for_pair.get_opening_price() * TRAILING_STOP_LOSS_ACTIVATION_LEVEL:
                position_for_pair.trailing_stop_loss_pct = TRAILING_STOP_LOSS_PCT
                position_for_pair.stop_loss = float(price_latest * TRAILING_STOP_LOSS_PCT)
                    
        # Visualise our technical indicators.
        pair_slug = f"{pair.base.token_symbol}/{pair.quote.token_symbol}"
        
        # bollinger bands
        visualisation.plot_indicator(timestamp, f"{pair_slug} BB upper", PlotKind.technical_indicator_on_price, bb_upper.iloc[-1], pair=pair)
        visualisation.plot_indicator(timestamp, f"{pair_slug} BB lower", PlotKind.technical_indicator_on_price, bb_lower.iloc[-1], pair=pair)        
        visualisation.plot_indicator(timestamp, f"{pair_slug} moving avg", PlotKind.technical_indicator_on_price, bb_mid.iloc[-1], pair=pair)
        
        # rsi
        visualisation.plot_indicator(timestamp, f"{pair_slug} RSI", PlotKind.technical_indicator_detached, current_rsi, pair=pair)

    return trades


@pytest.fixture(scope="module")
def mock_exchange():
    return generate_exchange(
        exchange_id=random.randint(1, 1000),
        chain_id=ChainId.ethereum,
        address=generate_random_ethereum_address(),
        exchange_slug="uniswap-mock",
    )


@pytest.fixture(scope="module")
def strategy_universe(mock_exchange) -> TradingStrategyUniverse:

    # Set up fake assets
    mock_chain_id = ChainId.ethereum
    
    usdc = AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "USDC", 6, 1)
    
    # limited to use same quote token for all pairs
    # because of generate_simple_routing_model
    # see https://github.com/tradingstrategy-ai/trade-executor/blob/fc1868b1369671b2c76238f5b334a1af4b339b66/tradeexecutor/testing/synthetic_exchange_data.py#L31
    usdc = AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "USDC", 6, 1)
    weth = AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "WETH", 18, 2)
    weth_usdc = TradingPairIdentifier(
        weth,
        usdc,
        generate_random_ethereum_address(),
        mock_exchange.address,
        internal_id=99,
        internal_exchange_id=mock_exchange.exchange_id)
    
    pepe = AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "PEPE", 18, 3)
    pepe_usdc = TradingPairIdentifier(
        pepe,
        usdc,
        generate_random_ethereum_address(),
        mock_exchange.address,
        internal_id=100,
        internal_exchange_id=mock_exchange.exchange_id)
    
    bob = AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "BOB", 18, 4)
    bob_usdc = TradingPairIdentifier(
        bob,
        usdc,
        generate_random_ethereum_address(),
        mock_exchange.address,
        internal_id=101,
        internal_exchange_id=mock_exchange.exchange_id)
    
    time_bucket = CANDLE_TIME_BUCKET

    pair_universe = create_pair_universe_from_code(mock_chain_id, [weth_usdc, pepe_usdc, bob_usdc])
    exchange_universe = ExchangeUniverse(exchanges={mock_exchange.exchange_id: mock_exchange})

    candles_weth_usdc = generate_ohlcv_candles(time_bucket, START_AT_DATA, END_AT, pair_id=weth_usdc.internal_id, random_seed=1)
    candles_pepe_usdc = generate_ohlcv_candles(time_bucket, START_AT_DATA, END_AT, pair_id=pepe_usdc.internal_id, random_seed = 2)
    candles_bob_usdc = generate_ohlcv_candles(time_bucket, START_AT_DATA, END_AT, pair_id=bob_usdc.internal_id, random_seed = 4)
    
    candle_universe = GroupedCandleUniverse.create_from_multiple_candle_datafarames([candles_weth_usdc, candles_pepe_usdc, candles_bob_usdc])

    universe = Universe(
        time_bucket=time_bucket,
        chains={mock_chain_id},
        exchanges={mock_exchange},
        exchange_universe=exchange_universe,
        pairs=pair_universe,
        candles=candle_universe,
        liquidity=None
    )

    return TradingStrategyUniverse(
        data_universe=universe,
        backtest_stop_loss_candles=candle_universe,
        backtest_stop_loss_time_bucket=time_bucket,
        reserve_assets=[usdc]
    )


def test_visualise_strategy_state(
        logger: logging.Logger,
        strategy_universe,
    ):
    """Visualise strategy state as a bunch inline images."""

    routing_model = generate_simple_routing_model(strategy_universe)

    # Run the test
    state, strategy_universe, debug_dump = run_backtest_inline(
        start_at=START_AT,
        end_at=END_AT,
        client=None,  # None of downloads needed, because we are using synthetic data
        cycle_duration=TRADING_STRATEGY_CYCLE,
        decide_trades=decide_trades,
        create_trading_universe=None,
        universe=strategy_universe,
        initial_deposit=INITIAL_DEPOSIT,
        reserve_currency=ReserveCurrency.busd,
        trade_routing=TradeRouting.user_supplied_routing_model,
        routing_model=routing_model,
        log_level=logging.WARNING,
        allow_missing_fees=True,
    )

    universe = strategy_universe
    image = draw_multi_pair_strategy_state(state, unit_test_execution_context, universe)

    assert len(image.data) == 27
    assert len(image._grid_ref) == 6
    assert image.data[0]['x'][0] == datetime.datetime(2023,4,3,0,0)
    assert image.data[0]['x'][-1] == datetime.datetime(2023,6,5,0,0)

    image_no_detached = draw_multi_pair_strategy_state(state, unit_test_execution_context, universe, detached_indicators=False)

    assert len(image_no_detached.data) == 24
    assert len(image_no_detached._grid_ref) == 3
    assert image_no_detached.data[0]['x'][0] == datetime.datetime(2023,4,3,0,0)
    assert image_no_detached.data[0]['x'][-1] == datetime.datetime(2023,6,5,0,0)

    image_no_indicators = draw_multi_pair_strategy_state(state, unit_test_execution_context, universe, technical_indicators=False)

    assert len(image_no_indicators.data) == 15
    assert len(image_no_indicators._grid_ref) == 3

    # Test the image on a local screen
    # using a web brower
    if os.environ.get("SHOW_IMAGE"):
        open_plotly_figure_in_browser(image, height=2000, width=1000)
        open_plotly_figure_in_browser(image_no_detached, height=2000, width=1000)
        open_plotly_figure_in_browser(image_no_indicators, height=2000, width=1000)


def test_visualise_strategy_state_overriden_pairs(
    logger: logging.Logger,
    strategy_universe,
):
    """Visualise strategy state as a bunch inline images."""

    routing_model = generate_simple_routing_model(strategy_universe)

    # Run the test
    state, strategy_universe, debug_dump = run_backtest_inline(
        start_at=START_AT,
        end_at=END_AT,
        client=None,  # None of downloads needed, because we are using synthetic data
        cycle_duration=TRADING_STRATEGY_CYCLE,
        decide_trades=decide_trades,
        create_trading_universe=None,
        universe=strategy_universe,
        initial_deposit=INITIAL_DEPOSIT,
        reserve_currency=ReserveCurrency.busd,
        trade_routing=TradeRouting.user_supplied_routing_model,
        routing_model=routing_model,
        log_level=logging.WARNING,
        allow_missing_fees=True,
    )

    # visualise only one pair
    state.visualisation.set_visualised_pairs([
        strategy_universe.data_universe.pairs.get_pair_by_id(99),
    ])

    image = draw_multi_pair_strategy_state(state, unit_test_execution_context, strategy_universe)

    assert len(image.data) == 9
    assert len(image._grid_ref) == 2
    assert image.data[0]['x'][0] == datetime.datetime(2023,4,3,0,0)
    assert image.data[0]['x'][-1] == datetime.datetime(2023,6,5,0,0)

    image_no_detached = draw_multi_pair_strategy_state(state, unit_test_execution_context, strategy_universe, detached_indicators=False)

    assert len(image_no_detached.data) == 8
    assert len(image_no_detached._grid_ref) == 1
    assert image_no_detached.data[0]['x'][0] == datetime.datetime(2023,4,3,0,0)
    assert image_no_detached.data[0]['x'][-1] == datetime.datetime(2023,6,5,0,0)

    image_no_indicators = draw_multi_pair_strategy_state(state, unit_test_execution_context, strategy_universe, technical_indicators=False)

    assert len(image_no_indicators.data) == 5
    assert len(image_no_indicators._grid_ref) == 1

    # Test the image on a local screen
    # using a web brower
    if os.environ.get("SHOW_IMAGE"):
        open_plotly_figure_in_browser(image, height=2000, width=1000)
        open_plotly_figure_in_browser(image_no_detached, height=2000, width=1000)
        open_plotly_figure_in_browser(image_no_indicators, height=2000, width=1000)
