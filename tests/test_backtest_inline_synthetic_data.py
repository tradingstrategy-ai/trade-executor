"""Test backtesting where decide_trades and create_universe functions are passed directly.

"""
import logging
import random
import datetime
from typing import List, Dict

import pytest

import pandas as pd
from pandas_ta.overlap import ema

from tradeexecutor.analysis.trade_analyser import build_trade_analysis, expand_timeline, TimelineRowStylingMode, TradeAnalysis, PositionSummary
from tradeexecutor.backtest.backtest_runner import run_backtest_inline
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.visualisation import PlotKind
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, \
    create_pair_universe_from_code
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange, generate_simple_routing_model
from tradeexecutor.testing.synthetic_price_data import generate_ohlcv_candles
from tradeexecutor.visual.benchmark import visualise_benchmark
from tradingstrategy.candle import GroupedCandleUniverse
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.state.state import State
from tradingstrategy.universe import Universe
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.strategy_module import pregenerated_create_trading_universe
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.strategy_type import StrategyType
from tradeexecutor.strategy.default_routing_options import TradeRouting


# relative tolerance for floating point tests
APPROX_REL = 1e-6


# How much of the cash to put on a single trade
position_size = 0.10

#
# Strategy thinking specific parameter
#

batch_size = 90

slow_ema_candle_count = 20

fast_ema_candle_count = 5


def decide_trades(
        timestamp: pd.Timestamp,
        universe: Universe,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: Dict) -> List[TradeExecution]:
    """The brain function to decide the trades on each trading strategy cycle."""

    # The pair we are trading
    pair = universe.pairs.get_single()

    # How much cash we have in the hand
    cash = state.portfolio.get_current_cash()

    # Get OHLCV candles for our trading pair as Pandas Dataframe.
    # We could have candles for multiple trading pairs in a different strategy,
    # but this strategy only operates on single pair candle.
    # We also limit our sample size to N latest candles to speed up calculations.
    candles: pd.DataFrame = universe.candles.get_single_pair_data(timestamp, sample_count=batch_size)

    # We have data for open, high, close, etc.
    # We only operate using candle close values in this strategy.
    close = candles["close"]

    # Calculate exponential moving averages based on slow and fast sample numbers.
    # https://github.com/twopirllc/pandas-ta
    slow_ema_series = ema(close, length=slow_ema_candle_count)
    fast_ema_series = ema(close, length=fast_ema_candle_count)

    if slow_ema_series is None or fast_ema_series is None:
        # Cannot calculate EMA, because not enough samples in backtesting.
        # Return no trades made.
        return []

    slow_ema = slow_ema_series.iloc[-1]
    fast_ema = fast_ema_series.iloc[-1]

    # Get the last close price from close time series
    # that's Pandas's Series object
    # https://pandas.pydata.org/docs/reference/api/pandas.Series.iat.html
    current_price = close.iloc[-1]

    # List of any trades we decide on this cycle.
    # Because the strategy is simple, there can be
    # only zero (do nothing) or 1 (open or close) trades
    # decides
    trades = []

    # Create a position manager helper class that allows us easily to create
    # opening/closing trades for different positions
    position_manager = PositionManager(timestamp, universe, state, pricing_model)

    if current_price >= slow_ema:
        # Entry condition:
        # Close price is higher than the slow EMA
        if not position_manager.is_any_open():
            buy_amount = cash * position_size
            trades += position_manager.open_1x_long(pair, buy_amount)
    elif fast_ema >= slow_ema:
        # Exit condition:
        # Fast EMA crosses slow EMA
        if position_manager.is_any_open():
            trades += position_manager.close_all()

    # Visualize strategy
    # See available Plotly colours here
    # https://community.plotly.com/t/plotly-colours-list/11730/3?u=miohtama
    visualisation = state.visualisation
    visualisation.plot_indicator(timestamp, "Slow EMA", PlotKind.technical_indicator_on_price, slow_ema, colour="darkblue")
    visualisation.plot_indicator(timestamp, "Fast EMA", PlotKind.technical_indicator_on_price, fast_ema, colour="mediumpurple")

    return trades


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request, mute_requests=False)


@pytest.fixture(scope="module")
def universe() -> TradingStrategyUniverse:
    """Set up a mock universe."""

    start_at = datetime.datetime(2021, 6, 1)
    end_at = datetime.datetime(2022, 1, 1)

    # Set up fake assets
    mock_chain_id = ChainId.ethereum
    mock_exchange = generate_exchange(
        exchange_id=random.randint(1, 1000),
        chain_id=mock_chain_id,
        address=generate_random_ethereum_address())
    usdc = AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "USDC", 6, 1)
    weth = AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "WETH", 18, 2)
    weth_usdc = TradingPairIdentifier(
        weth,
        usdc,
        generate_random_ethereum_address(),
        mock_exchange.address,
        internal_id=random.randint(1, 1000),
        internal_exchange_id=mock_exchange.exchange_id,
        fee=0.0030,
    )

    time_bucket = TimeBucket.d1

    pair_universe = create_pair_universe_from_code(mock_chain_id, [weth_usdc])

    candles = generate_ohlcv_candles(time_bucket, start_at, end_at, pair_id=weth_usdc.internal_id)
    candle_universe = GroupedCandleUniverse.create_from_single_pair_dataframe(candles)

    universe = Universe(
        time_bucket=time_bucket,
        chains={mock_chain_id},
        exchanges={mock_exchange},
        pairs=pair_universe,
        candles=candle_universe,
        liquidity=None
    )

    return TradingStrategyUniverse(universe=universe, reserve_assets=[usdc])


def test_ema_on_universe(universe: TradingStrategyUniverse):
    """Calculate exponential moving average on single pair candle universe."""
    start_timestamp = pd.Timestamp("2021-6-1")
    batch_size = 20
    candles = universe.universe.candles.get_single_pair_data(start_timestamp, sample_count=batch_size, allow_current=True)
    assert len(candles) == 1

    # Not enough data to calculate EMA - we haave only 1 sample
    ema_20_series = ema(candles["close"], length=20)
    assert ema_20_series is None

    end_timestamp = pd.Timestamp("2021-12-31")
    candles = universe.universe.candles.get_single_pair_data(end_timestamp, sample_count=batch_size, allow_current=True)
    assert len(candles) == batch_size

    ema_20_series = ema(candles["close"], length=20)
    assert pd.isna(ema_20_series.iloc[-2])
    assert float(ema_20_series.iloc[-1]) == pytest.approx(1955.019773)


# to avoid running backtest multiple times
@pytest.fixture(scope="module")
def backtest_result(
    universe: TradingStrategyUniverse
) -> tuple[State, TradingStrategyUniverse, dict]:
    start_at, end_at = universe.universe.candles.get_timestamp_range()

    routing_model = generate_simple_routing_model(universe)

    # Run the test
    state, universe, debug_dump = run_backtest_inline(
        start_at=start_at.to_pydatetime(),
        end_at=end_at.to_pydatetime(),
        client=None,  # None of downloads needed, because we are using synthetic data
        cycle_duration=CycleDuration.cycle_1d,  # Override to use 24h cycles despite what strategy file says
        decide_trades=decide_trades,
        create_trading_universe=None,
        universe=universe,
        initial_deposit=10_000,
        reserve_currency=ReserveCurrency.busd,
        trade_routing=TradeRouting.user_supplied_routing_model,
        routing_model=routing_model,
        log_level=logging.WARNING,
    )

    return state, universe, debug_dump


def test_run_inline_synthetic_backtest(
        logger: logging.Logger,
        backtest_result: tuple[State, TradingStrategyUniverse, dict],
    ):
    """Run the strategy backtest using inline decide_trades function.
    """

    state, universe, debug_dump = backtest_result

    assert len(debug_dump) == 213


@pytest.fixture(scope = "module")
def analysis(
    backtest_result: tuple[State, TradingStrategyUniverse, dict]
) -> TradeAnalysis:
    state, universe, debug_dump = backtest_result
    analysis = build_trade_analysis(state.portfolio)

    return analysis


@pytest.fixture(scope = "module")
def summary(
    backtest_result: tuple[State, TradingStrategyUniverse, dict],
    analysis: TradeAnalysis
) -> PositionSummary:

    state, universe, debug_dump = backtest_result

    summary = analysis.calculate_summary_statistics(state = state)

    # Should not cause exception
    summary.to_dataframe()

    return summary


def test_basic_summary_statistics(
    summary: PositionSummary,
):
    """Analyse synthetic trading strategy adv_stats.

    TODO: Might move this test to its own module.
    # TODO summary stat test with stop losses involved
    """

    assert summary.initial_cash == 10_000
    assert summary.won_positions == 4
    assert summary.lost_positions == 7
    assert summary.realised_profit == pytest.approx(-47.17044385644749, rel=APPROX_REL)
    assert summary.open_value == pytest.approx(0, rel=APPROX_REL)
    assert summary.end_value == pytest.approx(9952.829556143553, rel=APPROX_REL)
    assert summary.won_position_percent == pytest.approx(0.36363636363636365, rel=APPROX_REL)
    assert summary.duration == datetime.timedelta(days=181)
    assert summary.trade_volume == pytest.approx(21900.29776619458, rel=APPROX_REL)
    assert summary.uninvested_cash == pytest.approx(9952.829556143553, rel=APPROX_REL)

    assert summary.stop_losses == 0
    assert summary.take_profits == 0
    assert summary.total_positions == 11
    assert summary.undecided_positions == 0
    assert summary.zero_loss_positions == 0

    assert summary.annualised_return_percent == pytest.approx(-0.0095122718274248, rel=APPROX_REL)
    assert summary.realised_profit == pytest.approx(-47.17044385644749, rel=APPROX_REL)
    assert summary.return_percent == pytest.approx(-0.004717044385644658, rel=APPROX_REL)

    assert summary.lp_fees_average_pc == pytest.approx(0.003004503819031923, rel=APPROX_REL)
    assert summary.lp_fees_paid == pytest.approx(65.79952827646791, rel=APPROX_REL)

    assert summary.average_duration_of_lost_positions == pd.Timedelta('8 days 13:42:51.428571428')
    assert summary.average_duration_of_won_positions == pd.Timedelta('19 days 00:00:00')

    assert summary.median_position == pytest.approx(-0.02569303244842014, rel=APPROX_REL)
    assert summary.average_lost_position_loss_pc == pytest.approx(-0.05157416459057936, rel=APPROX_REL)
    assert summary.average_net_profit == pytest.approx(-4.288222168767954, rel=APPROX_REL)
    assert summary.average_position == pytest.approx(-0.00398060248726169, rel=APPROX_REL)
    assert summary.average_won_position_profit_pc == pytest.approx(0.07930813119354424, rel=APPROX_REL)
    assert summary.avg_realised_risk == pytest.approx(-0.005157416459057936, rel=APPROX_REL)
    assert summary.biggest_lost_position_pc == pytest.approx(-0.14216816784355246, rel=APPROX_REL)
    assert summary.biggest_won_position_pc == pytest.approx(0.1518660490865238, rel=APPROX_REL)

    assert summary.max_loss_risk == pytest.approx(0.10000000000000002, rel=APPROX_REL)
    assert summary.max_neg_cons == 3
    assert summary.max_pos_cons == 1
    assert summary.max_pullback == pytest.approx(-0.01703492069936046, rel=APPROX_REL)



def test_advanced_summary_statistics(
    summary: PositionSummary
):
    full_stats = summary.get_full_stats()

    adv_stats = full_stats.to_dict()["Strategy"]

    assert adv_stats['Start Period'] == '2021-06-01'
    assert adv_stats['End Period'] == '2021-12-30'
    assert adv_stats['Risk-Free Rate'] == pytest.approx(0, rel=APPROX_REL)
    assert adv_stats['Time in Market'] == pytest.approx(0.7, rel=APPROX_REL)
    assert adv_stats['Cumulative Return'] == pytest.approx(0, rel=APPROX_REL)
    assert adv_stats['CAGR﹪'] == pytest.approx(-0.01, rel=APPROX_REL)
    assert adv_stats['Sharpe'] == pytest.approx(-0.14, rel=APPROX_REL)
    assert adv_stats['Prob. Sharpe Ratio'] == pytest.approx(0.45, rel=APPROX_REL)
    assert adv_stats['Smart Sharpe'] == pytest.approx(-0.13, rel=APPROX_REL)
    assert adv_stats['Sortino'] == pytest.approx(-0.2, rel=APPROX_REL)
    assert adv_stats['Smart Sortino'] == pytest.approx(-0.19, rel=APPROX_REL)
    assert adv_stats['Sortino/√2'] == pytest.approx(-0.14, rel=APPROX_REL)
    assert adv_stats['Smart Sortino/√2'] == pytest.approx(-0.13, rel=APPROX_REL)
    assert adv_stats['Omega'] == pytest.approx(0.98, rel=APPROX_REL)
    assert adv_stats['Max Drawdown'] == pytest.approx(-0.03, rel=APPROX_REL)
    assert adv_stats['Longest DD Days'] == 76
    assert adv_stats['Volatility (ann.)'] == pytest.approx(0.04, rel=APPROX_REL)
    assert adv_stats['Calmar'] == pytest.approx(-0.32, rel=APPROX_REL)
    assert adv_stats['Skew'] == pytest.approx(0.22, rel=APPROX_REL)
    assert adv_stats['Kurtosis'] == pytest.approx(0.08, rel=APPROX_REL)
    assert adv_stats['Expected Daily'] == pytest.approx(0, rel=APPROX_REL)
    assert adv_stats['Expected Monthly'] == pytest.approx(0, rel=APPROX_REL)
    assert adv_stats['Expected Yearly'] == pytest.approx(0, rel=APPROX_REL)
    assert adv_stats['Kelly Criterion'] == pytest.approx(-0.01, rel=APPROX_REL)
    assert adv_stats['Risk of Ruin'] == pytest.approx(0, rel=APPROX_REL)
    assert adv_stats['Daily Value-at-Risk'] == pytest.approx(0, rel=APPROX_REL)
    assert adv_stats['Expected Shortfall (cVaR)'] == pytest.approx(0, rel=APPROX_REL)
    assert adv_stats['Max Consecutive Wins'] == 5.0
    assert adv_stats['Max Consecutive Losses'] == 7.0
    assert adv_stats['Gain/Pain Ratio'] == pytest.approx(-0.02, rel=APPROX_REL)



def test_timeline(
    analysis: TradeAnalysis,
    backtest_result: tuple[State, TradingStrategyUniverse, dict]
):
    state, universe, debug_dump = backtest_result

    timeline = analysis.create_timeline()

    # Test expand timeline both colouring modes
    df, apply_styles = expand_timeline(
        universe.universe.exchanges,
        universe.universe.pairs,
        timeline,
        row_styling_mode=TimelineRowStylingMode.simple,
    )

    # Check HTML output does not crash
    # https://github.com/pandas-dev/pandas/issues/19358#issuecomment-359733504
    apply_styles(df).to_html()

    expanded_timeline, apply_styles = expand_timeline(
        universe.universe.exchanges,
        universe.universe.pairs,
        timeline,
        row_styling_mode=TimelineRowStylingMode.simple,
    )
    apply_styles(df).to_html()

    # Do checks for the first position
    # 0    1          2021-07-01   8 days                      WETH        USDC         $2,027.23    $27.23    2.72%   0.027230  $1,617.294181   $1,661.333561            2
    row = expanded_timeline.iloc[0]
    assert row["Opened at"] == "2021-07-02"
    assert row["Trade count"] == 2

    # 1    3          2021-07-10  26 days                      WETH        USDC         $1,002.72  $-137.39  -13.70%  -0.137013  $1,710.929622   $1,476.509241            2
    row2 = expanded_timeline.iloc[1]
    assert row2["Opened at"] == "2021-07-11"


def test_benchmark_synthetic_trading_portfolio(
    logger: logging.Logger,
    universe: TradingStrategyUniverse,
):
    """Build benchmark figures.

    TODO: Might move this test to its own module.
    """

    start_at, end_at = universe.universe.candles.get_timestamp_range()

    routing_model = generate_simple_routing_model(universe)

    # Run the test
    state, universe, debug_dump = run_backtest_inline(
        start_at=start_at.to_pydatetime(),
        end_at=end_at.to_pydatetime(),
        client=None,  # None of downloads needed, because we are using synthetic data
        cycle_duration=CycleDuration.cycle_1d,  # Override to use 24h cycles despite what strategy file says
        decide_trades=decide_trades,
        create_trading_universe=None,
        universe=universe,
        initial_deposit=10_000,
        reserve_currency=ReserveCurrency.busd,
        trade_routing=TradeRouting.user_supplied_routing_model,
        routing_model=routing_model,
        log_level=logging.WARNING,
    )

    # Visualise performance
    fig = visualise_benchmark(
        state.name,
        portfolio_statistics=state.stats.portfolio,
        all_cash=100_000,
        buy_and_hold_asset_name="ETH",
        buy_and_hold_price_series=universe.universe.candles.get_single_pair_data()["close"],
    )

    # Check that the diagram has 3 plots
    assert len(fig.data) == 3
