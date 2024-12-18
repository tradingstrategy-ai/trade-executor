"""Test backtesting where decide_trades and create_universe functions are passed directly.

"""
import logging
import random
import datetime
from typing import List, Dict

import pytest

import pandas as pd
from packaging import version
from pandas_ta.overlap import ema

from tradeexecutor.analysis.multi_asset_benchmark import compare_strategy_backtest_to_multiple_assets, get_benchmark_data
from tradeexecutor.analysis.trade_analyser import build_trade_analysis, expand_timeline, expand_timeline_raw, TimelineRowStylingMode, TradeAnalysis, TradeSummary
from tradeexecutor.backtest.backtest_runner import run_backtest_inline
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.visualisation import PlotKind
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, \
    create_pair_universe_from_code, translate_trading_pair
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange, generate_simple_routing_model
from tradeexecutor.testing.synthetic_price_data import generate_ohlcv_candles
from tradeexecutor.visual.benchmark import visualise_equity_curve_benchmark, visualise_long_short_benchmark, create_benchmark_equity_curves, \
    visualise_vs_returns, visualise_equity_curves
from tradingstrategy.candle import GroupedCandleUniverse
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.state.state import State
from tradingstrategy.universe import Universe
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.strategy_type import StrategyType
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.visual.equity_curve import calculate_equity_curve, calculate_returns, calculate_compounding_realised_trading_profitability, \
    calculate_long_compounding_realised_trading_profitability, calculate_rolling_sharpe, resample_returns
from tradeexecutor.analysis.advanced_metrics import visualise_advanced_metrics, AdvancedMetricsMode



# relative tolerance for floating point tests
APPROX_REL = 1e-2


# How much of the cash to put on a single trade
position_size = 0.10

# candle time bucket
time_bucket = TimeBucket.d1


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
    cash = state.portfolio.get_cash()

    # Get OHLCV candles for our trading pair as Pandas Dataframe.
    # We could have candles for multiple trading pairs in a different strategy,
    # but this strategy only operates on single pair candle.
    # We also limit our sample size to N latest candles to speed up calculations.
    candles: pd.DataFrame = universe.candles.get_single_pair_data(timestamp, sample_count=batch_size, raise_on_not_enough_data=False)

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
            trades += position_manager.open_spot(pair, buy_amount)
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

    strategy_universe = TradingStrategyUniverse(data_universe=universe, reserve_assets=[usdc])
    strategy_universe.data_universe.pairs.exchange_universe = strategy_universe.data_universe.exchange_universe
    return strategy_universe


@pytest.fixture(scope="module")
def strategy_universe(universe) -> TradingStrategyUniverse:
    return universe


def test_ema_on_universe(strategy_universe):
    """Calculate exponential moving average on single pair candle universe."""
    start_timestamp = pd.Timestamp("2021-6-1")
    batch_size = 20

    candles = strategy_universe.data_universe.candles.get_single_pair_data(
        start_timestamp,
        sample_count=batch_size,
        allow_current=True,
        raise_on_not_enough_data=False,
    )
    assert len(candles) == 1

    # Not enough data to calculate EMA - we haave only 1 sample
    ema_20_series = ema(candles["close"], length=20)
    assert ema_20_series is None

    end_timestamp = pd.Timestamp("2021-12-31")
    candles = strategy_universe.data_universe.candles.get_single_pair_data(end_timestamp, sample_count=batch_size, allow_current=True)
    assert len(candles) == batch_size

    ema_20_series = ema(candles["close"], length=20)
    assert pd.isna(ema_20_series.iloc[-2])
    assert float(ema_20_series.iloc[-1]) == pytest.approx(1955.019773)


# to avoid running backtest multiple times
@pytest.fixture(scope="module")
def backtest_result(
    strategy_universe
) -> tuple[State, TradingStrategyUniverse, dict]:
    start_at, end_at = strategy_universe.data_universe.candles.get_timestamp_range()

    routing_model = generate_simple_routing_model(strategy_universe)

    # Run the test
    state, strategy_universe, debug_dump = run_backtest_inline(
        start_at=start_at.to_pydatetime(),
        end_at=end_at.to_pydatetime(),
        client=None,  # None of downloads needed, because we are using synthetic data
        cycle_duration=CycleDuration.cycle_1d,  # Override to use 24h cycles despite what strategy file says
        decide_trades=decide_trades,
        create_trading_universe=None,
        universe=strategy_universe,
        initial_deposit=10_000,
        reserve_currency=ReserveCurrency.busd,
        trade_routing=TradeRouting.user_supplied_routing_model,
        routing_model=routing_model,
        log_level=logging.WARNING,
    )

    return state, strategy_universe, debug_dump


def test_run_inline_synthetic_backtest(
        logger: logging.Logger,
        backtest_result: tuple[State, TradingStrategyUniverse, dict],
    ):
    """Run the strategy backtest using inline decide_trades function.
    """

    state, universe, debug_dump = backtest_result

    assert len(debug_dump) == 215


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
) -> TradeSummary:

    state, universe, debug_dump = backtest_result

    summary = analysis.calculate_summary_statistics(state = state, time_bucket=time_bucket)

    # Should not cause exception
    summary.to_dataframe()
    summary.display()

    return summary


def test_basic_summary_statistics(
    summary: TradeSummary,
):
    """Analyse synthetic trading strategy adv_stats.

    TODO: Might move this test to its own module.
    # TODO summary stat test with stop losses involved
    """

    assert summary.initial_cash == 10_000
    assert summary.won == 4
    assert summary.lost == 7
    assert summary.realised_profit == pytest.approx(-47.17044385644749, rel=APPROX_REL)
    assert summary.open_value == pytest.approx(0, rel=APPROX_REL)
    assert summary.end_value == pytest.approx(9952.829556143553, rel=APPROX_REL)
    assert summary.win_percent == pytest.approx(0.36363636363636365, rel=APPROX_REL)
    assert summary.lost_percent == pytest.approx(0.6363636363636364, rel=APPROX_REL)
    assert summary.duration == datetime.timedelta(days=213)
    assert summary.trade_volume == pytest.approx(21900.29776619458, rel=APPROX_REL)
    assert summary.uninvested_cash == pytest.approx(9952.829556143553, rel=APPROX_REL)

    assert summary.stop_losses == 0
    assert summary.take_profits == 0
    assert summary.total_positions == 11
    assert summary.undecided == 0
    assert summary.zero_loss == 0

    assert summary.annualised_return_percent == pytest.approx(-0.00808319812565206, rel=APPROX_REL)
    assert summary.realised_profit == pytest.approx(-47.17044385644749, rel=APPROX_REL)
    assert summary.return_percent == pytest.approx(-0.004717044385644658, rel=APPROX_REL)

    assert summary.lp_fees_average_pc == pytest.approx(0.003004503819031923, rel=APPROX_REL)
    assert summary.lp_fees_paid == pytest.approx(65.79952827646791, rel=APPROX_REL)

    assert summary.average_duration_of_losing_trades == pd.Timedelta('8 days 13:42:51.428571')
    assert summary.average_duration_of_winning_trades == pd.Timedelta('19 days 00:00:00')

    assert summary.median_trade == pytest.approx(-0.02569303244842014, rel=APPROX_REL)
    assert summary.average_losing_trade_loss_pc == pytest.approx(-0.05157416459057936, rel=APPROX_REL)
    assert summary.average_net_profit == pytest.approx(-4.288222168767954, rel=APPROX_REL)
    assert summary.average_trade == pytest.approx(-0.00398060248726169, rel=APPROX_REL)
    assert summary.average_winning_trade_profit_pc == pytest.approx(0.07930813119354424, rel=APPROX_REL)
    assert summary.avg_realised_risk == pytest.approx(-0.005157416459057936, rel=APPROX_REL)
    assert summary.biggest_losing_trade_pc == pytest.approx(-0.14216816784355246, rel=APPROX_REL)
    assert summary.biggest_winning_trade_pc == pytest.approx(0.1518660490865238, rel=APPROX_REL)

    assert summary.max_loss_risk == pytest.approx(0.10000000000000002, rel=APPROX_REL)
    assert summary.max_neg_cons == 3
    assert summary.max_pos_cons == 1
    assert summary.max_pullback == pytest.approx(-0.01703492069936046, rel=APPROX_REL)

    assert summary.winning_stop_losses == 0
    assert summary.winning_stop_losses_percent is None
    assert summary.losing_stop_losses == 0
    assert summary.losing_stop_losses_percent is None
    assert summary.winning_take_profits == 0
    assert summary.winning_take_profits_percent is None

    assert summary.sharpe_ratio == pytest.approx(-0.16402323856225548, rel=APPROX_REL)
    assert summary.sortino_ratio == pytest.approx(-0.23988078508533023, rel=APPROX_REL)
    assert summary.profit_factor == pytest.approx(0.9754583954173234, rel=APPROX_REL)
    assert summary.average_duration_between_position_openings == pd.Timedelta('17 days 09:36:00')
    assert summary.average_position_frequency == pytest.approx(0.051643192488262914)
    assert summary.average_interest_paid_usd == pytest.approx(0.0, rel=APPROX_REL)
    assert summary.max_interest_paid_usd == pytest.approx(0.0, rel=APPROX_REL)
    assert summary.min_interest_paid_usd == pytest.approx(0.0, rel=APPROX_REL)
    assert summary.total_interest_paid_usd == pytest.approx(0.0, rel=APPROX_REL)
    assert summary.median_interest_paid_usd == pytest.approx(0.0, rel=APPROX_REL)
    assert summary.time_in_market == pytest.approx(0.6384976525821596)
    assert summary.time_in_market_volatile == pytest.approx(0.6384976525821596)


def test_compounding_formulas(
        backtest_result: tuple[State, TradingStrategyUniverse, dict],
        summary: TradeSummary
):
    """Test compounding formulas for long and overall."""
    state, universe, debug_dump = backtest_result
    profitability = calculate_compounding_realised_trading_profitability(state)
    assert profitability.iloc[-1] == pytest.approx(-0.004717044385644686, rel=APPROX_REL)
    assert profitability.equals(calculate_long_compounding_realised_trading_profitability(state))
    assert profitability.equals(summary.compounding_returns)
    assert profitability.iloc[-1] == pytest.approx(summary.return_percent, abs=1e-9)


def test_bars_display(backtest_result: tuple[State, TradingStrategyUniverse, dict],
    analysis: TradeAnalysis):

    state, universe, debug_dump = backtest_result

    # should not cause exception
    summary2 = analysis.calculate_summary_statistics(time_bucket)
    summary3 = analysis.calculate_summary_statistics(time_bucket, state)
    summary2.to_dataframe()
    summary3.display()

    df = summary3.to_dataframe(format_headings=False)
    assert df.loc["Average duration of winning positions"][0] == '19 days 0 hours'
    assert df.loc["Average duration of losing positions"][0] == '8 days 13 hours'

    assert df.loc["Average bars of winning positions"][0] == '19 bars'
    assert df.loc["Average bars of losing positions"][0] == '8 bars'


def test_core_stats(analysis: TradeAnalysis):
    summary = analysis.calculate_summary_statistics(time_bucket)
    core_stats = summary.get_trading_core_metrics()
    assert core_stats["Positions taken"] == "11"
    assert core_stats["Positions won"] == "36.36%"


def test_advanced_summary_statistics(
    backtest_result: tuple[State, TradingStrategyUniverse, dict],
    summary: TradeSummary
):
    

    state, universe, debug_dump = backtest_result

    equity = calculate_equity_curve(state)
    returns = calculate_returns(equity)
    metrics = visualise_advanced_metrics(returns, mode=AdvancedMetricsMode.full)
    adv_stats = metrics.to_dict()["Strategy"]

    assert adv_stats['Start Period'] == '2021-06-01'
    assert adv_stats['End Period'] == '2021-12-30'
    assert adv_stats['Risk-Free Rate'] == '0.0%'
    assert adv_stats['Time in Market'] == '70.0%'
    assert adv_stats['Cumulative Return'] == '-0.47%'

    # TODO: No idea why this is happening.
    # Leave for later to to fix the underlying libraries.
    if version.parse(pd.__version__) >= version.parse("2.0"):
        assert adv_stats['CAGR﹪'] == '-0.56%'
        assert adv_stats['Calmar'] == '-0.22'
        assert adv_stats['3Y (ann.)'] == '-0.56%'
        assert adv_stats['5Y (ann.)'] == '-0.56%'
        assert adv_stats['10Y (ann.)'] == '-0.56%'
        assert adv_stats['All-time (ann.)'] == '-0.56%'
        assert adv_stats['Recovery Factor'] == '0.17'
    else:
        assert adv_stats['CAGR﹪'] == '-0.46%'
        assert adv_stats['Calmar'] == '-0.32'
        assert adv_stats['3Y (ann.)'] == '-0.81%'
        assert adv_stats['5Y (ann.)'] == '-0.81%'
        assert adv_stats['10Y (ann.)'] == '-0.81%'
        assert adv_stats['All-time (ann.)'] == '-0.81%'
        assert adv_stats['Recovery Factor'] == '-0.19'

    assert adv_stats['Sharpe'] == '-0.16'
    assert adv_stats['Prob. Sharpe Ratio'] == '45.02%'
    assert adv_stats['Smart Sharpe'] == '-0.16'
    assert adv_stats['Sortino'] == '-0.24'
    assert adv_stats['Smart Sortino'] == '-0.23'
    assert adv_stats['Sortino/√2'] == '-0.17'
    assert adv_stats['Smart Sortino/√2'] == '-0.16'
    assert adv_stats['Omega'] == '0.98'
    assert adv_stats['Max Drawdown'] == '-2.52%'
    assert adv_stats['Longest DD Days'] == '76'
    assert adv_stats['Volatility (ann.)'] == '4.35%'
    assert adv_stats['Skew'] == '0.22'
    assert adv_stats['Kurtosis'] == '0.08'
    assert adv_stats['Expected Daily'] == '-0.0%'
    assert adv_stats['Expected Monthly'] == '-0.07%'
    assert adv_stats['Expected Yearly'] == '-0.47%'
    assert adv_stats['Kelly Criterion'] == '-1.13%'
    assert adv_stats['Risk of Ruin'] == '0.0%'
    assert adv_stats['Daily Value-at-Risk'] == '-0.38%'
    assert adv_stats['Expected Shortfall (cVaR)'] == '-0.38%'
    assert adv_stats['Max Consecutive Wins'] == '5'
    assert adv_stats['Max Consecutive Losses'] == '7'
    assert adv_stats['Gain/Pain Ratio'] == '-0.02'
    assert adv_stats['Gain/Pain (1M)'] == '-0.12'
    assert adv_stats['Payoff Ratio'] == '1.2'
    assert adv_stats['Profit Factor'] == '0.98'
    assert adv_stats['Common Sense Ratio'] == '1.05'
    assert adv_stats['CPC Index'] == '0.52'
    assert adv_stats['Tail Ratio'] == '1.08'
    assert adv_stats['Outlier Win Ratio'] == '4.23'
    assert adv_stats['Outlier Loss Ratio'] == '2.23'
    assert adv_stats['MTD'] == '-1.23%'
    assert adv_stats['3M'] == '-0.59%'
    assert adv_stats['6M'] == '-0.47%'
    assert adv_stats['YTD'] == '-0.47%'
    assert adv_stats['1Y'] == '-0.47%'
    assert adv_stats['Best Day'] == '0.55%'
    assert adv_stats['Worst Day'] == '-0.57%'
    assert adv_stats['Best Month'] == '1.11%'
    assert adv_stats['Worst Month'] == '-1.7%'
    assert adv_stats['Best Year'] == '-0.47%'
    assert adv_stats['Worst Year'] == '-0.47%'
    assert adv_stats['Avg. Drawdown'] == '-1.11%'
    assert adv_stats['Avg. Drawdown Days'] == '25'
    assert adv_stats['Ulcer Index'] == '0.01'
    assert adv_stats['Serenity Index'] == '-0.04'
    assert adv_stats['Avg. Up Month'] == '1.06%'
    assert adv_stats['Avg. Down Month'] == '-1.2%'
    assert adv_stats['Win Days'] == '44.9%'
    assert adv_stats['Win Month'] == '50.0%'
    assert adv_stats['Win Quarter'] == '50.0%'
    assert adv_stats['Win Year'] == '0.0%'




def test_timeline(
    analysis: TradeAnalysis,
    backtest_result: tuple[State, TradingStrategyUniverse, dict],
):
    state, universe, debug_dump = backtest_result

    timeline = analysis.create_timeline()

    # Test expand timeline both colouring modes
    expanded_timeline, apply_styles = expand_timeline(
        universe.data_universe.exchanges,
        universe.data_universe.pairs,
        timeline,
        row_styling_mode=TimelineRowStylingMode.simple,
    )

    # Check HTML output does not crash
    # https://github.com/pandas-dev/pandas/issues/19358#issuecomment-359733504
    apply_styles(expanded_timeline).to_html()

    # Do checks for the first position
    # 0    1          2021-07-01   8 days                      WETH        USDC         $2,027.23    $27.23    2.72%   0.027230  $1,617.294181   $1,661.333561            2
    row = expanded_timeline.iloc[0]
    assert row['Id'] == 1
    assert row['Remarks'] == ''
    assert row['Opened at'] == '2021-07-02'
    assert row['Duration'] == '8 days      '
    assert row['Exchange'] == ''
    assert row['Base asset'] == 'WETH'
    assert row['Quote asset'] == 'USDC'
    assert row['Position max value'] == '$1,000.00'
    assert row['PnL USD'] == '$21.09'
    assert row['PnL %'] == '2.11%'
    assert row['PnL % raw'] == pytest.approx(0.021094526830844895, 1e-6)
    assert row['Open mid price USD'] == '$1,617.279626'
    assert row['Close mid price USD'] == '$1,651.395374'
    assert row['Trade count'] == 2
    assert row['LP fees'] == '$6.07'

    # 1    3          2021-07-10  26 days                      WETH        USDC         $1,002.72  $-137.39  -13.70%  -0.137013  $1,710.929622   $1,476.509241            2
    row2 = expanded_timeline.iloc[1]
    assert row2['Id'] == 2
    assert row2['Remarks'] == ''
    assert row2['Opened at'] == '2021-07-11'
    assert row2['Duration'] == '26 days      '
    assert row2['Exchange'] == ''
    assert row2['Base asset'] == 'WETH'
    assert row2['Quote asset'] == 'USDC'
    assert row2['Position max value'] == '$1,002.11'
    assert row2['PnL USD'] == '$-142.47'
    assert row2['PnL %'] == '-14.22%'
    assert row2['PnL % raw'] == pytest.approx(-0.14216816784355246, 1e-6)
    assert row2['Open mid price USD'] == '$1,710.914224'
    assert row2['Close mid price USD'] == '$1,467.676683'
    assert row2['Trade count'] == 2
    assert row2['LP fees'] == '$5.59'
    
    last_row = expanded_timeline.iloc[-1]
    assert last_row['Id'] == 11
    assert last_row['Remarks'] == ''
    assert last_row['Opened at'] == '2021-12-23'
    assert last_row['Duration'] == '7 days      '
    assert last_row['Exchange'] == ''
    assert last_row['Base asset'] == 'WETH'
    assert last_row['Quote asset'] == 'USDC'
    assert last_row['Position max value'] == '$1,000.32'
    assert last_row['PnL USD'] == '$-50.32'
    assert last_row['PnL %'] == '-5.03%'
    assert last_row['PnL % raw'] == pytest.approx(-0.05030528788437061, 1e-6)
    assert last_row['Open mid price USD'] == '$2,004.138663'
    assert last_row['Close mid price USD'] == '$1,903.319891'
    assert last_row['Trade count'] == 2
    assert last_row['LP fees'] == '$5.86'
    

def test_timeline_raw(
    analysis: TradeAnalysis,
    backtest_result: tuple[State, TradingStrategyUniverse, dict],
):
    state, universe, debug_dump = backtest_result

    timeline = analysis.create_timeline()

    expanded_timeline_raw = expand_timeline_raw(
        timeline,
    )

    # Do checks for the first position
    row = expanded_timeline_raw.iloc[0]
    assert row['Id'] == 1
    assert row['Remarks'] == ''
    assert row['Opened at'] == '2021-07-02'
    assert row['Duration'] == '8 days      '
    assert row['position_max_size'] == pytest.approx(1000.0000000000001, rel=1e-4)
    assert row['pnl_usd'] == pytest.approx(21.094527, rel=1e-4)
    assert row['pnl_pct_raw'] == pytest.approx(0.021095, rel=1e-4)
    assert row['open_price_usd'] == pytest.approx(1617.279626, rel=1e-4)
    assert row['close_price_usd'] == pytest.approx(1651.395374, rel=1e-4)
    assert row['trade_count'] == 2

    row2 = expanded_timeline_raw.iloc[1]
    assert row2['Id'] == 2
    assert row2['Remarks'] == ''
    assert row2['Opened at'] == '2021-07-11'
    assert row2['Duration'] == '26 days      '
    assert row2['position_max_size'] == pytest.approx(1002.109453, rel=1e-4)
    assert row2['pnl_usd'] == pytest.approx(-142.468065, rel=1e-4)
    assert row2['pnl_pct_raw'] == pytest.approx(-0.142168, rel=1e-4)
    assert row2['open_price_usd'] == pytest.approx(1710.914224, rel=1e-4)
    assert row2['close_price_usd'] == pytest.approx(1467.676683, rel=1e-4)
    assert row2['trade_count'] == 2
    
    last_row = expanded_timeline_raw.iloc[-1]
    assert last_row['Id'] == 11
    assert last_row['Remarks'] == ''
    assert last_row['Opened at'] == '2021-12-23'
    assert last_row['Duration'] == '7 days      '
    assert last_row['position_max_size'] == pytest.approx(1000.315069, rel=1e-4)
    assert last_row['pnl_usd'] == pytest.approx(-50.321138, rel=1e-4)
    assert last_row['pnl_pct_raw'] == pytest.approx(-0.050305, rel=1e-4)
    assert last_row['open_price_usd'] == pytest.approx(2004.138663, rel=1e-4)
    assert last_row['close_price_usd'] == pytest.approx(1903.319891, rel=1e-4)
    assert last_row['trade_count'] == 2


def test_benchmark_synthetic_trading_portfolio(
    logger: logging.Logger,
    strategy_universe,
):
    """Build benchmark figures.

    TODO: Might move this test to its own module.
    """

    start_at, end_at = strategy_universe.data_universe.candles.get_timestamp_range()

    routing_model = generate_simple_routing_model(strategy_universe)

    # Run the test
    state, strategy_universe, debug_dump = run_backtest_inline(
        start_at=start_at.to_pydatetime(),
        end_at=end_at.to_pydatetime(),
        client=None,  # None of downloads needed, because we are using synthetic data
        cycle_duration=CycleDuration.cycle_1d,  # Override to use 24h cycles despite what strategy file says
        decide_trades=decide_trades,
        create_trading_universe=None,
        universe=strategy_universe,
        initial_deposit=10_000,
        reserve_currency=ReserveCurrency.busd,
        trade_routing=TradeRouting.user_supplied_routing_model,
        routing_model=routing_model,
        log_level=logging.WARNING,
    )

    # Visualise performance
    fig = visualise_equity_curve_benchmark(
        state.name,
        portfolio_statistics=state.stats.portfolio,
        all_cash=100_000,
        buy_and_hold_asset_name="ETH",
        buy_and_hold_price_series=strategy_universe.data_universe.candles.get_single_pair_data()["close"],
    )

    # Check that the diagram has 3 plots
    assert len(fig.data) == 3

    # Visualise long/short benchmark
    fig2 = visualise_long_short_benchmark(
        state,
        time_bucket,
    )

    assert len(fig2.data) == 3

    # Test visualise_vs_returns()
    #
    equity_curve = calculate_equity_curve(state)
    returns = calculate_returns(equity_curve)
    our_pair = translate_trading_pair(strategy_universe.data_universe.pairs.get_single())
    benchmark_indexes = create_benchmark_equity_curves(
        strategy_universe,
        {"ETH": our_pair},
        initial_cash=state.portfolio.get_initial_cash(),
    )
    benchmark_indexes["ETH"].attrs = {"colour": "blue", "name": "Buy and hold ETH"}
    fig3 = visualise_vs_returns(
        returns,
        benchmark_indexes,
    )
    assert len(fig3.data) == 2

    rolling_sharpe = calculate_rolling_sharpe(returns)
    assert len(rolling_sharpe) > 0


def test_compare_portfolios(
    backtest_result: tuple[State, TradingStrategyUniverse, dict],
    summary: TradeSummary
):
    """Compare multiple portfolios or return series."""

    state, universe, debug_dump = backtest_result
    df = compare_strategy_backtest_to_multiple_assets(
        state,
        universe,
    )
    assert len(df.columns) == 2
    assert df.columns[0] == "Strategy"
    assert df.columns[1] == "ETH"


def test_compare_portfolios(
    backtest_result: tuple[State, TradingStrategyUniverse, dict],
    summary: TradeSummary
):
    """Compare multiple portfolios or return series."""

    state, universe, debug_dump = backtest_result
    df = compare_strategy_backtest_to_multiple_assets(
        state,
        universe,
    )
    assert len(df.columns) == 2
    assert df.columns[0] == "Strategy"
    assert df.columns[1] == "ETH"


def test_visualise_comparison(
    backtest_result: tuple[State, TradingStrategyUniverse, dict],
    summary: TradeSummary
):
    """Compare multiple portfolios or return series."""

    state, strategy_universe, debug_dump = backtest_result

    # Get daily returns
    equity = calculate_equity_curve(state)

    benchmarks = get_benchmark_data(
        strategy_universe,
        cumulative_with_initial_cash=state.portfolio.get_initial_cash(),
    )

    fig = visualise_equity_curves(
        [equity, benchmarks["ETH"]],
    )

    assert len(fig.data) == 2

