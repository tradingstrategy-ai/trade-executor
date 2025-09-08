"""Grid search tests."""
import datetime
from pathlib import Path
from typing import List

import pandas as pd
import pandas_ta_classic as pandas_ta
import pytest
from packaging import version
from pandas.io.formats.style import Styler
from plotly.graph_objs import Figure

from tradeexecutor.backtest.backtest_execution import BacktestExecutionFailed
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet, DiskIndicatorStorage, IndicatorSource, IndicatorDependencyResolver
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.visual.grid_search import visualise_single_grid_search_result_benchmark, visualise_grid_search_equity_curves
from tradeexecutor.visual.grid_search_advanced import calculate_rolling_metrics, BenchmarkMetric, visualise_grid_single_rolling_metric, visualise_grid_rolling_metric_heatmap

from tradeexecutor.visual.grid_search_advanced import visualise_grid_rolling_metric_line_chart
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import Exchange
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

from tradeexecutor.analysis.grid_search import analyse_grid_search_result, render_grid_search_result_table, visualise_heatmap_2d, \
    find_best_grid_search_results
from tradeexecutor.backtest.grid_search import prepare_grid_combinations, run_grid_search_backtest, perform_grid_search, GridCombination, GridSearchResult, \
    pick_grid_search_result, pick_best_grid_search_result, GridParameter, create_grid_search_failed_result
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.visualisation import PlotKind
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, create_pair_universe_from_code
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange
from tradeexecutor.testing.synthetic_price_data import generate_ohlcv_candles


def runner(universe: TradingStrategyUniverse, **kwrags) -> State:
    return State()


@pytest.fixture
def mock_chain_id() -> ChainId:
    """Mock a chai id."""
    return ChainId.ethereum


@pytest.fixture
def mock_exchange(mock_chain_id) -> Exchange:
    """Mock an exchange."""
    return generate_exchange(exchange_id=1, chain_id=mock_chain_id, address=generate_random_ethereum_address())


@pytest.fixture
def usdc() -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "USDC", 6, 1)


@pytest.fixture
def weth() -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "WETH", 18, 2)


@pytest.fixture
def weth_usdc(mock_exchange, usdc, weth) -> TradingPairIdentifier:
    """Mock some assets"""
    return TradingPairIdentifier(
        weth,
        usdc,
        generate_random_ethereum_address(),
        mock_exchange.address,
        internal_id=555,
        internal_exchange_id=mock_exchange.exchange_id,
        fee=0.0030,
    )


@pytest.fixture
def result_path() -> Path:
    """Where the grid search data is stored"""
    return


@pytest.fixture()
def universe(mock_chain_id, mock_exchange, weth_usdc) -> TradingStrategyUniverse:
    """Generate synthetic trading data universe for a single trading pair.

    - Single mock exchange

    - Single mock trading pair

    - Random candles

    - No liquidity data available
    """

    start_date = datetime.datetime(2021, 6, 1)
    end_date = datetime.datetime(2022, 1, 1)

    time_bucket = TimeBucket.d1

    pair_universe = create_pair_universe_from_code(mock_chain_id, [weth_usdc])

    # Generate candles for pair_id = 1
    candles = generate_ohlcv_candles(time_bucket, start_date, end_date, pair_id=weth_usdc.internal_id)
    candle_universe = GroupedCandleUniverse.create_from_single_pair_dataframe(candles, time_bucket)

    universe = Universe(
        time_bucket=time_bucket,
        chains={mock_chain_id},
        exchanges={mock_exchange},
        pairs=pair_universe,
        candles=candle_universe,
        liquidity=None
    )

    return TradingStrategyUniverse(
        data_universe=universe,
        backtest_stop_loss_candles=candle_universe,
        backtest_stop_loss_time_bucket=time_bucket,
        reserve_assets=[weth_usdc.quote])


@pytest.fixture()
def strategy_universe(universe) -> TradingStrategyUniverse:
    return universe


@pytest.fixture
def indicator_storage(strategy_universe, tmp_path) -> DiskIndicatorStorage:
    """Mock some assets"""
    return DiskIndicatorStorage(Path(tmp_path), strategy_universe.get_cache_key())


def grid_search_worker(
        universe: TradingStrategyUniverse,
        combination: GridCombination,
) -> GridSearchResult:
    """Run a backtest for a single grid combination."""

    stop_loss_pct, slow_ema_candle_count, fast_ema_candle_count = combination.destructure()
    batch_size = fast_ema_candle_count + 1
    position_size = 0.50

    def decide_trades(
        timestamp: pd.Timestamp,
        universe: Universe,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: dict
    ):
        # The pair we are trading
        pair = universe.pairs.get_single()
        cash = state.portfolio.get_cash()
        candles: pd.DataFrame = universe.candles.get_single_pair_data(sample_count=batch_size)
        close = candles["close"]
        slow_ema = close.ewm(span=slow_ema_candle_count).mean().iloc[-1]
        fast_ema = close.ewm(span=fast_ema_candle_count).mean().iloc[-1]
        trades = []
        position_manager = PositionManager(timestamp, universe, state, pricing_model)
        current_price = close.iloc[-1]
        if current_price >= slow_ema:
            if not position_manager.is_any_open():
                buy_amount = cash * position_size
                trades += position_manager.open_spot(pair, buy_amount, stop_loss_pct=stop_loss_pct)
        elif fast_ema >= slow_ema:
            if position_manager.is_any_open():
                trades += position_manager.close_all()
        visualisation = state.visualisation
        visualisation.plot_indicator(timestamp, "Slow EMA", PlotKind.technical_indicator_on_price, slow_ema, colour="forestgreen")
        visualisation.plot_indicator(timestamp, "Fast EMA", PlotKind.technical_indicator_on_price, fast_ema, colour="limegreen")
        return trades

    return run_grid_search_backtest(
        combination,
        decide_trades,
        universe,
    )


def test_prepare_grid_search_parameters(tmp_path):
    """Prepare grid search parameters."""

    parameters = {
        "stop_loss": [0.9, 0.95],
        "max_asset_amount": [3, 4],
        "momentum_lookback_days": ["7d", "14d", "21d"]
    }

    combinations = prepare_grid_combinations(parameters, tmp_path)
    assert len(combinations) == 2 * 2 * 3

    first = combinations[0]
    assert first.parameters[0].name == "stop_loss"
    assert first.parameters[0].value == 0.9

    assert first.get_relative_result_path() == Path('stop_loss=0.9/max_asset_amount=3/momentum_lookback_days=7d')
    assert first.get_label() == "#1, stop_loss=0.9, max_asset_amount=3, momentum_lookback_days=7d"


def test_perform_grid_search_single_thread(
        strategy_universe,
        tmp_path,
):
    """Run a grid search.

    Use the basic single thread mode for better debuggability.
    """

    parameters = {
        "stop_loss_pct": [0.9, 0.95],
        "slow_ema_candle_count": [7, 9],
        "fast_ema_candle_count": [2, 3],
    }

    combinations = prepare_grid_combinations(parameters, tmp_path)

    results = perform_grid_search(
        grid_search_worker,
        strategy_universe,
        combinations,
        max_workers=1,
    )

    # Pick a result of a single grid search combination
    # and examine its trading metrics
    sample = pick_grid_search_result(
        results,
        stop_loss_pct=0.9,
        slow_ema_candle_count=7,
        fast_ema_candle_count=2)
    assert sample.summary.total_positions == 2

    sample = pick_grid_search_result(
        results,
        stop_loss_pct=1.0,
        slow_ema_candle_count=7,
        fast_ema_candle_count=2)
    assert sample is None

    sample = pick_best_grid_search_result(
        results,
        key=lambda r: r.metrics.loc["Max Drawdown"][0])
    assert sample is not None

    sample = pick_best_grid_search_result(results)
    assert sample is not None

    table = analyse_grid_search_result(
        results,
        min_positions_threshold=0,
        drop_duplicates=False,
    )
    assert len(table) == 2 * 2 * 2
    row = table.iloc[0]
    # assert row["stop_loss_pct"] == 0.9
    # Getting on Github:
    # Obtained: 0.011546587485546267
    # Expected: 0.011682534679563261 ± 1.2e-08

    # TODO: No idea why this is happening.
    # Leave for later to to fix the underlying libraries.
    if version.parse(pd.__version__) >= version.parse("2.0"):
        assert row["CAGR"] == pytest.approx(0.046278164019362356)
    else:
        assert row["CAGR"] == pytest.approx(0.06771955893113946)
    assert row["Positions"] == 2

    styler = render_grid_search_result_table(table)
    assert isinstance(styler, Styler)

    # Remove extra axis by focusing only stop_loss_pct=0.9
    heatmap_data = table.xs(0.9, level="stop_loss_pct")
    fig = visualise_heatmap_2d(heatmap_data, "fast_ema_candle_count", "slow_ema_candle_count", "CAGR")
    assert isinstance(fig, Figure)


def test_perform_grid_search_cached(
        strategy_universe,
        tmp_path,
):
    """Run a grid search twice and see we get cached results."""

    parameters = {
        "stop_loss_pct": [0.9, 0.95],
        "slow_ema_candle_count": [7],
        "fast_ema_candle_count": [1],
    }

    combinations = prepare_grid_combinations(parameters, tmp_path)

    results = perform_grid_search(
        grid_search_worker,
        strategy_universe,
        combinations,
        max_workers=1,
    )

    for r in results:
        assert not r.cached

    results_2 = perform_grid_search(
        grid_search_worker,
        strategy_universe,
        combinations,
        max_workers=1,
    )

    for r in results_2:
        assert r.cached


def test_perform_grid_search_threaded(
        strategy_universe,
        tmp_path,
):
    """Run a grid search using multiple threads."""

    parameters = {
        "stop_loss_pct": [0.9, 0.95],
        "slow_ema_candle_count": [7],
        "fast_ema_candle_count": [1],
    }

    combinations = prepare_grid_combinations(parameters, tmp_path)

    results = perform_grid_search(
        grid_search_worker,
        strategy_universe,
        combinations,
        max_workers=4,
    )

    # Check we got results back
    for r in results:
        assert r.metrics.loc["Sharpe"][0] != 0


def test_perform_grid_search_multiprocess(
        strategy_universe,
        tmp_path,
):
    """Run a grid search using multiple threads."""

    parameters = {
        "stop_loss_pct": [0.9, 0.95],
        "slow_ema_candle_count": [7],
        "fast_ema_candle_count": [1, 2],
    }

    combinations = prepare_grid_combinations(parameters, tmp_path)

    results = perform_grid_search(
        grid_search_worker,
        strategy_universe,
        combinations,
        max_workers=4,
        multiprocess=True,
    )

    assert len(results) == 4

    # Check we got results back
    for r in results:
        assert r.metrics.loc["Sharpe"][0] != 0
        assert r.process_id > 1


def _decide_trades_v3(
    timestamp: pd.Timestamp,
    parameters: StrategyParameters,
    strategy_universe: TradingStrategyUniverse,
    state: State,
    pricing_model: PricingModel) -> List[TradeExecution]:
    """A simple strategy that puts all in to our lending reserve."""
    assert "stop_loss_pct" in parameters
    assert "slow_ema_candle_count" in parameters
    assert "fast_ema_candle_count" in parameters
    assert type(parameters.stop_loss_pct) == float
    assert type(parameters.slow_ema_candle_count) == int, f"Got {type(parameters.slow_ema_candle_count)}"
    assert type(parameters.fast_ema_candle_count) == int
    return []


def test_perform_grid_search_engine_v4(
    strategy_universe,
    tmp_path,
):
    """Run a grid search using multiple threads, engine version 0.4.

    - Uses a new style paramaeter passing
    """
    class InputParameters:
        stop_loss_pct = [0.9, 0.95]
        slow_ema_candle_count = 7
        fast_ema_candle_count = [1, 2]
        cycle_duration = CycleDuration.cycle_1d
        initial_cash = 10_000

    combinations = prepare_grid_combinations(InputParameters, tmp_path)

    # Sanity check for searchable parameters
    cycle_duration: GridParameter = combinations[0].parameters[0]
    assert cycle_duration.name == "cycle_duration"
    assert cycle_duration.single

    fast_ema_candle_count: GridParameter = combinations[0].parameters[1]
    assert fast_ema_candle_count.name == "fast_ema_candle_count"
    assert not fast_ema_candle_count.single

    assert len(combinations[0].searchable_parameters) == 2
    assert len(combinations[0].parameters) == 6

    results = perform_grid_search(
        _decide_trades_v3,
        strategy_universe,
        combinations,
        max_workers=4,
        multiprocess=True,
        trading_strategy_engine_version="0.4",
    )

    assert len(results) == 4

    # Check we got results back
    for r in results:
        assert r.metrics.loc["Sharpe"][0] != 0
        assert r.process_id > 1

    # See we can render the results
    fig = visualise_grid_search_equity_curves(
        results,
    )
    assert fig is not None


def test_perform_grid_search_engine_v4_cached(
    strategy_universe,
    tmp_path,
):
    """Read cached grid serach results.

    - Check that grid search result caching works
    """
    class InputParameters:
        stop_loss_pct = [0.9, 0.95]
        slow_ema_candle_count = 7
        fast_ema_candle_count = [1, 2]
        cycle_duration = CycleDuration.cycle_1d
        initial_cash = 10_000

    combinations = prepare_grid_combinations(InputParameters, tmp_path)

    results = perform_grid_search(
        _decide_trades_v3,
        strategy_universe,
        combinations,
        max_workers=4,
        multiprocess=True,
        trading_strategy_engine_version="0.4",
    )

    already_run_results = perform_grid_search(
        _decide_trades_v3,
        strategy_universe,
        combinations,
        max_workers=4,
        multiprocess=True,
        trading_strategy_engine_version="0.4",
    )

    assert len(results) == 4
    assert len(already_run_results) == 4

    for r in results:
        assert not r.cached

    for r in already_run_results:
        assert r.cached



def _decide_trades_v4(input: StrategyInput) -> List[TradeExecution]:
    """Checks some indicator logic works over grid search."""
    parameters = input.parameters
    assert "slow_ema_candle_count" in parameters
    assert "fast_ema_candle_count" in parameters

    if input.indicators.get_indicator_value("slow_ema") is not None:
        assert input.indicators.get_indicator_value("slow_ema") > 0

    series = input.indicators.get_indicator_series("slow_ema", unlimited=True)
    assert len(series) > 0

    series = input.indicators.get_indicator_series("my_custom_indicator", unlimited=True)
    assert len(series) ==  0

    return []


def _decide_trades_combined_indicator(input: StrategyInput) -> List[TradeExecution]:
    input.indicators.get_indicator_value("combined_indicator")
    return []


def _decide_trades_out_of_balance(input: StrategyInput) -> List[TradeExecution]:
    strategy_universe = input.strategy_universe
    pair = strategy_universe.get_single_pair()
    position_manager = input.get_position_manager()
    # Buy too much, we have only $10,000
    trades = position_manager.open_spot(
        pair,
        value=99_999,
    )
    return trades


def my_custom_indicator(strategy_universe: TradingStrategyUniverse):
    return pd.Series(dtype="float64")


def test_perform_grid_search_engine_v5(
    strategy_universe,
    indicator_storage,
    tmp_path,
):
    """Run a grid search using multiple threads, engine version 0.5.

    - Uses DecideTradesProtocolV5

    - Indicators are calculated prior to the grid search in a separate step

    - The actual grid search loads cached indicator values from the disk

    """
    class MyParameters:
        cycle_duration = CycleDuration.cycle_1d
        initial_cash = 10_000

        # Indicator values that are searched in the grid search
        slow_ema_candle_count = 7
        fast_ema_candle_count = [2, 3]

    def create_indicators(parameters: StrategyParameters, indicators: IndicatorSet, strategy_universe: TradingStrategyUniverse, execution_context: ExecutionContext):
        indicators.add("slow_ema", pandas_ta.ema, {"length": parameters.slow_ema_candle_count})
        indicators.add("fast_ema", pandas_ta.ema, {"length": parameters.fast_ema_candle_count})
        indicators.add("my_custom_indicator", my_custom_indicator, source=IndicatorSource.strategy_universe)

    combinations = prepare_grid_combinations(
        MyParameters,
        tmp_path,
        strategy_universe=strategy_universe,
        create_indicators=create_indicators,
        execution_context=ExecutionContext(mode=ExecutionMode.unit_testing, grid_search=True),
    )

    # fast ema 1, slow ema 7, my custom indicator
    c = combinations[0]
    assert len(c.indicators) == 3

    # fast ema 2, slow ema 7, my custom indicator
    c = combinations[1]
    assert len(c.indicators) == 3

    # {<IndicatorKey slow_ema(length=7)-WETH-USDC>, <IndicatorKey fast_ema(length=2)-WETH-USDC>, <IndicatorKey fast_ema(length=1)-WETH-USDC>, <IndicatorKey my_custom_indicator()-universe>}
    all_indicators = GridCombination.get_all_indicators(combinations)
    assert len(all_indicators) == 4

    # Indicators were properly created
    for c in combinations:
        assert c.indicators is not None

    # Sanity check for searchable parameters
    cycle_duration: GridParameter = combinations[0].parameters[0]
    assert cycle_duration.name == "cycle_duration"
    assert cycle_duration.single

    fast_ema_candle_count: GridParameter = combinations[0].parameters[1]
    assert fast_ema_candle_count.name == "fast_ema_candle_count"
    assert not fast_ema_candle_count.single

    assert len(combinations[0].searchable_parameters) == 1
    assert len(combinations[0].parameters) == 5

    # Multiprocess
    results_2 = perform_grid_search(
        _decide_trades_v4,
        strategy_universe,
        combinations,
        max_workers=4,
        multiprocess=True,
        trading_strategy_engine_version="0.5",
        indicator_storage=indicator_storage,
    )
    assert len(results_2) == 2

    filtered_results = [r for r in results_2 if r.get_parameter("fast_ema_candle_count") == 2]
    assert len(filtered_results) == 1

    # Check we got results back
    for r in results_2:
        assert r.metrics.loc["Sharpe"][0] != 0
        assert r.process_id > 1

    # Single thread
    results = perform_grid_search(
        _decide_trades_v4,
        strategy_universe,
        combinations,
        max_workers=1,
        multiprocess=True,
        trading_strategy_engine_version="0.5",
        indicator_storage=indicator_storage,
    )
    assert len(results) == 2


def test_visualise_grid_search_equity_curve(
    strategy_universe,
    indicator_storage,
    tmp_path,
):
    """Visualise grid search equity curve and other results.
    """
    class Parameters:
        cycle_duration = CycleDuration.cycle_1d
        initial_cash = 10_000
        test_param = [1, 2]

    def _decide_trades_flip_buy_sell(input: StrategyInput) -> List[TradeExecution]:
        """Every other day buy, every other sell."""
        position_manager = input.get_position_manager()
        pair = input.strategy_universe.get_single_pair()
        cash = position_manager.get_current_cash()
        if input.cycle % 2 == 0:
            return position_manager.open_spot(pair, cash * 0.99)
        else:
            if position_manager.is_any_open():
                return position_manager.close_all()
        return []

    def create_indicators(timestamp: datetime.datetime, parameters: StrategyParameters, strategy_universe: TradingStrategyUniverse, execution_context: ExecutionContext):
        # No indicators needed
        return IndicatorSet()

    combinations = prepare_grid_combinations(
        Parameters,
        tmp_path,
        strategy_universe=strategy_universe,
        create_indicators=create_indicators,
        execution_context=ExecutionContext(mode=ExecutionMode.unit_testing, grid_search=True),
    )

    assert len(combinations) == 2

    # Single thread
    grid_search_results = perform_grid_search(
        _decide_trades_flip_buy_sell,
        strategy_universe,
        combinations,
        max_workers=1,
        trading_strategy_engine_version="0.5",
        indicator_storage=indicator_storage,
    )
    best_results = find_best_grid_search_results(grid_search_results)

    fig = visualise_single_grid_search_result_benchmark(best_results.cagr[0], strategy_universe)
    assert len(fig.data) == 2


def test_perform_grid_search_with_indicator_dependency_resolution(
    strategy_universe,
    indicator_storage,
    tmp_path,
):
    """Check that grid search indicators can read data from other indicators

    """
    class MyParameters:
        cycle_duration = CycleDuration.cycle_1d
        initial_cash = 10_000

        # Indicator values that are searched in the grid search
        slow_ema_candle_count = [21, 30]
        fast_ema_candle_count = [7, 12]
        combined_indicator_modes = [1, 2]

    def combined_indicator(close: pd.Series, mode: int, pair: TradingPairIdentifier, dependency_resolver: IndicatorDependencyResolver):
        # An indicator that peeks the earlier grid search indicator calculations
        match mode:
            case 1:
                # When we look up data in grid search we need to give the parameter of which data we want,
                # and the trading pair if needed
                fast_ema = dependency_resolver.get_indicator_data("slow_ema", pair=pair, parameters={"length": 21})
                slow_ema = dependency_resolver.get_indicator_data("fast_ema", pair=pair, parameters={"length": 7})
            case 2:
                # Look up one set of parameters
                fast_ema = dependency_resolver.get_indicator_data("slow_ema", pair=pair, parameters={"length": 30})
                slow_ema = dependency_resolver.get_indicator_data("fast_ema", pair=pair, parameters={"length": 12})
            case _:
                raise NotImplementedError()

        return fast_ema * slow_ema * close # Calculate something based on two indicators and price

    def create_indicators(parameters: StrategyParameters, indicators: IndicatorSet, strategy_universe: TradingStrategyUniverse, execution_context: ExecutionContext):
        indicators.add("slow_ema", pandas_ta.ema, {"length": parameters.slow_ema_candle_count})
        indicators.add("fast_ema", pandas_ta.ema, {"length": parameters.fast_ema_candle_count})
        indicators.add(
            "combined_indicator",
            combined_indicator,
            {"mode": parameters.combined_indicator_modes},
            source=IndicatorSource.ohlcv,
            order=2,
        )

    combinations = prepare_grid_combinations(
        MyParameters,
        tmp_path,
        strategy_universe=strategy_universe,
        create_indicators=create_indicators,
        execution_context=ExecutionContext(mode=ExecutionMode.unit_testing, grid_search=True),
    )

    # fast ema 1, slow ema 7, my custom indicator
    c = combinations[0]
    assert len(c.indicators) == 3

    # fast ema 2, slow ema 7, my custom indicator
    c = combinations[1]
    assert len(c.indicators) == 3

    # {<IndicatorKey slow_ema(length=7)-WETH-USDC>, <IndicatorKey fast_ema(length=2)-WETH-USDC>, <IndicatorKey fast_ema(length=1)-WETH-USDC>, <IndicatorKey my_custom_indicator()-universe>}
    all_indicators = GridCombination.get_all_indicators(combinations)
    assert len(all_indicators) == 6

    # Indicators were properly created
    for c in combinations:
        assert c.indicators is not None

    # Multiprocess
    results_2 = perform_grid_search(
        _decide_trades_combined_indicator,
        strategy_universe,
        combinations,
        max_workers=4,
        multiprocess=True,
        trading_strategy_engine_version="0.5",
        indicator_storage=indicator_storage,
    )
    assert len(results_2) == 8

    # Single thread
    results = perform_grid_search(
        _decide_trades_combined_indicator,
        strategy_universe,
        combinations,
        max_workers=1,
        multiprocess=True,
        trading_strategy_engine_version="0.5",
        indicator_storage=indicator_storage,
    )
    assert len(results) == 8



def test_create_failed_result(tmp_path):
    """See we can serialise a failed result."""

    parameters = {
        "stop_loss_pct": [0.9, 0.95],
        "slow_ema_candle_count": [7, 9],
        "fast_ema_candle_count": [2, 3],
    }

    combinations = prepare_grid_combinations(parameters, tmp_path)

    r = create_grid_search_failed_result(
        combination=combinations[0],
        state=State(),
        exception=RuntimeError(),
    )

    assert r.exception is not None



def test_grid_out_of_balance(
    strategy_universe,
    indicator_storage,
    tmp_path,
):
    """Gracefully handle out of balance."""
    class MyParameters:
        cycle_duration = CycleDuration.cycle_1d
        initial_cash = 10_000
        test_param = [1, 2]


    def create_indicators(parameters: StrategyParameters, indicators: IndicatorSet, strategy_universe: TradingStrategyUniverse, execution_context: ExecutionContext):
        return indicators

    combinations = prepare_grid_combinations(
        MyParameters,
        tmp_path,
        strategy_universe=strategy_universe,
        create_indicators=create_indicators,
        execution_context=ExecutionContext(mode=ExecutionMode.unit_testing, grid_search=True),
    )

    # Multiprocess
    results = perform_grid_search(
        _decide_trades_out_of_balance,
        strategy_universe,
        combinations,
        max_workers=1,
        multiprocess=False,
        trading_strategy_engine_version="0.5",
        indicator_storage=indicator_storage,
        ignore_wallet_errors=True,
    )

    assert len(results) == 2
    for r in results:
        assert isinstance(r.exception, BacktestExecutionFailed)



def test_grid_search_visualisation_line_chart(
    strategy_universe,
    indicator_storage,
    tmp_path,
):
    """Advanced calculations and visualisation for grid search results.
    """
    class Parameters:
        cycle_duration = CycleDuration.cycle_1d
        initial_cash = 10_000
        allocation = [0.50, 0.75, 0.99]
        cycle_divider = [2, 3, 4]
        foo_param = ["a", "b"]

    def _decide_trades_flip_buy_sell(input: StrategyInput) -> list[TradeExecution]:
        # Generate some random trades
        position_manager = input.get_position_manager()
        parameters = input.parameters
        pair = input.strategy_universe.get_single_pair()
        cash = position_manager.get_current_cash()
        if input.cycle % parameters.cycle_divider == 0:
            return position_manager.open_spot(pair, cash * parameters.allocation)
        else:
            if position_manager.is_any_open():
                return position_manager.close_all()
        return []

    def create_indicators(timestamp: datetime.datetime, parameters: StrategyParameters, strategy_universe: TradingStrategyUniverse, execution_context: ExecutionContext):
        # No indicators needed
        return IndicatorSet()

    combinations = prepare_grid_combinations(
        Parameters,
        tmp_path,
        strategy_universe=strategy_universe,
        create_indicators=create_indicators,
        execution_context=ExecutionContext(mode=ExecutionMode.unit_testing, grid_search=True),
    )

    assert len(combinations) == 18

    grid_search_results = perform_grid_search(
        _decide_trades_flip_buy_sell,
        strategy_universe,
        combinations,
        trading_strategy_engine_version="0.5",
        indicator_storage=indicator_storage,
        verbose=False,
        multiprocess=True,
    )

    # Calculate rolling sharpe for each month
    # x-axis: time
    # y-axis: sharpe
    # variables as line charts: allocation=0.50, allocation=0.75, allocation=0.99
    # other variables are set to their fixed values
    df = calculate_rolling_metrics(
        grid_search_results,
        visualised_parameters="allocation",
        fixed_parameters={"cycle_divider": 2, "foo_param": "a"},
        benchmarked_metric=BenchmarkMetric.sharpe,
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

    # Check range is right
    assert df.index[0] == pd.Timestamp("2021-06-1")
    assert df.index[-1] == pd.Timestamp("2021-12-1")


    # pull out some values
    # (all negative sharpes, strategy does not make sense)
    assert df.loc["2021-07-01"][0.50] < 0
    assert df.loc["2021-07-01"][0.75] < 0
    assert df.loc["2021-07-01"][0.99] < 0

    # Draw line chart over time
    fig = visualise_grid_single_rolling_metric(df)
    assert isinstance(fig, Figure)

    # Draw evolving series of charts as a sublot
    fig = visualise_grid_rolling_metric_line_chart(
        df,
        range_start="2021-07-01",
        range_end="2021-09-01",
    )
    assert isinstance(fig, Figure)


def test_grid_search_visualisation_heatmap(
        strategy_universe,
        indicator_storage,
        tmp_path,
):
    """Advanced calculations and visualisation for grid search results.
    """

    class Parameters:
        cycle_duration = CycleDuration.cycle_1d
        initial_cash = 10_000
        allocation = [0.50, 0.75, 0.99]
        cycle_divider = [2, 3, 4]
        foo_param = ["a", "b"]

    def _decide_trades_flip_buy_sell(input: StrategyInput) -> list[TradeExecution]:
        # Generate some random trades
        position_manager = input.get_position_manager()
        parameters = input.parameters
        pair = input.strategy_universe.get_single_pair()
        cash = position_manager.get_current_cash()
        if input.cycle % parameters.cycle_divider == 0:
            return position_manager.open_spot(pair, cash * parameters.allocation)
        else:
            if position_manager.is_any_open():
                return position_manager.close_all()
        return []

    def create_indicators(timestamp: datetime.datetime, parameters: StrategyParameters, strategy_universe: TradingStrategyUniverse, execution_context: ExecutionContext):
        # No indicators needed
        return IndicatorSet()

    combinations = prepare_grid_combinations(
        Parameters,
        tmp_path,
        strategy_universe=strategy_universe,
        create_indicators=create_indicators,
        execution_context=ExecutionContext(mode=ExecutionMode.unit_testing, grid_search=True),
    )

    assert len(combinations) == 18

    grid_search_results = perform_grid_search(
        _decide_trades_flip_buy_sell,
        strategy_universe,
        combinations,
        trading_strategy_engine_version="0.5",
        indicator_storage=indicator_storage,
        verbose=False,
        multiprocess=True,
    )

    # Calculate rolling sharpe for each month
    # x-axis: time
    # y-axis: sharpe
    # variables as line charts: allocation=0.50, allocation=0.75, allocation=0.99
    # other variables are set to their fixed values
    df = calculate_rolling_metrics(
        grid_search_results,
        visualised_parameters=("allocation", "foo_param"),
        fixed_parameters={"cycle_divider": 2},
        benchmarked_metric=BenchmarkMetric.sharpe,
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

    # Check range is right
    assert df.index[0] == pd.Timestamp("2021-06-1")
    assert df.index[-1] == pd.Timestamp("2021-12-1")

    assert df.columns[0] == (0.5, "a")

    # pull out some values
    # (all negative sharpes, strategy does not make sense)
    assert df.loc["2021-07-01"][(0.5, 'a')] < 0
    assert df.loc["2021-07-01"][(0.5, 'b')] < 0
    assert df.loc["2021-07-01"][(0.75, 'b')] < 0

    # Draw evolving series of charts as a sublot
    fig = visualise_grid_rolling_metric_heatmap(
        df,
        range_start="2021-07-01",
        range_end="2021-09-01",
    )
    assert isinstance(fig, Figure)
