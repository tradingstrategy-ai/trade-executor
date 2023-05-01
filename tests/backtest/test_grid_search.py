"""Grid search tests."""
import datetime
from pathlib import Path

import pandas as pd
import pytest
from plotly.graph_objs import Figure

from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import Exchange
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

from tradeexecutor.analysis.grid_search import analyse_grid_search_result, visualise_table, visualise_heatmap_2d
from tradeexecutor.backtest.grid_search import prepare_grid_combinations, run_grid_search_backtest, perform_grid_search, GridCombination, GridSearchResult, \
    pick_grid_search_result, pick_best_grid_search_result
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
    candle_universe = GroupedCandleUniverse.create_from_single_pair_dataframe(candles)

    universe = Universe(
        time_bucket=time_bucket,
        chains={mock_chain_id},
        exchanges={mock_exchange},
        pairs=pair_universe,
        candles=candle_universe,
        liquidity=None
    )

    return TradingStrategyUniverse(
        universe=universe,
        backtest_stop_loss_candles=candle_universe,
        backtest_stop_loss_time_bucket=time_bucket,
        reserve_assets=[weth_usdc.quote])


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
        cash = state.portfolio.get_current_cash()
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
                trades += position_manager.open_1x_long(pair, buy_amount, stop_loss_pct=stop_loss_pct)
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
        universe: TradingStrategyUniverse,
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
        universe,
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

    table = analyse_grid_search_result(results, drop_index=True)
    assert len(table) == 2 * 2 * 2
    row = table.iloc[0]
    # assert row["stop_loss_pct"] == 0.9
    assert row["Return"] == pytest.approx(0.0356848692865151)
    assert row["Positions"] == 2

    visualise_table(table)

    # Remove extra axis by focusing only stop_loss_pct=0.9
    heatmap_data = table.xs(0.9, level="stop_loss_pct")
    fig = visualise_heatmap_2d(heatmap_data, "fast_ema_candle_count", "slow_ema_candle_count", "Return")
    assert isinstance(fig, Figure)


def test_perform_grid_search_cached(
        universe: TradingStrategyUniverse,
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
        universe,
        combinations,
        max_workers=1,
    )

    for r in results:
        assert not r.cached

    results_2 = perform_grid_search(
        grid_search_worker,
        universe,
        combinations,
        max_workers=1,
    )

    for r in results_2:
        assert r.cached


def test_perform_grid_search_threaded(
        universe: TradingStrategyUniverse,
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
        universe,
        combinations,
        max_workers=4,
    )

    # Check we got results back
    for r in results:
        assert r.metrics.loc["Sharpe"][0] != 0


def test_perform_grid_search_multiprocess(
        universe: TradingStrategyUniverse,
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
        universe,
        combinations,
        max_workers=4,
        multiprocess=True,
    )

    assert len(results) == 4

    # Check we got results back
    for r in results:
        assert r.metrics.loc["Sharpe"][0] != 0
        assert r.process_id > 1
