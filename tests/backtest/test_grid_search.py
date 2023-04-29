"""Grid search tests."""
import datetime
from pathlib import Path

import pandas as pd
import pytest
from tradingstrategy.client import Client

from tradeexecutor.backtest.grid_search import prepare_grid_combinations, run_grid_search_backtest, perform_grid_search, GridCombination
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.visualisation import PlotKind
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, create_pair_universe_from_code
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange
from tradeexecutor.testing.synthetic_price_data import generate_ohlcv_candles
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import Exchange
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe


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
        internal_exchange_id=mock_exchange.exchange_id)


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

    return TradingStrategyUniverse(universe=universe, reserve_assets=[weth_usdc.quote])



def test_prepare_grid_search_parameters():
    """Prepare grid search parameters."""

    parameters = {
        "stop_loss": [0.9, 0.95],
        "max_asset_amount": [3, 4],
        "momentum_lookback_days": ["7d", "14d", "21d"]
    }

    combinations = prepare_grid_combinations(parameters)
    assert len(combinations) == 2 * 2 * 3

    first = combinations[0]
    assert first.parameters[0].name == "max_asset_amount"
    assert first.parameters[0].value == 3

    assert first.get_state_path() == Path('max_asset_amount=3/momentum_lookback_days=7d/stop_loss=0.9')



def test_perform_grid_search(
        universe: TradingStrategyUniverse,
        tmp_path,
):
    """Run a grid search."""

    def grid_search_worker(
            universe: TradingStrategyUniverse,
            combination: GridCombination,
    ):

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
            decide_trades,
            universe,
            name=f"Backtest slow:{slow_ema_candle_count} fast:{fast_ema_candle_count}",
        )


    parameters = {
        "stop_loss_pct": [0.9, 0.95],
        "slow_ema_candle_count": [7, 14, 21],
        "fast_ema_candle_count": [2, 5, 7],
    }

    combinations = prepare_grid_combinations(parameters)

    perform_grid_search(
        grid_search_worker,
        universe,
        combinations,
        tmp_path,
        max_workers=1,
    )