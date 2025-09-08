"""Optimiser tests."""
import datetime
from pathlib import Path
from typing import List

import pandas as pd
import pandas_ta_classic as pandas_ta
import pytest
from plotly.graph_objs import Figure
from skopt import space

from tradeexecutor.analysis.optimiser import profile_optimiser, plot_profile_duration_data
from tradeexecutor.backtest.optimiser import prepare_optimiser_parameters, perform_optimisation, OptimiserResult
from tradeexecutor.backtest.optimiser_functions import optimise_profit
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet, DiskIndicatorStorage
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.parameters import StrategyParameters
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import Exchange
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe


from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.state import State
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


def _decide_trades_v4(input: StrategyInput) -> List[TradeExecution]:
    """Checks some indicator logic works over grid search."""
    parameters = input.parameters
    assert "slow_ema_candle_count" in parameters
    assert "fast_ema_candle_count" in parameters

    if input.indicators.get_indicator_value("slow_ema") is not None:
        assert input.indicators.get_indicator_value("slow_ema") > 0

    series = input.indicators.get_indicator_series("slow_ema", unlimited=True)
    assert len(series) > 0

    assert 2 <=parameters.real_val <= 3

    return []


def test_prepare_optimiser_search_parameters(tmp_path):
    """Prepare grid search parameters."""

    class Parameters:
        stop_loss = space.Real(0.85, 0.99)
        max_asset_amount = space.Integer(3, 4)
        regime_filter_type = space.Categorical(["bull", "bull_and_bear"])

    parameters = prepare_optimiser_parameters(Parameters)
    assert isinstance(parameters, StrategyParameters)


def test_perform_optimisation_engine_v5_single_worker(
    strategy_universe,
    indicator_storage,
    tmp_path,
):
    """Run the optimiser using a single thread.
    """
    class Parameters:
        cycle_duration = CycleDuration.cycle_1d
        initial_cash = 10_000

        # Indicator values that are searched in the grid search
        slow_ema_candle_count = space.Integer(5, 7)
        fast_ema_candle_count = space.Integer(2, 4)
        real_val = space.Real(2, 3)

    def my_real_value_indicator(close: pd.Series, x: float):
        assert isinstance(x, float)
        return close

    def create_indicators(parameters: StrategyParameters, indicators: IndicatorSet, strategy_universe: TradingStrategyUniverse, execution_context: ExecutionContext):
        indicators.add("slow_ema", pandas_ta.ema, {"length": parameters.slow_ema_candle_count})
        indicators.add("fast_ema", pandas_ta.ema, {"length": parameters.fast_ema_candle_count})
        indicators.add("my_real_value_indicator", my_real_value_indicator, {"x": parameters.real_val})

    # Single process search
    result = perform_optimisation(
        iterations=3,
        search_func=optimise_profit,
        decide_trades=_decide_trades_v4,
        strategy_universe=strategy_universe,
        parameters=prepare_optimiser_parameters(Parameters),
        max_workers=1,
        indicator_storage=indicator_storage,
        result_path=tmp_path,
        create_indicators=create_indicators,
    )

    assert isinstance(result, OptimiserResult)

    # See the profile function runs
    profile_df = profile_optimiser(result)
    assert isinstance(profile_df, pd.DataFrame)

    fig = plot_profile_duration_data(profile_df)
    assert isinstance(fig, Figure)


def test_perform_optimisation_engine_v5_multi_worker(
    strategy_universe,
    indicator_storage,
    tmp_path,
):
    """Run the optimiser using multiprocess.
    """
    class Parameters:
        cycle_duration = CycleDuration.cycle_1d
        initial_cash = 10_000

        # Indicator values that are searched in the grid search
        slow_ema_candle_count = space.Integer(5, 7)
        fast_ema_candle_count = space.Integer(2, 4)
        real_val = space.Real(2, 3)

    def my_real_value_indicator(close: pd.Series, x: float):
        assert isinstance(x, float)
        assert 2 <= x <= 3
        return close

    def create_indicators(parameters: StrategyParameters, indicators: IndicatorSet, strategy_universe: TradingStrategyUniverse, execution_context: ExecutionContext):
        indicators.add("slow_ema", pandas_ta.ema, {"length": parameters.slow_ema_candle_count})
        indicators.add("fast_ema", pandas_ta.ema, {"length": parameters.fast_ema_candle_count})
        indicators.add("my_real_value_indicator", my_real_value_indicator, {"x": parameters.real_val})

    # Single process search
    result = perform_optimisation(
        iterations=3,
        search_func=optimise_profit,
        decide_trades=_decide_trades_v4,
        strategy_universe=strategy_universe,
        parameters=prepare_optimiser_parameters(Parameters),
        max_workers=3,
        indicator_storage=indicator_storage,
        result_path=tmp_path,
        create_indicators=create_indicators,
    )

    assert isinstance(result, OptimiserResult)

