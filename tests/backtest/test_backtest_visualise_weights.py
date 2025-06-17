"""Weights visualisation."""
import datetime
import random

import pandas as pd
import pytest
from plotly.graph_objs import Figure

from tradeexecutor.analysis.weights import calculate_asset_weights, visualise_weights, calculate_weights_statistics
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

from tradeexecutor.backtest.backtest_pricing import BacktestPricing
from tradeexecutor.backtest.backtest_routing import BacktestRoutingModel
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.backtest.backtest_runner import run_backtest_inline
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, create_pair_universe_from_code
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.strategy_module import StrategyParameters
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange, generate_simple_routing_model
from tradeexecutor.testing.synthetic_price_data import generate_ohlcv_candles



@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request, mute_requests=False)


@pytest.fixture(scope="module")
def strategy_universe() -> TradingStrategyUniverse:
    """Create ETH-USDC universe with only increasing data.

    - 1 months of data

    - Close price increase 1% every hour

    - Liquidity is a fixed 150,000 USD for the duration of the test
    """

    start_at = datetime.datetime(2021, 6, 1)
    end_at = datetime.datetime(2021, 7, 1)

    # Set up fake assets
    mock_chain_id = ChainId.ethereum
    mock_exchange = generate_exchange(
        exchange_id=random.randint(1, 1000),
        chain_id=mock_chain_id,
        address=generate_random_ethereum_address(),
        exchange_slug="my-dex",
    )
    usdc = AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "USDC", 6, 1)
    weth = AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "WETH", 18, 2)
    wbtc = AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "WBTC", 6, 3)
    weth_usdc = TradingPairIdentifier(
        weth,
        usdc,
        generate_random_ethereum_address(),
        mock_exchange.address,
        internal_id=weth.internal_id,
        internal_exchange_id=mock_exchange.exchange_id,
        fee=0.0030,
    )
    wbtc_usdc = TradingPairIdentifier(
        wbtc,
        usdc,
        generate_random_ethereum_address(),
        mock_exchange.address,
        internal_id=wbtc.internal_id,
        internal_exchange_id=mock_exchange.exchange_id,
        fee=0.0030,
    )

    time_bucket = TimeBucket.d1

    pair_universe = create_pair_universe_from_code(mock_chain_id, [weth_usdc, wbtc_usdc])

    # Create 1h underlying trade signal
    weth_candles = generate_ohlcv_candles(
        time_bucket,
        start_at,
        end_at,
        pair_id=weth_usdc.internal_id,
        daily_drift=(1.01, 1.01),
        high_drift=1.05,
        low_drift=0.90,
        random_seed=1,
    )
    wbtc_candles = generate_ohlcv_candles(
        time_bucket,
        start_at,
        end_at,
        pair_id=wbtc_usdc.internal_id,
        daily_drift=(1.01, 1.01),
        high_drift=1.05,
        low_drift=0.90,
        random_seed=1,
    )

    candle_universe = GroupedCandleUniverse.create_from_multiple_candle_dataframes(
        [weth_candles, wbtc_candles],
    )

    universe = Universe(
        time_bucket=TimeBucket.d1,
        chains={mock_chain_id},
        exchanges={mock_exchange},
        pairs=pair_universe,
        candles=candle_universe,
    )
    universe.pairs.exchange_universe = universe.exchange_universe

    return TradingStrategyUniverse(
        data_universe=universe,
        reserve_assets=[usdc],
        backtest_stop_loss_time_bucket=time_bucket,
    )


@pytest.fixture()
def routing_model(synthetic_universe) -> BacktestRoutingModel:
    return generate_simple_routing_model(synthetic_universe)


@pytest.fixture()
def pricing_model(synthetic_universe, routing_model) -> BacktestPricing:
    pricing_model = BacktestPricing(
        synthetic_universe.data_universe.candles,
        routing_model,
        allow_missing_fees=True,
    )
    return pricing_model


def create_indicators(timestamp: datetime.datetime, parameters: StrategyParameters, strategy_universe: TradingStrategyUniverse, execution_context: ExecutionContext):
    # No indicators needed
    return IndicatorSet()


def decide_trades(input: StrategyInput) -> list[TradeExecution]:
    # Buy and sell BTC or WETH randomly
    position_manager = input.get_position_manager()
    cash = input.state.portfolio.get_cash()
    strategy_universe = input.strategy_universe
    cycle = input.cycle
    state = input.state

    trades = []

    # Track which asset is our turn to buy
    asset_turn = state.other_data.load_latest("asset_turn", 1)

    if not position_manager.is_any_open():
        if asset_turn % 2 == 0:
            pair = strategy_universe.get_pair_by_id(2)
        else:
            pair = strategy_universe.get_pair_by_id(3)

        trades += position_manager.open_spot(
            pair,
            cash * 0.99,
        )

        asset_turn += 1
    else:
        trades += position_manager.close_all()

    state.other_data.save(cycle, "asset_turn", asset_turn)
    return trades


def test_visualise_weights(strategy_universe, tmp_path):
    """Test visualising portfolio weights of a backtest."""

    # Start with $1M cash, far exceeding the market size
    class Parameters:
        backtest_start = strategy_universe.data_universe.candles.get_timestamp_range()[0].to_pydatetime()
        backtest_end = strategy_universe.data_universe.candles.get_timestamp_range()[1].to_pydatetime()
        initial_cash = 1_000_000
        cycle_duration = CycleDuration.cycle_1d

    # Run the test
    result = run_backtest_inline(
        client=None,
        decide_trades=decide_trades,
        create_indicators=create_indicators,
        universe=strategy_universe,
        reserve_currency=ReserveCurrency.usdc,
        engine_version="0.5",
        parameters=StrategyParameters.from_class(Parameters),
        mode=ExecutionMode.unit_testing,
    )

    state = result.state
    weights_series = calculate_asset_weights(state)

    assert isinstance(weights_series.index, pd.MultiIndex)
    assets = weights_series.index.get_level_values(1).unique()
    assert set(assets) == {"USDC", "WBTC", "WETH"}

    import tradeexecutor.monkeypatch.plotly

    fig = visualise_weights(weights_series)
    assert isinstance(fig, Figure)

    fig.show()

    weight_stats = calculate_weights_statistics(weights_series)
    assert isinstance(weight_stats, pd.DataFrame)

