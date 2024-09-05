"""Run a dummy strategy stressing market limit and take profit orders.

- Run daily decide_trade cycle that prepares orders for hourly cycle
- Strategy decision cycle is 24h
- The backtest trigger check is 1h signal

"""
import datetime
import random
from typing import List, Dict

import pytest
import pandas as pd

from tradeexecutor.backtest.backtest_pricing import BacktestPricing
from tradeexecutor.backtest.backtest_routing import BacktestRoutingModel
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.statistics.summary import calculate_summary_statistics
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.strategy_module import StrategyParameters
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange, generate_simple_routing_model
from tradeexecutor.testing.synthetic_price_data import generate_ohlcv_candles
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket

from tradeexecutor.backtest.backtest_runner import run_backtest_inline
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, create_pair_universe_from_code
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.state.trade import TradeExecution
from tradingstrategy.universe import Universe
from tradingstrategy.utils.groupeduniverse import resample_candles


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request, mute_requests=False)


@pytest.fixture(scope="module")
def strategy_universe() -> TradingStrategyUniverse:
    """Create ETH-USDC universe with only increasing data.

    - Close price increase 1% every hour
    """

    start_at = datetime.datetime(2021, 6, 1)
    end_at = datetime.datetime(2022, 1, 1)

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
    weth_usdc = TradingPairIdentifier(
        weth,
        usdc,
        generate_random_ethereum_address(),
        mock_exchange.address,
        internal_id=random.randint(1, 1000),
        internal_exchange_id=mock_exchange.exchange_id,
        fee=0.0030,
    )

    time_bucket = TimeBucket.h1

    pair_universe = create_pair_universe_from_code(mock_chain_id, [weth_usdc])

    # Create 1h underlying trade signal
    stop_loss_candles = generate_ohlcv_candles(
        time_bucket,
        start_at,
        end_at,
        pair_id=weth_usdc.internal_id,
        daily_drift=(1.01, 1.01),
        high_drift=1.05,
        low_drift=0.90,
        random_seed=1,
    )
    stop_loss_candle_universe = GroupedCandleUniverse.create_from_single_pair_dataframe(stop_loss_candles)

    # Create upsampled daily candles
    daily_candles = resample_candles(stop_loss_candles, "D")
    candle_universe = GroupedCandleUniverse.create_from_single_pair_dataframe(daily_candles)

    universe = Universe(
        time_bucket=TimeBucket.d1,
        chains={mock_chain_id},
        exchanges={mock_exchange},
        pairs=pair_universe,
        candles=candle_universe,
        liquidity=None
    )
    universe.pairs.exchange_universe = universe.exchange_universe

    return TradingStrategyUniverse(
        data_universe=universe,
        reserve_assets=[usdc],
        backtest_stop_loss_candles=stop_loss_candle_universe,
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


def decide_trades(input: StrategyInput) -> List[TradeExecution]:
    """Example decide_trades function using market limits and partial take profits."""
    position_manager = input.get_position_manager()
    pair = input.get_default_pair()
    cash = input.state.portfolio.get_cash()
    indicators = input.indicators
    portfolio = input.state.portfolio

    midnight_price = indicators.get_price()
    if midnight_price is None:
        # Skip cycle 1
        # We do not have the previous day price available at the first cycle
        return []

    # Only set a trigger open if we do not have any position open/pending yet
    if not position_manager.get_current_position_for_pair(pair, pending=True):

        position_manager.log(f"Setting up a new market limit trigger position for {pair}")

        # Set market limit if we break above level during the day,
        # with a conditional open position
        position, pending_trades = position_manager.open_spot_with_market_limit(
            pair=pair,
            value=cash*0.99,  # Cannot do 100% because of floating point rounding errors
            trigger_price=midnight_price * 1.01,
            expires_at=input.timestamp + pd.Timedelta(hours=24),
            notes="Market limit test open trade",
        )

        assert len(portfolio.pending_positions) == 1
        assert len(portfolio.open_positions) == 0

        # We do not know the accurage quantity we need to close,
        # because of occuring slippage,
        # but we use the close flag below to close the remaining]
        # amount
        total_quantity = position.get_pending_quantity()
        assert total_quantity > 0

        # Set two take profits to 1.5% and 2% price increase
        # First will close 2/3 of position
        # The second will close the remaining position
        position_manager.prepare_take_profit_trades(
            position,
            [
                (midnight_price * 1.015, -total_quantity * 2 / 3, False),
                (midnight_price * 1.02, -total_quantity * 1 / 3, True),
            ]
        )

    else:
        position_manager.log("Existing position pending - do not create new")

    if input.cycle == 2:
        # Check that the trade decision logic above worked correctly.
        # At this stage, we should not yet have an open position.
        assert len(portfolio.pending_positions) == 1
        assert len(portfolio.open_positions) == 0
        assert len(portfolio.closed_positions) == 0

    # We return zero trades here, as all of trades we have constructed
    # are pending for a trigger, and do not need to be executed
    # on this decision cycle
    return []


def test_market_limit_take_profit_strategy(strategy_universe, tmp_path):
    """Test DecideTradesProtocolV4

    - Check that StrategyInput is passed correctly in backtesting (only backtesting, not live trading)
    """

    class Parameters:
        backtest_start = strategy_universe.data_universe.candles.get_timestamp_range()[0].to_pydatetime()
        backtest_end = strategy_universe.data_universe.candles.get_timestamp_range()[0].to_pydatetime()  + datetime.timedelta(days=4)
        initial_cash = 10_0000
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
    assert len(result.diagnostics_data) == 6  # Entry for each day + few extras
    assert len(state.portfolio.closed_positions) == 3
    assert len(state.portfolio.pending_positions) == 0
    assert len(state.portfolio.open_positions) == 0

    # Check these do not crash on market limit positions
    calculate_summary_statistics(
        state,
        ExecutionMode.unit_testing_trading,
        now_=datetime.datetime.utcnow(),
    )
