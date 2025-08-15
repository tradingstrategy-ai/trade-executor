"""Backtest based on trade size limit on TVL data."""
import datetime
import random

import pytest

from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.liquidity import GroupedLiquidityUniverse
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
from tradeexecutor.strategy.tvl_size_risk import USDTVLSizeRiskModel
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange, generate_simple_routing_model
from tradeexecutor.testing.synthetic_price_data import generate_ohlcv_candles, generate_tvl_candles



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

    # Create 1h underlying trade signal
    daily_candles = generate_ohlcv_candles(
        time_bucket,
        start_at,
        end_at,
        pair_id=weth_usdc.internal_id,
        daily_drift=(1.01, 1.01),
        high_drift=1.05,
        low_drift=0.90,
        random_seed=1,
    )
    candle_universe = GroupedCandleUniverse.create_from_single_pair_dataframe(daily_candles)

    # Pool liquidity for this pair stays at fixed $150k
    liquidity_candles = generate_tvl_candles(
        time_bucket,
        start_at,
        end_at,
        start_liquidity=150_000,
        pair_id=weth_usdc.internal_id,
        daily_drift=(1.00, 1.00),
        high_drift=1.00,
        low_drift=1.00,
        random_seed=1,
    )
    liquidity_candles = GroupedLiquidityUniverse.create_from_single_pair_dataframe(liquidity_candles)

    universe = Universe(
        time_bucket=TimeBucket.d1,
        chains={mock_chain_id},
        exchanges={mock_exchange},
        pairs=pair_universe,
        candles=candle_universe,
        liquidity=liquidity_candles
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
    """Example decide_trades function that opens a position with a trade size limit."""
    position_manager = input.get_position_manager()
    pair = input.get_default_pair()
    cash = input.state.portfolio.get_cash()
    timestamp = input.timestamp

    size_risker = USDTVLSizeRiskModel(
        input.pricing_model,
        per_trade_cap=0.02,  # Cap trade to 2% of pool TVL
    )

    trades = []
    if not position_manager.is_any_open():
        # Ask the size risk model what kind of estimation they give for this pair
        # and then cap the trade size based on this
        size_risk = size_risker.get_acceptable_size_for_buy(timestamp, pair, cash)
        # We never enter 100% position with out cash,
        # as floating points do not work well with ERC-20 18 decimal accuracy
        # and all kind of problematic rounding errors would happen.
        position_size = min(cash * 0.99, size_risk.accepted_size)
        trades += position_manager.open_spot(
            pair,
            position_size
        )
    else:
        trades += position_manager.close_all()
    return trades


def test_tvl_trade_size_limit(strategy_universe, tmp_path):
    """Test that we limit the size of the trade based TVL

    - We have much more excess cash than we can put on a position
    """

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
        three_leg_resolution=False,
    )

    state = result.state
    assert len(state.portfolio.closed_positions) == 14
    assert len(state.portfolio.open_positions) == 1

    for t in state.portfolio.get_all_trades():
        # We are limited to 2% of the pool size,
        # or 150_000 * 0.02,
        # add some rounding error
        assert t.get_volume() <= 3050


