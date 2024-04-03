"""Test EMA cross-over strategy on synthetic data.

- Used to test different codepaths in backtesting

- The strategy profitabiltiy itself is not interesting

"""
import datetime
import logging
import os
from pathlib import Path

import pytest

import pandas as pd

from tradeexecutor.backtest.backtest_routing import BacktestRoutingModel
from tradeexecutor.backtest.backtest_runner import run_backtest, setup_backtest_for_universe
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, create_pair_universe_from_code
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange, generate_simple_routing_model
from tradeexecutor.testing.synthetic_price_data import generate_ohlcv_candles
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import Exchange, ExchangeUniverse
from tradingstrategy.pair import PandasPairUniverse, DEXPair
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request, mute_requests=False)


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


@pytest.fixture()
def strategy_path() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.join(os.path.dirname(__file__), "../..", "strategies", "test_only", "pandas_buy_every_second_day.py"))


@pytest.fixture()
def synthetic_universe(mock_chain_id, mock_exchange, weth_usdc) -> TradingStrategyUniverse:
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

    return TradingStrategyUniverse(data_universe=universe, reserve_assets=[weth_usdc.quote])


@pytest.fixture()
def routing_model(synthetic_universe) -> BacktestRoutingModel:
    return generate_simple_routing_model(synthetic_universe)


def test_synthetic_candles(
        logger: logging.Logger,
        strategy_path: Path,
        synthetic_universe: TradingStrategyUniverse,
    ):
    """Generate synthetic candle data."""

    start, end = synthetic_universe.data_universe.candles.get_timestamp_range()
    assert start == pd.Timestamp('2021-06-01 00:00:00')
    assert end == pd.Timestamp('2021-12-31 00:00:00')

    candles_for_pair = synthetic_universe.data_universe.candles.get_candles_by_pair(555)
    start = candles_for_pair.iloc[0]["timestamp"]
    end = candles_for_pair.iloc[-1]["timestamp"]
    assert start == pd.Timestamp('2021-06-01 00:00:00')
    assert end == pd.Timestamp('2021-12-31 00:00:00')


def test_synthetic_data_backtest_run(
        logger: logging.Logger,
        strategy_path: Path,
        synthetic_universe: TradingStrategyUniverse,
        routing_model: BacktestRoutingModel,
    ):
    """Run the strategy backtest.

    - Use synthetic data

    - Run a strategy for 6 months
    """

    # Run the test
    setup = setup_backtest_for_universe(
        strategy_path,
        start_at=datetime.datetime(2021, 6, 1),
        end_at=datetime.datetime(2022, 1, 1),
        cycle_duration=CycleDuration.cycle_1d,  # Override to use 24h cycles despite what strategy file says
        candle_time_frame=TimeBucket.d1,  # Override to use 24h cycles despite what strategy file says
        initial_deposit=10_000,
        universe=synthetic_universe,
        routing_model=routing_model,
        allow_missing_fees=True,
    )

    state, universe, debug_dump = run_backtest(setup, allow_missing_fees=True)

    assert len(debug_dump) == 215

    portfolio = state.portfolio
    assert len(list(portfolio.get_all_trades())) == 214
    buys = [t for t in portfolio.get_all_trades() if t.is_buy()]
    sells = [t for t in portfolio.get_all_trades() if t.is_sell()]

    assert len(buys) == 107
    assert len(sells) == 107

    # The actual result might vary, but we should slowly leak
    # portfolio valuation because losses on trading fees
    assert portfolio.get_cash() > 9000
    assert portfolio.get_cash() < 10_500
