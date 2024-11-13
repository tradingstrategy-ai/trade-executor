"""Tests for alternative market data sources."""
import datetime
import os.path
from pathlib import Path

import pandas as pd
import pytest

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.strategy.pandas_trader.alternative_market_data import load_candle_universe_from_parquet, replace_candles
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, create_pair_universe_from_code
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange
from tradeexecutor.testing.synthetic_price_data import generate_ohlcv_candles
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import Exchange
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe


@pytest.fixture(scope="module")
def mock_chain_id() -> ChainId:
    """Mock a chai id."""
    return ChainId.ethereum


@pytest.fixture(scope="module")
def mock_exchange(mock_chain_id) -> Exchange:
    """Mock an exchange."""
    return generate_exchange(exchange_id=1, chain_id=mock_chain_id, address=generate_random_ethereum_address())


@pytest.fixture(scope="module")
def usdc() -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "USDC", 6, 1)


@pytest.fixture(scope="module")
def wbtc() -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "WBTC", 8, 2)


@pytest.fixture(scope="module")
def wbtc_usdc(mock_exchange, usdc, wbtc) -> TradingPairIdentifier:
    """Mock WBTC-USDC trading pair with 5 BPS fee"""
    return TradingPairIdentifier(
        wbtc,
        usdc,
        generate_random_ethereum_address(),
        mock_exchange.address,
        internal_id=555,
        internal_exchange_id=mock_exchange.exchange_id,
        fee=0.0005,
    )

@pytest.fixture(scope="module")
def synthetic_candles(wbtc_usdc) -> GroupedCandleUniverse:
    # Generate candles for pair_id = 1
    start_date = datetime.datetime(2021, 6, 1)
    end_date = datetime.datetime(2021, 7, 1)
    time_bucket = TimeBucket.h1
    candles = generate_ohlcv_candles(time_bucket, start_date, end_date, pair_id=wbtc_usdc.internal_id)
    candle_universe = GroupedCandleUniverse.create_from_single_pair_dataframe(candles, time_bucket)
    return candle_universe


@pytest.fixture()
def synthetic_universe(
        mock_chain_id: ChainId,
        mock_exchange: Exchange,
        wbtc_usdc: TradingPairIdentifier,
        synthetic_candles: GroupedCandleUniverse,
        usdc: AssetIdentifier) -> TradingStrategyUniverse:
    """Generate synthetic trading data universe for a single trading pair.

    Will be mutated in-place by tests.

    - Single mock exchange

    - Single mock trading pair

    - Random candles

    - No liquidity data available
    """

    pair_universe = create_pair_universe_from_code(mock_chain_id, [wbtc_usdc])

    universe = Universe(
        time_bucket=synthetic_candles.time_bucket,
        chains={mock_chain_id},
        exchanges={mock_exchange},
        pairs=pair_universe,
        candles=synthetic_candles,
        liquidity=None
    )

    return TradingStrategyUniverse(data_universe=universe, reserve_assets=[usdc])


@pytest.fixture()
def sample_file() -> Path:
    return Path(os.path.join(os.path.dirname(__file__), "binance-BTCUSDT-1h.parquet"))


def test_replace_candles(
    synthetic_universe: TradingStrategyUniverse,
    wbtc_usdc: TradingPairIdentifier,
    sample_file
):
    """Replace candles for WBTC-USDC from the Binance hourly feed."""

    assert synthetic_universe.data_universe.candles.time_bucket == TimeBucket.h1

    new_candles, _ = load_candle_universe_from_parquet(
        sample_file,
        pair=wbtc_usdc,
    )

    replace_candles(synthetic_universe, new_candles)

    start, end = synthetic_universe.data_universe.candles.get_timestamp_range()
    assert start == pd.Timestamp('2017-08-17 04:00:00')
    assert end == pd.Timestamp('2023-08-02 13:00:00')

    assert synthetic_universe.get_pair_count() == 1


def test_replace_candles_resample(
    synthetic_universe: TradingStrategyUniverse,
    wbtc_usdc,
    sample_file
):
    """Replace candles for WBTC-USDC from the Binance hourly feed."""

    assert synthetic_universe.data_universe.candles.time_bucket == TimeBucket.h1

    new_candles, stop_loss_candles = load_candle_universe_from_parquet(
        sample_file,
        resample=TimeBucket.h4,
        pair=wbtc_usdc,
    )

    assert new_candles.time_bucket == TimeBucket.h4

    replace_candles(synthetic_universe, new_candles, stop_loss_candles, ignore_time_bucket_mismatch=True)

    assert synthetic_universe.backtest_stop_loss_candles.time_bucket == TimeBucket.h1

    # Readjusted to 4h candles
    start, end = synthetic_universe.data_universe.candles.get_timestamp_range()
    assert start == pd.Timestamp('2017-08-17 04:00:00')
    assert end == pd.Timestamp('2023-08-02 12:00:00')


def test_replace_candles_resample_no_stop_loss(
    synthetic_universe: TradingStrategyUniverse,
    wbtc_usdc,
    sample_file
):
    """Replace candles for WBTC-USDC from the Binance hourly feed."""

    assert synthetic_universe.data_universe.candles.time_bucket == TimeBucket.h1

    new_candles, stop_loss_candles = load_candle_universe_from_parquet(
        sample_file,
        resample=TimeBucket.h4,
        include_as_trigger_signal=False,
        pair=wbtc_usdc,
    )

    assert stop_loss_candles is None

    replace_candles(synthetic_universe, new_candles, ignore_time_bucket_mismatch=True)

    start, end = synthetic_universe.data_universe.candles.get_timestamp_range()

    # Readjusted to 4h candles
    assert start == pd.Timestamp('2017-08-17 04:00:00')
    assert end == pd.Timestamp('2023-08-02 12:00:00')
