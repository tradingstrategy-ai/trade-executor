"""Indicator decorator tests."""

import datetime
import random

import pandas as pd
import pandas_ta
import pytest

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.strategy.execution_context import unit_test_execution_context
from tradeexecutor.strategy.pandas_trader.indicator import (
    DiskIndicatorStorage,
    IndicatorSource, IndicatorDependencyResolver,
    calculate_and_load_indicators_inline,
)
from tradeexecutor.strategy.pandas_trader.indicator_decorator import IndicatorRegistry
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, create_pair_universe_from_code
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange
from tradeexecutor.testing.synthetic_price_data import generate_multi_pair_candles
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe


@pytest.fixture(scope="module")
def strategy_universe() -> TradingStrategyUniverse:
    """Set up a mock universe with two pairs."""

    start_at = datetime.datetime(2021, 6, 1)
    end_at = datetime.datetime(2022, 1, 1)
    time_bucket = TimeBucket.d1

    # Set up fake assets
    mock_chain_id = ChainId.ethereum
    mock_exchange = generate_exchange(
        exchange_id=random.randint(1, 1000),
        chain_id=mock_chain_id,
        address=generate_random_ethereum_address(),
        exchange_slug="test-dex"
    )
    usdc = AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "USDC", 6, 1)
    weth = AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "WETH", 18, 2)
    wbtc = AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "WBTC", 18, 3)

    weth_usdc = TradingPairIdentifier(
        weth,
        usdc,
        generate_random_ethereum_address(),
        mock_exchange.address,
        internal_id=1,
        internal_exchange_id=mock_exchange.exchange_id,
        fee=0.0030,
    )

    wbtc_usdc = TradingPairIdentifier(
        wbtc,
        usdc,
        generate_random_ethereum_address(),
        mock_exchange.address,
        internal_id=2,
        internal_exchange_id=mock_exchange.exchange_id,
        fee=0.0030,
    )

    pair_universe = create_pair_universe_from_code(mock_chain_id, [weth_usdc, wbtc_usdc])

    candles = generate_multi_pair_candles(
        time_bucket,
        start_at,
        end_at,
        pairs={wbtc_usdc: 50_000, weth_usdc: 3000}
    )
    candle_universe = GroupedCandleUniverse(candles)

    universe = Universe(
        time_bucket=time_bucket,
        chains={mock_chain_id},
        exchanges={mock_exchange},
        pairs=pair_universe,
        candles=candle_universe,
        liquidity=None
    )

    return TradingStrategyUniverse(data_universe=universe, reserve_assets=[usdc])


@pytest.fixture
def indicator_storage(tmp_path, strategy_universe):
    return DiskIndicatorStorage(tmp_path, strategy_universe.get_cache_key())


def test_get_indicator_decorator_simple(strategy_universe):
    """Use decorator based indicators"""
    indicators = IndicatorRegistry()

    class Parameters:
        rsi_length = 20

    @indicators.define()
    def rsi(close, rsi_length):
        return pandas_ta.rsi(close, rsi_length)

    # Calculate indicators - will spawn multiple worker processed,
    # or load cached results from the disk
    parameters = StrategyParameters.from_class(Parameters)
    indicators = calculate_and_load_indicators_inline(
        strategy_universe=strategy_universe,
        parameters=parameters,
        create_indicators=indicators.create_indicators,
    )

    series = indicators.get_indicator_data_pairs_combined("rsi")

    assert isinstance(series, pd.Series)
    assert isinstance(series.index, pd.MultiIndex)

    pair_ids = series.index.get_level_values("pair_id")
    assert list(pair_ids.unique()) == [1, 2]


def test_indicator_decorator_order():
    """See our indicators decorator solves the depdendency order"""

    class Parameters:
        fast_ma_length = 7
        slow_ma_length = 21

    indicators = IndicatorRegistry()

    @indicators.define()
    def slow_sma(close, slow_ma_length):
        return pandas_ta.sma(close, slow_ma_length)

    @indicators.define()
    def fast_sma(close, fast_ma_length):
        return pandas_ta.sma(close, fast_ma_length)

    @indicators.define(dependencies=(slow_sma, fast_sma))
    def ma_crossover(
        close: pd.Series,
        pair: TradingPairIdentifier,
        dependency_resolver: IndicatorDependencyResolver,
    ) -> pd.Series:
        """Do cross-over calculation based on other two earlier moving average indicators.

        - This example calculates regions here fast moving average is above slow moving averag

        - Return pd.Series with True/False valeus and DatetimeIndex
        """
        slow_sma: pd.Series = dependency_resolver.get_indicator_data("slow_sma")
        fast_sma: pd.Series = dependency_resolver.get_indicator_data("fast_sma")
        return fast_sma > slow_sma

    @indicators.define(dependencies=(slow_sma, fast_sma))
    def ma_crossover_by_pair(
        close: pd.Series,
        pair: TradingPairIdentifier,
        dependency_resolver: IndicatorDependencyResolver,
    ) -> pd.Series:
        slow_sma: pd.Series = dependency_resolver.get_indicator_data("slow_sma", pair=pair)
        fast_sma: pd.Series = dependency_resolver.get_indicator_data("fast_sma", pair=pair)
        return fast_sma > slow_sma

    @indicators.define(dependencies=(slow_sma, fast_sma), source=IndicatorSource.dependencies_only_per_pair)
    def ma_crossover_by_pair_and_parameters(
        pair: TradingPairIdentifier,
        dependency_resolver: IndicatorDependencyResolver,
        fast_ma_length: int,
        slow_ma_length: int,
    ) -> pd.Series:
        slow_sma: pd.Series = dependency_resolver.get_indicator_data("slow_sma", pair=pair, parameters={"slow_ma_length": slow_ma_length})
        fast_sma: pd.Series = dependency_resolver.get_indicator_data("fast_sma", pair=pair, parameters={"fast_ma_length": fast_ma_length})
        return fast_sma > slow_sma

    @indicators.define(dependencies=(ma_crossover_by_pair_and_parameters,), source=IndicatorSource.dependencies_only_universe)
    def ma_universe(
        dependency_resolver: IndicatorDependencyResolver,
        fast_ma_length: int,
        slow_ma_length: int,
    ) -> pd.Series:
        df = dependency_resolver.get_indicator_data_pairs_combined(
            "ma_crossover_by_pair_and_parameters",
            parameters={
                "fast_ma_length": fast_ma_length,
                "slow_ma_length": slow_ma_length,
                "slow_ma_length": slow_ma_length,
            }
        )
        return df

    parameters = StrategyParameters.from_class(Parameters)
    indicator_set = indicators.create_indicators(
        timestamp=None,
        parameters=parameters,
        strategy_universe=strategy_universe,
        execution_context=unit_test_execution_context,
    )

    assert indicator_set.get_indicator("slow_sma").dependency_order == 1
    assert indicator_set.get_indicator("fast_sma").dependency_order == 1
    assert indicator_set.get_indicator("ma_crossover_by_pair").dependency_order == 2
    assert indicator_set.get_indicator("ma_crossover_by_pair_and_parameters").dependency_order == 2
    assert indicator_set.get_indicator("ma_crossover").dependency_order == 2
    assert indicator_set.get_indicator("ma_crossover").dependency_order == 2
    assert indicator_set.get_indicator("ma_universe").dependency_order == 3

