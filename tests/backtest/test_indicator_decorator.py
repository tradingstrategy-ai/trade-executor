"""Indicator decorator tests."""

import datetime
import random

import pandas as pd
import pandas_ta_classic as pandas_ta
import pytest
from pandas._libs.tslibs.offsets import MonthBegin
from pyasn1_modules.rfc8018 import id_PBMAC1

from tradeexecutor.backtest.backtest_runner import run_backtest_inline, BacktestResult
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.execution_context import unit_test_execution_context
from tradeexecutor.strategy.pandas_trader.indicator import (
    DiskIndicatorStorage,
    IndicatorSource, IndicatorDependencyResolver,
    calculate_and_load_indicators_inline, prepare_indicators, IndicatorDefinition, IndicatorNotFound, load_indicators, MemoryIndicatorStorage,
)
from tradeexecutor.strategy.pandas_trader.indicator_decorator import IndicatorRegistry, flatten_dict_permutations
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInputIndicators, StrategyInput, IndicatorWithVariations
from tradeexecutor.strategy.parameters import StrategyParameters, RollingParameter, RollingParameterValueNotAvailable, display_parameters
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

    parameters = StrategyParameters.from_class(Parameters)
    indicators = calculate_and_load_indicators_inline(
        strategy_universe=strategy_universe,
        parameters=parameters,
        create_indicators=indicators.create_indicators,
        verbose=False,
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
        slow_sma: pd.Series = dependency_resolver.get_indicator_data(slow_sma)
        fast_sma: pd.Series = dependency_resolver.get_indicator_data(fast_sma)
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

    df = indicators.get_diagnostics()
    # print("\n" + str(df))
    assert len(df) == 6

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


def test_get_indicator_decorator_arguments(strategy_universe):
    """See arguments passed to the indicator functions look good."""
    indicators = IndicatorRegistry()

    class Parameters:
        rsi_length = 20

    @indicators.define()
    def rsi(close, rsi_length, pair, dependency_resolver):
        assert isinstance(close, pd.Series)
        assert type(rsi_length) == int
        assert isinstance(pair, TradingPairIdentifier)
        assert isinstance(dependency_resolver, IndicatorDependencyResolver)
        return pd.Series([], dtype="float64")

    @indicators.define(source=IndicatorSource.dependencies_only_per_pair, dependencies=[rsi])
    def rsi_derivate(rsi_length, pair, dependency_resolver):
        assert type(rsi_length) == int
        assert isinstance(pair, TradingPairIdentifier)
        assert isinstance(dependency_resolver, IndicatorDependencyResolver)
        return pd.Series([], dtype="float64")

    parameters = StrategyParameters.from_class(Parameters)
    indicators = calculate_and_load_indicators_inline(
        strategy_universe=strategy_universe,
        parameters=parameters,
        create_indicators=indicators.create_indicators,
        verbose=False,
    )

    assert isinstance(indicators, StrategyInputIndicators)


def test_get_indicator_rolling_parameters(strategy_universe, indicator_storage, tmp_path):
    """We create multiple indicator parameter variations for rolling indicators.

    - Test reading rolling parameter values
    - Test calculating indicators for varying rolling parameters, with multiple copies
    """
    indicators = IndicatorRegistry()

    rolling_data = pd.Series(
        data=[21, 22, 23, 24, 25, 26],
        index=pd.Index([
            pd.Timestamp("2021-06-01"),
            pd.Timestamp("2021-07-01"),
            pd.Timestamp("2021-08-01"),
            pd.Timestamp("2021-09-01"),
            pd.Timestamp("2021-10-01"),
            pd.Timestamp("2021-12-01"),
        ]),
    )

    other_param_data = pd.Series(
        data=[1, 2, 3, 4, 5, 6],
        index=pd.Index([
            pd.Timestamp("2021-06-01"),
            pd.Timestamp("2021-07-01"),
            pd.Timestamp("2021-08-01"),
            pd.Timestamp("2021-09-01"),
            pd.Timestamp("2021-10-01"),
            pd.Timestamp("2021-12-01"),
        ]),
    )

    class Parameters:

        fixed_parameter = 10

        rsi_length = RollingParameter(
            name="rsi_length",
            freq=MonthBegin(1),
            values=rolling_data,
        )

        other_param = RollingParameter(
            name="other_param",
            freq=MonthBegin(1),
            values=other_param_data,
        )

        backtest_start = datetime.datetime(2021, 6, 1)
        backtest_end = datetime.datetime(2022, 1, 1)
        initial_cash = 10_000
        cycle_duration = CycleDuration.cycle_1d

    @indicators.define()
    def fixed_rsi(close, fixed_parameter, pair, dependency_resolver):
        assert isinstance(close, pd.Series)
        assert type(fixed_parameter) == int
        assert isinstance(pair, TradingPairIdentifier)
        assert isinstance(dependency_resolver, IndicatorDependencyResolver)
        return close

    @indicators.define()
    def rsi(close, rsi_length, pair, dependency_resolver):
        assert isinstance(close, pd.Series)
        assert type(rsi_length) == int
        assert isinstance(pair, TradingPairIdentifier)
        assert isinstance(dependency_resolver, IndicatorDependencyResolver)
        return close * rsi_length

    @indicators.define(source=IndicatorSource.dependencies_only_per_pair, dependencies=[rsi])
    def rsi_derivative(rsi_length, other_param, pair, dependency_resolver):
        assert type(rsi_length) == int
        assert type(other_param) == int
        assert isinstance(pair, TradingPairIdentifier)
        assert isinstance(dependency_resolver, IndicatorDependencyResolver)
        rsi = dependency_resolver.get_indicator_data(
            "rsi",
            pair=pair,
            parameters={
                "rsi_length": rsi_length,
            }
        )
        return rsi * other_param

    @indicators.define(dependencies=(rsi_derivative,), source=IndicatorSource.dependencies_only_universe)
    def third_degree(
        rsi_length: int,
        other_param: int,
        dependency_resolver: IndicatorDependencyResolver
    ) -> pd.Series:
        """Series where each timestamp is the list of pairs meeting all inclusion criteria.

        :return:
            Series with pair count for each timestamp
        """
        series = dependency_resolver.get_indicator_data_pairs_combined(
            "rsi_derivative",
            parameters={
                "rsi_length": rsi_length,
                "other_param": other_param,
            },
        )
        return series

    parameters = StrategyParameters.from_class(Parameters)
    df = display_parameters(parameters)
    assert "RollingParameter" in str(df)
    assert "[1, 2, 3, 4, 5, 6]" in str(df)

    # Test reading rolling parameter values
    val = parameters.get_rolling_parameter("rsi_length", pd.Timestamp("2021-06-02"))
    assert val == 21

    val = parameters.get_rolling_parameter("rsi_length", pd.Timestamp("2021-12-01"))
    assert val == 26

    val = parameters.get_rolling_parameter("rsi_length", pd.Timestamp("2021-12-02"))
    assert val == 26

    val = parameters.get_rolling_parameter("rsi_length", pd.Timestamp("2021-06-01"))
    assert val == 21

    with pytest.raises(RollingParameterValueNotAvailable):
        _ = parameters.get_rolling_parameter("rsi_length", pd.Timestamp("2021-05-29"))

    with pytest.raises(RollingParameterValueNotAvailable):
        _ = parameters.get_rolling_parameter("rsi_length", pd.Timestamp("2022-01-01"))

    # Test calculating indicators for varying rolling parameters, with multiple copies
    indicator_set = prepare_indicators(
        indicators.create_indicators,
        parameters,
        strategy_universe,
        unit_test_execution_context,
    )

    for ind in indicator_set.indicators.values():
        assert isinstance(ind, IndicatorDefinition)
        if not ind.name.startswith("fixed"):
            assert ind.variations is True

    assert len(indicator_set.indicators.values()) == 79

    all_combinations = list(indicator_set.generate_combinations(strategy_universe))
    all_combinations.sort(key=lambda x: str(x))
    assert len(all_combinations) == 122
    for combination in all_combinations:
        for key, value in combination.definition.parameters.items():
            assert not isinstance(value, list)

    storage = MemoryIndicatorStorage(strategy_universe.get_cache_key())
    strategy_input_indicators = calculate_and_load_indicators_inline(
        strategy_universe=strategy_universe,
        parameters=parameters,
        indicator_set=indicator_set,
        verbose=False,
        max_workers=1,
        storage=storage,
    )

    assert isinstance(strategy_input_indicators, StrategyInputIndicators)

    # Make sure we have access to every variation of the indicator
    def decide_trades(input: StrategyInput):
        timestamp = input.timestamp
        indicators = input.indicators

        pair = input.strategy_universe.get_pair_by_id(1)

        _ = indicators.get_indicator_value("fixed_rsi", pair=pair)

        with pytest.raises(IndicatorWithVariations):
            indicators.get_indicator_value("rsi", pair=pair)

        with pytest.raises(IndicatorWithVariations):
            indicators.get_indicator_value("rsi_derivative", pair=pair)

        rsi_1 = indicators.get_indicator_value(
            "rsi",
            pair=pair,
            parameters={"rsi_length": 21},
        )
        assert rsi_1 > 0

        rsi_2 = indicators.get_indicator_value(
            "rsi",
            pair=pair,
            parameters={"rsi_length": 22},
        )
        assert rsi_2 > 0

        rsi_3 = indicators.get_indicator_value(
            "rsi_derivative",
            pair=pair,
            parameters={
                "rsi_length": 22,
                "other_param": 2,
            },
        )
        assert rsi_3 > 3

        with pytest.raises(IndicatorNotFound):
            _ = indicators.get_indicator_value(
                "rsi_derivative",
                pair=pair,
                parameters={
                    "rsi_length": 0,
                    "other_param": 0,
                },
            )

        return []

    backtest_result = run_backtest_inline(
        start_at=datetime.datetime(2021, 6, 1),
        end_at=datetime.datetime(2022, 1, 1),
        client=None,
        cycle_duration=CycleDuration.cycle_1d,
        decide_trades=decide_trades,
        universe=strategy_universe,
        engine_version="0.5",
        create_indicators=indicators.create_indicators,
        parameters=parameters,
        max_workers=1,
    )

    assert isinstance(backtest_result, BacktestResult)

    # Make sure we can load these values
    strategy_input_indicators = calculate_and_load_indicators_inline(
        strategy_universe=strategy_universe,
        parameters=parameters,
        create_indicators=indicators.create_indicators,
        verbose=False,
        indicator_storage_path=tmp_path,
        max_workers=1,
    )

    indicator_set = prepare_indicators(
        indicators.create_indicators,
        parameters,
        strategy_universe,
        unit_test_execution_context,
        timestamp=None,
    )
    storage = DiskIndicatorStorage(tmp_path, strategy_universe.get_cache_key())
    all_combinations = set(indicator_set.generate_combinations(strategy_universe))

    for combination in all_combinations:
        assert storage.is_available(combination)
        for key, value in combination.definition.parameters.items():
            assert not isinstance(value, list), f"Bad key: {key}, value: {value}, {value.__class__}, {combination.definition}"
            assert not isinstance(value, RollingParameter), f"RollingParameter not expanded: {key}, value: {value}, {value.__class__}, {combination.definition}"

    strategy_input_indicators = load_indicators(
        strategy_universe=strategy_universe,
        storage=storage,
        indicator_set=indicator_set,
        all_combinations=all_combinations,
    )


def test_param_permutations():
    """We generate permutations correctly for varying-indicator parameters"""

    rolling_data = pd.Series(
        data=[21, 22, 23],
        index=pd.Index([
            pd.Timestamp("2021-06-01"),
            pd.Timestamp("2021-07-01"),
            pd.Timestamp("2021-08-01"),
        ]),
    )

    other_param_data = pd.Series(
        data=[1, 2],
        index=pd.Index([
            pd.Timestamp("2021-06-01"),
            pd.Timestamp("2021-07-01"),
        ]),
    )

    input = {
        "rsi":  RollingParameter(
            name="rsi_length",
            freq=MonthBegin(1),
            values=rolling_data,
        ),
        "other_param": RollingParameter(
            name="other_param",
            freq=MonthBegin(1),
            values=other_param_data,
        ),
        "fixed_val": 1,
    }
    permutations = flatten_dict_permutations(input)
    assert len(permutations) == 6

    input = {
        "rsi":  RollingParameter(
            name="rsi_length",
            freq=MonthBegin(1),
            values=rolling_data,
        ),
    }
    permutations = flatten_dict_permutations(input)
    assert len(permutations) == 3
