"""Indicator definition, building and caching tests."""

import datetime
import random
from pathlib import Path

import pandas as pd
import pandas_ta
import pytest


from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.strategy.execution_context import ExecutionContext, unit_test_execution_context, unit_test_trading_execution_context
from tradeexecutor.strategy.pandas_trader.indicator import (
    IndicatorSet, DiskIndicatorStorage, IndicatorDefinition, IndicatorFunctionSignatureMismatch,
    calculate_and_load_indicators, IndicatorKey, IndicatorSource, IndicatorDependencyResolver,
    IndicatorCalculationFailed, MemoryIndicatorStorage, calculate_and_load_indicators_inline, calculate_and_load_indicators_inline,
)
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInputIndicators
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, create_pair_universe_from_code, load_partial_data
from tradeexecutor.strategy.universe_model import UniverseOptions
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



def test_setup_up_indicator_storage_per_pair(tmp_path, strategy_universe):
    """Create an indicator storage for a trading pair indicator combo."""

    storage = DiskIndicatorStorage(Path(tmp_path), universe_key=strategy_universe.get_cache_key())
    assert storage.path == Path(tmp_path)
    assert storage.universe_key == "ethereum_1d_WETH-USDC-WBTC-USDC_2021-06-01-2021-12-31"

    pair = strategy_universe.get_pair_by_human_description((ChainId.ethereum, "test-dex", "WETH", "USDC"))

    ind = IndicatorDefinition(
        name="sma",
        func=pandas_ta.sma,
        parameters={"length": 21},
    )

    key = IndicatorKey(pair, ind)

    ind_path = storage.get_indicator_path(key)

    # pandas_ta function signature changes in every releease
    assert str(ind_path).startswith(str(Path(tmp_path) / storage.universe_key))
    assert str(ind_path).endswith("(length=21)-WETH-USDC.parquet")
    # / "sma_dfc27ff4"


def test_setup_up_indicator_universe(tmp_path, strategy_universe):
    """Create an indicator storage for a universe indicator."""

    storage = DiskIndicatorStorage(Path(tmp_path), universe_key=strategy_universe.get_cache_key())
    
    ind = IndicatorDefinition(
        name="foobar",
        func=lambda length: pd.Series(),
        parameters={"length": 21},
    )

    key = IndicatorKey(None, ind)

    ind_path = storage.get_indicator_path(key)
    # signature changes in every pandas_ta release
    #assert ind_path == Path(tmp_path) / storage.universe_key / "foobar_01b82463(length=21)-universe.parquet"
    assert str(ind_path).endswith("-universe.parquet")


def test_setup_up_indicator_storage_two_parameters(tmp_path, strategy_universe):
    """Create an indicator storage for test universe using two indicators."""

    storage = DiskIndicatorStorage(Path(tmp_path), universe_key=strategy_universe.get_cache_key())
    assert storage.path == Path(tmp_path)
    assert storage.universe_key == "ethereum_1d_WETH-USDC-WBTC-USDC_2021-06-01-2021-12-31"

    pair = strategy_universe.get_pair_by_human_description((ChainId.ethereum, "test-dex", "WETH", "USDC"))

    ind = IndicatorDefinition(
        name="sma",
        func=pandas_ta.sma,
        parameters={"length": 21, "offset": 1},
    )

    key = IndicatorKey(pair, ind)

    # Signature changes in every pandas_ta release
    ind_path = storage.get_indicator_path(key)
    # assert ind_path == Path(tmp_path) / storage.universe_key / "sma_dfc27ff4(length=21,offset=1)-WETH-USDC.parquet"
    assert str(ind_path).endswith("-WETH-USDC.parquet")



def test_bad_indicator_parameters(tmp_path, strategy_universe):
    """Check for path passed functional parameters."""

    with pytest.raises(IndicatorFunctionSignatureMismatch):
        IndicatorDefinition(
            name="sma",
            func=pandas_ta.sma,
            parameters={"lengthx": 21},
        )


def test_indicators_single_backtest_single_thread(strategy_universe, indicator_storage):
    """Parallel creation of indicators using a single run backtest.

    - 2 pairs, 3 indicators each

    - In-thread
    """

    assert strategy_universe.get_pair_count() == 2

    def create_indicators(parameters: StrategyParameters, indicators: IndicatorSet, strategy_universe: TradingStrategyUniverse, execution_context: ExecutionContext):
        indicators.add("rsi", pandas_ta.rsi, {"length": parameters.rsi_length})
        indicators.add("sma_long", pandas_ta.sma, {"length": parameters.sma_long})
        indicators.add("sma_short", pandas_ta.sma, {"length": parameters.sma_short})

    class MyParameters:
        rsi_length=20
        sma_long=200
        sma_short=12

    indicator_result = calculate_and_load_indicators(
        strategy_universe,
        indicator_storage,
        create_indicators=create_indicators,
        execution_context=unit_test_execution_context,
        parameters=StrategyParameters.from_class(MyParameters),
        max_workers=1,
        max_readers=1,
    )

    # 2 pairs, 3 indicators
    assert len(indicator_result) == 2 * 3

    exchange = strategy_universe.data_universe.exchange_universe.get_single()
    weth_usdc = strategy_universe.get_pair_by_human_description((ChainId.ethereum, exchange.exchange_slug, "WETH", "USDC"))
    wbtc_usdc = strategy_universe.get_pair_by_human_description((ChainId.ethereum, exchange.exchange_slug, "WBTC", "USDC"))

    keys = list(indicator_result.keys())
    keys = sorted(keys, key=lambda k: (k.pair.internal_id, k.definition.name))  # Ensure we read set in deterministic order

    # Check our pair x indicator matrix
    assert keys[0].pair== weth_usdc
    assert keys[0].definition.name == "rsi"
    assert keys[0].definition.parameters == {"length": 20}

    assert keys[1].pair== weth_usdc
    assert keys[1].definition.name == "sma_long"
    assert keys[1].definition.parameters == {"length": 200}

    assert keys[3].pair== wbtc_usdc
    assert keys[3].definition.name == "rsi"
    assert keys[3].definition.parameters == {"length": 20}

    for result in indicator_result.values():
        assert not result.cached
        assert isinstance(result.data, pd.Series)
        assert len(result.data) > 0

    # Rerun, now everything should be cached and loaede
    indicator_result = calculate_and_load_indicators(
        strategy_universe,
        indicator_storage,
        create_indicators=create_indicators,
        execution_context=unit_test_execution_context,
        parameters=StrategyParameters.from_class(MyParameters),
        max_workers=1,
        max_readers=1,
    )
    for result in indicator_result.values():
        assert result.cached
        assert isinstance(result.data, pd.Series)
        assert len(result.data) > 0



def test_indicators_single_backtest_multiprocess(strategy_universe, indicator_storage):
    """Parallel creation of indicators using a single run backtest.

    - Using worker pools
    """

    def create_indicators(parameters: StrategyParameters, indicators: IndicatorSet, strategy_universe: TradingStrategyUniverse, execution_context: ExecutionContext):
        indicators.add("rsi", pandas_ta.rsi, {"length": parameters.rsi_length})
        indicators.add("sma_long", pandas_ta.sma, {"length": parameters.sma_long})
        indicators.add("sma_short", pandas_ta.sma, {"length": parameters.sma_short})

    class MyParameters:
        rsi_length=20
        sma_long=200
        sma_short=12

    indicator_result = calculate_and_load_indicators(
        strategy_universe,
        indicator_storage,
        create_indicators=create_indicators,
        execution_context=unit_test_execution_context,
        parameters=StrategyParameters.from_class(MyParameters),
        max_workers=3,
        max_readers=3,
    )

    exchange = strategy_universe.data_universe.exchange_universe.get_single()
    weth_usdc = strategy_universe.get_pair_by_human_description((ChainId.ethereum, exchange.exchange_slug, "WETH", "USDC"))
    wbtc_usdc = strategy_universe.get_pair_by_human_description((ChainId.ethereum, exchange.exchange_slug, "WBTC", "USDC"))

    keys = list(indicator_result.keys())
    keys = sorted(keys, key=lambda k: (k.pair.internal_id, k.definition.name))  # Ensure we read set in deterministic order

    # Check our pair x indicator matrix
    assert keys[0].pair== weth_usdc
    assert keys[0].definition.name == "rsi"
    assert keys[0].definition.parameters == {"length": 20}

    assert keys[1].pair== weth_usdc
    assert keys[1].definition.name == "sma_long"
    assert keys[1].definition.parameters == {"length": 200}

    assert keys[3].pair == wbtc_usdc
    assert keys[3].definition.name == "rsi"
    assert keys[3].definition.parameters == {"length": 20}

    for result in indicator_result.values():
        assert not result.cached
        assert isinstance(result.data, pd.Series)
        assert len(result.data) > 0

    # Rerun, now everything should be cached and loaede
    indicator_result = calculate_and_load_indicators(
        strategy_universe,
        indicator_storage,
        create_indicators=create_indicators,
        execution_context=unit_test_execution_context,
        parameters=StrategyParameters.from_class(MyParameters),
        max_workers=3,
        max_readers=3,
    )
    for result in indicator_result.values():
        assert result.cached
        assert isinstance(result.data, pd.Series)
        assert len(result.data) > 0


def test_complex_indicator(strategy_universe, indicator_storage):
    """Create an indicator with multiple return values.

    - Use bollinger band

    - Deals with multi-column dataframe instead of series
    """

    assert strategy_universe.get_pair_count() == 2

    def create_indicators(parameters: StrategyParameters, indicators: IndicatorSet, strategy_universe: TradingStrategyUniverse, execution_context: ExecutionContext):
        indicators.add("bb", pandas_ta.bbands, {"length": parameters.bb_length})

    class MyParameters:
        bb_length=20

    indicator_result = calculate_and_load_indicators(
        strategy_universe,
        indicator_storage,
        create_indicators=create_indicators,
        execution_context=unit_test_execution_context,
        parameters=StrategyParameters.from_class(MyParameters),
        max_workers=1,
        max_readers=1,
    )

    # 2 pairs, 3 indicators
    assert len(indicator_result) == 2

    exchange = strategy_universe.data_universe.exchange_universe.get_single()
    weth_usdc = strategy_universe.get_pair_by_human_description((ChainId.ethereum, exchange.exchange_slug, "WETH", "USDC"))
    wbtc_usdc = strategy_universe.get_pair_by_human_description((ChainId.ethereum, exchange.exchange_slug, "WBTC", "USDC"))

    keys = list(indicator_result.keys())
    keys = sorted(keys, key=lambda k: (k.pair.internal_id, k.definition.name))  # Ensure we read set in deterministic order

    # Check our pair x indicator matrix
    assert keys[0].pair== weth_usdc
    assert keys[0].definition.name == "bb"
    assert keys[0].definition.parameters == {"length": 20}

    assert keys[1].pair== wbtc_usdc
    assert keys[1].definition.name == "bb"
    assert keys[1].definition.parameters == {"length": 20}

    for result in indicator_result.values():
        assert not result.cached
        assert isinstance(result.data, pd.DataFrame)
        assert len(result.data) > 0
        assert result.data.columns.to_list() == ['BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0']

    # Rerun, now everything should be cached and loaded
    indicator_result = calculate_and_load_indicators(
        strategy_universe,
        indicator_storage,
        create_indicators=create_indicators,
        execution_context=unit_test_execution_context,
        parameters=StrategyParameters.from_class(MyParameters),
        max_workers=1,
        max_readers=1,
    )
    for result in indicator_result.values():
        assert result.cached
        assert isinstance(result.data, pd.DataFrame)
        assert len(result.data) > 0


def test_custom_indicator(strategy_universe, indicator_storage):
    """Create a custom indicator.

    - Create an ETC/BTC price and ETC/BTC RSI indicators
    """

    def calculate_eth_btc(strategy_universe: TradingStrategyUniverse):
        weth_usdc = strategy_universe.get_pair_by_human_description((ChainId.ethereum, "test-dex", "WETH", "USDC"))
        wbtc_usdc = strategy_universe.get_pair_by_human_description((ChainId.ethereum, "test-dex", "WBTC", "USDC"))
        btc_price = strategy_universe.data_universe.candles.get_candles_by_pair(wbtc_usdc.internal_id)
        eth_price = strategy_universe.data_universe.candles.get_candles_by_pair(weth_usdc.internal_id)
        series = eth_price["close"] / btc_price["close"]  # Divide two series
        return series

    def calculate_eth_btc_rsi(strategy_universe: TradingStrategyUniverse, length: int):
        weth_usdc = strategy_universe.get_pair_by_human_description((ChainId.ethereum, "test-dex", "WETH", "USDC"))
        wbtc_usdc = strategy_universe.get_pair_by_human_description((ChainId.ethereum, "test-dex", "WBTC", "USDC"))
        btc_price = strategy_universe.data_universe.candles.get_candles_by_pair(wbtc_usdc.internal_id)
        eth_price = strategy_universe.data_universe.candles.get_candles_by_pair(weth_usdc.internal_id)
        eth_btc = eth_price["close"] / btc_price["close"]
        return pandas_ta.rsi(eth_btc, length=length)

    def create_indicators(parameters: StrategyParameters, indicators: IndicatorSet, strategy_universe: TradingStrategyUniverse, execution_context: ExecutionContext):
        indicators.add("eth_btc", calculate_eth_btc, source=IndicatorSource.strategy_universe)
        indicators.add("eth_btc_rsi", calculate_eth_btc_rsi, parameters={"length": parameters.eth_btc_rsi_length}, source=IndicatorSource.strategy_universe)

    class MyParameters:
        eth_btc_rsi_length = 20

    indicator_result = calculate_and_load_indicators(
        strategy_universe,
        indicator_storage,
        create_indicators=create_indicators,
        execution_context=unit_test_execution_context,
        parameters=StrategyParameters.from_class(MyParameters),
        max_workers=1,
        max_readers=1,
    )

    # 2 pairs, 3 indicators
    assert len(indicator_result) == 2
    for result in indicator_result.values():
        assert not result.cached
        assert isinstance(result.data, pd.Series)
        assert len(result.data) > 0

    # Rerun, now everything should be cached and loaded
    indicator_result = calculate_and_load_indicators(
        strategy_universe,
        indicator_storage,
        create_indicators=create_indicators,
        execution_context=unit_test_execution_context,
        parameters=StrategyParameters.from_class(MyParameters),
        max_workers=1,
        max_readers=1,
    )
    for result in indicator_result.values():
        assert result.cached
        assert isinstance(result.data, pd.Series)
        assert len(result.data) > 0


def test_ohlcv_indicator(strategy_universe, indicator_storage):
    """Create an OHLCV data based indicator.

    - Use Money Flow Index (MFI) using full OHCLV data
    """

    indicators = IndicatorSet()
    indicators.add(
        "mfi",
        pandas_ta.mfi,
        parameters={"length": 4},
        source=IndicatorSource.ohlcv,
    )

    indicator_results = calculate_and_load_indicators(
        strategy_universe,
        indicator_storage,
        indicators=indicators,
        execution_context=unit_test_execution_context,
        parameters=StrategyParameters({}),
        max_workers=1,
        max_readers=1,
    )

    # 2 pairs, 1 indicator
    assert len(indicator_results) == 2
    for result in indicator_results.values():
        assert isinstance(result.data, pd.Series)
        assert result.data.name == "MFI_4"
        assert len(result.data) > 0

    # Test reading MFI value,
    # read on the last day of backtest data for WBTC-USDC pair
    wbtc_usdc = strategy_universe.get_pair_by_human_description((ChainId.ethereum, "test-dex", "WBTC", "USDC"))
    first_day, last_day = strategy_universe.data_universe.candles.get_timestamp_range()
    assert last_day == pd.Timestamp('2021-12-31 00:00:00')

    input_indicators = StrategyInputIndicators(
        strategy_universe=strategy_universe,
        available_indicators=indicators,
        indicator_results=indicator_results,
        timestamp=last_day,
    )

    indicator_value = input_indicators.get_indicator_value("mfi", pair=wbtc_usdc)
    assert indicator_value in (0, None)  # TODO: Local and Github CI disagree what's the proper MFI value here


def test_get_close_price(strategy_universe, indicator_storage):
    """Get the close price."""

    indicators = IndicatorSet()

    indicator_results = calculate_and_load_indicators(
        strategy_universe,
        indicator_storage,
        indicators=indicators,
        execution_context=unit_test_execution_context,
        parameters=StrategyParameters({}),
        max_workers=1,
        max_readers=1,
    )

    first_day, last_day = strategy_universe.data_universe.candles.get_timestamp_range()

    input_indicators = StrategyInputIndicators(
        strategy_universe=strategy_universe,
        available_indicators=indicators,
        indicator_results=indicator_results,
        timestamp=last_day,
    )

    wbtc_usdc = strategy_universe.get_pair_by_human_description((ChainId.ethereum, "test-dex", "WBTC", "USDC"))

    # Check that we correctly read price if there is no prior data
    # Both absolute ts, and relative
    price = input_indicators.get_price(timestamp=first_day - pd.Timedelta(days=1), pair=wbtc_usdc)
    assert price is None

    price = input_indicators.get_price(timestamp=first_day + pd.Timedelta(days=1), pair=wbtc_usdc)
    assert price is not None

    input_indicators.timestamp = first_day
    assert input_indicators.get_price(pair=wbtc_usdc) is None

    input_indicators.timestamp = first_day + pd.Timedelta(days=1)
    assert input_indicators.get_price(pair=wbtc_usdc) is not None
    assert input_indicators.get_price(pair=wbtc_usdc, index=-2) is None
    assert input_indicators.get_price(pair=wbtc_usdc, index=-1) is not None


def test_indicator_single_pair_live_trading_universe(persistent_test_client, indicator_storage):
    """Indicators should always have only timestamp as the index

    - Repeat a production bug

    - Live data loader gives (pair, timestamp) index instead of (timestamp) index
    """

    client = persistent_test_client

    universe_options = UniverseOptions(history_period=datetime.timedelta(days=14))

    pair = (ChainId.polygon, "uniswap-v3", "WMATIC", "USDC", 0.0005)

    dataset = load_partial_data(
        client=client,
        time_bucket=TimeBucket.h1,
        pairs=[pair],
        execution_context=unit_test_trading_execution_context,
        universe_options=universe_options,
        liquidity=False,
    )

    strategy_universe = TradingStrategyUniverse.create_from_dataset(
        dataset,
        reserve_asset="USDC",
        forward_fill=True,
    )

    indicators = IndicatorSet()
    indicators.add(
        "ma",
        pandas_ta.sma,
        parameters={"length": 4},
    )

    indicator_results = calculate_and_load_indicators(
        strategy_universe,
        indicator_storage,
        indicators=indicators,
        execution_context=unit_test_execution_context,
        parameters=StrategyParameters({}),
        max_workers=1,
        max_readers=1,
    )

    first_day, last_day = strategy_universe.data_universe.candles.get_timestamp_range()

    input_indicators = StrategyInputIndicators(
        strategy_universe=strategy_universe,
        available_indicators=indicators,
        indicator_results=indicator_results,
        timestamp=last_day,
    )

    indicator_series = input_indicators.get_indicator_series("ma")
    assert isinstance(indicator_series.index, pd.DatetimeIndex)
    indicator_value = input_indicators.get_indicator_value("ma")
    assert 0 < indicator_value < 10_000


def test_indicator_dependency_resolution(persistent_test_client, indicator_storage):
    """Indicators can depend each other and consume data of indicators calculated earlier.

    - Create an example where we have one indicator (crossover) depending on two other indicators (smas)
    - Live data loader gives (pair, timestamp) index instead of (timestamp) index
    - We can control dependency order resolution using :py:attr:`~tradingstrategy.strategy.pandas_trader.indicator.IndicatorKey.order` parameter
    - Dependency resolved indicators must use :py:attr:`~tradingstrategy.strategy.pandas_trader.indicator.IndicatorSource.order` as indicator osurce
    """

    client = persistent_test_client

    universe_options = UniverseOptions(history_period=datetime.timedelta(days=14))

    pair = (ChainId.polygon, "uniswap-v3", "WMATIC", "USDC", 0.0005)

    dataset = load_partial_data(
        client=client,
        time_bucket=TimeBucket.h1,
        pairs=[pair],
        execution_context=unit_test_trading_execution_context,
        universe_options=universe_options,
        liquidity=False,
    )

    strategy_universe = TradingStrategyUniverse.create_from_dataset(
        dataset,
        reserve_asset="USDC",
        forward_fill=True,
    )

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

    def ma_crossover_by_pair(
        close: pd.Series,
        pair: TradingPairIdentifier,
        dependency_resolver: IndicatorDependencyResolver,
    ) -> pd.Series:
        slow_sma: pd.Series = dependency_resolver.get_indicator_data("slow_sma", pair=pair)
        fast_sma: pd.Series = dependency_resolver.get_indicator_data("fast_sma", pair=pair)
        return fast_sma > slow_sma

    def ma_crossover_by_pair_and_parameters(
        close: pd.Series,
        pair: TradingPairIdentifier,
        dependency_resolver: IndicatorDependencyResolver,
    ) -> pd.Series:
        slow_sma: pd.Series = dependency_resolver.get_indicator_data("slow_sma", pair=pair, parameters={"length": 21})
        fast_sma: pd.Series = dependency_resolver.get_indicator_data("fast_sma", pair=pair, parameters={"length": 7})
        return fast_sma > slow_sma

    indicators = IndicatorSet()
    # Slow moving average
    indicators.add(
        "fast_sma",
        pandas_ta.sma,
        parameters={"length": 7},
        order=1,
    )
    # Fast moving average
    indicators.add(
        "slow_sma",
        pandas_ta.sma,
        parameters={"length": 21},
        order=1,
    )
    # An indicator that depends on both fast MA and slow MA above
    indicators.add(
        "ma_crossover",
        ma_crossover,
        source=IndicatorSource.ohlcv,
        order=2,  # 2nd order indicators can depend on the data of 1st order indicators
    )
    indicators.add(
        "ma_crossover_by_pair",
        ma_crossover_by_pair,
        source=IndicatorSource.ohlcv,
        order=2,  # 2nd order indicators can depend on the data of 1st order indicators
    )
    indicators.add(
        "ma_crossover_by_pair_and_parameters",
        ma_crossover_by_pair_and_parameters,
        source=IndicatorSource.ohlcv,
        order=2,  # 2nd order indicators can depend on the data of 1st order indicators
    )
    indicator_results = calculate_and_load_indicators(
        strategy_universe,
        indicator_storage,
        indicators=indicators,
        execution_context=unit_test_execution_context,
        parameters=StrategyParameters({}),
        max_workers=2,
        max_readers=1,
    )

    first_day, last_day = strategy_universe.data_universe.candles.get_timestamp_range()

    input_indicators = StrategyInputIndicators(
        strategy_universe=strategy_universe,
        available_indicators=indicators,
        indicator_results=indicator_results,
        timestamp=last_day,
    )

    indicator_series = input_indicators.get_indicator_series("ma_crossover")
    assert isinstance(indicator_series.index, pd.DatetimeIndex)
    indicator_value = input_indicators.get_indicator_value("ma_crossover")
    assert indicator_value in (True, False)


def test_indicator_dependency_order_error(persistent_test_client, indicator_storage):
    """Try to calculate data in wrong dependency oredr.
    """

    client = persistent_test_client

    universe_options = UniverseOptions(history_period=datetime.timedelta(days=14))

    pair = (ChainId.polygon, "uniswap-v3", "WMATIC", "USDC", 0.0005)

    dataset = load_partial_data(
        client=client,
        time_bucket=TimeBucket.h1,
        pairs=[pair],
        execution_context=unit_test_trading_execution_context,
        universe_options=universe_options,
        liquidity=False,
    )

    strategy_universe = TradingStrategyUniverse.create_from_dataset(
        dataset,
        reserve_asset="USDC",
        forward_fill=True,
    )

    def ma_crossover(
        close: pd.Series,
        pair: TradingPairIdentifier,
        dependency_resolver: IndicatorDependencyResolver,
    ) -> pd.Series:
        # Fails because order parameter not set in indiactors.add()
        slow_sma: pd.Series = dependency_resolver.get_indicator_data("slow_sma")
        return slow_sma

    indicators = IndicatorSet()
    indicators.add(
        "slow_sma",
        pandas_ta.sma,
        parameters={"length": 21},
    )
    # order parameter is not set
    indicators.add(
        "ma_crossover",
        ma_crossover,
        source=IndicatorSource.ohlcv,
    )

    with pytest.raises(IndicatorCalculationFailed):
        # TODO: match the nested IndicatorOrderError here
        calculate_and_load_indicators(
            strategy_universe,
            indicator_storage,
            indicators=indicators,
            execution_context=unit_test_execution_context,
            parameters=StrategyParameters({}),
            max_workers=1,
            max_readers=1,
        )


def test_indicator_dependency_memory_storage(strategy_universe, mocker):
    """Test dependency resolution with memory storage."""

    def regime(
        close: pd.Series,
        pair: TradingPairIdentifier,
        length: int,
        dependency_resolver: IndicatorDependencyResolver,
    ) -> pd.Series:
        fast_sma: pd.Series = dependency_resolver.get_indicator_data("fast_sma", pair=pair, parameters={"length": length})
        return close > fast_sma

    def create_indicators(parameters: StrategyParameters, indicators: IndicatorSet, strategy_universe: TradingStrategyUniverse, execution_context: ExecutionContext):
        indicators.add("fast_sma", pandas_ta.sma, {"length": parameters.fast_sma}, order=1)
        indicators.add("regime", regime, {"length": parameters.fast_sma}, order=2)

    class MyParameters:
        fast_sma = 20

    indicator_storage = MemoryIndicatorStorage(strategy_universe.get_cache_key())

    indicator_result = calculate_and_load_indicators(
        strategy_universe,
        indicator_storage,
        create_indicators=create_indicators,
        execution_context=unit_test_execution_context,
        parameters=StrategyParameters.from_class(MyParameters),
    )

    exchange = strategy_universe.data_universe.exchange_universe.get_single()
    weth_usdc = strategy_universe.get_pair_by_human_description((ChainId.ethereum, exchange.exchange_slug, "WETH", "USDC"))
    wbtc_usdc = strategy_universe.get_pair_by_human_description((ChainId.ethereum, exchange.exchange_slug, "WBTC", "USDC"))

    keys = list(indicator_result.keys())
    keys = sorted(keys, key=lambda k: (k.pair.internal_id, k.definition.name))  # Ensure we read set in deterministic order

    # Check our pair x indicator matrix
    assert keys[0].pair== weth_usdc
    assert keys[0].definition.name == "fast_sma"
    assert keys[0].definition.parameters == {"length": 20}

    assert keys[1].pair== weth_usdc
    assert keys[1].definition.name == "regime"
    assert keys[1].definition.parameters == {"length": 20}

    assert keys[3].pair == wbtc_usdc
    assert keys[3].definition.name == "regime"

    for result in indicator_result.values():
        assert not result.cached
        assert isinstance(result.data, pd.Series)
        assert len(result.data) > 0

    # Run with higher workers count should still work since it should force single thread
    logger_mock = mocker.patch("tradeexecutor.strategy.pandas_trader.indicator.logger.warning")
    indicator_result = calculate_and_load_indicators(
        strategy_universe,
        indicator_storage,
        create_indicators=create_indicators,
        execution_context=unit_test_execution_context,
        parameters=StrategyParameters.from_class(MyParameters),
        max_workers=3,
        max_readers=3,
    )

    assert logger_mock.call_count == 1
    assert logger_mock.call_args[0][0] == "MemoryIndicatorStorage does not support multiprocessing, setting max_workers and max_readers to 1"


def test_calculate_indicators_inline(strategy_universe):
    """Test calculate_and_load_indicators_inline() """

    # Some example parameters we use to calculate indicators.
    # E.g. RSI length 
    class Parameters:
        rsi_length = 20
        sma_long = 200
        sma_short = 12

    # Create indicators.
    # Map technical indicator functions to their parameters, like length.
    # You can also use hardcoded values, but we recommend passing in parameter dict,
    # as this allows later to reuse the code for optimising/grid searches, etc.
    def create_indicators(
        timestamp,
        parameters,
        strategy_universe,
        execution_context,
    ) -> IndicatorSet:
        indicator_set = IndicatorSet()
        indicator_set.add("rsi", pandas_ta.rsi, {"length": parameters.rsi_length})
        indicator_set.add("sma_long", pandas_ta.sma, {"length": parameters.sma_long})
        indicator_set.add("sma_short", pandas_ta.sma, {"length": parameters.sma_short})
        return indicator_set

    # Calculate indicators - will spawn multiple worker processed,
    # or load cached results from the disk
    indicators = calculate_and_load_indicators_inline(
        strategy_universe=strategy_universe,
        parameters=StrategyParameters.from_class(Parameters),
        create_indicators=create_indicators,
    )

    # From calculated indicators, read one indicator (RSI for BTC)
    wbtc_usdc = strategy_universe.get_pair_by_human_description((ChainId.ethereum, "test-dex", "WBTC", "USDC"))
    rsi = indicators.get_indicator_series("rsi", pair=wbtc_usdc)
    assert len(rsi) == 214  # We have series data for 214 days





