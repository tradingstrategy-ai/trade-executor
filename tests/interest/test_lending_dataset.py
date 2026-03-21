"""Load lending datasets."""

import datetime

import pandas as pd
import pytest

from tradeexecutor.analysis.universe import analyse_long_short_universe
from tradeexecutor.state.identifier import TradingPairKind
from tradeexecutor.strategy.execution_context import unit_test_execution_context, unit_test_trading_execution_context
from tradeexecutor.strategy.trading_strategy_universe import load_partial_data, TradingStrategyUniverse, load_trading_and_lending_data, translate_trading_pair
from tradeexecutor.strategy.universe_model import default_universe_options, UniverseOptions
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.lending import LendingProtocolType, UnknownLendingReserve
from tradingstrategy.timebucket import TimeBucket
from eth_defi.compat import native_datetime_utc_now


def test_load_lending_dataset(persistent_test_client: Client):
    """Load trading pair and lending data for the same backtest"""
    client = persistent_test_client

    pairs = [
        (ChainId.polygon, "uniswap-v3", "WETH", "USDC", 0.0005)
    ]

    reverses = [
        (ChainId.polygon, LendingProtocolType.aave_v3, "USDC.e")
    ]

    dataset = load_partial_data(
        client,
        execution_context=unit_test_execution_context,
        time_bucket=TimeBucket.d1,
        pairs=pairs,
        universe_options=default_universe_options,
        start_at=pd.Timestamp("2023-01-01"),
        end_at=pd.Timestamp("2023-02-01"),
        lending_reserves=reverses,
    )

    # Trading pairs ok
    assert len(dataset.pairs) == 1

    # Lending reserves ok
    assert dataset.lending_reserves.get_count() == 1
    rates = dataset.lending_candles.supply_apr.get_rates_by_reserve(
        (ChainId.polygon, LendingProtocolType.aave_v3, "USDC.e")
    )
    assert rates["open"][pd.Timestamp("2023-01-01")] == pytest.approx(0.6998308843795215)
    assert rates["close"][pd.Timestamp("2023-01-01")] == pytest.approx(0.7001845728826266)    


def test_construct_trading_universe_with_lending(persistent_test_client: Client):
    """Create a trading universe that allows lending."""
    client = persistent_test_client

    pairs = [
        (ChainId.polygon, "uniswap-v3", "WETH", "USDC", 0.0005)
    ]

    reverses = [
        (ChainId.polygon, LendingProtocolType.aave_v3, "USDC.e")
    ]

    dataset = load_partial_data(
        client,
        execution_context=unit_test_execution_context,
        time_bucket=TimeBucket.d7,
        pairs=pairs,
        universe_options=default_universe_options,
        start_at=pd.Timestamp("2023-01-01"),
        end_at=pd.Timestamp("2023-02-01"),
        lending_reserves=reverses,
    )

    # Convert loaded data to a trading pair universe
    strategy_universe = TradingStrategyUniverse.create_single_pair_universe(dataset)

    data_universe = strategy_universe.data_universe
    assert data_universe.chains == {ChainId.polygon}

    # Lending reserves ok
    assert data_universe.lending_reserves.get_count() == 1
    rates = data_universe.lending_candles.supply_apr.get_rates_by_reserve(
        (ChainId.polygon, LendingProtocolType.aave_v3, "USDC.e")
    )
    assert rates["open"][pd.Timestamp("2023-01-01")] == pytest.approx(0.6998308843795215)
    assert rates["close"][pd.Timestamp("2023-01-01")] == pytest.approx(0.7001845728826266)

    # check lending candles are in 1h bucket regardless of the trading pair bucket
    supply_df = strategy_universe.data_universe.lending_candles.supply_apr.df.tail(2)
    assert supply_df.iloc[0].timestamp == pd.Timestamp("2023-01-31 23:00:00")
    assert supply_df.iloc[1].timestamp == pd.Timestamp("2023-02-01 00:00:00")


def test_get_credit_supply_trading_pair(persistent_test_client: Client):
    """Resolve a TradingPair for credit supply for the trading universe reserve currency.

    - How to get the supply APR we can earn for the strategy reserve currency
    """

    client = persistent_test_client

    pairs = [
        (ChainId.polygon, "uniswap-v3", "WETH", "USDC", 0.0005)
    ]

    reverses = [
        (ChainId.polygon, LendingProtocolType.aave_v3, "USDC.e")
    ]

    dataset = load_partial_data(
        client,
        execution_context=unit_test_execution_context,
        time_bucket=TimeBucket.d1,
        pairs=pairs,
        universe_options=default_universe_options,
        start_at=pd.Timestamp("2023-01-01"),
        end_at=pd.Timestamp("2023-02-01"),
        lending_reserves=reverses,
    )

    strategy_universe = TradingStrategyUniverse.create_single_pair_universe(dataset)

    # This trading pair identifies the trades where
    reserve_credit_supply_pair = strategy_universe.get_credit_supply_pair()
    assert reserve_credit_supply_pair.base.token_symbol == "aPolUSDC"
    assert reserve_credit_supply_pair.quote.token_symbol == "USDC.e"
    assert reserve_credit_supply_pair.kind == TradingPairKind.credit_supply

    data_universe = strategy_universe.data_universe
    rate_candles = data_universe.lending_candles.supply_apr.get_rates_by_id(reserve_credit_supply_pair.internal_id)
    assert rate_candles["open"][pd.Timestamp("2023-01-01")] == pytest.approx(0.6998308843795215)

    rate_candles = data_universe.lending_candles.variable_borrow_apr.get_rates_by_id(reserve_credit_supply_pair.internal_id)
    assert rate_candles["open"][pd.Timestamp("2023-01-01")] == pytest.approx(1.836242)


def test_load_trading_and_lending_data_historical(persistent_test_client: Client):
    """Load historical lending market data."""

    client = persistent_test_client
    start_at = datetime.datetime(2023, 9, 1)
    end_at = datetime.datetime(2023, 10, 1)

    # Load all trading and lending data on Polygon
    # for all lending markets on a relevant time period
    dataset = load_trading_and_lending_data(
        client,
        execution_context=unit_test_execution_context,
        universe_options=UniverseOptions(start_at=start_at, end_at=end_at),
        chain_id=ChainId.polygon,
        exchange_slugs="uniswap-v3",
        reserve_assets={"0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"},
    )

    strategy_universe = TradingStrategyUniverse.create_from_dataset(dataset)
    data_universe = strategy_universe.data_universe

    assert 1 < data_universe.pairs.get_count() < 25  # Number of supported trading pairs for Polygon lending pools

    assert data_universe.lending_candles is not None
    assert data_universe.lending_reserves is not None

    # Check one loaded reserve metadata
    usdc_reserve = data_universe.lending_reserves.get_by_chain_and_symbol(ChainId.polygon, "USDC.e")
    assert usdc_reserve.atoken_symbol == "aPolUSDC"
    assert usdc_reserve.vtoken_symbol == "variableDebtPolUSDC"

    # Check the historical rates
    lending_candles = data_universe.lending_candles.variable_borrow_apr
    rates = lending_candles.get_rates_by_reserve(usdc_reserve)

    assert rates["open"][pd.Timestamp("2023-09-01")] == pytest.approx(3.222019)
    assert rates["open"][pd.Timestamp("2023-10-01")] == pytest.approx(3.446714)


def test_load_trading_and_lending_data_historical_certain_assets_only(persistent_test_client: Client):
    """Load historical lending market data for certain tokens."""

    client = persistent_test_client
    start_at = datetime.datetime(2023, 9, 1)
    end_at = datetime.datetime(2023, 10, 1)

    # Load all trading and lending data on Polygon
    # for all lending markets on a relevant time period
    dataset = load_trading_and_lending_data(
        client,
        execution_context=unit_test_execution_context,
        universe_options=UniverseOptions(start_at=start_at, end_at=end_at),
        chain_id=ChainId.polygon,
        exchange_slugs="uniswap-v3",
        asset_ids={"LINK", "WETH"},
        trading_fee=0.0005,
        reserve_assets={"0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"}
    )

    strategy_universe = TradingStrategyUniverse.create_from_dataset(dataset)
    data_universe = strategy_universe.data_universe

    usdc_reserve = data_universe.lending_reserves.get_by_chain_and_symbol(ChainId.polygon, "USDC.e")
    assert usdc_reserve.atoken_symbol == "aPolUSDC"
    assert usdc_reserve.vtoken_symbol == "variableDebtPolUSDC"

    lending_reserves = data_universe.lending_reserves
    assert lending_reserves.get_count() == 6
    assert lending_reserves.get_by_chain_and_symbol(ChainId.polygon, "LINK") is not None
    assert lending_reserves.get_by_chain_and_symbol(ChainId.polygon, "WETH") is not None

    with pytest.raises(UnknownLendingReserve):
        lending_reserves.get_by_chain_and_symbol(ChainId.polygon, "WMATIC")

    eth_reserve = data_universe.lending_reserves.get_by_chain_and_symbol(ChainId.polygon, "WETH")

    # Check the historical rates
    lending_candles = data_universe.lending_candles.variable_borrow_apr
    rates = lending_candles.get_rates_by_reserve(eth_reserve)

    assert rates["open"][pd.Timestamp("2023-09-01")] == pytest.approx(2.3803235973323122)

    link_usdc = data_universe.pairs.get_pair_by_human_description((ChainId.polygon, None, "LINK", "USDC"))
    assert link_usdc.fee_tier == 0.0005

    # test again with protocol filtered
    dataset = load_trading_and_lending_data(
        client,
        execution_context=unit_test_execution_context,
        universe_options=UniverseOptions(start_at=start_at, end_at=end_at),
        chain_id=ChainId.polygon,
        exchange_slugs="uniswap-v3",
        lending_protocol=LendingProtocolType.aave_v3,
        asset_ids={"LINK", "WETH"},
        trading_fee=0.0005,
        reserve_assets={"0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"}
    )
    strategy_universe = TradingStrategyUniverse.create_from_dataset(dataset)
    data_universe = strategy_universe.data_universe

    assert data_universe.lending_reserves.get_count() == 3


def test_load_trading_and_lending_data_live(persistent_test_client: Client):
    """Load lending market data today."""

    client = persistent_test_client

    # Load all trading and lending data on Polygon
    # for all lending markets on a relevant time period
    dataset = load_trading_and_lending_data(
        client,
        execution_context=unit_test_trading_execution_context,
        universe_options=UniverseOptions(history_period=datetime.timedelta(days=7)),
        chain_id=ChainId.polygon,
        exchange_slugs="uniswap-v3",
        reserve_assets={"0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"},
    )

    assert dataset.history_period == datetime.timedelta(days=7)

    strategy_universe = TradingStrategyUniverse.create_from_dataset(dataset)
    data_universe = strategy_universe.data_universe

    # Check one loaded reserve metadata
    usdc_reserve = data_universe.lending_reserves.get_by_chain_and_symbol(ChainId.polygon, "USDC")

    # Check the historical rates
    lending_candles = data_universe.lending_candles.variable_borrow_apr
    rates = lending_candles.get_rates_by_reserve(usdc_reserve)

    # Check that we did not load too old lending data
    first_rate_sample = rates.index[0]
    assert first_rate_sample.to_pydatetime() > native_datetime_utc_now() - datetime.timedelta(days=8)

    # Check that we did not load too old price data
    desc = (ChainId.polygon, "uniswap-v3", "WETH", "USDC", 0.0005)
    pair = data_universe.pairs.get_pair_by_human_description(desc)
    price_feed = data_universe.candles.get_candles_by_pair(pair.pair_id)
    first_price_sample = price_feed.index[0]
    assert first_price_sample.to_pydatetime() > native_datetime_utc_now() - datetime.timedelta(days=8)

    # Check a single data sample looks correct
    # Lending rate data
    two_days_ago = pd.Timestamp(native_datetime_utc_now() - datetime.timedelta(days=2)).floor("D")
    open = rates["open"][two_days_ago]
    close = rates["close"][two_days_ago]
    assert rates["open"][two_days_ago] > 0
    assert rates["open"][two_days_ago] < 50, f"Got: open:{open}%, close:{close}%, on {usdc_reserve} at {two_days_ago}"

    # Check a single data sample looks correct
    # Price feed
    assert price_feed["open"][two_days_ago] > 0
    assert price_feed["open"][two_days_ago] < 10_000  # To the moon warning


@pytest.fixture(scope="module")
def polygon_lending_dataset(persistent_test_client: Client):
    """Shared lending dataset for long/short universe tests.

    - Loads QuickSwap + Aave lending data on Polygon
    - Reused across test_can_open_short and test_analyse_long_short_universe
      to avoid duplicate API calls
    - Date range covers MaticX enablement (2023-03-07) with minimal buffer
    """
    return load_trading_and_lending_data(
        persistent_test_client,
        execution_context=unit_test_execution_context,
        universe_options=UniverseOptions(
            start_at=datetime.datetime(2023, 1, 1),
            end_at=datetime.datetime(2023, 5, 1),
        ),
        chain_id=ChainId.polygon,
        exchange_slugs="quickswap",
        time_bucket=TimeBucket.d7,
        any_quote=True,
    )


def test_can_open_short(polygon_lending_dataset):
    """Check if we correctly detect when we have lending market data available.

    1. Load shared Polygon lending dataset
    2. Create strategy universe with USDC reserve
    3. Check MaticX short availability at various timestamps
    4. Verify availability matches MaticX enablement date (2023-03-07)

    - Aave v3 on Polygon enabled MaticX 2023-3-7 https://tradingstrategy.ai/trading-view/polygon/lending/aave_v3/maticx
    - Aave v3 on Polygon enabled USDC 2022-3-16
    """

    # 1. Load shared Polygon lending dataset
    dataset = polygon_lending_dataset

    # https://tradingstrategy.ai/trading-view/polygon/tokens/0x2791bca1f2de4661ed88a30c99a7a9449aa84174
    usdc_address = "0x2791bca1f2de4661ed88a30c99a7a9449aa84174"

    # 2. Create strategy universe with USDC reserve
    strategy_universe = TradingStrategyUniverse.create_from_dataset(
        dataset,
        reserve_asset=usdc_address,
    )

    data_universe = strategy_universe.data_universe

    # https://tradingstrategy.ai/trading-view/polygon/quickswap/maticx-matic#7d
    # Internal id 2648052
    desc = (ChainId.polygon, "quickswap", "MaticX", "WMATIC")
    pair = translate_trading_pair(data_universe.pairs.get_pair_by_human_description(desc))

    # 3. Check MaticX short availability at various timestamps
    # Does not exist
    assert not strategy_universe.can_open_short(
        pd.Timestamp("2000-1-1"),
        pair,
    )

    # MaticX reserve not available, MATIC reserve not available
    assert not strategy_universe.can_open_short(
        pd.Timestamp("2023-02-01"),
        pair,
    )

    # MaticX reserve not available, MATIC reserve available
    assert not strategy_universe.can_open_short(
        pd.Timestamp("2023-02-01"),
        pair,
    )

    # 4. Verify availability matches MaticX enablement date
    # Both reserves available
    assert strategy_universe.can_open_short(
        pd.Timestamp("2023-04-01"),
        pair,
    )

    # Does not exist
    assert not strategy_universe.can_open_short(
        pd.Timestamp("2099-1-1"),
        pair,
    )


def test_analyse_long_short_universe(polygon_lending_dataset):
    """Check analyse_long_short_universe() does not crash.

    1. Load shared Polygon lending dataset
    2. Create strategy universe with USDC reserve
    3. Run analyse_long_short_universe and verify non-empty output
    """

    # 1. Load shared Polygon lending dataset
    dataset = polygon_lending_dataset

    # https://tradingstrategy.ai/trading-view/polygon/tokens/0x2791bca1f2de4661ed88a30c99a7a9449aa84174
    usdc_address = "0x2791bca1f2de4661ed88a30c99a7a9449aa84174"

    # 2. Create strategy universe with USDC reserve
    strategy_universe = TradingStrategyUniverse.create_from_dataset(
        dataset,
        reserve_asset=usdc_address,
    )

    # 3. Run analyse_long_short_universe and verify non-empty output
    df = analyse_long_short_universe(
        strategy_universe,
    )

    assert len(df) > 0

