"""Load lending datasets."""

import datetime

import pandas as pd
import pytest

from tradeexecutor.state.identifier import TradingPairKind
from tradeexecutor.strategy.execution_context import unit_test_execution_context
from tradeexecutor.strategy.trading_strategy_universe import load_partial_data, TradingStrategyUniverse, load_trading_and_lending_data
from tradeexecutor.strategy.universe_model import default_universe_options, UniverseOptions
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.lending import LendingProtocolType
from tradingstrategy.timebucket import TimeBucket


def test_load_lending_dataset(persistent_test_client: Client):
    """Load trading pair and lending data for the same backtest"""
    client = persistent_test_client

    pairs = [
        (ChainId.polygon, "uniswap-v3", "WETH", "USDC", 0.0005)
    ]

    reverses = [
        (ChainId.polygon, LendingProtocolType.aave_v3, "USDC")
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
        (ChainId.polygon, LendingProtocolType.aave_v3, "USDC")
    )
    assert rates["open"][pd.Timestamp("2023-01-01")] == pytest.approx(0.6998308843795215)
    assert rates["close"][pd.Timestamp("2023-01-01")] == pytest.approx(0.6589076823528996)


def test_construct_trading_universe_with_lending(persistent_test_client: Client):
    """Create a trading universe that allows lending."""
    client = persistent_test_client

    pairs = [
        (ChainId.polygon, "uniswap-v3", "WETH", "USDC", 0.0005)
    ]

    reverses = [
        (ChainId.polygon, LendingProtocolType.aave_v3, "USDC")
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

    # Convert loaded data to a trading pair universe
    strategy_universe = TradingStrategyUniverse.create_single_pair_universe(dataset)

    data_universe = strategy_universe.data_universe
    assert data_universe.chains == {ChainId.polygon}

    # Lending reserves ok
    assert data_universe.lending_reserves.get_count() == 1
    rates = data_universe.lending_candles.supply_apr.get_rates_by_reserve(
        (ChainId.polygon, LendingProtocolType.aave_v3, "USDC")
    )
    assert rates["open"][pd.Timestamp("2023-01-01")] == pytest.approx(0.6998308843795215)
    assert rates["close"][pd.Timestamp("2023-01-01")] == pytest.approx(0.6589076823528996)


def test_get_credit_supply_trading_pair(persistent_test_client: Client):
    """Resolve a TradingPair for credit supply for the trading universe reserve currency.

    - How to get the supply APR we can earn for the strategy reserve currency
    """

    client = persistent_test_client

    pairs = [
        (ChainId.polygon, "uniswap-v3", "WETH", "USDC", 0.0005)
    ]

    reverses = [
        (ChainId.polygon, LendingProtocolType.aave_v3, "USDC")
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
    assert reserve_credit_supply_pair.quote.token_symbol == "USDC"
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
    )

    strategy_universe = TradingStrategyUniverse.create_from_dataset(dataset)
    data_universe = strategy_universe.data_universe

    assert 1 < data_universe.pairs.get_count() < 25  # Number of supported trading pairs for Polygon lending pools

    assert data_universe.lending_candles is not None
    assert data_universe.lending_reserves is not None

    # Check one loaded reserve metadata
    usdc_reserve = data_universe.lending_reserves.get_by_chain_and_symbol(ChainId.polygon, "USDC")
    assert usdc_reserve.atoken_symbol == "aPolUSDC"
    assert usdc_reserve.vtoken_symbol == "variableDebtPolUSDC"

    # Check the historical rates
    lending_candles = data_universe.lending_candles.variable_borrow_apr
    rates = lending_candles.get_rates_by_reserve(usdc_reserve)

    assert rates["open"][pd.Timestamp("2023-09-01")] == pytest.approx(3.222019)
    assert rates["open"][pd.Timestamp("2023-10-01")] == pytest.approx(3.446714)


def test_load_trading_and_lending_data_live(persistent_test_client: Client):
    """Load lending market data today."""

    client = persistent_test_client

    # Load all trading and lending data on Polygon
    # for all lending markets on a relevant time period
    dataset = load_trading_and_lending_data(
        client,
        execution_context=unit_test_execution_context,
        universe_options=UniverseOptions(history_period=datetime.timedelta(days=7)),
        chain_id=ChainId.polygon,
        exchange_slugs="uniswap-v3",
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
    assert first_rate_sample.to_pydatetime() > datetime.datetime.utcnow() - datetime.timedelta(days=8)

    # Check that we did not load too old price data
    trading_pair = (ChainId.polygon, "uniswap-v3", "WETH", "USDC", 0.0005)
    pair = data_universe.pairs.get_pair_by_human_description(trading_pair)
    price_feed = data_universe.candles.get_candles_by_pair(pair.pair_id)
    first_price_sample = price_feed.index[0]
    assert first_price_sample.to_pydatetime() > datetime.datetime.utcnow() - datetime.timedelta(days=8)

    # Check a single data sample looks correct
    # Lending rate data
    two_days_ago = pd.Timestamp(datetime.datetime.utcnow() - datetime.timedelta(days=2)).floor("D")
    assert rates["open"][two_days_ago] > 0
    assert rates["open"][two_days_ago] < 10  # Erdogan warnings

    # Check a single data sample looks correct
    # Price feed
    assert price_feed["open"][two_days_ago] > 0
    assert price_feed["open"][two_days_ago] < 10_000  # To the moon warning
