"""Load lending datasets."""
import pandas as pd
import pytest

from tradeexecutor.state.identifier import TradingPairKind
from tradeexecutor.strategy.execution_context import ExecutionContext, unit_test_execution_context
from tradeexecutor.strategy.trading_strategy_universe import load_partial_data, TradingStrategyUniverse
from tradeexecutor.strategy.universe_model import default_universe_options
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
    assert dataset.lending_reserves.get_size() == 1
    rates = dataset.lending_candles.supply_apr.get_rates_by_reserve(
        (ChainId.polygon, LendingProtocolType.aave_v3, "USDC")
    )
    assert rates["open"][pd.Timestamp("2023-01-01")] == pytest.approx(1.836242)
    assert rates["close"][pd.Timestamp("2023-01-01")] == pytest.approx(1.780513)


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

    data_universe = strategy_universe.universe
    assert data_universe.chains == {ChainId.polygon}

    # Lending reserves ok
    assert data_universe.lending_reserves.get_size() == 1
    rates = data_universe.lending_candles.supply_apr.get_rates_by_reserve(
        (ChainId.polygon, LendingProtocolType.aave_v3, "USDC")
    )
    assert rates["open"][pd.Timestamp("2023-01-01")] == pytest.approx(1.836242)
    assert rates["close"][pd.Timestamp("2023-01-01")] == pytest.approx(1.780513)


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

    data_universe = strategy_universe.universe
    rate_candles = data_universe.lending_candles.supply_apr.get_rates_by_id(reserve_credit_supply_pair.internal_id)
    assert rate_candles["open"][pd.Timestamp("2023-01-01")] == pytest.approx(1.836242)
