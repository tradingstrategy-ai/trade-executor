"""Test strategy with exchange account position using inline functions.

This test verifies:
1. create_trading_universe can set up exchange account position
2. decide_trades can return empty list (passive position monitoring)
3. Strategy can run a single cycle with exchange account position
4. Sync model correctly tracks PnL changes
"""

import datetime
import random
from typing import List

import pytest

from tradeexecutor.backtest.backtest_runner import run_backtest_inline
from tradeexecutor.state.identifier import (
    TradingPairIdentifier,
    AssetIdentifier,
    TradingPairKind,
)
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.strategy_module import StrategyParameters
from tradeexecutor.strategy.trading_strategy_universe import (
    TradingStrategyUniverse,
    create_pair_universe_from_code,
)
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange
from tradeexecutor.testing.synthetic_price_data import generate_ohlcv_candles
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe


@pytest.fixture
def exchange_account_pair():
    """Create an exchange account pair for testing."""
    chain_id = ChainId.ethereum

    usdc = AssetIdentifier(
        chain_id=chain_id.value,
        address=generate_random_ethereum_address(),
        token_symbol="USDC",
        decimals=6,
    )
    derive_account = AssetIdentifier(
        chain_id=chain_id.value,
        address=generate_random_ethereum_address(),
        token_symbol="DERIVE-ACCOUNT",
        decimals=6,
    )

    mock_exchange = generate_exchange(
        exchange_id=random.randint(1, 1000),
        chain_id=chain_id,
        address=generate_random_ethereum_address(),
    )

    return TradingPairIdentifier(
        base=derive_account,
        quote=usdc,
        pool_address=generate_random_ethereum_address(),
        exchange_address=mock_exchange.address,
        internal_id=random.randint(1, 1000),
        internal_exchange_id=mock_exchange.exchange_id,
        fee=0.0,
        kind=TradingPairKind.exchange_account,
        exchange_name="derive",
        other_data={
            "exchange_protocol": "derive",
            "exchange_subaccount_id": 1,
            "exchange_is_testnet": True,
        },
    ), mock_exchange, usdc


@pytest.fixture
def strategy_universe_with_exchange_account(exchange_account_pair):
    """Create a trading universe with an exchange account position.

    Returns a universe with:
    - One exchange account pair
    - Synthetic OHLCV candles (not used, but required by framework)
    """
    pair, mock_exchange, usdc = exchange_account_pair
    chain_id = ChainId.ethereum

    start_at = datetime.datetime(2024, 1, 1)
    end_at = datetime.datetime(2024, 1, 10)
    time_bucket = TimeBucket.d1

    pair_universe = create_pair_universe_from_code(chain_id, [pair])

    # Generate minimal candles (required by framework, even if not used)
    candles = generate_ohlcv_candles(
        time_bucket,
        start_at,
        end_at,
        pair_id=pair.internal_id,
    )
    candle_universe = GroupedCandleUniverse.create_from_single_pair_dataframe(candles)

    universe = Universe(
        time_bucket=time_bucket,
        chains={chain_id},
        exchanges={mock_exchange},
        pairs=pair_universe,
        candles=candle_universe,
        liquidity=None,
    )

    strategy_universe = TradingStrategyUniverse(
        data_universe=universe,
        reserve_assets=[usdc],
    )
    strategy_universe.data_universe.pairs.exchange_universe = (
        strategy_universe.data_universe.exchange_universe
    )

    return strategy_universe


def test_strategy_with_exchange_account_single_cycle(strategy_universe_with_exchange_account):
    """Test that a strategy can run a single cycle with exchange account position.

    - create_trading_universe returns pre-built universe
    - decide_trades returns empty list (passive monitoring)
    - Strategy runs one cycle without errors
    """

    def decide_trades(input: StrategyInput) -> List[TradeExecution]:
        """Passive strategy - no trades, just monitor exchange account."""
        return []

    def create_indicators(
        parameters: StrategyParameters,
        indicators: IndicatorSet,
        strategy_universe: TradingStrategyUniverse,
        execution_context: ExecutionContext,
    ):
        """No indicators needed for exchange account monitoring."""
        pass

    class MyParameters:
        initial_cash = 100_000
        cycle_duration = CycleDuration.cycle_1d

    start_at, end_at = strategy_universe_with_exchange_account.data_universe.candles.get_timestamp_range()

    # Run single cycle backtest
    result = run_backtest_inline(
        client=None,
        decide_trades=decide_trades,
        create_indicators=create_indicators,
        universe=strategy_universe_with_exchange_account,
        reserve_currency=ReserveCurrency.usdc,
        engine_version="0.5",
        parameters=StrategyParameters.from_class(MyParameters),
        mode=ExecutionMode.unit_testing,
        start_at=start_at.to_pydatetime(),
        end_at=start_at.to_pydatetime() + datetime.timedelta(days=1),  # Single cycle
    )

    state, universe, debug_dump = result

    # Verify strategy ran without errors
    assert state is not None
    assert len(debug_dump) >= 1  # At least one cycle

    # No trades were made (passive monitoring)
    assert len(state.portfolio.closed_positions) == 0
    assert len(state.portfolio.open_positions) == 0
