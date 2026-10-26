"""Keep CLI, decision and trigger slippage configuration consistent."""

import datetime
from contextlib import nullcontext
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket

from tradeexecutor.cli.slippage import configure_max_slippage_tolerance
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.pricing_model import FixedPricing
from tradeexecutor.strategy.routing import RoutingState
from tradeexecutor.strategy.runner import StrategyRunner
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange
from tradeexecutor.testing.synthetic_price_data import generate_ohlcv_candles
from tradeexecutor.testing.synthetic_universe_data import create_synthetic_single_pair_universe


@pytest.fixture
def universe() -> TradingStrategyUniverse:
    """Supply local candle data for real position managers without running a backtest."""
    exchange = generate_exchange(1, ChainId.ethereum, "0x" + "01" * 20)
    usdc = AssetIdentifier(1, "0x" + "02" * 20, "USDC", 6)
    token = AssetIdentifier(1, "0x" + "03" * 20, "TEST", 18)
    pair = TradingPairIdentifier(
        token, usdc, "0x" + "04" * 20, exchange.address,
        internal_id=1, internal_exchange_id=1, fee=0.003,
    )
    candles = generate_ohlcv_candles(
        TimeBucket.d1, datetime.datetime(2026, 1, 1), datetime.datetime(2026, 1, 3), pair_id=1,
    )
    return create_synthetic_single_pair_universe(candles, ChainId.ethereum, exchange, TimeBucket.d1, pair)


@pytest.mark.parametrize(
    "values, expected",
    [
        ({"slippage_tolerance_pct": 0.006}, 0.006),
        ({"slippage_tolerance": 0.009}, 0.009),
        ({"slippage_tolerance": 0.008, "slippage_tolerance_pct": 0.006}, 0.008),
        ({"slippage_tolerance": None, "slippage_tolerance_pct": 0.006}, 0.006),
    ],
)
def test_configured_slippage_reaches_decision_and_trigger_managers(
    values: dict,
    expected: float,
    universe: TradingStrategyUniverse,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply either configured parameter to trades and trigger checks.

    1. Compare CLI resolution with a real decision position manager and planned trade.
    2. Enter the trigger runner with external execution and synchronisation stubbed.
    3. Verify both managers use the same tolerance without mutating parameters.
    """
    # 1. Fixed prices and local candles keep this a unit test without simulation.
    parameters = StrategyParameters(values.copy())
    state = State()
    reserve = state.portfolio.initialise_reserves(universe.get_reserve_asset(), reserve_token_price=1.0)
    reserve.quantity = Decimal(1000)
    context = ExecutionContext(ExecutionMode.unit_testing)
    pricing = FixedPricing(1.0, 0.003)
    timestamp = pd.Timestamp("2026-01-02")
    strategy_input = StrategyInput(
        cycle=1, timestamp=timestamp, state=state, strategy_universe=universe,
        parameters=parameters, indicators=None, pricing_model=pricing,
        execution_context=context, other_data={},
    )
    configured = configure_max_slippage_tolerance(None, SimpleNamespace(parameters=parameters))
    manager = strategy_input.get_position_manager()
    trade = manager.open_spot(universe.get_single_pair(), 100.0)[0]

    # 2. Only the config hand-off is under test; no account or network calls run.
    execution = Mock(get_safe_latest_block=Mock(return_value=None))
    sync = Mock(has_async_deposits=Mock(return_value=True), sync_interests=Mock(return_value=[]))
    trigger_check = Mock(return_value=[])
    monkeypatch.setattr("tradeexecutor.strategy.runner.check_position_triggers", trigger_check)
    runner = SimpleNamespace(
        parameters=parameters, execution_model=execution, sync_model=sync,
        execution_context=context, check_accounts=Mock(), max_price_impact=None,
        approval_model=SimpleNamespace(confirm_trades=lambda state, trades: trades),
        timed_task_context_manager=lambda *args, **kwargs: nullcontext(),
    )
    StrategyRunner.check_position_triggers(
        runner, timestamp.to_pydatetime(), state, universe, pricing, Mock(spec=RoutingState),
    )

    # 3. Assert the actual manager and trade boundaries, not just the lookup helper.
    assert configured == expected
    assert manager.default_slippage_tolerance == expected
    assert trade.slippage_tolerance == expected
    assert trigger_check.call_args.args[0].default_slippage_tolerance == expected
    assert dict(parameters) == values
    assert configure_max_slippage_tolerance(0.012, SimpleNamespace(parameters=parameters)) == 0.012


def test_unconfigured_slippage_preserves_existing_defaults(
    universe: TradingStrategyUniverse,
) -> None:
    """Keep the existing per-consumer defaults when neither name is configured.

    1. Create a strategy input with empty parameters and a CLI without parameters.
    2. Resolve both paths without executing any strategy cycle.
    3. Verify the position-manager and CLI defaults remain unchanged.
    """
    # 1. No pricing or execution is requested from the local fixture.
    state = State()
    state.portfolio.initialise_reserves(universe.get_reserve_asset(), reserve_token_price=1.0)
    strategy_input = StrategyInput(
        cycle=1, timestamp=pd.Timestamp("2026-01-02"), state=state, strategy_universe=universe,
        parameters=StrategyParameters({}), indicators=None, pricing_model=FixedPricing(1.0, 0.003),
        execution_context=ExecutionContext(ExecutionMode.unit_testing), other_data={},
    )

    # 2. Both call sites retain their original fallback policy.
    manager = strategy_input.get_position_manager()
    cli_tolerance = configure_max_slippage_tolerance(None, SimpleNamespace(parameters=None))

    # 3. The compatibility lookup does not unify or change unrelated defaults.
    assert manager.default_slippage_tolerance == 0.017
    assert cli_tolerance == 0.005
