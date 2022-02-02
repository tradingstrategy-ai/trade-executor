import os
import datetime
from decimal import Decimal
from pathlib import Path

import pytest

from tradeexecutor.state.state import State, AssetIdentifier, ReservePosition

from tradeexecutor.strategy.bootstrap import bootstrap_strategy
from tradeexecutor.strategy.runner import Dataset, StrategyRunner
from tradeexecutor.utils.timer import timed_task
from tradingstrategy.chain import ChainId
from tradingstrategy.universe import Universe

@pytest.fixture
def usdc() -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(ChainId.ethereum.value, "0x0", "USDC", 6)


def test_execute_empty_strategy_cycle(strategy_folder, persistent_test_client):
    """Create an empty strategy and execute one tick."""
    clock_now = datetime.datetime(2022, 1, 1, tzinfo=None)
    client = persistent_test_client
    strategy_path = Path(os.path.join(strategy_folder, "empty.py"))
    dataset, universe, runner = bootstrap_strategy(client, timed_task, strategy_path, clock_now)
    state = State()
    trade_instructions = runner.on_clock(clock_now, universe, state)
    assert len(trade_instructions) == 0


def test_dummy_strategy_clock(strategy_folder, persistent_test_client, usdc):
    """Run a test on a strategy that trades a single pair and always returns the same trading instructions."""
    clock_now = datetime.datetime(2022, 1, 1, tzinfo=None)
    client = persistent_test_client
    strategy_path = Path(os.path.join(strategy_folder, "dummy.py"))
    dataset, universe, runner = bootstrap_strategy(client, timed_task, strategy_path, clock_now)

    assert isinstance(dataset, Dataset)
    assert isinstance(universe, Universe)
    assert isinstance(runner, StrategyRunner)

    state = State()

    reserve = ReservePosition(usdc, Decimal(500), clock_now, 1.0, clock_now)
    state.portfolio.reserves[reserve.get_identifier()] = reserve

    trade_instructions = runner.on_clock(clock_now, universe, state)
    assert len(trade_instructions) == 1
    instruction = trade_instructions[0]
    assert instruction.trade_id == 1
    assert instruction.pair.base_token_symbol == "WMATIC"
    assert instruction.pair.quote_token_symbol == "USDC"
    assert instruction.planned_quantity == 100
    assert instruction.planned_reserve == pytest.approx(Decimal(270))
