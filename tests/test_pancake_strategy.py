import os
import datetime
from pathlib import Path

import pytest

from tradeexecutor.state.state import State

from tradeexecutor.strategy.bootstrap import bootstrap_strategy
from tradeexecutor.strategy.runner import StrategyRunner
from tradeexecutor.utils.timer import timed_task
from tradingstrategy.universe import Universe


@pytest.mark.skip(msg="Not ready")
def test_pancake_simple(strategy_folder, persistent_test_client):
    """Tests a QSTrader based strategy runner that is more or less real."""
    clock_now = datetime.datetime(2022, 1, 1, tzinfo=datetime.timezone.utc)
    client = persistent_test_client
    strategy_path = Path(os.path.join(strategy_folder, "pancakeswap.py"))
    dataset, universe, runner = bootstrap_strategy(client, timed_task, strategy_path, clock_now, max_data_age=datetime.timedelta(days=30))

    assert isinstance(dataset, Dataset)
    assert isinstance(universe, Universe)
    assert isinstance(runner, StrategyRunner)

    state = State()
    trade_instructions = runner.on_clock(clock_now, universe, state)
    assert len(trade_instructions) == 1
    instruction = trade_instructions[0]
    assert instruction.trade_id == 1
    assert instruction.trading_pair.base.token_symbol == "WMATIC"
    assert instruction.trading_pair.quote.token_symbol == "USDC"
    assert instruction.requested_quantity == 100
