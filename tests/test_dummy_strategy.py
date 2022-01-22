from threading import Thread
import os
import requests
import datetime

from tradeexecutor.state.inmemory import InMemoryStore
from tradeexecutor.state.state import State
from tradeexecutor.strategy.importer import import_strategy_file
from tradeexecutor.trade.dummy import DummyExecutionModel
from tradeexecutor.webhook.server import create_webhook_server


def test_execute_empty_strategy_cycle(strategy_folder):
    """Creates a main loop for a dummy strategy and see it gets back with trade instructions."""

    strategy_path = os.path.join(strategy_folder, "empty.py")
    runner = import_strategy_file(strategy_path)

    state = State()
    clock_now = datetime.datetime(2022, 1, 1, tzinfo=datetime.timezone.utc)
    trade_instructions = runner.on_clock(clock_now, state)
    assert len(trade_instructions) == 0
