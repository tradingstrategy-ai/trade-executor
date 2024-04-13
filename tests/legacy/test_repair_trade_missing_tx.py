"""Check that we can repair trades that never generated blockchaint transactions due to a crash.

"""
import datetime
import os
from decimal import Decimal

import pytest

from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.monkeypatch.dataclasses_json import patch_dataclasses_json
from tradeexecutor.state.repair import repair_trades
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeStatus, TradeType
from tradeexecutor.statistics.core import calculate_statistics
from tradeexecutor.strategy.execution_context import ExecutionMode
from tradingstrategy.client import Client


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request, mute_requests=False)


@pytest.fixture(scope="module")
def state() -> State:
    """A state dump with some failed trades we need to unwind.

    Taken as a snapshot from alpha version trade execution run.
    """
    f = os.path.join(os.path.dirname(__file__), "trade-missing-tx.json")
    return State.from_json(open(f, "rt").read())


def test_repair_trade_missing_tx(
    state: State,
    persistent_test_client: Client,
):
    """Repair trades.

    We have both buys and sells.

    Failed positions are 1 (buy), 26 (buy), 36 (sell)
    """

    # Check how our positions look like
    # before repair
    portfolio = state.portfolio
    pos1 = portfolio.get_position_by_id(4)
    import ipdb ; ipdb.set_trace()
    pos2 = portfolio.get_position_by_id(26)
    pos3 = portfolio.get_position_by_id(36)



    # We can serialise the repaired state
    dump = state.to_json()
    state2 = State.from_json(dump)
    state2.perform_integrity_check()

