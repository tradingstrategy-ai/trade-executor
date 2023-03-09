"""Check that we repair data correctly.

- Use some collected live state dumps with failed trades (buy, sell) in them. Thanks Polygon!

"""
import os

import pytest

from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.state.repair import repair_trades
from tradeexecutor.state.state import State
from tradingstrategy.client import Client


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request, mute_requests=False)


@pytest.fixture(scope="module")
def state() -> State:
    """Read a random old data dump."""
    f = os.path.join(os.path.dirname(__file__), "legacy-repair-dump.json")
    return State.from_json(open(f, "rt").read())


def test_assess_repair_need(
        state: State,
        persistent_test_client: Client,
):
    """Get the trades that need repair.

    We have both buys and sells.
    """

    repair_report = repair_trades(state, attempt_repair=False, interactive=False)
    assert len(repair_report.frozen_positions) == 3
    assert len(repair_report.trades_needing_repair) == 3

    trades = repair_report.trades_needing_repair
    assert trades[0].is_buy()
    assert trades[1].is_buy()
    assert trades[2].is_sell()


def test_repair_trades(
        state: State,
        persistent_test_client: Client,
):
    """Repair trades.

    We have both buys and sells.
    """
    repair_report = repair_trades(state, attempt_repair=True, interactive=False)
    assert len(repair_report.unfrozen_positions) == 3
    assert len(repair_report.new_trades) == 3



