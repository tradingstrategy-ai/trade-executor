"""Check that we can repair trades that never generated blockchaint transactions due to a crash.

"""
import datetime
import os
from decimal import Decimal

import pytest

from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.monkeypatch.dataclasses_json import patch_dataclasses_json
from tradeexecutor.state.repair import repair_trades, repair_tx_not_generated
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
):
    """Repair trades.

    We have both buys and sells.

    Failed positions are 1 (buy), 26 (buy), 36 (sell)
    """

    # Check how our positions look like
    # before repair
    portfolio = state.portfolio
    pos = portfolio.get_position_by_id(4)
    # {6: <Buy #6 0.8714827051238582225784361754 WETH at 3504.1829635670897, success phase>, 8: <Sell #8 0.871478149181352629 WETH at 3478.199651761113, planned phase>, 11: <Buy #11 0.1727611356789154584852805521 WETH at 3232.903676743047, planned phase>, 13: <Buy #13 0.1784399145897731857878380438 WETH at 3253.869387693136, planned phase>}
    assert pos.trades[8].get_status() == TradeStatus.planned
    assert pos.trades[8].blockchain_transactions == []
    assert pos.trades[8].is_missing_blockchain_transactions()
    assert len(pos.trades) == 4

    assert pos.get_value() == pytest.approx(2822.223554)
    assert portfolio.calculate_total_equity() == pytest.approx(4049.33755226)

    repair_trades = repair_tx_not_generated(state, interactive=False)
    assert len(repair_trades) == 7  # 7 repairs across two positions

    # We went from planned -> repaird
    pos = portfolio.get_position_by_id(4)
    assert pos.trades[8].get_status() == TradeStatus.repaired
    assert pos.trades[8].blockchain_transactions == []
    assert pos.trades[8].is_repaired()
    assert not pos.trades[8].is_missing_blockchain_transactions()

    assert len(pos.trades) == 7  # 4 original + 3 repairs added

    assert pos.get_value() == pytest.approx(2822.223554)
    assert portfolio.calculate_total_equity() == pytest.approx(4049.33755226)
