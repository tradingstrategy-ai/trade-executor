"""Check that we repair data correctly.

- Use some collected live state dumps with failed trades (buy, sell) in them. Thanks Polygon!

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
    f = os.path.join(os.path.dirname(__file__), "legacy-repair-dump.json")
    return State.from_json(open(f, "rt").read())


def test_assess_repair_need(
        state: State,
        persistent_test_client: Client,
):
    """Get the trades that need repair.

    We have both buys and sells.
    """

    t = state.portfolio.get_trade_by_id(70)
    assert t.opened_at == datetime.datetime(2023, 3, 1, 19, 51, 0, 858)

    assert len(state.portfolio.frozen_positions) == 3
    repair_report = repair_trades(state, attempt_repair=False, interactive=False)
    assert len(repair_report.frozen_positions) == 3
    assert len(repair_report.trades_needing_repair) == 3

    trades = repair_report.trades_needing_repair
    assert trades[0].is_buy()
    assert trades[0].position_id == 1
    assert trades[1].is_buy()
    assert trades[1].position_id == 26
    assert trades[2].is_sell()
    assert trades[2].position_id == 36
    assert trades[2].trade_id == 70

    assert trades[2].opened_at == datetime.datetime(2023, 3, 1, 19, 51, 0, 858)

    # Before repair run some summary statistics to see they don't crash
    execution_mode = ExecutionMode.real_trading
    clock = datetime.datetime(2023, 4, 1)
    calculate_statistics(clock, state.portfolio, execution_mode)


def test_repair_trades(
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
    pos1 = portfolio.get_position_by_id(1)
    pos2 = portfolio.get_position_by_id(26)
    pos3 = portfolio.get_position_by_id(36)

    # We have $6.8 in reserves - after fixing positions this amount should be
    # restored from the reserve allocated for failed buys
    assert portfolio.get_cash() == 6.815099

    # Failed buy, allocated capital needs to be released
    assert pos1.get_quantity_old() == 0
    assert pos1.get_unexeuted_reserve() == pytest.approx(Decimal('69.505733499999990954165696166455745697021484375'))
    assert pos1.is_frozen()
    t = t1 = pos1.get_last_trade()
    assert t.is_buy()
    assert t.is_failed()
    assert t.get_value() == pytest.approx(69.50573349999999)
    assert t.get_reserve_quantity() == pytest.approx(Decimal('69.505733499999990954165696166455745697021484375'))

    assert pos2.is_frozen()
    assert pos2.get_quantity_old() == 0
    assert pos2.get_unexeuted_reserve() == pytest.approx(Decimal('60.87260269999999451329131261'))
    t = t2 = pos1.get_last_trade()
    assert t.is_buy()

    # Failed sell, the trade must be marked as never happened
    assert pos3.is_frozen()
    assert pos3.get_quantity_old() == pytest.approx(Decimal('3.180200722896299527'))
    assert pos3.get_unexeuted_reserve() == pytest.approx(Decimal('0'))
    t = t3 = pos3.get_last_trade()
    assert t.is_sell()
    assert t.is_failed()
    assert t.get_value() == pytest.approx(55.501273)
    assert t.get_reserve_quantity() == Decimal('0')

    repair_report = repair_trades(state, attempt_repair=True, interactive=False)
    assert len(repair_report.unfrozen_positions) == 3
    assert len(repair_report.new_trades) == 3

    # Check how our positions look like
    pos1 = state.portfolio.get_position_by_id(1)
    assert pos1.is_closed()
    assert pos1.get_quantity_old() == 0
    assert pos1.get_unexeuted_reserve() == 0
    assert pos1.get_value() == 0
    assert t1.get_reserve_quantity() == 0
    assert t1.is_repaired()
    assert not t1.is_repair_needed()

    pos2 = state.portfolio.get_position_by_id(26)
    assert pos2.is_closed()
    assert pos2.get_quantity_old() == 0
    assert pos2.get_unexeuted_reserve() == 0
    assert pos2.get_value() == 0
    assert t2.is_repaired()

    # When closing failed (sell) position repair is done,
    # the tokens are left on the position
    pos3 = state.portfolio.get_position_by_id(36)
    assert pos3.is_open()
    assert pos3.get_quantity_old() == Decimal('3.180200722896299527')
    assert pos3.get_unexeuted_reserve() == 0
    assert pos3.get_value() == pytest.approx(48.802913)  # Unsold tokens hold some value
    assert t3.is_repaired()

    # Check the counter trades
    t4 = pos1.get_last_trade()
    assert t4.repaired_trade_id == t1.trade_id
    assert t4.is_repair_trade()
    assert t4.get_status() == TradeStatus.success
    assert t4.trade_type == TradeType.repair
    assert t4.get_value() == 0
    assert t4.get_position_quantity() == 0

    assert portfolio.get_cash() == pytest.approx(137.19343519999998)
    assert len(list(portfolio.get_unfrozen_positions())) == 3

    # After repair run some summary statistics to see they don't crash
    execution_mode = ExecutionMode.real_trading
    clock = datetime.datetime(2023, 4, 1)
    calculate_statistics(clock, portfolio, execution_mode)

    # We can serialise the repaired state
    dump = state.to_json()
    state2 = State.from_json(dump)
    state2.perform_integrity_check()

