import os
from pathlib import Path

import pytest

from tradeexecutor.state.state import State


@pytest.fixture()
def state_file() -> Path:
    """Because we modifty state file when fixing it, we need to make a working copy from the master copy."""
    p = Path(os.path.join(os.path.dirname(__file__), "credit-no-profit.json"))
    assert p.exists(), f"{p} missing"
    return p


def test_credit_position_profitability(state_file):
    state = State.read_json_file(state_file)
    credit_position = state.portfolio.open_positions[2]
    profit = credit_position.estimate_gained_interest(interest_period="year")
    assert profit == pytest.approx(0.049043081055064434, rel=0.01)

    profit = credit_position.estimate_gained_interest(interest_period="position")
    assert profit == pytest.approx(0.000724640625, rel=0.10)