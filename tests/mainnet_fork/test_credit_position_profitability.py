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
    profit = credit_position.get_unrealised_and_realised_profitability_percent_credit()
    assert profit == pytest.approx(0.049043081055064434)