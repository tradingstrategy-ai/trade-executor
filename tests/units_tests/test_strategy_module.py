import os
from pathlib import Path

import pytest

from tradeexecutor.strategy.strategy_module import read_strategy_module


@pytest.fixture(scope="module")
def strategy_file() -> Path:
    """A state dump with some failed trades we need to unwind.

    Taken as a snapshot from alpha version trade execution run.
    """
    f = os.path.join(os.path.dirname(__file__), "..", "..", "strategies", "matic-breakout.py")
    p = Path(f)
    assert p.exists()
    return p


def test_read_strategy_module(
    strategy_file,
):
    """Read strategy module and check for some basic information."""
    mod = read_strategy_module(strategy_file)
    assert mod.sort_priority == 99
    assert mod.trading_strategy_protocol_fee == "0.02%"
    assert mod.strategy_developer_fee == "0.1%"
    assert mod.management_fee == "0.00%"
    assert mod.enzyme_protocol_fee == "0.0025%"