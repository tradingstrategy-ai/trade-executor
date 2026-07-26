"""Test for a bug case from a productoin state."""

import os
import shutil

from pathlib import Path

import pytest

from tradeexecutor.state.state import State


@pytest.fixture()
def state_file(tmp_path: Path) -> Path:
    """Return a mutable copy of the historic state fixture."""
    path = tmp_path / "test-correct-account.json"
    source = os.path.join(os.path.dirname(__file__), "correct-accounts-token-price-missing.json")
    shutil.copy(source, path)
    return path


def test_cli_correct_account_price_missing(
    state_file: Path,
):
    """Verify the historic state can calculate a previously missing token price.

    1. Load the copied historic state fixture.
    2. Select the affected closed position.
    3. Calculate its quantity value without a missing-price assertion.
    """
    # 1. Load the copied historic state fixture.
    state = State.read_json_file(state_file)

    # 2. Select the affected closed position.
    position = state.portfolio.closed_positions[1]

    # 3. Calculate its quantity value without a missing-price assertion.
    value = position.calculate_quantity_usd_value(1)
    assert value is not None
