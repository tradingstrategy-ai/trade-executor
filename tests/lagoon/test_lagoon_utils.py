"""Unit tests for shared Lagoon CLI helpers."""

from decimal import Decimal
from types import SimpleNamespace

import pytest

from tradeexecutor.cli.commands.lagoon_utils import get_lagoon_reserve_baseline
from tradeexecutor.state.state import State


def test_lighter_activation_baseline_has_no_reserve() -> None:
    """Accept the empty Safe/state baseline left by Lighter activation.

    1. Create a clean executor state with no reserve position.
    2. Represent the empty Lagoon Safe after its 1 USDC activation transfer.
    3. Verify the normal Lagoon deposit flow may initialise the reserve later.
    """
    # 1. Create a clean executor state with no reserve position.
    state = State(name="Lighter activation test")
    denomination_token = SimpleNamespace(
        address="0x0000000000000000000000000000000000000001",
        symbol="USDC",
    )

    # 2. Represent the empty Lagoon Safe after its 1 USDC activation transfer.
    safe_balance = Decimal(0)

    # 3. Verify the normal Lagoon deposit flow may initialise the reserve later.
    reserve = get_lagoon_reserve_baseline(state, denomination_token, safe_balance)
    assert reserve is None


def test_missing_reserve_rejects_nonempty_safe() -> None:
    """Reject untracked Safe USDC before an operator submits another deposit.

    1. Create a clean executor state with no reserve position.
    2. Represent a Safe that has USDC which the executor does not track.
    3. Verify the helper requires a Lagoon settlement before new capital moves.
    """
    # 1. Create a clean executor state with no reserve position.
    state = State(name="Untracked Safe test")
    denomination_token = SimpleNamespace(
        address="0x0000000000000000000000000000000000000001",
        symbol="USDC",
    )

    # 2. Represent a Safe that has USDC which the executor does not track.
    safe_balance = Decimal("1")

    # 3. Verify the helper requires a Lagoon settlement before new capital moves.
    with pytest.raises(ValueError, match="run lagoon-settle"):
        get_lagoon_reserve_baseline(state, denomination_token, safe_balance)
