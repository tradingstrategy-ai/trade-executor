"""Tests for home-chain async vault settlement handling in test trades.

Why this exists: closing a home-chain async vault position (Ostium V1.5 or
ERC-7540 Lagoon) is a ``requestWithdraw`` / ``requestRedeem`` that confirms
on-chain but leaves the trade in ``vault_settlement_pending`` — the request
succeeds but no USDC is paid out until the vault operator settles the queue
off-chain. ``make_test_trade()`` open path tolerated this; the close path did
not and fell through to ``assert sell_trade.is_success()``. For an async redeem
that assertion is always False at request time, and because nothing reverted
``get_revert_reason()`` is ``None`` — which is exactly how this crashed
``perform-test-trade`` / ``trade-ui`` in production when closing our Ostium
positions:

    AssertionError: Test sell failed

The fix routes both the open and the close path through
:func:`_resolve_home_chain_async_settlement` before their ``is_success()``
assertions, so the two paths can no longer diverge. These tests lock in its
three branches so the regression cannot return.

The real on-chain async settlement mechanics (force-settle, claim, escrow) are
covered by the fork tests in ``tests/erc_4626/test_vault_async_ostium_v15.py``
and the end-to-end CLI cycle in ``tests/erc_4626/test_vault_async_cli_commands.py``;
here we isolate the home-chain test-trade control-flow decision in isolation.
"""

from unittest.mock import MagicMock

import pytest

from tradeexecutor.cli import testtrade
from tradeexecutor.cli.testtrade import _resolve_home_chain_async_settlement
from tradeexecutor.state.trade import TradeStatus


def _make_trade(status: TradeStatus) -> MagicMock:
    """Build a minimal trade stub exposing the status interface the resolver reads."""
    trade = MagicMock()
    trade.trade_id = 17
    trade.get_status.return_value = status
    return trade


def test_synchronous_home_trade_continues(monkeypatch) -> None:
    """A synchronous home-chain trade lets the test-trade flow continue.

    1. Build a home-chain trade that already executed successfully (sync vault).
    2. Resolve the async settlement.
    3. Verify the resolver tells the caller to continue (return False).
    4. Verify it never forces settlement (nothing to settle).
    """
    # 1. Build a home-chain trade that already executed successfully (sync vault).
    trade = _make_trade(TradeStatus.success)

    # 2. Resolve the async settlement.
    forced = []
    monkeypatch.setattr(
        testtrade, "_force_vault_settlement_and_resolve",
        lambda *a, **k: forced.append(True),
    )
    # is_anvil must never even be consulted for a non-pending trade.
    monkeypatch.setattr(
        testtrade, "is_anvil",
        lambda web3: pytest.fail("is_anvil must not be checked for a non-pending trade"),
    )
    stop = _resolve_home_chain_async_settlement(
        trade=trade,
        web3=MagicMock(),
        state=MagicMock(),
        execution_model=MagicMock(),
    )

    # 3. Verify the resolver tells the caller to continue (return False).
    assert stop is False

    # 4. Verify it never forces settlement.
    assert forced == []


def test_pending_home_trade_on_mainnet_stops_without_crashing(monkeypatch) -> None:
    """A pending async home-chain close on a real chain stops gracefully.

    This is the exact production scenario: closing an Ostium position issues a
    ``requestWithdraw`` that succeeds, but settlement is off-chain, so the
    command must report and exit rather than crash with "Test sell failed".

    1. Build a close trade left in ``vault_settlement_pending``.
    2. Resolve it on a real chain (``is_anvil`` False).
    3. Verify the resolver tells the caller to stop (return True) and raises nothing.
    4. Verify it does NOT attempt to force settlement (cannot rush an operator).
    """
    # 1. Build a close trade left in vault_settlement_pending.
    trade = _make_trade(TradeStatus.vault_settlement_pending)

    # 2. Resolve it on a real chain (is_anvil False).
    monkeypatch.setattr(testtrade, "is_anvil", lambda web3: False)
    forced = []
    monkeypatch.setattr(
        testtrade, "_force_vault_settlement_and_resolve",
        lambda *a, **k: forced.append(True),
    )
    stop = _resolve_home_chain_async_settlement(
        trade=trade,
        web3=MagicMock(),
        state=MagicMock(),
        execution_model=MagicMock(),
    )

    # 3. Verify the resolver tells the caller to stop (return True) and raises nothing.
    assert stop is True

    # 4. Verify it does NOT attempt to force settlement (cannot rush an operator).
    assert forced == []


def test_pending_home_trade_on_anvil_force_settles(monkeypatch) -> None:
    """A pending async home-chain trade on Anvil force-settles and continues.

    On Anvil ``tryNewSettlement()`` is permissionless, so the test trade plays
    the operator, force-settles, and the helper returns False so the caller
    proceeds to its normal ``is_success()`` check.

    1. Build a trade left in ``vault_settlement_pending``.
    2. Resolve it with ``is_anvil`` True; the forced settlement marks it success.
    3. Verify the resolver tells the caller to continue (return False) after forcing.
    4. Verify settlement was forced with the home-chain connection, state and trade.
    """
    # 1. Build a trade left in vault_settlement_pending.
    trade = _make_trade(TradeStatus.vault_settlement_pending)

    # 2. Resolve it with is_anvil True; the forced settlement marks it success.
    monkeypatch.setattr(testtrade, "is_anvil", lambda web3: True)
    web3 = object()
    state = MagicMock()
    execution_model = MagicMock()
    captured = {}

    def fake_force(passed_web3, passed_state, passed_trade, passed_execution_model, web3config=None):
        captured["web3"] = passed_web3
        captured["state"] = passed_state
        captured["trade"] = passed_trade
        captured["execution_model"] = passed_execution_model
        # Simulate a successful operator settlement: the queue resolved, so the
        # trade is no longer pending (matches force_ostium_v15_settlement + claim).
        passed_trade.get_status.return_value = TradeStatus.success

    monkeypatch.setattr(testtrade, "_force_vault_settlement_and_resolve", fake_force)

    stop = _resolve_home_chain_async_settlement(
        trade=trade,
        web3=web3,
        state=state,
        execution_model=execution_model,
    )

    # 3. Verify the resolver tells the caller to continue (return False) after forcing.
    assert stop is False

    # 4. Verify settlement was forced with the home-chain connection, state and trade.
    assert captured["web3"] is web3
    assert captured["state"] is state
    assert captured["trade"] is trade
    assert captured["execution_model"] is execution_model


def test_pending_home_trade_on_anvil_unresolved_raises(monkeypatch) -> None:
    """If forced settlement on Anvil does not resolve, surface a clear error.

    A failed force-settle must not silently fall through to the caller's
    ``is_success()`` assertion (which would report a misleading "Test sell
    failed" with no revert reason). The helper asserts the trade left the
    pending state.

    1. Build a trade left in ``vault_settlement_pending``.
    2. Force settlement on Anvil but leave the trade still pending (no-op).
    3. Verify the helper raises AssertionError naming the trade.
    """
    # 1. Build a trade left in vault_settlement_pending.
    trade = _make_trade(TradeStatus.vault_settlement_pending)

    # 2. Force settlement on Anvil but leave the trade still pending (no-op).
    monkeypatch.setattr(testtrade, "is_anvil", lambda web3: True)
    monkeypatch.setattr(
        testtrade, "_force_vault_settlement_and_resolve",
        lambda *a, **k: None,  # does not change status -> stays pending
    )

    # 3. Verify the helper raises AssertionError naming the trade.
    with pytest.raises(AssertionError, match="did not resolve test trade #17"):
        _resolve_home_chain_async_settlement(
            trade=trade,
            web3=MagicMock(),
            state=MagicMock(),
            execution_model=MagicMock(),
        )
