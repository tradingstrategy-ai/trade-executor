"""Tests for cross-chain satellite async vault settlement handling in test trades.

Why this exists: a CCTP-bridged ERC-7540 (Lagoon) or Ostium V1.5 satellite
deposit lands in ``vault_settlement_pending`` after its ``requestDeposit``
transaction confirms — the request succeeds on-chain but no shares are minted
until the vault operator settles the queue off-chain. The cross-chain
test-trade flow (``_make_cross_chain_test_trade``) used to assert
``satellite_buy_trade.is_success()`` immediately after executing the satellite
deposit. For an async vault that assertion is always False at request time, and
because nothing reverted ``get_revert_reason()`` is ``None`` — which is exactly
how this crashed ``perform-test-trade`` / ``trade-ui`` in production:

    AssertionError: Satellite open failed: None

The fix routes every satellite trade through
``_resolve_satellite_async_settlement()`` before the ``is_success()`` assertion.
These tests lock in its three branches so the regression cannot return.

The real-chain async settlement mechanics (force-settle, claim, escrow,
restart-after-claim) are covered by the fork tests in
``tests/erc_4626/test_vault_async_lagoon_erc_7540.py``; here we isolate the
cross-chain test-trade control-flow decision that those tests do not exercise.
"""

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from tradingstrategy.chain import ChainId

from tradeexecutor.cli import testtrade
from tradeexecutor.cli.testtrade import _resolve_satellite_async_settlement, _make_cross_chain_test_trade
from tradeexecutor.state.trade import TradeStatus
from tradeexecutor.strategy.generic.generic_pricing_model import GenericPricing


BASE_CHAIN_ID = ChainId.base.value


def _make_trade(status: TradeStatus) -> MagicMock:
    """Build a minimal trade stub exposing the status interface the resolver reads."""
    trade = MagicMock()
    trade.trade_id = 14
    trade.get_status.return_value = status
    return trade


def test_synchronous_satellite_trade_continues(monkeypatch) -> None:
    """A synchronous satellite deposit lets the test-trade flow continue.

    1. Build a satellite trade that already executed successfully (sync vault).
    2. Resolve the satellite async settlement on a real chain.
    3. Verify the resolver tells the caller to continue (return False).
    4. Verify it never reaches for a chain connection or forces settlement.
    """
    # 1. Build a satellite trade that already executed successfully (sync vault).
    trade = _make_trade(TradeStatus.success)

    # 2. Resolve the satellite async settlement on a real chain.
    forced = []
    monkeypatch.setattr(
        testtrade, "_force_vault_settlement_and_resolve",
        lambda *a, **k: forced.append(True),
    )
    web3config = MagicMock()
    stop = _resolve_satellite_async_settlement(
        trade=trade,
        on_anvil=False,
        web3config=web3config,
        dest_chain_id=BASE_CHAIN_ID,
        chain_name="Base",
        state=MagicMock(),
        execution_model=MagicMock(),
    )

    # 3. Verify the resolver tells the caller to continue (return False).
    assert stop is False

    # 4. Verify it never reaches for a chain connection or forces settlement.
    web3config.get_connection.assert_not_called()
    assert forced == []


def test_pending_satellite_trade_on_mainnet_stops_without_crashing(monkeypatch) -> None:
    """A pending async satellite deposit on a real chain stops gracefully.

    This is the exact production scenario: the satellite ``requestDeposit``
    succeeded but settlement is off-chain, so the command must report and exit
    rather than crash with "Satellite open failed: None".

    1. Build a satellite trade left in ``vault_settlement_pending``.
    2. Resolve it with ``on_anvil=False`` (real chain semantics).
    3. Verify the resolver tells the caller to stop (return True) and raises nothing.
    4. Verify it does NOT attempt to force settlement (cannot rush an operator).
    """
    # 1. Build a satellite trade left in vault_settlement_pending.
    trade = _make_trade(TradeStatus.vault_settlement_pending)

    # 2. Resolve it with on_anvil=False (real chain semantics).
    forced = []
    monkeypatch.setattr(
        testtrade, "_force_vault_settlement_and_resolve",
        lambda *a, **k: forced.append(True),
    )
    web3config = MagicMock()
    stop = _resolve_satellite_async_settlement(
        trade=trade,
        on_anvil=False,
        web3config=web3config,
        dest_chain_id=BASE_CHAIN_ID,
        chain_name="Base",
        state=MagicMock(),
        execution_model=MagicMock(),
    )

    # 3. Verify the resolver tells the caller to stop (return True) and raises nothing.
    assert stop is True

    # 4. Verify it does NOT attempt to force settlement (cannot rush an operator).
    assert forced == []
    web3config.get_connection.assert_not_called()


def test_pending_satellite_trade_on_anvil_force_settles_on_destination_chain(monkeypatch) -> None:
    """A pending async satellite deposit on Anvil force-settles on the satellite chain.

    The destination-chain connection — not the home chain — must drive the
    operator-impersonation settlement, otherwise the settle transaction is sent
    to the wrong chain / wrong Safe module.

    1. Build a satellite trade left in ``vault_settlement_pending``.
    2. Resolve it with ``on_anvil=True`` and a web3config mapping Base -> a marker web3.
    3. Verify the resolver tells the caller to continue (return False) after forcing.
    4. Verify settlement was forced with the destination-chain connection and the
       web3config forwarded for chain-aware claiming.
    """
    # 1. Build a satellite trade left in vault_settlement_pending.
    trade = _make_trade(TradeStatus.vault_settlement_pending)

    # 2. Resolve it with on_anvil=True and a web3config mapping Base -> a marker web3.
    dest_web3 = object()
    web3config = MagicMock()
    web3config.get_connection.return_value = dest_web3
    captured = {}

    def fake_force(web3, state, trade, execution_model, web3config=None):
        captured["web3"] = web3
        captured["state"] = state
        captured["trade"] = trade
        captured["execution_model"] = execution_model
        captured["web3config"] = web3config
        # Simulate a successful operator settlement: the queue resolved, so the
        # trade is no longer pending (matches force_lagoon_settle + claim on Anvil).
        trade.get_status.return_value = TradeStatus.success

    monkeypatch.setattr(testtrade, "_force_vault_settlement_and_resolve", fake_force)

    state = MagicMock()
    execution_model = MagicMock()
    stop = _resolve_satellite_async_settlement(
        trade=trade,
        on_anvil=True,
        web3config=web3config,
        dest_chain_id=BASE_CHAIN_ID,
        chain_name="Base",
        state=state,
        execution_model=execution_model,
    )

    # 3. Verify the resolver tells the caller to continue (return False) after forcing.
    assert stop is False

    # 4. Verify settlement was forced with the destination-chain connection and the
    #    web3config forwarded for chain-aware claiming.
    web3config.get_connection.assert_called_once_with(ChainId(BASE_CHAIN_ID))
    assert captured["web3"] is dest_web3
    assert captured["web3config"] is web3config
    assert captured["trade"] is trade
    assert captured["state"] is state
    assert captured["execution_model"] is execution_model


def test_cross_chain_buy_does_not_crash_when_satellite_open_is_pending(monkeypatch) -> None:
    """The cross-chain buy flow tolerates an async satellite open (production crash).

    This drives the real ``_make_cross_chain_test_trade()`` buy flow with the
    satellite deposit landing in ``vault_settlement_pending`` on a real chain —
    the exact production scenario. The collaborators that touch the chain
    (execution, bridging, position sizing) are mocked so only the orchestration
    control flow is exercised. The old code asserted
    ``satellite_buy_trade.is_success()`` here and raised
    "Satellite open failed: None"; the fix must return gracefully instead.

    1. Mock a real chain (``is_anvil`` False) and a successful CCTP bridge-in trade.
    2. Make the internally-constructed PositionManager return a satellite deposit
       trade stuck in ``vault_settlement_pending``.
    3. Run the cross-chain buy-only test trade.
    4. Verify it returns without raising and the satellite trade was executed but
       never asserted to be successful.
    """
    # 1. Mock a real chain (is_anvil False) and a successful CCTP bridge-in trade.
    monkeypatch.setattr(testtrade, "is_anvil", lambda web3: False)
    # Guard: settlement must never be force-resolved on a real chain.
    monkeypatch.setattr(
        testtrade, "_force_vault_settlement_and_resolve",
        lambda *a, **k: pytest.fail("Must not force settlement on a real chain"),
    )

    bridge_trade = MagicMock()
    bridge_trade.is_success.return_value = True
    bridge_pm = MagicMock()
    bridge_pm.open_spot.return_value = [bridge_trade]

    # 2. Make the internally-constructed PositionManager return a pending satellite deposit.
    satellite_trade = _make_trade(TradeStatus.vault_settlement_pending)
    satellite_pm = MagicMock()
    satellite_pm.open_spot.return_value = [satellite_trade]
    monkeypatch.setattr(testtrade, "PositionManager", MagicMock(return_value=satellite_pm))

    execution_model = MagicMock()
    pricing_model = MagicMock(spec=GenericPricing)  # satisfies the isinstance() guard
    routing_model = MagicMock()  # has a pair_configurator attribute by default
    pair = MagicMock()
    pair.chain_id = BASE_CHAIN_ID

    # 3. Run the cross-chain buy-only test trade.
    result = _make_cross_chain_test_trade(
        web3=MagicMock(),
        web3config=MagicMock(),
        execution_model=execution_model,
        pricing_model=pricing_model,
        sync_model=MagicMock(),
        state=MagicMock(),
        universe=MagicMock(),
        routing_model=routing_model,
        routing_state=MagicMock(),
        position_manager=bridge_pm,
        pair=pair,
        bridge_pair=MagicMock(),
        amount=Decimal("5"),
        max_slippage=0.05,
        buy_only=True,
        close_only=False,
        gas_at_start=Decimal("1"),
        hot_wallet=MagicMock(),
        reserve_currency="USDC",
        reserve_currency_at_start=80.0,
    )

    # 4. Verify it returns without raising and the satellite trade was executed but
    #    never asserted to be successful.
    assert result is None
    bridge_pm.open_spot.assert_called_once()
    satellite_pm.open_spot.assert_called_once()
    execution_model.execute_trades.assert_called()
    satellite_trade.is_success.assert_not_called()
