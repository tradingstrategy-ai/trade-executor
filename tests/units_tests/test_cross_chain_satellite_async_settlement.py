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


def test_pending_satellite_simulation_on_anvil_stops_without_forcing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A simulated async satellite deposit remains a closed diagnostic.

    1. Build a satellite deposit pending operator settlement on an Anvil fork.
    2. Resolve it with forced settlement explicitly disabled for simulation.
    3. Verify the caller is told to stop and no chain mutation is attempted.
    """
    # 1. Build a satellite deposit pending operator settlement on an Anvil fork.
    trade = _make_trade(TradeStatus.vault_settlement_pending)
    monkeypatch.setattr(
        testtrade,
        "_force_vault_settlement_and_resolve",
        lambda *a, **k: pytest.fail("Simulation must not force async settlement"),
    )
    web3config = MagicMock()

    # 2. Resolve it with forced settlement explicitly disabled for simulation.
    stop = _resolve_satellite_async_settlement(
        trade=trade,
        on_anvil=True,
        web3config=web3config,
        dest_chain_id=BASE_CHAIN_ID,
        chain_name="Base",
        state=MagicMock(),
        execution_model=MagicMock(),
        force_on_anvil=False,
    )

    # 3. Verify the caller is told to stop and no chain mutation is attempted.
    assert stop is True
    web3config.get_connection.assert_not_called()


def test_cross_chain_buy_does_not_crash_when_satellite_open_is_pending(monkeypatch) -> None:
    """The cross-chain buy flow tolerates an async satellite open (production crash).

    This drives the real ``_make_cross_chain_test_trade()`` buy flow with the
    satellite deposit landing in ``vault_settlement_pending`` on a real chain —
    the exact production scenario. The collaborators that touch the chain
    (execution, bridging, position sizing) are mocked so only the orchestration
    control flow is exercised. The pending satellite trade is faithful to
    production: ``is_success()`` is ``False`` and ``get_revert_reason()`` is
    ``None`` (nothing reverted).

    The regression is locked by the ``is_success.assert_not_called()`` check: the
    fix must reach the early ``return`` *before* the ``is_success()`` assertion,
    whereas the old code called ``satellite_buy_trade.is_success()`` (and, with a
    real-bool ``False`` return, raised "Satellite open failed: None"). So this
    test fails on the old code — either on the ``assert_not_called`` check or on
    the AssertionError it would raise.

    1. Mock a real chain (``is_anvil`` False) and a successful CCTP bridge-in trade.
    2. Make the internally-constructed PositionManager return a satellite deposit
       trade stuck in ``vault_settlement_pending`` (is_success False, no revert).
    3. Run the cross-chain buy-only test trade.
    4. Verify it returns without raising and the satellite trade was executed but
       its success was never asserted.
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
    bridge_trade.executed_quantity = Decimal("5")
    bridge_pm = MagicMock()
    bridge_pm.open_cctp_bridge_position.return_value = [bridge_trade]

    # 2. Make the internally-constructed PositionManager return a pending satellite deposit.
    #    Faithful to production: requestDeposit confirmed but not settled, so the
    #    trade is not yet successful and nothing reverted.
    satellite_trade = _make_trade(TradeStatus.vault_settlement_pending)
    satellite_trade.is_success.return_value = False
    satellite_trade.get_revert_reason.return_value = None
    satellite_pm = MagicMock()
    satellite_pm.open_spot.return_value = [satellite_trade]
    monkeypatch.setattr(testtrade, "PositionManager", MagicMock(return_value=satellite_pm))

    execution_model = MagicMock()
    pricing_model = MagicMock(spec=GenericPricing)  # satisfies the isinstance() guard
    routing_model = MagicMock()  # has a pair_configurator attribute by default
    pair = MagicMock()
    pair.chain_id = BASE_CHAIN_ID

    # 3. Run the cross-chain buy-only test trade.
    state = MagicMock()
    bridge_position = MagicMock()
    bridge_position.other_data = {}
    bridge_position.get_available_bridge_capital.return_value = Decimal("4.95")
    state.portfolio.get_bridge_position_for_chain.return_value = bridge_position
    result = _make_cross_chain_test_trade(
        web3=MagicMock(),
        web3config=MagicMock(),
        execution_model=execution_model,
        pricing_model=pricing_model,
        sync_model=MagicMock(),
        state=state,
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
    bridge_pm.open_cctp_bridge_position.assert_called_once()
    satellite_pm.open_spot.assert_called_once()
    execution_model.execute_trades.assert_called()
    satellite_trade.is_success.assert_not_called()


def test_cross_chain_buy_resumes_from_available_bridge_capital_without_trade_history(monkeypatch) -> None:
    """A settled CCTP leg resumes safely even without a successful buy-trade record.

    1. Model a bridge position marked for resume with capital but no successful trade.
    2. Make the satellite deposit enter async settlement pending.
    3. Run the resumed cross-chain buy flow.
    4. Verify it deposits only available bridge capital and advances the resume phase.
    """
    # 1. Model a bridge position marked for resume with capital but no successful trade.
    monkeypatch.setattr(testtrade, "is_anvil", lambda web3: False)
    vault_address = "0x0000000000000000000000000000000000000001"
    pair = MagicMock()
    pair.chain_id = BASE_CHAIN_ID
    pair.pool_address = vault_address
    pair.get_ticker.return_value = "VAULT-USDC"
    bridge_attempt = {
        "vault_id": f"{BASE_CHAIN_ID}-{vault_address}",
        "phase": "bridge_out_pending",
    }
    bridge_position = MagicMock()
    bridge_position.other_data = {"vault_test_attempt": bridge_attempt}
    bridge_position.trades = {}
    bridge_position.get_available_bridge_capital.return_value = Decimal("4.75")
    state = MagicMock()
    state.portfolio.get_bridge_position_for_chain.return_value = bridge_position

    # 2. Make the satellite deposit enter async settlement pending.
    satellite_trade = _make_trade(TradeStatus.vault_settlement_pending)
    satellite_pm = MagicMock()
    satellite_pm.open_spot.return_value = [satellite_trade]
    monkeypatch.setattr(testtrade, "PositionManager", MagicMock(return_value=satellite_pm))

    # 3. Run the resumed cross-chain buy flow.
    original_position_manager = MagicMock()
    _make_cross_chain_test_trade(
        web3=MagicMock(),
        web3config=MagicMock(),
        execution_model=MagicMock(),
        pricing_model=MagicMock(spec=GenericPricing),
        sync_model=MagicMock(),
        state=state,
        universe=MagicMock(),
        routing_model=MagicMock(),
        routing_state=MagicMock(),
        position_manager=original_position_manager,
        pair=pair,
        bridge_pair=MagicMock(),
        amount=Decimal("5"),
        max_slippage=0.05,
        buy_only=True,
        close_only=False,
        gas_at_start=Decimal("1"),
        hot_wallet=MagicMock(),
        reserve_currency="USDC",
        reserve_currency_at_start=5.0,
    )

    # 4. Verify it deposits only available bridge capital and advances the resume phase.
    original_position_manager.open_cctp_bridge_position.assert_not_called()
    assert satellite_pm.open_spot.call_args.args[1] == pytest.approx(4.75)
    assert bridge_attempt["phase"] == "deposit_requested"


def test_cross_chain_close_resumes_bridge_back_after_async_redemption_settles(monkeypatch: pytest.MonkeyPatch) -> None:
    """A later invocation finishes CCTP after an async redemption has settled.

    1. Model a closed satellite position and its still-open CCTP bridge position.
    2. Make the position manager create a successful bridge-back trade.
    3. Run the close-only cross-chain flow as a later command invocation.
    4. Verify it skips a second vault redemption and closes only the bridge.
    """
    # 1. Model a closed satellite position and its still-open CCTP bridge position.
    monkeypatch.setattr(testtrade, "is_anvil", lambda web3: False)
    pair = MagicMock()
    pair.chain_id = BASE_CHAIN_ID
    pair.get_ticker.return_value = "VAULT-USDC"
    closed_satellite_position = MagicMock()
    closed_satellite_position.pair = pair
    bridge_position = MagicMock()
    bridge_position.is_open.return_value = True
    state = MagicMock()
    state.portfolio.get_position_by_trading_pair.return_value = None
    state.portfolio.closed_positions = {1: closed_satellite_position}
    state.portfolio.get_bridge_position_for_chain.return_value = bridge_position
    state.portfolio.get_default_reserve_position.return_value.get_value.return_value = 4.9
    state.portfolio.get_all_trades.return_value = []

    # 2. Make the position manager create a successful bridge-back trade.
    bridge_back_trade = _make_trade(TradeStatus.success)
    bridge_back_trade.is_success.return_value = True
    resumed_position_manager = MagicMock()
    resumed_position_manager.close_position.return_value = [bridge_back_trade]
    monkeypatch.setattr(testtrade, "PositionManager", MagicMock(return_value=resumed_position_manager))
    execution_model = MagicMock()
    hot_wallet = MagicMock()
    hot_wallet.get_native_currency_balance.return_value = Decimal("1")

    # 3. Run the close-only cross-chain flow as a later command invocation.
    result = _make_cross_chain_test_trade(
        web3=MagicMock(),
        web3config=MagicMock(),
        execution_model=execution_model,
        pricing_model=MagicMock(spec=GenericPricing),
        sync_model=MagicMock(),
        state=state,
        universe=MagicMock(),
        routing_model=MagicMock(),
        routing_state=MagicMock(),
        position_manager=MagicMock(),
        pair=pair,
        bridge_pair=MagicMock(),
        amount=Decimal("5"),
        max_slippage=0.05,
        buy_only=False,
        close_only=True,
        gas_at_start=Decimal("1"),
        hot_wallet=hot_wallet,
        reserve_currency="USDC",
        reserve_currency_at_start=5.0,
    )

    # 4. Verify it skips a second vault redemption and closes only the bridge.
    assert result is None
    resumed_position_manager.close_position.assert_called_once()
    assert resumed_position_manager.close_position.call_args.args[0] is bridge_position
    execution_model.execute_trades.assert_called_once()
