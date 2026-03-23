"""Test P8: Activation cost single-deduction behaviour.

Verifies that:
1. Activation cost is only deducted from the first buy trade in a cycle.
2. Activation cost is stored per-trade in trade.other_data.
3. Settlement reads the per-trade value, not a stale instance variable.
"""

from decimal import Decimal
from unittest.mock import MagicMock, patch

def _make_routing(simulate=True):
    """Create a HypercoreVaultRouting with mocked dependencies."""
    from tradeexecutor.ethereum.vault.hypercore_routing import HypercoreVaultRouting

    routing = object.__new__(HypercoreVaultRouting)
    routing.web3 = MagicMock()
    routing.lagoon_vault = MagicMock()
    routing.lagoon_vault.safe_address = "0xSAFE"
    routing.deployer = MagicMock()
    routing.chain_id = 999
    routing.is_testnet = False
    routing.simulate = simulate
    routing.reserve_token_address = "0xusdc"
    routing._session = MagicMock()
    routing.allowed_intermediary_pairs = {}
    return routing


def _make_trade(planned_reserve=Decimal("100.0"), is_buy=True):
    trade = MagicMock()
    trade.is_buy.return_value = is_buy
    trade.is_vault.return_value = True
    trade.get_planned_reserve.return_value = planned_reserve
    trade.trade_id = 1
    trade.blockchain_transactions = []
    trade.other_data = {}
    trade.pair = MagicMock()
    trade.pair.pool_address = "0xVAULT"
    trade.pair.other_data = {"vault_protocol": "hypercore"}
    return trade


def _make_routing_state():
    rs = MagicMock()
    # Ensure routing doesn't try to replace deployer/vault from routing_state
    del rs.tx_builder
    return rs


def test_activation_cost_only_deducted_from_first_buy():
    """Two buy trades in the same cycle: only the first bears activation cost."""
    routing = _make_routing(simulate=False)
    state = MagicMock()
    trade1 = _make_trade(planned_reserve=Decimal("100.0"))
    trade2 = _make_trade(planned_reserve=Decimal("50.0"))

    costs_seen = []

    def capture_cost(trade, activation_cost_raw=0):
        costs_seen.append(activation_cost_raw)
        return [MagicMock()]

    # 1. Build two buy trades in the same cycle.
    # 2. Simulate activation followed by transaction creation.
    # 3. Verify only the first trade carries the activation cost.
    with patch.object(routing, "_create_deposit_or_withdraw_txs", side_effect=capture_cost):
        with patch("tradeexecutor.ethereum.vault.hypercore_routing.is_account_activated", return_value=False):
            with patch("tradeexecutor.ethereum.vault.hypercore_routing.activate_account"):
                with patch("tradeexecutor.ethereum.vault.hypercore_routing.fetch_user_abstraction_mode", return_value="standard"):
                    routing.setup_trades(state, _make_routing_state(), [trade1, trade2])

    # First buy gets the activation cost, second does not
    assert costs_seen[0] == 2_000_000
    assert costs_seen[1] == 0

    # Only the first trade has activation cost persisted
    assert trade1.other_data.get("hypercore_activation_cost_raw") == 2_000_000
    assert "hypercore_activation_cost_raw" not in trade2.other_data


def test_activation_cost_not_applied_to_sell():
    """Withdrawal trades never get activation cost deducted."""
    routing = _make_routing(simulate=False)
    state = MagicMock()
    sell_trade = _make_trade(planned_reserve=Decimal("50.0"), is_buy=False)

    costs_seen = []

    def capture_cost(trade, activation_cost_raw=0):
        costs_seen.append(activation_cost_raw)
        return [MagicMock()]

    # 1. Build one sell trade.
    # 2. Simulate an already activated Safe in standard mode.
    # 3. Verify no activation cost is applied.
    with patch.object(routing, "_create_deposit_or_withdraw_txs", side_effect=capture_cost):
        with patch("tradeexecutor.ethereum.vault.hypercore_routing.is_account_activated", return_value=True):
            with patch("tradeexecutor.ethereum.vault.hypercore_routing.fetch_user_abstraction_mode", return_value="standard"):
                routing.setup_trades(state, _make_routing_state(), [sell_trade])

    assert costs_seen[0] == 0


def test_setup_trades_logs_account_mode_without_blocking(caplog):
    """Log Hyperliquid account mode for diagnostics without blocking routing.

    1. Create one buy trade for a live Hypercore routing instance.
    2. Simulate an already activated Safe whose Hyperliquid API mode is unified.
    3. Verify setup still builds the trade and logs the observed mode.
    """
    import logging

    routing = _make_routing(simulate=False)
    state = MagicMock()
    trade = _make_trade(planned_reserve=Decimal("25.0"))

    # 1. Create one buy trade for a live Hypercore routing instance.
    # 2. Simulate an activated Safe whose Hyperliquid API mode is unified.
    # 3. Verify transaction creation still proceeds and the mode is logged.
    with caplog.at_level(logging.INFO):
        with patch.object(routing, "_create_deposit_or_withdraw_txs", return_value=[MagicMock()]) as create_txs:
            with patch("tradeexecutor.ethereum.vault.hypercore_routing.is_account_activated", return_value=True):
                with patch("tradeexecutor.ethereum.vault.hypercore_routing.fetch_user_abstraction_mode", return_value="unifiedAccount"):
                    routing.setup_trades(state, _make_routing_state(), [trade])

    create_txs.assert_called_once()
    assert any("unifiedAccount" in r.message for r in caplog.records)


def test_setup_trades_tolerates_account_mode_lookup_failure():
    """Treat account mode lookup as diagnostics-only.

    1. Create one buy trade for an already activated Safe.
    2. Simulate the Hyperliquid account-mode API call failing.
    3. Verify setup still builds the trade instead of aborting.
    """
    routing = _make_routing(simulate=False)
    state = MagicMock()
    trade = _make_trade(planned_reserve=Decimal("25.0"))

    # 1. Create one buy trade for an already activated Safe.
    # 2. Simulate an account-mode lookup failure.
    # 3. Verify transaction creation still proceeds.
    with patch.object(routing, "_create_deposit_or_withdraw_txs", return_value=[MagicMock()]) as create_txs:
        with patch("tradeexecutor.ethereum.vault.hypercore_routing.is_account_activated", return_value=True):
            with patch("tradeexecutor.ethereum.vault.hypercore_routing.fetch_user_abstraction_mode", side_effect=RuntimeError("boom")):
                routing.setup_trades(state, _make_routing_state(), [trade])

    create_txs.assert_called_once()


def test_setup_trades_checks_mode_after_activation():
    """Check account mode after activation before building Hypercore trades.

    1. Create one buy trade for an unactivated Safe.
    2. Simulate successful activation and a unified Hyperliquid mode read.
    3. Verify activation runs first and transaction building proceeds afterwards.
    """
    routing = _make_routing(simulate=False)
    state = MagicMock()
    trade = _make_trade(planned_reserve=Decimal("25.0"))

    # 1. Create one buy trade for an unactivated Safe.
    # 2. Simulate successful activation and a unified mode read.
    # 3. Verify the trade is built only after activation completes.
    with patch.object(routing, "_create_deposit_or_withdraw_txs", return_value=[MagicMock()]) as create_txs:
        with patch("tradeexecutor.ethereum.vault.hypercore_routing.is_account_activated", return_value=False):
            with patch("tradeexecutor.ethereum.vault.hypercore_routing.activate_account") as activate:
                with patch("tradeexecutor.ethereum.vault.hypercore_routing.fetch_user_abstraction_mode", return_value="unifiedAccount") as fetch_mode:
                    routing.setup_trades(state, _make_routing_state(), [trade])

    activate.assert_called_once()
    fetch_mode.assert_called_once()
    create_txs.assert_called_once()


def test_create_buy_transactions_split_approve_and_deposit():
    """Build live deposit phase 1 as separate approve and deposit calls.

    1. Create one buy trade for live Hypercore routing.
    2. Mock the new ``eth_defi`` approve and deposit builders plus transaction signing.
    3. Verify routing returns two transactions in the expected order.
    """
    routing = _make_routing(simulate=False)
    trade = _make_trade(planned_reserve=Decimal("25.0"))
    approve_fn = MagicMock()
    deposit_fn = MagicMock()
    approve_tx = MagicMock()
    deposit_tx = MagicMock()

    # 1. Create one buy trade for live Hypercore routing.
    # 2. Mock the separate approve and deposit builders and signing.
    # 3. Verify routing returns the two phase-1 transactions in order.
    with patch("tradeexecutor.ethereum.vault.hypercore_routing.build_hypercore_approve_deposit_wallet_call", return_value=approve_fn) as build_approve:
        with patch("tradeexecutor.ethereum.vault.hypercore_routing.build_hypercore_deposit_to_spot_call", return_value=deposit_fn) as build_deposit:
            with patch.object(routing, "_sign_module_call", side_effect=[approve_tx, deposit_tx]) as sign_call:
                txs = routing._create_deposit_or_withdraw_txs(trade)

    assert txs == [approve_tx, deposit_tx]
    build_approve.assert_called_once()
    build_deposit.assert_called_once()
    assert sign_call.call_count == 2


def test_create_sell_transactions_build_vault_withdraw_phase1_only():
    """Build live withdrawal phase 1 as a standalone vault-withdraw call.

    1. Create one sell trade for live Hypercore routing.
    2. Mock the reusable ``eth_defi`` vault-withdraw builder plus transaction signing.
    3. Verify routing returns one phase-1 transaction for vault -> perp.
    """
    routing = _make_routing(simulate=False)
    trade = _make_trade(planned_reserve=Decimal("25.0"), is_buy=False)
    trade.pair.pool_address = "0x1111111111111111111111111111111111111111"
    withdraw_fn = MagicMock()
    signed_tx = MagicMock()

    # 1. Create one sell trade for live Hypercore routing.
    # 2. Mock the reusable vault-withdraw builder and signing.
    # 3. Verify routing returns exactly one phase-1 withdrawal transaction.
    with patch("tradeexecutor.ethereum.vault.hypercore_routing.build_hypercore_withdraw_from_vault_call", return_value=withdraw_fn) as build_withdraw:
        with patch.object(routing, "_sign_module_call", return_value=signed_tx) as sign_call:
            txs = routing._create_deposit_or_withdraw_txs(trade)

    assert txs == [signed_tx]
    build_withdraw.assert_called_once_with(
        routing.lagoon_vault,
        vault_address="0x1111111111111111111111111111111111111111",
        hypercore_usdc_amount=25_000_000,
    )
    sign_call.assert_called_once()


@patch("tradeexecutor.ethereum.vault.hypercore_routing.report_failure")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.get_block_timestamp")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.wait_for_evm_escrow_clear")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.wait_for_vault_deposit_confirmation")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.fetch_user_vault_equity")
def test_settlement_second_buy_no_activation_cost(
    mock_fetch_equity, mock_wait_confirm, mock_escrow, mock_block_ts,
    mock_report_failure,
):
    """_settle_deposit for a second buy must NOT deduct activation cost.

    Regression: a second buy in the same cycle has no
    "hypercore_activation_cost_raw" in other_data.  Settlement must
    treat activation_cost as 0 and deposit the full planned amount.
    """
    import datetime
    from eth_defi.hyperliquid.api import UserVaultEquity
    from hexbytes import HexBytes

    routing = _make_routing(simulate=False)

    # Second buy: no activation cost in other_data
    trade = _make_trade(planned_reserve=Decimal("50.0"))
    trade.blockchain_transactions = [MagicMock(tx_hash="0xaa")]
    trade.other_data = {}
    # Crucially, NO "hypercore_activation_cost_raw" key

    state = MagicMock()
    mock_block_ts.return_value = datetime.datetime(2025, 1, 1)
    mock_escrow.return_value = None

    receipts = {HexBytes("0xaa"): {"status": 1, "blockNumber": 100}}

    # Equity snapshot before phase 2
    mock_fetch_equity.return_value = UserVaultEquity(
        vault_address="0xVAULT",
        equity=Decimal("200.0"),
        locked_until=datetime.datetime(2030, 1, 1),
    )

    # Phase 2 broadcast
    phase2_tx = MagicMock(tx_hash="0xbb")
    phase2_receipt = {"status": 1, "blockNumber": 101}

    # Deposit confirmation
    confirmed_eq = UserVaultEquity(
        vault_address="0xVAULT",
        equity=Decimal("250.0"),
        locked_until=datetime.datetime(2030, 1, 1),
    )
    mock_wait_confirm.return_value = confirmed_eq

    mock_phase2 = MagicMock(return_value=(phase2_tx, phase2_receipt))
    with patch.object(routing, "_broadcast_phase2", mock_phase2):
        routing._settle_deposit(
            routing.web3, state, trade, receipts,
            stop_on_execution_failure=False,
        )

    # Trade succeeded
    state.mark_trade_success.assert_called_once()
    mock_report_failure.assert_not_called()

    # The executed_amount should be 50 USDC (full amount), NOT 48 USDC
    call_kwargs = state.mark_trade_success.call_args
    executed_amount = call_kwargs[1].get("executed_amount") or call_kwargs[0][2]
    assert executed_amount == Decimal("50.0"), (
        f"Second buy settled as {executed_amount} instead of 50.0 — "
        f"activation cost was incorrectly deducted"
    )

    # Verify _broadcast_phase2 received the full 50 USDC raw (not 48)
    deposit_raw_passed = mock_phase2.call_args[0][2]
    assert deposit_raw_passed == 50_000_000
