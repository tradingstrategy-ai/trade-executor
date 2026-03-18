"""Test P8: Activation cost single-deduction behaviour.

Verifies that:
1. Activation cost is only deducted from the first buy trade in a cycle.
2. Activation cost is stored per-trade in trade.other_data.
3. Settlement reads the per-trade value, not a stale instance variable.
"""

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest


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

    with patch.object(routing, "_create_deposit_or_withdraw_txs", side_effect=capture_cost):
        with patch("tradeexecutor.ethereum.vault.hypercore_routing.is_account_activated", return_value=False):
            with patch("tradeexecutor.ethereum.vault.hypercore_routing.activate_account"):
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

    with patch.object(routing, "_create_deposit_or_withdraw_txs", side_effect=capture_cost):
        with patch("tradeexecutor.ethereum.vault.hypercore_routing.is_account_activated", return_value=True):
            routing.setup_trades(state, _make_routing_state(), [sell_trade])

    assert costs_seen[0] == 0


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
