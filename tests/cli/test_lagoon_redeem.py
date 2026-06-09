"""CLI coverage for lagoon-redeem pre-flight deposit and redemption claiming."""

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from tradeexecutor.cli.commands.lagoon_redeem import _claim_leftover_deposits, _claim_leftover_redemptions


@pytest.mark.timeout(30)
def test_claim_leftover_redemptions() -> None:
    """Pre-flight handles both settled-unclaimed and pending-unsettled leftover redemptions.

    1. Call with maxRedeem() > 0 (state A): verify finalise_redeem with explicit raw_amount, no settlement.
    2. Call with maxRedeem() == 0 and pendingRedeemRequest() > 0 (state B): verify settle + poll + finalise.
    3. Call with both == 0: verify no transactions broadcast.
    """
    tx_hash = b"\x00" * 32

    def make_mocks():
        vault = MagicMock()
        vault.finalise_redeem.return_value = "mock_finalise"
        vault.post_new_valuation.return_value = "mock_post_val"
        vault.settle_via_trading_strategy_module.return_value = "mock_settle"
        vault.fetch_nav.return_value = Decimal("10000")
        hot_wallet = MagicMock()
        hot_wallet.address = "0xABCD"
        hot_wallet.transact_and_broadcast_with_contract.return_value = tx_hash
        web3 = MagicMock()
        share_token = MagicMock()
        share_token.symbol = "lagVault"
        share_token.convert_to_decimals.return_value = Decimal("500")
        return vault, hot_wallet, web3, share_token

    # 1. State A: settled but unclaimed — finalise immediately, no settlement
    vault, hot_wallet, web3, share_token = make_mocks()
    settled_raw = 500_000_000
    vault.vault_contract.functions.maxRedeem.return_value.call.return_value = settled_raw
    vault.vault_contract.functions.pendingRedeemRequest.return_value.call.return_value = 0

    with patch("tradeexecutor.cli.commands.lagoon_redeem.assert_transaction_success_with_explanation"):
        _claim_leftover_redemptions(vault, hot_wallet, web3, share_token)

    vault.finalise_redeem.assert_called_once_with("0xABCD", raw_amount=settled_raw)
    vault.post_new_valuation.assert_not_called()
    vault.settle_via_trading_strategy_module.assert_not_called()

    # 2. State B: pending unsettled — settle, poll maxRedeem (0 then >0), finalise
    vault, hot_wallet, web3, share_token = make_mocks()
    pending_raw = 750_000_000
    vault.vault_contract.functions.maxRedeem.return_value.call.side_effect = [
        0,            # state A check — nothing settled
        0,            # first poll attempt (stale RPC)
        pending_raw,  # second poll attempt
    ]
    vault.vault_contract.functions.pendingRedeemRequest.return_value.call.return_value = pending_raw

    with (
        patch("tradeexecutor.cli.commands.lagoon_redeem.assert_transaction_success_with_explanation"),
        patch("tradeexecutor.cli.commands.lagoon_redeem.time.sleep") as mock_sleep,
    ):
        _claim_leftover_redemptions(vault, hot_wallet, web3, share_token)
        mock_sleep.assert_called_once_with(5)

    vault.post_new_valuation.assert_called_once()
    vault.settle_via_trading_strategy_module.assert_called_once()
    vault.finalise_redeem.assert_called_once_with("0xABCD", raw_amount=pending_raw)

    # 3. No-op: both 0 — no transactions
    vault, hot_wallet, web3, share_token = make_mocks()
    vault.vault_contract.functions.maxRedeem.return_value.call.return_value = 0
    vault.vault_contract.functions.pendingRedeemRequest.return_value.call.return_value = 0

    _claim_leftover_redemptions(vault, hot_wallet, web3, share_token)

    vault.finalise_redeem.assert_not_called()
    hot_wallet.transact_and_broadcast_with_contract.assert_not_called()


@pytest.mark.timeout(30)
def test_claim_leftover_deposits() -> None:
    """Pre-flight claims unclaimed deposits so shares move from vault contract to hot wallet.

    1. Call with maxDeposit() > 0: verify finalise_deposit is called with explicit raw_amount.
    2. Call with maxDeposit() == 0: verify no transactions broadcast.
    """
    tx_hash = b"\x00" * 32

    def make_mocks():
        vault = MagicMock()
        vault.finalise_deposit.return_value = "mock_finalise_deposit"
        vault.denomination_token.symbol = "USDC"
        vault.denomination_token.convert_to_decimals.return_value = Decimal("10")
        vault.share_token.symbol = "MASTER"
        vault.share_token.fetch_balance_of.return_value = Decimal("10")
        hot_wallet = MagicMock()
        hot_wallet.address = "0xABCD"
        hot_wallet.transact_and_broadcast_with_contract.return_value = tx_hash
        web3 = MagicMock()
        return vault, hot_wallet, web3

    # 1. Unclaimed deposit: maxDeposit > 0 — finalise_deposit called
    vault, hot_wallet, web3 = make_mocks()
    deposit_raw = 10_000_000
    vault.vault_contract.functions.maxDeposit.return_value.call.return_value = deposit_raw

    with patch("tradeexecutor.cli.commands.lagoon_redeem.assert_transaction_success_with_explanation"):
        _claim_leftover_deposits(vault, hot_wallet, web3)

    vault.finalise_deposit.assert_called_once_with("0xABCD", raw_amount=deposit_raw)
    hot_wallet.transact_and_broadcast_with_contract.assert_called_once()

    # 2. No unclaimed deposits: maxDeposit == 0 — no transactions
    vault, hot_wallet, web3 = make_mocks()
    vault.vault_contract.functions.maxDeposit.return_value.call.return_value = 0

    _claim_leftover_deposits(vault, hot_wallet, web3)

    vault.finalise_deposit.assert_not_called()
    hot_wallet.transact_and_broadcast_with_contract.assert_not_called()
