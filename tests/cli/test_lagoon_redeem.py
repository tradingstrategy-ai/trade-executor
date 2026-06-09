"""CLI coverage for lagoon-redeem pre-flight and Phase 3 polling."""

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from tradeexecutor.cli.commands.lagoon_redeem import (
    _claim_leftover_redemptions,
    _poll_and_finalise_redeem,
)


@pytest.mark.timeout(30)
def test_preflight_claims_settled_unclaimed_redemption() -> None:
    """Pre-flight state A: settled but unclaimed redemption is claimed via finalise_redeem.

    1. Set up maxRedeem() > 0 and pendingRedeemRequest() == 0.
    2. Call _claim_leftover_redemptions.
    3. Verify finalise_redeem was called with the observed raw_amount.
    4. Verify no settlement transactions were broadcast (state A skips settle).
    """
    raw_amount = 500_000_000

    vault = MagicMock()
    vault.vault_contract.functions.maxRedeem.return_value.call.return_value = raw_amount
    vault.vault_contract.functions.pendingRedeemRequest.return_value.call.return_value = 0
    vault.finalise_redeem.return_value = "mock_finalise"

    hot_wallet = MagicMock()
    hot_wallet.address = "0xABCD"
    hot_wallet.transact_and_broadcast_with_contract.return_value = b"\x00" * 32

    web3 = MagicMock()

    share_token = MagicMock()
    share_token.symbol = "lagVault"
    share_token.convert_to_decimals.return_value = Decimal("500")

    # 2. Call _claim_leftover_redemptions
    with patch(
        "tradeexecutor.cli.commands.lagoon_redeem.assert_transaction_success_with_explanation"
    ):
        _claim_leftover_redemptions(vault, hot_wallet, web3, share_token)

    # 3. Verify finalise_redeem was called with the observed raw_amount
    vault.finalise_redeem.assert_called_once_with("0xABCD", raw_amount=raw_amount)

    # 4. No settlement transactions (post_new_valuation, settle_via_trading_strategy_module)
    vault.post_new_valuation.assert_not_called()
    vault.settle_via_trading_strategy_module.assert_not_called()


@pytest.mark.timeout(30)
def test_preflight_settles_and_claims_pending_unsettled_redemption() -> None:
    """Pre-flight state B: pending unsettled redemption is settled, polled, and claimed.

    1. Set up maxRedeem() == 0 initially (nothing settled yet) and pendingRedeemRequest() > 0.
    2. After settlement, maxRedeem() returns > 0 on second poll.
    3. Call _claim_leftover_redemptions.
    4. Verify post_new_valuation and settle_via_trading_strategy_module were called.
    5. Verify finalise_redeem was called with the polled raw_amount.
    """
    pending_raw = 750_000_000
    settled_raw = 750_000_000

    vault = MagicMock()
    # State A check: nothing settled yet
    # State B check: pending redemption exists
    # Then _poll_and_finalise_redeem polls maxRedeem: 0 first, then settled_raw
    vault.vault_contract.functions.maxRedeem.return_value.call.side_effect = [
        0,  # state A check
        0,  # first poll attempt inside _poll_and_finalise_redeem
        settled_raw,  # second poll attempt
    ]
    vault.vault_contract.functions.pendingRedeemRequest.return_value.call.return_value = pending_raw
    vault.finalise_redeem.return_value = "mock_finalise"
    vault.post_new_valuation.return_value = "mock_post_val"
    vault.settle_via_trading_strategy_module.return_value = "mock_settle"
    vault.fetch_nav.return_value = Decimal("10000")

    hot_wallet = MagicMock()
    hot_wallet.address = "0xABCD"
    hot_wallet.transact_and_broadcast_with_contract.return_value = b"\x00" * 32

    web3 = MagicMock()

    share_token = MagicMock()
    share_token.symbol = "lagVault"
    share_token.convert_to_decimals.return_value = Decimal("750")

    # 3. Call _claim_leftover_redemptions
    with (
        patch("tradeexecutor.cli.commands.lagoon_redeem.assert_transaction_success_with_explanation"),
        patch("tradeexecutor.cli.commands.lagoon_redeem.time.sleep"),
    ):
        _claim_leftover_redemptions(vault, hot_wallet, web3, share_token)

    # 4. Verify settlement was performed
    vault.post_new_valuation.assert_called_once()
    vault.settle_via_trading_strategy_module.assert_called_once()

    # 5. Verify finalise_redeem was called with the polled raw_amount
    vault.finalise_redeem.assert_called_once_with("0xABCD", raw_amount=settled_raw)


@pytest.mark.timeout(30)
def test_preflight_skips_when_no_leftover_redemptions() -> None:
    """Pre-flight does nothing when maxRedeem and pendingRedeemRequest are both 0.

    1. Set up maxRedeem() == 0 and pendingRedeemRequest() == 0.
    2. Call _claim_leftover_redemptions.
    3. Verify no transactions were broadcast.
    """
    vault = MagicMock()
    vault.vault_contract.functions.maxRedeem.return_value.call.return_value = 0
    vault.vault_contract.functions.pendingRedeemRequest.return_value.call.return_value = 0

    hot_wallet = MagicMock()
    hot_wallet.address = "0xABCD"

    web3 = MagicMock()
    share_token = MagicMock()

    # 2. Call _claim_leftover_redemptions
    _claim_leftover_redemptions(vault, hot_wallet, web3, share_token)

    # 3. No transactions broadcast
    vault.finalise_redeem.assert_not_called()
    vault.post_new_valuation.assert_not_called()
    vault.settle_via_trading_strategy_module.assert_not_called()
    hot_wallet.transact_and_broadcast_with_contract.assert_not_called()


@pytest.mark.timeout(30)
def test_poll_and_finalise_redeem_retries_on_stale_rpc() -> None:
    """_poll_and_finalise_redeem polls maxRedeem when the first call returns 0 (stale RPC).

    1. Set up maxRedeem() to return 0 on the first call, then > 0 on the second.
    2. Patch time.sleep to a no-op.
    3. Call _poll_and_finalise_redeem.
    4. Verify maxRedeem was called twice and finalise_redeem used the polled raw_amount.
    """
    raw_amount = 1_000_000_000

    vault = MagicMock()
    vault.vault_contract.functions.maxRedeem.return_value.call.side_effect = [0, raw_amount]
    vault.finalise_redeem.return_value = "mock_bound_func"

    hot_wallet = MagicMock()
    hot_wallet.address = "0xABCD"
    hot_wallet.transact_and_broadcast_with_contract.return_value = b"\x00" * 32

    web3 = MagicMock()

    share_token = MagicMock()
    share_token.symbol = "lagVault"
    share_token.convert_to_decimals.return_value = Decimal("1000")

    # 2. Patch time.sleep to a no-op
    # 3. Call _poll_and_finalise_redeem
    with (
        patch("tradeexecutor.cli.commands.lagoon_redeem.time.sleep") as mock_sleep,
        patch("tradeexecutor.cli.commands.lagoon_redeem.assert_transaction_success_with_explanation"),
    ):
        _poll_and_finalise_redeem(vault, hot_wallet, web3, share_token)
        mock_sleep.assert_called_once_with(5)

    # 4. Verify maxRedeem was called twice (first returned 0, second returned raw_amount)
    assert vault.vault_contract.functions.maxRedeem.return_value.call.call_count == 2

    # Verify finalise_redeem used the polled raw_amount
    vault.finalise_redeem.assert_called_once_with(
        "0xABCD", raw_amount=raw_amount
    )
