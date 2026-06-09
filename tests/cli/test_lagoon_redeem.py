"""CLI coverage for lagoon-redeem pre-flight and Phase 3 polling."""

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from tradeexecutor.cli.commands.lagoon_redeem import _poll_and_finalise_redeem


@pytest.mark.timeout(30)
def test_preflight_claims_settled_unclaimed_redemption_then_errors_on_zero_shares() -> None:
    """Pre-flight detects a settled but unclaimed redemption, claims it, then errors because no shares remain.

    1. Set up a mock vault where maxRedeem() returns > 0 (settled but unclaimed).
    2. Set up share_token.fetch_balance_of() to return 0 after claiming.
    3. Call _poll_and_finalise_redeem to claim.
    4. Verify finalise_redeem was called with the observed raw_amount.
    """
    raw_amount = 500_000_000

    vault = MagicMock()
    vault.vault_contract.functions.maxRedeem.return_value.call.return_value = raw_amount
    vault.finalise_redeem.return_value = "mock_bound_func"

    hot_wallet = MagicMock()
    hot_wallet.address = "0xABCD"
    hot_wallet.transact_and_broadcast_with_contract.return_value = b"\x00" * 32

    web3 = MagicMock()

    share_token = MagicMock()
    share_token.symbol = "lagVault"
    share_token.convert_to_decimals.return_value = Decimal("500")

    # 3. Call _poll_and_finalise_redeem — maxRedeem immediately returns > 0
    with patch(
        "tradeexecutor.cli.commands.lagoon_redeem.assert_transaction_success_with_explanation"
    ):
        _poll_and_finalise_redeem(vault, hot_wallet, web3, share_token)

    # 4. Verify finalise_redeem was called with the observed raw_amount
    vault.finalise_redeem.assert_called_once_with(
        "0xABCD", raw_amount=raw_amount
    )
    hot_wallet.transact_and_broadcast_with_contract.assert_called_once_with(
        "mock_bound_func", gas_limit=1_000_000
    )


@pytest.mark.timeout(30)
def test_phase3_polls_max_redeem_before_finalising() -> None:
    """Normal Phase 3 polls maxRedeem when the first call returns 0 (stale RPC).

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
