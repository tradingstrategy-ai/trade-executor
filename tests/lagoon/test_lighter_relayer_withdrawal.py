"""Test Lighter withdrawals that its relayer claims for the Safe.

``withdrawPendingBalance`` is permissionless, so Lighter's relayer can pay a
withdrawal before the executor claims it, or while its claim is in flight. The claim
then reverts although the money has arrived. A full lifecycle with the relayer
winning is covered by ``test_lagoon_lighter_multi_subaccount_e2e.py``. These tests
cover the race, which a fork cannot time.
"""

from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest
from eth_defi.confirmation import Reverted
from pytest_mock import MockerFixture

import tradeexecutor.cli.commands.lagoon_lighter_test_trade as test_trade_module
import tradeexecutor.ethereum.lighter.lighter_routing as routing_module
import tradeexecutor.ethereum.lighter.transfer_verification as verification_module
from tradeexecutor.ethereum.lighter.lighter_routing import LighterRouting
from tradeexecutor.ethereum.lighter.transfer_verification import get_lighter_relayer_payout, verify_lighter_transfer

SAFE = "0xc9129A57299b0Db10b752E27507569D3F37dBBA4"
USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
AMOUNT = Decimal(10)
CLAIM_HASH = "0x" + "11" * 32


def create_withdrawal(**metadata: Any) -> SimpleNamespace:
    """Create the parts of a Lighter withdrawal trade the verification reads."""
    other_data = {
        "lighter_safe_address": SAFE,
        "lighter_withdrawal_from_block": 100,
        "lighter_pending_balance_before_usdc": "0",
        **metadata,
    }
    return SimpleNamespace(
        other_data=other_data,
        planned_quantity=-AMOUNT,
        reserve_currency=SimpleNamespace(address=USDC, chain_id=1),
        is_buy=lambda: False,
        pair=SimpleNamespace(get_exchange_account_protocol=lambda: "lighter"),
        is_internal_account_transfer=lambda: False,
        blockchain_transactions=[SimpleNamespace(tx_hash=CLAIM_HASH)],
    )


@pytest.fixture()
def paid(mocker: MockerFixture) -> Any:
    """Replace the ``WithdrawPending`` read with a controllable answer."""
    mocker.patch.object(verification_module, "fetch_erc20_details")
    return mocker.patch.object(verification_module, "is_lighter_withdrawal_paid", return_value=True)


def test_relayer_payout_uses_the_saved_baseline(paid: Any) -> None:
    """The check reads from the recorded block, and a trade without evidence gets no answer.

    1. Check a withdrawal with a baseline.
    2. Check a withdrawal saved by an earlier release, which has none.
    """
    # 1. Check a withdrawal with a baseline.
    assert get_lighter_relayer_payout(object(), create_withdrawal()) == AMOUNT
    assert paid.call_args.kwargs["from_block"] == 100
    assert paid.call_args.kwargs["amount"] == AMOUNT

    # 2. Check a withdrawal saved by an earlier release, which has none.
    legacy = create_withdrawal()
    del legacy.other_data["lighter_withdrawal_from_block"]
    assert get_lighter_relayer_payout(object(), legacy) is None


def test_verification_accepts_a_payout_only_without_a_successful_claim(paid: Any) -> None:
    """A payout replaces a reverted or missing claim, and never a successful one.

    Our own claim emits the same event, so a successful claim keeps its receipt and
    Safe-credit evidence.

    1. A reverted claim is verified from the relayer's payout.
    2. No claim at all is verified the same way.
    3. A reverted claim with no payout stays a failure.
    4. A successful claim never uses the payout, even though its own event matches.
    """
    # 1. A reverted claim is verified from the relayer's payout.
    verification = verify_lighter_transfer(object(), create_withdrawal(), object(), receipts=[{"status": 0}])
    assert (verification.executed_amount, verification.executed_reserve) == (-AMOUNT, AMOUNT)
    assert verification.metadata_updates["lighter_relayer_paid_usdc"] == "10"

    # 2. No claim at all is verified the same way.
    trade = create_withdrawal(paid_without_transaction=True)
    trade.blockchain_transactions = []
    assert verify_lighter_transfer(object(), trade, object()).executed_reserve == AMOUNT

    # 3. A reverted claim with no payout stays a failure.
    paid.return_value = False
    with pytest.raises(verification_module.LighterTransferVerificationError, match="custody transaction failed"):
        verify_lighter_transfer(object(), create_withdrawal(), object(), receipts=[{"status": 0}])

    # 4. A successful claim never uses the payout, even though its own event matches.
    paid.return_value = True
    paid.reset_mock()
    with pytest.raises(verification_module.LighterTransferVerificationError, match="Safe-credit evidence"):
        verify_lighter_transfer(object(), create_withdrawal(), object(), receipts=[{"status": 1}])
    paid.assert_not_called()


def test_cli_claim_survives_losing_the_race(mocker: MockerFixture) -> None:
    """A revert is accepted only if the relayer paid, and is raised otherwise.

    The broadcast helper raises ``Reverted``, which is not a ``RuntimeError``.

    1. A paid withdrawal sends no claim.
    2. A claim that reverts because the relayer went first is accepted.
    3. A claim that reverts for any other reason is raised.
    """
    # 1. A paid withdrawal sends no claim.
    paid = mocker.patch.object(test_trade_module, "is_lighter_withdrawal_paid", return_value=True)
    claim = mocker.patch.object(test_trade_module, "claim_usdc_to_lagoon_safe_from_lighter", side_effect=Reverted("claim reverted"))
    vault = SimpleNamespace(safe_address=SAFE)
    wallet = SimpleNamespace(sync_nonce=lambda _web3: None)
    journal = {"withdrawal_from_block": 100, "withdrawal_pending_before": "0"}
    assert test_trade_module._claim_unless_paid(object(), wallet, vault, object(), AMOUNT, journal, hypersync_api_key="test-key") is None
    claim.assert_not_called()
    assert paid.call_args.kwargs["hypersync_api_key"] == "test-key"

    # 2. A claim that reverts because the relayer went first is accepted.
    paid.side_effect = [False, True]
    assert test_trade_module._claim_unless_paid(object(), wallet, vault, object(), AMOUNT, journal) is None
    claim.assert_called_once()

    # 3. A claim that reverts for any other reason is raised.
    paid.side_effect = [False, False]
    with pytest.raises(Reverted):
        test_trade_module._claim_unless_paid(object(), wallet, vault, object(), AMOUNT, journal)


def test_reverted_claim_completes_when_the_relayer_paid(paid: Any, mocker: MockerFixture) -> None:
    """Losing the race to the relayer completes the trade instead of failing it.

    1. Settle a reverted claim whose money arrived.
    2. Assert the transfer completed and no failure was reported.
    3. Settle a reverted claim with no payout, which is a real failure.
    """
    # 1. Settle a reverted claim whose money arrived.
    report_failure = mocker.patch.object(routing_module, "report_failure")
    complete = mocker.patch.object(routing_module, "complete_exchange_account_transfer")
    routing = LighterRouting(USDC, session=object())
    state = SimpleNamespace(portfolio=SimpleNamespace(find_position_for_trade=lambda _trade: object()))
    receipts = {CLAIM_HASH: {"status": 0}}
    routing.settle_trade(object(), state, create_withdrawal(), receipts)

    # 2. Assert the transfer completed and no failure was reported.
    assert complete.call_args.kwargs["executed_reserve"] == AMOUNT
    report_failure.assert_not_called()

    # 3. Settle a reverted claim with no payout, which is a real failure.
    complete.reset_mock()
    paid.return_value = False
    routing.settle_trade(object(), state, create_withdrawal(), receipts)
    report_failure.assert_called_once()
    complete.assert_not_called()
