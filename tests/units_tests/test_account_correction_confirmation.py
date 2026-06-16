"""Tests for interactive accounting correction confirmation handling."""

import datetime
from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from tradingstrategy.chain import ChainId

from tradeexecutor.state.balance_update import BalanceUpdateCause, BalanceUpdatePositionType
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.strategy.account_correction import (
    AccountingBalanceCheck,
    AccountingCorrectionAborted,
    AccountingCorrectionCause,
    correct_accounts,
    is_accounting_correction_confirmation_yes,
)


def test_correct_accounts_accepts_confirmation_with_terminal_whitespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Correct reserve surplus when terminal input contains whitespace.

    1. Create a state with an empty USDC reserve balance.
    2. Build an accounting correction matching a production reserve surplus.
    3. Confirm the interactive prompt with a terminal-style ``y`` response containing whitespace.
    4. Verify the reserve balance correction is applied instead of aborting.
    """
    # 1. Create a state with an empty USDC reserve balance.
    timestamp = datetime.datetime(2026, 6, 16, 9, 27, 31)
    usdc = AssetIdentifier(
        chain_id=ChainId.ethereum.value,
        address="0x0000000000000000000000000000000000000001",
        token_symbol="USDC",
        decimals=6,
    )
    state = State()
    reserve = state.portfolio.initialise_reserves(usdc, reserve_token_price=1.0)
    reserve.quantity = Decimal("0")

    # 2. Build an accounting correction matching a production reserve surplus.
    correction = AccountingBalanceCheck(
        type=AccountingCorrectionCause.unknown_cause,
        holding_address="0x0000000000000000000000000000000000000002",
        asset=usdc,
        positions={reserve},
        expected_amount=Decimal("0"),
        actual_amount=Decimal("55.010596"),
        dust_epsilon=Decimal("0.000001"),
        relative_epsilon=0.0,
        block_number=123,
        timestamp=timestamp,
        usd_value=55.010596,
        reserve_asset=True,
        mismatch=True,
        price=Decimal("1"),
        price_at=timestamp,
    )

    # 3. Confirm the interactive prompt with a terminal-style ``y`` response containing whitespace.
    monkeypatch.setattr("builtins.input", lambda prompt: " y\r\n")

    balance_updates = correct_accounts(
        state,
        [correction],
        strategy_cycle_included_at=None,
        tx_builder=MagicMock(),
        interactive=True,
    )

    # 4. Verify the reserve balance correction is applied instead of aborting.
    balance_updates = list(balance_updates)
    assert len(balance_updates) == 1
    update = balance_updates[0]
    assert update.position_type == BalanceUpdatePositionType.reserve
    assert update.cause == BalanceUpdateCause.correction
    assert update.quantity == Decimal("55.010596")
    assert reserve.quantity == Decimal("55.010596")
    assert state.sync.accounting.last_block_scanned == 123


def test_correct_accounts_rejects_negative_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Abort correction when the user does not explicitly confirm.

    1. Create a minimal reserve correction.
    2. Answer the interactive prompt with ``n``.
    3. Verify the correction is aborted before mutating reserves.
    """
    # 1. Create a minimal reserve correction.
    timestamp = datetime.datetime(2026, 6, 16, 9, 27, 31)
    usdc = AssetIdentifier(
        chain_id=ChainId.ethereum.value,
        address="0x0000000000000000000000000000000000000001",
        token_symbol="USDC",
        decimals=6,
    )
    state = State()
    reserve = state.portfolio.initialise_reserves(usdc, reserve_token_price=1.0)
    correction = AccountingBalanceCheck(
        type=AccountingCorrectionCause.unknown_cause,
        holding_address="0x0000000000000000000000000000000000000002",
        asset=usdc,
        positions={reserve},
        expected_amount=Decimal("0"),
        actual_amount=Decimal("55.010596"),
        dust_epsilon=Decimal("0.000001"),
        relative_epsilon=0.0,
        block_number=123,
        timestamp=timestamp,
        usd_value=55.010596,
        reserve_asset=True,
        mismatch=True,
        price=Decimal("1"),
        price_at=timestamp,
    )

    # 2. Answer the interactive prompt with ``n``.
    monkeypatch.setattr("builtins.input", lambda prompt: "n")

    # 3. Verify the correction is aborted before mutating reserves.
    with pytest.raises(AccountingCorrectionAborted):
        list(
            correct_accounts(
                state,
                [correction],
                strategy_cycle_included_at=None,
                tx_builder=MagicMock(),
                interactive=True,
            )
        )
    assert reserve.quantity == Decimal("0")


def test_accounting_correction_confirmation_accepts_yes_variants() -> None:
    """Accept common affirmative terminal responses.

    1. Pass short and long affirmative responses with case and whitespace variants.
    2. Verify all affirmative responses are accepted.
    3. Verify a negative response is not accepted.
    """
    # 1. Pass short and long affirmative responses with case and whitespace variants.
    responses = ["y", "Y", " y\r\n", "yes", "YES", " yes\n"]

    # 2. Verify all affirmative responses are accepted.
    assert all(is_accounting_correction_confirmation_yes(response) for response in responses)

    # 3. Verify a negative response is not accepted.
    assert not is_accounting_correction_confirmation_yes("n")
