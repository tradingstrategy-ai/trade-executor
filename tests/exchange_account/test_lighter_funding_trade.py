"""State accounting for Lighter master to sub-account funding trades.

A funding trade is a buy on the sub-account position paid by the master
account, not by reserves. It executes off-chain, books the master decrease as
a principal outflow plus a fee expense, and blocks the executor while it is
unconfirmed.
"""

from decimal import Decimal
from types import SimpleNamespace

import pytest
from eth_defi.compat import native_datetime_utc_now
from eth_defi.lighter.transfer import LighterTransferRecord
from eth_defi.token import USDC_NATIVE_TOKEN
from pytest_mock import MockerFixture

from tradeexecutor.ethereum.lighter.transfer_verification import (
    LighterTransferVerificationError,
    reconcile_verified_lighter_transfers,
    verify_lighter_funding,
)
from tradeexecutor.exchange_account.lighter import create_lighter_exchange_account_pairs
from tradeexecutor.exchange_account.state import (
    ExchangeAccountTransferError,
    complete_exchange_account_funding,
    create_exchange_account_funding,
    open_exchange_account_position,
)
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State, UncleanState
from tradeexecutor.state.trade import TradeExecution, TradeStatus

#: Master account index.
MASTER_INDEX = 700
#: Funded sub-account index.
SUB_INDEX = 2**48 - 1000
#: Reserve held in the Safe throughout.
SAFE_USDC = Decimal(20)
#: Master collateral before funding.
MASTER_USDC = Decimal(200)


def _create_state() -> tuple[State, AssetIdentifier, TradingPosition, TradingPosition]:
    """Create a reserve, a funded master position and an empty sub-account position."""
    usdc = AssetIdentifier(chain_id=1, address=USDC_NATIVE_TOKEN[1], token_symbol="USDC", decimals=6)
    state = State()
    state.portfolio.initialise_reserves(usdc, reserve_token_price=1.0)
    state.portfolio.get_reserve_position(usdc).quantity = SAFE_USDC + MASTER_USDC
    master_pair, sub_pair = create_lighter_exchange_account_pairs(usdc, MASTER_INDEX, [SUB_INDEX])
    open_exchange_account_position(state=state, strategy_cycle_at=native_datetime_utc_now(), pair=master_pair, reserve_currency=usdc, reserve_amount=MASTER_USDC)
    open_exchange_account_position(state=state, strategy_cycle_at=native_datetime_utc_now(), pair=sub_pair, reserve_currency=usdc, reserve_amount=Decimal(0))
    master, sub = state.portfolio.open_positions.values()
    return state, usdc, master, sub


def _start_funding(state: State, usdc: AssetIdentifier, master: TradingPosition, sub: TradingPosition, amount: Decimal) -> TradeExecution:
    """Create, start and broadcast one funding trade as the router does."""
    trade = create_exchange_account_funding(state=state, source_position=master, destination_position=sub, strategy_cycle_at=native_datetime_utc_now(), reserve_currency=usdc, amount=amount, notes="Fund sub-account")
    state.start_execution(native_datetime_utc_now(), trade)
    trade.other_data["lighter_transfer_tx_hash"] = "0xtransfer"
    trade.mark_broadcasted(native_datetime_utc_now())
    return trade


def test_lighter_funding_trade_moves_capital_between_accounts() -> None:
    """Complete a funding trade without touching reserves or reporting the principal as a loss.

    1. Create and start a funding trade and check it reserves no Safe cash and blocks a restart.
    2. Verify it from transfer-history evidence while equity is unchanged, and complete it with a fee.
    3. Check positions, reserves, balance updates and share prices.
    4. Refuse a second completion and history evidence that names another account.
    """
    state, usdc, master, sub = _create_state()
    reserve = state.portfolio.get_reserve_position(usdc)

    # 1. Create and start a funding trade and check it reserves no Safe cash and blocks a restart.
    trade = _start_funding(state, usdc, master, sub, Decimal(100))
    assert reserve.quantity == SAFE_USDC
    assert trade.is_internal_account_transfer_pending() and trade.is_exchange_account_transfer_pending()
    assert not trade.is_external_account_transfer_pending()
    assert trade.get_execution_sort_position() > 0
    with pytest.raises(UncleanState):
        state.check_if_clean()

    # 2. Verify it from transfer-history evidence while equity is unchanged, and complete it with a fee.
    assert verify_lighter_funding(trade) is None
    trade.other_data.update({"lighter_transfer_from_account": MASTER_INDEX, "lighter_transfer_to_account": SUB_INDEX, "lighter_transfer_amount_usdc": "100", "lighter_transfer_fee_usdc": "0.5"})
    funding = verify_lighter_funding(trade)
    complete_exchange_account_funding(state=state, trade=trade, executed_amount=funding.amount, fee=funding.fee, executed_at=native_datetime_utc_now())

    # 3. Check positions, reserves, balance updates and share prices.
    assert trade.get_status() == TradeStatus.success
    assert (master.get_quantity(), sub.get_quantity(), reserve.quantity) == (Decimal("99.5"), Decimal(100), SAFE_USDC)
    assert [update.quantity for update in master.balance_updates.values()] == [Decimal("-0.5"), Decimal(-100)]
    assert sorted(trade.other_data["funding_source_balance_update_ids"]) == sorted(master.balance_updates)
    assert master.get_share_price_profit().current_share_price == pytest.approx(0.9975)
    assert sub.get_total_profit_usd() == 0
    state.check_if_clean()

    # 4. Refuse a second completion and history evidence that names another account.
    with pytest.raises(ExchangeAccountTransferError):
        complete_exchange_account_funding(state=state, trade=trade, executed_amount=Decimal(100), fee=Decimal("0.5"), executed_at=native_datetime_utc_now(), recovery=True)
    other = _start_funding(state, usdc, master, sub, Decimal(10))
    with pytest.raises(LighterTransferVerificationError, match="accounts differ"):
        verify_lighter_funding(other, LighterTransferRecord("0xtransfer", MASTER_INDEX, SUB_INDEX - 1, Decimal(10), Decimal(0), 1))


def test_lighter_funding_trade_recovery(mocker: MockerFixture) -> None:
    """Keep an unconfirmed funding blocking until history proves it, and release a definitive rejection.

    1. Mark a rejected submission and check it no longer blocks the executor.
    2. Leave a timed-out transfer pending when reconciliation has no operator record.
    3. Reconcile it from transfer history with the operator record and check it is applied once.
    """
    state, usdc, master, sub = _create_state()

    # 1. Mark a rejected submission and check it no longer blocks the executor.
    rejected = create_exchange_account_funding(state=state, source_position=master, destination_position=sub, strategy_cycle_at=native_datetime_utc_now(), reserve_currency=usdc, amount=Decimal(5), notes="Rejected")
    state.start_execution(native_datetime_utc_now(), rejected)
    state.mark_trade_failed(native_datetime_utc_now(), rejected)
    rejected.other_data["outcome"] = "transfer_rejected"
    state.check_if_clean()

    # 2. Leave a timed-out transfer pending when reconciliation has no operator record.
    trade = _start_funding(state, usdc, master, sub, Decimal(40))
    assert reconcile_verified_lighter_transfers(state, web3=None, mutate=True) == []
    assert trade.is_internal_account_transfer_pending()

    # 3. Reconcile it from transfer history with the operator record and check it is applied once.
    # Mocked because authenticated Lighter transfer history needs a live API key.
    record = LighterTransferRecord("0xtransfer", MASTER_INDEX, SUB_INDEX, Decimal(40), Decimal(0), 1)
    fetch = mocker.patch("tradeexecutor.ethereum.lighter.transfer_verification.fetch_lighter_funding_transfer", return_value=record)
    operator = SimpleNamespace(account_index=MASTER_INDEX)
    assert reconcile_verified_lighter_transfers(state, web3=None, mutate=True, operator_record=operator) == [trade]
    assert reconcile_verified_lighter_transfers(state, web3=None, mutate=True, operator_record=operator) == []
    assert fetch.call_count == 1
    assert (master.get_quantity(), sub.get_quantity()) == (MASTER_USDC - 40, Decimal(40))
    state.check_if_clean()
