"""CLI coverage for lagoon-redeem pre-flight deposit and redemption claiming."""

from decimal import Decimal
import datetime
from unittest.mock import MagicMock, patch

import pytest

from tradeexecutor.cli.commands.lagoon_redeem import (
    _claim_leftover_deposits,
    _claim_leftover_redemptions,
    _get_redemption_nav,
    _check_drained_pool_state,
)
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeStatus, TradeType


@pytest.mark.parametrize("status", [TradeStatus.started, TradeStatus.broadcasted])
def test_pool_redemption_refuses_unbooked_submissions(status: TradeStatus) -> None:
    """Terminal redemption cannot treat an unbooked submission as an empty portfolio.

    1. Create a non-pool buy with zero held shares and no saved capital allocation.
    2. Refuse its unresolved status even though recorded equity is zero.
    3. Permit terminal failed history, but refuse a successful holding with a zero price.

    Missing allocation represents incomplete historical bookkeeping; no venue
    is mocked because this check must refuse before any external transaction.
    """
    # 1. Create a non-pool buy with zero held shares and no saved capital allocation.
    state = State()
    ts = datetime.datetime(2026, 11, 4)
    usdc = AssetIdentifier(1, "0x" + "11" * 20, "USDC", 6)
    share = AssetIdentifier(1, "0x" + "22" * 20, "SHARE", 18)
    pair = TradingPairIdentifier(share, usdc, "0x" + "33" * 20, "0x" + "44" * 20)
    position, trade, _ = state.create_trade(ts, pair, Decimal(1), None, 1.0, TradeType.rebalance, usdc, 1.0)
    trade.started_at = ts
    if status == TradeStatus.broadcasted:
        trade.broadcasted_at = ts

    # 2. Refuse its unresolved status even though recorded equity is zero.
    assert position.get_quantity() == 0 and position.get_equity() == 0
    with pytest.raises(RuntimeError, match="reconcile submissions"):
        _check_drained_pool_state(state)

    # 3. Permit terminal failed history, but refuse a successful holding with a zero price.
    trade.mark_failed(ts)
    _check_drained_pool_state(state)
    trade.failed_at = None
    trade.mark_success(ts, 1.0, Decimal(1), Decimal(1), 0, None, force=True)
    position.last_token_price = 0.0
    assert position.get_equity() == 0
    with pytest.raises(RuntimeError, match="Close all positions"):
        _check_drained_pool_state(state)


@pytest.mark.timeout(30)
def test_claim_leftover_redemptions() -> None:
    """Pre-flight handles both settled-unclaimed and pending-unsettled leftover redemptions.

    1. Claim settled shares, then ignore an obsolete request when the global queue is empty.
    2. Call with maxRedeem() == 0 and pendingRedeemRequest() > 0 (state B): verify settle + poll + finalise.
    3. Resume after an already completed claim: ignore its obsolete request and broadcast nothing.
    4. Derive a drained pool deployment's terminal NAV from actual Safe cash.
    """
    tx_hash = b"\x00" * 32

    def make_mocks():
        vault = MagicMock()
        vault.finalise_redeem.return_value = "mock_finalise"
        vault.post_new_valuation.return_value = "mock_post_val"
        vault.settle_via_trading_strategy_module.return_value = "mock_settle"
        vault.fetch_nav.return_value = Decimal("10000")
        vault.get_flow_manager.return_value.fetch_pending_redemption.return_value = Decimal(0)
        hot_wallet = MagicMock()
        hot_wallet.address = "0xABCD"
        hot_wallet.transact_and_broadcast_with_contract.return_value = tx_hash
        web3 = MagicMock()
        share_token = MagicMock()
        share_token.symbol = "lagVault"
        share_token.convert_to_decimals.return_value = Decimal("500")
        return vault, hot_wallet, web3, share_token

    # 1. Claim settled shares, then ignore an obsolete request when the global queue is empty.
    vault, hot_wallet, web3, share_token = make_mocks()
    settled_raw = 500_000_000
    vault.vault_contract.functions.maxRedeem.return_value.call.return_value = settled_raw
    vault.vault_contract.functions.pendingRedeemRequest.return_value.call.return_value = settled_raw
    vault.get_flow_manager.return_value.fetch_pending_redemption.return_value = Decimal(0)

    with patch("tradeexecutor.cli.commands.lagoon_redeem._broadcast_and_wait"):
        assert _claim_leftover_redemptions(vault, hot_wallet, web3, share_token, None)

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
    vault.get_flow_manager.return_value.fetch_pending_redemption.return_value = Decimal(750)

    with (
        patch("tradeexecutor.cli.commands.lagoon_redeem._broadcast_and_wait"),
        patch("tradeexecutor.cli.commands.lagoon_redeem.time.sleep") as mock_sleep,
    ):
        assert _claim_leftover_redemptions(vault, hot_wallet, web3, share_token, None)
        mock_sleep.assert_called_once_with(5)

    vault.post_new_valuation.assert_called_once()
    vault.settle_via_trading_strategy_module.assert_called_once()
    vault.finalise_redeem.assert_called_once_with("0xABCD", raw_amount=pending_raw)

    # 3. Resume after an already completed claim: ignore its obsolete request and broadcast nothing.
    vault, hot_wallet, web3, share_token = make_mocks()
    vault.vault_contract.functions.maxRedeem.return_value.call.return_value = 0
    vault.vault_contract.functions.pendingRedeemRequest.return_value.call.return_value = settled_raw

    assert not _claim_leftover_redemptions(vault, hot_wallet, web3, share_token, None)

    vault.finalise_redeem.assert_not_called()
    hot_wallet.transact_and_broadcast_with_contract.assert_not_called()

    # 4. A verified drained pool uses actual Safe cash, never the exchange-account
    # dust rule or a previous on-chain NAV. The fork test covers the preflight.
    assert _get_redemption_nav(vault, Decimal("19.99"), [127], pool_account_drained=True) == Decimal("19.99")
    vault.fetch_nav.assert_not_called()


@pytest.mark.timeout(30)
def test_get_redemption_nav_ignores_small_lighter_balance() -> None:
    """Small Lighter equity does not prevent a full Safe-funded redemption.

    1. Read a normal Lagoon NAV that includes one USDC left in Lighter.
    2. Read the public Lighter equity for the configured account.
    3. Verify the Safe balance is posted when the equity is below five USDC.
    """
    # 1. Read a normal Lagoon NAV that includes one USDC left in Lighter.
    vault = MagicMock()
    vault.fetch_nav.return_value = Decimal("21")

    # 2. Read the public Lighter equity for the configured account.
    equity = MagicMock()
    equity.get_total.return_value = Decimal("1")

    with (
        patch("tradeexecutor.cli.commands.lagoon_redeem.create_lighter_session") as mock_session,
        patch("tradeexecutor.cli.commands.lagoon_redeem.fetch_lighter_total_equity", return_value=equity) as mock_equity,
    ):
        # 3. Verify the Safe balance is posted when the equity is below five USDC.
        assert _get_redemption_nav(vault, Decimal("20"), [747918]) == Decimal("20")
        mock_session.assert_called_once_with()
        mock_equity.assert_called_once_with(mock_session.return_value.__enter__.return_value, 747918)


@pytest.mark.timeout(30)
def test_get_redemption_nav_keeps_material_lighter_balance() -> None:
    """Material Lighter equity across a master and a sub-account remains in the settlement NAV.

    The threshold applies to the combined equity of all accounts, so two
    accounts that are each below it can still hold material capital.

    1. Read a normal Lagoon NAV that includes Lighter equity.
    2. Read public Lighter equity of 2.5 USDC on each of two accounts, together equal to the five-USDC threshold.
    3. Verify the normal NAV remains unchanged because only lower combined values are ignored.
    """
    # 1. Read a normal Lagoon NAV that includes Lighter equity.
    vault = MagicMock()
    vault.fetch_nav.return_value = Decimal("25")

    # 2. Read public Lighter equity of 2.5 USDC on each of two accounts, together equal to the five-USDC threshold.
    equity = MagicMock()
    equity.get_total.return_value = Decimal("2.5")

    with (
        patch("tradeexecutor.cli.commands.lagoon_redeem.create_lighter_session"),
        patch("tradeexecutor.cli.commands.lagoon_redeem.fetch_lighter_total_equity", return_value=equity) as mock_equity,
    ):
        # 3. Verify the normal NAV remains unchanged because only lower combined values are ignored.
        assert _get_redemption_nav(vault, Decimal("20"), [747918, 2**47 + 1]) == Decimal("25")
        assert mock_equity.call_count == 2


@pytest.mark.timeout(30)
def test_claim_leftover_deposits() -> None:
    """Pre-flight claims unclaimed deposits so shares move from vault contract to hot wallet.

    1. Call with maxDeposit() > 0: verify 3-arg deposit() called via _broadcast_and_wait.
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

    # 1. Unclaimed deposit: maxDeposit > 0 — 3-arg deposit() called
    vault, hot_wallet, web3 = make_mocks()
    deposit_raw = 10_000_000
    vault.vault_contract.functions.maxDeposit.return_value.call.return_value = deposit_raw

    with patch("tradeexecutor.cli.commands.lagoon_redeem._broadcast_and_wait") as mock_bw:
        _claim_leftover_deposits(vault, hot_wallet, web3)

    vault.vault_contract.functions.deposit.assert_called_once_with(deposit_raw, "0xABCD", "0xABCD")
    mock_bw.assert_called_once()

    # 2. No unclaimed deposits: maxDeposit == 0 — no transactions
    vault, hot_wallet, web3 = make_mocks()
    vault.vault_contract.functions.maxDeposit.return_value.call.return_value = 0

    _claim_leftover_deposits(vault, hot_wallet, web3)

    hot_wallet.transact_and_broadcast_with_contract.assert_not_called()
