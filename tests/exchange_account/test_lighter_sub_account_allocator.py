"""Pure sub-account rebalance planning for Lighter vaults.

The planner returns one kind of movement per call, sized from observed
balances, so a cycle never depends on another movement in the same cycle.
"""

from decimal import Decimal

import pytest

from tradeexecutor.exchange_account.cash_manager import (
    ExchangeCashManagementError,
    SubAccountBalance,
    plan_sub_account_rebalance,
)

#: Sub-account indices in creation order.
SUB_1, SUB_2 = 2**48 - 1000, 2**48 - 1001


def test_plan_sub_account_rebalance_converges() -> None:
    """Fund, withdraw, cap and converge towards sub-account target weights.

    1. Fund two empty sub-accounts from the master with the fee budgeted on top of each transfer.
    2. Withdraw an overweight sub-account instead of funding the other one in the same plan.
    3. Cap a withdrawal to free collateral when margin is in use.
    4. Return no movement for a converged snapshot or below the minimum.
    """
    # 1. Fund two empty sub-accounts from the master with the fee budgeted on top of each transfer.
    empty = {SUB_1: SubAccountBalance(Decimal(0), Decimal(0)), SUB_2: SubAccountBalance(Decimal(0), Decimal(0))}
    weights = {SUB_1: Decimal("0.5"), SUB_2: Decimal("0.5")}
    plan = plan_sub_account_rebalance(master_free_usdc=Decimal(200), sub_accounts=empty, target_weights=weights, minimum_transfer_usdc=Decimal(1), rebalance_band_usdc=Decimal(0), transfer_fee_usdc=Decimal("0.5"))
    assert plan == [(SUB_1, Decimal(100)), (SUB_2, Decimal(99))]

    # 2. Withdraw an overweight sub-account instead of funding the other one in the same plan.
    funded = {SUB_1: SubAccountBalance(Decimal(100), Decimal(100)), SUB_2: SubAccountBalance(Decimal(99), Decimal(99))}
    plan = plan_sub_account_rebalance(master_free_usdc=Decimal(0), sub_accounts=funded, target_weights={SUB_1: Decimal("0.75"), SUB_2: Decimal("0.25")}, minimum_transfer_usdc=Decimal(1), rebalance_band_usdc=Decimal(0))
    assert plan == [(SUB_2, Decimal("-49.25"))]

    # 3. Cap a withdrawal to free collateral when margin is in use.
    margined = {SUB_1: SubAccountBalance(Decimal(150), Decimal(5)), SUB_2: SubAccountBalance(Decimal(50), Decimal(50))}
    plan = plan_sub_account_rebalance(master_free_usdc=Decimal(0), sub_accounts=margined, target_weights={SUB_1: Decimal("0.25"), SUB_2: Decimal("0.75")}, minimum_transfer_usdc=Decimal(1), rebalance_band_usdc=Decimal(0))
    assert plan == [(SUB_1, Decimal(-5))]

    # 4. Return no movement for a converged snapshot or below the minimum.
    assert plan_sub_account_rebalance(master_free_usdc=Decimal(0), sub_accounts=funded, target_weights={SUB_1: Decimal(100) / Decimal(199), SUB_2: Decimal(99) / Decimal(199)}, minimum_transfer_usdc=Decimal(1), rebalance_band_usdc=Decimal("0.01")) == []
    assert plan_sub_account_rebalance(master_free_usdc=Decimal("0.5"), sub_accounts=empty, target_weights=weights, minimum_transfer_usdc=Decimal(1), rebalance_band_usdc=Decimal(0)) == []


def test_plan_sub_account_rebalance_rejects_invalid_weights() -> None:
    """Refuse weights that would allocate more than the investable capital.

    1. Reject weights summing above one.
    2. Reject a weight for an unknown sub-account.
    """
    balances = {SUB_1: SubAccountBalance(Decimal(0), Decimal(0))}

    # 1. Reject weights summing above one.
    with pytest.raises(ExchangeCashManagementError, match="sum to at most one"):
        plan_sub_account_rebalance(master_free_usdc=Decimal(10), sub_accounts=balances, target_weights={SUB_1: Decimal("1.1")}, minimum_transfer_usdc=Decimal(1), rebalance_band_usdc=Decimal(0))

    # 2. Reject a weight for an unknown sub-account.
    with pytest.raises(ExchangeCashManagementError, match="unknown sub-accounts"):
        plan_sub_account_rebalance(master_free_usdc=Decimal(10), sub_accounts=balances, target_weights={SUB_2: Decimal("0.5")}, minimum_transfer_usdc=Decimal(1), rebalance_band_usdc=Decimal(0))
