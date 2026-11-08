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
from tradeexecutor.exchange_account.lighter_strategy import get_sub_account_targets
from tradeexecutor.exchange_account.sub_account import SubAccount

#: Sub-account indices in creation order.
SUB_1, SUB_2 = "trend", "arb"


def test_plan_sub_account_rebalance_converges() -> None:
    """Fund, withdraw, cap and converge towards sub-account target weights.

    The planner decides every sub-account movement, so it must size each one
    from observed balances, respect margin in use and stop once converged.

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

    Invalid weights would plan transfers the accounts cannot cover.

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


def test_fixed_sub_account_minimums_preserve_cash_and_profits() -> None:
    """Fund fixed floors without trimming profitable sleeves or scaling to NAV.

    1. Fund fixed minimums and leave surplus on the master.
    2. Keep absolute targets when cash is short, funding the largest deficit first.
    3. Leave above-minimum equity in the sleeve, funding only other shortfalls.
    4. Leave balances within the rebalance band and converged targets alone.
    """
    targets = {"vol": Decimal(1000), "long-short": Decimal(6000), "btc": Decimal(2000), "momentum": Decimal(0)}
    empty = {account_id: SubAccountBalance(Decimal(0), Decimal(0)) for account_id in targets}
    common = dict(minimum_amounts_usdc=targets, minimum_transfer_usdc=Decimal(1), rebalance_band_usdc=Decimal(250))

    # 1. Fund fixed minimums and leave surplus on the master.
    plan = plan_sub_account_rebalance(master_free_usdc=Decimal(12000), sub_accounts=empty, **common)
    assert plan == [("long-short", Decimal(6000)), ("btc", Decimal(2000)), ("vol", Decimal(1000))]
    assert Decimal(12000) - sum(amount for _, amount in plan) == Decimal(3000)

    # 2. Keep absolute targets when cash is short, funding the largest deficit first.
    current = dict(empty, vol=SubAccountBalance(Decimal(200), Decimal(200)), **{"long-short": SubAccountBalance(Decimal(5850), Decimal(5270))})
    assert plan_sub_account_rebalance(master_free_usdc=Decimal(0), sub_accounts=current, **common) == []
    assert plan_sub_account_rebalance(master_free_usdc=Decimal(1000), sub_accounts=current, **common) == [("btc", Decimal(1000))]
    assert plan_sub_account_rebalance(master_free_usdc=Decimal(1), sub_accounts=current, transfer_fee_usdc=Decimal("0.5"), **common) == []

    # 3. Leave above-minimum equity in the sleeve, funding only other shortfalls.
    current["long-short"] = SubAccountBalance(Decimal(6500), Decimal(75))
    current["momentum"] = SubAccountBalance(Decimal(300), Decimal(300))
    assert plan_sub_account_rebalance(master_free_usdc=Decimal(3000), sub_accounts=current, **common) == [("btc", Decimal(2000)), ("vol", Decimal(800))]

    # 4. Leave balances within the rebalance band and converged targets alone.
    converged = {account_id: SubAccountBalance(amount, amount) for account_id, amount in targets.items()}
    assert plan_sub_account_rebalance(master_free_usdc=Decimal(3000), sub_accounts=converged, **common) == []
    converged["vol"] = SubAccountBalance(Decimal(800), Decimal(800))
    assert plan_sub_account_rebalance(master_free_usdc=Decimal(3000), sub_accounts=converged, **common) == []


def test_fixed_sub_account_target_validation() -> None:
    """Reject ambiguous budgets before they can cause custody movements.

    1. Resolve fixed minimums, including zero, without managing untargeted accounts.
    2. Reject negative, non-finite and incorrectly typed amounts or two target modes.
    3. Reject unknown accounts and strategies mixing fixed minimums with weights.
    4. Reject invalid planner inputs independently of strategy declarations.
    """
    # 1. Resolve fixed minimums, including zero, without managing untargeted accounts.
    accounts = [SubAccount("vol", "Volatility", minimum_usd=Decimal(1000)), SubAccount("empty", "Empty", minimum_usd=Decimal(0)), SubAccount("unmanaged", "Unmanaged")]
    assert get_sub_account_targets(accounts, ["vol", "empty", "unmanaged"]) == ({}, {"vol": Decimal(1000), "empty": Decimal(0)})
    assert get_sub_account_targets([SubAccount("vol", "Volatility", weight=0.5)], ["vol"]) == ({"vol": Decimal("0.5")}, {})

    # 2. Reject negative, non-finite and incorrectly typed amounts or two target modes.
    for amount in [Decimal(-1), Decimal("NaN"), Decimal("Infinity"), 1000]:
        with pytest.raises(ValueError, match="finite non-negative Decimal"):
            SubAccount("vol", "Volatility", minimum_usd=amount)
    with pytest.raises(ValueError, match="both a weight and a fixed minimum"):
        SubAccount("vol", "Volatility", weight=0, minimum_usd=Decimal(1000))

    # 3. Reject unknown accounts and strategies mixing fixed minimums with weights.
    with pytest.raises(ValueError, match="not in the strategy universe"):
        get_sub_account_targets(accounts, ["empty"])
    with pytest.raises(ValueError, match="either weights or fixed minimums"):
        get_sub_account_targets([accounts[0], SubAccount("weighted", "Weighted", weight=1)], ["vol", "weighted"])

    # 4. Reject invalid planner inputs independently of strategy declarations.
    common = dict(master_free_usdc=Decimal(10), sub_accounts={"vol": SubAccountBalance(Decimal(0), Decimal(0))}, minimum_transfer_usdc=Decimal(1), rebalance_band_usdc=Decimal(0))
    with pytest.raises(ExchangeCashManagementError, match="either target weights or fixed minimum amounts"):
        plan_sub_account_rebalance(target_weights={"vol": Decimal(1)}, minimum_amounts_usdc={"vol": Decimal(1000)}, **common)
    for amount in [Decimal(-1), Decimal("NaN"), Decimal("Infinity")]:
        with pytest.raises(ExchangeCashManagementError, match="finite non-negative Decimal"):
            plan_sub_account_rebalance(minimum_amounts_usdc={"vol": amount}, **common)
    with pytest.raises(ExchangeCashManagementError, match="unknown sub-accounts"):
        plan_sub_account_rebalance(minimum_amounts_usdc={"unknown": Decimal(1000)}, **common)
