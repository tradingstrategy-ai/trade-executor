"""Fixed sleeve minimums through the normal Lighter cash-policy entry point."""

from decimal import Decimal
from types import SimpleNamespace

import pytest
from eth_defi.token import USDC_NATIVE_TOKEN
from pytest_mock import MockerFixture

from tradeexecutor.exchange_account.lighter import create_lighter_exchange_account_pairs
from tradeexecutor.exchange_account.lighter_strategy import decide_lighter_cash_management
from tradeexecutor.exchange_account.sub_account import SubAccount
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.strategy.parameters import StrategyParameters


def test_fixed_lighter_cash_management_keeps_surplus_in_safe(mocker: MockerFixture) -> None:
    """Leave unallocated USDC in the Safe while funding fixed sleeve minimums.

    1. Mock exchange balances and custody submissions, retaining the real planner.
    2. Cap deposits by available Safe cash and the 9,000-USDC funding need.
    3. Fund sleeves before sweeping surplus, retaining profits and unused Safe cash.
    4. Return excess master cash, avoiding churn and preserving its configured buffer.
    5. Preserve redemption priority and the pending-transfer gate.
    """
    # 1. Mock exchange balances and custody submissions, retaining the real planner.
    # Existing lifecycle tests cover position initialisation and real custody execution.
    mocker.patch("tradeexecutor.exchange_account.lighter_strategy.ensure_lighter_positions", return_value=False)
    manager = mocker.Mock()
    manager.is_lighter_transfer_pending.return_value = False
    manager.get_current_cash.return_value = 12000
    balances = {"master": Decimal(0), "vol": Decimal(0), "long-short": Decimal(0), "btc": Decimal(0), "momentum": Decimal(0)}
    manager.get_lighter_free_collateral.side_effect = lambda account_id="master": balances[account_id]
    manager.get_lighter_balance.side_effect = lambda account_id: balances[account_id]
    # Only trade construction is replaced: record the real policy's selected amounts.
    manager.deposit_to_lighter.side_effect = lambda amount, **kwargs: SimpleNamespace(other_data={}, amount=amount)
    manager.withdraw_from_lighter.side_effect = lambda account, amount, **kwargs: SimpleNamespace(other_data={}, account=account, amount=amount)
    manager.transfer_lighter_funds.side_effect = lambda source, account, amount, **kwargs: SimpleNamespace(other_data={}, account=account, amount=amount)
    targets = {"vol": Decimal(1000), "long-short": Decimal(6000), "btc": Decimal(2000), "momentum": Decimal(0)}
    usdc = AssetIdentifier(chain_id=1, address=USDC_NATIVE_TOKEN[1], token_symbol="USDC", decimals=6)
    pairs = create_lighter_exchange_account_pairs(usdc, 700, {2**48 - i: account for i, account in enumerate(targets, 1)})
    state = State()
    parameters = StrategyParameters({
        "lighter_sub_accounts": [SubAccount(account, account, minimum_usd=amount) for account, amount in targets.items()],
        "lighter_safe_cash_buffer_usd": Decimal(0),
        "lighter_free_collateral_buffer_usd": Decimal(0),
        "lighter_rebalance_band_usd": Decimal(250),
        "lighter_keep_surplus_in_safe": True,
    })
    input = SimpleNamespace(
        execution_context=SimpleNamespace(mode=SimpleNamespace(is_backtesting=lambda: False)),
        get_position_manager=lambda: manager,
        strategy_universe=SimpleNamespace(iterate_pairs=lambda: iter(pairs)),
        parameters=parameters,
        state=state,
    )

    # 2. Cap deposits by available Safe cash and the 9,000-USDC funding need.
    manager.get_current_cash.return_value = 500
    assert decide_lighter_cash_management(input)[0].amount == Decimal(500)
    manager.get_current_cash.return_value = 12000
    trades = decide_lighter_cash_management(input)
    assert [(trade.other_data["policy"], trade.amount) for trade in trades] == [("idle_safe_cash", Decimal(9000))]

    # 3. Fund sleeves before sweeping surplus, retaining profits and unused Safe cash.
    manager.get_current_cash.return_value = 3000
    balances["master"] = Decimal(12000)
    trades = decide_lighter_cash_management(input)
    assert [(trade.account, trade.amount) for trade in trades] == [("long-short", Decimal(6000)), ("btc", Decimal(2000)), ("vol", Decimal(1000))]
    balances.update(targets, master=Decimal(0))
    assert decide_lighter_cash_management(input) == []
    balances["vol"] = Decimal(800)
    assert decide_lighter_cash_management(input) == []
    balances["vol"] = Decimal(1500)
    assert decide_lighter_cash_management(input) == []
    manager.withdraw_from_lighter.assert_not_called()
    balances["vol"] = targets["vol"]
    assert manager.deposit_to_lighter.call_count == 2

    # 4. Return excess master cash, avoiding churn and preserving its configured buffer.
    balances["master"] = Decimal(3000)
    trades = decide_lighter_cash_management(input)
    assert [(trade.account, trade.amount, trade.other_data["policy"]) for trade in trades] == [("master", Decimal(3000), "unallocated_cash")]
    balances["master"] = Decimal(260)
    balances["vol"] = Decimal(760)
    # Price drift moved a previously fundable shortfall inside the band: retain its cash.
    assert decide_lighter_cash_management(input) == []
    balances["vol"] = targets["vol"]
    balances["master"] = Decimal(250)
    assert decide_lighter_cash_management(input) == []
    parameters.lighter_free_collateral_buffer_usd = Decimal(100)
    balances["master"] = Decimal(50)
    balances["vol"] = Decimal(0)
    assert decide_lighter_cash_management(input)[0].amount == Decimal(1050)
    parameters.lighter_free_collateral_buffer_usd = Decimal(0)
    balances["vol"] = targets["vol"]

    # 5. Preserve redemption priority and the pending-transfer gate.
    balances["master"] = Decimal(0)
    manager.get_current_cash.return_value = 0
    state.sync.treasury.pending_redemptions = 100
    trades = decide_lighter_cash_management(input)
    assert [(trade.account, trade.amount, trade.other_data["policy"]) for trade in trades] == [("long-short", Decimal(100), "safe_liquidity")]
    manager.is_lighter_transfer_pending.return_value = True
    assert decide_lighter_cash_management(input) == []


def test_lighter_surplus_cash_requires_fixed_minimums(mocker: MockerFixture) -> None:
    """Reject an incompatible cash policy before moving any Safe capital.

    1. Provide an existing master position with idle Safe cash and weighted targets.
    2. Request Safe surplus retention and verify refusal before a deposit.
    """
    # 1. Provide an existing master position with idle Safe cash and weighted targets.
    # Position creation and exchange reads are covered by existing lifecycle tests.
    mocker.patch("tradeexecutor.exchange_account.lighter_strategy.ensure_lighter_positions", return_value=False)
    manager = mocker.Mock()
    manager.is_lighter_transfer_pending.return_value = False
    manager.get_lighter_free_collateral.return_value = Decimal(0)
    manager.get_lighter_balance.return_value = Decimal(0)
    usdc = AssetIdentifier(chain_id=1, address=USDC_NATIVE_TOKEN[1], token_symbol="USDC", decimals=6)
    pairs = create_lighter_exchange_account_pairs(usdc, 700, {2**48 - 1: "vol"})
    input = SimpleNamespace(
        execution_context=SimpleNamespace(mode=SimpleNamespace(is_backtesting=lambda: False)),
        get_position_manager=lambda: manager,
        strategy_universe=SimpleNamespace(iterate_pairs=lambda: iter(pairs)),
        parameters=StrategyParameters({"lighter_sub_accounts": [SubAccount("vol", "Volatility", weight=1)], "lighter_free_collateral_buffer_usd": Decimal(0), "lighter_keep_surplus_in_safe": True}),
    )

    # 2. Request Safe surplus retention and verify refusal before a deposit.
    with pytest.raises(ValueError, match="requires fixed sub-account minimums"):
        decide_lighter_cash_management(input)
    manager.deposit_to_lighter.assert_not_called()
