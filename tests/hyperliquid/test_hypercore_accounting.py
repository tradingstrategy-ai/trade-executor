"""Test Hypercore vault position accounting consistency.

Verifies that:
1. Deposit records the USDC delta (not total equity) as executed_amount.
2. Valuation computes per-unit price (equity/quantity) so value = equity.
3. A second deposit adds only the delta to position quantity.
4. Valuation stays 1:1 with equity after multiple deposits.
"""

import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from eth_defi.compat import native_datetime_utc_now
from tradingstrategy.chain import ChainId

from tradeexecutor.ethereum.vault.hypercore_valuation import HypercoreVaultPricing, HypercoreVaultValuator
from tradeexecutor.ethereum.vault.hypercore_vault import create_hypercore_vault_pair
from tradeexecutor.cli.double_position import check_double_position
from tradeexecutor.state.balance_update import (
    BalanceUpdate,
    BalanceUpdateCause,
    BalanceUpdatePositionType,
)
from tradeexecutor.state.identifier import TradingPairIdentifier, TradingPairKind, AssetIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.repair import (
    close_hypercore_dust_positions,
)
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeFlag, TradeType
from tradeexecutor.strategy.account_correction import (
    UnexpectedAccountingCorrectionIssue,
    _build_hypercore_vault_account_checks,
)
from tradeexecutor.strategy.dust import (
    get_close_epsilon_for_pair,
    get_dust_epsilon_for_pair,
    HYPERLIQUID_VAULT_CLOSE_EPSILON,
    HYPERLIQUID_VAULT_RELATIVE_EPSILON,
    DEFAULT_VAULT_EPSILON,
)
from tradeexecutor.strategy.execution_context import unit_test_execution_context
from tradeexecutor.strategy.runner import StrategyRunner
from tradeexecutor.strategy.sync_model import OnChainBalance


def test_valuation_computes_per_unit_price():
    """Value = quantity * (equity / quantity) = equity."""

    # Position with quantity=100 (deposited 100 USDC), price=1.0
    position = MagicMock()
    position.is_vault.return_value = True
    position.get_quantity.return_value = Decimal("100.0")
    position.last_token_price = 1.0

    # API returns equity = 105 (5% gain)
    def value_func(pair):
        return Decimal("105.0")

    valuator = HypercoreVaultValuator(value_func=value_func, simulate=False)
    ts = native_datetime_utc_now()
    result = valuator(ts, position)

    # Per-unit price should be 105/100 = 1.05
    assert position.revalue_base_asset.call_count == 1
    call_args = position.revalue_base_asset.call_args
    new_price = call_args[0][1]
    assert pytest.approx(new_price, rel=1e-6) == 1.05


def test_valuation_second_deposit_stays_correct():
    """After depositing 100 then 50, equity=155 → value should be 155."""

    position = MagicMock()
    position.is_vault.return_value = True
    # After two deposits: 100 + 50 = 150 quantity
    position.get_quantity.return_value = Decimal("150.0")
    position.last_token_price = 1.0

    # API returns equity = 155 (vault grew from 150 to 155)
    def value_func(pair):
        return Decimal("155.0")

    valuator = HypercoreVaultValuator(value_func=value_func, simulate=False)
    ts = native_datetime_utc_now()
    result = valuator(ts, position)

    # Per-unit price: 155/150 = 1.0333...
    call_args = position.revalue_base_asset.call_args
    new_price = call_args[0][1]
    assert pytest.approx(new_price, rel=1e-4) == 155.0 / 150.0


def test_valuation_zero_quantity_uses_default_price():
    """Edge case: quantity=0 should use price=1.0 to avoid division by zero."""

    position = MagicMock()
    position.is_vault.return_value = True
    position.get_quantity.return_value = Decimal("0")
    position.last_token_price = 1.0

    def value_func(pair):
        return Decimal("0")

    valuator = HypercoreVaultValuator(value_func=value_func, simulate=False)
    ts = native_datetime_utc_now()
    result = valuator(ts, position)

    call_args = position.revalue_base_asset.call_args
    new_price = call_args[0][1]
    assert new_price == 1.0


def test_pricing_model_returns_one():
    """Trade pricing for vault deposits/withdrawals is always 1.0 USDC."""

    # Even with non-zero equity, trade price should be 1.0
    def value_func(pair):
        return Decimal("500.0")

    pricing = HypercoreVaultPricing(value_func=value_func, simulate=False)

    base = AssetIdentifier(chain_id=999, address="0x0000000000000000000000000000000000000001", token_symbol="VAULT", decimals=6)
    quote = AssetIdentifier(chain_id=999, address="0x0000000000000000000000000000000000000002", token_symbol="USDC", decimals=6)
    pair = TradingPairIdentifier(
        base=base, quote=quote,
        pool_address="0x0000000000000000000000000000000000000003", exchange_address="0x0000000000000000000000000000000000000004",
        internal_id=1, internal_exchange_id=1,
        fee=0.0, kind=TradingPairKind.vault,
    )

    buy_pricing = pricing.get_buy_price(None, pair, Decimal("100"))
    assert buy_pricing.price == 1.0

    sell_pricing = pricing.get_sell_price(None, pair, Decimal("50"))
    assert sell_pricing.price == 1.0

    mid = pricing.get_mid_price(None, pair)
    assert mid == 1.0


def test_lockup_func_populates_expires_at():
    """Valuator with lockup_func stores ISO timestamp in position.other_data.

    1. Create a mock lockup func returning a fixed datetime
    2. Run the valuator
    3. Verify other_data contains the ISO string
    """
    import datetime as dt

    position = MagicMock()
    position.is_vault.return_value = True
    position.get_quantity.return_value = Decimal("100.0")
    position.last_token_price = 1.0
    position.other_data = {}

    expires = dt.datetime(2026, 3, 27, 14, 30, 0)

    def value_func(pair):
        return Decimal("105.0")

    def lockup_func(pair):
        return expires

    valuator = HypercoreVaultValuator(value_func=value_func, lockup_func=lockup_func)
    ts = native_datetime_utc_now()
    valuator(ts, position)

    assert position.other_data["vault_lockup_expires_at"] == "2026-03-27T14:30:00"


def test_lockup_func_none_position():
    """Valuator with lockup_func stores None when no vault position found.

    1. Create a mock lockup func returning None (no position)
    2. Run the valuator
    3. Verify other_data contains None
    """

    position = MagicMock()
    position.is_vault.return_value = True
    position.get_quantity.return_value = Decimal("100.0")
    position.last_token_price = 1.0
    position.other_data = {}

    def value_func(pair):
        return Decimal("105.0")

    def lockup_func(pair):
        return None

    valuator = HypercoreVaultValuator(value_func=value_func, lockup_func=lockup_func)
    ts = native_datetime_utc_now()
    valuator(ts, position)

    assert position.other_data["vault_lockup_expires_at"] is None


def test_old_bug_equity_squared():
    """Regression: the old code set quantity=equity and price=equity → value=equity².

    With the fix, quantity=deposited_amount and price=equity/quantity → value=equity.
    This test verifies the specific scenario from the review: a 5 USDC deposit
    with API equity=5.5 should value at 5.5, not 27.5.
    """

    position = MagicMock()
    position.is_vault.return_value = True
    # Deposited 5 USDC (quantity tracks USDC deposited)
    position.get_quantity.return_value = Decimal("5.0")
    position.last_token_price = 1.0

    # API says equity is 5.5
    def value_func(pair):
        return Decimal("5.5")

    valuator = HypercoreVaultValuator(value_func=value_func, simulate=False)
    ts = native_datetime_utc_now()
    result = valuator(ts, position)

    call_args = position.revalue_base_asset.call_args
    new_price = call_args[0][1]
    # Per-unit price: 5.5 / 5.0 = 1.1
    assert pytest.approx(new_price, rel=1e-6) == 1.1
    # Value would be: 5.0 * 1.1 = 5.5 (correct), NOT 5.0 * 5.5 = 27.5 (old bug)


def test_hypercore_vault_dust_epsilon_covers_safety_margin():
    """Hypercore vault close epsilon is large enough to cover withdrawal safety margin dust.

    HYPERCORE_WITHDRAWAL_SAFETY_MARGIN_RAW (100_000 raw = $0.10) is subtracted
    from live vault equity during full-close withdrawals, leaving ~$0.10 residual
    in the position.  The close epsilon must exceed this so can_be_closed()
    recognises the position as effectively closed.

    1. Build a Hypercore vault pair using create_hypercore_vault_pair().
    2. Verify get_close_epsilon_for_pair() and get_dust_epsilon_for_pair()
       return the Hypercore-specific epsilon.
    3. Create a TradingPosition with dust quantity (0.01) and assert can_be_closed().
    4. Same position with non-dust quantity (1.0) must NOT be closeable.
    5. Build a non-Hypercore vault pair and verify it still gets DEFAULT_VAULT_EPSILON.
    """

    # 1. Build a Hypercore vault pair
    quote = AssetIdentifier(
        chain_id=ChainId.hypercore.value,
        address="0x0000000000000000000000000000000000000002",
        token_symbol="USDC",
        decimals=6,
    )
    hypercore_pair = create_hypercore_vault_pair(
        quote=quote,
        vault_address="0x1111111111111111111111111111111111111111",
    )
    assert hypercore_pair.is_hyperliquid_vault()

    # 2. Epsilon functions return Hypercore-specific value
    assert get_close_epsilon_for_pair(hypercore_pair) == HYPERLIQUID_VAULT_CLOSE_EPSILON
    assert get_dust_epsilon_for_pair(hypercore_pair) == HYPERLIQUID_VAULT_CLOSE_EPSILON

    # 3. Position with dust quantity (0.10 USDC) can be closed
    ts = native_datetime_utc_now()
    position = TradingPosition(
        position_id=1,
        pair=hypercore_pair,
        opened_at=ts,
        last_pricing_at=ts,
        last_token_price=1.0,
        last_reserve_price=1.0,
        reserve_currency=quote,
    )
    # is_spot() asserts at least one trade exists, so add a dummy
    dummy_trade = MagicMock()
    dummy_trade.is_spot.return_value = False
    position.trades[1] = dummy_trade

    with patch.object(position, "get_quantity", return_value=Decimal("0.10")):
        assert position.can_be_closed() is True

    # 4. Position with non-dust quantity (1.0 USDC) must NOT be closeable
    with patch.object(position, "get_quantity", return_value=Decimal("1.0")):
        assert position.can_be_closed() is False

    # 5. Non-Hypercore vault pair still gets DEFAULT_VAULT_EPSILON
    non_hypercore_base = AssetIdentifier(
        chain_id=999,
        address="0x0000000000000000000000000000000000000001",
        token_symbol="VAULT",
        decimals=6,
    )
    non_hypercore_quote = AssetIdentifier(
        chain_id=999,
        address="0x0000000000000000000000000000000000000002",
        token_symbol="USDC",
        decimals=6,
    )
    non_hypercore_pair = TradingPairIdentifier(
        base=non_hypercore_base,
        quote=non_hypercore_quote,
        pool_address="0x0000000000000000000000000000000000000003",
        exchange_address="0x0000000000000000000000000000000000000004",
        internal_id=2,
        internal_exchange_id=1,
        fee=0.0,
        kind=TradingPairKind.vault,
    )
    assert not non_hypercore_pair.is_hyperliquid_vault()
    assert get_close_epsilon_for_pair(non_hypercore_pair) == DEFAULT_VAULT_EPSILON


def test_hypercore_account_check_compares_equity_not_quantity() -> None:
    """Test Hypercore account checks compare expected equity against live equity.

    This covers the live Hyper-AI crash pattern where the vault checker
    compared API equity to position quantity, even though Hypercore
    valuation stores quantity and price separately.

    1. Create a Hypercore position with quantity from deposited USDC and a price below 1.0.
    2. Feed the account checker the live vault equity returned by the Hyperliquid API path.
    3. Verify the check uses expected USD equity, reports zero USD diff, and stays clean.
    """

    # 1. Create a Hypercore position with quantity from deposited USDC and a price below 1.0.
    quote = AssetIdentifier(
        chain_id=ChainId.hypercore.value,
        address="0x0000000000000000000000000000000000000002",
        token_symbol="USDC",
        decimals=6,
    )
    hypercore_pair = create_hypercore_vault_pair(
        quote=quote,
        vault_address="0x1111111111111111111111111111111111111111",
    )

    class DummyHypercorePosition:
        """Minimal Hypercore position stub for account-check regression coverage."""

        pair = hypercore_pair
        last_token_price = 0.9939891061405017
        last_pricing_at = native_datetime_utc_now()

        def __hash__(self) -> int:
            return id(self)

        def get_quantity(self) -> Decimal:
            return Decimal("56.104634")

        def calculate_quantity_usd_value(self, quantity: Decimal) -> float:
            assert quantity == Decimal("56.104634")
            return 55.767395

        def get_human_readable_name(self) -> str:
            return "Loop Fund"

    position = DummyHypercorePosition()

    state = MagicMock()
    state.portfolio.get_open_and_frozen_positions.return_value = [position]

    sync_model = MagicMock()
    sync_model.web3 = MagicMock()
    sync_model.get_token_storage_address.return_value = "0xa8F8DEbb722c6174B814b432169BF569603F673F"

    live_equity = Decimal("55.767395")
    live_balance = OnChainBalance(
        block_number=None,
        timestamp=native_datetime_utc_now(),
        asset=hypercore_pair.base,
        amount=live_equity,
    )

    # 2. Feed the account checker the live vault equity returned by the Hyperliquid API path.
    with patch(
        "tradeexecutor.strategy.account_correction.fetch_onchain_balances_multichain",
        return_value=iter([live_balance]),
    ):
        corrections = _build_hypercore_vault_account_checks(state, sync_model)

    # 3. Verify the check uses expected USD equity, reports zero USD diff, and stays clean.
    assert len(corrections) == 1
    correction = corrections[0]
    assert correction.expected_amount == Decimal("55.767395")
    assert correction.actual_amount == live_equity
    assert correction.relative_epsilon == HYPERLIQUID_VAULT_RELATIVE_EPSILON
    assert correction.usd_value == 0.0
    assert correction.mismatch is False


def test_post_trade_hypercore_revaluation_runs_only_for_open_hypercore_positions() -> None:
    """Revalue Hypercore positions immediately before post-trade account checks.

    1. Build a minimal runner with one open Hypercore vault position and one non-Hypercore position.
    2. Run the post-trade Hypercore refresh helper and verify only the Hypercore position is revalued.
    3. Leave only a frozen Hypercore vault position and verify the helper still performs the refresh.
    4. Remove all Hypercore positions and verify the helper skips the extra valuation pass entirely.
    """

    class DummyRunner(StrategyRunner):
        """Minimal runner used to exercise post-trade Hypercore revaluation."""

        def pretick_check(self, ts: datetime.datetime, universe) -> None:
            return None

    # 1. Build a minimal runner with one open Hypercore vault position and one non-Hypercore position.
    runner = DummyRunner(
        timed_task_context_manager=MagicMock(),
        execution_model=MagicMock(),
        approval_model=MagicMock(),
        valuation_model_factory=MagicMock(),
        sync_model=None,
        pricing_model_factory=MagicMock(),
        execution_context=unit_test_execution_context,
    )

    reserve_asset = AssetIdentifier(
        chain_id=999,
        address="0xb88339cb7199b77e23db6e890353e22632ba630f",
        token_symbol="USDC",
        decimals=6,
    )
    hypercore_pair = create_hypercore_vault_pair(
        quote=reserve_asset,
        vault_address="0x1111111111111111111111111111111111111111",
    )
    spot_pair = TradingPairIdentifier(
        base=AssetIdentifier(
            chain_id=1,
            address="0x0000000000000000000000000000000000000001",
            token_symbol="WETH",
            decimals=18,
        ),
        quote=AssetIdentifier(
            chain_id=1,
            address="0x0000000000000000000000000000000000000002",
            token_symbol="USDC",
            decimals=6,
        ),
        pool_address="0x0000000000000000000000000000000000000003",
        exchange_address="0x0000000000000000000000000000000000000004",
        internal_id=2,
        internal_exchange_id=2,
        fee=0.003,
    )

    hypercore_position = MagicMock()
    hypercore_position.pair = hypercore_pair
    frozen_hypercore_position = MagicMock()
    frozen_hypercore_position.pair = hypercore_pair
    spot_position = MagicMock()
    spot_position.pair = spot_pair

    state = MagicMock()
    state.portfolio.open_positions = {
        1: hypercore_position,
        2: spot_position,
    }
    state.portfolio.get_open_and_frozen_positions.return_value = [
        hypercore_position,
        spot_position,
    ]

    valuation_model = MagicMock()
    universe = MagicMock()
    runner.setup_routing_context = MagicMock(
        return_value=MagicMock(valuation_model=valuation_model),
    )

    # 2. Run the post-trade Hypercore refresh helper and verify only the Hypercore position is revalued.
    runner._revalue_open_hypercore_positions_before_post_trade_account_check(
        universe,
        state,
    )

    assert valuation_model.call_count == 1
    assert valuation_model.call_args[0][1] is hypercore_position

    # 3. Leave only a frozen Hypercore position and verify the helper still refreshes it.
    valuation_model.reset_mock()
    runner.setup_routing_context.reset_mock()
    state.portfolio.open_positions = {2: spot_position}
    state.portfolio.get_open_and_frozen_positions.return_value = [
        spot_position,
        frozen_hypercore_position,
    ]

    runner._revalue_open_hypercore_positions_before_post_trade_account_check(
        universe,
        state,
    )

    assert valuation_model.call_count == 1
    assert valuation_model.call_args[0][1] is frozen_hypercore_position

    # 4. Remove all Hypercore positions and verify the helper skips the extra valuation pass entirely.
    valuation_model.reset_mock()
    runner.setup_routing_context.reset_mock()
    state.portfolio.get_open_and_frozen_positions.return_value = [spot_position]

    runner._revalue_open_hypercore_positions_before_post_trade_account_check(
        universe,
        state,
    )

    valuation_model.assert_not_called()
    runner.setup_routing_context.assert_not_called()


def test_hypercore_dust_position_is_reused_without_planned_close() -> None:
    """Test Hypercore dust positions are reused unless the cycle is already closing them.

    1. Build a state with one open Hypercore vault position whose residual quantity is below the dust epsilon.
    2. Create a second buy trade for the same vault without any planned closing trade on the old position.
    3. Verify the trade reuses the existing position instead of opening a duplicate position.
    """

    # 1. Build a state with one open Hypercore vault dust position.
    reserve_asset = AssetIdentifier(
        chain_id=999,
        address="0xb88339cb7199b77e23db6e890353e22632ba630f",
        token_symbol="USDC",
        decimals=6,
    )
    pair = create_hypercore_vault_pair(
        quote=reserve_asset,
        vault_address="0x1111111111111111111111111111111111111111",
    )
    state = State()
    state.portfolio.initialise_reserves(reserve_asset, reserve_token_price=1.0)
    state.portfolio.adjust_reserves(reserve_asset, Decimal("100"), "Initial reserve")

    position, trade, created = state.create_trade(
        strategy_cycle_at=datetime.datetime(2026, 4, 13),
        pair=pair,
        quantity=None,
        reserve=Decimal("1.00"),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=reserve_asset,
        reserve_currency_price=1.0,
        notes="Create dust Hypercore position",
    )
    trade.mark_success(
        executed_at=datetime.datetime(2026, 4, 13, 0, 1),
        executed_price=1.0,
        executed_quantity=Decimal("1.00"),
        executed_reserve=Decimal("1.00"),
        lp_fees=0,
        native_token_price=0,
        force=True,
    )
    position.balance_updates[1] = BalanceUpdate(
        balance_update_id=1,
        cause=BalanceUpdateCause.vault_flow,
        position_type=BalanceUpdatePositionType.open_position,
        asset=pair.base,
        block_mined_at=datetime.datetime(2026, 4, 13, 0, 2),
        strategy_cycle_included_at=datetime.datetime(2026, 4, 13),
        chain_id=pair.base.chain_id,
        quantity=Decimal("-0.90"),
        old_balance=Decimal("1.00"),
        usd_value=-0.90,
        position_id=position.position_id,
        notes="Simulate Hypercore withdrawal dust",
        block_number=1,
    )

    assert created is True
    assert position.can_be_closed()
    assert len(state.portfolio.open_positions) == 1

    # 2. Create a second buy trade for the same vault.
    position2, trade2, created2 = state.create_trade(
        strategy_cycle_at=datetime.datetime(2026, 4, 14),
        pair=pair,
        quantity=None,
        reserve=Decimal("10"),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=reserve_asset,
        reserve_currency_price=1.0,
        notes="Increase the same Hypercore position",
    )

    # 3. Verify the existing position is reused and no duplicate is opened.
    assert created2 is False
    assert position2.position_id == position.position_id
    assert trade2.position_id == position.position_id
    assert len(state.portfolio.open_positions) == 1


def test_hypercore_dust_position_is_not_about_to_close_without_planned_trades() -> None:
    """Test Hypercore dust does not look like a planned close unless the cycle really has closing trades.

    1. Build a Hypercore position whose live quantity is below the close epsilon.
    2. Verify is_about_to_close() stays false while there are no planned trades.
    3. Mock a planned closing state and verify is_about_to_close() turns true.
    """

    # 1. Build a Hypercore position whose live quantity is below the close epsilon.
    reserve_asset = AssetIdentifier(
        chain_id=999,
        address="0xb88339cb7199b77e23db6e890353e22632ba630f",
        token_symbol="USDC",
        decimals=6,
    )
    pair = create_hypercore_vault_pair(
        quote=reserve_asset,
        vault_address="0x3333333333333333333333333333333333333333",
    )
    state = State()
    state.portfolio.initialise_reserves(reserve_asset, reserve_token_price=1.0)
    state.portfolio.adjust_reserves(reserve_asset, Decimal("100"), "Initial reserve")

    position, trade, _created = state.create_trade(
        strategy_cycle_at=datetime.datetime(2026, 4, 13),
        pair=pair,
        quantity=None,
        reserve=Decimal("1.00"),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=reserve_asset,
        reserve_currency_price=1.0,
        notes="Create dust Hypercore position",
    )
    trade.mark_success(
        executed_at=datetime.datetime(2026, 4, 13, 0, 1),
        executed_price=1.0,
        executed_quantity=Decimal("1.00"),
        executed_reserve=Decimal("1.00"),
        lp_fees=0,
        native_token_price=0,
        force=True,
    )
    position.balance_updates[1] = BalanceUpdate(
        balance_update_id=1,
        cause=BalanceUpdateCause.vault_flow,
        position_type=BalanceUpdatePositionType.open_position,
        asset=pair.base,
        block_mined_at=datetime.datetime(2026, 4, 13, 0, 2),
        strategy_cycle_included_at=datetime.datetime(2026, 4, 13),
        chain_id=pair.base.chain_id,
        quantity=Decimal("-0.90"),
        old_balance=Decimal("1.00"),
        usd_value=-0.90,
        position_id=position.position_id,
        notes="Simulate Hypercore withdrawal dust",
        block_number=1,
    )

    # 2. Verify is_about_to_close() stays false while there are no planned trades.
    assert position.can_be_closed()
    assert position.has_planned_trades() is False
    assert position.is_about_to_close() is False

    # 3. Mock a planned closing state and verify is_about_to_close() turns true.
    #    We mock here because create_trade() quite rightly refuses dust-sized
    #    execution trades. This regression targets the helper semantics only:
    #    dust must not look "about to close" until the cycle really has a
    #    planned closing trade against the position.
    with patch.object(position, "has_planned_trades", return_value=True):
        assert position.is_about_to_close() is True


def test_check_double_position_distinguishes_different_hypercore_vaults() -> None:
    """Test duplicate-position checks do not merge distinct Hypercore vaults.

    1. Build two open Hypercore positions with the same synthetic pair metadata but different vault addresses.
    2. Verify Hypercore pair equality still reproduces the broad identifier semantics.
    3. Verify the duplicate-position tripwire does not report a duplicate because the vault addresses differ.
    """

    # 1. Build two open Hypercore positions with different vault addresses.
    reserve_asset = AssetIdentifier(
        chain_id=999,
        address="0xb88339cb7199b77e23db6e890353e22632ba630f",
        token_symbol="USDC",
        decimals=6,
    )
    pair_1 = create_hypercore_vault_pair(
        quote=reserve_asset,
        vault_address="0x5555555555555555555555555555555555555555",
    )
    pair_2 = create_hypercore_vault_pair(
        quote=reserve_asset,
        vault_address="0x6666666666666666666666666666666666666666",
    )

    state = State()
    state.portfolio.initialise_reserves(reserve_asset, reserve_token_price=1.0)
    state.portfolio.adjust_reserves(reserve_asset, Decimal("100"), "Initial reserve")

    for idx, pair in enumerate((pair_1, pair_2), start=1):
        _position, trade, _created = state.create_trade(
            strategy_cycle_at=datetime.datetime(2026, 4, 13, idx),
            pair=pair,
            quantity=None,
            reserve=Decimal("10"),
            assumed_price=1.0,
            trade_type=TradeType.rebalance,
            reserve_currency=reserve_asset,
            reserve_currency_price=1.0,
            notes=f"Create Hypercore position {idx}",
            flags={TradeFlag.ignore_open},
        )
        trade.mark_success(
            executed_at=datetime.datetime(2026, 4, 13, idx, 1),
            executed_price=1.0,
            executed_quantity=Decimal("10"),
            executed_reserve=Decimal("10"),
            lp_fees=0,
            native_token_price=0,
            force=True,
        )

    # 2. Verify Hypercore pair equality still reproduces the broad identifier semantics.
    assert pair_1 == pair_2
    assert pair_1.get_identifier() != pair_2.get_identifier()

    # 3. Verify the duplicate-position tripwire does not report a duplicate.
    assert check_double_position(state, crash=True) is False


def test_hypercore_account_check_rejects_duplicate_vault_positions() -> None:
    """Test Hypercore account checks fail early with a direct duplicate-vault diagnosis.

    1. Build a state with a dusty Hypercore position and a forced second open position for the same vault.
    2. Run the Hypercore account-check builder.
    3. Verify it raises the targeted duplicate-Hypercore error instead of producing a misleading diff table.
    """

    # 1. Build a state with one dust position and one live duplicate position.
    reserve_asset = AssetIdentifier(
        chain_id=999,
        address="0xb88339cb7199b77e23db6e890353e22632ba630f",
        token_symbol="USDC",
        decimals=6,
    )
    pair = create_hypercore_vault_pair(
        quote=reserve_asset,
        vault_address="0x4444444444444444444444444444444444444444",
    )
    state = State()
    state.portfolio.initialise_reserves(reserve_asset, reserve_token_price=1.0)
    state.portfolio.adjust_reserves(reserve_asset, Decimal("100"), "Initial reserve")

    dust_position, dust_trade, _created = state.create_trade(
        strategy_cycle_at=datetime.datetime(2026, 4, 13),
        pair=pair,
        quantity=None,
        reserve=Decimal("1.00"),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=reserve_asset,
        reserve_currency_price=1.0,
        notes="Create dust Hypercore position",
    )
    dust_trade.mark_success(
        executed_at=datetime.datetime(2026, 4, 13, 0, 1),
        executed_price=1.0,
        executed_quantity=Decimal("1.00"),
        executed_reserve=Decimal("1.00"),
        lp_fees=0,
        native_token_price=0,
        force=True,
    )
    dust_position.balance_updates[1] = BalanceUpdate(
        balance_update_id=1,
        cause=BalanceUpdateCause.vault_flow,
        position_type=BalanceUpdatePositionType.open_position,
        asset=pair.base,
        block_mined_at=datetime.datetime(2026, 4, 13, 0, 2),
        strategy_cycle_included_at=datetime.datetime(2026, 4, 13),
        chain_id=pair.base.chain_id,
        quantity=Decimal("-0.90"),
        old_balance=Decimal("1.00"),
        usd_value=-0.90,
        position_id=dust_position.position_id,
        notes="Simulate Hypercore withdrawal dust",
        block_number=1,
    )

    _live_position, live_trade, live_created = state.create_trade(
        strategy_cycle_at=datetime.datetime(2026, 4, 14),
        pair=pair,
        quantity=None,
        reserve=Decimal("25"),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=reserve_asset,
        reserve_currency_price=1.0,
        notes="Force a second open Hypercore position for regression coverage",
        flags={TradeFlag.ignore_open},
    )
    live_trade.mark_success(
        executed_at=datetime.datetime(2026, 4, 14, 0, 1),
        executed_price=1.0,
        executed_quantity=Decimal("25"),
        executed_reserve=Decimal("25"),
        lp_fees=0,
        native_token_price=0,
        force=True,
    )

    assert live_created is True
    assert len(state.portfolio.open_positions) == 2

    sync_model = MagicMock()
    sync_model.web3 = MagicMock()
    sync_model.get_token_storage_address.return_value = "0xa8F8DEbb722c6174B814b432169BF569603F673F"

    # 2. Run the Hypercore account-check builder.
    # 3. Verify it raises the targeted duplicate-Hypercore error.
    with pytest.raises(UnexpectedAccountingCorrectionIssue, match="Duplicate Hypercore vault positions detected"):
        _build_hypercore_vault_account_checks(state, sync_model)


def test_close_hypercore_dust_positions_closes_duplicate_residual_state() -> None:
    """Test Hypercore dust cleanup closes the stale residual position and keeps the live one open.

    1. Build a state with a dusty Hypercore position and a forced second open position for the same vault.
    2. Run the Hypercore dust cleanup helper.
    3. Verify the residual dust position is closed with a repair trade while the live position stays open.
    """

    # 1. Build a state with one dust position and one live duplicate position.
    reserve_asset = AssetIdentifier(
        chain_id=999,
        address="0xb88339cb7199b77e23db6e890353e22632ba630f",
        token_symbol="USDC",
        decimals=6,
    )
    pair = create_hypercore_vault_pair(
        quote=reserve_asset,
        vault_address="0x2222222222222222222222222222222222222222",
    )
    state = State()
    state.portfolio.initialise_reserves(reserve_asset, reserve_token_price=1.0)
    state.portfolio.adjust_reserves(reserve_asset, Decimal("100"), "Initial reserve")

    dust_position, dust_trade, _created = state.create_trade(
        strategy_cycle_at=datetime.datetime(2026, 4, 13),
        pair=pair,
        quantity=None,
        reserve=Decimal("1.00"),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=reserve_asset,
        reserve_currency_price=1.0,
        notes="Create dust Hypercore position",
    )
    dust_trade.mark_success(
        executed_at=datetime.datetime(2026, 4, 13, 0, 1),
        executed_price=1.0,
        executed_quantity=Decimal("1.00"),
        executed_reserve=Decimal("1.00"),
        lp_fees=0,
        native_token_price=0,
        force=True,
    )
    dust_position.balance_updates[1] = BalanceUpdate(
        balance_update_id=1,
        cause=BalanceUpdateCause.vault_flow,
        position_type=BalanceUpdatePositionType.open_position,
        asset=pair.base,
        block_mined_at=datetime.datetime(2026, 4, 13, 0, 2),
        strategy_cycle_included_at=datetime.datetime(2026, 4, 13),
        chain_id=pair.base.chain_id,
        quantity=Decimal("-0.90"),
        old_balance=Decimal("1.00"),
        usd_value=-0.90,
        position_id=dust_position.position_id,
        notes="Simulate Hypercore withdrawal dust",
        block_number=1,
    )

    live_position, live_trade, live_created = state.create_trade(
        strategy_cycle_at=datetime.datetime(2026, 4, 14),
        pair=pair,
        quantity=None,
        reserve=Decimal("25"),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=reserve_asset,
        reserve_currency_price=1.0,
        notes="Force a second open Hypercore position for regression coverage",
        flags={TradeFlag.ignore_open},
    )
    live_trade.mark_success(
        executed_at=datetime.datetime(2026, 4, 14, 0, 1),
        executed_price=1.0,
        executed_quantity=Decimal("25"),
        executed_reserve=Decimal("25"),
        lp_fees=0,
        native_token_price=0,
        force=True,
    )

    assert live_created is True
    assert len(state.portfolio.open_positions) == 2
    assert dust_position.can_be_closed()
    assert not live_position.can_be_closed()

    # 2. Run the Hypercore dust cleanup helper.
    created_trades = close_hypercore_dust_positions(
        state.portfolio,
        now=datetime.datetime(2026, 4, 15),
    )

    # 3. Verify only the dust position is closed and the live one remains open.
    assert len(created_trades) == 1
    assert dust_position.position_id in state.portfolio.closed_positions
    assert dust_position.position_id not in state.portfolio.open_positions
    assert live_position.position_id in state.portfolio.open_positions
    assert live_position.position_id not in state.portfolio.closed_positions
    assert created_trades[0].trade_type == TradeType.repair
