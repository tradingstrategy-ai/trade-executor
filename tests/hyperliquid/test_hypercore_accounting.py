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

from tradeexecutor.cli.commands.correct_accounts import _sync_hypercore_vault_positions
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
    get_hypercore_withdrawal_safety_margin,
    get_close_epsilon_for_pair,
    get_dust_epsilon_for_pair,
    HYPERLIQUID_VAULT_CLOSE_EPSILON,
    HYPERLIQUID_VAULT_RELATIVE_EPSILON,
    DEFAULT_VAULT_EPSILON,
)
from tradeexecutor.strategy.execution_model import AssetManagementMode
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.sync_model import OnChainBalance
from tradeexecutor.visual.equity_curve import calculate_compounding_unrealised_trading_profitability


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


def test_correct_accounts_marks_hypercore_equity_without_vault_flow() -> None:
    """Test Hypercore account correction does not book vault performance as a balance flow.

    1. Create an open Hypercore vault position with 100 USDC deposited.
    2. Mock the Hyperliquid equity API to report 105 USDC.
    3. Run the Hypercore account correction sync helper.
    4. Verify the position is revalued to 105 USDC without creating a vault_flow balance update.
    """

    # 1. Create an open Hypercore vault position with 100 USDC deposited.
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
    state.portfolio.adjust_reserves(reserve_asset, Decimal("200"), "Initial reserve")
    position, trade, _created = state.create_trade(
        strategy_cycle_at=datetime.datetime(2026, 4, 28),
        pair=pair,
        quantity=None,
        reserve=Decimal("100"),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=reserve_asset,
        reserve_currency_price=1.0,
        notes="Open Hypercore vault position",
    )
    state.mark_trade_success(
        executed_at=datetime.datetime(2026, 4, 28, 0, 1),
        trade=trade,
        executed_price=1.0,
        executed_amount=Decimal("100"),
        executed_reserve=Decimal("100"),
        lp_fees=0,
        native_token_price=0,
        force=True,
    )

    # 2. Mock the Hyperliquid equity API to report 105 USDC.
    universe = MagicMock()
    dex_pair = object()
    universe.data_universe.pairs.iterate_pairs.return_value = [dex_pair]
    sync_model = MagicMock()
    sync_model.get_token_storage_address.return_value = "0xa8F8DEbb722c6174B814b432169BF569603F673F"
    web3 = MagicMock()
    web3.eth.chain_id = 999
    next_balance_update_id = state.portfolio.next_balance_update_id

    with (
        patch("tradeexecutor.strategy.trading_strategy_universe.translate_trading_pair", return_value=pair),
        patch("tradeexecutor.strategy.account_correction.translate_trading_pair", return_value=pair),
        patch("eth_defi.hyperliquid.session.create_hyperliquid_session", return_value=MagicMock()),
        patch(
            "tradeexecutor.ethereum.vault.hypercore_vault.create_hypercore_vault_value_func",
            return_value=lambda pair: Decimal("105"),
        ),
    ):
        # 3. Run the Hypercore account correction sync helper.
        _sync_hypercore_vault_positions(
            asset_management_mode=AssetManagementMode.lagoon,
            universe=universe,
            sync_model=sync_model,
            web3=web3,
            state=state,
        )

    # 4. Verify the position is revalued to 105 USDC without creating a vault_flow balance update.
    assert position.get_quantity() == Decimal("100")
    assert position.get_value() == pytest.approx(105)
    assert position.last_token_price == pytest.approx(1.05)
    assert len(position.valuation_updates) == 1
    assert len(position.balance_updates) == 0
    assert state.portfolio.next_balance_update_id == next_balance_update_id
    assert len(state.sync.accounting.balance_update_refs) == 0


def test_hypercore_profit_uses_internal_share_price_with_vault_flow() -> None:
    """Test Hypercore vault profit is calculated from internal share price state.

    1. Create an open Hypercore vault position with 100 USDC deposited.
    2. Add a legacy vault_flow balance update that increases tracked quantity by 20 USDC.
    3. Verify profit uses internal share price state instead of spot average-price logic.
    """

    # 1. Create an open Hypercore vault position with 100 USDC deposited.
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
    state.portfolio.adjust_reserves(reserve_asset, Decimal("200"), "Initial reserve")
    position, trade, _created = state.create_trade(
        strategy_cycle_at=datetime.datetime(2026, 4, 28),
        pair=pair,
        quantity=None,
        reserve=Decimal("100"),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=reserve_asset,
        reserve_currency_price=1.0,
        notes="Open Hypercore vault position",
    )
    state.mark_trade_success(
        executed_at=datetime.datetime(2026, 4, 28, 0, 1),
        trade=trade,
        executed_price=1.0,
        executed_amount=Decimal("100"),
        executed_reserve=Decimal("100"),
        lp_fees=0,
        native_token_price=0,
        force=True,
    )

    # 2. Add a legacy vault_flow balance update that increases tracked quantity by 20 USDC.
    position.balance_updates[1] = BalanceUpdate(
        balance_update_id=1,
        cause=BalanceUpdateCause.vault_flow,
        position_type=BalanceUpdatePositionType.open_position,
        asset=pair.base,
        block_mined_at=datetime.datetime(2026, 4, 28, 0, 2),
        strategy_cycle_included_at=datetime.datetime(2026, 4, 28),
        chain_id=pair.base.chain_id,
        quantity=Decimal("20"),
        old_balance=Decimal("100"),
        usd_value=20,
        position_id=position.position_id,
        notes="Legacy Hypercore vault equity sync",
        block_number=1,
    )

    # 3. Verify profit uses internal share price state instead of spot average-price logic.
    assert position.share_price_state is not None
    assert position.get_quantity() == Decimal("120")
    assert position.get_unrealised_profit_usd() == pytest.approx(20)
    assert position.get_total_profit_usd() == pytest.approx(20)
    assert position.get_unrealised_profit_pct() == pytest.approx(0.20)
    assert position.get_total_profit_percent() == pytest.approx(0.20)


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
    position = MagicMock()
    position.is_vault.return_value = True
    position.get_quantity.return_value = Decimal("100.0")
    position.last_token_price = 1.0
    position.other_data = {}

    expires = datetime.datetime(2026, 3, 27, 14, 30, 0)

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

    The 1.50 USDC fixed safety-margin floor is subtracted from small live vault
    equities during full-close withdrawals. The default close epsilon must
    exceed this floor so can_be_closed() recognises the position as effectively
    closed. Larger percentage-derived headroom is accounted for by verified
    full-close settlement and must not widen this bookkeeping-only threshold.

    1. Build a Hypercore vault pair using create_hypercore_vault_pair().
    2. Verify get_close_epsilon_for_pair() returns the Hypercore-specific epsilon
       and get_dust_epsilon_for_pair() returns the smaller vault default.
    3. Create a TradingPosition with safety-margin dust quantity (1.50) and assert can_be_closed().
    4. Same position with non-dust quantity (2.50) must NOT be closeable.
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

    # 2. Close epsilon returns Hypercore-specific value; dust epsilon uses
    #    the smaller vault default so that buy trades are not rejected
    assert get_close_epsilon_for_pair(hypercore_pair) == HYPERLIQUID_VAULT_CLOSE_EPSILON
    assert get_dust_epsilon_for_pair(hypercore_pair) == DEFAULT_VAULT_EPSILON

    # 3. Position with safety-margin dust quantity (1.50 USDC) can be closed
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

    with patch.object(position, "get_quantity", return_value=Decimal("1.50")):
        assert position.can_be_closed() is True

    # 4. Position with non-dust quantity (2.50 USDC) must NOT be closeable
    with patch.object(position, "get_quantity", return_value=Decimal("2.50")):
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


def test_hypercore_reduction_planning_reserves_relative_margin_with_fixed_floor() -> None:
    """HyperCore reduction planning keeps relative headroom without weakening small withdrawals.

    1. Build a real state position with enough executed quantity for both reduction scenarios.
    2. Plan a large cap-bound redemption and verify it retains 0.5% of the live cap.
    3. Plan a small cap-bound redemption and verify it retains the 1.50 USDC floor.
    4. Plan an unconstrained full close and verify safety headroom does not make it partial.
    """

    # 1. Build a real state position with enough executed quantity for both reduction scenarios.
    reserve_asset = AssetIdentifier(
        chain_id=ChainId.hypercore.value,
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
    state.portfolio.adjust_reserves(reserve_asset, Decimal("10000"), "Initial reserve")
    position, opening_trade, _created = state.create_trade(
        strategy_cycle_at=datetime.datetime(2026, 8, 22),
        pair=pair,
        quantity=None,
        reserve=Decimal("7000"),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=reserve_asset,
        reserve_currency_price=1.0,
    )
    opening_trade.mark_success(
        executed_at=datetime.datetime(2026, 8, 22, 0, 1),
        executed_price=1.0,
        executed_quantity=Decimal("7000"),
        executed_reserve=Decimal("7000"),
        lp_fees=0,
        native_token_price=0,
        force=True,
    )
    position.last_token_price = 1.0
    position_manager = object.__new__(PositionManager)

    # 2. Plan a large cap-bound redemption and verify it retains 0.5% of the live cap.
    large_cap = Decimal("6389.537474")
    large_plan = position_manager.prepare_hypercore_position_reduction(
        position,
        dollar_delta=-7000,
        max_redemption=float(large_cap),
    )
    large_margin = get_hypercore_withdrawal_safety_margin(large_cap)
    assert large_margin == Decimal("31.947688")
    assert large_plan.effective_quantity_delta == pytest.approx(-float(large_cap - large_margin))
    assert large_plan.redemption_cap_bound is True
    assert large_plan.treat_as_full_close is False

    # 3. Plan a small cap-bound redemption and verify it retains the 1.50 USDC floor.
    small_cap = Decimal("100")
    small_plan = position_manager.prepare_hypercore_position_reduction(
        position,
        dollar_delta=-7000,
        max_redemption=float(small_cap),
    )
    assert get_hypercore_withdrawal_safety_margin(small_cap) == Decimal("1.500000")
    assert small_plan.effective_quantity_delta == pytest.approx(-98.5)
    assert small_plan.redemption_cap_bound is True
    assert small_plan.treat_as_full_close is False

    # 4. Plan an unconstrained full close and verify safety headroom does not make it partial.
    full_close_plan = position_manager.prepare_hypercore_position_reduction(
        position,
        dollar_delta=-7000,
        max_redemption=7000,
    )
    assert full_close_plan.effective_quantity_delta == pytest.approx(-7000)
    assert full_close_plan.redemption_cap_bound is False
    assert full_close_plan.treat_as_full_close is True


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
        reserve=Decimal("5.00"),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=reserve_asset,
        reserve_currency_price=1.0,
        notes="Create dust Hypercore position",
    )
    trade.mark_success(
        executed_at=datetime.datetime(2026, 4, 13, 0, 1),
        executed_price=1.0,
        executed_quantity=Decimal("5.00"),
        executed_reserve=Decimal("5.00"),
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
        quantity=Decimal("-3.50"),
        old_balance=Decimal("5.00"),
        usd_value=-3.50,
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
        reserve=Decimal("5.00"),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=reserve_asset,
        reserve_currency_price=1.0,
        notes="Create dust Hypercore position",
    )
    trade.mark_success(
        executed_at=datetime.datetime(2026, 4, 13, 0, 1),
        executed_price=1.0,
        executed_quantity=Decimal("5.00"),
        executed_reserve=Decimal("5.00"),
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
        quantity=Decimal("-3.50"),
        old_balance=Decimal("5.00"),
        usd_value=-3.50,
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
        reserve=Decimal("5.00"),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=reserve_asset,
        reserve_currency_price=1.0,
        notes="Create dust Hypercore position",
    )
    dust_trade.mark_success(
        executed_at=datetime.datetime(2026, 4, 13, 0, 1),
        executed_price=1.0,
        executed_quantity=Decimal("5.00"),
        executed_reserve=Decimal("5.00"),
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
        quantity=Decimal("-3.50"),
        old_balance=Decimal("5.00"),
        usd_value=-3.50,
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
        reserve=Decimal("5.00"),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=reserve_asset,
        reserve_currency_price=1.0,
        notes="Create dust Hypercore position",
    )
    dust_trade.mark_success(
        executed_at=datetime.datetime(2026, 4, 13, 0, 1),
        executed_price=1.0,
        executed_quantity=Decimal("5.00"),
        executed_reserve=Decimal("5.00"),
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
        quantity=Decimal("-3.50"),
        old_balance=Decimal("5.00"),
        usd_value=-3.50,
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


def test_close_hypercore_dust_positions_skips_unexecuted_and_planned_positions() -> None:
    """Dust cleanup must not treat unfinished HyperCore trades as redeemed positions.

    1. Create the Hyper AI crash shape: a new position whose opening trade is still planned.
    2. Create an executed dust position that also has a started follow-up trade.
    3. Run dust cleanup and verify neither unfinished position is repaired or closed.
    """

    # 1. Create the Hyper AI crash shape: a new position whose opening trade is still planned.
    reserve_asset = AssetIdentifier(
        chain_id=ChainId.hypercore.value,
        address="0xb88339cb7199b77e23db6e890353e22632ba630f",
        token_symbol="USDC",
        decimals=6,
    )
    state = State()
    state.portfolio.initialise_reserves(reserve_asset, reserve_token_price=1.0)
    state.portfolio.adjust_reserves(reserve_asset, Decimal("10000"), "Initial reserve")
    planned_pair = create_hypercore_vault_pair(
        quote=reserve_asset,
        vault_address="0x1111111111111111111111111111111111111111",
        internal_id=1,
    )
    planned_position, planned_opening_trade, _created = state.create_trade(
        strategy_cycle_at=datetime.datetime(2026, 8, 22),
        pair=planned_pair,
        quantity=None,
        reserve=Decimal("3248.872789"),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=reserve_asset,
        reserve_currency_price=1.0,
    )
    assert planned_opening_trade.is_planned()
    assert planned_position.get_quantity() == 0

    # 2. Create an executed dust position that also has a started follow-up trade.
    follow_up_pair = create_hypercore_vault_pair(
        quote=reserve_asset,
        vault_address="0x2222222222222222222222222222222222222222",
        internal_id=2,
    )
    follow_up_position, successful_opening_trade, _created = state.create_trade(
        strategy_cycle_at=datetime.datetime(2026, 8, 21),
        pair=follow_up_pair,
        quantity=None,
        reserve=Decimal("1.00"),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=reserve_asset,
        reserve_currency_price=1.0,
        flags={TradeFlag.ignore_open},
    )
    successful_opening_trade.mark_success(
        executed_at=datetime.datetime(2026, 8, 21, 0, 1),
        executed_price=1.0,
        executed_quantity=Decimal("1.00"),
        executed_reserve=Decimal("1.00"),
        lp_fees=0,
        native_token_price=0,
        force=True,
    )
    _, planned_follow_up_trade, created = state.create_trade(
        strategy_cycle_at=datetime.datetime(2026, 8, 22),
        pair=follow_up_pair,
        quantity=None,
        reserve=Decimal("25.00"),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=reserve_asset,
        reserve_currency_price=1.0,
        position=follow_up_position,
    )
    assert created is False
    state.start_execution(
        datetime.datetime(2026, 8, 22, 0, 1),
        planned_follow_up_trade,
    )
    assert planned_follow_up_trade.is_started()
    assert follow_up_position.can_be_closed()

    # 3. Run dust cleanup and verify neither unfinished position is repaired or closed.
    created_trades = close_hypercore_dust_positions(
        state.portfolio,
        now=datetime.datetime(2026, 8, 23),
    )

    assert created_trades == []
    assert planned_position.position_id in state.portfolio.open_positions
    assert follow_up_position.position_id in state.portfolio.open_positions
    assert state.portfolio.closed_positions == {}
    assert len(planned_position.trades) == 1
    assert len(follow_up_position.trades) == 2


def test_correct_accounts_closes_phantom_position_from_untracked_withdrawal() -> None:
    """Test that correct-accounts detects and closes a phantom Hypercore vault position.

    A phantom position occurs when a Hypercore vault withdrawal completed
    on Hyperliquid but the executor failed to confirm it (timeout/restart
    during the 3-phase settlement). The repair command zeroes out the
    unconfirmed trade, leaving the state with positive quantity but zero
    on-chain equity.

    1. Create an open Hypercore vault position with 304 USDC deposited.
    2. Record reserve quantity after setup.
    3. Mock Hyperliquid equity API to return 0 (no position found).
    4. Run the Hypercore account correction sync helper.
    5. Verify the position is closed with a zero-proceeds repair trade.
    6. Verify reserves were not changed (USDC reconciliation is deferred).
    7. Verify equity curve calculation does not crash.
    """

    # 1. Create an open Hypercore vault position with 304 USDC deposited.
    reserve_asset = AssetIdentifier(
        chain_id=999,
        address="0xb88339cb7199b77e23db6e890353e22632ba630f",
        token_symbol="USDC",
        decimals=6,
    )
    pair = create_hypercore_vault_pair(
        quote=reserve_asset,
        vault_address="0xf6f3d773e11023e3e686cbda883ecba631fefc15",
    )
    state = State()
    state.portfolio.initialise_reserves(reserve_asset, reserve_token_price=1.0)
    state.portfolio.adjust_reserves(reserve_asset, Decimal("500"), "Initial reserve")
    position, trade, _created = state.create_trade(
        strategy_cycle_at=datetime.datetime(2026, 3, 30),
        pair=pair,
        quantity=None,
        reserve=Decimal("304"),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=reserve_asset,
        reserve_currency_price=1.0,
        notes="Open Hypercore vault position (simulating YEELON deposit)",
    )
    state.mark_trade_success(
        executed_at=datetime.datetime(2026, 3, 30, 0, 1),
        trade=trade,
        executed_price=1.0,
        executed_amount=Decimal("304"),
        executed_reserve=Decimal("304"),
        lp_fees=0,
        native_token_price=0,
        force=True,
    )
    assert position.get_quantity() == Decimal("304")
    assert position.is_open()

    # 2. Record reserve quantity after setup.
    reserve_position = state.portfolio.get_default_reserve_position()
    reserves_after_setup = reserve_position.quantity

    # 3. Mock Hyperliquid equity API to return 0 (no position found).
    universe = MagicMock()
    dex_pair = object()
    universe.data_universe.pairs.iterate_pairs.return_value = [dex_pair]
    sync_model = MagicMock()
    sync_model.get_token_storage_address.return_value = "0xa8F8DEbb722c6174B814b432169BF569603F673F"
    web3 = MagicMock()
    web3.eth.chain_id = 999

    with (
        patch("tradeexecutor.strategy.trading_strategy_universe.translate_trading_pair", return_value=pair),
        patch("tradeexecutor.strategy.account_correction.translate_trading_pair", return_value=pair),
        patch("eth_defi.hyperliquid.session.create_hyperliquid_session", return_value=MagicMock()),
        patch(
            "tradeexecutor.ethereum.vault.hypercore_vault.create_hypercore_vault_value_func",
            return_value=lambda pair: Decimal("0"),
        ),
    ):
        # 4. Run the Hypercore account correction sync helper.
        _sync_hypercore_vault_positions(
            asset_management_mode=AssetManagementMode.lagoon,
            universe=universe,
            sync_model=sync_model,
            web3=web3,
            state=state,
        )

    # 5. Verify the position is closed with a zero-proceeds repair trade.
    assert position.position_id in state.portfolio.closed_positions
    assert position.position_id not in state.portfolio.open_positions
    assert position.get_quantity() == Decimal(0)

    last_trade = list(position.trades.values())[-1]
    assert last_trade.trade_type == TradeType.repair
    assert last_trade.executed_quantity == Decimal("-304")
    assert last_trade.executed_reserve == Decimal(0)
    assert last_trade.is_success()

    # 6. Verify reserves were not changed (USDC reconciliation is deferred).
    assert reserve_position.quantity == reserves_after_setup

    # 7. Verify equity curve calculation does not crash.
    result = calculate_compounding_unrealised_trading_profitability(state)
    assert result is not None


def test_correct_accounts_closes_phantom_position_already_valued_at_zero() -> None:
    """Test that correct-accounts closes a phantom position even when already valued at zero.

    In production, the valuator tick runs before correct-accounts and sets
    last_token_price=0.0 when the Hyperliquid API returns zero equity. This
    makes old_value=0 and diff=0. The phantom position detection must still
    fire despite the diff==0 condition.

    1. Create an open Hypercore vault position with 304 USDC deposited.
    2. Set last_token_price=0.0 to simulate a prior valuator tick.
    3. Record reserve quantity after setup.
    4. Mock Hyperliquid equity API to return 0.
    5. Run the Hypercore account correction sync helper.
    6. Verify the position is closed despite diff==0.
    7. Verify reserves were not changed.
    """

    # 1. Create an open Hypercore vault position with 304 USDC deposited.
    reserve_asset = AssetIdentifier(
        chain_id=999,
        address="0xb88339cb7199b77e23db6e890353e22632ba630f",
        token_symbol="USDC",
        decimals=6,
    )
    pair = create_hypercore_vault_pair(
        quote=reserve_asset,
        vault_address="0xf6f3d773e11023e3e686cbda883ecba631fefc15",
    )
    state = State()
    state.portfolio.initialise_reserves(reserve_asset, reserve_token_price=1.0)
    state.portfolio.adjust_reserves(reserve_asset, Decimal("500"), "Initial reserve")
    position, trade, _created = state.create_trade(
        strategy_cycle_at=datetime.datetime(2026, 3, 30),
        pair=pair,
        quantity=None,
        reserve=Decimal("304"),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=reserve_asset,
        reserve_currency_price=1.0,
    )
    state.mark_trade_success(
        executed_at=datetime.datetime(2026, 3, 30, 0, 1),
        trade=trade,
        executed_price=1.0,
        executed_amount=Decimal("304"),
        executed_reserve=Decimal("304"),
        lp_fees=0,
        native_token_price=0,
        force=True,
    )

    # 2. Set last_token_price=0.0 to simulate a prior valuator tick.
    position.last_token_price = 0.0
    position.revalue_base_asset(datetime.datetime(2026, 6, 8), 0.0)

    # 3. Record reserve quantity after setup.
    reserve_position = state.portfolio.get_default_reserve_position()
    reserves_after_setup = reserve_position.quantity

    # 4. Mock Hyperliquid equity API to return 0.
    universe = MagicMock()
    dex_pair = object()
    universe.data_universe.pairs.iterate_pairs.return_value = [dex_pair]
    sync_model = MagicMock()
    sync_model.get_token_storage_address.return_value = "0xa8F8DEbb722c6174B814b432169BF569603F673F"
    web3 = MagicMock()
    web3.eth.chain_id = 999

    with (
        patch("tradeexecutor.strategy.trading_strategy_universe.translate_trading_pair", return_value=pair),
        patch("tradeexecutor.strategy.account_correction.translate_trading_pair", return_value=pair),
        patch("eth_defi.hyperliquid.session.create_hyperliquid_session", return_value=MagicMock()),
        patch(
            "tradeexecutor.ethereum.vault.hypercore_vault.create_hypercore_vault_value_func",
            return_value=lambda pair: Decimal("0"),
        ),
    ):
        # 5. Run the Hypercore account correction sync helper.
        _sync_hypercore_vault_positions(
            asset_management_mode=AssetManagementMode.lagoon,
            universe=universe,
            sync_model=sync_model,
            web3=web3,
            state=state,
        )

    # 6. Verify the position is closed despite diff==0.
    assert position.position_id in state.portfolio.closed_positions
    assert position.position_id not in state.portfolio.open_positions
    assert position.get_quantity() == Decimal(0)

    last_trade = list(position.trades.values())[-1]
    assert last_trade.trade_type == TradeType.repair
    assert last_trade.executed_quantity == Decimal("-304")

    # 7. Verify reserves were not changed.
    assert reserve_position.quantity == reserves_after_setup


def test_valuation_accepts_zero_price() -> None:
    """Test that the valuation PnL calculation accepts a zero valuation price.

    A valuation price of 0.0 is valid when a Hypercore vault position has
    lost all value (trading losses or untracked withdrawal). The code must
    use ``is not None`` checks instead of truthiness to avoid crashing on
    zero prices.

    1. Create a mock position with last_token_price=0.0 and positive quantity.
    2. Call get_unrealised_and_realised_profit_percent().
    3. Verify no AssertionError is raised and the result is sensible.
    """

    # 1. Create a mock position with last_token_price=0.0 and positive quantity.
    reserve_asset = AssetIdentifier(
        chain_id=999,
        address="0xb88339cb7199b77e23db6e890353e22632ba630f",
        token_symbol="USDC",
        decimals=6,
    )
    pair = create_hypercore_vault_pair(
        quote=reserve_asset,
        vault_address="0xf6f3d773e11023e3e686cbda883ecba631fefc15",
    )
    state = State()
    state.portfolio.initialise_reserves(reserve_asset, reserve_token_price=1.0)
    state.portfolio.adjust_reserves(reserve_asset, Decimal("500"), "Initial reserve")
    position, trade, _created = state.create_trade(
        strategy_cycle_at=datetime.datetime(2026, 3, 30),
        pair=pair,
        quantity=None,
        reserve=Decimal("100"),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=reserve_asset,
        reserve_currency_price=1.0,
    )
    state.mark_trade_success(
        executed_at=datetime.datetime(2026, 3, 30, 0, 1),
        trade=trade,
        executed_price=1.0,
        executed_amount=Decimal("100"),
        executed_reserve=Decimal("100"),
        lp_fees=0,
        native_token_price=0,
        force=True,
    )

    # Simulate zero valuation (total loss)
    position.last_token_price = 0.0

    # 2. Call get_unrealised_and_realised_profit_percent().
    # This previously crashed with AssertionError because 0.0 is falsy.
    result = position.get_unrealised_and_realised_profit_percent()

    # 3. Verify no AssertionError is raised and the result is sensible.
    assert result is not None
