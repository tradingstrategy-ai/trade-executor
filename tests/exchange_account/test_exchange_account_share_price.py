"""Test internal share price PnL tracking for exchange account positions.

Exchange account positions use price=1.0 with value changes arriving via
balance updates. The internal share price model must:

1. NOT initialise from the placeholder trade (avoids bootstrap-as-profit bug)
2. Initialise from the first valuation sync (actual capital from exchange API)
3. Track PnL correctly on subsequent valuations
4. Migrate correctly when backfilling existing positions
"""

import datetime
from decimal import Decimal
from unittest.mock import Mock

import pytest

from eth_defi.compat import native_datetime_utc_now

from tradeexecutor.exchange_account.pricing import ExchangeAccountPricingModel
from tradeexecutor.exchange_account.state import open_exchange_account_position
from tradeexecutor.exchange_account.valuation import ExchangeAccountValuator
from tradeexecutor.state.balance_update import (
    BalanceUpdate,
    BalanceUpdateCause,
    BalanceUpdatePositionType,
)
from tradeexecutor.state.identifier import (
    TradingPairIdentifier,
    AssetIdentifier,
    TradingPairKind,
)
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeType
from tradeexecutor.strategy.position_internal_share_price import (
    backfill_share_price_state,
    migrate_share_price_state,
)


@pytest.fixture
def exchange_account_pair() -> TradingPairIdentifier:
    """Create exchange account pair for testing."""
    usdc = AssetIdentifier(
        chain_id=901,
        address="0x0000000000000000000000000000000000000001",
        token_symbol="USDC",
        decimals=6,
    )
    account = AssetIdentifier(
        chain_id=901,
        address="0x0000000000000000000000000000000000000002",
        token_symbol="GMX-ACCOUNT",
        decimals=6,
    )
    return TradingPairIdentifier(
        base=account,
        quote=usdc,
        pool_address="0x0000000000000000000000000000000000000003",
        exchange_address="0x0000000000000000000000000000000000000004",
        internal_id=1,
        internal_exchange_id=1,
        fee=0.0,
        kind=TradingPairKind.exchange_account,
        exchange_name="GMX",
        other_data={
            "exchange_protocol": "gmx",
        },
    )


def _create_position_with_placeholder_trade(
    pair: TradingPairIdentifier,
    reserve_amount: Decimal = Decimal(1),
    opened_at: datetime.datetime | None = None,
) -> TradingPosition:
    """Create an exchange account position with a placeholder trade (mimics correct-accounts)."""
    opened_at = opened_at or datetime.datetime(2024, 1, 1)
    position = TradingPosition(
        position_id=1,
        pair=pair,
        opened_at=opened_at,
        last_pricing_at=opened_at,
        last_token_price=1.0,
        last_reserve_price=1.0,
        reserve_currency=pair.quote,
    )
    trade = TradeExecution(
        trade_id=1,
        position_id=1,
        trade_type=TradeType.rebalance,
        pair=pair,
        opened_at=opened_at,
        planned_quantity=reserve_amount,
        planned_price=1.0,
        planned_reserve=reserve_amount,
        reserve_currency=pair.quote,
    )
    trade.started_at = opened_at
    trade.mark_broadcasted(opened_at)
    trade.mark_success(
        executed_at=opened_at,
        executed_price=1.0,
        executed_quantity=reserve_amount,
        executed_reserve=reserve_amount,
        lp_fees=0.0,
        native_token_price=1.0,
    )
    position.trades[1] = trade
    return position


def _valuate(position: TradingPosition, api_value: Decimal) -> None:
    """Run one valuation cycle against the position."""
    mock_pricing = Mock(spec=ExchangeAccountPricingModel)
    mock_pricing.get_account_value.return_value = api_value
    valuator = ExchangeAccountValuator(mock_pricing)
    valuator(ts=native_datetime_utc_now(), position=position)


def test_placeholder_trade_does_not_initialise_share_price(exchange_account_pair):
    """Share price state must not be created from the placeholder trade.

    1. Open exchange account position with $1 placeholder via open_exchange_account_position
    2. Verify share_price_state is None
    3. Verify unrealised PnL is 0 (not $current_value - $1)
    """
    # 1. Open exchange account position with $1 placeholder
    state = State()
    state.portfolio.initialise_reserves(exchange_account_pair.quote, reserve_token_price=1.0)
    state.portfolio.adjust_reserves(
        exchange_account_pair.quote, Decimal(1000), "seed",
    )
    open_exchange_account_position(
        state=state,
        strategy_cycle_at=datetime.datetime(2024, 1, 1),
        pair=exchange_account_pair,
        reserve_currency=exchange_account_pair.quote,
        reserve_amount=Decimal(1),
    )
    position = list(state.portfolio.open_positions.values())[0]

    # 2. Verify share_price_state is None
    assert position.share_price_state is None

    # 3. Verify unrealised PnL is 0
    assert position.get_unrealised_profit_usd() == 0
    assert position.get_unrealised_profit_pct() == 0


def test_first_valuation_establishes_initial_capital(exchange_account_pair):
    """First valuation sync must create share_price_state with the API value as initial capital.

    1. Create position with $1 placeholder
    2. First valuation: API says $100,000
    3. Verify share_price_state is created with total_invested=$100,000
    4. Verify PnL is 0% (initial capital, not profit)
    """
    # 1. Create position with $1 placeholder
    position = _create_position_with_placeholder_trade(exchange_account_pair)
    assert position.share_price_state is None

    # 2. First valuation: API says $100,000
    _valuate(position, Decimal("100000"))

    # 3. Verify share_price_state is created with total_invested=$100,000
    sp = position.share_price_state
    assert sp is not None
    assert sp.total_invested == pytest.approx(100_000)
    assert sp.total_supply == pytest.approx(100_000)
    assert sp.initial_share_price == 1.0
    assert sp.current_share_price == pytest.approx(1.0)

    # 4. Verify PnL is 0% (initial capital, not profit)
    assert position.get_unrealised_profit_usd() == pytest.approx(0, abs=0.01)
    assert position.get_unrealised_profit_pct() == pytest.approx(0, abs=1e-6)


def test_subsequent_valuations_track_pnl(exchange_account_pair):
    """PnL must update correctly after the initial capital is established.

    1. Create position with $1 placeholder
    2. First valuation: $100,000 (initial capital)
    3. Second valuation: $105,000 (5% profit)
    4. Third valuation: $98,000 (loss from initial)
    """
    # 1. Create position with $1 placeholder
    position = _create_position_with_placeholder_trade(exchange_account_pair)

    # 2. First valuation: $100,000 (initial capital)
    _valuate(position, Decimal("100000"))
    assert position.get_unrealised_profit_usd() == pytest.approx(0, abs=0.01)
    assert position.get_unrealised_profit_pct() == pytest.approx(0, abs=1e-6)

    # 3. Second valuation: $105,000 (5% profit)
    _valuate(position, Decimal("105000"))
    assert position.get_unrealised_profit_usd() == pytest.approx(5_000, abs=0.01)
    assert position.get_unrealised_profit_pct() == pytest.approx(0.05, abs=1e-6)

    # 4. Third valuation: $98,000 (2% loss from initial)
    _valuate(position, Decimal("98000"))
    assert position.get_unrealised_profit_usd() == pytest.approx(-2_000, abs=0.01)
    assert position.get_unrealised_profit_pct() == pytest.approx(-0.02, abs=1e-6)


def test_zero_placeholder_trade(exchange_account_pair):
    """Position opened with reserve_amount=0 (strategy default) also works.

    1. Create position with $0 placeholder
    2. First valuation: $50,000
    3. Second valuation: $52,500 (5% profit)
    """
    # 1. Create position with $0 placeholder
    position = _create_position_with_placeholder_trade(
        exchange_account_pair, reserve_amount=Decimal(0),
    )

    # 2. First valuation: $50,000
    _valuate(position, Decimal("50000"))
    sp = position.share_price_state
    assert sp is not None
    assert sp.total_invested == pytest.approx(50_000)
    assert position.get_unrealised_profit_pct() == pytest.approx(0, abs=1e-6)

    # 3. Second valuation: $52,500 (5% profit)
    _valuate(position, Decimal("52500"))
    assert position.get_unrealised_profit_usd() == pytest.approx(2_500, abs=0.01)
    assert position.get_unrealised_profit_pct() == pytest.approx(0.05, abs=1e-6)


def test_migration_skips_placeholder_trade(exchange_account_pair):
    """Migration must treat the first balance update as initial capital, not the trade.

    1. Create position with $1 placeholder trade
    2. Manually add balance updates simulating: $1 -> $100,000 -> $105,000
    3. Run migrate_share_price_state
    4. Verify PnL is 5% (not 10,499,900%)
    """
    # 1. Create position with $1 placeholder trade
    position = _create_position_with_placeholder_trade(exchange_account_pair)

    t1 = datetime.datetime(2024, 1, 2)
    t2 = datetime.datetime(2024, 1, 3)

    # 2. Manually add balance updates: $1 -> $100,000 -> $105,000
    position.balance_updates[1] = BalanceUpdate(
        balance_update_id=1,
        position_type=BalanceUpdatePositionType.open_position,
        cause=BalanceUpdateCause.vault_flow,
        asset=exchange_account_pair.base,
        block_mined_at=t1,
        strategy_cycle_included_at=t1,
        chain_id=901,
        old_balance=Decimal(1),
        usd_value=99_999.0,
        quantity=Decimal("99999"),
        owner_address=None,
        tx_hash=None,
        log_index=None,
        position_id=1,
    )
    position.balance_updates[2] = BalanceUpdate(
        balance_update_id=2,
        position_type=BalanceUpdatePositionType.open_position,
        cause=BalanceUpdateCause.vault_flow,
        asset=exchange_account_pair.base,
        block_mined_at=t2,
        strategy_cycle_included_at=t2,
        chain_id=901,
        old_balance=Decimal("100000"),
        usd_value=5_000.0,
        quantity=Decimal("5000"),
        owner_address=None,
        tx_hash=None,
        log_index=None,
        position_id=1,
    )

    # 3. Run migrate_share_price_state
    migrate_share_price_state(position)

    # 4. Verify PnL is 5% (not 10,499,900%)
    assert position.share_price_state is not None
    sp = position.share_price_state
    assert sp.total_invested == pytest.approx(100_000)
    assert sp.total_supply == pytest.approx(100_000)
    assert sp.cumulative_quantity == pytest.approx(105_000)
    assert sp.current_share_price == pytest.approx(1.05)

    assert position.get_unrealised_profit_usd() == pytest.approx(5_000, abs=0.01)
    assert position.get_unrealised_profit_pct() == pytest.approx(0.05, abs=1e-6)


def test_backfill_includes_exchange_accounts(exchange_account_pair):
    """Backfill must process exchange account positions.

    1. Create state with an exchange account position that has balance updates
    2. Run backfill_share_price_state
    3. Verify share_price_state was created
    """
    # 1. Create state with exchange account position
    state = State()
    position = _create_position_with_placeholder_trade(exchange_account_pair)
    position.balance_updates[1] = BalanceUpdate(
        balance_update_id=1,
        position_type=BalanceUpdatePositionType.open_position,
        cause=BalanceUpdateCause.vault_flow,
        asset=exchange_account_pair.base,
        block_mined_at=datetime.datetime(2024, 1, 2),
        strategy_cycle_included_at=datetime.datetime(2024, 1, 2),
        chain_id=901,
        old_balance=Decimal(1),
        usd_value=49_999.0,
        quantity=Decimal("49999"),
        owner_address=None,
        tx_hash=None,
        log_index=None,
        position_id=1,
    )
    state.portfolio.open_positions[1] = position

    # 2. Run backfill
    count = backfill_share_price_state(state)

    # 3. Verify share_price_state was created
    assert count == 1
    assert position.share_price_state is not None
    assert position.share_price_state.total_invested == pytest.approx(50_000)


def test_no_balance_updates_no_migration(exchange_account_pair):
    """Migration must not crash when position has no balance updates yet.

    1. Create position with only a placeholder trade (no BUs)
    2. Run migrate_share_price_state
    3. Verify share_price_state is still None
    """
    # 1. Create position with only a placeholder trade
    position = _create_position_with_placeholder_trade(exchange_account_pair)

    # 2. Run migrate
    migrate_share_price_state(position)

    # 3. Still None — no BUs to establish initial capital
    assert position.share_price_state is None


def test_valuation_no_change_still_initialises(exchange_account_pair):
    """First valuation with zero diff must still create share_price_state.

    This covers the edge case where the placeholder matches the actual
    exchange value (unlikely but possible).

    1. Create position with $100 placeholder
    2. Valuation returns $100 (diff=0, no BU created)
    3. Verify share_price_state is initialised anyway
    """
    # 1. Create position with $100 placeholder
    position = _create_position_with_placeholder_trade(
        exchange_account_pair, reserve_amount=Decimal(100),
    )

    # 2. Valuation returns $100 (no diff)
    _valuate(position, Decimal("100"))

    # 3. Share price state initialised
    sp = position.share_price_state
    assert sp is not None
    assert sp.total_invested == pytest.approx(100)
    assert sp.total_supply == pytest.approx(100)
    assert len(position.balance_updates) == 0  # No BU since diff was 0
