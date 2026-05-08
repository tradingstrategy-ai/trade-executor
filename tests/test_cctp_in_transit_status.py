"""Test CCTP in-transit trade status and related helper methods.

Verifies that:
- The new ``cctp_in_transit`` status is correctly resolved by ``get_status()``
- ``mark_expired()`` only works on planned trades
- ``get_execution_sort_position()`` places bridge trades in the correct phases
- ``mark_bridge_in_transit()`` sets metadata and locks bridge-back capital
"""

import datetime
from decimal import Decimal

import pytest

from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.state.identifier import (
    AssetIdentifier,
    TradingPairIdentifier,
    TradingPairKind,
)
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeStatus, TradeType


USDC_ARBITRUM_ADDRESS = "0xaf88d065e77c8cC2239327C5EDb3A432268e5831"
USDC_BASE_ADDRESS = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"


@pytest.fixture()
def usdc_arbitrum() -> AssetIdentifier:
    """USDC on Arbitrum."""
    return AssetIdentifier(
        chain_id=42161,
        address=USDC_ARBITRUM_ADDRESS,
        token_symbol="USDC",
        decimals=6,
    )


@pytest.fixture()
def usdc_base() -> AssetIdentifier:
    """USDC on Base."""
    return AssetIdentifier(
        chain_id=8453,
        address=USDC_BASE_ADDRESS,
        token_symbol="USDC",
        decimals=6,
    )


@pytest.fixture()
def cctp_pair(usdc_arbitrum: AssetIdentifier, usdc_base: AssetIdentifier) -> TradingPairIdentifier:
    """CCTP bridge pair: Arbitrum -> Base."""
    return TradingPairIdentifier(
        base=usdc_base,
        quote=usdc_arbitrum,
        pool_address="0x28b5a0e9C621a5BadaA536219b3a228C8168cf5d",
        exchange_address="0x28b5a0e9C621a5BadaA536219b3a228C8168cf5d",
        fee=0,
        kind=TradingPairKind.cctp_bridge,
        other_data={"bridge_protocol": "cctp"},
    )


def _create_bridge_trade(
    state: State,
    cctp_pair: TradingPairIdentifier,
    reserve_asset: AssetIdentifier,
    reserve_amount: Decimal,
    ts: datetime.datetime,
    *,
    sell: bool = False,
) -> TradeExecution:
    """Helper to create a bridge-out (buy) or bridge-back (sell) trade."""
    if sell:
        quantity = -reserve_amount
        reserve = None
    else:
        quantity = None
        reserve = reserve_amount

    _, trade, _ = state.create_trade(
        strategy_cycle_at=ts,
        pair=cctp_pair,
        quantity=quantity,
        reserve=reserve,
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=reserve_asset,
        reserve_currency_price=1.0,
    )
    return trade


def test_get_status_cctp_in_transit(
    cctp_pair: TradingPairIdentifier,
    usdc_arbitrum: AssetIdentifier,
):
    """Test that get_status() returns cctp_in_transit and that executed_at wins.

    1. Create a bridge-out trade and advance it to broadcasted
    2. Set cctp_in_transit_at and verify status is cctp_in_transit
    3. Set executed_at and verify status is success (executed_at takes priority)
    """
    state = State()
    ts = datetime.datetime(2025, 1, 1)

    state.portfolio.initialise_reserves(usdc_arbitrum)
    reserve = state.portfolio.get_default_reserve_position()
    reserve.quantity = Decimal(10_000)
    reserve.reserve_token_price = 1.0

    # 1. Create a bridge-out trade and advance it to broadcasted
    trade = _create_bridge_trade(state, cctp_pair, usdc_arbitrum, Decimal(500), ts)
    state.start_execution(ts, trade)
    trade.mark_broadcasted(ts)
    assert trade.get_status() == TradeStatus.broadcasted

    # 2. Set cctp_in_transit_at and verify status is cctp_in_transit
    trade.cctp_in_transit_at = ts
    assert trade.get_status() == TradeStatus.cctp_in_transit

    # 3. Set executed_at and verify status is success (executed_at takes priority)
    trade.executed_at = ts
    assert trade.get_status() == TradeStatus.success


def test_mark_expired(
    cctp_pair: TradingPairIdentifier,
    usdc_arbitrum: AssetIdentifier,
):
    """Test that mark_expired() works on planned trades and rejects other statuses.

    1. Create a planned trade
    2. Verify mark_expired() succeeds on a planned trade
    3. Create another trade, start it, and verify mark_expired() raises
    """
    state = State()
    ts = datetime.datetime(2025, 1, 1)

    state.portfolio.initialise_reserves(usdc_arbitrum)
    reserve = state.portfolio.get_default_reserve_position()
    reserve.quantity = Decimal(10_000)
    reserve.reserve_token_price = 1.0

    # 1. Create a planned trade
    trade = _create_bridge_trade(state, cctp_pair, usdc_arbitrum, Decimal(500), ts)
    assert trade.get_status() == TradeStatus.planned

    # 2. Verify mark_expired() succeeds on a planned trade
    trade.mark_expired(ts)
    assert trade.get_status() == TradeStatus.expired

    # 3. Create another trade, start it, and verify mark_expired() raises
    trade2 = _create_bridge_trade(state, cctp_pair, usdc_arbitrum, Decimal(200), ts)
    state.start_execution(ts, trade2)
    with pytest.raises(AssertionError):
        trade2.mark_expired(ts)


def test_get_execution_sort_position_cctp_bridge(
    cctp_pair: TradingPairIdentifier,
    usdc_arbitrum: AssetIdentifier,
):
    """Test that bridge trades sort to the correct phases.

    1. Create a bridge-out (buy) trade and check sort position is in +30M range
    2. Create a bridge-back (sell) trade and check sort position is in -30M range
    3. Verify bridge sell with closing=True still uses bridge phase, not close phase
    """
    state = State()
    ts = datetime.datetime(2025, 1, 1)

    state.portfolio.initialise_reserves(usdc_arbitrum)
    reserve = state.portfolio.get_default_reserve_position()
    reserve.quantity = Decimal(10_000)
    reserve.reserve_token_price = 1.0

    # 1. Create a bridge-out (buy) trade and check sort position is in +30M range
    buy_trade = _create_bridge_trade(state, cctp_pair, usdc_arbitrum, Decimal(500), ts)
    buy_sort = buy_trade.get_execution_sort_position()
    assert buy_sort == buy_trade.trade_id + 30_000_000

    # First bridge-out must complete before we can test bridge-back
    state.start_execution(ts, buy_trade)
    buy_trade.mark_broadcasted(ts)
    state.mark_trade_success(
        executed_at=ts,
        trade=buy_trade,
        executed_price=1.0,
        executed_amount=Decimal(500),
        executed_reserve=Decimal(500),
        lp_fees=0,
        native_token_price=0,
    )

    # 2. Create a bridge-back (sell) trade and check sort position is in -30M range
    sell_trade = _create_bridge_trade(
        state, cctp_pair, usdc_arbitrum, Decimal(500), ts, sell=True,
    )
    sell_sort = sell_trade.get_execution_sort_position()
    assert sell_sort == -sell_trade.trade_id - 30_000_000

    # 3. Verify bridge sell with closing=True still uses bridge phase, not close phase
    sell_trade.closing = True
    closing_sort = sell_trade.get_execution_sort_position()
    # Must still be in the bridge range, not the generic close range (-100M)
    assert closing_sort == -sell_trade.trade_id - 30_000_000


def test_mark_bridge_in_transit(
    cctp_pair: TradingPairIdentifier,
    usdc_arbitrum: AssetIdentifier,
):
    """Test that mark_bridge_in_transit() sets metadata and locks bridge-back capital.

    1. Create state with reserves and open a bridge position (bridge-out)
    2. Create a bridge-back (sell) trade and advance to broadcasted
    3. Call mark_bridge_in_transit() and verify fields and capital lock
    4. Verify bridge-out buy path sets chain IDs correctly
    """
    state = State()
    ts = datetime.datetime(2025, 1, 1)

    # 1. Create state with reserves and open a bridge position (bridge-out)
    state.portfolio.initialise_reserves(usdc_arbitrum)
    reserve = state.portfolio.get_default_reserve_position()
    reserve.quantity = Decimal(10_000)
    reserve.reserve_token_price = 1.0

    buy_trade = _create_bridge_trade(state, cctp_pair, usdc_arbitrum, Decimal(1000), ts)
    state.start_execution(ts, buy_trade)
    buy_trade.mark_broadcasted(ts)

    # Add a fake blockchain transaction so mark_bridge_in_transit can extract burn tx
    approve_tx = BlockchainTransaction()
    approve_tx.tx_hash = "0xapprove"
    burn_tx = BlockchainTransaction()
    burn_tx.tx_hash = "0xburn_abc123"
    buy_trade.blockchain_transactions = [approve_tx, burn_tx]

    # 4. Verify bridge-out buy path sets chain IDs correctly
    state.mark_bridge_in_transit(ts, buy_trade)
    assert buy_trade.cctp_in_transit_at == ts
    assert buy_trade.get_status() == TradeStatus.cctp_in_transit
    assert buy_trade.other_data["cctp_burn_tx_hash"] == "0xburn_abc123"
    assert buy_trade.other_data["cctp_source_chain_id"] == 42161  # Arbitrum (quote)
    assert buy_trade.other_data["cctp_dest_chain_id"] == 8453  # Base (base)

    # Complete the buy so we have a bridge position
    buy_trade.cctp_in_transit_at = None  # Reset for completion
    state.mark_trade_success(
        executed_at=ts,
        trade=buy_trade,
        executed_price=1.0,
        executed_amount=Decimal(1000),
        executed_reserve=Decimal(1000),
        lp_fees=0,
        native_token_price=0,
    )

    bridge_position = state.portfolio.get_position_by_id(buy_trade.position_id)
    assert bridge_position is not None
    assert bridge_position.bridge_capital_allocated == Decimal(0)

    # 2. Create a bridge-back (sell) trade and advance to broadcasted
    sell_trade = _create_bridge_trade(
        state, cctp_pair, usdc_arbitrum, Decimal(500), ts, sell=True,
    )
    state.start_execution(ts, sell_trade)
    sell_trade.mark_broadcasted(ts)

    sell_approve_tx = BlockchainTransaction()
    sell_approve_tx.tx_hash = "0xsell_approve"
    sell_burn_tx = BlockchainTransaction()
    sell_burn_tx.tx_hash = "0xsell_burn_def456"
    sell_trade.blockchain_transactions = [sell_approve_tx, sell_burn_tx]

    # 3. Call mark_bridge_in_transit() and verify fields and capital lock
    state.mark_bridge_in_transit(ts, sell_trade)
    assert sell_trade.cctp_in_transit_at == ts
    assert sell_trade.get_status() == TradeStatus.cctp_in_transit
    assert sell_trade.other_data["cctp_burn_tx_hash"] == "0xsell_burn_def456"
    assert sell_trade.other_data["cctp_source_chain_id"] == 8453  # Base (base for sell)
    assert sell_trade.other_data["cctp_dest_chain_id"] == 42161  # Arbitrum (quote for sell)

    # Verify bridge-back capital was locked
    assert bridge_position.bridge_capital_allocated == Decimal(500)
