"""Test trade execution sort ordering.

Verify that get_execution_sort_position() produces the correct
execution order for all trade types including vault deposits
and withdrawals.
"""
import datetime
from decimal import Decimal

import pytest
from tradingstrategy.chain import ChainId

from tradeexecutor.state.identifier import (
    AssetIdentifier,
    TradingPairIdentifier,
    TradingPairKind,
)
from tradeexecutor.state.trade import TradeExecution, TradeFlag, TradeType
from tradeexecutor.strategy.runner import prepare_sorted_trades


@pytest.fixture
def usdc():
    return AssetIdentifier(ChainId.ethereum.value, "0x0000000000000000000000000000000000000000", "USDC", 6)


@pytest.fixture
def spot_pair(usdc):
    weth = AssetIdentifier(ChainId.ethereum.value, "0x0000000000000000000000000000000000000001", "WETH", 18)
    return TradingPairIdentifier(
        base=weth, quote=usdc,
        pool_address="0x0000000000000000000000000000000000000010",
        exchange_address="0x0000000000000000000000000000000000000020",
        internal_id=1, kind=TradingPairKind.spot_market_hold, fee=0,
    )


@pytest.fixture
def credit_pair(usdc):
    ausdc = AssetIdentifier(ChainId.ethereum.value, "0x0000000000000000000000000000000000000002", "aUSDC", 6)
    return TradingPairIdentifier(
        base=ausdc, quote=usdc,
        pool_address="0x0000000000000000000000000000000000000011",
        exchange_address="0x0000000000000000000000000000000000000021",
        internal_id=2, kind=TradingPairKind.credit_supply, fee=0,
    )


@pytest.fixture
def vault_pair(usdc):
    vault_token = AssetIdentifier(ChainId.ethereum.value, "0x0000000000000000000000000000000000000003", "hlVault", 6)
    return TradingPairIdentifier(
        base=vault_token, quote=usdc,
        pool_address="0x0000000000000000000000000000000000000012",
        exchange_address="0x0000000000000000000000000000000000000022",
        internal_id=3, kind=TradingPairKind.vault, fee=0,
    )


def _make_trade(trade_id, pair, quantity, reserve_currency, flags=None, closing=None):
    ts = datetime.datetime(2025, 1, 1)
    return TradeExecution(
        trade_id=trade_id,
        position_id=trade_id,
        trade_type=TradeType.rebalance,
        pair=pair,
        opened_at=ts,
        planned_quantity=Decimal(quantity),
        planned_price=1.0,
        planned_reserve=Decimal(abs(quantity)),
        reserve_currency=reserve_currency,
        flags=flags or set(),
        closing=closing,
    )


def test_vault_sort_ordering(spot_pair, credit_pair, vault_pair, usdc):
    """Verify the full 7-tier execution order including vault trades."""
    credit_withdraw = _make_trade(1, credit_pair, -100, usdc, flags={TradeFlag.close})
    close_trade = _make_trade(2, spot_pair, -50, usdc, closing=True)
    vault_withdraw = _make_trade(3, vault_pair, -200, usdc)
    spot_sell = _make_trade(4, spot_pair, -75, usdc)
    spot_buy = _make_trade(5, spot_pair, 100, usdc)
    vault_deposit = _make_trade(6, vault_pair, 300, usdc)
    credit_supply = _make_trade(7, credit_pair, 500, usdc, flags={TradeFlag.open})

    # Deliberately scrambled input order
    trades = [spot_buy, vault_deposit, credit_supply, vault_withdraw, spot_sell, close_trade, credit_withdraw]
    sorted_trades = prepare_sorted_trades(trades)

    assert sorted_trades == [
        credit_withdraw,
        close_trade,
        vault_withdraw,
        spot_sell,
        spot_buy,
        vault_deposit,
        credit_supply,
    ]


def test_vault_withdraw_before_deposit(vault_pair, usdc):
    """Vault withdrawals must execute before vault deposits
    regardless of trade_id assignment."""
    deposit = _make_trade(1, vault_pair, 500, usdc)
    withdraw = _make_trade(10, vault_pair, -200, usdc)

    sorted_trades = prepare_sorted_trades([deposit, withdraw])

    assert sorted_trades[0] is withdraw
    assert sorted_trades[1] is deposit
