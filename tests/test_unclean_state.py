"""Unclean state checks.

Check for the conditions that could be "unclean" state that needs manual intervetion
to get the accouting back to the track.

We will do tests by constructing a problematic state by hand.
"""
import datetime
from decimal import Decimal

import pytest
from hexbytes import HexBytes
from tradingstrategy.chain import ChainId

from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.reserve import ReservePosition
from tradeexecutor.state.state import State, UncleanState
from tradeexecutor.state.trade import TradeType, TradeExecution


@pytest.fixture
def mock_exchange_address() -> str:
    """Mock an exchange"""
    return "0x1"


@pytest.fixture
def usdc() -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(ChainId.ethereum.value, "0x0", "USDC", 6)


@pytest.fixture
def weth() -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(ChainId.ethereum.value, "0x1", "WETH", 18)


@pytest.fixture
def aave() -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(ChainId.ethereum.value, "0x3", "AAVE", 18)


@pytest.fixture
def weth_usdc(mock_exchange_address, usdc, weth) -> TradingPairIdentifier:
    """Mock some assets"""
    return TradingPairIdentifier(weth, usdc, "0x4", mock_exchange_address, internal_id=1)


@pytest.fixture
def aave_usdc(mock_exchange_address, usdc, aave) -> TradingPairIdentifier:
    """Mock some assets"""
    return TradingPairIdentifier(aave, usdc, "0x5", mock_exchange_address, internal_id=2)


@pytest.fixture
def start_ts(usdc, weth) -> datetime.datetime:
    """Timestamp of action started"""
    return datetime.datetime(2022, 1, 1, tzinfo=None)


def test_broadcasted_trade(start_ts, weth_usdc, weth, usdc):
    """Clean check fails if we have broadcasted, but not confirmed trades."""

    portfolio = Portfolio()

    reserve = ReservePosition(usdc, Decimal(500), start_ts, 1.0, start_ts)
    portfolio.reserves[reserve.get_identifier()] = reserve

    # Create a trade that was broadcasted,
    # but not confirmed
    trade = TradeExecution(
        trade_id = 1,
        position_id =1,
        trade_type = TradeType.rebalance,
        pair=weth_usdc,
        opened_at = start_ts,
        planned_quantity = Decimal(0.1),
        planned_price=1670.0,
        planned_reserve=Decimal(167),
        reserve_currency = usdc,
        started_at = start_ts,
        reserve_currency_allocated = 167,
        broadcasted_at =start_ts,
        lp_fees_paid =2.5,
        native_token_price=1.9,
    )

    tx = BlockchainTransaction(
        tx_hash=HexBytes("0x01"),
        nonce=1,
        realised_gas_units_consumed=150_000,
        realised_gas_price=15,
    )
    trade.blockchain_transactions = [tx]

    assert not trade.is_success()
    assert not trade.is_success()
    assert trade.is_unfinished()

    position = TradingPosition(
        position_id=1,
        pair=weth_usdc,
        opened_at=start_ts,
        last_token_price=1660,
        last_reserve_price=1,
        last_pricing_at=start_ts,
        reserve_currency=usdc,
        trades={1: trade},
    )

    portfolio.open_positions = {1: position}

    state = State(portfolio=portfolio)

    with pytest.raises(UncleanState):
        state.check_if_clean()
