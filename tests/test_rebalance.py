"""Test portfolio rebalancing.

"""
import datetime
from decimal import Decimal
from typing import Tuple

import pytest
from hexbytes import HexBytes

from tradeexecutor.monkeypatch.dataclasses_json import patch_dataclasses_json
from tradeexecutor.state.state import State, TradeType
from tradeexecutor.state.portfolio import NotEnoughMoney
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.trade import TradeExecution, TradeStatus
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.state.reserve import ReservePosition
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.statistics.core import update_statistics
from tradeexecutor.strategy.pandas_trader.rebalance import get_existing_portfolio_weights, rebalance_portfolio, \
    get_weight_diffs, clip_to_normalised
from tradeexecutor.testing.dummy_trader import DummyTestTrader
from tradingstrategy.chain import ChainId
from tradingstrategy.types import USDollarAmount



@pytest.fixture
def mock_exchange_address() -> str:
    """Mock some assets"""
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


@pytest.fixture
def single_asset_portfolio(start_ts, weth_usdc, weth, usdc) -> Portfolio:
    """Creates a mock portfolio that holds some reserve currency and WETH"""
    p = Portfolio()

    reserve = ReservePosition(usdc, Decimal(500), start_ts, 1.0, start_ts)
    p.reserves[reserve.get_identifier()] = reserve

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
        executed_at = start_ts,
        executed_price=1660,
        executed_quantity=Decimal(0.095),
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

    assert trade.is_buy()
    assert trade.is_success()

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

    p.open_positions = {1: position}
    p.next_trade_id = 2
    return p


def test_get_portfolio_weights(
    single_asset_portfolio: Portfolio,
    weth_usdc: TradingPairIdentifier,
):
    """"Get weights of exising portfolio."""

    portfolio = single_asset_portfolio

    weights = get_existing_portfolio_weights(portfolio)
    assert weights == {
        weth_usdc.internal_id: 1.0
    }


def test_weight_diffs(
    single_asset_portfolio: Portfolio,
    weth_usdc: TradingPairIdentifier,
    aave_usdc: TradingPairIdentifier,
):
    """"Create weight diffs between exising and new portfolio."""

    portfolio = single_asset_portfolio

    new_weights = {
        aave_usdc.internal_id: 1
    }

    diffs = get_weight_diffs(portfolio, new_weights)

    assert diffs == {
        aave_usdc.internal_id: 1,
        weth_usdc.internal_id: -1
    }


def test_weight_diffs_partial(
    single_asset_portfolio: Portfolio,
    weth_usdc: TradingPairIdentifier,
    aave_usdc: TradingPairIdentifier,
):
    """"Create 50%/50% portfolio."""

    portfolio = single_asset_portfolio

    new_weights = {
        aave_usdc.internal_id: 0.5,
        weth_usdc.internal_id: 0.5,
    }

    diffs = get_weight_diffs(portfolio, new_weights)

    assert diffs == {
        aave_usdc.internal_id: 0.5,
        weth_usdc.internal_id: -0.5
    }


def test_clip():
    """"Make sure normalised weights are summed precise 1."""

    weights = {
        0: 0.8,
        1: 0.21,
    }

    clipped = clip_to_normalised(weights)

    assert clipped == {
        0: 0.79,
        1: 0.21,
    }

