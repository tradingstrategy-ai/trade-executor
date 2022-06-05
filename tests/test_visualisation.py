"""Visualisation tests."""
import datetime
from decimal import Decimal

import pytest

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.reserve import ReservePosition
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeType
from tradeexecutor.testing.trader import DummyTestTrader
from tradingstrategy.chain import ChainId


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
def weth_usdc(mock_exchange_address, usdc, weth) -> TradingPairIdentifier:
    """Mock some assets"""
    return TradingPairIdentifier(weth, usdc, "0x4", mock_exchange_address)


def test_visualise_trades_with_indicator(usdc, weth, weth_usdc):
    """Do a single token purchase."""
    state = State()
    state.update_reserves([ReservePosition(usdc, Decimal(1000), start_ts, 1.0, start_ts)])

    trader = DummyTestTrader(state)

    # Day 1
    trader.time_travel(datetime.datetime(2021, 1, 1))

    # Buy 10 ETH at 1700 USD/ETH
    trader.buy(weth_usdc, 10, 1700)



