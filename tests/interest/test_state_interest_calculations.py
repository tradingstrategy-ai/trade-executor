"""Unit tests for interest calculations.

Not connected to any data source, purely stress state functions.
"""
import datetime
from _decimal import Decimal

import pytest

from eth_defi.uniswap_v2.utils import ZERO_ADDRESS
from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier, TradingPairKind
from tradeexecutor.state.reserve import ReservePosition
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeType
from tradingstrategy.chain import ChainId


@pytest.fixture()
def usdc() -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(ChainId.ethereum.value, "0x0", "USDC", 6)

@pytest.fixture()
def ausdc() -> AssetIdentifier:
    """Mock some assets"""
    # https://etherscan.io/token/0xbcca60bb61934080951369a648fb03df4f96263c#readProxyContract
    return AssetIdentifier(ChainId.ethereum.value, "0x1", "aUSDC", 6)


@pytest.fixture()
def lending_protocol_address() -> str:
    """Mock some assets.

    TODO: What is unique address to identify Aave deployments?
    """
    return ZERO_ADDRESS


@pytest.fixture()
def lending_pool_identifier(usdc, ausdc) -> TradingPairIdentifier:
    """Sets up a lending pool"""
    # https://etherscan.io/token/0xbcca60bb61934080951369a648fb03df4f96263c
    return TradingPairIdentifier(
        usdc,
        ausdc,
        "0x1",
        ZERO_ADDRESS,
        internal_id=1,
        kind=TradingPairKind.credit_supply,
    )


@pytest.fixture()
def state(usdc):
    """Set up a state with a starting balance."""
    state = State()
    reserve_position = ReservePosition(
        usdc,
        Decimal(10_000),
        reserve_token_price=1,
        last_pricing_at=datetime.datetime.utcnow(),
        last_sync_at=datetime.datetime.utcnow(),
    )
    state.portfolio.reserves = {usdc.address: reserve_position}
    return state


def test_open_supply_credit(
        state: State,
        lending_pool_identifier: TradingPairIdentifier,
        usdc: AssetIdentifier,
):
    """Open a credit supply position.

    Check that the position variables are correctly initialised.
    """
    assert lending_pool_identifier.kind.is_interest_accruing()

    credit_supply_position, trade, created = state.create_trade(
        datetime.datetime.utcnow(),
        lending_pool_identifier,
        quantity=None,
        reserve=Decimal(9000),
        assumed_price=1.0,
        trade_type=TradeType.supply_credit,
        reserve_currency=usdc,
        reserve_currency_price=1.0,
    )

    assert created
    assert credit_supply_position.interest is not None
    assert credit_supply_position.interest.opening_amount == Decimal(9000)
    assert credit_supply_position.interest.last_accrued_interest == 0

    assert trade.trade_type == TradeType.supply_credit



