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
from tradeexecutor.strategy.interest import update_credit_supply_interest
from tradeexecutor.testing.dummy_trader import DummyTestTrader
from tradingstrategy.chain import ChainId


@pytest.fixture()
def usdc() -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(ChainId.ethereum.value, "0x0", "USDC", 6)

@pytest.fixture()
def ausdc() -> AssetIdentifier:
    """Mock some assets"""
    # https://etherscan.io/address/0x98c23e9d8f34fefb1b7bd6a91b7ff122f4e16f5c#readProxyContract
    return AssetIdentifier(ChainId.ethereum.value, "0x1", "aEthUSDC", 6)


@pytest.fixture()
def lending_pool_address() -> str:
    """Mock Aave v3 pool"""
    # https://etherscan.io/address/0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2
    return ZERO_ADDRESS


@pytest.fixture()
def lending_reserve_identifier(usdc, ausdc, lending_pool_address) -> TradingPairIdentifier:
    """Sets up a lending reserve"""
    # https://etherscan.io/token/0x98c23e9d8f34fefb1b7bd6a91b7ff122f4e16f5c
    return TradingPairIdentifier(
        ausdc,
        usdc,
        "0x1",
        lending_pool_address,
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
        lending_reserve_identifier: TradingPairIdentifier,
        usdc: AssetIdentifier,
):
    """Open a credit supply position.

    Check that the position variables are correctly initialised.
    """
    assert lending_reserve_identifier.kind.is_interest_accruing()
    assert lending_reserve_identifier.base.token_symbol == "aEthUSDC"
    assert lending_reserve_identifier.quote.token_symbol == "USDC"

    trader = DummyTestTrader(state)

    credit_supply_position, trade, created = state.create_trade(
        datetime.datetime.utcnow(),
        lending_reserve_identifier,
        quantity=None,
        reserve=Decimal(9000),
        assumed_price=1.0,
        trade_type=TradeType.supply_credit,
        reserve_currency=usdc,
        reserve_currency_price=1.0,
    )

    trader.set_perfectly_executed(trade)

    assert credit_supply_position.last_token_price == 1.0

    assert created
    assert trade.is_success()
    assert trade.trade_type == TradeType.supply_credit
    assert credit_supply_position.interest is not None
    assert credit_supply_position.interest.opening_amount == Decimal(9000)
    assert credit_supply_position.interest.last_accrued_interest == 0
    assert credit_supply_position.get_value() == Decimal(9000)

    assert state.portfolio.get_total_equity() == 10000


def test_accrue_interest(
        state: State,
        lending_reserve_identifier: TradingPairIdentifier,
        usdc: AssetIdentifier,
):
    """See that the credit supply position gains interest.

    """
    opened_at = datetime.datetime(2020, 1, 1)

    trader = DummyTestTrader(state)

    credit_supply_position, trade, _ = state.create_trade(
        opened_at,
        lending_reserve_identifier,
        quantity=None,
        reserve=Decimal(9000),
        assumed_price=1.0,
        trade_type=TradeType.supply_credit,
        reserve_currency=usdc,
        reserve_currency_price=1.0,
    )

    trader.set_perfectly_executed(trade)

    assert credit_supply_position.get_value() == pytest.approx(9000)

    interest_event_1_at = datetime.datetime(2020, 1, 2)
    update_credit_supply_interest(
        state,
        credit_supply_position,
        new_atoken_amount=Decimal(9000.01),
        event_at=interest_event_1_at,
    )

    assert credit_supply_position.interest.last_accrued_interest == pytest.approx(Decimal(0.01))
    assert credit_supply_position.interest.last_event_at == interest_event_1_at

    assert credit_supply_position.get_value() == pytest.approx(9000.01)
    assert credit_supply_position.get_quantity() == pytest.approx(Decimal(9000.01))

    assert state.portfolio.get_total_equity() == 10000.01
