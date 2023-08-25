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
from tradeexecutor.testing.unit_test_trader import UnitTestTrader
from tradingstrategy.chain import ChainId


@pytest.fixture()
def usdc() -> AssetIdentifier:
    """Mock USDC."""
    return AssetIdentifier(ChainId.ethereum.value, "0x0", "USDC", 6)


@pytest.fixture()
def ausdc(usdc) -> AssetIdentifier:
    """Mock aUSDC. Aave's interest accruing USDC where balanceOf() is dynamic."""
    # https://etherscan.io/token/0xbcca60bb61934080951369a648fb03df4f96263c#readProxyContract
    return AssetIdentifier(
        ChainId.ethereum.value,
        "0x1",
        "aUSDC",
        6,
        underlying=usdc,
    )


@pytest.fixture()
def lending_protocol_address() -> str:
    """Mock some assets.

    TODO: What is unique address to identify Aave deployments?
    """
    return ZERO_ADDRESS


@pytest.fixture()
def lending_pool_identifier(usdc, ausdc) -> TradingPairIdentifier:
    """Sets up a lending pool"""

    #
    # For "credit only"
    # position both base and quote of the trading pair identifier
    # are the atoken
    #
    return TradingPairIdentifier(
        ausdc,
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
    assert lending_pool_identifier.base.token_symbol == "aUSDC"
    assert lending_pool_identifier.base.underlying.token_symbol == "USDC"
    assert lending_pool_identifier.quote.token_symbol == "aUSDC"
    assert lending_pool_identifier.quote.underlying.token_symbol == "USDC"

    trader = UnitTestTrader(state)

    credit_supply_position, trade, created = state.supply_credit(
        datetime.datetime.utcnow(),
        lending_pool_identifier,
        collateral_quantity=Decimal(9000),
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
        collateral_asset_price=1.0,
    )

    assert trade.planned_loan_update
    trader.set_perfectly_executed(trade)
    assert trade.executed_loan_update

    assert credit_supply_position.last_token_price == 1.0

    assert created
    assert trade.is_success()
    assert trade.is_credit_supply()

    interest = credit_supply_position.loan.collateral_interest

    assert interest is not None
    assert interest.opening_amount == Decimal(9000)
    assert interest.last_accrued_interest == 0
    assert credit_supply_position.get_value() == Decimal(9000)

    loan = credit_supply_position.loan
    assert loan.get_net_asset_value() == 9000
    assert loan.collateral.get_usd_value() == 9000
    assert loan.get_collateral_interest() == 0

    assert credit_supply_position.get_value() == pytest.approx(9000)
    assert state.portfolio.get_net_asset_value() == 10000


def test_accrue_interest(
        state: State,
        lending_pool_identifier: TradingPairIdentifier,
        usdc: AssetIdentifier,
        ausdc: AssetIdentifier,
):
    """See that the credit supply position gains interest.

    """
    opened_at = datetime.datetime(2020, 1, 1)

    trader = UnitTestTrader(state)

    # Open credit supply position
    credit_supply_position, trade, _ = state.create_trade(
        opened_at,
        lending_pool_identifier,
        quantity=None,
        reserve=Decimal(9000),
        assumed_price=1.0,
        trade_type=TradeType.supply_credit,
        reserve_currency=usdc,
        reserve_currency_price=1.0,
    )
    trader.set_perfectly_executed(trade)

    assert credit_supply_position.get_value() == pytest.approx(9000)

    # Generate first interest accruing event
    interest_event_1_at = datetime.datetime(2020, 1, 2)
    update_credit_supply_interest(
        state,
        credit_supply_position,
        ausdc,
        new_atoken_amount=Decimal(9000.01),
        event_at=interest_event_1_at,
    )

    # The position has refreshed its accrued interest once
    assert len(credit_supply_position.balance_updates) == 1
    assert credit_supply_position.balance_updates[1].quantity == pytest.approx(Decimal(0.01))

    # Position and portfolio valuations reflect the accrued interest
    loan = credit_supply_position.loan
    interest = loan.collateral_interest
    assert interest.last_accrued_interest == pytest.approx(Decimal(0.01))
    assert interest.last_event_at == interest_event_1_at

    assert loan.get_collateral_interest() == pytest.approx(0.01)
    assert loan.get_borrow_interest() == 0
    assert loan.get_net_interest() == pytest.approx(0.01)

    assert credit_supply_position.get_value() == pytest.approx(9000.01)
    assert credit_supply_position.get_quantity() == pytest.approx(Decimal(9000.01))

    assert state.portfolio.get_net_asset_value(include_interest=True) == 10000.01
    assert state.portfolio.get_net_asset_value(include_interest=False) == 10000.00
