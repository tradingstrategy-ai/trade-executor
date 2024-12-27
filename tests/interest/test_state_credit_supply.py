"""Unit tests for interest calculations.

Not connected to any data source, purely stress state functions.
"""
import datetime
from _decimal import Decimal

import pytest

from eth_defi.abi import ZERO_ADDRESS
from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier, TradingPairKind
from tradeexecutor.state.reserve import ReservePosition
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeType
from tradeexecutor.strategy.interest import update_interest
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
        usdc,
        "0x1",
        ZERO_ADDRESS,
        internal_id=1,
        kind=TradingPairKind.credit_supply,
    )


@pytest.fixture()
def state(usdc: AssetIdentifier):
    """Set up a state with a starting balance."""
    state = State()
    reserve_position = ReservePosition(
        usdc,
        Decimal(10_000),
        reserve_token_price=1,
        last_pricing_at=datetime.datetime.utcnow(),
        last_sync_at=datetime.datetime.utcnow(),
    )
    state.portfolio.reserves = {usdc.get_identifier(): reserve_position}
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
    assert lending_pool_identifier.quote.token_symbol == "USDC"

    trader = UnitTestTrader(state)

    credit_supply_position, trade, created = state.supply_credit(
        datetime.datetime.utcnow(),
        lending_pool_identifier,
        collateral_quantity=Decimal(9000),
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
        collateral_asset_price=1.0,
    )

    assert trade.planned_quantity == Decimal(9000)  # Amount of aToken received
    assert trade.planned_reserve == Decimal(9000)  # Amount of USDC deposited

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
    credit_supply_position, trade, _ = state.supply_credit(
        opened_at,
        lending_pool_identifier,
        collateral_quantity=Decimal(9000),
        collateral_asset_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
    )
    trader.set_perfectly_executed(trade)

    assert credit_supply_position.get_value() == pytest.approx(9000)

    # Generate first interest accruing event
    interest_event_1_at = datetime.datetime(2020, 1, 2)
    update_interest(
        state,
        credit_supply_position,
        ausdc,
        new_token_amount=Decimal(9000.01),
        event_at=interest_event_1_at,
        asset_price=1.0,
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


def test_close_credit_position(
        state: State,
        lending_pool_identifier: TradingPairIdentifier,
        usdc: AssetIdentifier,
        ausdc: AssetIdentifier,
):
    """We close a credit position and return any accrued interest to reserves.

    """
    opened_at = datetime.datetime(2020, 1, 1)

    trader = UnitTestTrader(state)
    portfolio = state.portfolio

    # Open credit supply position
    credit_supply_position, trade, _ = state.supply_credit(
        opened_at,
        lending_pool_identifier,
        collateral_quantity=Decimal(9000.00),
        collateral_asset_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
    )
    trader.set_perfectly_executed(trade)
    assert portfolio.get_cash() == 1000
    assert portfolio.get_all_loan_nav() == 9000
    assert portfolio.get_net_asset_value() == 10000.00

    # Generate first interest accruing event
    interest_event_1_at = datetime.datetime(2020, 1, 2)
    update_interest(
        state,
        credit_supply_position,
        ausdc,
        new_token_amount=Decimal(9000.50),
        event_at=interest_event_1_at,
        asset_price=1.0,
    )

    assert state.portfolio.get_net_asset_value() == 10000.50
    assert credit_supply_position.get_unrealised_profit_usd() == 0.50

    # Close credit supply position
    #
    # This trade will bring the collateral
    # quantity to negative, but it's ok
    # because the difference will be taken from aTokens
    # when we redeem accured aToken interest to USDC
    #
    _, trade_2, _ = state.supply_credit(
        opened_at,
        lending_pool_identifier,
        collateral_quantity=-Decimal(9000.50),
        collateral_asset_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
        closing=True,
    )

    assert not trade_2.is_spot()
    assert trade_2.is_credit_based()

    trader.set_perfectly_executed(trade_2)

    # The closing trade claims the interest
    assert credit_supply_position.is_closed()
    assert trade_2.is_success()
    assert trade_2.claimed_interest == Decimal(0.50)
    assert trade_2.get_claimed_interest() == 0.50
    assert trade_2.planned_quantity == -Decimal(9000.50)
    assert trade_2.executed_reserve == Decimal(9000.50)

    # Loan is now repaid
    loan = credit_supply_position.loan
    assert loan.collateral_interest.interest_payments == Decimal(0.50)
    assert loan.collateral_interest.get_remaining_interest() == 0
    assert loan.get_collateral_value() == 0.0  # TODO: This value is incorrect as loan object currently does not track repaid interest perfectly
    assert loan.get_net_asset_value(include_interest=True) == 0.0
    assert loan.get_net_asset_value(include_interest=False) == 0.0

    assert credit_supply_position.get_claimed_interest() == 0.50
    assert credit_supply_position.get_value() == 0
    assert credit_supply_position.get_unrealised_profit_usd() == 0
    assert credit_supply_position.get_realised_profit_usd() == 0.50
    assert credit_supply_position.get_accrued_interest() == 0

    # All profits are in portfolio cash
    portfolio = state.portfolio
    assert portfolio.get_cash() == pytest.approx(10000.50)
    assert portfolio.get_net_asset_value() == pytest.approx(10000.50)


def test_accrue_unrealistic_interest(
    state: State,
    lending_pool_identifier: TradingPairIdentifier,
    usdc: AssetIdentifier,
    ausdc: AssetIdentifier,
):
    """See that the credit supply position gains unrealistic interest should raise error
    """
    opened_at = datetime.datetime(2020, 1, 1)

    trader = UnitTestTrader(state)

    # Open credit supply position
    credit_supply_position, trade, _ = state.supply_credit(
        opened_at,
        lending_pool_identifier,
        collateral_quantity=Decimal(9000),
        collateral_asset_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
    )
    trader.set_perfectly_executed(trade)

    assert credit_supply_position.get_value() == pytest.approx(9000)

    # Generate first interest accruing event
    interest_event_1_at = datetime.datetime(2020, 1, 2)

    # negative interest
    with pytest.raises(AssertionError) as e:
        update_interest(
            state,
            credit_supply_position,
            ausdc,
            new_token_amount=Decimal(5_000_000),
            event_at=interest_event_1_at,
            asset_price=1.0,
        )

    assert str(e.value) == "Unlikely gained_interest for <aUSDC (USDC) at 0x1>: 4991000 (diff 55455.56%, threshold 5.0%), old quantity: 9000, new quantity: 5000000"
