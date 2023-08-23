"""Test opening and closing short positions.

- Test state calculations

- Fees are assumed 0% for the sake of simplifying tests
"""
import datetime
from _decimal import Decimal

import pytest

from eth_defi.uniswap_v2.utils import ZERO_ADDRESS
from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier, TradingPairKind, AssetType
from tradeexecutor.state.reserve import ReservePosition
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeType
from tradeexecutor.strategy.interest import update_credit_supply_interest
from tradeexecutor.testing.unit_test_trader import UnitTestTrader
from tradingstrategy.chain import ChainId
from tradingstrategy.lending import LendingProtocolType


@pytest.fixture()
def usdc() -> AssetIdentifier:
    """Mock USDC."""
    return AssetIdentifier(ChainId.polygon.value, "0x0", "USDC", 6)


@pytest.fixture()
def weth() -> AssetIdentifier:
    """Mock WETH."""
    return AssetIdentifier(ChainId.polygon.value, "0x2", "WETH", 18)


@pytest.fixture()
def ausdc(usdc: AssetIdentifier) -> AssetIdentifier:
    """Aave collateral."""
    return AssetIdentifier(
        ChainId.polygon.value,
        "0x3",
        "aPolUSDC",
        18,
        underlying=usdc,
        type=AssetType.collateral,
    )


@pytest.fixture()
def vweth(weth: AssetIdentifier) -> AssetIdentifier:
    """Variable debt token."""
    return AssetIdentifier(
        ChainId.polygon.value,
        "0x4",
        "variableDebtPolWETH",
        18,
        underlying=weth,
        type=AssetType.borrowed,
    )


@pytest.fixture()
def lending_protocol_address() -> str:
    """Mock some assets.

    TODO: What is unique address to identify Aave deployments?
    """
    return ZERO_ADDRESS


@pytest.fixture()
def weth_short_identifier(ausdc: AssetIdentifier, vweth: AssetIdentifier) -> TradingPairIdentifier:
    """Sets up a lending pool"""
    return TradingPairIdentifier(
        vweth,
        ausdc,
        "0x1",
        ZERO_ADDRESS,
        internal_id=1,
        kind=TradingPairKind.lending_protocol_short,
        collateral_factor=0.8,
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
    state.portfolio.reserves = {usdc.address: reserve_position}
    return state


def test_open_short(
        state: State,
        weth_short_identifier: TradingPairIdentifier,
        usdc: AssetIdentifier,
):
    """Opening a short position.

    - Supply 1000 USDC as a collateral

    - Borrow out WETH (default leverage ETH 80% of USDC value, or 0.8x)

    - Sell WETH

    - In our wallet, we have now USDC (our capital), USDC (from sold WETH),
      vWETH to mark out debt

    - Short position gains interest on the borrowed ETH (negative),
      which we can read from vWETH balance

    """
    assert weth_short_identifier.kind.is_shorting()
    assert weth_short_identifier.get_lending_protocol() == LendingProtocolType.aave_v3
    assert weth_short_identifier.base.token_symbol == "variableDebtPolWETH"
    assert weth_short_identifier.base.underlying.token_symbol == "WETH"
    assert weth_short_identifier.quote.token_symbol == "aPolUSDC"
    assert weth_short_identifier.quote.underlying.token_symbol == "USDC"
    assert weth_short_identifier.get_max_leverage_at_open() == pytest.approx(5.00)

    trader = UnitTestTrader(state)

    # Aave allows us to borrow 80% ETH against our USDC collateral
    start_ltv = 0.8

    # How many ETH (vWETH) we expect when we go in
    # with our max leverage available
    # based on the collateral ratio
    expected_eth_shorted_amount = 1000 * start_ltv / 1500

    # Take 1000 USDC reserves and open a ETH short using it.
    # We should get 800 USDC worth of ETH for this.
    short_position, trade, created = state.create_trade(
        strategy_cycle_at=datetime.datetime.utcnow(),
        pair=weth_short_identifier,
        quantity=-Decimal(expected_eth_shorted_amount),
        reserve=Decimal(1000),
        assumed_price=float(1500),  # USDC/ETH price we are going to sell
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
        reserve_currency_price=1.0,
    )

    trader.set_perfectly_executed(trade)

    assert created
    assert trade.is_success()
    assert trade.is_sell()
    assert trade.trade_type == TradeType.rebalance

    # Check loan data structures
    loan = short_position.loan
    assert loan is not None
    assert loan.pair == weth_short_identifier

    assert loan.collateral.asset.token_symbol == "aPolUSDC"
    assert loan.collateral.quantity == Decimal(1000)
    assert loan.collateral.asset.underlying.token_symbol == "USDC"
    assert loan.collateral.last_usd_price == 1.0

    assert loan.borrowed.quantity == pytest.approx(Decimal(expected_eth_shorted_amount))
    assert loan.borrowed.asset.token_symbol == "variableDebtPolWETH"
    assert loan.borrowed.asset.underlying.token_symbol == "WETH"
    assert loan.borrowed.last_usd_price == 1500
    assert loan.get_loan_to_value() == 0.8

    # Check position data structures
    assert short_position.is_short()
    assert short_position.is_open()
    assert short_position.get_opening_price() == 1500
    assert short_position.get_unrealised_profit_usd() == 0
    assert short_position.get_realised_profit_usd() == 0
    assert short_position.get_value() == 800  # -800 USD worth of ETH
    assert short_position.get_borrowed() == 800  # 800 USD worth of ETH
    assert short_position.get_equity() == 0  # Because we are not holding spot tokens, it does not count as equity
    assert short_position.get_collateral() == 1000

    # Check that we track the equity value correctly
    assert state.portfolio.get_borrowed() == 800
    assert state.portfolio.get_position_equity_and_collateral() == 1000  # 1000 USDC collateral
    assert state.portfolio.get_current_cash() == 9000
    assert state.portfolio.get_loan_net_asset_value() == 200
    assert state.portfolio.get_total_equity() == 10000


def test_short_unrealised_profit(
        state: State,
        weth_short_identifier: TradingPairIdentifier,
        usdc: AssetIdentifier,
):
    """Opening a short position and get some unrealised profit.

    - ETH price goes 1500 -> 1400 so we get unrealised PnL

    """

    trader = UnitTestTrader(state)

    # Aave allows us to borrow 80% ETH against our USDC collateral
    start_ltv = 0.8

    # How many ETH (vWETH) we expect when we go in
    # with our max leverage available
    # based on the collateral ratio
    expected_eth_shorted_amount = 1000 * start_ltv / 1500

    # Take 1000 USDC reserves and open a ETH short using it.
    # We should get 800 USDC worth of ETH for this.
    short_position, trade, created = state.create_trade(
        strategy_cycle_at=datetime.datetime.utcnow(),
        pair=weth_short_identifier,
        quantity=-Decimal(expected_eth_shorted_amount),
        reserve=Decimal(1000),
        assumed_price=float(1500),  # USDC/ETH price we are going to sell
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
        reserve_currency_price=1.0,
    )

    trader.set_perfectly_executed(trade)
    assert state.portfolio.get_total_equity() == 10000

    # ETH price 1500 -> 1400
    short_position.revalue_base_asset(
        datetime.datetime.utcnow(),
        1400.0,
    )

    # New ETH loan worth of 746.6666666666666 USD
    assert short_position.get_current_price() == 1400
    assert short_position.get_average_price() == 1500
    assert short_position.get_unrealised_profit_usd() == pytest.approx(800 - 746.6666666666666)
    assert short_position.get_realised_profit_usd() == 0

    loan = short_position.loan
    assert loan.borrowed.quantity == pytest.approx(Decimal(expected_eth_shorted_amount))
    assert loan.borrowed.asset.token_symbol == "variableDebtPolWETH"
    assert loan.borrowed.asset.underlying.token_symbol == "WETH"
    assert loan.borrowed.last_usd_price == 1400
    assert loan.get_loan_to_value() == pytest.approx(0.746666)

    # Check that we track the equity value correctly
    assert state.portfolio.get_loan_net_asset_value() == pytest.approx(253.33333333333337)
    assert state.portfolio.get_current_cash() == 9000
    assert state.portfolio.get_theoretical_value() == pytest.approx(10053.333333333334)


def test_short_unrealised_profit_partially_closed(
        state: State,
        weth_short_identifier: TradingPairIdentifier,
        usdc: AssetIdentifier,
):
    """Opening a short position and get some unrealised profit.

    - ETH price goes 1500 -> 1400 so we get unrealised PnL

    - Close 50% of this position at this price
    """

    trader = UnitTestTrader(state)

    # Aave allows us to borrow 80% ETH against our USDC collateral
    start_ltv = 0.8

    # How many ETH (vWETH) we expect when we go in
    # with our max leverage available
    # based on the collateral ratio
    expected_eth_shorted_amount = 1000 * start_ltv / 1500

    # Take 1000 USDC reserves and open a ETH short using it.
    # We should get 800 USDC worth of ETH for this.
    short_position, trade, created = state.create_trade(
        strategy_cycle_at=datetime.datetime.utcnow(),
        pair=weth_short_identifier,
        quantity=-Decimal(expected_eth_shorted_amount),
        reserve=Decimal(1000),
        assumed_price=float(1500),  # USDC/ETH price we are going to sell
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
        reserve_currency_price=1.0,
    )

    assert trade.planned_loan_update is not None
    assert trade.executed_loan_update is None
    trader.set_perfectly_executed(trade)
    assert trade.planned_loan_update is not None
    assert trade.executed_loan_update is not None

    assert state.portfolio.get_total_equity() == 10000
    loan = short_position.loan
    assert loan.borrowed.quantity == pytest.approx(Decimal(expected_eth_shorted_amount))
    assert loan.collateral.quantity == pytest.approx(1000)

    # ETH price 1500 -> 1400
    short_position.revalue_base_asset(
        datetime.datetime.utcnow(),
        1400.0,
    )

    # Close 50% of the position
    short_position_ref_2, trade_2, created = state.create_trade(
        strategy_cycle_at=datetime.datetime.utcnow(),
        pair=weth_short_identifier,
        # Position quantity for short position means reduce the position
        quantity=Decimal(expected_eth_shorted_amount / 2),  # Short position quantity is counted as negative. When we close the quantity goes towards zero from neagtive.
        reserve=-Decimal(500), # Because this is reducing short exposure, the reserve allocated for this position is negative (reduced)
        assumed_price=float(1400),  # USDC/ETH price we are going to sell
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
        reserve_currency_price=1.0,
    )

    # Loan does not change until the trade is executed
    assert short_position_ref_2 == short_position
    assert not created
    assert short_position.loan.borrowed.quantity == pytest.approx(Decimal(expected_eth_shorted_amount))

    assert trade_2.planned_loan_update is not None
    assert trade_2.executed_loan_update is None
    trader.set_perfectly_executed(trade_2)
    assert trade_2.planned_loan_update is not None
    assert trade_2.executed_loan_update is not None

    # New ETH loan worth of 746.6666666666666 USD
    assert short_position.get_current_price() == 1400
    assert short_position.get_average_price() == 1500
    assert short_position.get_unrealised_profit_usd() == pytest.approx((800 - 746.6666666666666) / 2)
    assert short_position.get_realised_profit_usd() == pytest.approx((800 - 746.6666666666666) / 2)

    # Loan is 50% reduced from 0.53 ETH to 0.26 ETH
    # Because price has decreased, the USDC value goes down faster
    loan = short_position.loan
    assert loan.collateral.quantity == pytest.approx(500)
    assert loan.collateral.get_usd_value() == pytest.approx(500)
    assert loan.borrowed.quantity == pytest.approx(Decimal(expected_eth_shorted_amount / 2))
    assert loan.borrowed.last_usd_price == 1400
    assert loan.borrowed.get_usd_value() == pytest.approx(373.333333)
    assert loan.get_loan_to_value() == pytest.approx(0.746666)

    # Check that we track the equity value correctly
    assert state.portfolio.get_loan_net_asset_value() == pytest.approx(253.33333333333337 / 2)
    assert state.portfolio.get_current_cash() == 9500 + short_position.get_realised_profit_usd()
    assert state.portfolio.get_total_equity() == pytest.approx(10_000 + short_position.get_realised_profit_usd())  # Any profits from closed short positions are moved to equity
    assert state.portfolio.get_theoretical_value() == pytest.approx(10053.333333333334)
