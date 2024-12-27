"""Test opening and closing short positions.

- Unit test state calculations i.e. now data used

- Any fees are assumed 0% for the sake of simplifying tests

"""
import datetime
from _decimal import Decimal

import pytest

from eth_defi.abi import ZERO_ADDRESS
from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier, TradingPairKind, AssetType
from tradeexecutor.utils.leverage_calculations import LeverageEstimate
from tradeexecutor.state.reserve import ReservePosition
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeType
from tradeexecutor.strategy.interest import estimate_interest, update_leveraged_position_interest, prepare_interest_distribution
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
        liquidation_threshold=0.85,  # From Aave UI
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
def weth_usdc_spot(weth: AssetIdentifier, usdc: AssetIdentifier) -> TradingPairIdentifier:
    """Sets up a lending pool short trading pair 0% fee"""
    return TradingPairIdentifier(
        weth,
        usdc,
        "0x1",
        ZERO_ADDRESS,
        internal_id=2,
        kind=TradingPairKind.lending_protocol_short,
        fee=0,  # See
    )



@pytest.fixture()
def weth_short_identifier(ausdc: AssetIdentifier, vweth: AssetIdentifier, weth_usdc_spot) -> TradingPairIdentifier:
    """Sets up a lending pool short trading pair 0% fee"""
    return TradingPairIdentifier(
        vweth,
        ausdc,
        "0x2",
        ZERO_ADDRESS,
        internal_id=3,
        kind=TradingPairKind.lending_protocol_short,
        underlying_spot_pair=weth_usdc_spot,
    )


@pytest.fixture()
def weth_short_identifier_5bps(ausdc: AssetIdentifier, vweth: AssetIdentifier, weth_usdc_spot) -> TradingPairIdentifier:
    """Sets up a lending pool 5 BPS fee"""
    weth_usdc_spot.fee = 0.0005
    return TradingPairIdentifier(
        vweth,
        ausdc,
        "0x3",
        ZERO_ADDRESS,
        internal_id=4,
        kind=TradingPairKind.lending_protocol_short,
        underlying_spot_pair=weth_usdc_spot,
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
    assert weth_short_identifier.get_max_leverage_at_open() == pytest.approx(5.6666666666)

    trader = UnitTestTrader(state)

    # Aave allows us to borrow 80% ETH against our USDC collateral
    start_ltv = 0.8

    # How many ETH (vWETH) we expect when we go in
    # with our max leverage available
    # based on the collateral ratio
    expected_eth_shorted_amount = 1000 * start_ltv / 1500

    # Take 1000 USDC reserves and open a ETH short using it.
    # We should get 800 USDC worth of ETH for this.
    short_position, trade, created = state.trade_short(
        strategy_cycle_at=datetime.datetime.utcnow(),
        pair=weth_short_identifier,
        borrowed_quantity=-Decimal(expected_eth_shorted_amount),
        collateral_quantity=Decimal(1000),
        borrowed_asset_price=float(1500),  # USDC/ETH price we are going to sell
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
        collateral_asset_price=1.0,
    )

    trader.set_perfectly_executed(trade)

    assert created
    assert trade.is_success()
    assert trade.is_sell()
    assert trade.trade_type == TradeType.rebalance
    assert trade.executed_collateral_allocation == trade.planned_collateral_allocation
    assert trade.executed_collateral_consumption == trade.executed_collateral_consumption

    # Check loan data structures
    loan = short_position.loan
    assert loan is not None
    assert loan.pair == weth_short_identifier

    assert loan.collateral.asset.token_symbol == "aPolUSDC"
    assert loan.collateral.quantity == pytest.approx(Decimal(1800))
    assert loan.collateral.asset.underlying.token_symbol == "USDC"
    assert loan.collateral.last_usd_price == 1.0

    assert loan.borrowed.quantity == pytest.approx(Decimal(expected_eth_shorted_amount))
    assert loan.borrowed.asset.token_symbol == "variableDebtPolWETH"
    assert loan.borrowed.asset.underlying.token_symbol == "WETH"
    assert loan.borrowed.last_usd_price == 1500
    assert loan.get_loan_to_value() == pytest.approx(0.44444444)
    assert loan.get_health_factor() == pytest.approx(1.9125)
    assert loan.get_leverage() == pytest.approx(0.8)

    # Check position data structures
    assert short_position.is_short()
    assert short_position.is_open()
    assert short_position.get_opening_price() == 1500
    assert short_position.get_unrealised_profit_usd() == 0
    assert short_position.get_realised_profit_usd() is None
    assert short_position.get_value() == 1000  # 1800 USD collateral - 800 USD ETH
    assert short_position.get_borrowed() == 800  # 800 USD worth of ETH
    assert short_position.get_equity() == 0  # Because we are not holding spot tokens, it does not count as equity
    assert short_position.get_collateral() == 1800

    # Check that we track the equity value correctly
    assert state.portfolio.get_borrowed() == 800
    assert state.portfolio.get_position_equity_and_loan_nav() == 1000  # 1000 USDC collateral
    assert state.portfolio.get_cash() == 9000
    assert state.portfolio.get_loan_net_asset_value() == 1000
    assert state.portfolio.get_total_equity() == 10000

    apolusdc = loan.collateral.asset
    vdebtpolweth = loan.borrowed.asset



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
    short_position, trade, created = state.trade_short(
        strategy_cycle_at=datetime.datetime.utcnow(),
        pair=weth_short_identifier,
        borrowed_quantity=-Decimal(expected_eth_shorted_amount),
        collateral_quantity=Decimal(1000),
        borrowed_asset_price=float(1500),  # USDC/ETH price we are going to sell
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
        collateral_asset_price=1.0,
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
    assert short_position.get_realised_profit_usd() is None

    loan = short_position.loan
    assert loan.borrowed.quantity == pytest.approx(Decimal(expected_eth_shorted_amount))
    assert loan.borrowed.asset.token_symbol == "variableDebtPolWETH"
    assert loan.borrowed.asset.underlying.token_symbol == "WETH"
    assert loan.borrowed.last_usd_price == 1400
    assert loan.get_loan_to_value() == pytest.approx(0.4148148148148148)

    # Check that we track the equity value correctly
    assert state.portfolio.get_loan_net_asset_value() == pytest.approx(1053.3333333333335)
    assert state.portfolio.get_cash() == 9000
    assert state.portfolio.get_net_asset_value() == pytest.approx(10053.333333333334)


def test_short_unrealised_profit_partially_closed_keep_collateral(
        state: State,
        weth_short_identifier: TradingPairIdentifier,
        usdc: AssetIdentifier,
):
    """Opening a short position and get some unrealised profit.

    - ETH price goes 1500 -> 1400 so we get USD 56 unrealised PnL

    - Close 50% of this position at this price, we get USD 27 realised profit

    - Any closed profit is kept on the collateral
    """

    trader = UnitTestTrader(state)

    portfolio = state.portfolio

    # Start by shorting  ETH 2:1 with 1000 USD
    expected_eth_shorted_amount = 1000 / 1500

    # Take 1000 USDC reserves and open a ETH short using it.
    # We should get 800 USDC worth of ETH for this.
    # 800 USDC worth of ETH is
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

    loan = short_position.loan
    assert loan.borrowed.quantity == pytest.approx(Decimal(expected_eth_shorted_amount))
    assert loan.collateral.quantity == pytest.approx(2000)
    assert loan.get_leverage() == 1.0

    assert portfolio.get_total_equity() == 10000
    assert portfolio.get_cash() == 9000

    # ETH price 1500 -> 1400
    short_position.revalue_base_asset(
        datetime.datetime.utcnow(),
        1400.0,
    )

    # Loan value 1000 USD -> 933 USD
    assert short_position.loan.borrowed.get_usd_value() == pytest.approx(933.3333333333333)

    # Close 50% of the position
    #
    short_position_ref_2, trade_2, created = state.trade_short(
        strategy_cycle_at=datetime.datetime.utcnow(),
        pair=weth_short_identifier,
        # Position quantity for short position means reduce the position
        borrowed_quantity=Decimal(expected_eth_shorted_amount / 2),  # Short position quantity is counted as negative. When we close the quantity goes towards zero from neagtive.
        collateral_quantity=Decimal(0),  # Reserve will be calculated generated from the released collateral
        borrowed_asset_price=float(1400),  # USDC/ETH price we are going to sell
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
        collateral_asset_price=1.0,
    )

    # Loan does not change until the trade is executed
    assert short_position_ref_2 == short_position
    assert not created
    assert short_position.loan.borrowed.quantity == pytest.approx(Decimal(expected_eth_shorted_amount))

    # Trade 2 will be excuted,
    # planned loan update moves sto executed
    # we get rid of half of the position
    assert trade_2.planned_loan_update is not None
    assert trade_2.executed_loan_update is None
    trader.set_perfectly_executed(trade_2)
    assert trade_2.planned_loan_update is not None
    assert trade_2.executed_loan_update is not None
    assert trade_2.planned_quantity == pytest.approx(Decimal(expected_eth_shorted_amount / 2))
    assert trade_2.executed_quantity == pytest.approx(Decimal(expected_eth_shorted_amount / 2))
    assert trade_2.planned_reserve == pytest.approx(Decimal(0))
    assert trade_2.executed_reserve == pytest.approx(Decimal(0))

    # Because we closed half, we realised 50% of profit, left 50% profit on the table.
    total_profit = (1500-1400) * expected_eth_shorted_amount
    assert total_profit == pytest.approx(66.6666666)
    assert short_position.get_current_price() == 1400
    assert short_position.get_average_price() == 1500
    assert short_position.get_unrealised_profit_usd() == pytest.approx(total_profit / 2)
    assert short_position.get_realised_profit_usd() == pytest.approx(total_profit / 2)

    # Loan is 50% reduced from 0.53 ETH to 0.26 ETH.
    # We buy the ETH to repay using the USDC loan from collateral
    loan = short_position.loan
    released_collateral = 1400 * expected_eth_shorted_amount / 2
    assert released_collateral == pytest.approx(466.66666666666663)
    left_collateral = Decimal(2000 - released_collateral)
    assert left_collateral == pytest.approx(Decimal(1533.33333333333348491578362882137298583984375))
    assert loan.collateral.quantity == pytest.approx(left_collateral)
    assert loan.collateral.get_usd_value() == pytest.approx(float(left_collateral))
    assert loan.borrowed.quantity == pytest.approx(Decimal(expected_eth_shorted_amount / 2))
    assert loan.borrowed.last_usd_price == 1400
    assert loan.borrowed.get_usd_value() == pytest.approx(466.66666666666663)
    assert loan.get_loan_to_value() == pytest.approx(0.3043478260869565)
    assert loan.get_leverage() == pytest.approx(0.4375000000000002)

    # Check that we track the portfolio value correctly
    # after realising profit
    assert portfolio.get_cash() == 9000  # No changes in cash, because we paid back from the collateral
    assert portfolio.get_net_asset_value() == pytest.approx(10066.66666)  # Should be same with our without reducing position as we have no fees, see test above
    assert portfolio.get_loan_net_asset_value() == pytest.approx(1066.6666666666665)
    assert portfolio.get_total_equity() == pytest.approx(10066.666)  # Any profits from closed short positions are not moved to equity, unless told so


def test_short_unrealised_profit_partially_closed_release_collateral(
        state: State,
        weth_short_identifier: TradingPairIdentifier,
        usdc: AssetIdentifier,
):
    """Opening a short position and get some unrealised profit.

    - ETH price goes 1500 -> 1400 so we get USD 56 unrealised PnL

    - Close 50% of this position at this price, we get USD 27 realised profit

    - Release collateral in the same proportions as and get USD 27 realised profit
      back to the cash reserves
    """

    trader = UnitTestTrader(state)

    portfolio = state.portfolio

    # How many ETH (vWETH) we expect when we go in
    # with our max leverage available
    # based on the collateral ratio
    expected_eth_shorted_amount = 1000 / 1500

    # Take 1000 USDC reserves and open a ETH short using it.
    # We should get 800 USDC worth of ETH for this.
    # 800 USDC worth of ETH is
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
    loan = short_position.loan
    assert loan.borrowed.quantity == pytest.approx(Decimal(expected_eth_shorted_amount))
    assert loan.collateral.quantity == pytest.approx(2000)
    assert loan.get_leverage() == 1.0

    assert portfolio.get_cash() == 9000
    assert portfolio.get_net_asset_value() == pytest.approx(10000)

    # ETH price 1500 -> 1400
    short_position.revalue_base_asset(
        datetime.datetime.utcnow(),
        1400.0,
    )

    assert portfolio.get_net_asset_value() == pytest.approx(10066.66666)

    # Make a trade that
    #
    # - Reduces position size 50%
    #
    # - Reduces collateral in the same ratio,
    #   so that 2:1 short leverage is maintained
    #
    # - Takes 100% accured profits back to cash
    #   reserves
    #

    # Calculate how much extra collateral we have
    # because of profit of shorts due to falling ETH price
    released_eth = expected_eth_shorted_amount / 2

    start_ltv = 0.5
    target_collateral = loan.calculate_collateral_for_target_ltv(start_ltv, expected_eth_shorted_amount / 2)
    expected_collateral_release = Decimal(expected_eth_shorted_amount/2 * 1400)

    # How much collateral we need to maintain 2:1 short ratio
    # using the new ETH prcie
    collateral_left_needed = Decimal(expected_eth_shorted_amount / 2 * 1400) * 2
    assert collateral_left_needed == pytest.approx(Decimal(933.3333333333333))

    collateral_adjustment = collateral_left_needed + expected_collateral_release - loan.collateral.quantity

    _, trade_2, _ = state.create_trade(
        strategy_cycle_at=datetime.datetime.utcnow(),
        pair=weth_short_identifier,
        quantity=Decimal(expected_eth_shorted_amount / 2),  # Short position quantity is counted as negative. When we close the quantity goes towards zero from neagtive.
        reserve=None,  # Reserve will be calculated generated from the released collateral
        assumed_price=float(1400),  # USDC/ETH price we are going to sell
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
        reserve_currency_price=1.0,
        planned_collateral_allocation=collateral_adjustment,
    )

    assert trade_2.planned_collateral_allocation == collateral_adjustment

    # Trade 2 will be executed,
    # planned loan update moves sto executed
    # we get rid of half of the position
    trader.set_perfectly_executed(trade_2)

    assert trade_2.executed_collateral_allocation == trade_2.planned_collateral_allocation

    # Check we are in our target collateral level
    loan = short_position.loan
    assert loan.borrowed.get_usd_value() == pytest.approx(466.6666)
    assert loan.collateral.quantity == pytest.approx(Decimal(933.3333333333333))
    assert loan.get_leverage() == 1.0

    assert portfolio.get_cash() == pytest.approx(9600)  # We have now cashed out our USD 53 profit unlike in the previous test
    assert portfolio.get_net_asset_value() == pytest.approx(10066.6666666)  # Should be same with our without reducing position as we have no fees, see test above
    assert portfolio.get_loan_net_asset_value() == pytest.approx(466.6666)


def test_short_close_fully_profitable(
        state: State,
        weth_short_identifier: TradingPairIdentifier,
        usdc: AssetIdentifier,
):
    """Opening a short position and close it.

    - Start with 1000 USDC collateral

    - Take 1000 USDC worth of ETH loan

    - Sell ETH at 1500 USDC/ETH

    - Now you end up with 2000 USDC collateral, 0.66 ETH debt

    - ETH price goes 1500 -> 1400 so we get USD 53.333 unrealised PnL

    - Close the position fully

    - Close position example TX on 1delta
      https://polygonscan.com/tx/0x44dbfd62b0730f83f89474eb7ca45b797a414276ed38fee77c5df3e7c56ae399
    """

    trader = UnitTestTrader(state)

    portfolio = state.portfolio

    # We deposit 1000 USDC as reserves,
    # use it to take a loan WETH at price 1500 USD/ETH
    # for 1000 USDC,
    # This gives us short leverage 1x,
    # long leverage 2x.
    expected_eth_shorted_amount = 1000 / 1500

    # Take 1000 USDC reserves and open a ETH short using it.
    # We should get 800 USDC worth of ETH for this.
    # 800 USDC worth of ETH is
    short_position, trade, created = state.trade_short(
        strategy_cycle_at=datetime.datetime.utcnow(),
        pair=weth_short_identifier,
        borrowed_quantity=-Decimal(expected_eth_shorted_amount),
        collateral_quantity=Decimal(1000),
        borrowed_asset_price=float(1500),  # USDC/ETH price we are going to sell
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
        collateral_asset_price=1.0,
    )

    trader.set_perfectly_executed(trade)

    loan = short_position.loan

    # Collateral is our initial collateral 1000 USDC,
    # plus it is once looped
    assert loan.collateral.quantity == pytest.approx(Decimal(2000))

    # We start with 1000 USDC collateral
    # and then do short with full collateral amount
    # so we end up 2000 USDC collateral and half of its
    # worth of ETH
    start_ltv = 0.50
    assert loan.get_loan_to_value() == start_ltv

    # Health factor comes from LTV
    # using 85% liquidation threshold for USDC
    # 1 / 0.5 * 0.85
    start_health_factor = 1.7
    assert loan.get_health_factor() == start_health_factor
    assert loan.get_leverage() == 1  # 1000 USD worth of ETH, 2000 USDC collateral

    # Portfolio value should not change, because
    # we have not paid fees and any price has not changed yet
    assert loan.get_net_asset_value() == 1000
    assert portfolio.get_cash() == 9000
    assert portfolio.get_net_asset_value() == pytest.approx(10000)
    assert portfolio.get_all_loan_nav() == 1000

    # ETH price 1500 -> 1400
    short_position.revalue_base_asset(
        datetime.datetime.utcnow(),
        1400.0,
    )

    # Our short is USD 66 to profit after the price drop
    assert loan.get_net_asset_value() == pytest.approx(1066.66666)
    assert portfolio.get_net_asset_value() == pytest.approx(10066.6666)
    assert portfolio.get_all_loan_nav() == pytest.approx(1066.666)

    _, trade_2, _ = state.trade_short(
        closing=True,
        strategy_cycle_at=datetime.datetime.utcnow(),
        pair=weth_short_identifier,
        borrowed_asset_price=float(1400),
        planned_mid_price=float(1400),
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
        collateral_asset_price=1.0,
    )

    # Amount of USDC we can return from the collateral to the cash reserves
    assert trade_2.planned_collateral_allocation == pytest.approx(Decimal(-1066.666666666666718477074482))
    # Amount of USDC we need to repay the loan
    assert trade_2.planned_collateral_consumption== pytest.approx(Decimal(-933.333333))
    assert trade_2.planned_loan_update.collateral.quantity == Decimal(0)
    assert trade_2.planned_loan_update.borrowed.quantity == Decimal(0)

    # Trade 2 will be excuted,
    # planned loan update moves sto executed
    # we get rid of half of the position
    trader.set_perfectly_executed(trade_2)

    assert trade_2.executed_price == 1400
    assert trade_2.executed_collateral_allocation == trade_2.planned_collateral_allocation
    assert trade_2.executed_collateral_consumption == trade_2.executed_collateral_consumption

    # Check that loan has now been repaid
    loan = short_position.loan
    assert loan.collateral.quantity == pytest.approx(0)  # TODO: Epsilon issues
    assert loan.borrowed.quantity == pytest.approx(0)  # TODO: Epsilon issues

    assert short_position.is_closed()
    assert short_position.get_value() == 0
    assert short_position.get_quantity() == 0
    assert short_position.get_unrealised_profit_usd() == 0
    assert short_position.get_realised_profit_usd() == pytest.approx(66.666666)

    assert portfolio.get_all_loan_nav() == 0
    assert portfolio.get_cash() == pytest.approx(10066.666666)  # We have now cashed out our USD 53 profit unlike in the previous test
    assert portfolio.get_net_asset_value() == pytest.approx(10066.666666)  # Should be same with our without reducing position as we have no fees, see test above
    assert portfolio.get_total_equity() == pytest.approx(10066.666666)


def test_short_close_fully_loss(
        state: State,
        weth_short_identifier: TradingPairIdentifier,
        usdc: AssetIdentifier,
):
    """Opening a short position and close it.

    - Start with 1000 USDC collateral

    - Take 1000 USDC worth of ETH loan

    - Sell ETH at 1500 USDC/ETH

    - Now you end up with 2000 USDC collateral, 0.66 ETH debt

    - ETH price goes up 1500 -> 1600 so we unrealised PnL

    - Close the position fully
    """

    trader = UnitTestTrader(state)

    portfolio = state.portfolio

    # We deposit 1000 USDC as reserves,
    # use it to take a loan WETH at price 1500 USD/ETH
    # for 1000 USDC,
    # This gives us short leverage 1x,
    # long leverage 2x.
    expected_eth_shorted_amount = 1000 / 1500

    # Take 1000 USDC reserves and open a ETH short using it.
    # We should get 800 USDC worth of ETH for this.
    # 800 USDC worth of ETH is
    short_position, trade, created = state.trade_short(
        strategy_cycle_at=datetime.datetime.utcnow(),
        pair=weth_short_identifier,
        borrowed_quantity=-Decimal(expected_eth_shorted_amount),
        collateral_quantity=Decimal(1000),
        borrowed_asset_price=float(1500),  # USDC/ETH price we are going to sell
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
        collateral_asset_price=1.0,
    )

    trader.set_perfectly_executed(trade)

    loan = short_position.loan

    # Collateral is our initial collateral 1000 USDC,
    # plus it is once looped
    assert loan.collateral.quantity == pytest.approx(Decimal(2000))

    # Portfolio value should not change, because
    # we have not paid fees and any price has not changed yet
    assert loan.get_net_asset_value() == 1000
    assert portfolio.get_cash() == 9000
    assert portfolio.get_net_asset_value() == pytest.approx(10000)
    assert portfolio.get_all_loan_nav() == 1000

    # ETH price 1500 -> 1600
    short_position.revalue_base_asset(
        datetime.datetime.utcnow(),
        1600.0,
    )

    assert loan.get_net_asset_value() == pytest.approx(933.3333333333335)
    assert portfolio.get_net_asset_value() == pytest.approx(9933.333333333334)
    assert portfolio.get_all_loan_nav() == pytest.approx(933.3333333333335)

    _, trade_2, _ = state.trade_short(
        closing=True,
        strategy_cycle_at=datetime.datetime.utcnow(),
        pair=weth_short_identifier,
        borrowed_asset_price=float(1600),
        planned_mid_price=float(1600),
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
        collateral_asset_price=1.0,
    )

    # Amount of USDC we can return from the collateral to the cash reserves
    assert trade_2.planned_collateral_allocation == pytest.approx(Decimal(-933.333333333333392545227980))
    # Amount of USDC we need to repay the loan
    assert trade_2.planned_collateral_consumption== pytest.approx(Decimal(-1066.666666666666607454772020))
    assert trade_2.planned_loan_update.collateral.quantity == Decimal(0)
    assert trade_2.planned_loan_update.borrowed.quantity == Decimal(0)

    # Trade 2 will be excuted,
    # planned loan update moves sto executed
    # we get rid of half of the position
    trader.set_perfectly_executed(trade_2)

    assert trade_2.executed_price == 1600
    assert trade_2.executed_collateral_allocation == trade_2.planned_collateral_allocation
    assert trade_2.executed_collateral_consumption == trade_2.executed_collateral_consumption

    # Check that loan has now been repaid
    loan = short_position.loan
    assert loan.collateral.quantity == pytest.approx(0)  # TODO: Epsilon issues
    assert loan.borrowed.quantity == pytest.approx(0)  # TODO: Epsilon issues

    assert short_position.is_closed()
    assert short_position.get_value() == 0
    assert short_position.get_quantity() == 0
    assert short_position.get_unrealised_profit_usd() == 0
    assert short_position.get_realised_profit_usd() == pytest.approx(-66.666666)

    assert portfolio.get_all_loan_nav() == 0
    assert portfolio.get_cash() == pytest.approx(9933.333333333334)
    assert portfolio.get_net_asset_value() == pytest.approx(9933.333333333334)
    assert portfolio.get_total_equity() == pytest.approx(9933.333333333334)


def test_short_increase_leverage_and_close(
        state: State,
        weth_short_identifier: TradingPairIdentifier,
        usdc: AssetIdentifier,
):
    """Opening a short position and close it.

    - Start with 1000 USDC collateral

    - Take 1000 USDC worth of ETH loan

    - Sell ETH at 1500 USDC/ETH

    - Now you end up with 2000 USDC collateral, 0.66 ETH debt

    - ETH price goes up 1500 -> 1600 so we have unrealised PnL

    - Increase leverage to 3:1 short

    - ETH price goes up 1600 -> 1700, take more losses

    - Close the position fully
    """

    trader = UnitTestTrader(state)

    portfolio = state.portfolio

    # We deposit 1000 USDC as reserves,
    # use it to take a loan WETH at price 1500 USD/ETH
    # for 1000 USDC,
    # This gives us short leverage 1x,
    # long leverage 2x.
    expected_eth_shorted_amount = 1000 / 1500

    # Take 1000 USDC reserves and open a ETH short using it.
    # We should get 800 USDC worth of ETH for this.
    # 800 USDC worth of ETH is
    short_position, trade, created = state.trade_short(
        strategy_cycle_at=datetime.datetime.utcnow(),
        pair=weth_short_identifier,
        borrowed_quantity=-Decimal(expected_eth_shorted_amount),
        collateral_quantity=Decimal(1000),
        borrowed_asset_price=float(1500),  # USDC/ETH price we are going to sell
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
        collateral_asset_price=1.0,
    )

    trader.set_perfectly_executed(trade)

    loan = short_position.loan

    assert loan.collateral.quantity == pytest.approx(Decimal(2000))
    assert loan.get_net_asset_value() == 1000
    assert loan.get_leverage() == 1
    assert portfolio.get_cash() == 9000
    assert portfolio.get_net_asset_value() == pytest.approx(10000)
    assert portfolio.get_all_loan_nav() == 1000

    # ETH price 1500 -> 1600
    short_position.revalue_base_asset(
        datetime.datetime.utcnow(),
        1600.0,
    )

    assert loan.get_leverage() == pytest.approx(1.1428571428571423)
    assert portfolio.get_net_asset_value() == pytest.approx(9933.333333333334)

    # Trade to increase leverage
    #
    # Bring leverage to 3:1
    #
    target_collateral = loan.calculate_collateral_for_target_leverage(
        3.0,
        loan.borrowed.quantity,
    )
    collateral_adjustment = target_collateral - loan.collateral.quantity

    _, trade_2, _ = state.trade_short(
        strategy_cycle_at=datetime.datetime.utcnow(),
        pair=weth_short_identifier,
        borrowed_quantity=Decimal(0),
        collateral_quantity=Decimal(0),
        borrowed_asset_price=loan.borrowed.last_usd_price,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
        collateral_asset_price=1.0,
        planned_collateral_allocation=collateral_adjustment,
    )

    # Amount of USDC we can return from the collateral to the cash reserves
    assert trade_2.planned_collateral_allocation == pytest.approx(Decimal(-577.7777))
    assert trade_2.planned_collateral_consumption== pytest.approx(Decimal(0))
    assert trade_2.planned_loan_update.collateral.quantity == pytest.approx(Decimal(1422.2222))
    assert trade_2.planned_loan_update.borrowed.quantity == pytest.approx(Decimal(0.6666666666))

    trader.set_perfectly_executed(trade_2)

    assert trade_2.executed_collateral_allocation == trade_2.planned_collateral_allocation
    assert trade_2.executed_collateral_consumption == trade_2.executed_collateral_consumption

    # Loan leverage has changed, but our portfolio net asset value stays same
    loan = short_position.loan
    assert loan.get_leverage() == pytest.approx(3.0)
    assert loan.get_health_factor() == pytest.approx(1.1333333)
    assert portfolio.get_net_asset_value() == pytest.approx(9933.333333333334)

    # Even more losses
    # ETH price 1600 -> 1700
    short_position.revalue_base_asset(
        datetime.datetime.utcnow(),
        1700.0,
    )

    loan = short_position.loan
    assert loan.get_leverage() == pytest.approx(3.9230769230769256)
    assert loan.get_health_factor() == pytest.approx(1.0666666666666667)  # Risky
    assert portfolio.get_net_asset_value() == pytest.approx(9866.66666)

    # Close all
    _, trade_3, _ = state.trade_short(
        closing=True,
        strategy_cycle_at=datetime.datetime.utcnow(),
        pair=weth_short_identifier,
        borrowed_asset_price=loan.borrowed.last_usd_price,
        planned_mid_price=loan.borrowed.last_usd_price,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
        collateral_asset_price=1.0,
    )
    trader.set_perfectly_executed(trade_3)

    assert short_position.is_closed()
    assert portfolio.get_net_asset_value() == pytest.approx(9866.66666)


def test_short_unrealised_profit_leveraged(
        state: State,
        weth_short_identifier: TradingPairIdentifier,
        usdc: AssetIdentifier,
):
    """Opening a short position and get some unrealised profit.

    - Use 1000 USDC as collateral

    - Open a short with 3x leveraggem, so we get 3000 USDC worth of ETH

    - ETH price goes 1500 -> 1400 so we get unrealised PnL
    """

    trader = UnitTestTrader(state)
    portfolio = state.portfolio

    start_collateral = Decimal(1000)
    leverage = 3.0
    eth_price = 1500.0
    estimation = LeverageEstimate.open_short(
        starting_reserve=float(start_collateral),
        leverage=leverage,
        borrowed_asset_price=eth_price,
        shorting_pair=weth_short_identifier,
    )
    eth_short_value = estimation.borrowed_value
    collateral_value = estimation.total_collateral_quantity

    assert estimation.liquidation_price == pytest.approx(Decimal(1699.9999))

    # Check our loan parameters
    assert eth_short_value == pytest.approx(start_collateral * Decimal(leverage))
    assert collateral_value == pytest.approx(4000)

    eth_quantity = Decimal(eth_short_value / eth_price)
    assert eth_quantity == pytest.approx(Decimal(2))

    short_position, trade, _ = state.trade_short(
        strategy_cycle_at=datetime.datetime.utcnow(),
        pair=weth_short_identifier,
        borrowed_quantity=-eth_quantity,
        collateral_quantity=start_collateral,
        borrowed_asset_price=eth_price,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
        collateral_asset_price=1.0,
        # We loop our collateral this much more
        # to achieve our target collateral level
        planned_collateral_consumption=estimation.additional_collateral_quantity,
        # planned_collateral_allocation=estimation.starting_reserve,
    )

    # We move 1000 USDC from reserves to loan deposits
    assert trade.planned_reserve == start_collateral
    # TODO: should this be the same as start_collateral?
    assert trade.planned_collateral_allocation == 0
    assert trade.planned_collateral_consumption == pytest.approx(Decimal(3000))
    assert trade.planned_quantity == pytest.approx(-eth_quantity)  # We loop the collateral this many USD to get our required level

    trader.set_perfectly_executed(trade)

    loan = short_position.loan
    assert loan.collateral.get_usd_value() == pytest.approx(4000)
    assert loan.get_net_asset_value() == pytest.approx(1000)
    assert loan.get_leverage() == pytest.approx(leverage)
    # assert portfolio.get_cash() == 9000
    assert portfolio.get_net_asset_value() == 10_000

    # ETH price 1500 -> 1400, make profit
    short_position.revalue_base_asset(
        datetime.datetime.utcnow(),
        1400.0,
    )

    loan = short_position.loan
    assert loan.get_net_asset_value() == pytest.approx(1200)
    assert short_position.get_unrealised_profit_usd() == pytest.approx(200)

    _, trade_2, _ = state.trade_short(
        closing=True,
        strategy_cycle_at=datetime.datetime.utcnow(),
        pair=weth_short_identifier,
        borrowed_asset_price=loan.borrowed.last_usd_price,
        planned_mid_price=loan.borrowed.last_usd_price,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
        collateral_asset_price=1.0,
    )
    trader.set_perfectly_executed(trade_2)

    # check closing trade numbers
    assert trade_2.planned_reserve == 0
    assert trade_2.planned_collateral_allocation == pytest.approx(-Decimal(1200))
    assert trade_2.planned_collateral_consumption == pytest.approx(-Decimal(2800))
    assert trade_2.planned_quantity == pytest.approx(eth_quantity)

    assert short_position.is_closed()
    assert short_position.get_realised_profit_usd() == pytest.approx(200)
    assert short_position.get_unrealised_profit_usd() == 0
    assert portfolio.get_cash() == pytest.approx(10200)


def test_short_unrealised_profit_no_leverage(
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
    short_position, trade, created = state.trade_short(
        strategy_cycle_at=datetime.datetime.utcnow(),
        pair=weth_short_identifier,
        borrowed_quantity=-Decimal(expected_eth_shorted_amount),
        collateral_quantity=Decimal(1000),
        borrowed_asset_price=float(1500),  # USDC/ETH price we are going to sell
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
        collateral_asset_price=1.0,
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
    assert short_position.get_realised_profit_usd() is None

    loan = short_position.loan
    assert loan.borrowed.quantity == pytest.approx(Decimal(expected_eth_shorted_amount))
    assert loan.borrowed.asset.token_symbol == "variableDebtPolWETH"
    assert loan.borrowed.asset.underlying.token_symbol == "WETH"
    assert loan.borrowed.last_usd_price == 1400
    assert loan.get_loan_to_value() == pytest.approx(0.4148148148148148)

    # Check that we track the equity value correctly
    assert state.portfolio.get_loan_net_asset_value() == pytest.approx(1053.3333333333335)
    assert state.portfolio.get_cash() == 9000
    assert state.portfolio.get_net_asset_value() == pytest.approx(10053.333333333334)


def test_short_unrealised_profit_leverage_all(
        state: State,
        weth_short_identifier: TradingPairIdentifier,
        usdc: AssetIdentifier,
):
    """Opening a short position and get some unrealised profit.

    - Use our all 10,000 USDC as collateral

    - Open a short with 3x leveragge

    - ETH price goes 1500 -> 1400 so we get unrealised PnL
    """

    trader = UnitTestTrader(state)
    portfolio = state.portfolio

    start_collateral = Decimal(10000)
    leverage = 3.0
    eth_price = 1500.0
    estimation = LeverageEstimate.open_short(
        starting_reserve=float(start_collateral),
        leverage=leverage,
        borrowed_asset_price=eth_price,
        shorting_pair=weth_short_identifier,
    )
    eth_short_value = estimation.borrowed_value
    collateral_value = estimation.total_collateral_quantity

    # Check our loan parameters
    assert eth_short_value == pytest.approx(start_collateral * Decimal(leverage))
    assert collateral_value == pytest.approx(start_collateral * Decimal(leverage + 1))

    eth_quantity = Decimal(eth_short_value / eth_price)

    short_position, trade, _ = state.trade_short(
        strategy_cycle_at=datetime.datetime.utcnow(),
        pair=weth_short_identifier,
        borrowed_quantity=-eth_quantity,
        collateral_quantity=start_collateral,
        borrowed_asset_price=eth_price,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
        collateral_asset_price=1.0,
        # We loop our collateral this much more
        # to achieve our target collateral level
        planned_collateral_consumption=Decimal(collateral_value) - start_collateral,
    )

    # We move 1000 USDC from reserves to loan deposits
    assert trade.planned_reserve == start_collateral
    assert trade.planned_collateral_consumption == pytest.approx(Decimal(30000))
    assert trade.planned_quantity == pytest.approx(-eth_quantity)  # We loop the collateral this many USD to get our required level

    trader.set_perfectly_executed(trade)

    loan = short_position.loan
    assert loan.collateral.get_usd_value() == pytest.approx(40000)
    assert loan.get_net_asset_value() == pytest.approx(10000)
    assert loan.get_leverage() == pytest.approx(leverage)
    assert portfolio.get_cash() == 0
    assert portfolio.get_net_asset_value() == 10_000

    # ETH price 1500 -> 1400, make profit
    short_position.revalue_base_asset(
        datetime.datetime.utcnow(),
        1400.0,
    )

    loan = short_position.loan
    assert loan.get_net_asset_value() == pytest.approx(12000)
    assert loan.get_collateral_interest() == pytest.approx(0)
    assert loan.collateral_interest.last_accrued_interest == 0
    assert short_position.get_unrealised_profit_usd() == pytest.approx(2000)

    _, trade_2, _ = state.trade_short(
        closing=True,
        strategy_cycle_at=datetime.datetime.utcnow(),
        pair=weth_short_identifier,
        borrowed_asset_price=loan.borrowed.last_usd_price,
        planned_mid_price=loan.borrowed.last_usd_price,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
        collateral_asset_price=1.0,
    )
    trader.set_perfectly_executed(trade_2)
    assert short_position.is_closed()
    assert short_position.get_realised_profit_usd() == pytest.approx(2000)
    assert short_position.get_unrealised_profit_usd() == 0
    assert portfolio.get_cash() == pytest.approx(12000)

    # How much USDC gained we got from the collateral
    assert trade_2.claimed_interest == pytest.approx(0)


def test_short_unrealised_interest_and_profit(
        state: State,
        weth_short_identifier: TradingPairIdentifier,
        usdc: AssetIdentifier,
):
    """Opening a short position and get some unrealised profit.

    - ETH price goes 1500 -> 1400 so we get unrealised PnL
    - We have 10% borrow cost on the ETH short position
    - We have 2% interest income on the USDC collateral
    - Wait half a year
    - See ``test_short_unrealised_profit`` for a comparison calculations
      with interest payments ignored
    """

    trader = UnitTestTrader(state)

    # Aave allows us to borrow 80% ETH against our USDC collateral
    start_ltv = 0.8

    # How many ETH (vWETH) we expect when we go in
    # with our max leverage available
    # based on the collateral ratio
    expected_eth_shorted_amount = 1000 * start_ltv / 1500

    start_at = datetime.datetime(2020, 1, 1)

    # Take 1000 USDC reserves and open a ETH short using it.
    # We should get 800 USDC worth of ETH for this.
    short_position, trade, created = state.trade_short(
        strategy_cycle_at=start_at,
        pair=weth_short_identifier,
        borrowed_quantity=-Decimal(expected_eth_shorted_amount),
        collateral_quantity=Decimal(1000),
        borrowed_asset_price=float(1500),  # USDC/ETH price we are going to sell
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
        collateral_asset_price=1.0,
    )

    trader.set_perfectly_executed(trade)
    assert state.portfolio.get_total_equity() == 10000

    # Move forward half a year
    now_at = datetime.datetime(2020, 6, 1)

    # ETH price 1500 -> 1400
    short_position.revalue_base_asset(
        datetime.datetime.utcnow(),
        1400.0,
    )

    atoken_interest = 1.02  # Receive 2% on USD collateral
    vtoken_interest = 1.10  # Pay 10% on ETH loan

    # Calculate simulated interest gains
    new_atoken = estimate_interest(
        start_at,
        now_at,
        short_position.loan.collateral.quantity,
        atoken_interest,
    )

    new_vtoken = estimate_interest(
        start_at,
        now_at,
        short_position.loan.borrowed.quantity,
        vtoken_interest,
    )

    assert new_atoken == pytest.approx(Decimal('1814.905206376357316063955614'))
    assert new_vtoken == pytest.approx(Decimal('0.5549274775700127945499953131'))

    # Tell strategy state about interest gains
    # Note that this BalanceUpdate event
    # is not stored with the state
    vevt, aevt = update_leveraged_position_interest(
        state,
        short_position,
        new_vtoken,
        new_atoken,
        now_at,
        vtoken_price=1400.0,
        atoken_price=1.0,
    )

    # We gain around 15 USDC in half a year
    assert aevt.quantity == pytest.approx(Decimal('14.905206376357327166185860'))

    # We need to pay ~ 0.02 ETH half a year,
    # worth 30 USD at 1400 ETH/USD
    assert vevt.quantity == pytest.approx(Decimal('0.0215941442366794686181488106'))

    assert aevt.get_update_period() == datetime.timedelta(days=152)
    assert vevt.get_update_period() == datetime.timedelta(days=152)
    assert aevt.get_effective_yearly_yield() == pytest.approx(0.019884504120505936, rel=0.01)
    assert vevt.get_effective_yearly_yield() == pytest.approx(0.097, rel=0.01)

    # New ETH loan worth of 746.6666666666666 USD
    assert short_position.get_current_price() == 1400
    assert short_position.get_average_price() == 1500

    assert short_position.loan.get_collateral_interest() == pytest.approx(14.905206376357327)
    assert short_position.loan.get_borrow_interest() == pytest.approx(30.23180193135126)
    assert short_position.get_accrued_interest() == pytest.approx(-15.326595554993933)
    assert short_position.get_value() == pytest.approx(1038.0067377783394)

    #  assert short_position.get_unrealised_profit_usd() == pytest.approx(800 - 746.6666666666666)
    assert short_position.get_realised_profit_usd() is None

    loan = short_position.loan
    assert loan.borrowed.quantity == pytest.approx(Decimal(expected_eth_shorted_amount))
    assert loan.borrowed.asset.token_symbol == "variableDebtPolWETH"
    assert loan.borrowed.asset.underlying.token_symbol == "WETH"
    assert loan.borrowed.last_usd_price == 1400
    assert loan.get_loan_to_value() == pytest.approx(0.4148148148148148)

    # Check that we track the equity value correctly
    assert state.portfolio.get_loan_net_asset_value() == pytest.approx(1038.0067377783394)
    assert state.portfolio.get_cash() == 9000
    assert state.portfolio.get_net_asset_value() == pytest.approx(10038.0067377783394)


def test_short_unrealised_interest_and_losses(
        state: State,
        weth_short_identifier: TradingPairIdentifier,
        usdc: AssetIdentifier,
):
    """Opening a short position and get some unrealised profit.

    - ETH price goes 1500 -> 1600 so we get unrealised PnL
    - We have 50% borrow cost on the ETH short position
    - We have 2% interest income on the USDC collateral
    - Wait 3 months
    """

    trader = UnitTestTrader(state)

    # Aave allows us to borrow 80% ETH against our USDC collateral
    start_ltv = 0.8

    # How many ETH (vWETH) we expect when we go in
    # with our max leverage available
    # based on the collateral ratio
    expected_eth_shorted_amount = 1000 * start_ltv / 1500

    start_at = datetime.datetime(2020, 1, 1)

    # Take 1000 USDC reserves and open a ETH short using it.
    # We should get 800 USDC worth of ETH for this.
    short_position, trade, created = state.trade_short(
        strategy_cycle_at=start_at,
        pair=weth_short_identifier,
        borrowed_quantity=-Decimal(expected_eth_shorted_amount),
        collateral_quantity=Decimal(1000),
        borrowed_asset_price=float(1500),  # USDC/ETH price we are going to sell
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
        collateral_asset_price=1.0,
    )

    trader.set_perfectly_executed(trade)
    assert state.portfolio.get_total_equity() == 10000

    # Move forward a financial year
    now_at = datetime.datetime(2020, 1, 1) + datetime.timedelta(days=365)

    # ETH price 1500 -> 1600,
    # cause short to go negative
    short_position.revalue_base_asset(
        datetime.datetime.utcnow(),
        1600.0,
    )

    atoken_interest = 1.02  # Receive 2% on USD collateral
    vtoken_interest = 1.50  # Pay 50% on ETH loan

    # Calculate simulated interest gains
    new_atoken = estimate_interest(
        start_at,
        now_at,
        short_position.loan.collateral.quantity,
        atoken_interest,
    )

    new_vtoken = estimate_interest(
        start_at,
        now_at,
        short_position.loan.borrowed.quantity,
        vtoken_interest,
    )

    old_vtoken = short_position.loan.borrowed.quantity
    # assert new_atoken == Decimal('1809.032784100456570042034639')  # We have gained 15 USDC on our dollar long
    # assert new_vtoken == Decimal('0.5463385917812319708720691214')
    assert new_vtoken / old_vtoken == pytest.approx(Decimal("1.500000000000000000000000000"))

    # Tell strategy state about interest gains
    # Note that this BalanceUpdate event
    # is not stored with the state
    vevt, aevt = update_leveraged_position_interest(
        state,
        short_position,
        new_vtoken,
        new_atoken,
        now_at,
        vtoken_price=1600.0,
        atoken_price=1.0,
        max_interest_gain=0.6,
    )

    # We gain around 15 USDC in half a year
    assert aevt.quantity == pytest.approx(Decimal('36.000000000000031752378504'))
    assert aevt.get_update_period() == datetime.timedelta(days=365)
    assert aevt.get_effective_yearly_yield() == pytest.approx(0.020000000000000018)
    assert vevt.get_effective_yearly_yield() == pytest.approx(0.50)

    assert short_position.get_current_price() == 1600

    assert short_position.loan.get_borrow_interest() == pytest.approx(426.6666666666667)

    # We go red
    assert state.portfolio.get_loan_net_asset_value() == pytest.approx(556)
    assert state.portfolio.get_net_asset_value() == pytest.approx(9556)


def test_short_realised_interest_and_profit(
        state: State,
        weth_short_identifier: TradingPairIdentifier,
        usdc: AssetIdentifier,
):
    """Opening a short position and get some unrealised profit.

    - ETH price goes 1500 -> 1400 so we get unrealised PnL
    - We have 10% borrow cost on the ETH short position
    - We have 2% interest income on the USDC collateral
    - Wait half a year
    - Close the position
    """

    trader = UnitTestTrader(state)
    start_ltv = 0.8
    expected_eth_shorted_amount = 1000 * start_ltv / 1500

    start_at = datetime.datetime(2020, 1, 1)
    short_position, trade, created = state.trade_short(
        strategy_cycle_at=start_at,
        pair=weth_short_identifier,
        borrowed_quantity=-Decimal(expected_eth_shorted_amount),
        collateral_quantity=Decimal(1000),
        borrowed_asset_price=float(1500),  # USDC/ETH price we are going to sell
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
        collateral_asset_price=1.0,
    )

    trader.set_perfectly_executed(trade)
    assert state.portfolio.get_total_equity() == 10000

    # Move forward half a year
    now_at = datetime.datetime(2020, 6, 1)

    # ETH price 1500 -> 1400
    short_position.revalue_base_asset(
        datetime.datetime.utcnow(),
        1400.0,
    )

    atoken_interest = 1.02  # Receive 2% on USD collateral
    vtoken_interest = 1.10  # Pay 10% on ETH loan

    # Calculate simulated interest gains
    new_atoken = estimate_interest(
        start_at,
        now_at,
        short_position.loan.collateral.quantity,
        atoken_interest,
    )

    new_vtoken = estimate_interest(
        start_at,
        now_at,
        short_position.loan.borrowed.quantity,
        vtoken_interest,
    )

    assert new_atoken == pytest.approx(Decimal('1814.905206376357316063955614'))  # We have gained 15 USDC on our dollar long
    assert new_vtoken == pytest.approx(Decimal('0.5549274775700127945499953131'))

    # Tell strategy state about interest gains
    # Note that this BalanceUpdate event
    # is not stored with the state
    update_leveraged_position_interest(
        state,
        short_position,
        new_vtoken,
        new_atoken,
        now_at,
        vtoken_price=1400.0,
        atoken_price=1.0,
    )

    # How much total increment vToken has seen
    assert short_position.get_base_token_balance_update_quantity() == pytest.approx(Decimal('0.0215941442366794686181488106'))

    # All vToken left
    assert short_position.get_quantity() == pytest.approx(Decimal('-0.5549274775700127945499953131'))

    # Calculate unrealised and realised PnL
    loan = short_position.loan
    assert loan.collateral_interest.last_token_amount == pytest.approx(Decimal(1814.905206376357316063955614))
    assert loan.collateral_interest.last_event_at == now_at
    assert loan.collateral_interest.last_accrued_interest == pytest.approx(Decimal(14.905206376357327166185860))
    assert loan.get_collateral_quantity() == pytest.approx(Decimal(1814.905206376357316063955614))
    assert loan.get_collateral_value() == pytest.approx(1814.905206376357316063955614)
    assert loan.get_borrow_value() == pytest.approx(776.8984685980179)
    assert loan.get_collateral_interest() == pytest.approx(14.905206376357327166185860)
    assert loan.get_borrow_interest() == pytest.approx(30.23180193135126)
    assert loan.get_net_interest() == pytest.approx(-15.326595554993933)
    assert loan.get_net_asset_value(include_interest=False) == pytest.approx(1053.3333333333335)
    assert loan.get_net_asset_value(include_interest=True) == pytest.approx(1038.0067377783394)
    assert loan.borrowed.quantity == Decimal('0.5333333333333333259318465025')
    assert loan.get_borrowed_principal_and_interest_quantity() == Decimal('0.5549274775700127945499953131')

    assert short_position.get_accrued_interest() == pytest.approx(-15.326595554993933)
    assert short_position.get_quantity() == Decimal('-0.5549274775700127945499953131')
    assert short_position.get_net_quantity() == Decimal('-0.5549274775700127945499953131')

    unrealised_equity = (short_position.get_current_price() - short_position.get_average_price()) * float(short_position.get_net_quantity())
    assert unrealised_equity == pytest.approx(55.49274775700128)

    assert short_position.get_accrued_interest_with_repayments() == pytest.approx(-15.326595554993933)
    assert short_position.get_unrealised_profit_usd(include_interest=False) == pytest.approx(55.49274775700128)
    assert short_position.get_unrealised_profit_usd(include_interest=True) == pytest.approx(40.16615220200735)

    assert short_position.get_realised_profit_usd() is None
    assert short_position.get_value() == pytest.approx(1038.0067377783394)
    assert short_position.loan.get_net_asset_value() == pytest.approx(1038.0067377783394)

    # Prepare a closing trade and check it matches full vToken amount
    assert short_position.loan.borrowed.quantity == pytest.approx(Decimal('0.5333333333333333259318465025'))
    assert short_position.loan.borrowed_interest.last_accrued_interest == pytest.approx(Decimal('0.0215941442366794686181488106'))
    principal_and_interest = short_position.loan.get_borrowed_principal_and_interest_quantity()
    assert principal_and_interest == pytest.approx(Decimal('0.5549274775700127945499953131'))

    _, trade_2, _ = state.trade_short(
        closing=True,
        strategy_cycle_at=datetime.datetime.utcnow(),
        pair=weth_short_identifier,
        borrowed_asset_price=float(1400),
        planned_mid_price=float(1400),
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
        collateral_asset_price=1.0,
    )

    # Check that the position closes for vToken principal + interest amount
    assert short_position.get_quantity() == -principal_and_interest

    trader.set_perfectly_executed(trade_2)

    # Trade executes, value and quantity checks out
    assert trade_2.is_success()
    assert trade_2.executed_quantity == principal_and_interest
    assert trade_2.get_position_quantity() == principal_and_interest
    assert trade_2.get_value() == pytest.approx(776.898468598018)

    # How much USDC gained we got from the collateral
    assert trade_2.claimed_interest == pytest.approx(Decimal('14.905206376357327166185860'))

    # How much ETH this trade paid in interest cose
    assert trade_2.paid_interest == pytest.approx(Decimal('0.0215941442366794686181488106'))

    # The difference between opening and closing trades quantity
    # is the interest amount
    assert trade.executed_quantity + trade_2.executed_quantity == pytest.approx(Decimal('0.0215941442366794686181488106'))

    # This trade included repaid interest
    assert trade_2.get_repaid_interest() == pytest.approx(30.23180193135126)

    # Net asset value does not correctly work when the interest is repaid
    loan = short_position.loan
    assert loan.collateral_interest.get_remaining_interest() == 0
    assert loan.borrowed_interest.get_remaining_interest() == 0
    assert loan.collateral_interest.interest_payments == pytest.approx(Decimal('14.905206376357327166185860'))
    assert loan.borrowed_interest.interest_payments == pytest.approx(Decimal('0.0215941442366794686181488106'))
    assert loan.get_net_asset_value() == 0

    # Position is properly closed
    assert short_position.get_quantity() == 0
    assert short_position.is_closed()
    assert short_position.get_unrealised_profit_usd() == 0
    assert short_position.get_repaid_interest() == pytest.approx(30.23180193135126)
    assert short_position.get_claimed_interest() == pytest.approx(14.905206376357327)  # Profit we got from collateral interest
    assert short_position.get_realised_profit_usd(include_interest=False) == pytest.approx(55.49274775700128)
    assert short_position.get_realised_profit_usd(include_interest=True) == pytest.approx(40.16615220200735)

    portfolio = state.portfolio
    assert portfolio.get_cash() == pytest.approx(10038.00673777834)


def test_short_open_with_fee(
        state: State,
        weth_short_identifier_5bps: TradingPairIdentifier,
        usdc: AssetIdentifier,
):
    """Opening a short position and loss some money on trading fees..

    See :py:class:`LeverageEstimate` docstring for the explanation.
    """

    trader = UnitTestTrader(state)
    portfolio = state.portfolio

    start_collateral = Decimal(10000)
    leverage = 3.0
    eth_price = 1500.0
    estimate = LeverageEstimate.open_short(
        start_collateral,
        leverage,
        borrowed_asset_price=eth_price,
        fee=weth_short_identifier_5bps.get_pricing_pair().fee,
        shorting_pair=weth_short_identifier_5bps,
    )

    # Check our loan parameters
    eth_quantity = estimate.borrowed_quantity
    assert estimate.borrowed_value == pytest.approx(start_collateral * Decimal(leverage))
    assert estimate.additional_collateral_quantity == pytest.approx(Decimal(29984.9999))
    assert estimate.total_collateral_quantity == pytest.approx(Decimal(39984.9999))
    assert estimate.lp_fees == pytest.approx(15)

    short_position, trade, _ = state.trade_short(
        strategy_cycle_at=datetime.datetime.utcnow(),
        pair=weth_short_identifier_5bps,
        borrowed_quantity=-estimate.borrowed_quantity,
        collateral_quantity=start_collateral,
        borrowed_asset_price=eth_price,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
        collateral_asset_price=1.0,
        planned_collateral_consumption=estimate.additional_collateral_quantity,
        lp_fees_estimated=estimate.lp_fees,
    )

    # We move 1000 USDC from reserves to loan deposits
    assert trade.planned_reserve == start_collateral
    assert trade.planned_collateral_consumption == pytest.approx(Decimal(29984.9999))

    # When we open a short position the token amount for the borrowed token is
    # is shorted amount - fees
    assert trade.planned_quantity == pytest.approx(-eth_quantity)
    assert trade.lp_fees_estimated == 15

    trader.set_perfectly_executed(trade)

    total_shorted_usd = -float(trade.executed_quantity) * trade.executed_price
    assert total_shorted_usd == pytest.approx(30_000)
    assert trade.lp_fees_paid == 15  # 5 BPS of 20_000 of opened position

    loan = short_position.loan
    assert loan.borrowed.get_usd_value() == pytest.approx(30_000)
    assert loan.collateral.get_usd_value() == pytest.approx(39984.9999)
    assert loan.get_net_asset_value() == pytest.approx(9985.0)
    assert loan.get_leverage() == pytest.approx(3.0045067)
    assert portfolio.get_cash() == 0
    assert portfolio.get_net_asset_value() == 9985.0


def test_short_close_with_fee_no_price_movement(
        state: State,
        weth_short_identifier_5bps: TradingPairIdentifier,
        usdc: AssetIdentifier,
):
    """Close a short position and calculate money lost on trading fees.

    - Price does not move between open and close

    See :py:class:`LeverageEstimate` docstring for the explanation.
    """

    trader = UnitTestTrader(state)
    portfolio = state.portfolio

    open_estimate = LeverageEstimate.open_short(
        starting_reserve=Decimal(10_000),
        leverage=3.0,
        borrowed_asset_price=1500.0,
        fee=weth_short_identifier_5bps.get_pricing_pair().fee,
        shorting_pair=weth_short_identifier_5bps,
    )

    assert open_estimate.lp_fees == pytest.approx(15)

    short_position, trade, _ = state.trade_short(
        strategy_cycle_at=datetime.datetime.utcnow(),
        pair=weth_short_identifier_5bps,
        borrowed_quantity=-open_estimate.borrowed_quantity,
        collateral_quantity=open_estimate.starting_reserve,
        borrowed_asset_price=open_estimate.borrowed_asset_price,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
        collateral_asset_price=1.0,
        planned_collateral_consumption=open_estimate.additional_collateral_quantity,
        lp_fees_estimated=open_estimate.lp_fees,
    )

    trader.set_perfectly_executed(trade)

    assert portfolio.get_cash() == 0
    assert portfolio.get_net_asset_value() == 9985

    assert short_position.loan.collateral.quantity == pytest.approx(Decimal(39984.9999))
    assert short_position.loan.borrowed.quantity == pytest.approx(Decimal(20))

    #
    # Close the short position with trading fees
    #

    estimate = LeverageEstimate.close_short(
        start_collateral=short_position.loan.collateral.quantity,
        start_borrowed=short_position.loan.borrowed.quantity,
        close_size=short_position.loan.borrowed.quantity,
        fee=weth_short_identifier_5bps.get_pricing_pair().fee,
        borrowed_asset_price=1500.0,
    )

    assert estimate.leverage == 1.0  # Reduced USDC leverage to 1.0
    assert estimate.additional_collateral_quantity == pytest.approx(Decimal(-30015.00750375187))  # USDC needed to reduce from collateral to close position + fees
    assert estimate.borrowed_quantity == pytest.approx(Decimal(-20))  # How much ETH is bought to close the short
    assert estimate.total_collateral_quantity == pytest.approx(Decimal(9969.992496248124))  # Collateral left after closing the position
    assert estimate.total_borrowed_quantity == 0  # open vWETH debt left after close
    assert estimate.lp_fees == pytest.approx(15.007503751875939)

    # Now close the position
    _, trade_2, _ = state.trade_short(
        closing=True,
        strategy_cycle_at=datetime.datetime.utcnow(),
        pair=weth_short_identifier_5bps,
        borrowed_asset_price=open_estimate.borrowed_asset_price,
        planned_mid_price=open_estimate.borrowed_asset_price,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
        collateral_asset_price=1.0,
    )

    trader.set_perfectly_executed(trade_2)

    # After opening and closing short, we have lost money on fees
    assert portfolio.get_cash() == pytest.approx(9969.992496248124)
    assert portfolio.get_net_asset_value() == pytest.approx(9969.992496248124)


def test_short_close_profit_with_fee(
        state: State,
        weth_short_identifier_5bps: TradingPairIdentifier,
        usdc: AssetIdentifier,
):
    """Close a short position for profit.

    - See :py:func:`test_short_close_fully_profitable` for an example
      without fees.

    - See :py:func:`test_short_close_with_fee_no_price_movement` for an example
      without price movement.

    - ETH price moves 1500 -> 1400 between open and close
    - Close for profit
    - See that fees have been deducted from the profit
    """

    trader = UnitTestTrader(state)
    portfolio = state.portfolio

    open_estimate = LeverageEstimate.open_short(
        starting_reserve=Decimal(10_000),
        leverage=3.0,
        borrowed_asset_price=1500.0,
        fee=weth_short_identifier_5bps.get_pricing_pair().fee,
        shorting_pair=weth_short_identifier_5bps,
    )

    assert open_estimate.lp_fees == pytest.approx(15)

    short_position, trade, _ = state.trade_short(
        strategy_cycle_at=datetime.datetime.utcnow(),
        pair=weth_short_identifier_5bps,
        borrowed_quantity=-open_estimate.borrowed_quantity,
        collateral_quantity=open_estimate.starting_reserve,
        borrowed_asset_price=open_estimate.borrowed_asset_price,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
        collateral_asset_price=1.0,
        planned_collateral_consumption=open_estimate.additional_collateral_quantity,
        lp_fees_estimated=open_estimate.lp_fees,
    )

    trader.set_perfectly_executed(trade)

    # ETH price 1500 -> 1400
    short_position.revalue_base_asset(
        datetime.datetime.utcnow(),
        1400.0,
    )

    # TODO: Net asset value calculation does not account for fees
    # paid to close a short position

    assert portfolio.get_net_asset_value() == pytest.approx(11985.0)
    assert portfolio.get_unrealised_profit_usd() == pytest.approx(2000)

    assert short_position.loan.collateral.quantity == pytest.approx(Decimal(39984.9999))
    assert short_position.loan.borrowed.quantity == pytest.approx(Decimal(20))
    assert short_position.loan.get_borrow_value() == pytest.approx(28000)

    #
    # Close the short position with trading fees
    #

    estimate = LeverageEstimate.close_short(
        start_collateral=short_position.loan.collateral.quantity,
        start_borrowed=short_position.loan.borrowed.quantity,
        close_size=short_position.loan.borrowed.quantity,
        fee=weth_short_identifier_5bps.get_pricing_pair().fee,
        borrowed_asset_price=1400.0,
    )

    assert estimate.leverage == 1.0  # Reduced USDC leverage to 1.0
    assert estimate.additional_collateral_quantity == pytest.approx(Decimal(-28014.00700350))  # USDC needed to reduce from collateral to close position + fees
    assert estimate.borrowed_quantity == pytest.approx(Decimal(-20))  # How much ETH is bought to close the short
    assert estimate.total_collateral_quantity == pytest.approx(Decimal(11970.99299))  # Collateral left after closing the position
    assert estimate.total_borrowed_quantity == 0  # open vWETH debt left after close
    assert estimate.lp_fees == pytest.approx(14.007)

    # Now close the position
    _, trade_2, _ = state.trade_short(
        closing=True,
        strategy_cycle_at=datetime.datetime.utcnow(),
        pair=weth_short_identifier_5bps,
        borrowed_asset_price=estimate.borrowed_asset_price,
        planned_mid_price=estimate.borrowed_asset_price,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
        collateral_asset_price=1.0,
    )

    trader.set_perfectly_executed(trade_2)

    # After opening and closing short, we have lost money on fees
    assert portfolio.get_cash() == pytest.approx(11970.99299649825)
    assert portfolio.get_net_asset_value() == pytest.approx(11970.99299649825)


def test_short_increase_size(
    state: State,
    weth_short_identifier: TradingPairIdentifier,
    usdc: AssetIdentifier,
):
    """Increase short position.

    Increase short position and maintain the leverage ratio same.

    - Start with 1000 USDC collateral

    - Take 1000 USDC worth of ETH loan, ETH at 1500 USDC/ETH

    - Now you end up with 2000 USDC collateral, 0.66 ETH debt

    - ETH price goes up 1500 -> 1600 so we have unrealised PnL,
      our leverage goes up a bit

    - Increase short size, by allocating more collateral and maintaining the current
      leverage

    - Increaese collateral with $500, while maintaining the current leverage.


    - Make sure the short maintains the existing leverage,
      and buy the additional token amount for the increase collateral

    Based on ``test_short_increase_leverage_and_close``.
    """

    trader = UnitTestTrader(state)

    portfolio = state.portfolio

    # We deposit 1000 USDC as reserves,
    # use it to take a loan WETH at price 1500 USD/ETH
    # for 1000 USDC,
    # This gives us short leverage 1x,
    # long leverage 2x.
    expected_eth_shorted_amount = 1000 / 1500

    # Take 1000 USDC reserves and open a ETH short using it.
    # We should get 800 USDC worth of ETH for this.
    # 800 USDC worth of ETH is
    short_position, trade, created = state.trade_short(
        strategy_cycle_at=datetime.datetime.utcnow(),
        pair=weth_short_identifier,
        borrowed_quantity=-Decimal(expected_eth_shorted_amount),
        collateral_quantity=Decimal(1000),
        borrowed_asset_price=float(1500),  # USDC/ETH price we are going to sell
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
        collateral_asset_price=1.0,
    )

    trader.set_perfectly_executed(trade)

    loan = short_position.loan

    # ETH price 1500 -> 1600
    short_position.revalue_base_asset(
        datetime.datetime.utcnow(),
        1600.0,
    )

    assert loan.get_leverage() == pytest.approx(1.1428571428571423)
    assert loan.get_collateral_quantity() == pytest.approx(Decimal(2000))
    assert loan.get_net_asset_value() == pytest.approx(933.333333)
    assert portfolio.get_net_asset_value() == pytest.approx(9933.333333333334)

    # Trade to increase short position size (sell more)
    #
    # We want to maintain the same leverage and add
    # in another $1000
    #
    # The target should b e
    # $933 + $500 net asset value
    # 1.143x leverage
    # 1600 ETH/USD
    #
    #

    target_params = LeverageEstimate.open_short(
        933.333333 + 500,
        loan.get_leverage(),
        1600.0,
        weth_short_identifier
    )

    assert target_params.borrowed_quantity == pytest.approx(Decimal(1.023809523571428343350194986))
    assert target_params.total_collateral_quantity == pytest.approx(Decimal(3071.42857071428538809751031))
    assert target_params.leverage == pytest.approx(1.1428571428571423)

    collateral_adjustment = Decimal(500)
    increase_borrowed_quantity = loan.calculate_size_adjust(collateral_adjustment)
    assert increase_borrowed_quantity == pytest.approx(Decimal('0.3571428571428570536427926638'))

    assert increase_borrowed_quantity + loan.borrowed.quantity == pytest.approx(Decimal(1.023809523571428343350194986))

    _, trade_2, _ = state.trade_short(
        strategy_cycle_at=datetime.datetime.utcnow(),
        pair=weth_short_identifier,
        borrowed_quantity=-increase_borrowed_quantity,
        collateral_quantity=collateral_adjustment,
        borrowed_asset_price=loan.borrowed.last_usd_price,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
        collateral_asset_price=1.0,
        planned_collateral_consumption=target_params.total_collateral_quantity - loan.collateral.quantity - 500,
    )

    # Check that we have trade and loan parameters correct
    assert trade_2.planned_reserve == 500
    assert trade_2.planned_loan_update.get_leverage() == pytest.approx(1.1428571428571423)
    assert trade_2.planned_quantity == pytest.approx(-increase_borrowed_quantity)
    assert trade_2.planned_loan_update.borrowed.quantity == pytest.approx(Decimal(1.023809523809523683302025176))
    assert trade_2.planned_loan_update.collateral.quantity == pytest.approx(Decimal(3071.428570714285388097510315))
    assert trade_2.planned_collateral_consumption == pytest.approx(Decimal(571.428570714285443608661546))

    trader.set_perfectly_executed(trade_2)

    assert trade_2.executed_collateral_allocation == trade_2.planned_collateral_allocation
    assert trade_2.executed_collateral_consumption == trade_2.executed_collateral_consumption

    # Loan leverage has changed, but our portfolio net asset value stays same
    loan = short_position.loan
    assert loan.get_leverage() == pytest.approx(1.142857142857143)
    assert loan.collateral.quantity == pytest.approx(Decimal(3071.42857071428538809751031))
    assert portfolio.get_cash() == pytest.approx(8500)  # We used $500 more of our cash reserves
    assert portfolio.get_net_asset_value() == pytest.approx(9933.333333333334)


def test_short_decrease_size(
    state: State,
    weth_short_identifier: TradingPairIdentifier,
    usdc: AssetIdentifier,
):
    """Decrease short position.

    Decrease short position and maintain the leverage ratio same.

    - Start with 1000 USDC collateral

    - Take 1000 USDC worth of ETH loan, ETH at 1500 USDC/ETH

    - Now you end up with 2000 USDC collateral, 0.66 ETH debt

    - ETH price goes up 1500 -> 1600 so we have unrealised PnL,
      our leverage goes up a bit

    - Increase short size, by allocating more collateral and maintaining the current
      leverage

    - Decrease collateral with $500, while maintaining the current leverage.

    - Make sure the short maintains the existing leverage,
      and buy the additional token amount for the increase collateral

    Based on ``test_short_increase_leverage_and_close``.
    """

    trader = UnitTestTrader(state)

    portfolio = state.portfolio

    # We deposit 1000 USDC as reserves,
    # use it to take a loan WETH at price 1500 USD/ETH
    # for 1000 USDC,
    # This gives us short leverage 1x,
    # long leverage 2x.
    expected_eth_shorted_amount = 1000 / 1500

    # Take 1000 USDC reserves and open a ETH short using it.
    # We should get 800 USDC worth of ETH for this.
    # 800 USDC worth of ETH is
    short_position, trade, created = state.trade_short(
        strategy_cycle_at=datetime.datetime.utcnow(),
        pair=weth_short_identifier,
        borrowed_quantity=-Decimal(expected_eth_shorted_amount),
        collateral_quantity=Decimal(1000),
        borrowed_asset_price=float(1500),  # USDC/ETH price we are going to sell
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
        collateral_asset_price=1.0,
    )

    trader.set_perfectly_executed(trade)

    loan = short_position.loan

    # ETH price 1500 -> 1600
    short_position.revalue_base_asset(
        datetime.datetime.utcnow(),
        1600.0,
    )

    assert loan.get_leverage() == pytest.approx(1.1428571428571423)
    assert loan.get_collateral_quantity() == pytest.approx(Decimal(2000))
    assert loan.get_net_asset_value() == pytest.approx(933.333333)
    assert portfolio.get_net_asset_value() == pytest.approx(9933.333333333334)

    # Trade to increase short position size (sell more)
    #
    # We want to maintain the same leverage and remove
    # $500 so that our total porfolio exposure for this position
    # is $433

    #
    # The target should b e
    # $433 net asset value
    # 1.143x leverage
    # 1600 ETH/USD
    #
    #

    target_params = LeverageEstimate.open_short(
        933.333333 - 500,
        loan.get_leverage(),
        1600.0,
        weth_short_identifier
    )

    assert target_params.borrowed_quantity == pytest.approx(Decimal(0.3095238092857142360646096573))
    assert target_params.total_collateral_quantity == pytest.approx(Decimal(928.5714278571428164405737896))
    assert target_params.leverage == pytest.approx(1.1428571428571423)

    # How much we will pay back our vToken debt
    decrease_borrow_quantity = loan.borrowed.quantity - target_params.borrowed_quantity
    assert decrease_borrow_quantity == pytest.approx(Decimal('0.3571428573809523935946228552'))

    reserves_released = Decimal(500)

    _, trade_2, _ = state.trade_short(
        strategy_cycle_at=datetime.datetime.utcnow(),
        pair=weth_short_identifier,
        borrowed_quantity=decrease_borrow_quantity, # Buy back shorted tokens to decrease exposute
        collateral_quantity=0,  # Not used when releasing reserves
        borrowed_asset_price=loan.borrowed.last_usd_price,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
        collateral_asset_price=1.0,
        planned_collateral_allocation=-reserves_released,
        # See comments in update_short_loan()
        planned_collateral_consumption=target_params.total_collateral_quantity - loan.collateral.quantity + reserves_released,
    )

    # Check that we have trade and loan parameters correct
#    assert trade_2.planned_reserve == -500
    assert trade_2.planned_quantity == pytest.approx(decrease_borrow_quantity)

    assert trade_2.planned_loan_update.borrowed.quantity == pytest.approx(Decimal(0.3095238092857142360646096573))
    assert trade_2.planned_loan_update.collateral.quantity == pytest.approx(Decimal(928.571427857142816440573790))
    assert trade_2.planned_loan_update.get_leverage() == pytest.approx(1.1428571428571423)

    # See comments in update_short_loan()
    assert trade_2.planned_collateral_consumption == pytest.approx(Decimal('-571.428572142857128048274979'))
    assert trade_2.planned_collateral_allocation == pytest.approx(Decimal('-500'))

    trader.set_perfectly_executed(trade_2)

    assert trade_2.executed_collateral_allocation == trade_2.planned_collateral_allocation
    assert trade_2.executed_collateral_consumption == trade_2.executed_collateral_consumption

    # Loan leverage has changed, but our portfolio net asset value stays same
    loan = short_position.loan
    assert loan.get_leverage() == pytest.approx(1.142857142857143)
    assert loan.collateral.quantity == pytest.approx(Decimal(928.571427857142816440573790))
    assert loan.get_net_asset_value() == pytest.approx(433.33333300000004)
    assert portfolio.get_cash() == pytest.approx(9500)  # We get $500 back
    assert portfolio.get_net_asset_value() == pytest.approx(9933.333333333334)  # NAV does not change in zero fee trade

