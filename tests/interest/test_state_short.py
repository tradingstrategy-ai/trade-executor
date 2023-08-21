"""Test opening short positions.

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
from tradingstrategy.lending import LendingProtocolType


@pytest.fixture()
def usdc() -> AssetIdentifier:
    """Mock USDC."""
    return AssetIdentifier(ChainId.polygon.value, "0x0", "USDC", 6)


@pytest.fixture()
def ausdc() -> AssetIdentifier:
    """Mock aUSDC. Aave's interest accruing USDC where balanceOf() is dynamic."""
    # https://etherscan.io/token/0xbcca60bb61934080951369a648fb03df4f96263c#readProxyContract
    return AssetIdentifier(ChainId.polygon.value, "0x1", "aPolUSDC", 6)


@pytest.fixture()
def weth() -> AssetIdentifier:
    return AssetIdentifier(ChainId.polygon.value, "0x2", "WETH", 18)


@pytest.fixture()
def ausdc() -> AssetIdentifier:
    return AssetIdentifier(ChainId.polygon.value, "0x3", "aPolUSDC", 18)


@pytest.fixture()
def vweth() -> AssetIdentifier:
    return AssetIdentifier(ChainId.polygon.value, "0x4", "variableDebtPolWETH", 18)


@pytest.fixture()
def lending_protocol_address() -> str:
    """Mock some assets.

    TODO: What is unique address to identify Aave deployments?
    """
    return ZERO_ADDRESS


@pytest.fixture()
def weth_short_identifier(usdc, weth) -> TradingPairIdentifier:
    """Sets up a lending pool"""
    return TradingPairIdentifier(
        weth,
        usdc,
        "0x1",
        ZERO_ADDRESS,
        internal_id=1,
        kind=TradingPairKind.lending_protocol_short,
        collateral_factor=0.8,
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


def test_open_short(
        state: State,
        weth_short_identifier: TradingPairIdentifier,
        usdc: AssetIdentifier,
):
    """Opening a short position.

    - Supply USDC as a collateral

    - Borrow out WETH (default leverage ETH 80% of USDC value, or 0.8x)

    - Sell WETH

    - In our wallet, we have now USDC (our capital), USDC (from sold WETH),
      vWETH to mark out debt

    - Short position gains interest on the borrowed ETH (negative),
      which we can read from vWETH balance

    """
    assert weth_short_identifier.kind.is_shorting()
    assert weth_short_identifier.get_lending_protocol() == LendingProtocolType.aave_v3
    assert weth_short_identifier.base.token_symbol == "WETH"
    assert weth_short_identifier.quote.token_symbol == "USDC"
    assert weth_short_identifier.get_max_leverage_at_open() == 5.00

    trader = DummyTestTrader(state)

    # Take 1000 USDC reserves and open a ETH short using it.
    # We should get 800 USDC worth of ETH for this.
    short_position, trade, created = state.create_short(
        strategy_cycle_at=datetime.datetime.utcnow(),
        pair=weth_short_identifier,
        quantity=None,
        reserve=Decimal(1000),
        assumed_price=1500,  # USDC/ETH price we are going to sell
        trade_type=TradeType.lending_protocol_short,
        reserve_currency=usdc,
        reserve_currency_price=1.0,
        leverage=0.8,
    )

    trader.set_perfectly_executed(trade)

    expected_eth_shorted_amount = 1000 * 0.8 * 1500

    assert created
    assert trade.is_success()
    assert trade.trade_type == TradeType.lending_protocol_short

    assert short_position.is_short()
    assert short_position.interest is not None
    assert short_position.interest.opening_amount == pytest.approx(Decimal(expected_eth_shorted_amount))
    assert short_position.interest.last_accrued_interest == 0
    assert short_position.get_value() == Decimal(1000)

    assert short_position.get_opening_price() == 1500
    assert short_position.loan is not None
    assert short_position.loan.collateral.asset.token_symbol == "USDC"
    assert short_position.loan.collateral.quantity == Decimal(1000)
    assert short_position.loan.collateral.presentation.token_symbol == "aPolUSDC"
    assert short_position.loan.borrowed.quantity == pytest.approx(Decimal(expected_eth_shorted_amount))
    assert short_position.loan.borrowed.asset.token_symbol == "WETH"
    assert short_position.loan.presentation.token_symbol == "variableDebtPolWETH"
    assert short_position.loan.get_leverage() == 0.8
    assert short_position.get_unrealised_profit_usd() == 0
    assert short_position.get_realised_profit_usd() == 0

    assert state.portfolio.get_total_equity() == 10000


