"""Cash-aware CCTP bridge-out sizing regression tests.

The bridge-out branch of :py:func:`inject_cctp_bridge_trades` used to size a
bridge-out purely from the alpha model's net planned buy on a satellite chain,
ignoring:

- capital already idle on the satellite chain (tracked as available bridge
  capital on the chain's bridge position), and
- whether the primary chain can actually fund the bridge-out.

This let a cross-chain backtest over-bridge and drive the primary-chain reserve
negative, crashing deep in ``simulate_bridge()`` with ``OutOfSimulatedBalance``
(surfaced as ``BacktestExecutionFailed``).

These tests pin the cash-aware behaviour:

- ``test_bridge_out_funds_from_idle_satellite_capital`` reproduces the crash
  via the simulated wallet and asserts no redundant bridge-out is injected.
- ``test_bridge_out_nets_against_partial_idle_capital`` checks only the
  shortfall is bridged.
- ``test_bridge_out_raises_clearly_when_underfunded`` checks an explicit early
  :py:class:`NotEnoughMoney` instead of a deep execution crash.
"""

import datetime
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from tradeexecutor.backtest.backtest_execution import BacktestExecution
from tradeexecutor.backtest.backtest_routing import BacktestRoutingIgnoredModel, BacktestRoutingState
from tradeexecutor.backtest.simulated_wallet import SimulatedWallet
from tradeexecutor.ethereum.cctp.planner import inject_cctp_bridge_trades
from tradeexecutor.state.identifier import (
    AssetIdentifier,
    TradingPairIdentifier,
    TradingPairKind,
)
from tradeexecutor.state.portfolio import NotEnoughMoney
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeType

#: Arbitrum native USDC — the portfolio reserve / primary chain
USDC_ARBITRUM_ADDRESS = "0xaf88d065e77c8cc2239327c5edb3a432268e5831"

#: Base native USDC — satellite chain stablecoin
USDC_BASE_ADDRESS = "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"

PRIMARY_CHAIN_ID = 42161  # Arbitrum
SATELLITE_CHAIN_ID = 8453  # Base

TS = datetime.datetime(2025, 1, 1, 12, 0, 0)


@pytest.fixture()
def usdc_arbitrum() -> AssetIdentifier:
    """USDC on Arbitrum — the reserve currency on the primary chain."""
    return AssetIdentifier(
        chain_id=PRIMARY_CHAIN_ID,
        address=USDC_ARBITRUM_ADDRESS,
        token_symbol="USDC",
        decimals=6,
    )


@pytest.fixture()
def usdc_base() -> AssetIdentifier:
    """USDC on Base — the satellite chain stablecoin."""
    return AssetIdentifier(
        chain_id=SATELLITE_CHAIN_ID,
        address=USDC_BASE_ADDRESS,
        token_symbol="USDC",
        decimals=6,
    )


@pytest.fixture()
def cctp_pair(usdc_arbitrum: AssetIdentifier, usdc_base: AssetIdentifier) -> TradingPairIdentifier:
    """CCTP bridge pair: Arbitrum -> Base."""
    return TradingPairIdentifier(
        base=usdc_base,
        quote=usdc_arbitrum,
        pool_address="0x28b5a0e9c621a5badaa536219b3a228c8168cf5d",
        exchange_address="0x28b5a0e9c621a5badaa536219b3a228c8168cf5d",
        internal_id=1,
        fee=0,
        kind=TradingPairKind.cctp_bridge,
        other_data={"bridge_protocol": "cctp"},
    )


@pytest.fixture()
def satellite_pair(usdc_base: AssetIdentifier) -> TradingPairIdentifier:
    """A satellite-chain spot pair on Base, quoted in native USDC.

    Priced 1:1 so reserve == quantity, which keeps the bridge-capital
    accounting easy to assert.
    """
    base = AssetIdentifier(
        chain_id=SATELLITE_CHAIN_ID,
        address="0x0000000000000000000000000000000000000011",
        token_symbol="satBASE",
        decimals=18,
    )
    return TradingPairIdentifier(
        base=base,
        quote=usdc_base,
        pool_address="0x0000000000000000000000000000000000000022",
        exchange_address="0x0000000000000000000000000000000000000033",
        internal_id=2,
        fee=0,
        kind=TradingPairKind.spot_market_hold,
    )


def _make_mock_universe(pairs: list[TradingPairIdentifier]):
    """Build a mock strategy universe that yields the given pairs."""
    mock = MagicMock()
    mock.iterate_pairs.return_value = pairs
    return mock


def _make_state(usdc_arbitrum: AssetIdentifier, reserve_amount: Decimal) -> State:
    """Create a fresh state with reserves on the primary chain."""
    state = State()
    state.portfolio.initialise_reserves(usdc_arbitrum)
    reserve = state.portfolio.get_default_reserve_position()
    reserve.quantity = reserve_amount
    reserve.reserve_token_price = 1.0
    return state


def _routing(wallet: SimulatedWallet, usdc_arbitrum: AssetIdentifier):
    """Minimal routing model/state for the backtest executor."""
    routing_model = BacktestRoutingIgnoredModel(reserve_token_address=usdc_arbitrum.address)
    routing_state = BacktestRoutingState(pair_universe=None, wallet=wallet)
    return routing_model, routing_state


def _establish_idle_satellite_capital(
    state: State,
    execution: BacktestExecution,
    wallet: SimulatedWallet,
    cctp_pair: TradingPairIdentifier,
    usdc_arbitrum: AssetIdentifier,
    amount: Decimal,
):
    """Bridge ``amount`` from primary to satellite and settle it.

    Leaves the satellite chain holding ``amount`` idle USDC, represented as an
    open bridge position with ``amount`` available bridge capital.
    """
    _, bridge_out, _ = state.create_trade(
        strategy_cycle_at=TS,
        pair=cctp_pair,
        quantity=None,
        reserve=amount,
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_arbitrum,
        reserve_currency_price=1.0,
    )
    routing_model, routing_state = _routing(wallet, usdc_arbitrum)
    execution.execute_trades(
        ts=TS,
        state=state,
        trades=[bridge_out],
        routing_model=routing_model,
        routing_state=routing_state,
        check_balances=False,
    )
    assert bridge_out.is_success()


def _create_satellite_buy(
    state: State,
    satellite_pair: TradingPairIdentifier,
    usdc_arbitrum: AssetIdentifier,
    reserve: Decimal,
) -> TradeExecution:
    """Create (but do not execute) a satellite-chain spot buy."""
    _, buy, _ = state.create_trade(
        strategy_cycle_at=TS,
        pair=satellite_pair,
        quantity=None,
        reserve=reserve,
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_arbitrum,
        reserve_currency_price=1.0,
    )
    return buy


def test_bridge_out_funds_from_idle_satellite_capital(
    usdc_arbitrum: AssetIdentifier,
    cctp_pair: TradingPairIdentifier,
    satellite_pair: TradingPairIdentifier,
):
    """Idle satellite capital funds the buy — no redundant bridge-out, no crash.

    1. Seed 25_000 primary reserve and bridge 20_000 to the satellite, leaving
       5_000 primary reserve and 20_000 idle on the satellite.
    2. Ask for an 11_500 satellite buy and inject bridge trades.
    3. Execute the result.

    On the buggy code a second 11_500 bridge-out is injected, which drains the
    5_000 primary reserve and raises ``BacktestExecutionFailed`` during step 3.
    With the fix, no bridge-out is injected; the buy allocates from the idle
    bridge capital and the primary reserve is untouched.
    """
    wallet = SimulatedWallet()
    wallet.set_balance(usdc_arbitrum, Decimal(25_000))
    state = _make_state(usdc_arbitrum, Decimal(25_000))
    execution = BacktestExecution(wallet=wallet)

    # 1. Park 20_000 idle on the satellite; primary reserve falls to 5_000.
    _establish_idle_satellite_capital(state, execution, wallet, cctp_pair, usdc_arbitrum, Decimal(20_000))
    reserve = state.portfolio.get_default_reserve_position()
    assert reserve.quantity == Decimal(5_000)

    # 2. Inject bridge trades for an 11_500 satellite buy.
    buy = _create_satellite_buy(state, satellite_pair, usdc_arbitrum, Decimal(11_500))
    universe = _make_mock_universe([cctp_pair, satellite_pair])
    result = inject_cctp_bridge_trades(
        state=state,
        trades=[buy],
        strategy_universe=universe,
        primary_chain_id=PRIMARY_CHAIN_ID,
        ts=TS,
        reserve_asset=usdc_arbitrum,
    )

    # 3. Execute — must not raise (reproduces the crash on buggy code).
    routing_model, routing_state = _routing(wallet, usdc_arbitrum)
    execution.execute_trades(
        ts=TS,
        state=state,
        trades=sorted(result, key=lambda t: t.get_execution_sort_position()),
        routing_model=routing_model,
        routing_state=routing_state,
        check_balances=False,
    )

    # No redundant bridge-out was injected.
    assert [t for t in result if t.pair.is_cctp_bridge()] == []
    # The buy executed and was funded from the idle bridge capital.
    assert buy.is_success()
    assert reserve.quantity == Decimal(5_000)
    bridge_pos = state.portfolio.get_bridge_position_for_chain(SATELLITE_CHAIN_ID)
    assert bridge_pos.bridge_capital_allocated == Decimal(11_500)


def test_bridge_out_nets_against_partial_idle_capital(
    usdc_arbitrum: AssetIdentifier,
    cctp_pair: TradingPairIdentifier,
    satellite_pair: TradingPairIdentifier,
):
    """Only the shortfall above idle satellite capital is bridged out.

    Idle 4_000 on the satellite, an 11_500 buy, ample primary reserve -> a
    single 7_500 bridge-out (= 11_500 - 4_000).
    """
    wallet = SimulatedWallet()
    wallet.set_balance(usdc_arbitrum, Decimal(50_000))
    state = _make_state(usdc_arbitrum, Decimal(50_000))
    execution = BacktestExecution(wallet=wallet)

    _establish_idle_satellite_capital(state, execution, wallet, cctp_pair, usdc_arbitrum, Decimal(4_000))

    buy = _create_satellite_buy(state, satellite_pair, usdc_arbitrum, Decimal(11_500))
    universe = _make_mock_universe([cctp_pair, satellite_pair])
    result = inject_cctp_bridge_trades(
        state=state,
        trades=[buy],
        strategy_universe=universe,
        primary_chain_id=PRIMARY_CHAIN_ID,
        ts=TS,
        reserve_asset=usdc_arbitrum,
    )

    bridge_trades = [t for t in result if t.pair.is_cctp_bridge()]
    assert len(bridge_trades) == 1
    bridge_out = bridge_trades[0]
    assert bridge_out.is_buy()
    assert bridge_out.planned_reserve == Decimal(7_500)
    assert bridge_out.pair.get_destination_chain_id() == SATELLITE_CHAIN_ID


def test_bridge_out_raises_clearly_when_underfunded(
    usdc_arbitrum: AssetIdentifier,
    cctp_pair: TradingPairIdentifier,
    satellite_pair: TradingPairIdentifier,
):
    """A bridge-out that exceeds fundable primary reserve raises NotEnoughMoney.

    No idle satellite capital, an 11_500 buy, only 5_000 fundable primary
    reserve -> an explicit early :py:class:`NotEnoughMoney`, not a deep
    ``BacktestExecutionFailed``/``OutOfSimulatedBalance`` during execution.
    """
    state = _make_state(usdc_arbitrum, Decimal(5_000))

    buy = _create_satellite_buy(state, satellite_pair, usdc_arbitrum, Decimal(11_500))
    universe = _make_mock_universe([cctp_pair, satellite_pair])

    with pytest.raises(NotEnoughMoney):
        inject_cctp_bridge_trades(
            state=state,
            trades=[buy],
            strategy_universe=universe,
            primary_chain_id=PRIMARY_CHAIN_ID,
            ts=TS,
            reserve_asset=usdc_arbitrum,
        )


def test_bridge_back_capped_to_available_bridge_capital(
    usdc_arbitrum: AssetIdentifier,
    cctp_pair: TradingPairIdentifier,
    satellite_pair: TradingPairIdentifier,
):
    """A bridge-back is capped to available bridge capital, not the full net sell.

    Protects in-flight satellite deposits: when a net sell exceeds the currently
    available bridge capital (because the rest is committed to a not-yet-settled
    deposit), only the available amount may be bridged back. The remainder
    bridges back on a later cycle once the deposit settles.

    1. Bridge 10_000 to the satellite, then deploy 7_000 into a satellite buy,
       leaving 3_000 available bridge capital.
    2. Ask to bridge back the whole 7_000 position via a net sell.
    3. Assert the injected bridge-back is capped to 3_000, not 7_000.
    """
    wallet = SimulatedWallet()
    wallet.set_balance(usdc_arbitrum, Decimal(50_000))
    state = _make_state(usdc_arbitrum, Decimal(50_000))
    execution = BacktestExecution(wallet=wallet)

    # 1. Park 10_000 on the satellite, deploy 7_000 -> 3_000 available.
    _establish_idle_satellite_capital(state, execution, wallet, cctp_pair, usdc_arbitrum, Decimal(10_000))
    buy = _create_satellite_buy(state, satellite_pair, usdc_arbitrum, Decimal(7_000))
    routing_model, routing_state = _routing(wallet, usdc_arbitrum)
    execution.execute_trades(
        ts=TS,
        state=state,
        trades=[buy],
        routing_model=routing_model,
        routing_state=routing_state,
        check_balances=False,
    )
    bridge_pos = state.portfolio.get_bridge_position_for_chain(SATELLITE_CHAIN_ID)
    assert bridge_pos.get_available_bridge_capital() == Decimal(3_000)

    # 2. Net sell of the whole 7_000 satellite position.
    sat_pos = state.portfolio.get_open_position_for_pair(satellite_pair)
    _, sell, _ = state.create_trade(
        strategy_cycle_at=TS,
        pair=satellite_pair,
        quantity=-sat_pos.get_quantity(),
        reserve=None,
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_arbitrum,
        reserve_currency_price=1.0,
        position=sat_pos,
        closing=True,
    )
    universe = _make_mock_universe([cctp_pair, satellite_pair])
    result = inject_cctp_bridge_trades(
        state=state,
        trades=[sell],
        strategy_universe=universe,
        primary_chain_id=PRIMARY_CHAIN_ID,
        ts=TS,
        reserve_asset=usdc_arbitrum,
    )

    # 3. Bridge-back capped to the 3_000 available, not the full 7_000 net sell.
    bridge_backs = [t for t in result if t.pair.is_cctp_bridge()]
    assert len(bridge_backs) == 1
    assert bridge_backs[0].is_sell()
    assert bridge_backs[0].planned_quantity == Decimal(-3_000)
