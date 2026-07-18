"""Idle CCTP bridge capital sweep-back regression tests.

Before issue #1562, ``inject_cctp_bridge_trades`` only bridged satellite capital
back to the primary hub when a satellite chain had a net sell that cycle, or when
the primary chain was short of cash. Capital that became idle on a satellite with
no matching same-cycle demand — a settled async redemption, a withheld net-sell
excess, a profitable round-trip — stayed parked on the bridge position earning
nothing.

The planner now runs an idle-capital sweep after demand-driven bridge planning:
any free idle satellite capital above the configured dust buffer is bridged back
to the hub (controlled by ``sweep_idle_bridge_capital`` / ``bridge_sweep_min_usd``,
on by default). These tests pin that behaviour:

- ``test_idle_capital_swept_back_to_hub`` — a quiet cycle with settled idle
  capital and no directional trades produces a single closing bridge-back.
- ``test_idle_sweep_guard_paths`` — the sweep respects same-cycle buys, capital
  committed to unsettled async deposits, the dust buffer, the disable flag, and
  merges cleanly with a same-cycle net sell.
"""

import datetime
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from tradeexecutor.backtest.backtest_execution import BacktestExecution
from tradeexecutor.backtest.backtest_routing import BacktestRoutingIgnoredModel, BacktestRoutingState
from tradeexecutor.backtest.simulated_wallet import SimulatedWallet
from tradeexecutor.analysis.cctp import analyse_idle_bridge_capital
from tradeexecutor.ethereum.cctp.planner import inject_cctp_bridge_trades
from tradeexecutor.state.identifier import (
    AssetIdentifier,
    TradingPairIdentifier,
    TradingPairKind,
)
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
    """A satellite-chain spot pair on Base, quoted 1:1 in native USDC."""
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

    Leaves the satellite holding ``amount`` idle USDC, represented as an open
    bridge position with ``amount`` available bridge capital.
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


@pytest.mark.timeout(300)
def test_idle_capital_swept_back_to_hub(
    usdc_arbitrum: AssetIdentifier,
    cctp_pair: TradingPairIdentifier,
    satellite_pair: TradingPairIdentifier,
):
    """A quiet cycle with settled idle satellite capital sweeps it back to the hub.

    This is the core issue #1562 fix: with no directional demand at all, settled
    idle capital used to sit on the bridge position forever. The sweep now emits a
    single closing bridge-back that returns the cash to the primary reserve.

    1. Seed 25_000 primary reserve and bridge 20_000 to the satellite, leaving
       5_000 primary reserve and 20_000 idle bridge capital.
    2. Inject bridge trades with an EMPTY trade list (a quiet cycle) and the sweep
       enabled.
    3. Assert exactly one closing bridge-back for the full 20_000 tagged
       ``idle_sweep``.
    4. Execute it and assert the hub reserve is made whole (25_000) and no open
       bridge position retains capital above the dust buffer.
    """
    # 1. Park 20_000 idle on the satellite; primary reserve falls to 5_000.
    wallet = SimulatedWallet()
    wallet.set_balance(usdc_arbitrum, Decimal(25_000))
    state = _make_state(usdc_arbitrum, Decimal(25_000))
    execution = BacktestExecution(wallet=wallet)
    _establish_idle_satellite_capital(state, execution, wallet, cctp_pair, usdc_arbitrum, Decimal(20_000))
    assert state.portfolio.get_default_reserve_position().quantity == Decimal(5_000)

    # 2. Quiet cycle: no directional trades, sweep enabled (default).
    universe = _make_mock_universe([cctp_pair, satellite_pair])
    result = inject_cctp_bridge_trades(
        state=state,
        trades=[],
        strategy_universe=universe,
        primary_chain_id=PRIMARY_CHAIN_ID,
        ts=TS,
        reserve_asset=usdc_arbitrum,
    )

    # 3. Exactly one closing bridge-back for the full idle balance, tagged idle_sweep.
    bridge_backs = [t for t in result if t.pair.is_cctp_bridge()]
    assert len(bridge_backs) == 1
    sweep = bridge_backs[0]
    assert sweep.is_sell()
    assert sweep.planned_quantity == Decimal(-20_000)
    assert sweep.closing is True
    assert sweep.other_data["cctp_planning_amounts"] == {"idle_sweep": "20000"}

    # 4. Execute and assert the hub reserve is whole and no idle capital remains.
    routing_model, routing_state = _routing(wallet, usdc_arbitrum)
    execution.execute_trades(
        ts=TS,
        state=state,
        trades=sorted(result, key=lambda t: t.get_execution_sort_position()),
        routing_model=routing_model,
        routing_state=routing_state,
        check_balances=False,
    )
    assert sweep.is_success()
    assert state.portfolio.get_default_reserve_position().quantity == pytest.approx(Decimal(25_000))
    open_bridge = state.portfolio.get_bridge_position_for_chain(SATELLITE_CHAIN_ID)
    assert open_bridge is None or open_bridge.get_available_bridge_capital() < Decimal(1)


@pytest.mark.timeout(300)
def test_idle_sweep_guard_paths(
    usdc_arbitrum: AssetIdentifier,
    cctp_pair: TradingPairIdentifier,
    satellite_pair: TradingPairIdentifier,
):
    """The idle sweep respects same-cycle demand, commitments, buffers and the flag.

    Each numbered step is an independent scenario built on a fresh state, so one
    test covers every guard on the sweep without a proliferation of near-identical
    cases.

    1. Same-cycle satellite buy: only the free remainder above the buy is swept.
    2. Capital committed to an unsettled async deposit is excluded from the sweep.
    3. Idle below ``bridge_sweep_min_usd`` is left as a dust buffer (no trade).
    4. ``sweep_idle_bridge_capital=False`` reproduces the old behaviour (no trade).
    5. A same-cycle net sell and settled idle capital merge into one bridge-back
       whose planning-amount breakdown splits ``net_sell`` and ``idle_sweep``.
       The sweep only draws the physically settled idle capital, never the sell
       proceeds (which the net-sell bridge-back already carries home).
    """
    universe = _make_mock_universe([cctp_pair, satellite_pair])

    # 1. Same-cycle satellite buy: 20_000 idle, an 11_500 buy -> only 8_500 swept.
    wallet = SimulatedWallet()
    wallet.set_balance(usdc_arbitrum, Decimal(25_000))
    state = _make_state(usdc_arbitrum, Decimal(25_000))
    execution = BacktestExecution(wallet=wallet)
    _establish_idle_satellite_capital(state, execution, wallet, cctp_pair, usdc_arbitrum, Decimal(20_000))
    buy = _create_satellite_buy(state, satellite_pair, usdc_arbitrum, Decimal(11_500))
    result = inject_cctp_bridge_trades(
        state=state,
        trades=[buy],
        strategy_universe=universe,
        primary_chain_id=PRIMARY_CHAIN_ID,
        ts=TS,
        reserve_asset=usdc_arbitrum,
    )
    bridge_backs = [t for t in result if t.pair.is_cctp_bridge()]
    assert len(bridge_backs) == 1
    assert bridge_backs[0].planned_quantity == Decimal(-8_500)
    assert bridge_backs[0].other_data["cctp_planning_amounts"] == {"idle_sweep": "8500"}

    # 2. Committed capital: 20_000 idle with 15_000 locked to an unsettled async
    #    deposit (simulated by the allocation ledger a prior cycle would leave)
    #    -> only the free 5_000 is swept.
    wallet = SimulatedWallet()
    wallet.set_balance(usdc_arbitrum, Decimal(25_000))
    state = _make_state(usdc_arbitrum, Decimal(25_000))
    execution = BacktestExecution(wallet=wallet)
    _establish_idle_satellite_capital(state, execution, wallet, cctp_pair, usdc_arbitrum, Decimal(20_000))
    bridge_pos = state.portfolio.get_bridge_position_for_chain(SATELLITE_CHAIN_ID)
    bridge_pos.bridge_capital_allocated = Decimal(15_000)
    result = inject_cctp_bridge_trades(
        state=state,
        trades=[],
        strategy_universe=universe,
        primary_chain_id=PRIMARY_CHAIN_ID,
        ts=TS,
        reserve_asset=usdc_arbitrum,
    )
    bridge_backs = [t for t in result if t.pair.is_cctp_bridge()]
    assert len(bridge_backs) == 1
    assert bridge_backs[0].planned_quantity == Decimal(-5_000)

    # 3. Dust buffer: 50 idle with a 100 minimum -> nothing swept.
    wallet = SimulatedWallet()
    wallet.set_balance(usdc_arbitrum, Decimal(1_000))
    state = _make_state(usdc_arbitrum, Decimal(1_000))
    execution = BacktestExecution(wallet=wallet)
    _establish_idle_satellite_capital(state, execution, wallet, cctp_pair, usdc_arbitrum, Decimal(50))
    result = inject_cctp_bridge_trades(
        state=state,
        trades=[],
        strategy_universe=universe,
        primary_chain_id=PRIMARY_CHAIN_ID,
        ts=TS,
        reserve_asset=usdc_arbitrum,
        bridge_sweep_min_usd=100.0,
    )
    assert [t for t in result if t.pair.is_cctp_bridge()] == []

    # 4. Disabled: 20_000 idle, sweep off -> old behaviour, no bridge trade.
    wallet = SimulatedWallet()
    wallet.set_balance(usdc_arbitrum, Decimal(25_000))
    state = _make_state(usdc_arbitrum, Decimal(25_000))
    execution = BacktestExecution(wallet=wallet)
    _establish_idle_satellite_capital(state, execution, wallet, cctp_pair, usdc_arbitrum, Decimal(20_000))
    result = inject_cctp_bridge_trades(
        state=state,
        trades=[],
        strategy_universe=universe,
        primary_chain_id=PRIMARY_CHAIN_ID,
        ts=TS,
        reserve_asset=usdc_arbitrum,
        sweep_idle_bridge_capital=False,
    )
    assert [t for t in result if t.pair.is_cctp_bridge()] == []

    # 5. Merge: bridge 12_000, deploy 2_000 then sell it non-closing, leaving
    #    10_000 physically settled idle plus a 2_000 same-cycle sync sell. The
    #    net-sell bridge-back carries the 2_000 sell home; the sweep adds the
    #    8_000 settled idle it can prove is physically present (10_000 idle minus
    #    the 2_000 the bridge-back already reserved). One merged bridge-back of
    #    10_000 splits net_sell 2_000 + idle_sweep 8_000.
    wallet = SimulatedWallet()
    wallet.set_balance(usdc_arbitrum, Decimal(50_000))
    state = _make_state(usdc_arbitrum, Decimal(50_000))
    execution = BacktestExecution(wallet=wallet)
    _establish_idle_satellite_capital(state, execution, wallet, cctp_pair, usdc_arbitrum, Decimal(12_000))
    buy = _create_satellite_buy(state, satellite_pair, usdc_arbitrum, Decimal(2_000))
    routing_model, routing_state = _routing(wallet, usdc_arbitrum)
    execution.execute_trades(
        ts=TS,
        state=state,
        trades=[buy],
        routing_model=routing_model,
        routing_state=routing_state,
        check_balances=False,
    )
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
        closing=False,
    )
    result = inject_cctp_bridge_trades(
        state=state,
        trades=[sell],
        strategy_universe=universe,
        primary_chain_id=PRIMARY_CHAIN_ID,
        ts=TS,
        reserve_asset=usdc_arbitrum,
    )
    bridge_backs = [t for t in result if t.pair.is_cctp_bridge()]
    assert len(bridge_backs) == 1
    assert bridge_backs[0].planned_quantity == Decimal(-10_000)
    assert bridge_backs[0].other_data["cctp_planning_amounts"] == {
        "net_sell": "2000",
        "idle_sweep": "8000",
    }


@pytest.mark.timeout(300)
def test_sweep_captures_realised_satellite_profits(
    usdc_arbitrum: AssetIdentifier,
    usdc_base: AssetIdentifier,
    cctp_pair: TradingPairIdentifier,
    satellite_pair: TradingPairIdentifier,
):
    """The sweep captures realised satellite profits, not just the gross bridged amount.

    Satellite trades only ever mutate ``bridge_capital_allocated`` — the bridge
    position quantity stays at the gross bridged amount — so a profitable
    round-trip drives ``allocated`` negative and the physical satellite USDC
    (``available = quantity − allocated``) exceeds the position quantity. The
    sweep must move the full physical balance; clamping to ``get_quantity()``
    would strand the profit unsweepable forever (review finding on the first
    implementation).

    1. Bridge 10_000 to the satellite and deploy 2_000 into a satellite position.
    2. Sell the position for 2_500 (a 500 profit), leaving ``allocated`` at −500
       and 10_500 physical satellite USDC against a 10_000 position quantity.
    3. Run a quiet-cycle sweep and execute it.
    4. Assert the full 10_500 (profit included) lands back on the hub reserve and
       no idle capital remains on the satellite.
    """
    # 1. Bridge 10_000, deploy 2_000.
    wallet = SimulatedWallet()
    wallet.set_balance(usdc_arbitrum, Decimal(50_000))
    state = _make_state(usdc_arbitrum, Decimal(50_000))
    execution = BacktestExecution(wallet=wallet)
    _establish_idle_satellite_capital(state, execution, wallet, cctp_pair, usdc_arbitrum, Decimal(10_000))
    buy = _create_satellite_buy(state, satellite_pair, usdc_arbitrum, Decimal(2_000))
    routing_model, routing_state = _routing(wallet, usdc_arbitrum)
    execution.execute_trades(
        ts=TS,
        state=state,
        trades=[buy],
        routing_model=routing_model,
        routing_state=routing_state,
        check_balances=False,
    )

    # 2. Sell for 2_500 — price moved 1.0 -> 1.25, realising a 500 profit.
    sat_pos = state.portfolio.get_open_position_for_pair(satellite_pair)
    _, sell, _ = state.create_trade(
        strategy_cycle_at=TS,
        pair=satellite_pair,
        quantity=-sat_pos.get_quantity(),
        reserve=None,
        assumed_price=1.25,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_arbitrum,
        reserve_currency_price=1.0,
        position=sat_pos,
        closing=True,
    )
    execution.execute_trades(
        ts=TS,
        state=state,
        trades=[sell],
        routing_model=routing_model,
        routing_state=routing_state,
        check_balances=False,
    )
    bridge_pos = state.portfolio.get_bridge_position_for_chain(SATELLITE_CHAIN_ID)
    assert bridge_pos.bridge_capital_allocated == Decimal(-500)
    assert bridge_pos.get_available_bridge_capital() == Decimal(10_500)
    assert wallet.get_balance(usdc_base) == Decimal(10_500)

    # 3. Quiet-cycle sweep.
    universe = _make_mock_universe([cctp_pair, satellite_pair])
    result = inject_cctp_bridge_trades(
        state=state,
        trades=[],
        strategy_universe=universe,
        primary_chain_id=PRIMARY_CHAIN_ID,
        ts=TS,
        reserve_asset=usdc_arbitrum,
    )
    bridge_backs = [t for t in result if t.pair.is_cctp_bridge()]
    assert len(bridge_backs) == 1
    assert bridge_backs[0].planned_quantity == Decimal(-10_500)
    execution.execute_trades(
        ts=TS,
        state=state,
        trades=sorted(result, key=lambda t: t.get_execution_sort_position()),
        routing_model=routing_model,
        routing_state=routing_state,
        check_balances=False,
    )

    # 4. Full 10_500 back on the hub; nothing idle on the satellite.
    assert state.portfolio.get_default_reserve_position().quantity == pytest.approx(Decimal(50_500))
    assert wallet.get_balance(usdc_base) == Decimal(0)
    open_bridge = state.portfolio.get_bridge_position_for_chain(SATELLITE_CHAIN_ID)
    assert open_bridge is None or open_bridge.get_available_bridge_capital() < Decimal(1)


@pytest.mark.timeout(300)
def test_analyse_idle_bridge_capital_classifies_remaining(
    usdc_arbitrum: AssetIdentifier,
    cctp_pair: TradingPairIdentifier,
    satellite_pair: TradingPairIdentifier,
):
    """The end-of-run diagnostics classify why each bridge position kept capital.

    Covers the ``below_min_sweep`` (dust-buffer) and ``reserved_for_async_deposit``
    (capital committed to an unsettled async deposit) branches that the sweep and
    disabled-flag paths do not exercise, and pins that a plain positive
    ``bridge_capital_allocated`` (the normal deployed state of any synchronous
    satellite position) is NOT misreported as an async-deposit reservation.

    1. A position holding capital between one raw unit and the sweep threshold is
       classified ``below_min_sweep``.
    2. The same sub-threshold residue with capital allocated to a synchronous
       satellite position is still ``below_min_sweep``, not
       ``reserved_for_async_deposit``.
    3. With an actual unsettled async vault deposit on the chain
       (``vault_settlement_pending`` buy), the residue is classified
       ``reserved_for_async_deposit``.
    """
    # 1. Dust buffer: 0.5 idle with a 1.0 threshold -> below_min_sweep.
    wallet = SimulatedWallet()
    wallet.set_balance(usdc_arbitrum, Decimal(1_000))
    state = _make_state(usdc_arbitrum, Decimal(1_000))
    execution = BacktestExecution(wallet=wallet)
    _establish_idle_satellite_capital(state, execution, wallet, cctp_pair, usdc_arbitrum, Decimal("0.5"))
    idle_df = analyse_idle_bridge_capital(state, bridge_sweep_min_usd=1.0, sweep_enabled=True)
    assert list(idle_df["why_not_swept"]) == ["below_min_sweep"]

    # 2. Synchronous deployment: 20_000 bridged, 19_999.5 held by a sync satellite
    #    position (simulated via the allocation ledger), 0.5 idle residue -> still
    #    below_min_sweep, because no deposit is pending settlement.
    wallet = SimulatedWallet()
    wallet.set_balance(usdc_arbitrum, Decimal(50_000))
    state = _make_state(usdc_arbitrum, Decimal(50_000))
    execution = BacktestExecution(wallet=wallet)
    _establish_idle_satellite_capital(state, execution, wallet, cctp_pair, usdc_arbitrum, Decimal(20_000))
    bridge_pos = state.portfolio.get_bridge_position_for_chain(SATELLITE_CHAIN_ID)
    bridge_pos.bridge_capital_allocated = Decimal("19999.5")
    idle_df = analyse_idle_bridge_capital(state, bridge_sweep_min_usd=1.0, sweep_enabled=True)
    assert list(idle_df["why_not_swept"]) == ["below_min_sweep"]

    # 3. Same residue but with a genuine unsettled async deposit on the chain ->
    #    reserved_for_async_deposit.
    _, deposit, _ = state.create_trade(
        strategy_cycle_at=TS,
        pair=satellite_pair,
        quantity=None,
        reserve=Decimal(1_000),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_arbitrum,
        reserve_currency_price=1.0,
    )
    # Mark the deposit as awaiting async settlement; the status derives from
    # this timestamp field, so no execution machinery is needed for the check.
    deposit.vault_settlement_pending_at = TS
    idle_df = analyse_idle_bridge_capital(state, bridge_sweep_min_usd=1.0, sweep_enabled=True)
    assert list(idle_df["why_not_swept"]) == ["reserved_for_async_deposit"]
