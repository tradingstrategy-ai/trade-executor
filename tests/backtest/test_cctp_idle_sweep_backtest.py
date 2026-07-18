"""Integration coverage for the idle CCTP bridge capital sweep (issue #1562).

Proves the money path end to end through the real ``BacktestExecution`` bridge
simulation: settled satellite bridge capital with no directional demand is swept
back to the primary hub reserve, where the existing hub-side ``YieldManager`` sweep
(covered by ``tests/backtest/test_phase_aware_backtest.py``) then parks it in the
queue vault. This test owns the first half of that path — idle satellite capital ->
hub reserve — and the end-of-run diagnostics.

Why not ``run_backtest_inline``: no existing backtest harness runs a cross-chain
universe through the runner, and doing so requires the generic-router plus
per-chain pricing/settlement machinery, which is out of proportion to the
behaviour under test (the planner sweep and its landing in the hub reserve). The
full-run acceptance is instead exercised by the getting-started NB15 notebook.
The planner's quiet-cycle sweep with an empty trade list is unit-covered by
``tests/ethereum/test_cctp_idle_bridge_sweep.py``; the runner gate that calls the
planner on a quiet cycle threads the same flag through
``StrategyParameters.get("sweep_idle_bridge_capital", True)``.
"""

import datetime
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from tradeexecutor.analysis.cctp import analyse_idle_bridge_capital
from tradeexecutor.backtest.backtest_execution import BacktestExecution
from tradeexecutor.backtest.backtest_routing import BacktestRoutingIgnoredModel, BacktestRoutingState
from tradeexecutor.backtest.simulated_wallet import SimulatedWallet
from tradeexecutor.ethereum.cctp.planner import inject_cctp_bridge_trades
from tradeexecutor.state.identifier import (
    AssetIdentifier,
    TradingPairIdentifier,
    TradingPairKind,
)
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeType

USDC_ARBITRUM_ADDRESS = "0xaf88d065e77c8cc2239327c5edb3a432268e5831"
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


def _execute(state, execution, wallet, usdc_arbitrum, trades):
    """Sort by execution order and run the trades through the backtest executor."""
    routing_model, routing_state = _routing(wallet, usdc_arbitrum)
    execution.execute_trades(
        ts=TS,
        state=state,
        trades=sorted(trades, key=lambda t: t.get_execution_sort_position()),
        routing_model=routing_model,
        routing_state=routing_state,
        check_balances=False,
    )


def _establish_idle_satellite_capital(state, execution, wallet, cctp_pair, usdc_arbitrum, amount):
    """Bridge ``amount`` to the satellite and settle it, leaving it idle there."""
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
    _execute(state, execution, wallet, usdc_arbitrum, [bridge_out])
    assert bridge_out.is_success()


@pytest.mark.timeout(300)
def test_quiet_cycle_sweeps_idle_capital_back_to_hub(
    usdc_arbitrum: AssetIdentifier,
    cctp_pair: TradingPairIdentifier,
):
    """A quiet cycle sweeps settled satellite capital back to the hub reserve.

    This is the issue #1562 fix on the real execution path: idle bridge capital
    that no later allocation demands is returned to the hub, where the queue-vault
    sweep (tested elsewhere) can park it. With no queue vault the recovered cash
    simply sits safely as hub reserve — the issue's "absent queue vault" case.

    1. Seed 100_000 hub reserve and bridge 40_000 to the satellite, settling it so
       the satellite holds 40_000 idle capital and the hub reserve is 60_000.
    2. Run a quiet cycle (empty trade list) with the sweep enabled and execute the
       injected trades through the backtest bridge simulation.
    3. Assert the hub reserve is made whole (100_000), no bridge position retains
       capital above the dust buffer, and total equity is conserved throughout.
    4. Assert the end-of-run diagnostics report nothing left unswept.
    """
    # 1. Establish 40_000 idle on the satellite; hub reserve falls to 60_000.
    wallet = SimulatedWallet()
    wallet.set_balance(usdc_arbitrum, Decimal(100_000))
    state = _make_state(usdc_arbitrum, Decimal(100_000))
    execution = BacktestExecution(wallet=wallet)
    equity_start = state.portfolio.calculate_total_equity()
    _establish_idle_satellite_capital(state, execution, wallet, cctp_pair, usdc_arbitrum, Decimal(40_000))
    assert state.portfolio.get_default_reserve_position().quantity == Decimal(60_000)
    assert state.portfolio.calculate_total_equity() == pytest.approx(equity_start)

    # 2. Quiet cycle: no directional trades, sweep enabled; execute the result.
    universe = MagicMock()
    universe.iterate_pairs.return_value = [cctp_pair]
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
    assert bridge_backs[0].other_data["cctp_planning_amounts"] == {"idle_sweep": "40000"}
    _execute(state, execution, wallet, usdc_arbitrum, result)

    # 3. Hub reserve is whole again, no idle capital remains, equity conserved.
    assert state.portfolio.get_default_reserve_position().quantity == pytest.approx(Decimal(100_000))
    assert state.portfolio.calculate_total_equity() == pytest.approx(equity_start)
    open_bridge = state.portfolio.get_bridge_position_for_chain(SATELLITE_CHAIN_ID)
    assert open_bridge is None or open_bridge.get_available_bridge_capital() < Decimal(1)

    # 4. Diagnostics report nothing left unswept.
    idle_df = analyse_idle_bridge_capital(state, bridge_sweep_min_usd=1.0, sweep_enabled=True)
    assert (idle_df["why_not_swept"] != "not_swept").all()
    assert "sweep_disabled" not in set(idle_df["why_not_swept"])


@pytest.mark.timeout(300)
def test_full_balance_sweep_tolerates_state_wallet_drift(
    usdc_arbitrum: AssetIdentifier,
    usdc_base: AssetIdentifier,
    cctp_pair: TradingPairIdentifier,
):
    """A full-balance sweep survives sub-raw-unit state/wallet rounding drift.

    Over a long backtest the state's tracked bridge quantity and the simulated
    wallet balance can drift by a fraction of a raw unit from accumulated Decimal
    rounding. A sweep bridges the chain's entire settled balance, landing exactly
    on that boundary; the burn must tolerate the dust instead of crashing with
    ``OutOfSimulatedBalance`` (the failure this reproduces from getting-started
    NB15).

    1. Establish 40_000 idle satellite capital.
    2. Nudge the simulated wallet's satellite USDC a fraction of a raw unit below
       the state quantity, simulating accumulated rounding drift.
    3. Run the quiet-cycle sweep and execute it.
    4. Assert the burn does not crash, the bridge position clears, and the hub
       reserve is restored within rounding dust.
    """
    # 1. Establish 40_000 idle on the satellite.
    wallet = SimulatedWallet()
    wallet.set_balance(usdc_arbitrum, Decimal(100_000))
    state = _make_state(usdc_arbitrum, Decimal(100_000))
    execution = BacktestExecution(wallet=wallet)
    _establish_idle_satellite_capital(state, execution, wallet, cctp_pair, usdc_arbitrum, Decimal(40_000))

    # 2. Nudge the wallet's satellite USDC 2e-7 below the state quantity.
    wallet.update_balance(usdc_base, Decimal("-0.0000002"), "simulate rounding drift")

    # 3. Quiet-cycle sweep and execute.
    universe = MagicMock()
    universe.iterate_pairs.return_value = [cctp_pair]
    result = inject_cctp_bridge_trades(
        state=state,
        trades=[],
        strategy_universe=universe,
        primary_chain_id=PRIMARY_CHAIN_ID,
        ts=TS,
        reserve_asset=usdc_arbitrum,
    )
    _execute(state, execution, wallet, usdc_arbitrum, result)

    # 4. No crash; the bridge position cleared and the hub reserve is restored.
    open_bridge = state.portfolio.get_bridge_position_for_chain(SATELLITE_CHAIN_ID)
    assert open_bridge is None or open_bridge.get_available_bridge_capital() < Decimal(1)
    assert state.portfolio.get_default_reserve_position().quantity == pytest.approx(Decimal(100_000), abs=Decimal("0.001"))


@pytest.mark.timeout(300)
def test_sweep_disabled_leaves_idle_capital(
    usdc_arbitrum: AssetIdentifier,
    cctp_pair: TradingPairIdentifier,
):
    """With the sweep disabled, idle capital stays put and diagnostics flag it.

    This reproduces the pre-fix behaviour (issue #1562) and proves the feature is
    configurable off for strategies that deliberately keep satellite-chain cash.

    1. Establish 40_000 idle satellite capital as before.
    2. Run a quiet cycle with ``sweep_idle_bridge_capital=False``.
    3. Assert no bridge trade is injected and the capital remains on the satellite.
    4. Assert the diagnostics classify the remaining capital ``sweep_disabled``.
    """
    # 1. Establish 40_000 idle on the satellite.
    wallet = SimulatedWallet()
    wallet.set_balance(usdc_arbitrum, Decimal(100_000))
    state = _make_state(usdc_arbitrum, Decimal(100_000))
    execution = BacktestExecution(wallet=wallet)
    _establish_idle_satellite_capital(state, execution, wallet, cctp_pair, usdc_arbitrum, Decimal(40_000))

    # 2. Quiet cycle with the sweep disabled.
    universe = MagicMock()
    universe.iterate_pairs.return_value = [cctp_pair]
    result = inject_cctp_bridge_trades(
        state=state,
        trades=[],
        strategy_universe=universe,
        primary_chain_id=PRIMARY_CHAIN_ID,
        ts=TS,
        reserve_asset=usdc_arbitrum,
        sweep_idle_bridge_capital=False,
    )

    # 3. No bridge trade; capital remains idle on the satellite.
    assert [t for t in result if t.pair.is_cctp_bridge()] == []
    bridge_pos = state.portfolio.get_bridge_position_for_chain(SATELLITE_CHAIN_ID)
    assert bridge_pos.get_available_bridge_capital() == Decimal(40_000)

    # 4. Diagnostics classify the leak as sweep_disabled.
    idle_df = analyse_idle_bridge_capital(state, bridge_sweep_min_usd=1.0, sweep_enabled=False)
    assert list(idle_df["why_not_swept"]) == ["sweep_disabled"]
    assert idle_df.iloc[0]["available_idle"] == pytest.approx(40_000.0)
