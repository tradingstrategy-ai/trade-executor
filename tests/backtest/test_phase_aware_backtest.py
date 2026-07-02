"""Synthetic, no-network integration backtest for PhaseAwareAlphaModel (cleanup CU-3).

Drives the full park -> deposit-on-open + yield-sweep pipeline through a real
``run_backtest_inline`` over a synthetic single-chain universe with two synchronous vaults:

- a **window-gated target vault** (``VT``) whose deposit window is closed for the first few cycles
  then opens, via ``vault_window_overrides`` gating ``can_deposit`` (no async settlement delay);
- an always-open **synchronous queue venue** (``VQ``), wired as a single-slot ``YieldRuleset`` so
  idle cash sweeps into it while the target is closed and releases to fund the deposit on open.

One module-scoped backtest run feeds several small independent assertions (per the cleanup plan's
"land the obligations as small tests, not one all-or-nothing package"): the window cycle
(park -> promote -> executed deposit), the idle-cash floor after the sweep, and the invariants
reassigned here from CU-1 (venue never traded by the model / stays in equity / does not inflate the
buy budget). Method-agnostic (park -> promote under the simple vs waterfall normaliser) is a second
run. Per-cycle invariant evidence is collected in ``_OBSERVATIONS`` because the venue is a transient
holding (swept while the target is closed, released to fund the deposit on open), so the final state
alone cannot show it held idle cash.

Reproduces the synthetic-vault machinery of ``tests/backtest/test_backtest_async_vault.py`` inline
(tests cannot import from the ``tests`` tree); the helpers it uses live in ``tradeexecutor.testing``.
"""
import datetime

import pandas as pd
import pytest

from tradeexecutor.backtest.backtest_runner import run_backtest_inline
from tradeexecutor.backtest.vault_windows import VaultWindowSchedule
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.chart.definition import ChartInput
from tradeexecutor.strategy.chart.standard.weight import QUEUE_VENUE_BAND, equity_curve_by_chain
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionMode, unit_test_execution_context
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.pandas_trader.yield_manager import (
    YieldDecisionInput,
    YieldManager,
    YieldRuleset,
    YieldWeightingRule,
)
from tradeexecutor.strategy.phase_aware import (
    EVENT_PARK,
    EVENT_PROMOTE,
    PhaseAwareAlphaModel,
    is_queue_vault_position,
    iter_all_events,
    queue_vault_pair_ids,
    queue_venue_redeemable,
)
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, create_pair_universe_from_code
from tradeexecutor.strategy.tvl_size_risk import USDTVLSizeRiskModel
from tradeexecutor.strategy.weighting import weight_passthrouh
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange, generate_simple_routing_model
from tradeexecutor.testing.synthetic_price_data import generate_fixed_price_candles, generate_tvl_candles
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.liquidity import GroupedLiquidityUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe


START_AT = datetime.datetime(2024, 1, 1)
END_AT = datetime.datetime(2024, 1, 15)  # ~14 daily cycles
INITIAL_DEPOSIT = 100_000
FIXED_PRICE = 100.0
ALLOCATION = 0.95  # directional + yield target; 5% stays as the always-in-cash reserve

TARGET_INTERNAL_ID = 600  # window-gated deposit target (VT)
VENUE_INTERNAL_ID = 601  # always-open synchronous queue venue (VQ)
DIRECTIONAL_INTERNAL_ID = 602  # always-open directional vault (VD), used only by the gate-survival test

# Closed 1-day window every 5 days, anchored so backtest start is in the closed phase. With the
# engine's 1-based input.cycle: the target is closed on cycles 1-4, opens on cycle 5, closes again,
# reopens on cycle 10 (retry headroom).
WINDOW = VaultWindowSchedule(
    cadence=datetime.timedelta(days=5),
    open_duration=datetime.timedelta(days=1),
    anchor=START_AT - datetime.timedelta(days=1),
)

#: Per-cycle invariant evidence recorded by decide_trades during the observed run.
_OBSERVATIONS: list[dict] = []

#: Whether the size-risk (waterfall) normaliser actually ran, recorded per cycle during the
#: method-agnostic run. ``size_risk`` on a signal is set only by the size-risk normalisers, never by
#: ``_normalise_weights_simple`` - so it is a regression-proof witness that waterfall was exercised.
_SIZE_RISK_RAN: list[bool] = []


class _Parameters:
    cycle_duration = CycleDuration.cycle_1d
    initial_cash = INITIAL_DEPOSIT
    allocation = ALLOCATION


def _create_indicators(parameters, indicators, strategy_universe, execution_context) -> None:
    """No indicators - the target signal is forced in decide_trades."""


def _make_universe() -> tuple[TradingStrategyUniverse, TradingPairIdentifier, TradingPairIdentifier]:
    """Build a synthetic universe with two synchronous vault pairs and a window override on the target."""
    chain_id = ChainId.ethereum
    exchange = generate_exchange(exchange_id=1, chain_id=chain_id, address=generate_random_ethereum_address())
    reserve_asset = AssetIdentifier(chain_id.value, generate_random_ethereum_address(), "USDC", 6, 1)

    pairs: list[TradingPairIdentifier] = []
    # VT window-gated target, VQ queue venue, VD an always-open directional vault (only the
    # gate-survival test signals VD; it stays inert - no position - in the other runs).
    for idx, symbol in enumerate(("VT", "VQ", "VD")):
        share = AssetIdentifier(chain_id.value, generate_random_ethereum_address(), symbol, 18, 10 + idx)
        pair = TradingPairIdentifier(
            share,
            reserve_asset,
            generate_random_ethereum_address(),
            exchange.address,
            internal_id=600 + idx,
            internal_exchange_id=exchange.exchange_id,
            fee=0,
            kind=TradingPairKind.vault,  # is_vault() True; features=None -> synchronous (is_async_vault() False)
        )
        pair.other_data["vault_protocol"] = "test_sync_vault"
        pairs.append(pair)

    pair_universe = create_pair_universe_from_code(chain_id, pairs)
    candles = generate_fixed_price_candles(TimeBucket.d1, START_AT, END_AT, {p: FIXED_PRICE for p in pairs})
    # Deep, flat TVL per vault so the size-risk (waterfall) normaliser has data to run but its cap
    # never binds (100M pool vs a 95k deposit), i.e. deposits still succeed. Only the method-agnostic
    # run consults it; the simple-normaliser runs ignore liquidity.
    liquidity_df = pd.concat([
        generate_tvl_candles(
            TimeBucket.d1, START_AT, END_AT,
            start_liquidity=100_000_000,
            pair_id=p.internal_id,
            daily_drift=(1.0, 1.0), high_drift=1.0, low_drift=1.0, random_seed=1,
        )
        for p in pairs
    ])
    universe = Universe(
        time_bucket=TimeBucket.d1,
        chains={chain_id},
        exchanges={exchange},
        pairs=pair_universe,
        candles=GroupedCandleUniverse(candles),
        liquidity=GroupedLiquidityUniverse(liquidity_df),
    )
    strategy_universe = TradingStrategyUniverse(data_universe=universe, reserve_assets=[reserve_asset])
    strategy_universe.data_universe.pairs.exchange_universe = strategy_universe.data_universe.exchange_universe

    target_pair, venue_pair = pairs[0], pairs[1]
    # Gate only the target vault's deposit window; the venue and VD are always open.
    strategy_universe.vault_window_overrides = {target_pair.internal_id: WINDOW}
    return strategy_universe, target_pair, venue_pair


def _make_yield_rules(venue_pair: TradingPairIdentifier) -> YieldRuleset:
    return YieldRuleset(
        position_allocation=ALLOCATION,
        buffer_pct=0.01,
        cash_change_tolerance_usd=5.0,
        weights=[YieldWeightingRule(pair=venue_pair, max_concentration=1.0)],
    )


def _phase_aware_decide_trades(
    input: StrategyInput,
    *,
    size_risk: bool = False,
    observe: bool = False,
    signal_scores: dict[int, float] | None = None,
) -> list[TradeExecution]:
    """Force deposits into the signalled vaults and run the phase-aware + YieldManager two-step."""
    state = input.state
    timestamp = input.timestamp
    position_manager = input.get_position_manager()
    strategy_universe = input.strategy_universe

    venue_pair = strategy_universe.get_pair_by_id(VENUE_INTERNAL_ID)
    yield_rules = _make_yield_rules(venue_pair)
    venue_pair_ids = queue_vault_pair_ids(yield_rules)

    # Venue value at the start of the cycle (before this cycle's trades) - inv-3 evidence.
    venue_value_before = queue_venue_redeemable(state.portfolio, venue_pair_ids)

    alpha_model = PhaseAwareAlphaModel(timestamp, cycle=input.cycle, venue_pair_ids=venue_pair_ids)
    # Default: only the window-gated target. The venue is never signalled -> excluded from candidates.
    for pair_id, score in (signal_scores or {TARGET_INTERNAL_ID: 0.999}).items():
        alpha_model.set_signal(strategy_universe.get_pair_by_id(pair_id), score)

    locked = alpha_model.carry_forward_non_redeemable_positions(position_manager)
    deployable = max(state.portfolio.get_total_equity() * ALLOCATION - locked, 0.0)

    alpha_model.select_top_signals(count=5)
    alpha_model.assign_weights(method=weight_passthrouh)
    if size_risk:
        # Real size-risk + waterfall normaliser: normalise_weights dispatches to
        # _normalise_weights_waterfall only when a size_risk_model is given (waterfall=True alone
        # falls back to the simple normaliser). A deep pool means the cap does not bind.
        size_risker = USDTVLSizeRiskModel(input.pricing_model, per_position_cap=0.99)
        alpha_model.normalise_weights(
            investable_equity=deployable,
            size_risk_model=size_risker,
            max_weight=1.0,
            max_positions=5,
            waterfall=True,
        )
        # Witness that the size-risk path actually ran: the size-risk normalisers set
        # accepted_investable_equity, the simple normaliser never does - so this catches a future
        # regression back to the simple path (waterfall=True alone silently falling back).
        _SIZE_RISK_RAN.append(alpha_model.accepted_investable_equity is not None)
    else:
        alpha_model.normalise_weights(investable_equity=deployable, max_weight=1.0)
    alpha_model.update_old_weights(state.portfolio, ignore_credit=False)  # venue excluded (inv-2)
    # The simple normaliser sets normalised_weight but not position_target, so pass investable_equity.
    alpha_model.calculate_target_positions(position_manager, investable_equity=deployable)

    alpha_model.apply_phase_aware_intent(position_manager)  # park closed-window deposit before generation

    trades = alpha_model.generate_rebalance_trades_and_triggers(
        position_manager,
        min_trade_threshold=1.0,
        individual_rebalance_min_threshold=1.0,
        sell_rebalance_min_threshold=1.0,
        execution_context=input.execution_context,
    )
    alpha_model.reconcile_phase_aware_events(position_manager, trades)  # finalise emitted promotes

    if observe:
        _OBSERVATIONS.append({
            "cycle": input.cycle,
            # inv-1: the alpha model's directional trades must never target the venue.
            "model_traded_venue": any(t.pair.internal_id == VENUE_INTERNAL_ID for t in trades),
            # inv-3b: directional buys this cycle must not exceed the deployable investable equity.
            "directional_buys": sum(max(s.position_adjust_usd, 0.0) for s in alpha_model.signals.values()),
            "deployable": deployable,
            # inv-3: the venue value is part of equity while the target is closed.
            "venue_value_before": venue_value_before,
        })

    yield_manager = YieldManager(position_manager=position_manager, rules=yield_rules)
    yield_input = YieldDecisionInput(
        execution_mode=input.execution_context.mode,
        cycle=input.cycle,
        timestamp=timestamp,
        total_equity=state.portfolio.get_total_equity(),
        directional_trades=trades,
        pending_redemptions=position_manager.get_pending_redemptions(),
    )
    trades += yield_manager.calculate_yield_management_safe(yield_input).trades
    return trades


def _run(decide) -> State:
    strategy_universe, _, _ = _make_universe()
    routing_model = generate_simple_routing_model(strategy_universe)
    result = run_backtest_inline(
        start_at=START_AT,
        end_at=END_AT,
        client=None,
        decide_trades=decide,
        create_indicators=_create_indicators,
        universe=strategy_universe,
        cycle_duration=CycleDuration.cycle_1d,
        initial_deposit=INITIAL_DEPOSIT,
        reserve_currency=ReserveCurrency.usdc,
        trade_routing=TradeRouting.user_supplied_routing_model,
        routing_model=routing_model,
        engine_version="0.5",
        parameters=_Parameters,
        mode=ExecutionMode.unit_testing,
        allow_missing_fees=True,
        name="phase-aware-backtest",
    )
    return result.state


@pytest.fixture(scope="module")
def primary_run() -> tuple[State, list[dict]]:
    """Run the window-cycle backtest once (simple normaliser); reused by the small assertions."""
    _OBSERVATIONS.clear()
    state = _run(lambda i: _phase_aware_decide_trades(i, observe=True))
    return state, list(_OBSERVATIONS)


def test_window_cycle_park_then_promote(primary_run: tuple[State, list[dict]]):
    """The window-gated vault parks while closed then promotes exactly once when the window opens.

    1. The durable event log holds a park and a promote for the same target vault.
    2. There is exactly one promote (a single deposit-on-open, not repeated).
    """
    state, _ = primary_run
    events = list(iter_all_events(state.other_data))
    parked = {e.vault_internal_id for e in events if e.kind == EVENT_PARK}
    promoted = {e.vault_internal_id for e in events if e.kind == EVENT_PROMOTE}

    # 1. Same vault parked and promoted.
    assert TARGET_INTERNAL_ID in parked
    assert TARGET_INTERNAL_ID in (parked & promoted)
    # 2. Exactly one promote for the target.
    assert sum(1 for e in events if e.kind == EVENT_PROMOTE and e.vault_internal_id == TARGET_INTERNAL_ID) == 1


def test_executed_deposit_holds_value(primary_run: tuple[State, list[dict]]):
    """The promoted deposit actually executes: the target vault ends with a realised holding.

    1. The target vault has an open position with value > 0 (~95% of equity) at run end.
    """
    state, _ = primary_run
    held = {p.pair.internal_id: p.get_value() for p in state.portfolio.open_positions.values()}
    # 1. The target vault holds a realised position from the deposit-on-open.
    assert held.get(TARGET_INTERNAL_ID, 0.0) > 0


def test_idle_cash_returns_to_reserve_floor(primary_run: tuple[State, list[dict]]):
    """After the deposit, idle cash sits at the always-in-cash reserve floor, not leaking.

    1. Final idle cash is ~ (1 - allocation) of equity (the 5% reserve), i.e. the capital was
       deployed rather than left idle.
    """
    state, _ = primary_run
    equity = state.portfolio.get_total_equity()
    idle = state.portfolio.get_cash()
    # 1. Idle cash at the reserve floor.
    assert idle / equity == pytest.approx(1 - ALLOCATION, abs=0.03)


def test_inv1_model_never_trades_the_venue(primary_run: tuple[State, list[dict]]):
    """Invariant 1: every venue trade comes from YieldManager; the alpha model emits none.

    1. On no cycle did the alpha model's directional trades target the queue venue.
    """
    _, observations = primary_run
    assert observations, "expected per-cycle observations from the run"
    # 1. The model never traded the venue on any cycle.
    assert not any(o["model_traded_venue"] for o in observations)


def test_inv3_venue_held_in_equity_while_closed(primary_run: tuple[State, list[dict]]):
    """Invariant 3: the venue stays inside equity - it holds swept idle cash while the target is closed.

    1. On at least one cycle the queue venue held a positive redeemable balance (idle cash swept in,
       part of total equity, available to fund the later deposit).
    """
    _, observations = primary_run
    # 1. The venue held value at some point (proving the sweep, and that the venue is in equity).
    assert any(o["venue_value_before"] > 0 for o in observations)


def test_inv3b_directional_buys_within_investable_equity(primary_run: tuple[State, list[dict]]):
    """Invariant 3b: directional buys never exceed the deployable investable equity.

    1. Every cycle's directional buys are <= that cycle's deployable target - the venue, in equity
       but excluded from old-weights, does not inflate the buy budget.
    """
    _, observations = primary_run
    # 1. No cycle over-allocates beyond investable equity.
    assert all(o["directional_buys"] <= o["deployable"] + 1.0 for o in observations)


def test_method_agnostic_park_promote_under_size_risk_waterfall():
    """The park -> deposit-on-open fires under the size-risk waterfall normaliser, not just the simple one.

    The phase-aware pass operates on self.signals after normalisation, so it is allocation-method
    agnostic. This run uses a real ``USDTVLSizeRiskModel`` so ``normalise_weights`` dispatches to
    ``_normalise_weights_waterfall`` (waterfall=True alone falls back to the simple normaliser), and
    must still park the closed-window target and promote it on open - the slot-holding property under
    a genuinely different normaliser code path, not asserted-by-construction.

    1. Run the scenario with a size-risk model so the waterfall normaliser genuinely runs.
    2. Confirm the size-risk (waterfall) path actually executed (guards against a silent fallback to
       the simple normaliser).
    3. The target vault is still parked and promoted, and ends holding a realised position.
    """
    _SIZE_RISK_RAN.clear()
    # 1. Run under the real size-risk waterfall normaliser.
    state = _run(lambda i: _phase_aware_decide_trades(i, size_risk=True, observe=False))

    # 2. The size-risk normaliser genuinely ran (size_risk set on the target signal - simple never does).
    assert _SIZE_RISK_RAN and all(_SIZE_RISK_RAN), _SIZE_RISK_RAN

    # 3. Same park -> promote -> held outcome as the simple normaliser.
    events = list(iter_all_events(state.other_data))
    parked = {e.vault_internal_id for e in events if e.kind == EVENT_PARK}
    promoted = {e.vault_internal_id for e in events if e.kind == EVENT_PROMOTE}
    held = {p.pair.internal_id: p.get_value() for p in state.portfolio.open_positions.values()}
    assert TARGET_INTERNAL_ID in (parked & promoted)
    assert held.get(TARGET_INTERNAL_ID, 0.0) > 0


def test_gate_survival_other_deposit_survives_a_park():
    """Parking the window-gated deposit does not cancel a co-occurring valid deposit (gate-survival happy path).

    The bad path (parking the only above-threshold trade cancels the cycle) is a unit test in CU-2;
    this is the happy path, which needs real trade generation: while the window-gated VT is parked on
    the closed cycles, an always-open directional vault VD is deposited normally - the parked VT's
    zeroed adjust does not lower the min-trade gate's max_diff below the threshold for VD.

    1. Signal both the window-gated VT (dominant) and an always-open VD.
    2. VD ends holding a realised deposit (its trades survived the cycles where VT was parked).
    3. VT itself still parks then promotes.
    """
    # 1. Both vaults signalled; VT is window-gated, VD is always open.
    state = _run(lambda i: _phase_aware_decide_trades(
        i, signal_scores={TARGET_INTERNAL_ID: 0.999, DIRECTIONAL_INTERNAL_ID: 0.9},
    ))

    held = {p.pair.internal_id: p.get_value() for p in state.portfolio.open_positions.values()}
    events = list(iter_all_events(state.other_data))
    parked = {e.vault_internal_id for e in events if e.kind == EVENT_PARK}
    promoted = {e.vault_internal_id for e in events if e.kind == EVENT_PROMOTE}

    # 2. VD deposited despite VT being parked on the closed cycles (the cycle was not cancelled).
    assert held.get(DIRECTIONAL_INTERNAL_ID, 0.0) > 0
    # 3. VT still parked then promoted.
    assert TARGET_INTERNAL_ID in (parked & promoted)


def test_queue_venue_identified_and_split_into_own_chart_band(primary_run: tuple[State, list[dict]]):
    """The queue venue is identified from state alone and split into its own chart band (CU-4 piece 1).

    Charts have no YieldRuleset config and the venue position tag has no writer, so venue identity
    comes from YieldManager's durable ``yield_decision`` trade marker via ``is_queue_vault_position``.

    1. is_queue_vault_position identifies the venue (VQ) and not the window-gated target (VT).
    2. equity_curve_by_chain renders the venue as a distinct band, split from the chain bands.
    """
    state, _ = primary_run
    # 1. Venue identified from state; the target vault is not a venue.
    venue_ids = {p.pair.internal_id for p in state.portfolio.get_all_positions() if is_queue_vault_position(p)}
    assert VENUE_INTERNAL_ID in venue_ids
    assert TARGET_INTERNAL_ID not in venue_ids

    # 2. The equity-by-chain chart splits the venue into its own band.
    _fig, df = equity_curve_by_chain(ChartInput(execution_context=unit_test_execution_context, state=state))
    assert QUEUE_VENUE_BAND in df.columns
