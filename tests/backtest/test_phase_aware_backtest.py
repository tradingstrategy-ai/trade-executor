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
buy budget). Method-agnostic (park -> promote under the simple vs the real size-risk normalisers) is
a second run. Per-cycle invariant evidence is collected in an observations list because the venue is
a transient holding (swept while the target is closed, released to fund the deposit on open), so the
final state alone cannot show it held idle cash.

A second module-scoped run (cleanup CU-6) adds an **async vault** (``VA``, simulated async via a
backtest settlement-delay override) so the two async-dependent behaviours run end-to-end rather than
only as unit tests: the invariant-4 widening through ``_cap_buys_by_async_sell_proceeds`` when a
promotion coincides with a same-cycle async sell (the parent plan's "engineer a coincident async
sell ... not vacuous"), and the redemption-side sweep (settled async-redeem proceeds land in the
queue venue, not idle).

A third run (cleanup CU-7) exercises the redemption-locked diagnostics: the same window schedule
gates ``check_redemption``, so a held VT is locked while its window is closed and an exit signalled
into a closed window waits until the reopen - asserting the durable redeem-block / redeem-clear
event sequence and the negative ``pending_trigger_queue`` band.

Reproduces the synthetic-vault machinery of ``tests/backtest/test_backtest_async_vault.py`` inline
(tests cannot import from the ``tests`` tree); the helpers it uses live in ``tradeexecutor.testing``.
"""
import datetime
from typing import Callable

import pandas as pd
import pytest

from tradeexecutor.backtest.backtest_runner import run_backtest_inline
from tradeexecutor.backtest.vault_windows import VaultWindowSchedule
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.alpha_model import TradingPairSignalFlags
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
from tradeexecutor.strategy.chart.standard.vault import pending_trigger_queue
from tradeexecutor.strategy.phase_aware import (
    EVENT_PARK,
    EVENT_PROMOTE,
    EVENT_REDEEM_BLOCK,
    EVENT_REDEEM_CLEAR,
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
DATA_END_AT = datetime.datetime(2024, 2, 1)  # synthetic candles/TVL beyond the longest run
INITIAL_DEPOSIT = 100_000
FIXED_PRICE = 100.0
ALLOCATION = 0.95  # directional + yield target; 5% stays as the always-in-cash reserve

TARGET_INTERNAL_ID = 600  # window-gated deposit target (VT)
VENUE_INTERNAL_ID = 601  # always-open synchronous queue venue (VQ)
DIRECTIONAL_INTERNAL_ID = 602  # always-open directional vault (VD), used only by the gate-survival test
ASYNC_INTERNAL_ID = 603  # async vault (VA), used only by the async run (settlement-delay override)

#: Backtest settlement delay that makes VA behave as an async vault (deposit and redemption both
#: *request* then settle two days later). VA carries no async feature flags - asyncness comes purely
#: from the ``vault_settlement_delay_overrides`` execution path, which is also what forces the cap's
#: ``position.has_async_vault_flow()`` fallback (rather than ``pair.is_async_vault()``) to classify it.
ASYNC_SETTLEMENT_DELAY = datetime.timedelta(days=2)

#: Cap the target vault's weight in the async run so the settled async-redeem proceeds have no
#: directional home and must be swept into the queue venue (asserting the sweep, not a re-buy).
ASYNC_RUN_MAX_WEIGHT = 0.6

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

#: Whether a size-risk normaliser actually ran, recorded per cycle during the method-agnostic runs.
#: ``accepted_investable_equity`` is set by all three size-risk normalisers (waterfall, size-risk,
#: size-risk-positions), never by ``_normalise_weights_simple`` - so it is a regression-proof
#: witness that a size-risk path was exercised rather than the silent simple fallback.
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
    # gate-survival test signals VD), VA an async vault (only the async run signals VA; asyncness
    # comes from the settlement-delay override in _run, not from feature flags). Unsignalled vaults
    # stay inert - no position - in the runs that do not use them.
    for idx, symbol in enumerate(("VT", "VQ", "VD", "VA")):
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
    candles = generate_fixed_price_candles(TimeBucket.d1, START_AT, DATA_END_AT, {p: FIXED_PRICE for p in pairs})
    # Deep, flat TVL per vault so the size-risk (waterfall) normaliser has data to run but its cap
    # never binds (100M pool vs a 95k deposit), i.e. deposits still succeed. Only the method-agnostic
    # run consults it; the simple-normaliser runs ignore liquidity.
    liquidity_df = pd.concat([
        generate_tvl_candles(
            TimeBucket.d1, START_AT, DATA_END_AT,
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
    waterfall: bool = True,
    signal_scores: dict[int, float] | None = None,
    signal_fn: Callable[[int], dict[int, float]] | None = None,
    max_weight: float = 1.0,
    observations: list[dict] | None = None,
) -> list[TradeExecution]:
    """Force deposits into the signalled vaults and run the phase-aware + YieldManager two-step.

    :param signal_fn:
        Optional per-cycle signal source ``(cycle) -> {pair_id: score}``; overrides ``signal_scores``.
        Used by the async run to switch signals on the promote cycle.

    :param max_weight:
        Cap any single signal's normalised weight, so a run can leave deliberate capacity slack
        (undeployed deployable) for the yield sweep to absorb.

    :param observations:
        When given, per-cycle invariant evidence is appended here (after trade generation and the
        phase-aware reconcile, before the yield step).
    """
    state = input.state
    timestamp = input.timestamp
    position_manager = input.get_position_manager()
    strategy_universe = input.strategy_universe

    venue_pair = strategy_universe.get_pair_by_id(VENUE_INTERNAL_ID)
    yield_rules = _make_yield_rules(venue_pair)
    venue_pair_ids = queue_vault_pair_ids(yield_rules)

    # Venue value and raw cash at the start of the cycle (before this cycle's trades) - inv-3 /
    # invariant-4 evidence.
    venue_value_before = queue_venue_redeemable(state.portfolio, venue_pair_ids)
    raw_cash_before = state.portfolio.get_cash()

    alpha_model = PhaseAwareAlphaModel(timestamp, cycle=input.cycle, venue_pair_ids=venue_pair_ids)
    # Default: only the window-gated target. The venue is never signalled -> excluded from candidates.
    if signal_fn is not None:
        scores = signal_fn(input.cycle)
    else:
        scores = signal_scores or {TARGET_INTERNAL_ID: 0.999}
    for pair_id, score in scores.items():
        alpha_model.set_signal(strategy_universe.get_pair_by_id(pair_id), score)

    locked = alpha_model.carry_forward_non_redeemable_positions(position_manager)
    deployable = max(state.portfolio.get_total_equity() * ALLOCATION - locked, 0.0)

    alpha_model.select_top_signals(count=5)
    alpha_model.assign_weights(method=weight_passthrouh)
    if size_risk:
        # Real size-risk normalisers: normalise_weights dispatches to _normalise_weights_waterfall
        # (waterfall=True) or _normalise_weights_size_risk (waterfall=False) only when a
        # size_risk_model is given (waterfall=True alone falls back to the simple normaliser).
        # A deep pool means the cap does not bind.
        size_risker = USDTVLSizeRiskModel(input.pricing_model, per_position_cap=0.99)
        alpha_model.normalise_weights(
            investable_equity=deployable,
            size_risk_model=size_risker,
            max_weight=max_weight,
            max_positions=5,
            waterfall=waterfall,
        )
        # Witness that a size-risk path actually ran: the size-risk normalisers set
        # accepted_investable_equity, the simple normaliser never does - so this catches a future
        # regression back to the simple path (waterfall=True alone silently falling back).
        _SIZE_RISK_RAN.append(alpha_model.accepted_investable_equity is not None)
    else:
        alpha_model.normalise_weights(investable_equity=deployable, max_weight=max_weight)
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

    if observations is not None:
        target_signal = alpha_model.signals.get(TARGET_INTERNAL_ID)
        async_pair = strategy_universe.get_pair_by_id(ASYNC_INTERNAL_ID)
        async_position = position_manager.get_current_position_for_pair(async_pair, pending=True)
        observations.append({
            "cycle": input.cycle,
            # inv-1: the alpha model's directional trades must never target the venue.
            "model_traded_venue": any(t.pair.internal_id == VENUE_INTERNAL_ID for t in trades),
            # inv-3b: directional buys this cycle must not exceed the deployable investable equity.
            "directional_buys": sum(max(s.position_adjust_usd, 0.0) for s in alpha_model.signals.values()),
            "deployable": deployable,
            # inv-3: the venue value is part of equity while the target is closed.
            "venue_value_before": venue_value_before,
            # inv-4 evidence (async run): what funded the promote, and whether the cap engaged.
            "raw_cash_before": raw_cash_before,
            "promote_emitted": target_signal is not None and TradingPairSignalFlags.promoted_from_queue_vault in target_signal.flags,
            "target_flags": {f.value for f in target_signal.flags} if target_signal is not None else set(),
            "target_adjust_after": target_signal.position_adjust_usd if target_signal is not None else 0.0,
            "async_sell_emitted": any(t.pair.internal_id == ASYNC_INTERNAL_ID and t.is_sell() for t in trades),
            # The cap classifies the delay-override VA as async via the position's own settlement
            # history (has_async_vault_flow), not feature flags - record that it actually did.
            "async_position_has_flow": async_position is not None and async_position.has_async_vault_flow(),
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


def _run(decide, *, async_vault: bool = False, end_at: datetime.datetime = END_AT) -> State:
    strategy_universe, _, _ = _make_universe()
    routing_model = generate_simple_routing_model(strategy_universe)
    settlement_overrides = None
    if async_vault:
        # VA settles deposits and redemptions ASYNC_SETTLEMENT_DELAY after the request - the
        # backtest-delay-override way to simulate an async vault with no async feature flags.
        va_pair = strategy_universe.get_pair_by_id(ASYNC_INTERNAL_ID)
        settlement_overrides = {va_pair.pool_address: ASYNC_SETTLEMENT_DELAY}
    result = run_backtest_inline(
        start_at=START_AT,
        end_at=end_at,
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
        vault_settlement_delay_overrides=settlement_overrides,
    )
    return result.state


@pytest.fixture(scope="module")
def primary_run() -> tuple[State, list[dict]]:
    """Run the window-cycle backtest once (simple normaliser); reused by the small assertions."""
    _OBSERVATIONS.clear()
    state = _run(lambda i: _phase_aware_decide_trades(i, observations=_OBSERVATIONS))
    return state, list(_OBSERVATIONS)


def _async_signals(cycle: int) -> dict[int, float]:
    """Per-cycle signals for the async run.

    Closed cycles 1-4: VT (0.6 share) parks while VA (0.4 share) holds a directional position.
    From cycle 5 (VT's window opens): promote VT and simultaneously exit VA - engineering the
    coincident same-cycle async sell the invariant-4 test needs, in the very cycle of the promotion.
    """
    if cycle < 5:
        return {TARGET_INTERNAL_ID: 0.6, ASYNC_INTERNAL_ID: 0.4}
    return {TARGET_INTERNAL_ID: 0.999, ASYNC_INTERNAL_ID: 0.0}


#: Per-cycle evidence recorded during the async run.
_ASYNC_OBSERVATIONS: list[dict] = []


@pytest.fixture(scope="module")
def async_run() -> tuple[State, list[dict]]:
    """Run the async-vault backtest once (cleanup CU-6); reused by the async assertions.

    VA is made async via a settlement-delay override (no async feature flags), VT is weight-capped
    at ``ASYNC_RUN_MAX_WEIGHT`` so the settled VA proceeds have no directional home and must be
    swept into the queue venue rather than re-bought.
    """
    _ASYNC_OBSERVATIONS.clear()
    state = _run(
        lambda i: _phase_aware_decide_trades(
            i,
            signal_fn=_async_signals,
            max_weight=ASYNC_RUN_MAX_WEIGHT,
            observations=_ASYNC_OBSERVATIONS,
        ),
        async_vault=True,
    )
    return state, list(_ASYNC_OBSERVATIONS)


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


@pytest.mark.parametrize("waterfall", [True, False], ids=["waterfall", "positions"])
def test_method_agnostic_park_promote_under_size_risk(waterfall: bool):
    """The park -> deposit-on-open fires under both real size-risk normalisers, not just the simple one.

    The phase-aware pass operates on self.signals after normalisation, so it is allocation-method
    agnostic. These runs use a real ``USDTVLSizeRiskModel`` so ``normalise_weights`` dispatches to
    ``_normalise_weights_waterfall`` (waterfall=True) or - because the harness passes
    ``max_positions`` - ``_normalise_weights_size_risk_positions`` (waterfall=False); waterfall=True
    *alone* silently falls back to the simple normaliser. Both runs must still park the
    closed-window target and promote it on open: the slot-holding property under genuinely
    different normaliser code paths, not asserted-by-construction.

    1. Run the scenario with a size-risk model so the chosen size-risk normaliser genuinely runs.
    2. Confirm a size-risk path actually executed (guards against a silent fallback to the simple
       normaliser).
    3. The target vault is still parked and promoted, and ends holding a realised position.
    """
    _SIZE_RISK_RAN.clear()
    # 1. Run under the chosen real size-risk normaliser.
    state = _run(lambda i: _phase_aware_decide_trades(i, size_risk=True, waterfall=waterfall))

    # 2. A size-risk normaliser genuinely ran (accepted_investable_equity set - simple never sets it).
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


def test_promote_survives_coincident_async_sell(async_run: tuple[State, list[dict]]):
    """Invariant 4 end-to-end: a promotion coinciding with a same-cycle async sell is not scaled down.

    ``_cap_buys_by_async_sell_proceeds`` engages only when the cycle has an async sell; it scales
    buys down to raw cash + sync sells - unless the phase-aware widening adds the queue venue's
    redeemable balance. This is the parent plan's "engineer a coincident same-cycle async sell so
    the widening is actually exercised, not vacuous" - here through a real backtest run, not stubs
    (the unit-level proof is ``test_cap_buys_widened_by_venue_through_async_sell``).

    1. Exactly one promote cycle, on which an async VA sell was emitted too - and the cap genuinely
       classified VA as async via the position's own settlement history (guards against the cap
       early-returning, which would make this test vacuous).
    2. The promotion buy was not scaled: no ``capped_by_pending_settlement_cash`` flag, and the
       emitted size far exceeds the raw cash available that cycle (the venue balance funded it -
       the base model would have scaled the buy down to raw cash).
    3. The durable promote event persists the same unscaled size (a log-consistency check - both
       numbers read the post-generation adjust, so this proves persistence, not an independent fact).
    """
    state, observations = async_run

    # 1. One promote cycle, with the engineered coincident async sell, classified async by the cap.
    promote_cycles = [o for o in observations if o["promote_emitted"]]
    assert len(promote_cycles) == 1
    o = promote_cycles[0]
    assert o["async_sell_emitted"], "the VA exit did not coincide with the promote cycle"
    assert o["async_position_has_flow"], "cap would early-return: VA position shows no async flow"

    # 2. The promotion was not scaled by the cap and was funded beyond raw cash.
    assert "capped_by_pending_settlement_cash" not in o["target_flags"]
    assert o["target_adjust_after"] > o["raw_cash_before"] + 1_000, \
        f"promotion {o['target_adjust_after']} should exceed raw cash {o['raw_cash_before']} (venue-funded)"

    # 3. The promote event records the unscaled deposit size.
    promote_events = [
        e for e in iter_all_events(state.other_data)
        if e.kind == EVENT_PROMOTE and e.vault_internal_id == TARGET_INTERNAL_ID
    ]
    assert len(promote_events) == 1
    assert promote_events[0].usd == pytest.approx(o["target_adjust_after"], rel=0.01)


def test_settled_async_redeem_proceeds_swept_to_venue(async_run: tuple[State, list[dict]]):
    """Redemption side end-to-end: settled async-redeem proceeds are swept into the venue, not idle.

    The promote cycle drains the venue to fund the deposit; from then on the only cash inflow is
    VA's settling redemption (VT is weight-capped so it cannot re-absorb the proceeds). The venue
    recovering to ~the sale value therefore proves the settled proceeds were swept, attacking the
    original "cumulative churn withheld across settlement windows" leak (the unit-level sweep
    arithmetic is ``test_yield_safe_sweeps_reserve_withholding_pending_redemptions``).

    1. VA's exit executed as an async redemption (request-then-settle, the ``vault_async_flow``
       marker) and succeeded.
    2. The venue was drained by the promote, then recovered to >= ~the VA sale value - only the
       settled proceeds could have refilled it.
    3. Idle cash ends at the reserve floor: the proceeds are in the venue, not idle.
    """
    state, observations = async_run

    # 1. The VA exit was a settled async redemption.
    va_sells = [
        t
        for p in state.portfolio.get_all_positions(pending=True)
        if p.pair.internal_id == ASYNC_INTERNAL_ID
        for t in p.trades.values()
        if t.is_sell()
    ]
    assert len(va_sells) == 1
    va_sell = va_sells[0]
    assert va_sell.other_data.get("vault_async_flow"), "VA sell did not go through async settlement"
    assert va_sell.is_success()
    va_sale_usd = va_sell.get_planned_value()
    assert va_sale_usd > 10_000  # sanity: the position was substantial

    # 2. Venue drained by the promote, then recovered from the settled proceeds alone.
    promote_cycle = next(o["cycle"] for o in observations if o["promote_emitted"])
    after_promote = [o for o in observations if o["cycle"] > promote_cycle]
    assert after_promote, "run too short: no cycles after the promote"
    assert after_promote[0]["venue_value_before"] < 0.35 * va_sale_usd, "venue was not drained by the promote"
    assert after_promote[-1]["venue_value_before"] >= 0.75 * va_sale_usd, \
        "settled redeem proceeds did not come back to the venue"

    # 3. Idle cash at the reserve floor at run end.
    equity = state.portfolio.get_total_equity()
    assert state.portfolio.get_cash() / equity == pytest.approx(1 - ALLOCATION, abs=0.03)


def _redeem_signals(cycle: int) -> dict[int, float]:
    """Per-cycle signals for the redemption-wait run.

    Hold VT until cycle 10 (parks on the closed cycles 1-4, promotes on 5). From cycle 11 - VT's
    window has closed again - exit VT: the redemption is blocked until the window reopens on
    cycle 15.
    """
    if cycle < 11:
        return {TARGET_INTERNAL_ID: 0.999}
    return {TARGET_INTERNAL_ID: 0.0}


def test_redemption_window_wait_and_clear():
    """Redemption-locked value is durably recorded across closed windows and cleared on reopen (CU-7).

    The redemption side is passive (the carry-forward pin owns the behaviour); the phase-aware model
    mirrors the per-cycle ``missed_redemption_usd`` markers into durable redeem-block / redeem-clear
    events so the redemption-locked buffer is chartable. The same ``vault_window_overrides``
    schedule gates ``check_redemption``: a *held* VT is locked whenever the window is closed (the
    precautionary carry-forward pin - true lock, exit wanted or not), and exiting VT while closed
    waits until the reopen.

    1. Run three weeks: VT promotes on cycle 5 and is then held through the closed windows; the
       signal drops to zero on cycle 11 (window closed on cycles 11-14, reopens on 15).
    2. The lock/unlock event sequence is exact and deduped: block on cycle 6 (held while the window
       closed - re-blocks at the unchanged value append nothing), clear on 10 (window reopens),
       block on 11 (closed again, exit now waiting), clear on 15 (reopened - the sell executes).
    3. The exit actually executed: no VT position remains open at run end.
    4. pending_trigger_queue renders the locked value as a negative band.
    """
    # 1. Three-week run: promote, hold across closed windows, then exit into a closed window.
    state = _run(
        lambda i: _phase_aware_decide_trades(i, signal_fn=_redeem_signals),
        end_at=datetime.datetime(2024, 1, 22),
    )

    # 2. The exact lock/unlock sequence, deduped within each closed window.
    events = list(iter_all_events(state.other_data))
    blocks = [e for e in events if e.kind == EVENT_REDEEM_BLOCK and e.vault_internal_id == TARGET_INTERNAL_ID]
    clears = [e for e in events if e.kind == EVENT_REDEEM_CLEAR and e.vault_internal_id == TARGET_INTERNAL_ID]
    assert [e.cycle for e in blocks] == [6, 11], f"expected deduped blocks on cycles 6 and 11, got {blocks}"
    assert [e.cycle for e in clears] == [10, 15], f"expected clears on the window reopens, got {clears}"
    assert all(e.usd > 10_000 for e in blocks)  # the locked value is the substantial VT holding

    # 3. The redemption executed once the window opened: VT no longer held.
    held = {p.pair.internal_id for p in state.portfolio.open_positions.values()}
    assert TARGET_INTERNAL_ID not in held

    # 4. The chart shows the locked value as a negative band.
    _fig, df = pending_trigger_queue(ChartInput(execution_context=unit_test_execution_context, state=state))
    assert "Redemption locked" in df.columns
    assert df["Redemption locked"].min() == pytest.approx(-blocks[0].usd, rel=0.01)
