"""Unit tests for PhaseAwareAlphaModel (PR-B core behaviour).

Covers the two correctness invariants the model enforces (the same-cycle cash
budget includes redeemable venue balance; queue-venue positions are excluded from
old-weight accounting), the pre-generation park step (a closed-window deposit is
deferred with its adjustment zeroed, so the min-trade gate and the same-cycle cash
cap never see it), and the post-generation promote reconciliation (a promote is
only logged once the deposit trade has actually emitted).

The phase-aware pass operates on ``self.signals`` regardless of which
``normalise_weights`` variant produced them, so the promote/stale tests set
``self.signals`` directly - that is exactly the allocation-method-agnostic property.

Small local doubles (``_Stub*``) stand in for PositionManager / state / pricing /
trades so the model methods can be exercised in isolation without a full backtest.
"""
import dataclasses
import datetime

import pytest

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.other_data import OtherData
from tradeexecutor.strategy.alpha_model import AlphaModel, TradingPairSignal, TradingPairSignalFlags
from tradeexecutor.strategy.phase_aware import (
    EVENT_PARK,
    EVENT_REDEEM_BLOCK,
    EVENT_REDEEM_CLEAR,
    PhaseAwareAlphaModel,
    QueueVaultEvent,
    append_queue_event,
    iter_all_events,
    read_open_park_events,
    read_open_redeem_block_events,
)
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradingstrategy.chain import ChainId


def _make_pair(internal_id: int, symbol: str = "VLT") -> TradingPairIdentifier:
    base = AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), symbol, 18, internal_id)
    quote = AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "USDC", 6, 999999)
    return TradingPairIdentifier(
        base,
        quote,
        generate_random_ethereum_address(),
        generate_random_ethereum_address(),
        internal_id=internal_id,
    )


def _make_signal(pair: TradingPairIdentifier, adjust_usd: float, carry_forward: bool = False) -> TradingPairSignal:
    signal = TradingPairSignal(pair=pair, signal=1.0)
    signal.position_adjust_usd = adjust_usd
    signal.position_adjust_quantity = 1.0
    signal.carry_forward_position = carry_forward
    return signal


@dataclasses.dataclass
class _StubPosition:
    pair: TradingPairIdentifier
    value: float = 0.0
    credit: bool = False
    vault: bool = False
    async_flow: bool = False
    pending_settlement: bool = False

    def get_value(self) -> float:
        return self.value

    def is_credit_supply(self) -> bool:
        return self.credit

    def is_vault(self) -> bool:
        return self.vault

    def has_async_vault_flow(self) -> bool:
        return self.async_flow

    def has_pending_vault_settlement(self) -> bool:
        return self.pending_settlement

    def get_vault_settlement_pending_value(self) -> float:
        return 0.0


@dataclasses.dataclass
class _StubPortfolio:
    open_positions: dict


@dataclasses.dataclass
class _StubPricing:
    open_pairs: set

    def can_deposit(self, timestamp, pair) -> bool:
        return pair.internal_id in self.open_pairs


@dataclasses.dataclass
class _StubState:
    other_data: OtherData
    portfolio: _StubPortfolio


@dataclasses.dataclass
class _StubPositionManager:
    state: _StubState
    pricing_model: _StubPricing
    cash: float = 0.0
    #: pair.internal_id -> position, returned by get_current_position_for_pair (the async-sell
    #: fallback path in _cap_buys_by_async_sell_proceeds queries this).
    pending_positions: dict | None = None

    def get_current_cash(self) -> float:
        return self.cash

    def get_current_portfolio(self) -> _StubPortfolio:
        return self.state.portfolio

    def get_current_position_for_pair(self, pair: TradingPairIdentifier, pending: bool = False):
        return (self.pending_positions or {}).get(pair.internal_id)

    def is_async_vault_sell_pair(self, pair: TradingPairIdentifier, *, position_pair: TradingPairIdentifier | None = None) -> bool:
        # Mirrors PositionManager.is_async_vault_sell_pair over the stub's position map.
        if pair.is_async_vault():
            return True
        position = self.get_current_position_for_pair(position_pair or pair, pending=True)
        return position is not None and position.has_async_vault_flow()


@dataclasses.dataclass
class _StubTrade:
    pair: TradingPairIdentifier
    buy: bool = True

    def is_buy(self) -> bool:
        return self.buy


def _make_pm(
    other_data: OtherData,
    open_pairs: set,
    cash: float = 0.0,
    positions: dict | None = None,
    pending_positions: dict | None = None,
) -> _StubPositionManager:
    return _StubPositionManager(
        state=_StubState(other_data, _StubPortfolio(positions or {})),
        pricing_model=_StubPricing(open_pairs=open_pairs),
        cash=cash,
        pending_positions=pending_positions,
    )


def test_available_same_cycle_cash_includes_venue():
    """The same-cycle buy budget adds the redeemable queue-venue balance (invariant 4).

    1. A venue position (internal id in venue_pair_ids) holds $500; raw cash is $100.
    2. _available_same_cycle_cash returns base cash + venue = $600.
    """
    venue_pos = _StubPosition(pair=_make_pair(101), value=500.0)
    pm = _make_pm(OtherData(), open_pairs=set(), cash=100.0, positions={1: venue_pos})
    alpha = PhaseAwareAlphaModel(datetime.datetime(2024, 1, 1), cycle=1, venue_pair_ids={101})

    # 1-2. Base cash plus the redeemable venue value.
    assert alpha._available_same_cycle_cash(pm) == pytest.approx(600.0)


def test_count_position_in_old_weights_excludes_venue():
    """Queue-venue positions are excluded from old-weight accounting (invariant 2).

    1. A venue position (internal id in venue_pair_ids) is excluded (returns False).
    2. A directional spot position delegates to the base predicate (returns True).
    """
    alpha = PhaseAwareAlphaModel(datetime.datetime(2024, 1, 1), cycle=1, venue_pair_ids={101})
    venue_pos = _StubPosition(pair=_make_pair(101))
    directional = _StubPosition(pair=_make_pair(202))

    # 1. Venue excluded before the base predicate runs.
    assert alpha._count_position_in_old_weights(venue_pos, ignore_credit=True, portfolio_pairs=None) is False
    # 2. Directional spot delegates to base (not bridge/credit/vault) -> counted.
    assert alpha._count_position_in_old_weights(directional, ignore_credit=True, portfolio_pairs=None) is True


def test_apply_parks_closed_window_and_zeros_adjust():
    """A closed-window deposit is parked before generation: adjustment zeroed, event logged.

    1. A positive deposit signal whose window is closed.
    2. apply_phase_aware_intent zeros position_adjust_usd / quantity, marks ignored,
       flags parked_in_queue_vault, records parked_usd, and logs a park event.
    """
    other_data = OtherData()
    signal = _make_signal(_make_pair(101), 1000.0)
    pm = _make_pm(other_data, open_pairs=set())  # 101 closed
    alpha = PhaseAwareAlphaModel(datetime.datetime(2024, 1, 1), cycle=5, venue_pair_ids={999})
    alpha.signals = {101: signal}

    # 1-2. Park before generation - the deferred buy is zeroed out.
    alpha.apply_phase_aware_intent(pm)
    assert signal.position_adjust_usd == pytest.approx(0.0)
    assert signal.position_adjust_quantity == pytest.approx(0.0)
    assert signal.position_adjust_ignored is True
    assert TradingPairSignalFlags.parked_in_queue_vault in signal.flags
    assert signal.other_data["parked_usd"] == pytest.approx(1000.0)
    open_events = read_open_park_events(other_data)
    assert set(open_events) == {101}
    assert open_events[101].usd == pytest.approx(1000.0)
    assert open_events[101].cycle == 5


def test_promote_finalised_only_when_buy_emits():
    """A promotion is logged/flagged only once the deposit trade actually emits.

    1. Vault A parked on an earlier cycle; this cycle its window is open and it is targeted.
    2. apply marks A a promote candidate but logs no promote event and sets no flag yet.
    3. reconcile with an emitted buy on A logs the promote, flags the signal, closes the park.
    """
    other_data = OtherData()
    append_queue_event(other_data, QueueVaultEvent(EVENT_PARK, 101, 1000.0, 1))
    pair = _make_pair(101)
    signal = _make_signal(pair, 1200.0)
    pm = _make_pm(other_data, open_pairs={101})  # A open
    alpha = PhaseAwareAlphaModel(datetime.datetime(2024, 1, 2), cycle=2, venue_pair_ids={999})
    alpha.signals = {101: signal}

    # 1-2. apply marks the candidate but does not promote yet.
    alpha.apply_phase_aware_intent(pm)
    assert alpha._promote_candidates == {101}
    assert TradingPairSignalFlags.promoted_from_queue_vault not in signal.flags
    assert set(read_open_park_events(other_data)) == {101}  # still open until the buy emits

    # 3. reconcile with the emitted buy -> promote + close.
    alpha.reconcile_phase_aware_events(pm, [_StubTrade(pair=pair)])
    assert TradingPairSignalFlags.promoted_from_queue_vault in signal.flags
    assert read_open_park_events(other_data) == {}


def test_promote_candidate_stays_open_when_buy_suppressed():
    """A promotion candidate whose buy did not emit keeps its park event open.

    1. Vault A parked earlier; open + targeted this cycle -> promote candidate.
    2. reconcile with no buy trade for A leaves the park event open and unflagged (no
       premature promote when the min-trade gate or dust suppresses the buy).
    """
    other_data = OtherData()
    append_queue_event(other_data, QueueVaultEvent(EVENT_PARK, 101, 1000.0, 1))
    signal = _make_signal(_make_pair(101), 1200.0)
    pm = _make_pm(other_data, open_pairs={101})
    alpha = PhaseAwareAlphaModel(datetime.datetime(2024, 1, 2), cycle=2, venue_pair_ids={999})
    alpha.signals = {101: signal}
    alpha.apply_phase_aware_intent(pm)

    # 2. No emitted buy for A -> the park event stays open, no promote flag.
    alpha.reconcile_phase_aware_events(pm, [])
    assert set(read_open_park_events(other_data)) == {101}
    assert TradingPairSignalFlags.promoted_from_queue_vault not in signal.flags


def test_stale_close_when_no_longer_targeted():
    """A parked vault dropped from the ranked set this cycle is stale-closed.

    1. Vault B parked earlier; absent from self.signals this cycle.
    2. apply logs a close event; the park event is no longer open.
    """
    other_data = OtherData()
    append_queue_event(other_data, QueueVaultEvent(EVENT_PARK, 202, 2000.0, 1))
    pm = _make_pm(other_data, open_pairs=set())
    alpha = PhaseAwareAlphaModel(datetime.datetime(2024, 1, 2), cycle=2, venue_pair_ids={999})
    alpha.signals = {}  # B no longer targeted

    alpha.apply_phase_aware_intent(pm)
    assert read_open_park_events(other_data) == {}


def test_settlement_pending_park_stays_open():
    """A parked vault now held by a carry-forward (settlement) pin is not stale-closed.

    1. Vault V parked earlier; this cycle it has a carry_forward_position signal (settling, adjust 0).
    2. apply leaves the park event open (the deposit is in flight, not stale).
    """
    other_data = OtherData()
    append_queue_event(other_data, QueueVaultEvent(EVENT_PARK, 101, 1000.0, 1))
    signal = _make_signal(_make_pair(101), 0.0, carry_forward=True)
    pm = _make_pm(other_data, open_pairs=set())
    alpha = PhaseAwareAlphaModel(datetime.datetime(2024, 1, 2), cycle=2, venue_pair_ids={999})
    alpha.signals = {101: signal}

    alpha.apply_phase_aware_intent(pm)
    assert set(read_open_park_events(other_data)) == {101}  # left open, not stale-closed


def test_still_closed_reparks_and_stays_open():
    """A vault whose window is still closed is re-parked and its event stays open.

    1. Vault V parked earlier; still closed + targeted this cycle.
    2. apply re-parks it (zeros the adjust) and the park event remains open, refreshed.
    """
    other_data = OtherData()
    append_queue_event(other_data, QueueVaultEvent(EVENT_PARK, 101, 1000.0, 1))
    signal = _make_signal(_make_pair(101), 1100.0)
    pm = _make_pm(other_data, open_pairs=set())  # still closed
    alpha = PhaseAwareAlphaModel(datetime.datetime(2024, 1, 2), cycle=2, venue_pair_ids={999})
    alpha.signals = {101: signal}

    alpha.apply_phase_aware_intent(pm)
    assert signal.position_adjust_usd == pytest.approx(0.0)
    assert TradingPairSignalFlags.parked_in_queue_vault in signal.flags
    open_events = read_open_park_events(other_data)
    assert set(open_events) == {101}
    assert open_events[101].usd == pytest.approx(1100.0)  # refreshed to this cycle's target
    assert open_events[101].cycle == 2


def test_no_cycle_disables_phase_aware():
    """Without a cycle number, both phase-aware passes are inert.

    1. apply_phase_aware_intent logs nothing, marks no candidates, and does not park.
    2. reconcile is a no-op.
    """
    other_data = OtherData()
    pair = _make_pair(101)
    signal = _make_signal(pair, 1000.0)
    pm = _make_pm(other_data, open_pairs=set())
    alpha = PhaseAwareAlphaModel(datetime.datetime(2024, 1, 1), cycle=None, venue_pair_ids={999})
    alpha.signals = {101: signal}

    # 1. Inert apply: nothing logged, no candidate, signal untouched.
    alpha.apply_phase_aware_intent(pm)
    assert read_open_park_events(other_data) == {}
    assert alpha._promote_candidates == set()
    assert signal.position_adjust_usd == pytest.approx(1000.0)

    # 2. Inert reconcile.
    alpha.reconcile_phase_aware_events(pm, [_StubTrade(pair=pair)])
    assert read_open_park_events(other_data) == {}


def test_repark_same_amount_does_not_duplicate_event():
    """Re-parking a vault at the same amount does not append a duplicate park event.

    1. Vault parked earlier at $1000; still closed and targeted at the same $1000 this cycle.
    2. apply zeros the adjust again but appends no new event (the durable log stays bounded).
    """
    other_data = OtherData()
    append_queue_event(other_data, QueueVaultEvent(EVENT_PARK, 101, 1000.0, 1))
    signal = _make_signal(_make_pair(101), 1000.0)
    pm = _make_pm(other_data, open_pairs=set())  # still closed
    alpha = PhaseAwareAlphaModel(datetime.datetime(2024, 1, 2), cycle=2, venue_pair_ids={999})
    alpha.signals = {101: signal}

    # 1-2. Re-parked (still parked) but no duplicate event appended.
    alpha.apply_phase_aware_intent(pm)
    events = list(iter_all_events(other_data))
    assert len(events) == 1
    assert events[0].cycle == 1
    assert signal.position_adjust_usd == pytest.approx(0.0)


def test_cap_buys_widened_by_venue_through_async_sell():
    """The same-cycle async cap draws on venue balance for the phase-aware model (invariant 4, non-vacuous).

    Exercises ``_cap_buys_by_async_sell_proceeds`` end-to-end, not just the ``_available_same_cycle_cash``
    hook: a coincident async-vault sell keeps the cap from early-returning (``async_sell_usd > 0``) and
    the buys exceed raw cash + synchronous sells, so the cap actually bites. The base ``AlphaModel``
    scales the buy down to raw cash; ``PhaseAwareAlphaModel``, whose budget includes the redeemable
    queue-venue balance, funds the buy in full. This is the non-vacuous shape the design plan demanded
    (the earlier test only called the hook directly).

    1. Signals: an async-vault sell (adjust -$300, detected via the position's async-vault flow) plus a
       $500 buy; raw cash $100, a $1000 queue-venue position, no synchronous sells.
    2. Base AlphaModel budget = raw cash $100, so the $500 buy is scaled to $100 and flagged
       capped_by_pending_settlement_cash.
    3. PhaseAwareAlphaModel budget = cash $100 + venue $1000 = $1100 >= $500, so the buy is left
       unscaled and unflagged.
    """
    venue_pair = _make_pair(101, "VENUE")
    async_pair = _make_pair(202, "ASYNC")
    buy_pair = _make_pair(303, "BUY")

    # 1. Same signal set for both models; a fresh position_manager per run (the cap mutates in place).
    def build_signals() -> dict:
        return {202: _make_signal(async_pair, -300.0), 303: _make_signal(buy_pair, 500.0)}

    def make_pm() -> _StubPositionManager:
        # The async sell is detected via the position fallback (has_async_vault_flow), not pair
        # features; the venue position (id 101) supplies the phase-aware model's widened budget.
        return _make_pm(
            OtherData(),
            open_pairs=set(),
            cash=100.0,
            positions={101: _StubPosition(pair=venue_pair, value=1000.0, vault=True)},
            pending_positions={202: _StubPosition(pair=async_pair, value=300.0, async_flow=True)},
        )

    # 2. Base model: budget is raw cash $100, so the $500 buy is scaled to $100.
    base = AlphaModel(datetime.datetime(2024, 1, 1))
    base.signals = build_signals()
    base._cap_buys_by_async_sell_proceeds(make_pm())
    assert base.signals[303].position_adjust_usd == pytest.approx(100.0)
    assert TradingPairSignalFlags.capped_by_pending_settlement_cash in base.signals[303].flags

    # 3. Phase-aware model: budget $100 + $1000 venue covers the $500 buy -> unscaled, unflagged.
    phase = PhaseAwareAlphaModel(datetime.datetime(2024, 1, 1), cycle=1, venue_pair_ids={101})
    phase.signals = build_signals()
    phase._cap_buys_by_async_sell_proceeds(make_pm())
    assert phase.signals[303].position_adjust_usd == pytest.approx(500.0)
    assert TradingPairSignalFlags.capped_by_pending_settlement_cash not in phase.signals[303].flags


def test_park_dominant_cancels_cycle_via_min_trade_gate():
    """Parking the dominant deposit can cancel the cycle at the whole-portfolio min-trade gate.

    apply_phase_aware_intent zeroes a parked deposit's adjustment *before* trade generation, so the
    pre-cap gate (``max_diff = max(abs(position_adjust_usd))``) no longer counts it. When the parked
    buy was the only adjustment above the threshold, ``max_diff`` drops below it and the whole cycle
    is cancelled cleanly - the documented gate hazard the design plan warned about, exercised here
    rather than left implicit. (The converse - a non-dominant park leaving a valid cycle intact -
    reaches the full per-signal trade-creation loop and is covered by the CU-3 integration backtest.)

    1. A dominant closed-window deposit ($1000) plus a tiny open directional adjust ($5); gate
       threshold $50.
    2. apply_phase_aware_intent parks the dominant deposit, zeroing its adjustment.
    3. generate_rebalance_trades_and_triggers returns no trades: max_diff is now the $5 adjust
       (< $50), so the gate cancels the whole cycle.
    """
    other_data = OtherData()
    dominant = _make_signal(_make_pair(101), 1000.0)  # closed-window deposit, will be parked
    tiny = _make_signal(_make_pair(202), 5.0)  # open, below the gate threshold
    pm = _make_pm(other_data, open_pairs={202})  # 101 closed, 202 open
    alpha = PhaseAwareAlphaModel(datetime.datetime(2024, 1, 1), cycle=1, venue_pair_ids={999})
    alpha.signals = {101: dominant, 202: tiny}

    # 2. Park the dominant deposit (window closed) -> its adjustment is zeroed before generation.
    alpha.apply_phase_aware_intent(pm)
    assert dominant.position_adjust_usd == pytest.approx(0.0)

    # 3. The gate now sees only the $5 adjust (< $50) and cancels the whole cycle.
    trades = alpha.generate_rebalance_trades_and_triggers(pm, min_trade_threshold=50.0)
    assert trades == []
    assert alpha.max_position_adjust_usd == pytest.approx(5.0)  # parked dominant no longer counts


def test_venue_pair_rejected_from_signals():
    """The queue venue cannot receive an alpha signal - invariant 1 enforced, not just conventional.

    A venue that slipped into the candidate universe would be treated as a fresh directional buy
    (it is excluded from old-weight accounting) while YieldManager trades the same pair in the
    same cycle - conflicting same-pair trades. set_signal fails fast instead.

    1. set_signal on a venue pair raises with a descriptive message.
    2. set_signal on a directional pair still works.
    """
    alpha = PhaseAwareAlphaModel(datetime.datetime(2024, 1, 1), cycle=1, venue_pair_ids={101})

    # 1. The venue pair is rejected at signal time.
    with pytest.raises(AssertionError, match="must not receive an alpha signal"):
        alpha.set_signal(_make_pair(101), 0.5)

    # 2. A directional pair is unaffected.
    directional = _make_pair(202)
    alpha.set_signal(directional, 0.5)
    assert 202 in alpha.raw_signals


def test_carry_forward_skips_queue_venue_pairs():
    """Queue venues are skipped by the phase-aware carry-forward pin even if not redeemable.

    A deep synchronous venue can transiently report maxRedeem == 0 under high utilisation.
    YieldManager owns that venue, so the phase-aware carry-forward loop must not call
    set_signal() on it; the invariant-1 signal guard remains for genuine misconfiguration.

    1. A queue-venue position is open while pricing reports it non-redeemable.
    2. carry_forward_non_redeemable_positions returns without creating a venue signal.
    3. The same non-redeemable pair is still rejected when signalled directly.
    """
    venue_pair = _make_pair(101, "VENUE")
    venue_pos = _StubPosition(pair=venue_pair, value=500.0, vault=True)
    pm = _make_pm(OtherData(), open_pairs=set(), positions={1: venue_pos})
    alpha = PhaseAwareAlphaModel(datetime.datetime(2024, 1, 1), cycle=1, venue_pair_ids={101})

    # 1-2. Non-redeemable venue does not get carry-forward pinned into raw_signals.
    locked = alpha.carry_forward_non_redeemable_positions(pm)
    assert locked == pytest.approx(0.0)
    assert alpha.raw_signals == {}

    # 3. Direct alpha signals on the venue are still a configuration error.
    with pytest.raises(AssertionError, match="must not receive an alpha signal"):
        alpha.set_signal(venue_pair, 1.0)


def test_redemption_wait_events_lifecycle():
    """Blocked redemptions are mirrored to the durable event log: block, dedup, re-block, clear (CU-7).

    The redemption side is passive - the settlement pin and the redemption checks own the
    behaviour and stamp per-cycle ``missed_redemption_usd`` on the signals - so
    ``reconcile_phase_aware_events`` mirrors those markers into durable redeem-block /
    redeem-clear events for the redemption-locked chart, without touching any trade decision.

    1. Cycle 1: a signal carrying ``missed_redemption_usd`` logs a redeem-block event.
    2. Cycle 2: the same amount is still blocked - no duplicate event is appended.
    3. Cycle 3: the locked amount changes - a new block event is appended (no intervening clear)
       and the fold reflects only the latest amount (no double count).
    4. Cycle 4: the marker is gone (the redemption executed or is no longer wanted) - a
       redeem-clear closes the open block.
    """
    other_data = OtherData()
    pair = _make_pair(701)
    pm = _make_pm(other_data, open_pairs=set())

    def _cycle(cycle: int, missed_usd: float | None) -> None:
        alpha = PhaseAwareAlphaModel(datetime.datetime(2024, 1, cycle), cycle=cycle)
        signal = _make_signal(pair, 0.0)
        if missed_usd is not None:
            signal.other_data["missed_redemption_usd"] = missed_usd
        alpha.signals = {701: signal}
        alpha.reconcile_phase_aware_events(pm, [])

    # 1. Cycle 1: a blocked redemption opens a redeem-block event.
    _cycle(1, 500.0)
    assert read_open_redeem_block_events(other_data)[701].usd == pytest.approx(500.0)

    # 2. Cycle 2: unchanged amount -> deduped, still exactly one block event in the log.
    _cycle(2, 500.0)
    assert sum(1 for e in iter_all_events(other_data) if e.kind == EVENT_REDEEM_BLOCK) == 1

    # 3. Cycle 3: changed amount -> a second block event, and the fold holds only the latest value.
    _cycle(3, 750.0)
    assert sum(1 for e in iter_all_events(other_data) if e.kind == EVENT_REDEEM_BLOCK) == 2
    open_blocks = read_open_redeem_block_events(other_data)
    assert len(open_blocks) == 1  # dict overwrite: latest per vault, no double count
    assert open_blocks[701].usd == pytest.approx(750.0)

    # 4. Cycle 4: marker gone -> the open block is closed with a redeem-clear.
    _cycle(4, None)
    assert read_open_redeem_block_events(other_data) == {}
    assert sum(1 for e in iter_all_events(other_data) if e.kind == EVENT_REDEEM_CLEAR) == 1
