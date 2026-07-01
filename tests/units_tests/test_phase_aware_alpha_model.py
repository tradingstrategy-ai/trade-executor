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
from tradeexecutor.strategy.alpha_model import TradingPairSignal, TradingPairSignalFlags
from tradeexecutor.strategy.phase_aware import (
    EVENT_PARK,
    PhaseAwareAlphaModel,
    QueueVaultEvent,
    append_queue_event,
    iter_all_events,
    read_open_park_events,
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

    def get_value(self) -> float:
        return self.value

    def is_credit_supply(self) -> bool:
        return self.credit

    def is_vault(self) -> bool:
        return self.vault


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

    def get_current_cash(self) -> float:
        return self.cash


@dataclasses.dataclass
class _StubTrade:
    pair: TradingPairIdentifier
    buy: bool = True

    def is_buy(self) -> bool:
        return self.buy


def _make_pm(other_data: OtherData, open_pairs: set, cash: float = 0.0, positions: dict | None = None) -> _StubPositionManager:
    return _StubPositionManager(
        state=_StubState(other_data, _StubPortfolio(positions or {})),
        pricing_model=_StubPricing(open_pairs=open_pairs),
        cash=cash,
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
