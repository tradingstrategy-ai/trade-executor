"""Unit tests for the PR-0 phase-aware scaffolding.

Covers (1) the durable queue-venue event log's full-history-fold reader — the
correctness fix that makes park events survive strategy cycles that log nothing,
which a naive ``OtherData.load_latest`` read would silently drop — the venue
helpers, and (2) the three behaviour-preserving hooks extracted from
:py:class:`AlphaModel` so ``PhaseAwareAlphaModel`` can override them.

The tests use small local doubles (``_Stub*``) rather than full ``Portfolio`` /
``PositionManager`` objects: the functions under test are pure and only touch a
couple of duck-typed attributes, so a lightweight double keeps the unit tests
fast and isolated.
"""
import dataclasses
import datetime

import pytest

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.other_data import OtherData
from tradeexecutor.strategy.alpha_model import AlphaModel, TradingPairSignalFlags
from tradeexecutor.strategy.phase_aware import (
    EVENT_CLOSE,
    EVENT_PARK,
    EVENT_PROMOTE,
    IS_QUEUE_VAULT_KEY,
    QUEUE_VAULT_EVENT_LOG_KEY,
    QueueVaultEvent,
    append_queue_event,
    is_queue_vault,
    mark_queue_vault,
    queue_vault_pair_ids,
    queue_venue_redeemable,
    read_open_park_events,
)
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradingstrategy.chain import ChainId


@dataclasses.dataclass
class _StubPair:
    cctp: bool = False
    internal_id: int | None = None
    async_vault: bool = False

    def is_cctp_bridge(self) -> bool:
        return self.cctp

    def is_async_vault(self) -> bool:
        return self.async_vault


@dataclasses.dataclass
class _StubRule:
    pair: _StubPair


@dataclasses.dataclass
class _StubRuleset:
    weights: list


@dataclasses.dataclass
class _StubPosition:
    value: float = 0.0
    other_data: dict = dataclasses.field(default_factory=dict)
    pair: _StubPair = dataclasses.field(default_factory=_StubPair)
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
class _StubPositionManager:
    cash: float

    def get_current_cash(self) -> float:
        return self.cash


@dataclasses.dataclass
class _StubSignal:
    pair: object
    position_adjust_usd: float
    position_adjust_ignored: bool = False
    flags: set = dataclasses.field(default_factory=set)
    other_data: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class _StubPricingModel:
    can_deposit_result: bool

    def can_deposit(self, timestamp, pair) -> bool:
        return self.can_deposit_result


@dataclasses.dataclass
class _StubSkipPositionManager:
    pricing_model: _StubPricingModel

    def is_problematic_pair(self, pair) -> bool:
        return False


def test_queue_event_log_full_history_fold():
    """Open park events survive cycles that log no queue event (the load_latest trap).

    1. Park vaults A and B on cycle 1.
    2. Store unrelated bookkeeping on cycle 2 with no queue event (a "quiet" cycle).
    3. The full-history reader still sees A and B, but ``OtherData.load_latest`` sees nothing.
    4. Promote A and stale-close B on cycle 3 — both become closed.
    5. Re-parking A twice keeps a single open event with the latest USD (dedup).
    """
    other_data = OtherData()

    # 1. Park vaults A (id 10) and B (id 20) on cycle 1.
    append_queue_event(other_data, QueueVaultEvent(EVENT_PARK, 10, 1000.0, 1))
    append_queue_event(other_data, QueueVaultEvent(EVENT_PARK, 20, 2000.0, 1))

    # 2. A quiet queue cycle: the framework still writes bookkeeping to other_data.
    other_data.save(2, "decision_cycle_ended_at", "2024-01-02")

    # 3. Full-history fold keeps both open; load_latest would have dropped them.
    open_events = read_open_park_events(other_data)
    assert set(open_events) == {10, 20}
    assert open_events[10].usd == pytest.approx(1000.0)
    assert other_data.load_latest(QUEUE_VAULT_EVENT_LOG_KEY) is None

    # 4. Promote A and stale-close B on cycle 3 — both become closed.
    append_queue_event(other_data, QueueVaultEvent(EVENT_PROMOTE, 10, 1000.0, 3))
    append_queue_event(other_data, QueueVaultEvent(EVENT_CLOSE, 20, 0.0, 3))
    assert read_open_park_events(other_data) == {}

    # 5. Re-parking A twice keeps a single open event with the latest USD (dedup).
    append_queue_event(other_data, QueueVaultEvent(EVENT_PARK, 10, 1500.0, 4))
    append_queue_event(other_data, QueueVaultEvent(EVENT_PARK, 10, 1800.0, 5))
    reopened = read_open_park_events(other_data)
    assert set(reopened) == {10}
    assert reopened[10].usd == pytest.approx(1800.0)


def test_queue_venue_helpers():
    """Venue identification, redeemable value, and event JSON round-trip.

    1. A tagged position is a queue vault; an untagged one is not.
    2. queue_venue_redeemable sums value over tagged positions only.
    3. A QueueVaultEvent survives a to_dict/from_dict round-trip (durable, JSON-primitive).
    """
    # 1. The is_queue_vault tag identifies queue-venue positions.
    tagged = _StubPosition(value=500.0, other_data={IS_QUEUE_VAULT_KEY: True})
    directional = _StubPosition(value=999.0, other_data={})
    assert is_queue_vault(tagged) is True
    assert is_queue_vault(directional) is False

    # 2. Only tagged positions count toward the redeemable venue value.
    portfolio = _StubPortfolio(open_positions={1: tagged, 2: directional})
    assert queue_venue_redeemable(portfolio) == pytest.approx(500.0)

    # 3. The event round-trips through its JSON-primitive dict form.
    event = QueueVaultEvent(EVENT_PARK, 42, 1234.5, 7)
    assert QueueVaultEvent.from_dict(event.to_dict()) == event


def test_alpha_model_extracted_hooks():
    """The three PR-0 hooks preserve AlphaModel's base behaviour.

    1. _available_same_cycle_cash returns get_current_cash() unchanged.
    2. _count_position_in_old_weights keeps the existing bridge/credit/vault filters.
    3. _on_deposit_window_closed skips the buy and records the miss.
    """
    alpha_model = AlphaModel(timestamp=datetime.datetime(2024, 1, 1))

    # 1. The cash hook is a pass-through in the base class.
    assert alpha_model._available_same_cycle_cash(_StubPositionManager(cash=250.0)) == pytest.approx(250.0)

    # 2. The old-weight inclusion predicate keeps the current filters.
    spot = _StubPosition(pair=_StubPair())
    bridge = _StubPosition(pair=_StubPair(cctp=True))
    vault = _StubPosition(pair=_StubPair(), vault=True)
    assert alpha_model._count_position_in_old_weights(spot, ignore_credit=True, portfolio_pairs=None) is True
    assert alpha_model._count_position_in_old_weights(bridge, ignore_credit=True, portfolio_pairs=None) is False
    assert alpha_model._count_position_in_old_weights(vault, ignore_credit=True, portfolio_pairs=None) is False
    # ignore_credit=False keeps the vault position in (the reference-strategy path).
    assert alpha_model._count_position_in_old_weights(vault, ignore_credit=False, portfolio_pairs=None) is True

    # 3. A closed deposit window skips the buy and records the missed deposit.
    signal = _StubSignal(pair="vault-A", position_adjust_usd=1000.0)
    result = alpha_model._on_deposit_window_closed(signal, _StubPositionManager(cash=0.0))
    assert result is True
    assert signal.position_adjust_ignored is True
    assert TradingPairSignalFlags.cannot_deposit in signal.flags
    assert signal.other_data["missed_deposit_usd"] == pytest.approx(1000.0)


def test_queue_event_log_survives_state_serialisation():
    """Queue events survive an OtherData JSON serialise/reload (durability + reload recovery).

    1. Park two vaults on cycle 2 and cycle 10 (10 vs 2 exposes lexical key sorting).
    2. Promote one, then JSON-serialise and reload the OtherData (as a state save/restore does).
    3. read_open_park_events on the reloaded store returns only the still-open park.
    """
    other_data = OtherData()

    # 1. Park A (cycle 2) and B (cycle 10) — non-adjacent cycles crossing a 10-boundary.
    append_queue_event(other_data, QueueVaultEvent(EVENT_PARK, 10, 1000.0, 2))
    append_queue_event(other_data, QueueVaultEvent(EVENT_PARK, 20, 2000.0, 10))

    # 2. Promote A on cycle 10, then round-trip the store through JSON.
    append_queue_event(other_data, QueueVaultEvent(EVENT_PROMOTE, 10, 1000.0, 10))
    reloaded = OtherData.from_json(other_data.to_json())

    # 3. Only the still-open park (B, id 20) survives, read back correctly after reload.
    open_events = read_open_park_events(reloaded)
    assert set(open_events) == {20}
    assert open_events[20].usd == pytest.approx(2000.0)


def test_should_skip_calls_deposit_window_closed_hook():
    """_should_skip_signal_rebalance routes a closed deposit window through the hook (call-through).

    1. A positive-adjust signal whose vault reports can_deposit=False.
    2. _should_skip_signal_rebalance returns True (skip), delegating to _on_deposit_window_closed.
    3. The signal is flagged cannot_deposit with the missed deposit recorded.
    """
    alpha_model = AlphaModel(timestamp=datetime.datetime(2024, 1, 1))
    pm = _StubSkipPositionManager(pricing_model=_StubPricingModel(can_deposit_result=False))
    signal = _StubSignal(pair="vault-A", position_adjust_usd=1000.0)

    # 1-2. Deposits closed for a positive buy ⇒ skip via the hook.
    skipped = alpha_model._should_skip_signal_rebalance(
        signal,
        pm,
        frozen_pairs=set(),
        individual_rebalance_min_threshold=0.0,
        sell_rebalance_min_threshold=None,
    )
    assert skipped is True

    # 3. The hook recorded the miss at the real call site.
    assert signal.position_adjust_ignored is True
    assert TradingPairSignalFlags.cannot_deposit in signal.flags
    assert signal.other_data["missed_deposit_usd"] == pytest.approx(1000.0)


def test_queue_vault_signal_flags_exist():
    """The two PhaseAwareAlphaModel diagnostic flags exist with snake_case values.

    1. parked_in_queue_vault and promoted_from_queue_vault are members.
    2. Their values are snake_case strings equal to their names (repo enum convention).
    """
    # 1-2. Members exist with snake_case value == name.
    assert TradingPairSignalFlags.parked_in_queue_vault.value == "parked_in_queue_vault"
    assert TradingPairSignalFlags.promoted_from_queue_vault.value == "promoted_from_queue_vault"


def test_mark_queue_vault_tag():
    """mark_queue_vault tags a position so is_queue_vault recognises it.

    1. An untagged position is not a queue vault.
    2. After mark_queue_vault it is, and queue_venue_redeemable counts its value.
    """
    position = _StubPosition(value=750.0, other_data={})

    # 1. Untagged position is not a queue vault.
    assert is_queue_vault(position) is False

    # 2. Tagged position is recognised and its value is counted.
    mark_queue_vault(position)
    assert is_queue_vault(position) is True
    portfolio = _StubPortfolio(open_positions={1: position})
    assert queue_venue_redeemable(portfolio) == pytest.approx(750.0)


def test_next_open_getters():
    """TradingPairIdentifier surfaces deposit/redemption next-open times (diagnostics only).

    1. Build a vault-like pair and stamp the deposit next-open time in other_data.
    2. get_deposit_next_open returns that datetime.
    3. An absent redemption key returns None.
    """
    base = AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "VLT", 18, 1)
    quote = AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "USDC", 6, 2)
    pair = TradingPairIdentifier(
        base,
        quote,
        generate_random_ethereum_address(),
        generate_random_ethereum_address(),
        internal_id=999,
    )

    # 1-2. A stamped deposit next-open time is returned by the getter. The key
    # "deposit_next_open" is the one written by dex_data_translation.py; keep the
    # getter and that writer in sync if either key is renamed.
    deposit_open = datetime.datetime(2026, 2, 1, 18, 0)
    pair.other_data["deposit_next_open"] = deposit_open
    assert pair.get_deposit_next_open() == deposit_open

    # 3. An absent redemption next-open key returns None.
    assert pair.get_redemption_next_open() is None


def test_queue_vault_membership_identity():
    """queue_vault_pair_ids + membership identify venue positions, reload-safe, without the tag.

    1. queue_vault_pair_ids extracts the venue internal ids from a YieldRuleset.
    2. queue_venue_redeemable with those ids counts only the venue position, even
       though no is_queue_vault tag is set (the primary, reload-safe identity path).
    """
    venue_pos = _StubPosition(value=400.0, other_data={}, pair=_StubPair(internal_id=101))
    directional = _StubPosition(value=900.0, other_data={}, pair=_StubPair(internal_id=202))
    ruleset = _StubRuleset(weights=[_StubRule(pair=_StubPair(internal_id=101))])

    # 1. Extract venue ids from the ruleset.
    venue_ids = queue_vault_pair_ids(ruleset)
    assert venue_ids == {101}

    # 2. Membership counts only the venue position, with no tag set on it.
    portfolio = _StubPortfolio(open_positions={1: venue_pos, 2: directional})
    assert is_queue_vault(venue_pos) is False
    assert queue_venue_redeemable(portfolio, venue_ids) == pytest.approx(400.0)


def test_queue_vault_pair_ids_rejects_async_venue():
    """An async vault cannot be a queue venue - the precondition is enforced, not just documented.

    An async venue cannot release cash same-cycle, so the invariant-4 budget widening and the
    promotion funding would be built on cash that settles only cycles later.

    1. A ruleset whose venue reports is_async_vault() trips the assert.
    2. A synchronous venue passes.
    """
    # 1. Async venue -> rejected at wiring time.
    async_ruleset = _StubRuleset(weights=[_StubRule(pair=_StubPair(internal_id=101, async_vault=True))])
    with pytest.raises(AssertionError, match="must be synchronous"):
        queue_vault_pair_ids(async_ruleset)

    # 2. Sync venue -> accepted.
    sync_ruleset = _StubRuleset(weights=[_StubRule(pair=_StubPair(internal_id=101))])
    assert queue_vault_pair_ids(sync_ruleset) == {101}
