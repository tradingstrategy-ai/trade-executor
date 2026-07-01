"""Phase-aware alpha model helpers.

Pure, dependency-light functions shared by ``PhaseAwareAlphaModel``, the
diagnostics charts, and tests, so there is a single source of truth for the
queue-venue event log, venue identification, and venue redeemable value.

See ``.claude/plans/phase-aware-alpha-model.md``.

The event log is stored durably in
:py:class:`tradeexecutor.state.other_data.OtherData` keyed by strategy cycle. It
**must** be read back with :py:func:`read_open_park_events`, which folds the whole
history — never ``OtherData.load_latest``, which returns only the most recent
cycle that stored anything and would silently drop open park events on any cycle
that recorded nothing (the framework writes bookkeeping to ``other_data`` every
cycle, so a "quiet" queue cycle still exists in the store without the queue key).
"""
import dataclasses
from typing import Iterable

from tradeexecutor.state.other_data import OtherData
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.types import USDollarAmount


#: ``state.other_data`` key under which the park/promote event log is stored.
QUEUE_VAULT_EVENT_LOG_KEY = "phase_aware_queue_events"

#: ``position.other_data`` flag marking a position as a queue-venue holding.
IS_QUEUE_VAULT_KEY = "is_queue_vault"

#: Cash was deferred into the queue venue for a vault whose deposit window is closed.
EVENT_PARK = "park"

#: Parked cash was deposited into the target vault once its window opened.
EVENT_PROMOTE = "promote"

#: A park was abandoned without depositing (the vault is no longer targeted).
EVENT_CLOSE = "close"

#: Event kinds that close an open park event.
_CLOSING_KINDS = frozenset({EVENT_PROMOTE, EVENT_CLOSE})


@dataclasses.dataclass(slots=True, frozen=True)
class QueueVaultEvent:
    """A single park / promote / close event in the queue-venue event log.

    JSON-primitive by construction: a string kind, an int vault id, a float USD
    and an int cycle — no :py:class:`~tradeexecutor.state.identifier.TradingPairIdentifier`
    objects — so it round-trips through the state file.
    """

    #: One of :py:data:`EVENT_PARK`, :py:data:`EVENT_PROMOTE`, :py:data:`EVENT_CLOSE`.
    kind: str

    #: Internal id of the target vault the parked cash is destined for.
    vault_internal_id: int

    #: USD amount parked / promoted at event time (a diagnostic snapshot).
    usd: float

    #: Strategy cycle number the event was logged on.
    cycle: int

    def to_dict(self) -> dict:
        """Serialise to a JSON-primitive dict for ``state.other_data``."""
        return {
            "kind": self.kind,
            "vault_internal_id": int(self.vault_internal_id),
            "usd": float(self.usd),
            "cycle": int(self.cycle),
        }

    @staticmethod
    def from_dict(data: dict) -> "QueueVaultEvent":
        """Deserialise from a ``state.other_data`` dict."""
        return QueueVaultEvent(
            kind=data["kind"],
            vault_internal_id=int(data["vault_internal_id"]),
            usd=float(data["usd"]),
            cycle=int(data["cycle"]),
        )


def is_queue_vault(position: TradingPosition) -> bool:
    """Is this position a queue-venue (YieldManager-managed) holding?

    Reads the :py:data:`IS_QUEUE_VAULT_KEY` tag in ``other_data``. The tag has no
    natural writer — YieldManager writes ``trade.other_data`` (not
    ``position.other_data``) and PhaseAwareAlphaModel must not touch venue
    positions — so :py:func:`queue_vault_pair_ids` membership is the preferred,
    reload-safe identity path; this tag is an optional override.
    """
    return bool(position.other_data.get(IS_QUEUE_VAULT_KEY))


def mark_queue_vault(position: TradingPosition) -> None:
    """Optionally tag a position as a queue-venue holding (override path).

    The preferred identity is :py:func:`queue_vault_pair_ids` membership
    (config-derived, reload-safe); this tag is available if a caller wants to mark
    a specific position explicitly. Idempotent.
    """
    position.other_data[IS_QUEUE_VAULT_KEY] = True


def queue_vault_pair_ids(yield_rules) -> set:
    """Internal ids of the venues declared in a ``YieldRuleset``.

    Duck-typed on ``yield_rules.weights[i].pair`` to avoid importing YieldManager.
    This config-derived set is the **primary, reload-safe** way to identify
    queue-venue positions (see :py:func:`is_queue_vault` for why the tag path is
    only an override).
    """
    return {rule.pair.internal_id for rule in yield_rules.weights}


def queue_venue_redeemable(
    portfolio: Portfolio,
    venue_pair_ids: set | None = None,
) -> USDollarAmount:
    """Instantly-redeemable USD value held in queue-venue positions.

    This is the balance the same-cycle cash cap adds to ``get_current_cash()``: a
    synchronous venue redeems same-cycle, so YieldManager can release this to fund
    a promotion buy in the same cycle.

    Venue positions are identified by ``YieldRuleset`` membership when
    ``venue_pair_ids`` is given (from :py:func:`queue_vault_pair_ids`; reload-safe,
    preferred), otherwise by the :py:func:`is_queue_vault` tag.
    """
    total: USDollarAmount = 0.0
    for position in portfolio.open_positions.values():
        if venue_pair_ids is not None:
            is_venue = position.pair.internal_id in venue_pair_ids
        else:
            is_venue = is_queue_vault(position)
        if is_venue:
            total += position.get_value()
    return total


def append_queue_event(other_data: OtherData, event: QueueVaultEvent) -> None:
    """Append one event to the durable log, in ``event.cycle``'s slot.

    Read back with :py:func:`read_open_park_events` (a full-history fold), never
    ``OtherData.load_latest``.
    """
    existing = other_data.data.get(event.cycle, {}).get(QUEUE_VAULT_EVENT_LOG_KEY, [])
    other_data.save(event.cycle, QUEUE_VAULT_EVENT_LOG_KEY, list(existing) + [event.to_dict()])


def iter_all_events(other_data: OtherData) -> Iterable[QueueVaultEvent]:
    """Yield every logged event across all cycles, in cycle order.

    Cycles are sorted by their integer value (``key=int``) so ordering is correct
    both for the in-memory int-keyed store and after a JSON state round-trip,
    which turns the dict's keys into strings (``"10"`` would otherwise sort before
    ``"2"`` lexically).
    """
    data = other_data.data or {}
    for cycle in sorted(data, key=int):
        for raw in data[cycle].get(QUEUE_VAULT_EVENT_LOG_KEY, []) or []:
            yield QueueVaultEvent.from_dict(raw)


def read_open_park_events(other_data: OtherData) -> dict[int, QueueVaultEvent]:
    """Reconstruct the currently-open park events by folding the whole history.

    A :py:data:`EVENT_PARK` opens a target vault's waiting-deposit; a later
    :py:data:`EVENT_PROMOTE` or :py:data:`EVENT_CLOSE` for the same vault closes
    it. Returns ``{vault_internal_id: latest open park event}``.

    This deliberately folds the full ``other_data.data`` history rather than
    calling ``OtherData.load_latest`` (which returns only the most recent cycle
    that stored anything): because the framework writes bookkeeping to
    ``other_data`` every cycle, ``load_latest`` would return ``None`` on any cycle
    that logged no queue event and silently drop every open park event — breaking
    dedup, promotion detection, and reload recovery.
    """
    open_events: dict[int, QueueVaultEvent] = {}
    for event in iter_all_events(other_data):
        if event.kind == EVENT_PARK:
            open_events[event.vault_internal_id] = event
        elif event.kind in _CLOSING_KINDS:
            open_events.pop(event.vault_internal_id, None)
    return open_events
