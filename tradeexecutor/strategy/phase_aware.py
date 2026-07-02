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
from tradeexecutor.strategy.alpha_model import AlphaModel, TradingPairSignal, TradingPairSignalFlags
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager


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

#: A held position's redemption is blocked (closed redemption window / lock-up) - diagnostic only.
#:
#: Its value is *redemption-locked*: the strategy could not exit this cycle even if it wanted to.
#: This covers both a wanted-but-blocked exit and the precautionary carry-forward pin on a
#: closed-window holding - once pinned the two are indistinguishable at the model level, and the
#: operator-relevant quantity is the same: how much of the book cannot be redeemed right now.
#: (In-flight async settlements are a separate concern, charted by ``pending_vault_settlements``.)
#:
#: The redemption side of the phase-aware design is *passive* (the settlement pin owns the
#: behaviour; the model adds no redemption logic), so these events change no trade decision -
#: they exist to make the redemption-locked buffer durable for the charts, mirroring how park
#: events make the waiting-deposit buffer durable.
EVENT_REDEEM_BLOCK = "redeem_block"

#: A previously-blocked redemption unblocked (the window opened / lock-up expired - the position
#: became redeemable again, or was exited).
EVENT_REDEEM_CLEAR = "redeem_clear"

#: Event kinds that close an open park event.
_CLOSING_KINDS = frozenset({EVENT_PROMOTE, EVENT_CLOSE})

#: Event kinds that close an open redeem-block event.
_REDEEM_CLOSING_KINDS = frozenset({EVENT_REDEEM_CLEAR})


@dataclasses.dataclass(slots=True, frozen=True)
class QueueVaultEvent:
    """A single event in the durable queue-venue event log.

    Deposit side: park / promote / close (drives behaviour - promotion detection and dedup).
    Redemption side: redeem-block / redeem-clear (diagnostic only - the settlement pin owns
    the behaviour; these make the waiting-redemption buffer durable for the charts).

    JSON-primitive by construction: a string kind, an int vault id, a float USD
    and an int cycle — no :py:class:`~tradeexecutor.state.identifier.TradingPairIdentifier`
    objects — so it round-trips through the state file.
    """

    #: One of :py:data:`EVENT_PARK`, :py:data:`EVENT_PROMOTE`, :py:data:`EVENT_CLOSE`,
    #: :py:data:`EVENT_REDEEM_BLOCK`, :py:data:`EVENT_REDEEM_CLEAR`.
    kind: str

    #: Internal id of the target vault the parked cash is destined for (deposit side),
    #: or the vault whose redemption is waiting (redemption side).
    vault_internal_id: int

    #: USD amount parked / promoted at event time (a diagnostic snapshot).
    usd: float

    #: Strategy cycle number the event was logged on.
    cycle: int

    #: ISO-8601 cycle timestamp the event was logged at, or ``None`` for older logs.
    #:
    #: The cycle number alone cannot be placed on a wall-clock time axis (backtest state carries no
    #: durable cycle->timestamp map), so charts read this. Optional and defaulted for backward
    #: compatibility with state files written before this field existed.
    timestamp: str | None = None

    def to_dict(self) -> dict:
        """Serialise to a JSON-primitive dict for ``state.other_data``."""
        return {
            "kind": self.kind,
            "vault_internal_id": int(self.vault_internal_id),
            "usd": float(self.usd),
            "cycle": int(self.cycle),
            "timestamp": self.timestamp,
        }

    @staticmethod
    def from_dict(data: dict) -> "QueueVaultEvent":
        """Deserialise from a ``state.other_data`` dict (``timestamp`` absent in older logs)."""
        return QueueVaultEvent(
            kind=data["kind"],
            vault_internal_id=int(data["vault_internal_id"]),
            usd=float(data["usd"]),
            cycle=int(data["cycle"]),
            timestamp=data.get("timestamp"),
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


def is_queue_vault_position(position: TradingPosition) -> bool:
    """Is this open/closed position a queue-venue (YieldManager-managed) holding, from state alone?

    Charts and post-run diagnostics have no ``YieldRuleset`` config (so :py:func:`queue_vault_pair_ids`
    is unreachable) and the :py:data:`IS_QUEUE_VAULT_KEY` tag has no natural writer, so this reads the
    durable marker YieldManager stamps on every venue trade: ``trade.other_data["yield_decision"]``
    (set only on the vault-venue adjust path, never on a directional buy). A position is a queue venue
    iff it is a vault and any of its trades carries that marker. The tag is still honoured as an
    override.
    """
    if is_queue_vault(position):
        return True
    if not position.is_vault():
        return False
    return any(trade.get_yield_decision() is not None for trade in position.trades.values())


def queue_vault_pair_ids(yield_rules) -> set:
    """Internal ids of the venues declared in a ``YieldRuleset``.

    Duck-typed on ``yield_rules.weights[i].pair`` to avoid importing YieldManager.
    This config-derived set is the **primary, reload-safe** way to identify
    queue-venue positions (see :py:func:`is_queue_vault` for why the tag path is
    only an override).

    Enforces the synchronous-venue precondition: an async (ERC-7540 / Lagoon /
    Ostium) venue cannot release cash same-cycle, so the invariant-4 budget
    widening and the promotion funding would both be built on cash that settles
    only cycles later (``NotEnoughMoney`` at execution, or stranded parks).
    Window-gated *deposit* schedules on a venue cannot be detected from the pair
    alone - keep the venue always-open (Gauntlet/Steakhouse-style deep sync USDC
    vaults; avoid D2/Gains-style epoch vaults).
    """
    pair_ids = set()
    for rule in yield_rules.weights:
        pair = rule.pair
        assert not pair.is_async_vault(), (
            f"Queue venue {pair} is an async vault: the queue venue must be synchronous "
            f"(instantly redeemable) so YieldManager can release its cash same-cycle to "
            f"fund promotions (invariant 4). Pick a sync ERC-4626 venue instead."
        )
        pair_ids.add(pair.internal_id)
    return pair_ids


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


def _read_open_events(
    other_data: OtherData,
    open_kind: str,
    closing_kinds: frozenset,
) -> dict[int, QueueVaultEvent]:
    """Fold the whole event history into the currently-open events of one kind.

    An ``open_kind`` event opens (or updates) a vault's entry; a later event in
    ``closing_kinds`` for the same vault closes it. Returns
    ``{vault_internal_id: latest open event}``.
    """
    open_events: dict[int, QueueVaultEvent] = {}
    for event in iter_all_events(other_data):
        if event.kind == open_kind:
            open_events[event.vault_internal_id] = event
        elif event.kind in closing_kinds:
            open_events.pop(event.vault_internal_id, None)
    return open_events


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
    return _read_open_events(other_data, EVENT_PARK, _CLOSING_KINDS)


def read_open_redeem_block_events(other_data: OtherData) -> dict[int, QueueVaultEvent]:
    """Reconstruct the currently-waiting blocked redemptions by folding the whole history.

    The redemption-side sibling of :py:func:`read_open_park_events`:
    a :py:data:`EVENT_REDEEM_BLOCK` opens a vault's waiting-redemption, a later
    :py:data:`EVENT_REDEEM_CLEAR` for the same vault closes it. Returns
    ``{vault_internal_id: latest open redeem-block event}``. Same full-history-fold
    rationale — never ``OtherData.load_latest``.
    """
    return _read_open_events(other_data, EVENT_REDEEM_BLOCK, _REDEEM_CLOSING_KINDS)


class PhaseAwareAlphaModel(AlphaModel):
    """AlphaModel that defers window-closed vault deposits into a yield-bearing queue venue.

    Instead of *skipping* a vault whose deposit window is closed, it **defers** the
    buy and logs a durable park event, so the capital waits in the queue venue and
    is deposited on a later cycle once the window opens. It reuses the overridable
    hooks extracted from :py:class:`AlphaModel` in PR-0 and the queue-venue helpers
    in this module. It is orthogonal to the allocation method (works with any
    ``normalise_weights`` variant): the phase-aware pass operates on ``self.signals``
    after targets are computed, whatever produced them.

    Cross-chain deposits compose with no phase-aware-specific code: a promoted buy into a
    satellite-chain vault is funded through the existing CCTP planner, provided the queue venue
    is on the primary/hub chain so a same-cycle venue release can bridge out. Robust cross-chain
    funding over long horizons (a single hub venue can be swept with cash a same-cycle satellite
    bridge still needs) is a chain-aware ``YieldManager`` follow-up.

    Wiring in ``decide_trades`` (after the usual set_signal / normalise /
    update_old_weights / calculate_target_positions sequence, and before
    ``generate_rebalance_trades_and_triggers``)::

        rules = create_yield_rules(...)
        alpha = PhaseAwareAlphaModel(
            timestamp,
            cycle=input.cycle,
            venue_pair_ids=queue_vault_pair_ids(rules),
        )
        # ... set_signal() for each candidate, then (order matters):
        alpha.carry_forward_non_redeemable_positions(position_manager)   # MUST precede select_top_signals
        alpha.select_top_signals(count=...)                              # so settlement pins reach self.signals
        alpha.assign_weights(...)
        alpha.normalise_weights(...)
        alpha.update_old_weights(state.portfolio, ignore_credit=False)   # venue excluded (inv. 2)
        alpha.calculate_target_positions(position_manager)
        alpha.apply_phase_aware_intent(position_manager)                 # park closed-window deposits; mark promotes
        trades = alpha.generate_rebalance_trades_and_triggers(...)       # cap sees venue cash (inv. 4)
        alpha.reconcile_phase_aware_events(position_manager, trades)     # finalise promotes that actually emitted
        # then the existing YieldManager two-step sweeps idle cash into the venue.

    The stale-close guard relies on settlement-pending pins reaching ``self.signals``,
    so ``carry_forward_non_redeemable_positions()`` must run **before**
    ``select_top_signals()`` (as the reference strategy does); otherwise an in-flight
    parked deposit could be wrongly stale-closed.

    See ``.claude/plans/phase-aware-alpha-model.md``.
    """

    def __init__(
        self,
        timestamp=None,
        *,
        cycle: int | None = None,
        venue_pair_ids: set | None = None,
        **kwargs,
    ):
        """
        :param cycle:
            Strategy cycle number, used as the durable event-log key. When ``None`` the
            phase-aware passes (:py:meth:`apply_phase_aware_intent` /
            :py:meth:`reconcile_phase_aware_events`) are inert and behaviour degrades to the base
            :py:class:`AlphaModel`: a closed-window deposit is *skipped*, not deferred (the base
            ``_on_deposit_window_closed`` path sets the ``cannot_deposit`` flag and records
            ``missed_deposit_usd``). Pass the real cycle number to enable park / deposit-on-open.

        :param venue_pair_ids:
            Internal ids of the queue-venue pairs (from
            :py:func:`queue_vault_pair_ids`). Used to (a) add their redeemable value
            to the same-cycle cash budget and (b) exclude them from old-weight
            accounting.
        """
        super().__init__(timestamp=timestamp, **kwargs)
        # This subclass is intentionally not slotted, so instances get a __dict__
        # for these transient per-cycle attributes (not part of the serialised state).
        self.phase_aware_cycle = cycle
        self.venue_pair_ids: set = set(venue_pair_ids) if venue_pair_ids else set()
        #: Vaults with an open park event whose window opened and stayed targeted this
        #: cycle. The promote event is finalised in
        #: :py:meth:`reconcile_phase_aware_events`, only once the deposit trade has
        #: actually been generated.
        self._promote_candidates: set = set()

    # -- Invariant 1: the queue venue is never an alpha candidate --
    def set_signal(self, pair, alpha, **kwargs):
        """Reject signals on queue-venue pairs - YieldManager owns every venue trade.

        A queue venue that slips into the candidate universe would otherwise be treated
        as a fresh directional buy (it is excluded from old-weight accounting, invariant 2)
        while YieldManager sells/rebalances the same pair in the same cycle - conflicting
        same-pair trades. Fail fast at signal time instead. This also fires if a venue
        position is carry-forward pinned (``carry_forward_non_redeemable_positions`` calls
        ``set_signal``), which would equally violate the venue's fully-liquid precondition.
        """
        assert pair.internal_id not in self.venue_pair_ids, (
            f"Queue-venue pair {pair} must not receive an alpha signal: "
            f"YieldManager owns all venue trades (invariant 1). "
            f"Remove the venue from the candidate universe / signal set."
        )
        return super().set_signal(pair, alpha, **kwargs)

    # -- Invariant 4: the same-cycle buy budget includes instantly-redeemable venue balance --
    def _available_same_cycle_cash(self, position_manager: PositionManager) -> USDollarAmount:
        base = super()._available_same_cycle_cash(position_manager)
        return base + queue_venue_redeemable(position_manager.state.portfolio, self.venue_pair_ids)

    # -- Invariant 2: exclude queue-venue positions from alpha old-weight accounting --
    def _count_position_in_old_weights(self, position, ignore_credit, portfolio_pairs) -> bool:
        if position.pair.internal_id in self.venue_pair_ids:
            return False
        return super()._count_position_in_old_weights(position, ignore_credit, portfolio_pairs)

    # -- Phase 1 (park): defer closed-window deposits BEFORE trade generation --
    def apply_phase_aware_intent(self, position_manager: PositionManager) -> None:
        """Park closed-window deposits and classify open park events, before trade generation.

        Call after :py:meth:`calculate_target_positions` and before
        :py:meth:`generate_rebalance_trades_and_triggers`. Parking here (rather than
        during trade generation) removes the deferred buys from the whole-portfolio
        min-trade gate and the same-cycle cash cap, which would otherwise count them
        toward - or scale them to zero within - the buy budget. Promotions are only
        *marked* here; they are finalised in :py:meth:`reconcile_phase_aware_events`
        once the deposit has actually emitted.

        For each fresh positive deposit signal (carry-forward pins - settlement-pending
        / non-redeemable - are left to the base model):

        - **window closed** -> defer: zero the adjustment, flag ``parked_in_queue_vault``,
          record ``parked_usd``, and log a park event.
        - **window open, with an existing open park event** -> a promotion candidate.

        Then each still-open park event that was neither re-parked nor promoted this
        cycle, and is not held by an in-flight settlement pin, is stale-closed.
        """
        if self.phase_aware_cycle is None:
            return
        other_data = position_manager.state.other_data
        pricing_model = position_manager.pricing_model
        open_events = read_open_park_events(other_data)
        self._promote_candidates = set()

        # 1. Park every fresh positive deposit whose window is closed.
        for vault_id, signal in self.signals.items():
            if signal.carry_forward_position or signal.position_adjust_usd <= 0:
                continue
            if pricing_model.can_deposit(self.timestamp, signal.pair):
                if vault_id in open_events:
                    self._promote_candidates.add(vault_id)
            else:
                self._park_signal(other_data, signal, open_events.get(vault_id))

        # 2. Stale-close open park events whose vault is no longer a live parked deposit.
        for vault_id in open_events:
            if vault_id in self._promote_candidates:
                continue
            signal = self.signals.get(vault_id)
            if signal is not None and (
                signal.carry_forward_position
                or TradingPairSignalFlags.parked_in_queue_vault in signal.flags
            ):
                # in-flight settlement pin, or re-parked this cycle -> keep the event open
                continue
            self._log_phase_aware_event(other_data, EVENT_CLOSE, vault_id, 0.0)

    # -- Phase 2 (deposit-on-open): finalise promotions AFTER trade generation --
    def reconcile_phase_aware_events(self, position_manager: PositionManager, trades) -> None:
        """Log a promote event only for candidates whose deposit trade actually emitted.

        Call after :py:meth:`generate_rebalance_trades_and_triggers`. A promotion
        candidate whose buy was suppressed (min-trade threshold, dust, or a whole-cycle
        cancellation) keeps its park event open to retry on a later cycle, rather than
        closing it with no deposit.

        Also records the cycle's blocked redemptions to the durable event log
        (:py:meth:`_reconcile_redemption_waits`) - diagnostic only, no trade decision
        reads those events.
        """
        if self.phase_aware_cycle is None:
            return
        other_data = position_manager.state.other_data
        # Diagnostic redemption-wait events first: they must be recorded even on a cycle
        # with no promotion candidates.
        self._reconcile_redemption_waits(other_data)
        if not self._promote_candidates:
            return
        # A vault deposit long is built on synthetic_pair == pair (map_pair_for_signal
        # returns the underlying for signal > 0), so trade.pair.internal_id matches the
        # signal/vault key used for the promote candidates.
        deposited = {t.pair.internal_id for t in trades if t.is_buy() and t.pair is not None}
        for vault_id in self._promote_candidates:
            if vault_id not in deposited:
                continue  # buy suppressed -> leave the park event open, retry next cycle
            signal = self.signals.get(vault_id)
            if signal is not None:
                signal.flags.add(TradingPairSignalFlags.promoted_from_queue_vault)
            usd = signal.position_adjust_usd if signal is not None else 0.0
            self._log_phase_aware_event(other_data, EVENT_PROMOTE, vault_id, usd)
        self._promote_candidates = set()

    def _reconcile_redemption_waits(self, other_data: OtherData) -> None:
        """Mirror this cycle's redemption-locked value into the durable event log (diagnostic only).

        The redemption side is *passive* - the settlement pin and the redemption checks own
        the behaviour, and they record ``missed_redemption_usd`` on the cycle's signals
        (``_mark_signal_cannot_redeem``: the carry-forward pin, the skip-rebalance path and
        the reduce path all funnel there). Signals are per-cycle, so this scan makes the
        redemption-locked buffer durable for the charts: a signal carrying
        ``missed_redemption_usd`` opens (or updates) a :py:data:`EVENT_REDEEM_BLOCK`; an open
        block whose vault no longer reports one this cycle - the window opened / lock-up
        expired, or the position is gone - is closed with :py:data:`EVENT_REDEEM_CLEAR`.
        Re-blocking at an unchanged amount appends nothing, to bound the log over long runs.
        """
        open_blocks = read_open_redeem_block_events(other_data)
        blocked_now: dict[int, float] = {}
        for vault_id, signal in self.signals.items():
            usd = signal.other_data.get("missed_redemption_usd")
            if usd:
                blocked_now[vault_id] = float(usd)

        for vault_id, usd in blocked_now.items():
            existing = open_blocks.get(vault_id)
            if existing is None or existing.usd != usd:
                self._log_phase_aware_event(other_data, EVENT_REDEEM_BLOCK, vault_id, usd)

        for vault_id in open_blocks:
            if vault_id not in blocked_now:
                self._log_phase_aware_event(other_data, EVENT_REDEEM_CLEAR, vault_id, 0.0)

    def _park_signal(self, other_data: OtherData, signal: TradingPairSignal, existing_event=None) -> None:
        """Defer a closed-window deposit into the queue venue and log a park event.

        Re-parking a vault whose open park event already records the same amount does
        not append a duplicate event, to bound the durable log over long runs.
        """
        signal.flags.add(TradingPairSignalFlags.parked_in_queue_vault)
        usd = float(signal.position_adjust_usd)
        signal.other_data["parked_usd"] = signal.position_adjust_usd
        if existing_event is None or existing_event.usd != usd:
            self._log_phase_aware_event(other_data, EVENT_PARK, signal.pair.internal_id, usd)
        # Zero the deferred adjustment so it is excluded from the min-trade gate, the
        # same-cycle cash cap, and trade generation (the base model then skips it).
        signal.position_adjust_usd = 0.0
        signal.position_adjust_quantity = 0.0
        signal.position_adjust_ignored = True

    def _log_phase_aware_event(self, other_data: OtherData, kind: str, vault_internal_id: int, usd: float) -> None:
        """Append a park/promote/close event to the durable log (no-op without a cycle)."""
        if self.phase_aware_cycle is None:
            return
        append_queue_event(
            other_data,
            QueueVaultEvent(
                kind,
                int(vault_internal_id),
                float(usd),
                self.phase_aware_cycle,
                timestamp=self.timestamp.isoformat() if self.timestamp is not None else None,
            ),
        )
