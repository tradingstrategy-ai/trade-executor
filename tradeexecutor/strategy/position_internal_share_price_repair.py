"""Rebuild the internal share price of an exchange-account position.

An exchange-account position's internal share price measures trading
results only when every capital flow is booked as one: a transfer trade, or a
balance update applied with :py:func:`apply_capital_outflow`. Before
``lagoon-lighter-test-trade`` booked its transfers, its withdrawal to the
Safe arrived as an ordinary valuation balance update, so the share price
reported the withdrawn principal as a trading loss. On lighter-ai this turned
an 11% loss into 95%.

This module replays such a position's history with the live share-price
functions, treating chosen balance updates as withdrawals, and rewrites the
stored share-price state and position statistics. Withdrawals are found from
the test-trade journal kept next to the state file, or named by the operator.
The ``rebuild-exchange-account-share-price`` command drives it.

The replay is only trusted when, without the new withdrawals, it reproduces the
stored state and every stored statistics sample. That proves it applies events
in the order and with the formulas the live executor used.
"""
import dataclasses
import datetime
import json
import logging
import math
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from tradeexecutor.state.balance_update import BalanceUpdate, BalanceUpdateCause
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.position_internal_share_price import PositionInternalSharePriceState
from tradeexecutor.state.state import State
from tradeexecutor.state.statistics import PositionStatistics
from tradeexecutor.state.trade import TradeExecution, TradeFlag
from tradeexecutor.strategy.position_internal_share_price import (
    apply_capital_outflow,
    create_share_price_state,
    create_share_price_state_for_exchange_account,
    update_share_price_state,
    update_share_price_state_for_balance_update,
)

logger = logging.getLogger(__name__)


#: ``BalanceUpdate.other_data`` key marking a balance update as a capital flow.
#:
#: Written by the repair so that a later rebuild applies the same
#: classification, which makes the repair idempotent.
SHARE_PRICE_CAPITAL_FLOW_KEY = "share_price_capital_flow"

#: Journal phases after the test trade's withdrawal reached the Safe.
LIGHTER_TEST_TRADE_WITHDRAWN_PHASES = frozenset({"withdrawn", "complete"})

#: A single valuation update moving the share price more than this is listed
#: as a suspected unbooked capital flow. Never applied automatically, because
#: a liquidation looks the same.
SUSPECT_SHARE_PRICE_MOVE = 0.5

#: Statistics are taken after a cycle's execution, so a sample can include
#: events up to this long after its timestamp.
STATISTICS_EXECUTION_WINDOW = datetime.timedelta(hours=1)

#: Notes prefixes of balance updates that only revalue an exchange account.
#: Written by ``ExchangeAccountValuator`` and ``ExchangeAccountSyncModel``.
VALUATION_NOTES_PREFIXES = ("Exchange account valuation:", "Exchange account PnL sync:")

_REL_TOL = 1e-9
_ABS_TOL = 1e-6


class SharePriceRepairError(Exception):
    """The repair cannot be planned safely; nothing was changed."""


@dataclass(slots=True)
class ReplayEntry:
    """The share-price state after one replayed position event."""

    #: Event timestamp: ``executed_at`` of a trade, ``block_mined_at`` of a balance update.
    at: datetime.datetime

    #: ``"trade"`` or ``"balance_update"``.
    kind: str

    #: Trade id or balance update id.
    event_id: int

    #: Position quantity after the event, from the Decimal books.
    quantity: Decimal

    #: Share-price state after the event; ``None`` before any capital.
    state: PositionInternalSharePriceState | None


@dataclass(slots=True)
class Suspect:
    """A valuation balance update that alone moved the share price a lot."""

    balance_update_id: int
    at: datetime.datetime
    quantity: Decimal
    share_price_before: float
    share_price_after: float


@dataclass(slots=True)
class SharePriceRepair:
    """A planned rebuild of one position, ready to apply."""

    position_id: int

    #: Balance update id to the reason it is applied as a withdrawal.
    withdrawals: dict[int, str]

    stored_state: PositionInternalSharePriceState
    rebuilt_state: PositionInternalSharePriceState

    #: Rewritten statistics samples, aligned with the stored list.
    samples: list[PositionStatistics]
    samples_changed: int

    suspects: list[Suspect] = field(default_factory=list)

    def is_noop(self) -> bool:
        """Applying would change nothing."""
        return self.samples_changed == 0 and _same_state(self.stored_state, self.rebuilt_state) and self.stored_state.total_withdrawn == self.rebuilt_state.total_withdrawn


def _close(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=_REL_TOL, abs_tol=_ABS_TOL)


def _same_state(a: PositionInternalSharePriceState, b: PositionInternalSharePriceState) -> bool:
    return all(
        _close(getattr(a, name), getattr(b, name))
        for name in ("current_share_price", "total_supply", "cumulative_quantity", "total_invested")
    )


def is_valuation_balance_update(balance_update: BalanceUpdate) -> bool:
    """Is this an exchange-account revaluation rather than a booked flow."""
    return balance_update.cause == BalanceUpdateCause.vault_flow and (balance_update.notes or "").startswith(VALUATION_NOTES_PREFIXES)


def get_marked_withdrawal_ids(position: TradingPosition) -> set[int]:
    """Balance updates an earlier repair marked as withdrawals."""
    return {
        bu.balance_update_id
        for bu in position.balance_updates.values()
        if (bu.other_data or {}).get(SHARE_PRICE_CAPITAL_FLOW_KEY) == "withdrawal"
    }


def get_funding_principal_ids(state: State) -> set[int]:
    """Source-account balance updates booked as funding principal.

    :py:func:`tradeexecutor.exchange_account.state.complete_exchange_account_funding`
    books the fee, then the principal, and records both ids on the funding
    trade. The principal went through :py:func:`apply_capital_outflow`.
    """
    principal_ids = set()
    for position in state.portfolio.get_all_positions():
        for trade in position.trades.values():
            booked = (trade.other_data or {}).get("funding_source_balance_update_ids")
            if booked:
                principal_ids.add(int(booked[-1]))
    return principal_ids


def _is_funding_fee(balance_update: BalanceUpdate) -> bool:
    return (balance_update.notes or "").startswith("Exchange-account funding trade #") and (balance_update.notes or "").endswith(": exchange fee")


def _is_custody_transfer(trade: TradeExecution) -> bool:
    """Was this trade completed through ``State.mark_trade_success()``."""
    return TradeFlag.external_account_transfer in (trade.flags or set()) or trade.is_internal_account_transfer()


def replay_exchange_account_share_price(
    position: TradingPosition,
    withdrawal_ids: set[int],
    funding_principal_ids: set[int] = frozenset(),
) -> list[ReplayEntry]:
    """Replay a position's share price from its trades and balance updates.

    Events are applied in time order with the functions the live executor
    used for each, so without extra withdrawals the result reproduces the
    stored state.

    :param position:
        Open exchange-account position.
    :param withdrawal_ids:
        Balance updates to apply as capital withdrawals with
        :py:func:`apply_capital_outflow`.
    :param funding_principal_ids:
        Balance updates booked as funding principal, also capital outflows.
    :return:
        One entry per event, in replay order.
    :raise SharePriceRepairError:
        If a balance update is of a kind this replay does not model.
    """
    events: list[tuple[datetime.datetime, int, int, TradeExecution | BalanceUpdate]] = []
    for trade in position.trades.values():
        if trade.is_success():
            events.append((trade.executed_at, 0, trade.trade_id, trade))
    for bu in position.balance_updates.values():
        events.append((bu.block_mined_at, 1, bu.balance_update_id, bu))
    events.sort(key=lambda e: (e[0], e[1], e[2]))

    entries = []
    state: PositionInternalSharePriceState | None = None
    quantity = Decimal(0)
    for at, _order, event_id, event in events:
        if isinstance(event, TradeExecution):
            trade = event
            quantity += trade.executed_quantity or Decimal(0)
            if _is_custody_transfer(trade):
                # State.mark_trade_success()
                if state is None:
                    state = create_share_price_state(trade)
                else:
                    state = update_share_price_state(state, trade, remaining_quantity=quantity)
            else:
                # open_exchange_account_position(): a placeholder seeds no state
                reserve_amount = float(trade.executed_reserve or 0)
                if state is not None:
                    if reserve_amount > 0:
                        state = update_share_price_state(state, trade)
                elif reserve_amount > 1:
                    state = create_share_price_state(trade)
            entries.append(ReplayEntry(at, "trade", event_id, quantity, state))
            continue

        bu = event
        quantity += bu.quantity
        if bu.balance_update_id in withdrawal_ids or bu.balance_update_id in funding_principal_ids:
            if state is None:
                raise SharePriceRepairError(f"Position #{position.position_id}: withdrawal balance update #{event_id} precedes any capital")
            state = apply_capital_outflow(state, float(-bu.quantity), at, remaining_quantity=quantity)
        elif is_valuation_balance_update(bu) or _is_funding_fee(bu):
            if state is None:
                # ExchangeAccountValuator: the first positive value is the starting capital
                if quantity > 0:
                    state = create_share_price_state_for_exchange_account(float(quantity), at)
            else:
                state = update_share_price_state_for_balance_update(state, bu)
        else:
            raise SharePriceRepairError(
                f"Position #{position.position_id}: balance update #{event_id} ({bu.cause.name}: {bu.notes!r}) is not a valuation, funding or withdrawal update"
            )
        entries.append(ReplayEntry(at, "balance_update", event_id, quantity, state))
    return entries


def _sample_figures(
    value: float,
    state: PositionInternalSharePriceState,
    use_withdrawn: bool,
) -> dict[str, float]:
    """Share-price statistics as ``TradingPosition.get_share_price_profit()`` computes them.

    Exchange-account quantity is valued at a price of 1.0.
    """
    supply = state.total_supply
    share_price = value / supply if supply > 0 else state.current_share_price
    profit_pct = share_price / state.initial_share_price - 1
    if use_withdrawn:
        profit_usd = value + state.total_withdrawn - state.total_invested
    elif supply > 0:
        profit_usd = value - supply * state.initial_share_price
    else:
        raise SharePriceRepairError("Cannot recompute a closed share series; only open positions are supported")
    return {
        "internal_share_price": share_price,
        "internal_total_supply": supply,
        "internal_profit_pct": profit_pct,
        "internal_profit_usd": profit_usd,
        "profitability": profit_pct,
        "profit_usd": profit_usd,
    }


def _match_sample(
    sample: PositionStatistics,
    entries: list[ReplayEntry],
    start: int,
) -> int | None:
    """Find the replay entry a stored sample was computed from.

    A sample reads the position's quantity and share supply when statistics
    run, after the cycle's execution, so it is matched by content: share
    supply and value. Of the matching entries from ``start``, take the latest
    no later than the sample's timestamps allow, else the first.
    """
    bound = max(sample.calculated_at, sample.last_valuation_at) + STATISTICS_EXECUTION_WINDOW
    first = None
    best = None
    for index in range(start, len(entries)):
        entry = entries[index]
        if entry.state is None:
            continue
        if entry.at > bound and first is not None:
            break
        if _close(entry.state.total_supply, sample.internal_total_supply) and _close(float(entry.quantity), sample.value):
            if first is None:
                first = index
            if entry.at <= bound:
                best = index
    return best if best is not None else first


def plan_share_price_repair(
    state: State,
    position: TradingPosition,
    withdrawals: dict[int, str],
) -> SharePriceRepair:
    """Plan the rebuild of one open exchange-account position.

    :param state:
        Strategy state, not modified.
    :param position:
        Open exchange-account position.
    :param withdrawals:
        Balance update ids to apply as withdrawals, with the reason for each.
        Balance updates already marked by an earlier repair are added.
    :return:
        The planned repair.
    :raise SharePriceRepairError:
        If the replay does not reproduce the stored history, or an input is wrong.
    """
    pid = position.position_id
    if not position.is_exchange_account() or pid not in state.portfolio.open_positions:
        raise SharePriceRepairError(f"Position #{pid} is not an open exchange-account position")
    stored = position.share_price_state
    if stored is None:
        raise SharePriceRepairError(f"Position #{pid} has no share-price state")

    marked = get_marked_withdrawal_ids(position)
    for bu_id in withdrawals:
        bu = position.balance_updates.get(bu_id)
        if bu is None:
            raise SharePriceRepairError(f"Balance update #{bu_id} does not belong to position #{pid}")
        if not is_valuation_balance_update(bu) and bu_id not in marked:
            raise SharePriceRepairError(f"Balance update #{bu_id} is not a valuation update: {bu.notes!r}")
        if bu.quantity >= 0:
            raise SharePriceRepairError(f"Balance update #{bu_id} is not negative ({bu.quantity}), so it cannot be a withdrawal")

    funding_principal_ids = get_funding_principal_ids(state)
    uncorrected = replay_exchange_account_share_price(position, marked, funding_principal_ids)
    applied = marked | set(withdrawals)
    corrected = replay_exchange_account_share_price(position, applied, funding_principal_ids)

    # Faithfulness: the replay without the new withdrawals is the live history
    final_uncorrected = uncorrected[-1].state if uncorrected else None
    if final_uncorrected is None or not _same_state(final_uncorrected, stored):
        raise SharePriceRepairError(
            f"Position #{pid}: replay does not reproduce the stored share-price state "
            f"(stored {stored}, replayed {final_uncorrected}); refusing to rebuild"
        )

    rebuilt = dataclasses.replace(corrected[-1].state)

    # The stored withdrawn total was unknown when the samples were taken,
    # if it is unknown now: it never changes from None.
    stored_used_withdrawn = stored.total_withdrawn is not None
    stored_samples = state.stats.positions.get(pid, [])
    new_samples = []
    changed = 0
    cursor = 0
    for sample in stored_samples:
        if sample.internal_share_price is None:
            new_samples.append(sample)
            continue
        index = _match_sample(sample, uncorrected, cursor)
        if index is None:
            raise SharePriceRepairError(
                f"Position #{pid}: statistics sample at {sample.calculated_at} "
                f"(supply {sample.internal_total_supply}, value {sample.value}) matches no replayed event"
            )
        cursor = index
        expected = _sample_figures(sample.value, uncorrected[index].state, stored_used_withdrawn)
        for name, value in expected.items():
            if not _close(getattr(sample, name), value):
                raise SharePriceRepairError(
                    f"Position #{pid}: statistics sample at {sample.calculated_at} has {name}={getattr(sample, name)}, "
                    f"the replay gives {value}"
                )
        figures = _sample_figures(sample.value, corrected[index].state, True)
        new_sample = dataclasses.replace(sample, **figures)
        if any(not _close(getattr(sample, name), value) for name, value in figures.items()):
            changed += 1
        new_samples.append(new_sample)

    suspects = []
    for before, after in zip(uncorrected, uncorrected[1:]):
        if after.kind != "balance_update" or after.event_id in applied or before.state is None or after.state is None:
            continue
        old_price = before.state.current_share_price
        new_price = after.state.current_share_price
        if old_price > 0 and abs(new_price / old_price - 1) > SUSPECT_SHARE_PRICE_MOVE:
            bu = position.balance_updates[after.event_id]
            if is_valuation_balance_update(bu):
                suspects.append(Suspect(after.event_id, after.at, bu.quantity, old_price, new_price))

    reasons = {bu_id: "marked by an earlier repair" for bu_id in marked}
    reasons.update(withdrawals)
    return SharePriceRepair(
        position_id=pid,
        withdrawals=reasons,
        stored_state=stored,
        rebuilt_state=rebuilt,
        samples=new_samples,
        samples_changed=changed,
        suspects=suspects,
    )


def apply_share_price_repair(state: State, repair: SharePriceRepair) -> None:
    """Write a planned repair into the state.

    Changes only the position's share-price state, the statistics samples and
    the marker on each applied balance update. Quantities, trades, balance
    amounts and reserves are untouched.
    """
    position = state.portfolio.open_positions[repair.position_id]
    equity_before = state.portfolio.calculate_total_equity()
    quantity_before = position.get_quantity()
    for bu_id in repair.withdrawals:
        bu = position.balance_updates[bu_id]
        bu.other_data = {**(bu.other_data or {}), SHARE_PRICE_CAPITAL_FLOW_KEY: "withdrawal"}
    position.share_price_state = repair.rebuilt_state
    if repair.position_id in state.stats.positions:
        state.stats.positions[repair.position_id] = repair.samples
    assert position.get_quantity() == quantity_before
    assert state.portfolio.calculate_total_equity() == equity_before


def find_lighter_test_trade_journals(state_file: Path, executor_id: str) -> list[Path]:
    """Test-trade journals kept next to the state file.

    ``lagoon-lighter-test-trade`` defaults its journal to
    ``<state folder>/<executor-id>.lighter-test-trade.json``. A rerun needs a
    new journal path, which operators keep under the same prefix.
    """
    return sorted(state_file.parent.glob(f"{executor_id}.lighter-test-trade*.json"))


def detect_test_trade_withdrawals(
    state: State,
    journal_files: list[Path],
) -> tuple[dict[int, dict[int, str]], list[str]]:
    """Find test-trade withdrawals booked as valuation updates.

    A journal past its withdrawal names the claimed amount and the Lighter
    account. The withdrawal is that account's one valuation balance update of
    exactly minus that amount, at or after the request.

    :return:
        Position id to ``{balance update id: reason}``, and report lines.
    :raise SharePriceRepairError:
        A journal for another vault, an ambiguous match, or a misbooked deposit.
    """
    found: dict[int, dict[int, str]] = {}
    report = []
    vault_address = (state.sync.deployment.address or "").lower()
    for path in journal_files:
        journal = json.loads(path.read_text())
        name = path.name
        if vault_address and journal.get("vault_address", "").lower() != vault_address:
            raise SharePriceRepairError(f"Journal {name} belongs to vault {journal.get('vault_address')}, not {vault_address}")
        if journal.get("phase") not in LIGHTER_TEST_TRADE_WITHDRAWN_PHASES or "withdrawal_usdc" not in journal:
            report.append(f"{name}: phase {journal.get('phase')!r} has no completed withdrawal, skipped")
            continue
        account_index = int(journal["account_index"])
        positions = [
            p for p in state.portfolio.open_positions.values()
            if p.is_exchange_account() and int(p.pair.get_exchange_account_id()) == account_index
        ]
        if len(positions) != 1:
            report.append(f"{name}: no single open position for Lighter account {account_index}, skipped")
            continue
        position = positions[0]
        if "withdrawal_trade_id" in journal:
            report.append(f"{name}: withdrawal booked as trade #{journal['withdrawal_trade_id']}, nothing to repair")
            continue
        _check_journal_deposit(state, position, journal, name)
        amount = Decimal(str(journal["withdrawal_usdc"]))
        requested_at = datetime.datetime.fromtimestamp(int(journal["withdrawal_requested_at"]), datetime.timezone.utc).replace(tzinfo=None)
        matches = [
            bu for bu in position.balance_updates.values()
            if bu.quantity == -amount and bu.block_mined_at >= requested_at
            and (is_valuation_balance_update(bu) or bu.balance_update_id in get_marked_withdrawal_ids(position))
        ]
        if not matches:
            report.append(f"{name}: no balance update of -{amount} on position #{position.position_id} after {requested_at}, skipped")
            continue
        if len(matches) > 1:
            ids = ", ".join(f"#{bu.balance_update_id}" for bu in matches)
            raise SharePriceRepairError(
                f"{name}: withdrawal of {amount} matches balance updates {ids} on position #{position.position_id}; "
                "choose one with --capital-withdrawal-balance-update-id and --ignore-balance-update-id"
            )
        bu = matches[0]
        found.setdefault(position.position_id, {})[bu.balance_update_id] = f"journal {name}: test-trade withdrawal of {amount} USDC"
        report.append(f"{name}: position #{position.position_id} balance update #{bu.balance_update_id} is the test-trade withdrawal of {amount} USDC")
    return found, report


def _check_journal_deposit(state: State, position: TradingPosition, journal: dict, name: str) -> None:
    """Refuse a test-trade deposit that was booked as profit.

    The deposit is correct capital when the share-price state did not exist
    yet: the first valuation after it seeds the state. If the position already
    had capital when the test trade started, the deposit raised the share
    price instead. This repair does not reclassify deposits.
    """
    if "deposit_trade_id" in journal or position.share_price_state is None:
        return
    created_at = datetime.datetime.fromisoformat(journal["created_at"]).replace(tzinfo=None)
    entries = replay_exchange_account_share_price(position, get_marked_withdrawal_ids(position), get_funding_principal_ids(state))
    seeded_at = next((entry.at for entry in entries if entry.state is not None), None)
    if seeded_at is not None and seeded_at < created_at:
        raise SharePriceRepairError(
            f"{name}: the test-trade deposit went into position #{position.position_id}, which already had capital "
            f"from {seeded_at}; it was booked as profit, which this repair does not fix"
        )
