"""Backtest vault deposit/redemption window modelling.

Some vaults (e.g. D2 / HYPE++, Ostium) only accept deposits or redemptions for a
few days per epoch. The historical ``vault_state`` availability frame may not
capture those windows (it starts only when the scanner began recording, and may
mark a vault always-open), so a backtest cannot exercise the park -> deposit-on-open
path without a synthetic schedule.

:py:class:`VaultWindowSchedule` is that synthetic schedule, and
:py:func:`get_assumed_open_close_time` resolves one for a vault from a layered
config. When wired into :py:class:`~tradeexecutor.backtest.backtest_pricing.BacktestPricing`
as a *window override*, it takes precedence over the real ``vault_state`` (so a
stale/always-open frame cannot stop the assumed schedule firing).

Everything here is **look-ahead-free**: a schedule is a pure periodic function of
the decision timestamp and the vault's own config, never of future price data.

See ``.claude/plans/phase-aware-alpha-model.md``.
"""
import dataclasses
import datetime


@dataclasses.dataclass(frozen=True, slots=True)
class VaultWindowSchedule:
    """A periodic deposit/redemption open window for a vault (backtest modelling).

    The window is open for ``open_duration`` at the start of every ``cadence``
    period measured from ``anchor`` (e.g. a 3-day window every 30-day epoch). It is
    a pure function of the timestamp, so it introduces no look-ahead.
    """

    #: Length of one epoch (e.g. 30 days for a monthly vault).
    cadence: datetime.timedelta

    #: How long the window stays open at the start of each epoch.
    open_duration: datetime.timedelta

    #: Start of one open window; all other windows are ``anchor + n * cadence``.
    anchor: datetime.datetime

    def __post_init__(self):
        assert self.cadence > datetime.timedelta(0), f"cadence must be positive: {self.cadence}"
        assert datetime.timedelta(0) < self.open_duration <= self.cadence, \
            f"open_duration must be in (0, cadence]: {self.open_duration} vs {self.cadence}"

    def is_open_at(self, ts: datetime.datetime) -> bool:
        """Whether the window is open at ``ts``.

        ``(ts - anchor) % cadence`` is a non-negative remainder even for timestamps
        before the anchor, so the schedule is periodic in both directions.
        """
        phase = (ts - self.anchor) % self.cadence
        return phase < self.open_duration

    # Direction-specific aliases: same window for both here, but distinct call sites
    # let a future schedule gate deposits and redemptions independently.
    def is_deposit_open(self, ts: datetime.datetime) -> bool:
        """Whether the deposit window is open at ``ts``."""
        return self.is_open_at(ts)

    def is_redemption_open(self, ts: datetime.datetime) -> bool:
        """Whether the redemption window is open at ``ts``."""
        return self.is_open_at(ts)


def get_assumed_open_close_time(
    vault,
    vault_universe=None,
    *,
    overrides: dict[int, VaultWindowSchedule] | None = None,
    protocol_cadences: dict[str, VaultWindowSchedule] | None = None,
) -> VaultWindowSchedule | None:
    """Resolve an assumed deposit/redemption window schedule for ``vault``.

    Layered, first hit wins:

    1. **Explicit per-vault override** - ``overrides`` keyed by ``vault.internal_id``.
    2. **Protocol-default cadence** - ``protocol_cadences`` keyed by the vault's
       protocol slug (``vault.get_vault_protocol()``), e.g. ``{"d2": <30d schedule>}``.

    Returns ``None`` when nothing applies (the caller then treats the vault as
    unwindowed). Real per-timestamp ``vault_state`` is *not* a schedule and is not
    resolved here - :py:class:`BacktestPricing` applies it separately, and a schedule
    returned here, passed as a *window override*, deliberately takes precedence over
    it. Share-price-spike inference is a deferred layer; ``vault_universe`` is accepted
    for it and is unused today.
    """
    if overrides and vault.internal_id in overrides:
        return overrides[vault.internal_id]
    if protocol_cadences:
        protocol = vault.get_vault_protocol()
        if protocol is not None and protocol in protocol_cadences:
            return protocol_cadences[protocol]
    return None
