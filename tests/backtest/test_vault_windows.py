"""Unit tests for backtest vault deposit/redemption window modelling.

Covers three layers of :py:mod:`tradeexecutor.backtest.vault_windows` and its
wiring into :py:class:`~tradeexecutor.backtest.backtest_pricing.BacktestPricing`:

1. :py:class:`VaultWindowSchedule` — the periodic open/closed function is correct,
   look-ahead-free, periodic in both directions, and validates its config.
2. :py:func:`get_assumed_open_close_time` — the layered resolver picks an explicit
   per-vault override over a protocol-default cadence, and returns ``None`` otherwise.
3. ``BacktestPricing`` override precedence — a window override beats the real
   (possibly stale / always-open) ``vault_state`` for both ``can_deposit`` and
   ``check_redemption``, while non-overridden pairs still answer from ``vault_state``.
"""
import datetime

import pandas as pd
import pytest

from tradeexecutor.backtest.backtest_pricing import BacktestPricing
from tradeexecutor.backtest.vault_windows import (
    VaultWindowSchedule,
    get_assumed_open_close_time,
)
from tradeexecutor.strategy.redemption import RedemptionBlockReason
from tradingstrategy.candle import GroupedCandleUniverse


# A 3-day window open at the start of every 30-day epoch, anchored 2026-03-01.
# Open:   [03-01, 03-04)  and every anchor + n*30d (also before the anchor).
# Closed: [03-04, 03-31), so 03-15 is deep inside a closed stretch.
ANCHOR = datetime.datetime(2026, 3, 1)
SCHEDULE = VaultWindowSchedule(
    cadence=datetime.timedelta(days=30),
    open_duration=datetime.timedelta(days=3),
    anchor=ANCHOR,
)


# ---------------------------------------------------------------------------
# 1. VaultWindowSchedule
# ---------------------------------------------------------------------------


def test_window_open_closed_reopen():
    """The window is a periodic open/closed function of the timestamp.

    1. Open at the anchor and within ``open_duration``.
    2. Closed on the half-open boundary once ``open_duration`` has elapsed.
    3. Re-open exactly one ``cadence`` later.
    4. An ``open_duration == cadence`` schedule (permitted by ``__post_init__``) is always open.
    """
    # 1. Open at anchor and one day in (phase 0d and 1d < 3d).
    assert SCHEDULE.is_open_at(ANCHOR) is True
    assert SCHEDULE.is_open_at(ANCHOR + datetime.timedelta(days=1)) is True

    # 2. Closed on the exact boundary (phase == open_duration is closed: the interval is
    #    half-open) and deep inside the closed stretch (phase 14d >= 3d).
    assert SCHEDULE.is_open_at(ANCHOR + datetime.timedelta(days=3)) is False
    assert SCHEDULE.is_open_at(datetime.datetime(2026, 3, 15)) is False

    # 3. Re-open one cadence later (phase back to 0d).
    assert SCHEDULE.is_open_at(ANCHOR + datetime.timedelta(days=30)) is True
    assert SCHEDULE.is_open_at(ANCHOR + datetime.timedelta(days=31)) is True

    # 4. open_duration == cadence -> the whole epoch is one open window.
    always_open = VaultWindowSchedule(
        cadence=datetime.timedelta(days=30),
        open_duration=datetime.timedelta(days=30),
        anchor=ANCHOR,
    )
    assert always_open.is_open_at(ANCHOR) is True
    assert always_open.is_open_at(datetime.datetime(2026, 3, 15)) is True
    assert always_open.is_open_at(ANCHOR + datetime.timedelta(days=29, hours=23)) is True


def test_window_periodic_before_anchor():
    """The window is periodic in both directions, so timestamps before the anchor work.

    1. A ts one day into the pre-anchor window (anchor - cadence) is open.
    2. A ts deep in the pre-anchor closed stretch is closed.
    3. The same holds for a ``pd.Timestamp`` — negative-modulo interop is the risky edge.
    """
    # 1. anchor - 30d = 2026-01-30 opens a window; +1d into it is open.
    assert SCHEDULE.is_open_at(datetime.datetime(2026, 1, 31)) is True
    # 2. 2026-02-27 is far from any window start -> closed.
    assert SCHEDULE.is_open_at(datetime.datetime(2026, 2, 27)) is False
    # 3. pd.Timestamp before the anchor: (Timestamp - datetime) is a negative Timedelta,
    #    and % timedelta must still yield a non-negative phase (what BacktestPricing feeds).
    assert SCHEDULE.is_open_at(pd.Timestamp("2026-01-31")) is True
    assert SCHEDULE.is_open_at(pd.Timestamp("2026-02-27")) is False


def test_window_deposit_redemption_aliases_and_pandas_ts():
    """Deposit/redemption aliases mirror ``is_open_at`` and accept ``pd.Timestamp``.

    1. Both aliases equal ``is_open_at`` for open and closed instants.
    2. A ``pd.Timestamp`` (what BacktestPricing feeds) gives the same answer.
    """
    open_ts = datetime.datetime(2026, 3, 2)
    closed_ts = datetime.datetime(2026, 3, 15)

    # 1. Aliases mirror the base predicate.
    assert SCHEDULE.is_deposit_open(open_ts) is True
    assert SCHEDULE.is_redemption_open(open_ts) is True
    assert SCHEDULE.is_deposit_open(closed_ts) is False
    assert SCHEDULE.is_redemption_open(closed_ts) is False

    # 2. pandas Timestamp interop (Timestamp - datetime -> Timedelta % timedelta).
    assert SCHEDULE.is_open_at(pd.Timestamp("2026-03-02")) is True
    assert SCHEDULE.is_open_at(pd.Timestamp("2026-03-15")) is False


def test_window_validation():
    """The dataclass rejects nonsensical cadence / open_duration at construction.

    1. Non-positive cadence is rejected.
    2. open_duration outside (0, cadence] is rejected.
    """
    # 1. Zero/negative cadence.
    with pytest.raises(AssertionError):
        VaultWindowSchedule(
            cadence=datetime.timedelta(0),
            open_duration=datetime.timedelta(days=1),
            anchor=ANCHOR,
        )
    # 2. open_duration longer than cadence.
    with pytest.raises(AssertionError):
        VaultWindowSchedule(
            cadence=datetime.timedelta(days=3),
            open_duration=datetime.timedelta(days=30),
            anchor=ANCHOR,
        )


# ---------------------------------------------------------------------------
# 2. get_assumed_open_close_time resolver
# ---------------------------------------------------------------------------


class _FakeVault:
    def __init__(self, internal_id: int, protocol: str | None):
        self.internal_id = internal_id
        self._protocol = protocol

    def get_vault_protocol(self) -> str | None:
        return self._protocol


def test_resolver_override_wins_over_protocol():
    """An explicit per-vault override beats a matching protocol cadence.

    1. Both an override (by internal_id) and a protocol cadence match the vault.
    2. The override schedule is returned.
    """
    protocol_schedule = VaultWindowSchedule(
        cadence=datetime.timedelta(days=7),
        open_duration=datetime.timedelta(days=1),
        anchor=ANCHOR,
    )
    vault = _FakeVault(internal_id=42, protocol="d2")

    # 1. + 2. Override registered for id 42 also matches protocol "d2" -> override wins.
    resolved = get_assumed_open_close_time(
        vault,
        overrides={42: SCHEDULE},
        protocol_cadences={"d2": protocol_schedule},
    )
    assert resolved is SCHEDULE


def test_resolver_protocol_fallback_and_none():
    """The resolver falls back to protocol cadence, then to ``None``.

    1. No override but a matching protocol cadence -> protocol schedule.
    2. An override alone (no protocol_cadences supplied) still resolves.
    3. Neither an override nor a protocol match -> ``None``.
    """
    # 1. No override, protocol "d2" matches.
    vault = _FakeVault(internal_id=7, protocol="d2")
    assert get_assumed_open_close_time(vault, protocol_cadences={"d2": SCHEDULE}) is SCHEDULE

    # 2. Override alone, protocol_cadences omitted -> override resolves (no None-key crash).
    assert get_assumed_open_close_time(vault, overrides={7: SCHEDULE}) is SCHEDULE

    # 3. Nothing applies (unknown protocol, empty maps) -> None.
    other = _FakeVault(internal_id=7, protocol="morpho")
    assert get_assumed_open_close_time(other, protocol_cadences={"d2": SCHEDULE}) is None
    assert get_assumed_open_close_time(other) is None


# ---------------------------------------------------------------------------
# 3. BacktestPricing override precedence
# ---------------------------------------------------------------------------


class _FakePair:
    def __init__(self, internal_id: int):
        self.internal_id = internal_id
        self.pool_address = f"0x{internal_id:040x}"

    def get_ticker(self) -> str:
        return f"VAULT{self.internal_id}-USDC"


def _candle_universe() -> GroupedCandleUniverse:
    candles = pd.DataFrame(
        [{"pair_id": 1, "timestamp": pd.Timestamp("2026-03-01"), "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 0}]
    ).set_index("timestamp", drop=False)
    return GroupedCandleUniverse(candles)


def _always_closed_state() -> pd.DataFrame:
    """A vault_state that marks both pairs 1 and 2 permanently closed on 2026-03-01.

    Used to prove precedence in both directions: an *open* override must still allow a
    pair the frame closes, and a *non-overridden* pair must still be blocked by the frame.
    """
    rows = [
        {
            "pair_id": pair_id,
            "address": f"0x{pair_id:040x}",
            "timestamp": pd.Timestamp("2026-03-01"),
            "deposits_open": False,
            "redemption_open": False,
            "deposit_closed_reason": "closed in data",
            "redemption_closed_reason": "closed in data",
            "max_deposit": float("nan"),
            "max_redeem": float("nan"),
        }
        for pair_id in (1, 2)
    ]
    df = pd.DataFrame(rows)
    df["deposits_open"] = df["deposits_open"].astype("boolean")
    df["redemption_open"] = df["redemption_open"].astype("boolean")
    return df


@pytest.fixture
def pricing() -> BacktestPricing:
    # Pair 1 has a window override; pair 2 does not. vault_state closes both.
    return BacktestPricing(
        _candle_universe(),
        routing_model=None,
        data_delay_tolerance=pd.Timedelta("2d"),
        vault_state=_always_closed_state(),
        vault_window_overrides={1: SCHEDULE},
    )


def test_override_beats_vault_state_can_deposit(pricing):
    """A deposit window override overrides the (closed) ``vault_state`` in both directions.

    1. Override open (03-02) allows a deposit the frame would block.
    2. Override closed (03-15) blocks it.
    3. A non-overridden pair still answers from ``vault_state`` (blocked).
    """
    # 1. Override says open -> allowed, despite vault_state deposits_open=False.
    assert pricing.can_deposit(pd.Timestamp("2026-03-02"), _FakePair(1)) is True
    # 2. Override says closed -> blocked.
    assert pricing.can_deposit(pd.Timestamp("2026-03-15"), _FakePair(1)) is False
    # 3. Pair 2 has no override -> falls through to vault_state (closed).
    assert pricing.can_deposit(pd.Timestamp("2026-03-02"), _FakePair(2)) is False


def test_override_beats_vault_state_check_redemption(pricing: BacktestPricing):
    """A redemption window override overrides the (closed) ``vault_state`` in both directions.

    1. Override open (03-02) allows redemption the frame would block.
    2. Override closed (03-15) blocks it with the window-closed reason and zeroed caps.
    3. A non-overridden pair still answers from ``vault_state`` (blocked).
    """
    # 1. Override open -> can_redeem True despite vault_state redemption_open=False.
    assert pricing.check_redemption(pd.Timestamp("2026-03-02"), _FakePair(1)).can_redeem is True

    # 2. Override closed -> blocked, with the distinct window-closed reason and zeroed caps.
    blocked = pricing.check_redemption(pd.Timestamp("2026-03-15"), _FakePair(1))
    assert blocked.can_redeem is False
    assert blocked.reason_code == RedemptionBlockReason.redemption_window_closed
    assert blocked.max_redemption == 0.0
    assert blocked.max_withdrawable == 0.0

    # 3. Pair 2 has no override -> falls through to vault_state (closed).
    assert pricing.check_redemption(pd.Timestamp("2026-03-02"), _FakePair(2)).can_redeem is False


def test_override_none_timestamp_allows(pricing: BacktestPricing):
    """With no decision timestamp the window cannot be evaluated, so the vault is allowed.

    1. ``can_deposit(None, ...)`` is True even though the frame closes the pair.
    2. ``check_redemption(None, ...)`` is allowed for the same reason.
    """
    # 1. + 2. ts=None -> override present but unevaluable -> default allow (not the frame's block).
    assert pricing.can_deposit(None, _FakePair(1)) is True
    assert pricing.check_redemption(None, _FakePair(1)).can_redeem is True


def _always_open_state() -> pd.DataFrame:
    """A vault_state marking both pairs 1 and 2 OPEN at 2026-03-15.

    Mirror of :py:func:`_always_closed_state`. The sample sits exactly on the query
    timestamp so a non-overridden pair genuinely reads an *open* frame there (not an
    out-of-tolerance "unknown -> allowed"), letting us prove a *closed* override still
    blocks while the frame alone would allow.
    """
    rows = [
        {
            "pair_id": pair_id,
            "address": f"0x{pair_id:040x}",
            "timestamp": pd.Timestamp("2026-03-15"),
            "deposits_open": True,
            "redemption_open": True,
            "deposit_closed_reason": None,
            "redemption_closed_reason": None,
            "max_deposit": float("nan"),
            "max_redeem": float("nan"),
        }
        for pair_id in (1, 2)
    ]
    df = pd.DataFrame(rows)
    df["deposits_open"] = df["deposits_open"].astype("boolean")
    df["redemption_open"] = df["redemption_open"].astype("boolean")
    return df


def test_override_closed_beats_open_vault_state():
    """A *closed* window override blocks even when the real ``vault_state`` reports open.

    1. The override-closed pair (03-15) is blocked for both deposit and redemption,
       beating a frame that reports the vault open at that exact timestamp.
    2. A non-overridden pair reads that same open frame and stays allowed.
    """
    pricing = BacktestPricing(
        _candle_universe(),
        routing_model=None,
        data_delay_tolerance=pd.Timedelta("2d"),
        vault_state=_always_open_state(),
        vault_window_overrides={1: SCHEDULE},
    )
    # 1. Override closed at 03-15 -> blocked, despite the open frame at 03-15.
    assert pricing.can_deposit(pd.Timestamp("2026-03-15"), _FakePair(1)) is False
    assert pricing.check_redemption(pd.Timestamp("2026-03-15"), _FakePair(1)).can_redeem is False
    # 2. Pair 2 has no override -> genuinely reads the open frame -> allowed.
    assert pricing.can_deposit(pd.Timestamp("2026-03-15"), _FakePair(2)) is True
    assert pricing.check_redemption(pd.Timestamp("2026-03-15"), _FakePair(2)).can_redeem is True
