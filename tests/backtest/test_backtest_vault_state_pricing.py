"""Unit tests for historical deposit/redemption availability in BacktestPricing.

The backtest pricing model answers ``can_deposit`` / ``check_redemption`` / ``get_max_*`` from a
per-(vault, timestamp) availability frame so the alpha model can skip impossible rebalances.

Critical semantics under test:

- explicit ``deposits_open=False`` (or ``max_deposit==0``) -> blocked
- explicit ``deposits_open=True`` -> allowed
- unknown / NA / pre-history / out-of-tolerance / missing pair / no state frame -> allowed for
  generic vaults, but unavailable post-cutoff HyperCore deposits fail closed
- no look-ahead: a sample stamped strictly after the decision timestamp is never used
"""
import datetime
from decimal import Decimal

import pandas as pd
import pytest
from eth_defi.erc_4626.core import ERC4626Feature

from tradeexecutor.backtest.backtest_pricing import BacktestPricing
from tradeexecutor.backtest.vault_windows import VaultWindowSchedule
from tradeexecutor.strategy.redemption import DepositBlockReason, DepositCheckStage
from tradingstrategy.candle import GroupedCandleUniverse


class _FakePair:
    def __init__(self, internal_id: int, features=None, async_vault=False, protocol=None, hypercore=False):
        self.internal_id = internal_id
        self.pool_address = f"0x{internal_id:040x}"
        self._features = features
        self._async_vault = async_vault
        self._protocol = protocol
        self._hypercore = hypercore

    def get_ticker(self) -> str:
        return f"VAULT{self.internal_id}-USDC"

    def get_vault_features(self):
        return self._features

    def is_async_vault(self) -> bool:
        return self._async_vault

    def get_vault_protocol(self) -> str | None:
        return self._protocol

    def is_hyperliquid_vault(self) -> bool:
        return self._hypercore


def _candle_universe() -> GroupedCandleUniverse:
    candles = pd.DataFrame(
        [{"pair_id": 1, "timestamp": pd.Timestamp("2026-03-01"), "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 0}]
    ).set_index("timestamp", drop=False)
    return GroupedCandleUniverse(candles)


def _vault_state() -> pd.DataFrame:
    """Daily state for pair 1 (deposits) and pair 2 (redemptions)."""
    rows = [
        # pair 1 — deposits toggle open -> closed -> reopened
        _state_row(1, "2026-03-05", deposits_open=True),
        _state_row(1, "2026-03-06", deposits_open=False, deposit_closed_reason="Vault deposits disabled by leader"),
        _state_row(1, "2026-03-07", deposits_open=False, deposit_closed_reason="Vault deposits disabled by leader"),
        _state_row(1, "2026-03-10", deposits_open=True),
        # pair 1 — unknown deposits but explicit zero hard cap on this day
        _state_row(1, "2026-03-11", deposits_open=None, max_deposit=0.0),
        # pair 2 — redemptions closed on 03-06
        _state_row(2, "2026-03-05", redemption_open=None),
        _state_row(2, "2026-03-06", redemption_open=False, redemption_closed_reason="Redemptions paused", max_redeem=0.0),
        _state_row(2, "2026-03-07", redemption_open=True, max_redeem=1234.0),
    ]
    df = pd.DataFrame(rows)
    df["deposits_open"] = df["deposits_open"].astype("boolean")
    df["redemption_open"] = df["redemption_open"].astype("boolean")
    return df


def _state_row(pair_id, day, deposits_open=None, redemption_open=None, deposit_closed_reason=None, redemption_closed_reason=None, max_deposit=float("nan"), max_redeem=float("nan")):
    return {
        "pair_id": pair_id,
        "address": f"0x{pair_id:040x}",
        "timestamp": pd.Timestamp(day),
        "deposits_open": deposits_open,
        "redemption_open": redemption_open,
        "deposit_closed_reason": deposit_closed_reason,
        "redemption_closed_reason": redemption_closed_reason,
        "max_deposit": max_deposit,
        "max_redeem": max_redeem,
    }


@pytest.fixture
def pricing() -> BacktestPricing:
    return BacktestPricing(
        _candle_universe(),
        routing_model=None,
        data_delay_tolerance=pd.Timedelta("2d"),
        vault_state=_vault_state(),
    )


def test_can_deposit_open(pricing):
    assert pricing.can_deposit(pd.Timestamp("2026-03-05"), _FakePair(1)) is True


def test_can_deposit_closed(pricing):
    assert pricing.can_deposit(pd.Timestamp("2026-03-06"), _FakePair(1)) is False


def test_can_deposit_backfill_within_closed_stretch(pricing):
    # No exact sample at 03-07 12:00 -> backward-fill to the 03-07 (closed) sample.
    assert pricing.can_deposit(pd.Timestamp("2026-03-07 12:00"), _FakePair(1)) is False


def test_can_deposit_reopened(pricing):
    assert pricing.can_deposit(pd.Timestamp("2026-03-10"), _FakePair(1)) is True


def test_can_deposit_zero_hard_cap_blocks(pricing):
    # deposits_open unknown but max_deposit == 0 -> blocked.
    assert pricing.can_deposit(pd.Timestamp("2026-03-11"), _FakePair(1)) is False


def test_check_deposit_records_historical_closed_reason(pricing):
    result = pricing.check_deposit(
        pd.Timestamp("2026-03-06"),
        _FakePair(1),
        stage=DepositCheckStage.buy_rebalance,
    )
    assert result.can_deposit is False
    assert result.reason_code == DepositBlockReason.vault_deposits_closed
    assert result.message == "Vault deposits disabled by leader"
    assert result.max_deposit is None


def test_check_deposit_records_zero_hard_cap(pricing):
    result = pricing.check_deposit(pd.Timestamp("2026-03-11"), _FakePair(1))
    assert result.can_deposit is False
    assert result.reason_code == DepositBlockReason.vault_max_deposit_zero
    assert result.max_deposit == 0.0


def test_hypercore_cutoff_assumes_open_before_enforcing_archived_state():
    """Apply the HyperCore cutoff while retaining explicit backtest overrides.

    1. Build HyperCore state spanning the assumption boundary and an explicit window override.
    2. Check that pre-cutoff closure is ignored, while fresh open and closed state are honoured.
    3. Assert that the explicit window override still takes precedence.
    """
    state = pd.DataFrame([
        _state_row(3, "2026-04-10", deposits_open=False, deposit_closed_reason="closed before cutoff"),
        _state_row(3, "2026-04-11", deposits_open=True),
        _state_row(3, "2026-04-12", deposits_open=False, deposit_closed_reason="closed after cutoff"),
    ])
    hypercore_pair = _FakePair(3, hypercore=True)
    override = VaultWindowSchedule(
        cadence=datetime.timedelta(days=3),
        open_duration=datetime.timedelta(days=2),
        anchor=datetime.datetime(2026, 4, 11),
    )
    pricing_with_state = BacktestPricing(
        _candle_universe(),
        routing_model=None,
        data_delay_tolerance=pd.Timedelta("2d"),
        vault_state=state,
        vault_window_overrides={3: override},
    )
    pricing_without_override = BacktestPricing(
        _candle_universe(),
        routing_model=None,
        data_delay_tolerance=pd.Timedelta("2d"),
        vault_state=state,
    )

    # 1-2. The override is tested separately; archived state is tested without it.
    assert pricing_without_override.can_deposit(pd.Timestamp("2026-04-10"), hypercore_pair) is True
    assert pricing_without_override.can_deposit(pd.Timestamp("2026-04-11"), hypercore_pair) is True
    closed = pricing_without_override.check_deposit(pd.Timestamp("2026-04-12"), hypercore_pair)
    assert closed.can_deposit is False
    assert closed.reason_code == DepositBlockReason.vault_deposits_closed
    assert closed.message == "closed after cutoff"

    # 3. The explicit schedule beats archived state, including the cutoff rule.
    assert pricing_with_state.can_deposit(pd.Timestamp("2026-04-12"), hypercore_pair) is True


def test_hypercore_open_permission_with_zero_policy_capacity_blocks_buy():
    """Backtest admission matches the live low-share safety policy.

    1. Supply an explicitly open HyperCore observation with zero policy cap.
    2. Verify the result is a capacity block rather than a false closure.
    """
    # 1. Producer records permission and amount policy in separate fields.
    pair = _FakePair(7, hypercore=True)
    state = pd.DataFrame([_state_row(7, "2026-09-23", deposits_open=True, max_deposit=0.0)])
    pricing_with_state = BacktestPricing(
        _candle_universe(),
        routing_model=None,
        data_delay_tolerance=pd.Timedelta("2d"),
        vault_state=state,
    )

    # 2. A zero policy cap blocks new capital without claiming source closure.
    result = pricing_with_state.check_deposit(pd.Timestamp("2026-09-24"), pair)
    assert result.can_deposit is False
    assert result.reason_code == DepositBlockReason.vault_max_deposit_zero
    assert result.max_deposit == pytest.approx(0.0)
    assert pricing_with_state.get_max_deposit(pd.Timestamp("2026-09-24"), pair) == Decimal(0)


def test_hypercore_missing_state_fails_closed_after_cutoff_but_other_pairs_do_not():
    """Fail closed only for unavailable post-cutoff HyperCore deposit state.

    1. Build pricing with no state for a HyperCore pair and an equivalent generic pair.
    2. Check missing, nullable, and stale HyperCore observations after the cutoff.
    3. Assert non-HyperCore unknown-state behaviour and redemption behaviour remain unchanged.
    """
    nullable_state = pd.DataFrame([
        _state_row(4, "2026-04-11", deposits_open=None),
        _state_row(5, "2026-04-08", deposits_open=True),
    ])
    pricing_with_state = BacktestPricing(
        _candle_universe(),
        routing_model=None,
        data_delay_tolerance=pd.Timedelta("2d"),
        vault_state=nullable_state,
    )
    hypercore_pair = _FakePair(4, hypercore=True)
    stale_pair = _FakePair(5, hypercore=True)
    generic_pair = _FakePair(6)

    # 1-2. Missing, nullable, and stale HyperCore state block new deposits.
    for pair, timestamp in [
        (hypercore_pair, "2026-04-11"),
        (stale_pair, "2026-04-12"),
    ]:
        result = pricing_with_state.check_deposit(pd.Timestamp(timestamp), pair)
        assert result.can_deposit is False
        assert result.reason_code == DepositBlockReason.unknown
        assert "missing, unknown, or stale" in (result.message or "")

    # 3. Generic unknown state remains allowed and redemption is unaffected.
    assert pricing_with_state.can_deposit(pd.Timestamp("2026-04-12"), generic_pair) is True
    assert pricing_with_state.check_redemption(pd.Timestamp("2026-04-12"), generic_pair).can_redeem is True


def test_historical_settlement_event_uses_naive_utc_timestamp():
    """Settlement evidence must be found from a naive UTC decision timestamp.

    1. Create one settlement observation at a non-midnight timestamp.
    2. Query just before the observation and confirm that no future event leaks.
    3. Query at the observation and confirm that it is returned without applying
       the machine's local timezone offset.
    """
    event_at = datetime.datetime(2026, 3, 6, 0, 30)
    pricing_with_event = BacktestPricing(
        _candle_universe(),
        routing_model=None,
        vault_state=pd.DataFrame([{
            "timestamp": event_at,
            "pair_id": 1,
            "address": "0x0000000000000000000000000000000000000001",
            "vault_settlement_at": event_at,
        }]),
    )

    # 1-2. The historical lookup must not see a later same-day event.
    pair = _FakePair(1)
    assert pricing_with_event.get_vault_settlement_event_at(event_at - pd.Timedelta(seconds=1), pair) is None

    # 3. The event is visible exactly at its naive UTC timestamp.
    assert pricing_with_event.get_vault_settlement_event_at(event_at, pair) == event_at


@pytest.mark.parametrize(
    ("pair", "description"),
    [
        (_FakePair(1, features={ERC4626Feature.lagoon_like}), "Lagoon"),
        (_FakePair(1, features={ERC4626Feature.morpho_v2_like}), "Morpho V2"),
        (_FakePair(1, features={ERC4626Feature.yearn_v3_like}), "Yearn V3"),
    ],
)
def test_zero_max_deposit_does_not_close_protocols_where_it_is_not_capacity(pricing, pair, description):
    """Treat advisory maximum-function zeros as unknown capacity, not a closure."""
    result = pricing.check_deposit(pd.Timestamp("2026-03-11"), pair)

    assert pricing.can_deposit(pd.Timestamp("2026-03-11"), pair) is True, description
    assert result.can_deposit is True
    assert result.reason_code is None
    assert result.max_deposit is None


@pytest.mark.parametrize(
    "pair",
    [
        _FakePair(1, features={ERC4626Feature.lagoon_like}),
        _FakePair(1, features={ERC4626Feature.morpho_v2_like}),
        _FakePair(1, features={ERC4626Feature.yearn_v3_like}),
    ],
)
def test_incomplete_protocol_history_never_closes_deposits(pricing, pair):
    """Keep designated protocols open even when historical flags say closed."""
    result = pricing.check_deposit(pd.Timestamp("2026-03-06"), pair)

    assert result.can_deposit is True
    assert result.reason_code is None


@pytest.mark.parametrize("protocol", ["lagoon-finance", "yearn"])
def test_unclassified_protocol_history_never_closes_deposits(pricing, protocol):
    """Apply the historical guard to legacy Lagoon and Yearn snapshots.

    1. Create a pair from a snapshot without classified ERC-4626 features.
    2. Give it a legacy protocol slug whose historical admission data is unreliable.
    3. Verify that a closed marker does not manufacture a deposit closure.
    """
    # 1. + 2. A legacy Lagoon/Yearn pair lacks a reliable protocol-specific admission field.
    pair = _FakePair(1, features=None, protocol=protocol)

    # 3. The generic closed marker is advisory for this explicit compatibility fallback.
    assert pricing.can_deposit(pd.Timestamp("2026-03-06"), pair) is True


def test_can_deposit_pre_history_allowed(pricing):
    # Before the first sample -> unknown -> allowed.
    assert pricing.can_deposit(pd.Timestamp("2026-03-04"), _FakePair(1)) is True


def test_can_deposit_out_of_tolerance_allowed(pricing):
    # Last pair-1 sample is 03-11; 03-25 is well beyond the 2d tolerance -> allowed.
    assert pricing.can_deposit(pd.Timestamp("2026-03-25"), _FakePair(1)) is True


def test_can_deposit_no_look_ahead(pricing):
    # At 03-05 23:00 the only sample at-or-before is 03-05 (open). The 03-06 (closed)
    # sample is in the future and must NOT be used.
    assert pricing.can_deposit(pd.Timestamp("2026-03-05 23:00"), _FakePair(1)) is True


def test_unknown_pair_allowed(pricing):
    assert pricing.can_deposit(pd.Timestamp("2026-03-06"), _FakePair(999)) is True


def test_none_timestamp_allowed(pricing):
    assert pricing.can_deposit(None, _FakePair(1)) is True


def test_no_vault_state_allowed():
    p = BacktestPricing(_candle_universe(), routing_model=None, vault_state=None)
    assert p.can_deposit(pd.Timestamp("2026-03-06"), _FakePair(1)) is True
    assert p.check_redemption(pd.Timestamp("2026-03-06"), _FakePair(1)).can_redeem is True
    assert p.get_max_deposit(pd.Timestamp("2026-03-06"), _FakePair(1)) is None


def test_check_redemption_closed(pricing):
    result = pricing.check_redemption(pd.Timestamp("2026-03-06"), _FakePair(2))
    assert result.can_redeem is False
    assert result.max_redemption == 0.0
    assert result.message == "Redemptions paused"


def test_check_redemption_unknown_allowed(pricing):
    # redemption_open is NA on 03-05 -> allowed.
    assert pricing.check_redemption(pd.Timestamp("2026-03-05"), _FakePair(2)).can_redeem is True


def test_check_redemption_open_with_cap(pricing):
    # A positive historical cap is reported on the result (consumed by the Hyperliquid reduction
    # path); redemption stays allowed.
    result = pricing.check_redemption(pd.Timestamp("2026-03-07"), _FakePair(2))
    assert result.can_redeem is True
    assert result.max_redemption == 1234.0


def test_get_max_redemption_reports_historical_cap(pricing):
    # Faithful accessor: returns the recorded cap, or None when unknown (NA).
    assert pricing.get_max_redemption(pd.Timestamp("2026-03-07"), _FakePair(2)) == Decimal("1234.0")
    assert pricing.get_max_redemption(pd.Timestamp("2026-03-05"), _FakePair(2)) is None
