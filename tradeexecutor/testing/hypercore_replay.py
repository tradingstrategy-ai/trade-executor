"""Reusable Hypercore historical replay helpers for tests.

This module provides a small market-data replay layer for Hypercore vault
tests.  The goal is deliberately modest:

- replay *real* historical ``tvl`` and ``account_pnl`` values from
  :mod:`eth_defi.hyperliquid.daily_metrics`
- derive a deterministic test share-price path ourselves
- hardcode the Hyperliquid fields we do not have historical coverage for yet
- keep the behaviour stable and easy to reason about in test assertions

The replay logic here is intentionally **not** a full Hyperliquid simulator.
It only models the parts that the trade executor needs for live-style
Hypercore tests:

- position valuation
- TVL-based sizing
- deposit / redemption gating

Why share price is calculated here
==================================

For the first test generation we only trust two historical inputs from
``eth_defi.hyperliquid.daily_metrics``:

- ``tvl``
- ``cumulative_pnl`` (exposed here as ``account_pnl_usd``)

We do **not** treat the daily-metrics ``share_price`` column as source truth
for these tests.  Instead, we reconstruct a deterministic share price path
inside the mock.  This keeps the replay rules explicit and documents the
assumption in one place.

Why some fields are hardcoded
=============================

Some live Hyperliquid fields needed by the trade executor, such as
``leader_fraction``, do not currently have the historical fidelity we need
for deterministic replay.  For the first generation of tests these values are
hardcoded to known working defaults.  The caller may still override them for
individual tests when needed.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

import pandas as pd

from tradeexecutor.state.identifier import TradingPairIdentifier


def _to_decimal(value: object) -> Decimal:
    """Normalise numeric replay inputs to :class:`Decimal`.

    Daily metrics data typically arrives as floats from DuckDB / pandas.
    Converting through ``str()`` gives deterministic decimal behaviour for
    assertions without pulling binary floating-point artefacts into the mock.
    """
    if isinstance(value, Decimal):
        return value
    if value is None:
        return Decimal(0)
    return Decimal(str(value))


def _normalise_timestamp(ts: datetime.datetime | datetime.date | pd.Timestamp) -> pd.Timestamp:
    """Turn replay lookup inputs into a naive UTC-normalised pandas timestamp."""
    if isinstance(ts, pd.Timestamp):
        value = ts
    elif isinstance(ts, datetime.datetime):
        value = pd.Timestamp(ts)
    elif isinstance(ts, datetime.date):
        value = pd.Timestamp(ts)
    else:
        raise TypeError(f"Unsupported timestamp type: {type(ts)}")

    if value.tzinfo is not None:
        value = value.tz_convert(None)
    return value.normalize()


@dataclass(frozen=True, slots=True)
class HypercoreReplaySnapshot:
    """One replayed Hypercore market-data snapshot."""

    vault_address: str
    requested_at: datetime.datetime
    data_date: datetime.date
    tvl_usd: Decimal
    account_pnl_usd: Decimal
    share_price: Decimal
    equity_usd: Decimal | None
    is_closed: bool
    allow_deposits: bool
    relationship_type: str
    leader_fraction: Decimal
    max_withdrawable_usd: Decimal | None
    lockup_expired: bool


class HypercoreVaultMarketDataSource(Protocol):
    """Interface used by Hypercore pricing and valuation tests."""

    def get_snapshot(
        self,
        timestamp: datetime.datetime | datetime.date | pd.Timestamp,
        pair: TradingPairIdentifier,
        *,
        safe_address: str | None = None,
        net_deposited_usdc: Decimal | None = None,
        current_equity_usd: Decimal | None = None,
    ) -> HypercoreReplaySnapshot:
        """Return one as-of market-data snapshot for a Hypercore vault pair."""


@dataclass(frozen=True, slots=True)
class HypercoreReplayDefaults:
    """Hardcoded defaults for missing historical Hypercore fields."""

    leader_fraction: Decimal = Decimal("0.10")
    allow_deposits: bool = True
    is_closed: bool = False
    relationship_type: str = "normal"
    lockup_expired: bool = True
    initial_share_price: Decimal = Decimal("1.0")


class HypercoreDailyMetricsReplay:
    """Replay Hypercore market data from historical daily metrics.

    The provider accepts either raw daily-metrics dataframes or dataframes
    loaded through :class:`eth_defi.hyperliquid.daily_metrics.HyperliquidDailyMetricsDatabase`.

    Only the following fields are treated as historical truth in v1:

    - ``tvl``
    - ``cumulative_pnl`` / ``account_pnl_usd``

    Everything else is deterministic test scaffolding layered on top of those
    values.
    """

    def __init__(
        self,
        frames_by_vault: dict[str, pd.DataFrame],
        *,
        defaults: HypercoreReplayDefaults | None = None,
        lockup_expired_after: datetime.datetime | datetime.date | pd.Timestamp | None = None,
    ):
        self.defaults = defaults or HypercoreReplayDefaults()
        self.lockup_expired_after = (
            _normalise_timestamp(lockup_expired_after)
            if lockup_expired_after is not None
            else None
        )
        self._frames: dict[str, pd.DataFrame] = {}
        self._initial_capital_by_vault: dict[str, Decimal] = {}

        for vault_address, df in frames_by_vault.items():
            normalised_address = vault_address.lower()
            prepared = self._prepare_frame(df.copy(), normalised_address)
            self._frames[normalised_address] = prepared
            self._initial_capital_by_vault[normalised_address] = self._calculate_initial_capital(prepared)

    @classmethod
    def from_daily_metrics_database(
        cls,
        db,
        vault_addresses: list[str],
        **kwargs,
    ) -> "HypercoreDailyMetricsReplay":
        """Load replay data from ``HyperliquidDailyMetricsDatabase``.

        The *db* object is duck-typed on purpose so this helper stays usable in
        environments where importing DuckDB-backed classes eagerly would be
        undesirable.
        """
        frames = {}
        for vault_address in vault_addresses:
            frames[vault_address.lower()] = db.get_vault_daily_prices(vault_address)
        return cls(frames, **kwargs)

    @classmethod
    def from_single_vault_dataframe(
        cls,
        vault_address: str,
        df: pd.DataFrame,
        **kwargs,
    ) -> "HypercoreDailyMetricsReplay":
        """Convenience constructor for small unit-test fixtures."""
        return cls({vault_address.lower(): df}, **kwargs)

    @staticmethod
    def _prepare_frame(df: pd.DataFrame, vault_address: str) -> pd.DataFrame:
        if "date" not in df.columns:
            raise KeyError(f"Replay frame for {vault_address} is missing 'date'")

        if "account_pnl_usd" not in df.columns:
            if "cumulative_pnl" in df.columns:
                df["account_pnl_usd"] = df["cumulative_pnl"]
            else:
                raise KeyError(
                    f"Replay frame for {vault_address} is missing both 'account_pnl_usd' and 'cumulative_pnl'"
                )

        if "tvl_usd" not in df.columns:
            if "tvl" in df.columns:
                df["tvl_usd"] = df["tvl"]
            else:
                raise KeyError(f"Replay frame for {vault_address} is missing both 'tvl_usd' and 'tvl'")

        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.sort_values("date").reset_index(drop=True)
        return df

    @staticmethod
    def _calculate_initial_capital(df: pd.DataFrame) -> Decimal:
        first = df.iloc[0]
        first_tvl = _to_decimal(first["tvl_usd"])
        first_account_pnl = _to_decimal(first["account_pnl_usd"])

        # Share price is reconstructed by the mock.  We use the first day's
        # implied principal base as the denominator:
        #
        #     principal ~= tvl - cumulative_pnl
        #
        # This is intentionally simple and deterministic.  It is good enough
        # for replay-based sizing and valuation tests, even though it is not a
        # perfect reproduction of every Hyperliquid accounting edge case.
        initial_capital = first_tvl - first_account_pnl
        if initial_capital <= 0:
            if first_tvl > 0:
                return first_tvl
            return Decimal(1)
        return initial_capital

    def _lookup_row(
        self,
        vault_address: str,
        timestamp: datetime.datetime | datetime.date | pd.Timestamp,
    ) -> pd.Series:
        key = vault_address.lower()
        frame = self._frames.get(key)
        if frame is None:
            raise KeyError(f"No replay data configured for Hypercore vault {vault_address}")

        requested_date = _normalise_timestamp(timestamp)
        matches = frame.loc[frame["date"] <= requested_date]
        if matches.empty:
            raise LookupError(
                f"No replay row for vault {vault_address} on or before {requested_date.date()}"
            )
        return matches.iloc[-1]

    def calculate_share_price(
        self,
        vault_address: str,
        row: pd.Series,
    ) -> Decimal:
        """Reconstruct the share-price multiplier for one replay row."""
        initial_capital = self._initial_capital_by_vault[vault_address.lower()]
        account_pnl_usd = _to_decimal(row["account_pnl_usd"])
        gross_equity = initial_capital + account_pnl_usd
        if gross_equity <= 0:
            return Decimal(0)
        return self.defaults.initial_share_price * (gross_equity / initial_capital)

    def get_snapshot(
        self,
        timestamp: datetime.datetime | datetime.date | pd.Timestamp,
        pair: TradingPairIdentifier,
        *,
        safe_address: str | None = None,
        net_deposited_usdc: Decimal | None = None,
        current_equity_usd: Decimal | None = None,
    ) -> HypercoreReplaySnapshot:
        """Replay one Hypercore market-data snapshot.

        ``safe_address`` is accepted for interface symmetry with the live
        Hyperliquid path, even though the v1 replay rules do not use it yet.

        ``net_deposited_usdc`` is the test-side position quantity.  Because the
        replay share price is calculated by us, the replay equity can be
        reconstructed as:

        ``equity_usd = net_deposited_usdc * share_price``

        When only redemption gating is needed and the caller already has a
        current equity estimate, it can pass ``current_equity_usd`` instead.
        """
        del safe_address

        vault_address = pair.other_data.get("hypercore_vault_address")
        if not vault_address:
            raise AssertionError(f"No hypercore_vault_address in pair other_data: {pair}")

        row = self._lookup_row(vault_address, timestamp)
        share_price = self.calculate_share_price(vault_address, row)

        equity_usd: Decimal | None
        if current_equity_usd is not None:
            equity_usd = Decimal(current_equity_usd)
        elif net_deposited_usdc is not None:
            equity_usd = Decimal(net_deposited_usdc) * share_price
        else:
            equity_usd = None

        if equity_usd is None:
            max_withdrawable_usd = None
        else:
            max_withdrawable_usd = min(equity_usd, _to_decimal(row["tvl_usd"]))

        requested_at = _normalise_timestamp(timestamp).to_pydatetime()
        if self.lockup_expired_after is None:
            lockup_expired = self.defaults.lockup_expired
        else:
            lockup_expired = _normalise_timestamp(timestamp) >= self.lockup_expired_after

        return HypercoreReplaySnapshot(
            vault_address=vault_address.lower(),
            requested_at=requested_at,
            data_date=row["date"].date(),
            tvl_usd=_to_decimal(row["tvl_usd"]),
            account_pnl_usd=_to_decimal(row["account_pnl_usd"]),
            share_price=share_price,
            equity_usd=equity_usd,
            is_closed=self.defaults.is_closed,
            allow_deposits=self.defaults.allow_deposits,
            relationship_type=self.defaults.relationship_type,
            leader_fraction=self.defaults.leader_fraction,
            max_withdrawable_usd=max_withdrawable_usd,
            lockup_expired=lockup_expired,
        )

