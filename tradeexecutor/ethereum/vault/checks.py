"""Vault data quality checks.

Guards against making allocation decisions on stale vault data.
The framework forward-fill pipeline (see ``create_from_dataset()``)
extends every pair's candle and TVL data to the decision timestamp,
which prevents indicator crashes but also masks stale data. A vault
whose real data stopped days ago will have forward-filled rows that
look valid — ``tvl()`` sees the last real TVL repeated, ``age()``
keeps growing, and ``age_ramp_weight()`` increases. Without an
explicit staleness check, ``decide_trades()`` would allocate to
vaults based on synthetic forward-filled data.

This module provides ``check_stale_vault_data()`` which inspects the
``forward_filled`` column in the candle universe to find the last
*real* data timestamp per vault pair and raises if any vault exceeds
a staleness tolerance.

It also provides ``get_vault_data_freshness()`` which returns a
diagnostic DataFrame of freshness information for all vault pairs,
covering both candle and TVL data. This is intended for operator
dashboards and strategy chart registries.
"""

import datetime
import logging

import pandas as pd

from tradeexecutor.strategy.execution_context import ExecutionMode
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.state.types import USDollarAmount

logger = logging.getLogger(__name__)


class StaleVaultData(Exception):
    """Raised when one or more vaults have data older than the tolerance."""
    pass


def check_stale_vault_data(
    strategy_universe: TradingStrategyUniverse,
    decision_timestamp: datetime.datetime,
    execution_mode: ExecutionMode,
    tolerance: datetime.timedelta = datetime.timedelta(hours=36),
    min_tvl: USDollarAmount = 5_000,
) -> None:
    """Check that vault candle data is fresh enough for allocation decisions.

    Iterates over all vault pairs in the universe, finds the last
    non-forward-filled candle timestamp for each, and raises
    :class:`StaleVaultData` if any vault's real data is older than
    ``tolerance`` relative to ``decision_timestamp``.

    Vaults whose latest TVL is below ``min_tvl`` are silently skipped,
    because low-TVL vaults often have sporadic data and would otherwise
    cause false-positive staleness failures.

    Call this at the start of ``decide_trades()`` to bail out early
    rather than allocating based on synthetic forward-filled data.

    Silently returns for non-live execution modes (backtesting,
    simulation) where data is always complete by construction.

    :param strategy_universe:
        The trading universe with forward-filled candle data.
    :param decision_timestamp:
        The current decision cycle timestamp (naive UTC).
    :param execution_mode:
        The current execution mode. The check only runs for live
        trading modes; backtesting and simulation are skipped.
    :param tolerance:
        Maximum allowed age of real (non-forward-filled) candle data.
        Defaults to 36 hours, which accommodates the 24h parquet cache
        plus normal pipeline lag.
    :param min_tvl:
        Minimum TVL in USD for a vault to be checked. Vaults below
        this threshold are skipped. Defaults to 5000 USD.
    :raises StaleVaultData:
        If any vault pair has stale data. The exception message lists
        every stale vault with its name, address, last real timestamp,
        and data age.
    """
    if not execution_mode.is_live_trading():
        return

    candles = strategy_universe.data_universe.candles
    if candles is None:
        return

    liquidity = strategy_universe.data_universe.liquidity
    has_forward_filled_column = "forward_filled" in candles.df.columns

    stale_vaults = []
    now = pd.Timestamp(decision_timestamp)

    for pair in strategy_universe.iterate_pairs():
        if not pair.is_vault():
            continue

        # Look up the latest TVL from liquidity data.
        # Skip vaults below the min_tvl threshold — they often have
        # sporadic data and would cause false-positive staleness failures.
        latest_tvl = None
        if liquidity is not None:
            try:
                pair_liquidity = liquidity.get_samples_by_pair(pair.internal_id)
                if len(pair_liquidity) > 0:
                    latest_tvl = float(pair_liquidity["close"].iloc[-1])
                    if latest_tvl < min_tvl:
                        continue
            except KeyError:
                pass

        try:
            pair_candles = candles.get_samples_by_pair(pair.internal_id)
        except KeyError:
            continue

        if len(pair_candles) == 0:
            continue

        # Find the last real (non-forward-filled) timestamp
        if has_forward_filled_column and "forward_filled" in pair_candles.columns:
            real_candles = pair_candles[pair_candles["forward_filled"] != True]
            if len(real_candles) == 0:
                # All data is forward-filled — vault has no real data at all
                stale_vaults.append((pair, None, None, latest_tvl))
                continue
            last_real_ts = real_candles["timestamp"].max()
        else:
            # No forward_filled column — use last timestamp as-is
            last_real_ts = pair_candles["timestamp"].max()

        age = now - pd.Timestamp(last_real_ts)
        if age > pd.Timedelta(tolerance):
            stale_vaults.append((pair, last_real_ts, age, latest_tvl))

    if stale_vaults:
        lines = []
        for pair, last_ts, age, pair_tvl in stale_vaults:
            tvl_str = f", TVL ${pair_tvl:,.0f}" if pair_tvl is not None else ""
            if last_ts is None:
                lines.append(f"  - {pair.get_ticker()} ({pair.pool_address}): no real data (all forward-filled){tvl_str}")
            else:
                lines.append(f"  - {pair.get_ticker()} ({pair.pool_address}): last real data {last_ts}, age {age}{tvl_str}")
        detail = "\n".join(lines)
        raise StaleVaultData(
            f"{len(stale_vaults)} vault(s) have stale candle data (tolerance {tolerance}, min TVL ${min_tvl:,.0f}):\n{detail}"
        )


def _count_trailing_forward_filled(pair_data: pd.DataFrame) -> int:
    """Count contiguous forward-filled rows at the tail of a per-pair DataFrame.

    Only counts the trailing stale stretch, not forward-filled rows
    scattered throughout the series (which may be legitimate gap fills).

    Returns 0 if no ``forward_filled`` column exists or the tail is real data.
    Returns the total row count if every row is forward-filled.
    """
    if "forward_filled" not in pair_data.columns:
        return 0

    ff_col = pair_data["forward_filled"]
    count = 0
    for val in reversed(ff_col.values):
        if val is True or val == True:  # noqa: E712 — deliberate identity check for np.bool_
            count += 1
        else:
            break
    return count


def _get_last_real_timestamp(
    pair_data: pd.DataFrame,
    has_forward_filled_column: bool,
) -> pd.Timestamp | None:
    """Find the last non-forward-filled timestamp in a per-pair DataFrame.

    Returns ``None`` when all rows are forward-filled or the column is absent
    and the DataFrame is empty.
    """
    if has_forward_filled_column and "forward_filled" in pair_data.columns:
        real_rows = pair_data[pair_data["forward_filled"] != True]
        if len(real_rows) == 0:
            return None
        return pd.Timestamp(real_rows["timestamp"].max())
    # No forward_filled column — treat all data as real
    if len(pair_data) == 0:
        return None
    return pd.Timestamp(pair_data["timestamp"].max())


def _resolve_reference_timestamp(
    strategy_universe: TradingStrategyUniverse,
) -> pd.Timestamp | None:
    """Derive the reference timestamp from the universe data.

    Uses the global maximum candle timestamp across all pairs as the
    reference point. This matches the ``forward_fill_until`` value that
    ``create_from_dataset()`` uses when constructing the universe.

    Returns ``None`` when no candle data is available.
    """
    candles = strategy_universe.data_universe.candles
    if candles is None or len(candles.df) == 0:
        return None
    return pd.Timestamp(candles.df["timestamp"].max())


def get_vault_data_freshness(
    strategy_universe: TradingStrategyUniverse,
    reference_timestamp: datetime.datetime | None = None,
) -> pd.DataFrame:
    """Build a diagnostic table of data freshness for all vault pairs.

    For each vault pair in the universe, reports the last real (non-forward-filled)
    timestamp for both candle and TVL data and how old that data is relative to
    ``reference_timestamp``. This allows operators to see at a glance which vaults
    have stale data — whether or not the staleness is masked by framework
    forward-fill.

    Data age is computed as ``reference_timestamp - last_real_timestamp``, so it
    correctly reflects staleness even when the ``forward_filled`` column is absent
    (e.g. when forward-fill was skipped or the universe was built without it).

    Unlike :func:`check_stale_vault_data`, this function:

    - Always runs (no execution-mode or tolerance filtering)
    - Returns a DataFrame instead of raising
    - Reports both candle and TVL freshness separately
    - Includes all vault pairs regardless of TVL

    :param strategy_universe:
        The trading universe with (possibly forward-filled) candle and liquidity data.
    :param reference_timestamp:
        The point in time to measure freshness against. Defaults to the global
        maximum candle timestamp across all pairs (which matches the
        ``forward_fill_until`` value used by ``create_from_dataset()``).
    :return:
        A DataFrame sorted by worst staleness descending (stalest first),
        with ``NaT`` (no real data at all) appearing at the top.
    """
    candles = strategy_universe.data_universe.candles
    liquidity = strategy_universe.data_universe.liquidity

    has_candle_ff_col = candles is not None and "forward_filled" in candles.df.columns
    has_liquidity_ff_col = liquidity is not None and "forward_filled" in liquidity.df.columns

    if reference_timestamp is not None:
        ref_ts = pd.Timestamp(reference_timestamp)
    else:
        ref_ts = _resolve_reference_timestamp(strategy_universe)

    rows = []

    for pair in strategy_universe.iterate_pairs():
        if not pair.is_vault():
            continue

        vault_name = pair.get_vault_name() or pair.get_ticker()

        # -- Candle freshness --
        last_real_candle = None
        latest_candle = None
        candle_data_age = None
        trailing_stale_candles = 0

        if candles is not None:
            try:
                pair_candles = candles.get_samples_by_pair(pair.internal_id)
                if len(pair_candles) > 0:
                    latest_candle = pd.Timestamp(pair_candles["timestamp"].max())
                    last_real_candle = _get_last_real_timestamp(pair_candles, has_candle_ff_col)
                    trailing_stale_candles = _count_trailing_forward_filled(pair_candles)
                    if last_real_candle is not None and ref_ts is not None:
                        candle_data_age = ref_ts - last_real_candle
            except KeyError:
                pass

        # -- TVL freshness --
        last_real_tvl = None
        latest_tvl_ts = None
        tvl_data_age = None
        latest_tvl_usd = None

        if liquidity is not None:
            try:
                pair_liquidity = liquidity.get_samples_by_pair(pair.internal_id)
                if len(pair_liquidity) > 0:
                    latest_tvl_ts = pd.Timestamp(pair_liquidity["timestamp"].max())
                    latest_tvl_usd = float(pair_liquidity["close"].iloc[-1])
                    last_real_tvl = _get_last_real_timestamp(pair_liquidity, has_liquidity_ff_col)
                    if last_real_tvl is not None and ref_ts is not None:
                        tvl_data_age = ref_ts - last_real_tvl
            except KeyError:
                pass

        rows.append({
            "Vault": vault_name,
            "Address": pair.pool_address,
            "Last real candle": last_real_candle,
            "Latest candle": latest_candle,
            "Candle data age": candle_data_age,
            "Trailing stale candles": trailing_stale_candles,
            "Last real TVL": last_real_tvl,
            "Latest TVL timestamp": latest_tvl_ts,
            "TVL data age": tvl_data_age,
            "Latest TVL (USD)": latest_tvl_usd,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Sort by worst staleness across both candle and TVL; NaT sorts to top
    df["_sort_key"] = df[["Candle data age", "TVL data age"]].max(axis=1)
    df = df.sort_values("_sort_key", ascending=False, na_position="first")
    df = df.drop(columns=["_sort_key"])
    df = df.reset_index(drop=True)
    return df
