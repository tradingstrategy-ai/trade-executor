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
"""

import datetime
import logging

import pandas as pd

from tradeexecutor.strategy.execution_context import ExecutionMode
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse

logger = logging.getLogger(__name__)


class StaleVaultData(Exception):
    """Raised when one or more vaults have data older than the tolerance."""
    pass


def check_stale_vault_data(
    strategy_universe: TradingStrategyUniverse,
    decision_timestamp: datetime.datetime,
    execution_mode: ExecutionMode,
    tolerance: datetime.timedelta = datetime.timedelta(hours=36),
) -> None:
    """Check that vault candle data is fresh enough for allocation decisions.

    Iterates over all vault pairs in the universe, finds the last
    non-forward-filled candle timestamp for each, and raises
    :class:`StaleVaultData` if any vault's real data is older than
    ``tolerance`` relative to ``decision_timestamp``.

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

    has_forward_filled_column = "forward_filled" in candles.df.columns

    stale_vaults = []
    now = pd.Timestamp(decision_timestamp)

    for pair in strategy_universe.iterate_pairs():
        if not pair.is_vault():
            continue

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
                stale_vaults.append((pair, None, None))
                continue
            last_real_ts = real_candles["timestamp"].max()
        else:
            # No forward_filled column — use last timestamp as-is
            last_real_ts = pair_candles["timestamp"].max()

        age = now - pd.Timestamp(last_real_ts)
        if age > pd.Timedelta(tolerance):
            stale_vaults.append((pair, last_real_ts, age))

    if stale_vaults:
        lines = []
        for pair, last_ts, age in stale_vaults:
            if last_ts is None:
                lines.append(f"  - {pair.get_ticker()} ({pair.pool_address}): no real data (all forward-filled)")
            else:
                lines.append(f"  - {pair.get_ticker()} ({pair.pool_address}): last real data {last_ts}, age {age}")
        detail = "\n".join(lines)
        raise StaleVaultData(
            f"{len(stale_vaults)} vault(s) have stale candle data (tolerance {tolerance}):\n{detail}"
        )
