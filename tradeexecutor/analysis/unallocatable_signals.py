"""Analyse portfolio allocations blocked by unavailable trading venues."""

from __future__ import annotations

import pandas as pd

from tradeexecutor.state.state import State


def calculate_unallocatable_signal_weights(
    state: State,
) -> pd.Series:
    """Build a time series of blocked allocation weights by asset.

    Each strategy cycle must persist ``unallocatable_signals`` in its
    visualisation calculations. Missing assets are explicitly represented as
    zero, so a stacked-area chart shows when an unavailable opportunity ended.
    """
    timestamps = []
    rows = []

    for timestamp, calculations in state.visualisation.calculations.items():
        timestamp = pd.Timestamp(timestamp, unit="s")
        timestamps.append(timestamp)
        for signal in calculations.get("unallocatable_signals", []):
            weight = float(signal.get("unallocatable_weight", 0.0) or 0.0)
            if weight <= 0:
                continue

            rows.append({
                "timestamp": timestamp,
                "asset": signal.get("asset_label") or signal.get("pair_ticker") or "Unknown asset",
                "weight": weight,
            })

    if not timestamps:
        return pd.Series(
            dtype="float64",
            index=pd.MultiIndex.from_arrays([[], []], names=["timestamp", "asset"]),
        )

    assets = sorted({row["asset"] for row in rows})
    if not assets:
        return pd.Series(
            dtype="float64",
            index=pd.MultiIndex.from_arrays([[], []], names=["timestamp", "asset"]),
        )

    index = pd.MultiIndex.from_product(
        [sorted(set(timestamps)), assets],
        names=["timestamp", "asset"],
    )
    weights = pd.DataFrame(rows).groupby(["timestamp", "asset"])["weight"].sum()
    weights = weights.reindex(index, fill_value=0.0)
    weights.attrs["reserve_asset_symbol"] = ""
    weights.attrs["credit_supply_symbols"] = []
    weights.attrs["vault_symbols"] = []
    return weights
