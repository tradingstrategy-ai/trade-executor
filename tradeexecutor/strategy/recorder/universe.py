"""Capture the constructed strategy universe without mutating it."""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from tradeexecutor.strategy.recorder.serialisation import encode_frame_chunks, to_json_value


ObjectWriter = Callable[[str, dict[str, Any]], dict[str, str]]


def _frame_capture(
    frame: pd.DataFrame | pd.Series,
    put_object: ObjectWriter,
) -> dict[str, Any]:
    """Store a dataframe in content-addressed chunks and return its manifest."""

    schema, chunks = encode_frame_chunks(frame)
    refs = []
    for chunk in chunks:
        ref = put_object("frame_chunk", chunk["payload"])
        refs.append({"object": ref["object"], "row_positions": list(range(chunk["start"], chunk["start"] + chunk["count"]))})
    return {"schema": schema, "row_count": len(frame), "chunks": refs}


def _capture_vault_specs(specs: Any) -> Any:
    """Capture the serialisable vault specification fields used by the universe."""
    if specs is None:
        return None
    if hasattr(specs, "iterate_vaults"):
        records = []
        for vault in specs.iterate_vaults():
            record = {
                name: to_json_value(getattr(vault, name, None))
                for name in (
                    "chain_id", "vault_address", "denomination_token_address",
                    "denomination_token_symbol", "denomination_token_decimals",
                    "share_token_address", "share_token_symbol", "share_token_decimals",
                    "protocol_name", "protocol_slug", "name", "token_symbol",
                    "features", "deployed_at", "denormalised_data_updated_at",
                    "management_fee", "performance_fee", "tvl", "issued_shares",
                )
            }
            metadata = getattr(vault, "metadata", None)
            if metadata is not None:
                record["metadata"] = {
                    name: to_json_value(getattr(metadata, name, None))
                    for name in (
                        "lifetime_return_net", "cagr_net", "three_months_return_net",
                        "volatility", "sharpe", "max_drawdown", "tvl", "tvl_peak",
                        "deposit_closed_reason", "redemption_closed_reason",
                        "deposit_next_open", "redemption_next_open", "deposit_status",
                        "redemption_status", "deposit_permission", "deposit_status_source",
                        "deposit_status_observed_at", "deposit_status_observed_block",
                    )
                }
            records.append(record)
        return {"type": type(specs).__name__, "vaults": records}
    try:
        return to_json_value(specs)
    except TypeError:
        return {"type": f"{type(specs).__module__}.{type(specs).__qualname__}"}


def capture_universe(universe: Any, put_object: ObjectWriter) -> dict[str, str]:
    """Store the constructed universe and its decision-time data frames."""

    data = universe.data_universe
    pairs = []
    pair_objects = getattr(data, "pairs", None)
    if pair_objects is not None:
        for pair in pair_objects.iterate_pairs():
            address = getattr(pair, "pool_address", None) or getattr(pair, "address", None) or ""
            chain = getattr(pair, "chain_id", "")
            pair_key = f"{getattr(chain, 'value', chain)}:{str(address).lower()}"
            pairs.append({
                "pair_key": pair_key,
                "internal_id": getattr(pair, "internal_id", getattr(pair, "pair_id", None)),
                "identifier": to_json_value(pair),
                "extra_metadata": to_json_value(dict(getattr(pair, "other_data", {}) or {})),
            })

    metadata = {
        "strategy_universe": {
            name: to_json_value(getattr(universe, name, None))
            for name in ("options", "primary_chain", "required_history_period", "price_data_delay_tolerance", "other_data", "vault_history_diagnostics", "vault_window_overrides", "ignore_routing", "backtest_stop_loss_time_bucket")
        },
        "data_universe": {
            name: to_json_value(getattr(data, name, None))
            for name in ("time_bucket", "chains", "exchange_universe", "exchanges", "forward_filled", "start_hint", "end_hint")
        },
        "vault_specs": _capture_vault_specs(getattr(data, "vault_specs", None)),
        "pair_cache": {
            str(key): to_json_value(value)
            for key, value in getattr(universe, "pair_cache", {}).items()
        },
    }
    frames = {}
    frame_sources = {
        "data_universe.pairs.df": getattr(pair_objects, "df", None),
        "data_universe.candles.df": getattr(getattr(data, "candles", None), "df", None),
        "data_universe.liquidity.df": getattr(getattr(data, "liquidity", None), "df", None),
        "data_universe.resampled_liquidity.df": getattr(getattr(data, "resampled_liquidity", None), "df", None),
        "data_universe.lending_candles.df": getattr(getattr(data, "lending_candles", None), "df", None),
        "vault_state": getattr(universe, "vault_state", None),
    }
    for key, frame in frame_sources.items():
        if isinstance(frame, (pd.DataFrame, pd.Series)):
            frames[key] = _frame_capture(frame, put_object)

    return put_object("universe", {"metadata": metadata, "pairs": pairs, "reserves": [to_json_value(v) for v in getattr(universe, "reserve_assets", [])], "frames": frames})
