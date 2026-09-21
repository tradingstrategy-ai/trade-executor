"""Constructed-universe schema for a decision-input recorder.

The recorder captures the universe that actually reached ``decide_trades()``,
not the full remote API response that was used to build it. One ``universe``
object contains strategy and data-universe metadata, reserve assets, normalised
pair identifiers and metadata, relevant vault specifications, and manifests for
data frames. Each frame is split into content-addressed ``frame_chunk`` objects
so unchanged chunks are deduplicated across decisions.

Capture deliberately serialises copies of mutable domain objects. In particular,
some pair ``to_dict()`` encoders mutate ``other_data``; the serialisation helper
protects the live universe from that side effect.

:meth:`DecisionRecorder.begin
<tradeexecutor.strategy.recorder.recorder.DecisionRecorder.begin>` is the only
production caller. It invokes :func:`capture_universe` before strategy signal
calculations so later analysis sees the exact eligible data set rather than a
newly downloaded approximation.
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from tradeexecutor.strategy.recorder.serialisation import encode_frame_chunks, to_json_value


ObjectWriter = Callable[[str, dict[str, Any]], dict[str, str]]


def _frame_capture(
    frame: pd.DataFrame | pd.Series,
    put_object: ObjectWriter,
) -> dict[str, Any]:
    """Store a dataframe in content-addressed chunks and return its manifest.

    :func:`capture_universe` calls this for pair, candle, liquidity, lending,
    and vault-state frames present in the decision universe. Chunking exists so
    unchanged history is deduplicated when only the latest rows differ between
    live cycles.

    The manifest stores frame schema, row count, and each chunk reference with
    its original row positions. Chunks store only data values; schema is held
    once in the parent manifest.

    :param frame:
        Constructed-universe dataframe or series to persist.
    :param put_object:
        Content-addressed writer owned by the active recorder storage.
    :return:
        Parent manifest containing schema, row count, and chunk references.
    """
    schema, chunks = encode_frame_chunks(frame)
    refs = []
    for chunk in chunks:
        ref = put_object("frame_chunk", chunk["payload"])
        refs.append({"object": ref["object"], "row_positions": list(range(chunk["start"], chunk["start"] + chunk["count"]))})
    return {"schema": schema, "row_count": len(frame), "chunks": refs}


def _capture_vault_specs(specs: Any) -> Any:
    """Capture serialisable vault fields used by universe selection.

    :func:`capture_universe` calls this for ``data_universe.vault_specs``. The
    projection exists because vault eligibility depends on metadata that is not
    necessarily present in price frames, especially live deposit availability.

    This is a curated decision-data projection: identity, token, fee, TVL,
    issuance, protocol, and deposit/redemption availability metadata are kept,
    while full provider responses and arbitrary client caches are excluded.

    :param specs:
        Vault universe collection, serialisable value, or ``None``.
    :return:
        JSON-ready vault records, a safe type marker, or ``None``.
    """
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
    """Store the constructed decision universe and its data-frame manifests.

    Called by :meth:`DecisionRecorder.begin
    <tradeexecutor.strategy.recorder.recorder.DecisionRecorder.begin>` exactly
    once for each recorded live cycle. Its object reference becomes
    ``input_manifest.universe`` and is also listed as an input of each indicator
    fingerprint.

    :param universe:
        Universe passed to the active strategy decision.
    :param put_object:
        Content-addressed object writer supplied by :class:`RecorderStorage`.
    :return:
        ``{"object": <sha256>}`` reference to the parent ``universe`` object.
    """
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
