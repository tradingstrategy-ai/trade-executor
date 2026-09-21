"""Black-box coverage for recording live HyperCore vault selection.

This test enters through the public Typer ``start`` command and lets normal
bootstrap, live data loading, indicator calculation, scheduling, strategy
execution, and shutdown run for multiple cycles. It then treats the recorder
DuckDB as an external artefact and checks that an analyst can connect the
constructed vault universe and TVL data to the AlphaModel candidate and
selection observations.
"""

import json
import os
from pathlib import Path
from unittest import mock

import duckdb
import pytest
from click import Group
from typer.main import get_command
from tradingstrategy.vault_data_client import VAULT_PRO_API_KEY_ENV_VAR

from tradeexecutor.cli.commands import start as _start  # noqa: F401 - register the real start command
from tradeexecutor.cli.commands.app import app
from tradeexecutor.state.state import State
from tradeexecutor.strategy.recorder.serialisation import decode_json_value


REQUIRED_ENVIRONMENT_VARIABLES = (
    "TRADING_STRATEGY_API_KEY",
    VAULT_PRO_API_KEY_ENV_VAR,
    "JSON_RPC_HYPERLIQUID",
)

pytestmark = [
    pytest.mark.slow_test_group,
    pytest.mark.skipif(
        any(os.environ.get(name) is None for name in REQUIRED_ENVIRONMENT_VARIABLES),
        reason="Set TRADING_STRATEGY_API_KEY, VAULT_PRO_API_KEY, and JSON_RPC_HYPERLIQUID to run this test",
    ),
]


@pytest.fixture()
def hypercore_recorder_strategy_file() -> Path:
    """Return the checked-in live-data strategy loaded by the CLI test.

    The black-box test uses a normal strategy file so it covers strategy-module
    discovery and gives developers an inspectable example outside the test.
    """
    return Path(__file__).resolve().parents[2] / "strategies" / "test_only" / "hypercore_recorder_alpha_model.py"


def _read_object(connection: duckdb.DuckDBPyConnection, reference: dict[str, str]) -> tuple[str, dict]:
    """Resolve one content-addressed recorder object through its manifest ref.

    :param connection:
        Read-only connection to the completed recorder database.
    :param reference:
        Manifest mapping containing the object's SHA-256 identity.
    :return:
        Stored object kind and parsed payload.
    """
    row = connection.execute(
        "SELECT kind, payload FROM objects WHERE content_hash = ?",
        [reference["object"]],
    ).fetchone()
    assert row is not None
    return row[0], json.loads(row[1])


@pytest.mark.timeout(600)
def test_start_records_live_hypercore_universe_and_alpha_model_selection(
    tmp_path: Path,
    persistent_test_cache_path: str,
    hypercore_recorder_strategy_file: Path,
) -> None:
    """Record and inspect multiple live HyperCore AlphaModel decisions.

    1. Run the real ``start`` command for two one-second cycles against live HyperCore data.
    2. Confirm normal executor state and its adjacent recorder database were produced.
    3. Resolve each recorded constructed universe and verify vault identity, metadata, and TVL frames.
    4. Verify indicator fingerprints identify the TVL and age-ramp inputs used by the strategy.
    5. Join recorded candidates to the universe and confirm AlphaModel selected an equal-weight subset.
    """
    # 1. Run the real ``start`` command for two one-second cycles against live HyperCore data.
    executor_id = "hypercore-recorder-blackbox"
    state_file = tmp_path / "hypercore-recorder-state.json"
    record_file = tmp_path / f"{executor_id}-record.duckdb"
    environment = {
        # Preserve only the external services this live integration actually
        # uses; clearing the environment prevents a developer's deployment
        # variables from silently changing the black-box executor mode.
        "PATH": os.environ["PATH"],
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        VAULT_PRO_API_KEY_ENV_VAR: os.environ[VAULT_PRO_API_KEY_ENV_VAR],
        "JSON_RPC_HYPERLIQUID": os.environ["JSON_RPC_HYPERLIQUID"],
        "EXECUTOR_ID": executor_id,
        "STRATEGY_FILE": str(hypercore_recorder_strategy_file),
        "STATE_FILE": str(state_file),
        "CACHE_PATH": persistent_test_cache_path,
        "RESET_STATE": "true",
        # A real hot-wallet stack provides generic HyperCore routing. The
        # fixed test key is empty and the strategy returns no trades.
        "ASSET_MANAGEMENT_MODE": "hot_wallet",
        "PRIVATE_KEY": "0x111e53aed5e777996f26b4bdb89300bbc05b84743f32028c41be7193c0fe0b83",
        "MIN_GAS_BALANCE": "0",
        "GAS_BALANCE_WARNING_LEVEL": "0",
        "CHECK_ACCOUNTS": "false",
        "SYNC_TREASURY_ON_STARTUP": "true",
        # Run on the shortest scheduler cadence. The loop checks MAX_CYCLES
        # before the next callback, so a value of 3 yields two decisions.
        "STRATEGY_CYCLE_TRIGGER": "since_last_cycle_end",
        "CYCLE_DURATION": "1s",
        "MAX_CYCLES": "3",
        "MAX_DATA_DELAY_MINUTES": str(7 * 24 * 60),
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",
        "STATS_REFRESH_MINUTES": "0",
        "POSITION_TRIGGER_CHECK_MINUTES": "0",
        "VISUALISATION": "false",
    }

    # This patch controls process configuration only. It does not replace any
    # CLI, client, runner, universe, indicator, AlphaModel, or recorder method.
    with mock.patch.dict(os.environ, environment, clear=True):
        # Typer flattens a single-command app; other collected CLI tests can
        # register additional commands on the shared app.
        cli = get_command(app)
        cli.main(args=["start"] if isinstance(cli, Group) else [], standalone_mode=False)

    # 2. Confirm normal executor state and its adjacent recorder database were produced.
    assert state_file.exists()
    assert record_file.exists()
    state = State.from_json(state_file.read_text(encoding="utf-8"))
    assert len(state.uptime.cycles_completed_at) == 2

    connection = duckdb.connect(str(record_file), read_only=True)
    try:
        decisions = connection.execute(
            "SELECT cycle, status, input_manifest, observations FROM decisions ORDER BY cycle"
        ).fetchall()
        assert [row[0] for row in decisions] == [1, 2]
        assert all(row[1] == "completed" for row in decisions)

        for _cycle, _status, raw_manifest, raw_observations in decisions:
            manifest = json.loads(raw_manifest)
            observations = decode_json_value(json.loads(raw_observations))

            # 3. Resolve each recorded constructed universe and verify vault
            # identity, metadata, and TVL frames. This is the actual universe
            # passed into decide_trades(), not a fresh post-test API request.
            universe_kind, universe = _read_object(connection, manifest["universe"])
            assert universe_kind == "universe"
            assert 3 <= len(universe["pairs"]) <= 12
            assert all(pair["pair_key"].startswith("9999:") for pair in universe["pairs"])
            assert all(pair["identifier"] for pair in universe["pairs"])

            vault_specs = universe["metadata"]["vault_specs"]
            assert vault_specs["type"] == "VaultUniverse"
            assert len(vault_specs["vaults"]) == len(universe["pairs"])
            assert all(vault["vault_address"] for vault in vault_specs["vaults"])
            assert all(vault["name"] for vault in vault_specs["vaults"])
            assert all("deposit_status" in vault["metadata"] for vault in vault_specs["vaults"])
            assert all("deposit_closed_reason" in vault["metadata"] for vault in vault_specs["vaults"])

            liquidity = universe["frames"]["data_universe.liquidity.df"]
            assert liquidity["row_count"] > len(universe["pairs"])
            assert "close" in liquidity["schema"]["columns"]
            assert liquidity["chunks"]
            liquidity_rows = []
            for chunk_reference in liquidity["chunks"]:
                chunk_kind, chunk = _read_object(connection, chunk_reference)
                assert chunk_kind == "frame_chunk"
                assert chunk["rows"]
                liquidity_rows.extend(decode_json_value(chunk["rows"]))

            # Rebuild only the columns needed for this assertion. The recorder
            # stores every original TVL row in content-addressed chunks, which
            # lets us trace each calculated candidate TVL to its source history.
            liquidity_columns = liquidity["schema"]["columns"]
            pair_id_column = liquidity_columns.index("pair_id")
            assert "timestamp" in liquidity_columns
            close_column = liquidity_columns.index("close")
            tvl_history_by_pair: dict[int, list[float]] = {}
            for row in liquidity_rows:
                pair_id = int(row[pair_id_column])
                close = row[close_column]
                if close is None or close != close:
                    continue
                tvl_history_by_pair.setdefault(pair_id, []).append(float(close))

            # 4. Verify indicator fingerprints identify the TVL and age-ramp
            # inputs used by this decision, including reproducibility metadata.
            indicator_payloads = [
                _read_object(connection, reference)[1]
                for reference in manifest["indicators"]
            ]
            indicator_names = {payload["name"] for payload in indicator_payloads}
            assert {"tvl", "age", "age_ramp_weight"} <= indicator_names
            for indicator in indicator_payloads:
                fingerprint = indicator["result"]
                assert len(fingerprint["sha256"]) == 64
                assert fingerprint["length"] >= 0
                # Live metadata may include a newly discovered vault before it
                # has its first history row. Empty fingerprints are valuable
                # evidence of that condition; populated results must carry
                # their temporal bounds.
                if fingerprint["length"]:
                    assert fingerprint["start_at"] is not None
                    assert fingerprint["end_at"] is not None
            for required_name in ("tvl", "age", "age_ramp_weight"):
                assert any(
                    payload["name"] == required_name and payload["result"]["length"] > 0
                    for payload in indicator_payloads
                )

            # 5. Join candidates to the constructed universe and confirm the
            # AlphaModel selected a bounded equal-weight subset. TVL is asserted
            # on the candidate values actually consumed by the strategy.
            observations_by_name = {item["name"]: item for item in observations}
            candidates = observations_by_name["vault_candidates"]["value"]
            selected = observations_by_name["selected_vaults"]["value"]
            universe_addresses = {
                pair["pair_key"].split(":", maxsplit=1)[1]
                for pair in universe["pairs"]
            }
            candidate_by_address = {candidate["vault_address"]: candidate for candidate in candidates}

            assert set(candidate_by_address) == universe_addresses
            assert any(candidate["tvl"] is not None and candidate["tvl"] > 0 for candidate in candidates)
            assert all(candidate["tvl"] is None or candidate["tvl"] > 0 for candidate in candidates)
            assert all(
                candidate["rejection_reason"] == "missing_tvl"
                for candidate in candidates
                if candidate["tvl"] is None
            )
            for candidate in candidates:
                if candidate["tvl"] is not None:
                    # The explicit calculation value must be traceable to the
                    # captured TVL history for the same constructed pair.
                    assert any(
                        candidate["tvl"] == pytest.approx(recorded_tvl)
                        for recorded_tvl in tvl_history_by_pair[candidate["pair_id"]]
                    )
            eligible = [candidate for candidate in candidates if candidate["eligible"]]
            assert eligible
            expected = sorted(eligible, key=lambda candidate: candidate["signal"], reverse=True)[:3]
            assert [item["vault_address"] for item in selected] == [item["vault_address"] for item in expected]
            assert all(item["vault_address"] in candidate_by_address for item in selected)
            assert all(candidate_by_address[item["vault_address"]]["eligible"] for item in selected)
            assert all(candidate_by_address[item["vault_address"]]["tvl"] >= 7_500 for item in selected)
            assert [item["rank"] for item in selected] == list(range(1, len(selected) + 1))
            assert all(item["raw_weight"] == 1.0 for item in selected)
    finally:
        connection.close()
