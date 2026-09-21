"""Black-box CLI coverage for the live strategy-input recorder.

The test starts the real Typer command with the normal test-only strategy and
the local Anvil/Uniswap fixtures. It intentionally does not replace CLI
bootstrap, scheduler, universe construction, or runner methods.
"""

import json
import os
from pathlib import Path
from unittest import mock

import duckdb
import pytest
from eth_defi.provider.anvil import AnvilLaunch
from eth_defi.uniswap_v2.deployment import UniswapV2Deployment
from eth_typing import HexAddress

from tradeexecutor.cli.commands import start as _start  # noqa: F401 - register the real start command
from tradeexecutor.cli.commands.app import app
from tradeexecutor.state.state import State


@pytest.fixture()
def strategy_file() -> Path:
    """Use a checked-in normal strategy module, not an inline test module."""

    return Path(__file__).resolve().parents[2] / "strategies" / "test_only" / "strategy_input_recorder.py"


@pytest.mark.timeout(300)
def test_cli_live_recorder_creates_state_adjacent_duckdb(
    anvil: AnvilLaunch,
    uniswap_v2: UniswapV2Deployment,
    weth_usdc_uniswap_pair: HexAddress,
    tmp_path: Path,
    strategy_file: Path,
) -> None:
    """Run two real one-second CLI decisions and inspect their recorder database.

    1. Build the local Anvil/Uniswap CLI environment and run the real Typer command.
    2. Confirm the executor wrote its state and state-adjacent recorder database.
    3. Inspect the database to verify both completed decision records and observations.
    """

    # 1. Build the local Anvil/Uniswap CLI environment and run the real Typer command.
    del weth_usdc_uniswap_pair  # The fixture creates the pair read by the mock client.

    executor_id = "cli-recorder-blackbox"
    state_file = tmp_path / "executor-state.json"
    record_file = tmp_path / f"{executor_id}-record.duckdb"
    cache_path = tmp_path / "cache"

    environment = {
        "EXECUTOR_ID": executor_id,
        "STRATEGY_FILE": strategy_file.as_posix(),
        "STATE_FILE": state_file.as_posix(),
        "RESET_STATE": "true",
        "ASSET_MANAGEMENT_MODE": "dummy",
        "JSON_RPC_ANVIL": anvil.json_rpc_url,
        "TEST_EVM_UNISWAP_V2_ROUTER": uniswap_v2.router.address,
        "TEST_EVM_UNISWAP_V2_FACTORY": uniswap_v2.factory.address,
        "TEST_EVM_UNISWAP_V2_INIT_CODE_HASH": uniswap_v2.init_code_hash,
        "CACHE_PATH": cache_path.as_posix(),
        "STRATEGY_CYCLE_TRIGGER": "since_last_cycle_end",
        "CYCLE_DURATION": "1s",
        # The loop increments its cycle counter after each decision and stops
        # when it reaches MAX_CYCLES, so 3 yields two completed cycles.
        "MAX_CYCLES": "3",
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",
        "CHECK_ACCOUNTS": "false",
        "SYNC_TREASURY_ON_STARTUP": "true",
        "STATS_REFRESH_MINUTES": "0",
        "POSITION_TRIGGER_CHECK_MINUTES": "0",
        "VISUALISATION": "false",
    }

    # This mock only scopes process environment variables; CLI bootstrap and execution are real.
    with mock.patch.dict(os.environ, environment, clear=True):
        # ``start`` is registered directly on this focused Typer app.
        app([], standalone_mode=False)

    # 2. Confirm the executor wrote its state and state-adjacent recorder database.
    assert state_file.exists()
    assert record_file.exists()
    state = State.from_json(state_file.read_text(encoding="utf-8"))
    assert len(state.uptime.cycles_completed_at) == 2

    # 3. Inspect the database to verify both completed decision records and observations.
    connection = duckdb.connect(str(record_file), read_only=True)
    try:
        run_count, metadata = connection.execute(
            "SELECT count(*), any_value(metadata) FROM runs"
        ).fetchone()
        assert run_count == 1
        assert json.loads(metadata)["strategy_file"] == str(strategy_file)
        decisions = connection.execute(
            "SELECT cycle, status, state_path, observations FROM decisions ORDER BY cycle"
        ).fetchall()
        assert [row[0] for row in decisions] == [1, 2]
        assert all(row[1] == "completed" for row in decisions)
        assert all(row[2] == str(state_file) for row in decisions)
        assert [json.loads(row[3])[0]["name"] for row in decisions] == ["cli_cycle", "cli_cycle"]
    finally:
        connection.close()
