"""Integration coverage for webhook data preloading during startup."""

import os
import subprocess
import time
from pathlib import Path

import pytest
import requests
from eth_defi.compat import native_datetime_utc_now
from eth_defi.utils import find_free_port

from tradeexecutor.state.state import State
from tradeexecutor.state.store import JSONFileStore


@pytest.fixture()
def hyper_ai_strategy_path() -> Path:
    """Return the Hyper AI strategy used by the startup preload integration test."""
    return Path(__file__).resolve().parents[2] / "strategies" / "test_only" / "hyper-ai-test.py"


def _stop_process(proc: subprocess.Popen[str]) -> str:
    """Terminate a subprocess and return its combined output."""
    if proc.poll() is None:
        proc.terminate()
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate(timeout=10)
    else:
        stdout, stderr = proc.communicate(timeout=10)

    return stdout + stderr


@pytest.mark.timeout(300)
@pytest.mark.slow_test_group
@pytest.mark.skipif(
    os.environ.get("TRADING_STRATEGY_API_KEY") is None or os.environ.get("JSON_RPC_HYPERLIQUID") is None,
    reason="Set TRADING_STRATEGY_API_KEY and JSON_RPC_HYPERLIQUID environment variables to run this test",
)
def test_start_preloads_chart_pairs_before_first_cycle(
    tmp_path: Path,
    hyper_ai_strategy_path: Path,
) -> None:
    """Test startup preload makes indicator-backed webhook data available before the first cycle.

    1. Seed a state file so the rolling scheduler places the next live cycle in the future.
    2. Launch `trade-executor start` with the Hyper AI strategy and webhook preload enabled.
    3. Confirm `/chart-registry/pairs` becomes available while `/status` still reports no completed cycle.
    """
    state_file = tmp_path / "hyper-ai-state.json"
    store = JSONFileStore(state_file)
    state = State()
    state.other_data.save(state.cycle, "decision_cycle_ended_at", native_datetime_utc_now())
    store.sync(state)

    port = find_free_port(20_000, 40_000, 20)
    server_url = f"http://127.0.0.1:{port}"

    environment = os.environ.copy()
    environment.update(
        {
            "STRATEGY_FILE": str(hyper_ai_strategy_path),
            "STATE_FILE": str(state_file),
            "CACHE_PATH": str(tmp_path / "cache"),
            "HTTP_ENABLED": "true",
            "HTTP_HOST": "127.0.0.1",
            "HTTP_PORT": str(port),
            "HTTP_WAIT_GOOD_STARTUP_SECONDS": "0",
            "PRELOAD_WEBHOOK_DATA": "true",
            "UNIT_TESTING": "true",
            "ASSET_MANAGEMENT_MODE": "hot_wallet",
            "PRIVATE_KEY": "0x111e53aed5e777996f26b4bdb89300bbc05b84743f32028c41be7193c0fe0b83",
            "MIN_GAS_BALANCE": "0",
            "TRADE_IMMEDIATELY": "false",
            "STRATEGY_CYCLE_TRIGGER": "since_last_cycle_end",
            "CYCLE_DURATION": "1d",
            "CHECK_ACCOUNTS": "false",
            "SYNC_TREASURY_ON_STARTUP": "false",
            "VISUALISATION": "false",
            "MAX_DATA_DELAY_MINUTES": str(10 * 60 * 24 * 365),
            "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
            "JSON_RPC_HYPERLIQUID": os.environ["JSON_RPC_HYPERLIQUID"],
            "LOG_LEVEL": "disabled",
        }
    )

    proc = subprocess.Popen(
        ["trade-executor", "start"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )

    try:
        # 1. Seed a state file so the rolling scheduler places the next live cycle in the future.
        # 2. Launch `trade-executor start` with the Hyper AI strategy and webhook preload enabled.
        deadline = time.monotonic() + 180
        last_status_payload: dict | None = None
        last_pairs_reply: tuple[int, str] | None = None
        pairs_payload: dict | None = None

        # 3. Confirm `/chart-registry/pairs` becomes available while `/status` still reports no completed cycle.
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                output = _stop_process(proc)
                raise AssertionError(f"trade-executor exited before webhook preload succeeded:\n{output}")

            try:
                status_response = requests.get(f"{server_url}/status", timeout=1.0)
            except requests.RequestException:
                time.sleep(0.5)
                continue

            assert status_response.status_code == 200, status_response.text
            last_status_payload = status_response.json()
            assert last_status_payload["executor_running"] is True
            assert last_status_payload["exception"] is None
            assert last_status_payload["completed_cycle"] is None

            try:
                pairs_response = requests.get(f"{server_url}/chart-registry/pairs", timeout=1.0)
            except requests.RequestException:
                time.sleep(0.5)
                continue

            last_pairs_reply = (pairs_response.status_code, pairs_response.text)
            if pairs_response.status_code == 200:
                pairs_payload = pairs_response.json()
                break

            time.sleep(0.5)

        if pairs_payload is None:
            output = _stop_process(proc)
            raise AssertionError(
                "Webhook preload did not make /chart-registry/pairs available before timeout.\n"
                f"Last /status payload: {last_status_payload}\n"
                f"Last /chart-registry/pairs reply: {last_pairs_reply}\n"
                f"Process output:\n{output}"
            )

        assert "default_pairs" in pairs_payload
        assert "all_pairs" in pairs_payload
        assert len(pairs_payload["all_pairs"]) > 0
    finally:
        _stop_process(proc)
