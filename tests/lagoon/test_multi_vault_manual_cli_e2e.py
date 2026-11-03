"""Run the multi-vault manual shell tutorial end to end in simulated mode."""

import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import pytest

from tradeexecutor.state.state import State
from tradeexecutor.testing.multi_vault_manual import STRATEGY_FILE, get_target_pair, get_target_phase
from tradeexecutor.testing.erc4626_manual import load_universe


REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY / "scripts/lagoon/manual-multi-vault-cli.sh"

pytestmark = [
    pytest.mark.skipif(
        not (os.environ.get("JSON_RPC_ARBITRUM") and os.environ.get("JSON_RPC_ETHEREUM") and os.environ.get("TRADING_STRATEGY_API_KEY")),
        reason="JSON_RPC_ARBITRUM, JSON_RPC_ETHEREUM and TRADING_STRATEGY_API_KEY are needed to fork both chains",
    ),
    pytest.mark.skipif(shutil.which("anvil") is None or shutil.which("jq") is None, reason="anvil and jq are required"),
    pytest.mark.timeout(1800),
]


def _get_simulation_dir(run_dir: Path) -> Path:
    """Return the named simulation directory of a test run."""
    return Path(f"/tmp/multi-vault-sim-test-{run_dir.name}")


def _run_script(run_dir: Path, *arguments: str) -> subprocess.CompletedProcess:
    """Run the real Bash script with this interpreter, in simulated mode."""
    environment = {
        **os.environ,
        "SIMULATE": "true",
        "MULTI_VAULT_RUN_DIR": str(run_dir),
        "MULTI_VAULT_SIM_DIR": str(_get_simulation_dir(run_dir)),
        "MULTI_VAULT_PYTHON": sys.executable,
        "LOG_LEVEL": "warning",
    }
    return subprocess.run(["bash", str(SCRIPT), *arguments], env=environment, cwd=REPOSITORY, capture_output=True, text=True, timeout=1700)


def test_multi_vault_manual_cli_simulated() -> None:
    """The shell tutorial deploys, trades all three vault families and redeems the investor.

    Every Safe, Guard, Lagoon, Atoma, Upshift, CCTP and ZkLighter transaction
    is real on Arbitrum and Ethereum forks. Lighter's sequencer is the fake,
    Circle attestations are forged for real burns and only Atoma's share lock
    is advanced on the clock.

    1. Run the script from a fresh run directory with deployment and subscription.
    2. Check that every target is complete and no trade is unfinished.
    3. Check that a rerun after the forks stopped refuses instead of redeploying.
    """
    # A private temporary directory, not pytest's tmp_path: concurrent pytest
    # sessions of the same user rotate and delete older basetemps during this
    # long run. The script stops its forks on success; the stopped
    # simulation's files are removed afterwards.
    with tempfile.TemporaryDirectory(prefix="multi-vault-e2e-") as directory:
        run_dir = Path(directory) / f"multi-vault-{uuid.uuid4().hex[:8]}"
        try:
            # 1. Run the script from a fresh run directory with deployment and subscription.
            result = _run_script(run_dir, "--deploy", "--subscribe")
            assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-6000:]
            assert "Investor redeemed" in result.stdout

            # 2. Check that every target is complete and no trade is unfinished.
            state = State.read_json_file(run_dir / "state/multi-vault-cli.json")
            universe = load_universe(REPOSITORY / STRATEGY_FILE)
            for target in ("lighter", "atoma", "upshift"):
                assert get_target_phase(state, get_target_pair(universe, target)) == "done"
            assert all(trade.is_success() for trade in state.portfolio.get_all_trades())
            lighter_trades = [t for t in state.portfolio.get_all_trades() if t.pair.is_lighter_pool()]
            assert [t.other_data["lighter_pool"]["phase"] for t in lighter_trades] == ["complete", "complete"]

            # 3. Check that a rerun after the forks stopped refuses instead of redeploying.
            rerun = _run_script(run_dir)
            assert rerun.returncode == 1
            assert "No running simulation" in rerun.stdout
        finally:
            shutil.rmtree(_get_simulation_dir(run_dir), ignore_errors=True)
