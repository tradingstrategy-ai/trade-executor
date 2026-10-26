"""Black-box run of the Lighter transfer-accounting tutorial on the fixed Ethereum fork.

The tutorial ``scripts/lagoon/manual-trade-executor-lighter-transfers.py``
runs as a separate process in its simulate mode, reusing the shared Anvil
fork and installing the fake Lighter sequencer itself. Every Safe, guard,
Lagoon and ZkLighter transaction is real; only Lighter's sequencer is faked,
because Anvil cannot create Lighter accounts, execute L2 transfers or prove
withdrawals.
"""

import json
import os
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest
from eth_defi.provider.anvil import AnvilLaunch
from eth_defi.testing.anvil_fork_pool import AnvilForkPool
from eth_defi.testing.evm_snapshot_fixture import evm_snapshot_revert
from eth_defi.testing.fork_blocks import ETHEREUM_MIDNIGHT_BLOCK
from eth_defi.token import USDC_WHALE

from tradeexecutor.exchange_account.lighter import LIGHTER_MASTER_LABEL
from tradeexecutor.exchange_account.sub_account import get_lighter_account_resolution, get_sub_account_id_for_position
from tradeexecutor.state.state import State
from tradeexecutor.testing.lighter_scripted_transfers import SCRIPTED_TRANSFER_UNIT_USDC, build_lighter_transfer_schedule, verify_lighter_scripted_transfers
from tradeexecutor.testing.lighter_tutorial import SIMULATED_TRANSFER_FEE_USDC

#: Optional upstream RPC used to enable the fixed-block Ethereum fork.
JSON_RPC_ETHEREUM = os.environ.get("JSON_RPC_ETHEREUM")
#: API key required by the strategy factory even though this strategy has no market data.
TRADING_STRATEGY_API_KEY = os.environ.get("TRADING_STRATEGY_API_KEY")

pytestmark = [
    pytest.mark.skipif(
        not JSON_RPC_ETHEREUM or not TRADING_STRATEGY_API_KEY,
        reason="JSON_RPC_ETHEREUM and TRADING_STRATEGY_API_KEY are required",
    ),
    pytest.mark.warm_rpc_test_group,
    pytest.mark.xdist_group("fork:ethereum:midnight"),
]

#: Tutorial under test.
TUTORIAL = Path(__file__).resolve().parents[2] / "scripts" / "lagoon" / "manual-trade-executor-lighter-transfers.py"
#: Base amount of the scripted movements.
UNIT_USDC = SCRIPTED_TRANSFER_UNIT_USDC
#: Sub-account labels declared by the strategy, in order.
SUB_LABELS = ["trend", "arb"]
#: Fee the fake sequencer charges per transfer, exercising the fee accounting.
TRANSFER_FEE_USDC = SIMULATED_TRANSFER_FEE_USDC


@pytest.fixture()
def anvil_ethereum(anvil_fork_pool: AnvilForkPool) -> AnvilLaunch:
    """Reset the shared fixed Ethereum fork around the test."""
    launch = anvil_fork_pool.get_launch(
        JSON_RPC_ETHEREUM,
        ETHEREUM_MIDNIGHT_BLOCK,
        unlocked_addresses=[USDC_WHALE[1]],
        inherit_stdio=True,
    )
    snapshot = evm_snapshot_revert(launch)
    next(snapshot)
    try:
        yield launch
    finally:
        next(snapshot, None)


@pytest.mark.timeout(900)
def test_lighter_transfer_tutorial_simulated(anvil_ethereum: AnvilLaunch, tmp_path: Path) -> None:
    """Run the transfer tutorial in simulate mode and check it books every movement as a trade.

    The tutorial is the executable specification of Lighter custody
    accounting: each Safe deposit, transfer between accounts and withdrawal
    must be one successful trade between the right positions, with the fee
    booked on the source. Running it as a black box also proves the
    sub-account signer option of ``start`` and the tutorial's own verifier.

    1. Run the tutorial as a subprocess against the shared fork, in simulate mode with a transfer fee.
    2. Check it completed every phase and left no secrets in its output.
    3. Verify the saved state independently, check the state maps ids to accounts and positions, and check the eight movements and their fees.
    """
    # 1. Run the tutorial as a subprocess against the shared fork, in simulate mode with a transfer fee.
    run_dir = tmp_path / "run"
    environment = {
        **os.environ,
        "SIMULATE": "true",
        "LIGHTER_SIMULATE_RPC_URL": anvil_ethereum.json_rpc_url,
        "LIGHTER_RUN_DIR": str(run_dir),
    }
    result = subprocess.run([sys.executable, str(TUTORIAL)], env=environment, capture_output=True, text=True, timeout=800)
    assert result.returncode == 0, f"Tutorial failed:\n{result.stdout[-4000:]}\n{result.stderr[-4000:]}"

    # 2. Check it completed every phase and left no secrets in its output.
    progress = json.loads((run_dir / "progress.json").read_text())
    assert progress["phase"] == 7
    operator_record = json.loads((run_dir / "vault-record.json").read_text())
    master_key = operator_record["deployments"]["ethereum"]["lighter_account_setup"]["private_key"]
    output = result.stdout + result.stderr
    assert master_key not in output
    for account_id, index in zip(SUB_LABELS, progress["account_indices"][1:]):
        handover = json.loads((run_dir / f"vault-record.nautilus.{account_id}.json").read_text())
        assert handover["private_key"] not in output
        assert (handover["account_index"], handover["label"], handover["module_address"]) == (index, account_id, progress["module_address"])
    assert "All 8 scripted transfers were booked correctly" in output

    # 3. Verify the saved state independently, check the state maps ids to accounts and positions, and check the eight movements and their fees.
    state = State.read_json_file(run_dir / "state.json")
    accounts = get_lighter_account_resolution(state)
    assert accounts == {LIGHTER_MASTER_LABEL: progress["account_indices"][0], **dict(zip(SUB_LABELS, progress["account_indices"][1:]))}
    assert get_sub_account_id_for_position(state, next(position for position in state.portfolio.get_all_positions() if position.pair.get_ticker() == "LIGHTER-arb-USDC")) == "arb"
    rows = verify_lighter_scripted_transfers(state, build_lighter_transfer_schedule(SUB_LABELS, UNIT_USDC), accounts)
    assert [(row["kind"], row["source"], row["destination"]) for row in rows] == [
        ("deposit", "safe", "master"),
        ("funding", "master", "trend"),
        ("funding", "master", "arb"),
        ("funding", "trend", "arb"),
        ("funding", "arb", "master"),
        ("withdrawal", "trend", "safe"),
        ("withdrawal", "arb", "safe"),
        ("withdrawal", "master", "safe"),
    ]
    assert {position.pair.get_ticker() for position in state.portfolio.get_all_positions() if position.is_exchange_account()} == {"LIGHTER-ACCOUNT-USDC", "LIGHTER-trend-USDC", "LIGHTER-arb-USDC"}
    assert [row["fee"] for row in rows if row["kind"] == "funding"] == [TRANSFER_FEE_USDC] * 4
    assert rows[0]["amount"] == 5 * UNIT_USDC - 1
    assert sum((row["amount"] for row in rows if row["kind"] == "withdrawal"), Decimal(0)) == 5 * UNIT_USDC - 4 * TRANSFER_FEE_USDC
