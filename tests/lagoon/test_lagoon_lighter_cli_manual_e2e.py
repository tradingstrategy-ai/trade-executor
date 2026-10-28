"""Black-box run of the command-line Lighter sub-account manual test.

``scripts/lagoon/manual-lighter-subaccounts-cli.sh`` runs on mainnet only.
This test runs the same commands in the same order on the fixed Ethereum
fork, with only Lighter's sequencer faked: deployment with a key backup,
subscription through ``lagoon-settle``, sub-account funding with
``perform-test-trade --pair``, a ``--lighter-transfer`` between the
sub-accounts, the unwind through ``lighter-move-funds`` and
``lagoon-redeem``, and the ``lighter_manual`` helpers the script calls.

The fake sequencer charges a 0.5 USDC transfer fee, so the master and the
``trend`` sub-account end below Lighter's 1 USDC withdrawal minimum and the
unwind takes both its withdraw and its skip path.
"""

import io
import os
from decimal import Decimal
from pathlib import Path

import pytest
from eth_defi.provider.anvil import AnvilLaunch
from eth_defi.testing.anvil_fork_pool import AnvilForkPool
from eth_defi.testing.evm_snapshot_fixture import evm_snapshot_revert
from eth_defi.testing.fork_blocks import ETHEREUM_MIDNIGHT_BLOCK
from eth_defi.token import USDC_WHALE
from pytest import CaptureFixture, MonkeyPatch

from tradeexecutor.state.state import State
from tradeexecutor.testing import lighter_manual
from tradeexecutor.testing.lighter_tutorial import (
    SIMULATED_TRANSFER_FEE_USDC,
    STRATEGY_DIR,
    Tutorial,
    get_base_environment,
    get_cli_environment,
    initialise_state,
    load_tutorial_config,
    prepare_tutorial,
    record_deployment,
    run_cli,
)

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

#: Test capital, as in the script's default.
SUBSCRIPTION_USDC = Decimal(10)
#: USDC each sub-account is funded with.
FUNDING_USDC = Decimal(2)
#: USDC moved from trend to arb.
TRANSFER_USDC = Decimal(1)
#: The simulated fee of one transfer between Lighter accounts.
FEE = SIMULATED_TRANSFER_FEE_USDC


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


@pytest.fixture()
def tutorial(anvil_ethereum: AnvilLaunch, tmp_path: Path, monkeypatch: MonkeyPatch) -> Tutorial:
    """Prepare a two-sub-account vault in simulate mode on the shared fork."""
    monkeypatch.setenv("SIMULATE", "true")
    monkeypatch.setenv("LIGHTER_SIMULATE_RPC_URL", anvil_ethereum.json_rpc_url)
    monkeypatch.setenv("LIGHTER_RUN_DIR", str(tmp_path / "run"))
    config = load_tutorial_config(executor_id="lighter-cli", strategy_file=STRATEGY_DIR / "lighter_multi_subaccount_strategy.py", subscription_usdc=SUBSCRIPTION_USDC)
    tutorial = prepare_tutorial(config)
    try:
        yield tutorial
    finally:
        tutorial.close()


def manual(capsys: CaptureFixture, *args: str) -> str:
    """Run one ``lighter_manual`` subcommand in-process, so it sees the fake sequencer, and return its result line."""
    capsys.readouterr()
    code = lighter_manual.main(list(args))
    out = capsys.readouterr().out.strip()
    assert code == 0, f"lighter_manual {' '.join(args)} failed: {out}"
    return out


def expect(capsys: CaptureFixture, **balances: Decimal) -> None:
    """Assert the Safe's and the Lighter accounts' balances through ``expect-balances``."""
    assert manual(capsys, "expect-balances", *[f"{name}={value}" for name, value in balances.items()]) == "ok"


@pytest.mark.timeout(900)
def test_cli_manual_lighter_sub_accounts_flow(tutorial: Tutorial, tmp_path: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture) -> None:
    """Run the manual test script's commands end to end and check every balance and booking.

    The script is mainnet only, so its command sequence is tested here on the
    fork: if any command, option or helper it relies on changes, this fails.

    1. Record the deployer's starting balances with check-funds
    2. Deploy with a key backup directory and check the backups are removed after success
    3. Initialise the state and book the activation deposit
    4. Subscribe with request-deposit, lagoon-settle and claim-shares, and check a rerun does not subscribe again
    5. Fund trend and arb with perform-test-trade --pair --buy-only
    6. Check --lighter-transfer refuses before moving anything without every signer
    7. Move USDC from trend to arb with --lighter-transfer and check the booking
    8. Unwind: withdraw arb with lighter-move-funds answered over stdin, skip the accounts below the minimum
    9. Redeem every share, sweep the Safe and reconcile the deployer's balances
    """
    run_dir = tutorial.config.run_dir
    start_file = run_dir / "start.env"
    for key, value in get_base_environment(tutorial).items():
        monkeypatch.setenv(key, value)

    # 1. Record the deployer's starting balances with check-funds
    assert manual(capsys, "check-funds", "--usdc", str(SUBSCRIPTION_USDC + 1), "--start-file", str(start_file)) == "ok"
    assert manual(capsys, "check-funds", "--usdc", "1", "--start-file", str(start_file)) == "recorded", "A rerun must keep the original start balances"

    # 2. Deploy with a key backup directory and check the backups are removed after success
    backup_dir = run_dir / "key-backup"
    run_cli(
        ["lagoon-deploy-vault"],
        {
            **get_base_environment(tutorial),
            "VAULT_RECORD_FILE": str(tutorial.record_file),
            "FUND_NAME": "Lighter CLI test",
            "FUND_SYMBOL": "LCLI",
            "MULTISIG_OWNERS": tutorial.deployer.address,
            "ANY_ASSET": "true",
            "PERFORMANCE_FEE": "0",
            "MANAGEMENT_FEE": "0",
            "GENERATE_LIGHTER_API_KEY": "true",
            "LIGHTER_KEY_BACKUP_DIR": str(backup_dir),
        },
    )
    record_deployment(tutorial)
    assert backup_dir.is_dir() and list(backup_dir.iterdir()) == [], "Every key was backed up, then removed once the records were written"
    assert backup_dir.stat().st_mode & 0o077 == 0
    master, trend, arb = tutorial.account_indices
    for key, value in get_cli_environment(tutorial).items():
        monkeypatch.setenv(key, value)

    # 3. Initialise the state and book the activation deposit
    initialise_state(tutorial)
    expect(capsys, safe=Decimal(0), master=Decimal(1), trend=Decimal(0), arb=Decimal(0))

    # 4. Subscribe with request-deposit, lagoon-settle and claim-shares, and check a rerun does not subscribe again
    marker = run_dir / "deposit-requested"
    assert manual(capsys, "request-deposit", str(SUBSCRIPTION_USDC), "--marker", str(marker)) == "requested"
    assert manual(capsys, "request-deposit", str(SUBSCRIPTION_USDC), "--marker", str(marker)) == "pending"
    run_cli(["lagoon-settle"], get_cli_environment(tutorial))
    assert manual(capsys, "claim-shares").startswith("held ")
    assert manual(capsys, "request-deposit", str(SUBSCRIPTION_USDC), "--marker", str(marker)) == "settled"
    run_cli(["check-accounts"], get_cli_environment(tutorial))
    expect(capsys, safe=SUBSCRIPTION_USDC, master=Decimal(1), trend=Decimal(0), arb=Decimal(0))

    # 5. Fund trend and arb with perform-test-trade --pair --buy-only
    for name, master_after in (("trend", Decimal(1) - FEE), ("arb", Decimal(1) - 2 * FEE)):
        run_cli(["perform-test-trade", "--pair", f"(ethereum, lighter, LIGHTER-{name}, USDC)", "--buy-only", "--amount", str(FUNDING_USDC)], get_cli_environment(tutorial))
        run_cli(["check-accounts"], get_cli_environment(tutorial))
    expect(capsys, safe=SUBSCRIPTION_USDC - 2 * FUNDING_USDC, master=master_after, trend=FUNDING_USDC, arb=FUNDING_USDC)

    # 6. Check --lighter-transfer refuses before moving anything without every signer
    trade_count = len(list(State.read_json_file(tutorial.state_file).portfolio.get_all_trades()))
    transfer = ["perform-test-trade", "--lighter-transfer", "trend:arb", "--amount", str(TRANSFER_USDC)]
    with pytest.raises(ValueError, match=r"no signer for Lighter account\(s\) \['trend'\]"):
        run_cli([*transfer, "--buy-only"], get_cli_environment(tutorial))
    with pytest.raises(ValueError, match=r"no signer for Lighter account\(s\) \['arb'\]"):
        run_cli(transfer, get_cli_environment(tutorial, LIGHTER_SUB_ACCOUNT_OPERATOR_RECORD_FILES=str(tutorial.get_handover_file("trend"))))
    assert len(list(State.read_json_file(tutorial.state_file).portfolio.get_all_trades())) == trade_count
    expect(capsys, trend=FUNDING_USDC, arb=FUNDING_USDC)

    # 7. Move USDC from trend to arb with --lighter-transfer and check the booking
    signers = ",".join(str(tutorial.get_handover_file(name)) for name in ("trend", "arb"))
    run_cli([*transfer, "--buy-only"], get_cli_environment(tutorial, LIGHTER_SUB_ACCOUNT_OPERATOR_RECORD_FILES=signers))
    run_cli(["check-accounts"], get_cli_environment(tutorial, LIGHTER_SUB_ACCOUNT_OPERATOR_RECORD_FILES=signers))
    trend_left = FUNDING_USDC - TRANSFER_USDC - FEE
    expect(capsys, master=Decimal(0), trend=trend_left, arb=FUNDING_USDC + TRANSFER_USDC)
    state = State.read_json_file(tutorial.state_file)
    [booked] = [trade for trade in state.portfolio.get_all_trades() if trade.trade_id > trade_count]
    assert booked.is_internal_account_transfer() and booked.is_success()
    assert int(booked.pair.get_exchange_account_id()) == arb and int(booked.other_data["funding_source_account_id"]) == trend
    assert booked.executed_quantity == pytest.approx(TRANSFER_USDC) and Decimal(booked.other_data["funding_fee_usdc"]) == FEE

    # 8. Unwind: withdraw arb with lighter-move-funds answered over stdin, skip the accounts below the minimum
    plan = manual(capsys, "withdrawal-plan", str(arb))
    assert plan == f"withdraw {lighter_manual.format_usdc(FUNDING_USDC + TRANSFER_USDC)}"
    monkeypatch.setattr("sys.stdin", io.StringIO(f"w\n{plan.removeprefix('withdraw ')}\ny\n"))
    run_cli(["lighter-move-funds"], get_cli_environment(tutorial, LIGHTER_ACCOUNT_INDEX=str(arb)))
    assert manual(capsys, "withdrawal-plan", str(arb)) == "skip"
    assert manual(capsys, "withdrawal-plan", str(trend)) == "skip"
    assert manual(capsys, "withdrawal-plan", str(master)) == "skip"
    run_cli(["check-accounts"], get_cli_environment(tutorial))
    expect(capsys, safe=SUBSCRIPTION_USDC - 2 * FUNDING_USDC + FUNDING_USDC + TRANSFER_USDC, arb=Decimal(0))

    # 9. Redeem every share, sweep the Safe and reconcile the deployer's balances
    assert manual(capsys, "redeem-status") == "redeem"
    run_cli(["lagoon-redeem"], get_cli_environment(tutorial))
    assert manual(capsys, "redeem-status") == "done"
    assert manual(capsys, "sweep-safe").startswith("swept ")
    assert manual(capsys, "reconcile", "--start-file", str(start_file)) == "ok"
    assert tutorial.usdc.fetch_balance_of(tutorial.safe_address) == 0
