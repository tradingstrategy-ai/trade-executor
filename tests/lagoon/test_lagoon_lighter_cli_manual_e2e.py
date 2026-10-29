"""Black-box run of the command-line Lighter sub-account manual test.

``scripts/lagoon/manual-lighter-subaccounts-cli.sh`` runs on mainnet only.
This test runs the same commands in the same order on the fixed Ethereum
fork, with only Lighter's sequencer faked: deployment with a key backup,
subscription through ``lagoon-settle``, sub-account funding with
``perform-test-trade --pair``, a ``--lighter-transfer`` between the
sub-accounts, the audit of clean books with ``check-wallet``,
``correct-accounts`` and ``lagoon-settle``, the detection and correction of
a trader's transfer and a direct Safe deposit the executor did not make, the
unwind through ``lighter-move-funds`` and ``lagoon-redeem``, and the
``lighter_manual`` helpers the script calls.

The fake sequencer charges a 0.5 USDC transfer fee. The amounts are larger
than the script's, and the master gets a 1 USDC top-up, so every account
ends above Lighter's 1 USDC withdrawal minimum and the unwind withdraws all
three: two sub-accounts with the key-free L1 withdrawal and the master with
the secure withdrawal, all through ``lighter-move-funds`` as the script does.
"""

import io
import os
import shutil
from decimal import Decimal
from pathlib import Path

import pytest
from eth_defi.provider.anvil import AnvilLaunch
from eth_defi.provider.receipt import wait_for_transaction_receipt_robust
from eth_defi.testing.anvil_fork_pool import AnvilForkPool
from eth_defi.testing.evm_snapshot_fixture import evm_snapshot_revert
from eth_defi.testing.fork_blocks import ETHEREUM_MIDNIGHT_BLOCK
from eth_defi.token import USDC_WHALE
from pytest import CaptureFixture, MonkeyPatch

from eth_defi.compat import native_datetime_utc_now

from tradeexecutor.cli.commands.check_accounts import ACCOUNTS_DO_NOT_MATCH
from tradeexecutor.exchange_account.state import create_exchange_account_transfer
from tradeexecutor.state.state import State, UncleanState
from tradeexecutor.testing import lighter_manual
from tradeexecutor.testing.lighter_tutorial import (
    SIMULATED_TRANSFER_FEE_USDC,
    STRATEGY_DIR,
    Tutorial,
    deploy,
    get_base_environment,
    get_cli_environment,
    load_tutorial_config,
    prepare_tutorial,
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
#: USDC each sub-account is funded with; more than the script's 2, so both stay above the withdrawal minimum after the fees.
FUNDING_USDC = Decimal(3)
#: USDC deposited to the master after the fees emptied it, so its secure withdrawal is tested too.
MASTER_TOP_UP_USDC = Decimal(1)
#: USDC moved from trend to arb.
TRANSFER_USDC = Decimal(1)
#: USDC the trader of arb moves back to trend outside the executor.
TRADER_TRANSFER_USDC = Decimal(1)
#: USDC the deployer sends straight to the Safe outside the vault.
DONATION_USDC = Decimal(1)
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

    1. Fund the deployer for the donation too and record its starting balances with check-funds
    2. Deploy with a key backup directory and check the backups are removed after success
    3. Initialise the state, book the activation deposit and check the wallet
    4. Subscribe with request-deposit, lagoon-settle and claim-shares, and check a rerun does not subscribe again
    5. Fund trend and arb with perform-test-trade --pair --buy-only
    6. Check --lighter-transfer refuses before moving anything without every signer
    7. Move USDC from trend to arb with --lighter-transfer and check the booking
    8. Audit clean books: check-wallet, a correct-accounts that books nothing, and a settled NAV that matches the balances
    9. Move USDC between the sub-accounts with the trader's key, check check-accounts detects it and correct-accounts books it
    10. Send USDC straight to the Safe, check check-accounts reports it with its mismatch message, correct-accounts books it and the NAV rises
    11. Check a withdrawal started without a recorded request is reported as unconfirmed, not resumed
    12. Top up the master, then withdraw every account with lighter-move-funds answered over stdin
    13. Redeem every share, sweep the Safe and reconcile the deployer's balances
    """
    run_dir = tutorial.config.run_dir
    start_file = run_dir / "start.env"
    for key, value in get_base_environment(tutorial).items():
        monkeypatch.setenv(key, value)

    # 1. Fund the deployer for the donation too and record its starting balances with check-funds
    tx_hash = tutorial.usdc.transfer(tutorial.deployer.address, DONATION_USDC).transact({"from": USDC_WHALE[1]})
    wait_for_transaction_receipt_robust(tutorial.web3, tx_hash)
    assert manual(capsys, "check-funds", "--usdc", str(SUBSCRIPTION_USDC + 1 + DONATION_USDC), "--start-file", str(start_file)) == "ok"
    assert manual(capsys, "check-funds", "--usdc", "1", "--start-file", str(start_file)) == "recorded", "A rerun must keep the original start balances"
    assert lighter_manual.main(["check-funds", "--usdc", "1", "--min-eth", "100000000", "--start-file", str(run_dir / "short.env")]) == 1, "Too little ETH must fail"

    # 2. Deploy with a key backup directory and check the backups are removed after success
    deploy(tutorial, "Lighter CLI test", "LCLI")
    backup_dir = tutorial.key_backup_dir
    assert backup_dir.is_dir() and list(backup_dir.iterdir()) == [], "Every key was backed up, then removed once the records were written"
    assert backup_dir.stat().st_mode & 0o077 == 0
    master, trend, arb = tutorial.account_indices
    # As in the script, every command after deployment gets both sub-account keys.
    signers = ",".join(str(tutorial.get_handover_file(name)) for name in ("trend", "arb"))
    env = get_cli_environment(tutorial, LIGHTER_SUB_ACCOUNT_OPERATOR_RECORD_FILES=signers)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    # 3. Initialise the state, book the activation deposit and check the wallet
    run_cli(["init"], env)
    run_cli(["correct-accounts", "--process-redemption"], env)
    run_cli(["check-accounts"], env)
    run_cli(["check-wallet"], env)
    expect(capsys, safe=Decimal(0), master=Decimal(1), trend=Decimal(0), arb=Decimal(0))

    # 4. Subscribe with request-deposit, lagoon-settle and claim-shares, and check a rerun does not subscribe again
    marker = run_dir / "deposit-requested"
    assert manual(capsys, "request-deposit", str(SUBSCRIPTION_USDC), "--marker", str(marker)) == "requested"
    assert manual(capsys, "request-deposit", str(SUBSCRIPTION_USDC), "--marker", str(marker)) == "pending"
    run_cli(["lagoon-settle"], env)
    assert manual(capsys, "claim-shares").startswith("held ")
    assert manual(capsys, "request-deposit", str(SUBSCRIPTION_USDC), "--marker", str(marker)) == "settled"
    run_cli(["check-accounts"], env)
    expect(capsys, safe=SUBSCRIPTION_USDC, master=Decimal(1), trend=Decimal(0), arb=Decimal(0))

    # 5. Fund trend and arb with perform-test-trade --pair --buy-only
    for name in ("trend", "arb"):
        run_cli(["perform-test-trade", "--pair", f"(ethereum, lighter, LIGHTER-{name}, USDC)", "--buy-only", "--amount", str(FUNDING_USDC)], env)
        run_cli(["check-accounts"], env)
    expect(capsys, safe=SUBSCRIPTION_USDC - 2 * FUNDING_USDC, master=Decimal(1) - 2 * FEE, trend=FUNDING_USDC, arb=FUNDING_USDC)

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
    run_cli([*transfer, "--buy-only"], env)
    run_cli(["check-accounts"], env)
    trend_left = FUNDING_USDC - TRANSFER_USDC - FEE
    expect(capsys, master=Decimal(0), trend=trend_left, arb=FUNDING_USDC + TRANSFER_USDC)
    state = State.read_json_file(tutorial.state_file)
    [booked] = [trade for trade in state.portfolio.get_all_trades() if trade.trade_id > trade_count]
    assert booked.is_internal_account_transfer() and booked.is_success()
    assert int(booked.pair.get_exchange_account_id()) == arb and int(booked.other_data["funding_source_account_id"]) == trend
    assert booked.executed_quantity == pytest.approx(TRANSFER_USDC) and Decimal(booked.other_data["funding_fee_usdc"]) == FEE

    # 8. Audit clean books: check-wallet, a correct-accounts that books nothing, and a settled NAV that matches the balances
    run_cli(["check-wallet"], env)
    books = manual(capsys, "books")
    run_cli(["correct-accounts", "--dry-run"], env)
    run_cli(["correct-accounts"], env)
    assert manual(capsys, "books") == books, "correct-accounts must book nothing when the books match"
    run_cli(["check-accounts"], env)
    run_cli(["lagoon-settle"], env)
    nav = SUBSCRIPTION_USDC + 1 - 3 * FEE
    assert manual(capsys, "expect-nav", "--nav", str(nav)) == "ok"
    run_cli(["show-valuation"], env)
    run_cli(["show-positions"], env)

    # 9. Move USDC between the sub-accounts with the trader's key, check check-accounts detects it and correct-accounts books it
    trader_marker = run_dir / "trader-transfer"
    assert manual(capsys, "trader-transfer", "arb", "trend", str(TRADER_TRANSFER_USDC), "--marker", str(trader_marker)) == "sent"
    assert manual(capsys, "trader-transfer", "arb", "trend", str(TRADER_TRANSFER_USDC), "--marker", str(trader_marker)) == "sent-earlier", "A rerun must not send again"
    trend_left += TRADER_TRANSFER_USDC
    arb_left = FUNDING_USDC + TRANSFER_USDC - TRADER_TRANSFER_USDC - FEE
    expect(capsys, trend=trend_left, arb=arb_left)
    assert run_cli(["check-accounts"], env, check=False) == 1, "check-accounts must report a transfer the executor did not make"
    run_cli(["correct-accounts", "--dry-run"], env)
    assert manual(capsys, "books") == books, "A dry run must book nothing"
    run_cli(["correct-accounts"], env)
    assert manual(capsys, "books") != books, "correct-accounts must book the new equity"
    run_cli(["check-accounts"], env)
    run_cli(["lagoon-settle"], env)
    nav -= FEE
    assert manual(capsys, "expect-nav", "--nav", str(nav)) == "ok"

    # 10. Send USDC straight to the Safe, check check-accounts reports it with its mismatch message, correct-accounts books it and the NAV rises
    donation_marker = run_dir / "donation"
    assert manual(capsys, "donate", str(DONATION_USDC), "--marker", str(donation_marker)) == "sent"
    assert manual(capsys, "donate", str(DONATION_USDC), "--marker", str(donation_marker)) == "sent-earlier", "A rerun must not send again"
    expect(capsys, safe=SUBSCRIPTION_USDC - 2 * FUNDING_USDC + DONATION_USDC)
    # The script accepts the failure only with this message in the output, so pin it through the raising form.
    with pytest.raises(UncleanState, match=ACCOUNTS_DO_NOT_MATCH):
        run_cli(["check-accounts"], {**env, "RAISE_ON_UNCLEAN": "true"})
    run_cli(["correct-accounts"], env)
    run_cli(["check-accounts"], env)
    run_cli(["lagoon-settle"], env)
    nav += DONATION_USDC
    assert manual(capsys, "expect-nav", "--nav", str(nav)) == "ok"

    # 11. Check a withdrawal started without a recorded request is reported as unconfirmed, not resumed
    saved = tmp_path / "state-before-unconfirmed.json"
    shutil.copyfile(tutorial.state_file, saved)
    state = State.read_json_file(tutorial.state_file)
    arb_position = next(position for position in state.portfolio.open_positions.values() if position.pair.is_exchange_account() and int(position.pair.get_exchange_account_id()) == arb)
    started = create_exchange_account_transfer(state=state, position=arb_position, strategy_cycle_at=native_datetime_utc_now(), reserve_currency=arb_position.reserve_currency, amount=Decimal(1), deposit=False, notes="Interrupted before submission", metadata={})
    state.start_execution(native_datetime_utc_now(), started)
    state.write_json_file(tutorial.state_file)
    assert manual(capsys, "withdrawal-plan", str(arb)) == f"unconfirmed {started.trade_id}"
    shutil.copyfile(saved, tutorial.state_file)

    # 12. Top up the master, then withdraw every account with lighter-move-funds answered over stdin
    run_cli(["perform-test-trade", "--single-pair", "--buy-only", "--amount", str(MASTER_TOP_UP_USDC)], env)
    expect(capsys, master=MASTER_TOP_UP_USDC, trend=trend_left, arb=arb_left)
    safe = SUBSCRIPTION_USDC - 2 * FUNDING_USDC + DONATION_USDC - MASTER_TOP_UP_USDC
    for name, index, amount, options in (
        ("arb", arb, arb_left, ["--lighter-account-index", str(arb)]),
        ("trend", trend, trend_left, ["--lighter-account-index", str(trend)]),
        ("master", master, MASTER_TOP_UP_USDC, []),
    ):
        plan = manual(capsys, "withdrawal-plan", str(index))
        assert plan == f"withdraw {lighter_manual.format_usdc(amount)}", name
        monkeypatch.setattr("sys.stdin", io.StringIO(f"w\n{plan.removeprefix('withdraw ')}\ny\n"))
        run_cli(["lighter-move-funds", *options], env)
        run_cli(["check-accounts"], env)
        safe += amount
        assert manual(capsys, "withdrawal-plan", str(index)) == "skip"
        expect(capsys, safe=safe, **{name: Decimal(0)})

    # 13. Redeem every share, sweep the Safe and reconcile the deployer's balances
    assert manual(capsys, "redeem-status") == "redeem"
    run_cli(["lagoon-redeem"], env)
    assert manual(capsys, "redeem-status") == "done"
    assert manual(capsys, "sweep-safe").startswith("swept ")
    run_cli(["check-wallet"], env)
    assert manual(capsys, "reconcile", "--start-file", str(start_file), "--trader-transfer-marker", str(trader_marker)) == "ok"
    assert tutorial.usdc.fetch_balance_of(tutorial.safe_address) == 0
