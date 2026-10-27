"""Black-box ``check-accounts`` and ``correct-accounts`` for Lighter sub-account balance changes.

An external trader's profit, loss or own transfer changes a sub-account's
equity without an executor trade. The commands must report and record such
changes per sub-account. The vault runs on the fixed Ethereum fork through
the tutorial helpers; only Lighter's sequencer is faked.
"""

import logging
import os
from decimal import Decimal
from pathlib import Path

import pytest
from eth_defi.compat import native_datetime_utc_now
from eth_defi.provider.anvil import AnvilLaunch
from eth_defi.testing.anvil_fork_pool import AnvilForkPool
from eth_defi.testing.evm_snapshot_fixture import evm_snapshot_revert
from eth_defi.testing.fork_blocks import ETHEREUM_MIDNIGHT_BLOCK
from eth_defi.token import USDC_WHALE
from pytest import CaptureFixture, LogCaptureFixture, MonkeyPatch

from tradeexecutor.state.balance_update import BalanceUpdateCause
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeFlag
from tradeexecutor.testing.lighter_scripted_transfers import SCRIPTED_STEP_KEY, SCRIPTED_TRANSFER_UNIT_USDC
from tradeexecutor.testing.lighter_tutorial import (
    SIMULATED_TRANSFER_FEE_USDC,
    STRATEGY_DIR,
    Tutorial,
    deploy,
    get_cli_environment,
    initialise_state,
    load_tutorial_config,
    prepare_tutorial,
    run_cli,
    run_start,
    subscribe,
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

#: Base amount of the scripted movements.
UNIT_USDC = SCRIPTED_TRANSFER_UNIT_USDC
#: Subscription of the scripted strategy: five units minus the 1 USDC activation deposit.
SUBSCRIPTION_USDC = 5 * UNIT_USDC - Decimal(1)
#: Trader profit credited to the first sub-account outside the executor.
TREND_PROFIT_USDC = Decimal(3)
#: Trader loss debited from the second sub-account outside the executor.
ARB_LOSS_USDC = Decimal("0.5")
#: USDC the trader moves from the first sub-account to the second with its own key.
TRADER_TRANSFER_USDC = Decimal(1)


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
    """Prepare the transfer tutorial in simulate mode on the shared fork."""
    monkeypatch.setenv("SIMULATE", "true")
    monkeypatch.setenv("LIGHTER_SIMULATE_RPC_URL", anvil_ethereum.json_rpc_url)
    monkeypatch.setenv("LIGHTER_RUN_DIR", str(tmp_path / "run"))
    config = load_tutorial_config(executor_id="lighter-corrections", strategy_file=STRATEGY_DIR / "lighter_scripted_transfers_strategy.py", subscription_usdc=SUBSCRIPTION_USDC)
    tutorial = prepare_tutorial(config)
    try:
        yield tutorial
    finally:
        tutorial.close()


def _positions_by_account(state: State) -> dict[int, TradingPosition]:
    """Return the Lighter positions keyed by account index."""
    return {int(position.pair.get_exchange_account_id()): position for position in state.portfolio.get_open_positions() if position.is_exchange_account()}


def _run_and_read(tutorial: Tutorial, args: list[str], caplog: LogCaptureFixture, capsys: CaptureFixture[str], **overrides: str) -> tuple[int, str]:
    """Run a command and return its exit code with everything it logged and printed."""
    caplog.clear()
    capsys.readouterr()
    with caplog.at_level(logging.INFO):
        code = run_cli(args, get_cli_environment(tutorial, **overrides), check=False)
    # The command installs its own console handler, so read both captures.
    captured = capsys.readouterr()
    return code, captured.out + captured.err + caplog.text


@pytest.mark.timeout(900)
def test_cli_lighter_sub_account_balance_corrections(tutorial: Tutorial, caplog: LogCaptureFixture, capsys: CaptureFixture[str]) -> None:
    """Report and record sub-account balance changes made outside the executor.

    ``check-accounts`` must name each mismatched sub-account, and
    ``correct-accounts`` must book the net change as a balance update on that
    account's position, so the NAV and later custody decisions use the real
    balances while the master stays untouched.

    1. Deploy a vault with two sub-accounts, initialise state, subscribe and fund both sub-accounts with the scripted strategy.
    2. Check accounts are clean, then change the fake sequencer's balances: a profit on one sub-account, a loss on the other and a trader transfer between them.
    3. Check that check-accounts fails and names both sub-accounts.
    4. Run correct-accounts and check each sub-account position carries one balance update for its net change, in the accounting log, with the master unchanged.
    5. Check that check-accounts passes, that the NAV matches Lighter, and that the next start cycle continues the script from the corrected balances.
    6. Interrupt the next scripted transfer: check-accounts reports it as pending, correct-accounts --dry-run previews its reconciliation without saving, and correct-accounts completes it.
    7. Run the last scripted cycle, which drains every account, and check that check-accounts passes.
    8. Freeze a sub-account position and credit its account: correct-accounts revalues the frozen position without opening a second one.
    9. Stop declaring the sub-accounts: correct-accounts closes the drained one, reports the funded one with the drain command, and closes it once it holds only Lighter dust.
    """
    # 1. Deploy a vault with two sub-accounts, initialise state, subscribe and fund both sub-accounts with the scripted strategy.
    deploy(tutorial, "Lighter corrections", "LCT")
    initialise_state(tutorial)
    signers = {"LIGHTER_SUB_ACCOUNT_OPERATOR_RECORD_FILES": ",".join(str(tutorial.get_handover_file(account_id)) for account_id in tutorial.sub_account_ids)}
    subscribe(tutorial, **signers)
    run_start(tutorial, **signers)
    master, trend, arb = tutorial.account_indices
    # The fake sequencer stands in for Lighter's L2, which Anvil cannot run; its balances are the "real" ones here.
    fake = tutorial.simulation.fake_core
    fee = SIMULATED_TRANSFER_FEE_USDC
    assert [fake.get_account(index).collateral for index in (master, trend, arb)] == [UNIT_USDC - 2 * fee, 2 * UNIT_USDC, 2 * UNIT_USDC]
    state = State.read_json_file(tutorial.state_file)
    positions = _positions_by_account(state)
    updates_before = {index: set(position.balance_updates) for index, position in positions.items()}

    # 2. Check accounts are clean, then change the fake sequencer's balances: a profit on one sub-account, a loss on the other and a trader transfer between them.
    run_cli(["check-accounts"], get_cli_environment(tutorial))
    fake.get_account(trend).collateral += TREND_PROFIT_USDC - TRADER_TRANSFER_USDC
    fake.get_account(arb).collateral += TRADER_TRANSFER_USDC - ARB_LOSS_USDC
    trend_change = TREND_PROFIT_USDC - TRADER_TRANSFER_USDC
    arb_change = TRADER_TRANSFER_USDC - ARB_LOSS_USDC

    # 3. Check that check-accounts fails and names both sub-accounts.
    caplog.clear()
    capsys.readouterr()
    with caplog.at_level(logging.INFO), pytest.raises(SystemExit) as exit_info:
        run_cli(["check-accounts"], get_cli_environment(tutorial, LOG_LEVEL="info"))
    assert exit_info.value.code == 1
    # The command installs its own console handler, so read both captures.
    captured = capsys.readouterr()
    report = captured.out + captured.err + caplog.text
    assert "LIGHTER-trend-USDC" in report and "LIGHTER-arb-USDC" in report

    # 4. Run correct-accounts and check each sub-account position carries one balance update for its net change, in the accounting log, with the master unchanged.
    run_cli(["correct-accounts"], get_cli_environment(tutorial))
    state = State.read_json_file(tutorial.state_file)
    positions = _positions_by_account(state)
    recorded_ids = {ref.balance_event_id for ref in state.sync.accounting.balance_update_refs}
    for index, change in ((trend, trend_change), (arb, arb_change)):
        new_updates = [update for update_id, update in positions[index].balance_updates.items() if update_id not in updates_before[index]]
        assert len(new_updates) == 1, f"Account {index} got {len(new_updates)} corrections"
        update = new_updates[0]
        assert update.quantity == change
        assert update.cause == BalanceUpdateCause.vault_flow
        assert update.position_id == positions[index].position_id
        assert update.balance_update_id in recorded_ids
    assert set(positions[master].balance_updates) == updates_before[master]
    assert {index: position.get_quantity() for index, position in positions.items()} == {index: fake.get_account(index).collateral for index in (master, trend, arb)}

    # 5. Check that check-accounts passes, that the NAV matches Lighter, and that the next start cycle continues the script from the corrected balances.
    run_cli(["check-accounts"], get_cli_environment(tutorial))
    lighter_total = sum((fake.get_account(index).collateral for index in (master, trend, arb)), Decimal(0))
    assert state.portfolio.get_net_asset_value() == pytest.approx(float(tutorial.usdc.fetch_balance_of(tutorial.safe_address) + lighter_total))
    run_start(tutorial, **signers)
    state = State.read_json_file(tutorial.state_file)
    positions = _positions_by_account(state)
    step_3 = [trade for trade in state.portfolio.get_all_trades() if (trade.other_data or {}).get(SCRIPTED_STEP_KEY) == 3]
    assert len(step_3) == 1 and step_3[0].is_success()
    assert positions[trend].get_quantity() == 2 * UNIT_USDC + trend_change - UNIT_USDC - fee
    assert positions[arb].get_quantity() == 2 * UNIT_USDC + arb_change + UNIT_USDC
    assert {index: position.get_quantity() for index, position in positions.items()} == {index: fake.get_account(index).collateral for index in (master, trend, arb)}
    state.check_if_clean()

    # 6. Interrupt the next scripted transfer: check-accounts reports it as pending, correct-accounts --dry-run previews its reconciliation without saving, and correct-accounts completes it.
    fake.execute_transfers = False
    with pytest.raises(TimeoutError):
        run_start(tutorial, **signers)
    state = State.read_json_file(tutorial.state_file)
    [pending] = [trade for trade in state.portfolio.get_all_trades() if trade.is_exchange_account_transfer_pending()]
    assert TradeFlag.internal_account_transfer in pending.flags and pending.other_data["lighter_transfer_tx_hash"]
    code, report = _run_and_read(tutorial, ["check-accounts"], caplog, capsys, **signers)
    assert code == 1
    assert f"Lighter transfer trade #{pending.trade_id}" in report and "is pending" in report
    assert "LIGHTER-ACCOUNT-USDC" in report, "The balance table is still printed"
    fake.execute_pending_transfers()
    fake.execute_transfers = True
    saved = tutorial.state_file.read_bytes()
    code, report = _run_and_read(tutorial, ["correct-accounts", "--dry-run"], caplog, capsys, LOG_LEVEL="info", **signers)
    assert code == 0
    assert "Reconciled 1 verified Lighter transfer(s) (dry run, not saved)" in report
    assert tutorial.state_file.read_bytes() == saved
    run_cli(["correct-accounts"], get_cli_environment(tutorial, **signers))
    state = State.read_json_file(tutorial.state_file)
    assert state.portfolio.get_trade_by_id(pending.trade_id).is_success()

    # 7. Run the last scripted cycle, which drains every account, and check that check-accounts passes.
    run_start(tutorial, **signers)
    assert [fake.get_account(index).collateral for index in (master, trend, arb)] == [0, 0, 0]
    run_cli(["check-accounts"], get_cli_environment(tutorial))
    state = State.read_json_file(tutorial.state_file)
    assert [position.get_quantity() for position in _positions_by_account(state).values()] == [0, 0, 0]

    # 8. Freeze a sub-account position and credit its account: correct-accounts revalues the frozen position without opening a second one.
    trend_position = _positions_by_account(state)[trend]
    del state.portfolio.open_positions[trend_position.position_id]
    state.portfolio.frozen_positions[trend_position.position_id] = trend_position
    trend_position.frozen_at = native_datetime_utc_now()
    state.write_json_file(tutorial.state_file)
    updates_before = set(trend_position.balance_updates)
    fake.get_account(trend).collateral += TREND_PROFIT_USDC
    run_cli(["correct-accounts"], get_cli_environment(tutorial))
    state = State.read_json_file(tutorial.state_file)
    trend_position = state.portfolio.frozen_positions[trend_position.position_id]
    assert [update.quantity for update_id, update in trend_position.balance_updates.items() if update_id not in updates_before] == [TREND_PROFIT_USDC]
    assert trend not in _positions_by_account(state), "No second position was opened for the frozen account"
    run_cli(["check-accounts"], get_cli_environment(tutorial))
    # Return the position to the open positions, as repair would, for the next step.
    del state.portfolio.frozen_positions[trend_position.position_id]
    state.portfolio.open_positions[trend_position.position_id] = trend_position
    trend_position.frozen_at = None
    state.write_json_file(tutorial.state_file)

    # 9. Stop declaring the sub-accounts: correct-accounts closes the drained one, reports the funded one with the drain command, and closes it once it holds only Lighter dust.
    master_only = {"STRATEGY_FILE": str(STRATEGY_DIR / "minimal_lighter_strategy.py"), "LIGHTER_ACCOUNT_INDEX": str(master)}
    code, report = _run_and_read(tutorial, ["check-accounts"], caplog, capsys, **master_only)
    assert code == 1 and f"positions track Lighter accounts {sorted([trend, arb])} the strategy does not declare" in report
    code, report = _run_and_read(tutorial, ["correct-accounts"], caplog, capsys, **master_only)
    assert code == 1 and f"lighter-move-funds --lighter-account-index {trend}" in report
    state = State.read_json_file(tutorial.state_file)
    assert set(_positions_by_account(state)) == {master, trend}, "The drained, undeclared account was closed"
    fake.get_account(trend).collateral = Decimal("0.4")
    run_cli(["correct-accounts"], get_cli_environment(tutorial, **master_only))
    state = State.read_json_file(tutorial.state_file)
    assert set(_positions_by_account(state)) == {master}
    closed_trend = state.portfolio.closed_positions[trend_position.position_id]
    assert closed_trend.get_quantity() == 0
    assert [update.cause for update in closed_trend.balance_updates.values()][-1] == BalanceUpdateCause.correction
    run_cli(["check-accounts"], get_cli_environment(tutorial, **master_only))
