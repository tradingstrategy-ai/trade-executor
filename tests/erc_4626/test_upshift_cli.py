"""Upshift through public CLI commands, real Lagoon custody and persisted state."""

import json
import os
from collections.abc import Iterator
from decimal import Decimal
from functools import partial
from pathlib import Path

import pytest
from eth_defi.erc_4626.classification import create_vault_instance_autodetect
from eth_defi.erc_4626.vault_protocol.lagoon.vault import LagoonVault
from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import fund_erc20_on_anvil
from eth_defi.provider.receipt import wait_for_transaction_receipt_robust
from eth_defi.testing.anvil_fork_pool import AnvilForkPool
from eth_defi.testing.evm_snapshot_fixture import evm_snapshot_revert
from eth_defi.testing.fork_blocks import ETHEREUM_MIDNIGHT_BLOCK
from eth_defi.token import USDC_NATIVE_TOKEN, fetch_erc20_details
from eth_defi.vault.base import VaultSpec
from pytest import MonkeyPatch, TempPathFactory
from typer.main import get_command
from web3 import Web3

from tradeexecutor.cli.main import app
from tradeexecutor.cli.commands import perform_test_trade as perform_test_trade_command
from tradeexecutor.cli.testtrade import make_test_trade
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeStatus

JSON_RPC_ETHEREUM = os.environ.get("JSON_RPC_ETHEREUM")
DATA_KEYS = {name: os.environ.get(name) for name in ("TRADING_STRATEGY_API_KEY", "VAULT_PRO_API_KEY")}
STRATEGY_FILE = Path(__file__).resolve().parents[2] / "strategies/test_only/upshift.py"
SENTORA = "0x74aD2F789Ed583DBd141bbdafC673fE1F033718b"
NEMO = "0x955256B31097dDf47a9E47A95aDfDFB4460D8522"
CORE = "0xE9B725010A9E419412ed67d0fA5f3A5f40159D32"
pytestmark = [pytest.mark.skipif(not JSON_RPC_ETHEREUM, reason="JSON_RPC_ETHEREUM required"), pytest.mark.xdist_group("fork:ethereum:midnight")]


@pytest.fixture(scope="module")
def runtime(anvil_fork_pool: AnvilForkPool, tmp_path_factory: TempPathFactory) -> tuple[dict[str, str], Web3]:
    """Deploy and subscribe to Lagoon once using the normal public commands."""
    launch = anvil_fork_pool.get_launch(JSON_RPC_ETHEREUM, ETHEREUM_MIDNIGHT_BLOCK)
    web3 = anvil_fork_pool.get_web3(JSON_RPC_ETHEREUM, ETHEREUM_MIDNIGHT_BLOCK)
    root = tmp_path_factory.mktemp("upshift-cli")
    wallet = HotWallet.create_for_testing(web3, test_account_n=8, eth_amount=10)
    governance = HotWallet.create_for_testing(web3, test_account_n=7, eth_amount=1)
    token = fetch_erc20_details(web3, USDC_NATIVE_TOKEN[1])
    fund_erc20_on_anvil(web3, token.address, wallet.address, token.convert_to_raw(Decimal(500)))
    environment = {
        "PATH": os.environ["PATH"], "EXECUTOR_ID": "upshift-cli", "NAME": "Upshift CLI",
        "STRATEGY_FILE": str(STRATEGY_FILE), "STATE_FILE": str(root / "state.json"),
        "VAULT_RECORD_FILE": str(root / "record.txt"), "PRIVATE_KEY": Web3.to_hex(wallet.private_key),
        "JSON_RPC_ETHEREUM": launch.json_rpc_url, "ASSET_MANAGEMENT_MODE": "lagoon",
        "UNIT_TESTING": "true", "SIMULATE": "false", "LOG_LEVEL": "disabled",
        "CONFIRMATION_BLOCK_COUNT": "1", "MIN_GAS_BALANCE": "0.001", "GAS_BALANCE_WARNING_LEVEL": "0",
        "TRADING_STRATEGY_API_KEY": "secret-token:unused-static-universe", "CACHE_PATH": str(root / "cache"),
        "MAX_CYCLES": "1", "TRADE_IMMEDIATELY": "true", "POSITION_TRIGGER_CHECK_MINUTES": "0",
        "STATS_REFRESH_MINUTES": "0", "VISUALISATION": "false", "CHECK_ACCOUNTS": "true",
        "LAGOON_MAX_SETTLEMENT_AMOUNT": "5000",
    }
    cli = get_command(app)
    with MonkeyPatch.context() as patch:
        patch.setattr(os, "environ", environment.copy())
        cli.main(args=["lagoon-deploy-vault", "--fund-name", "Upshift CLI", "--fund-symbol", "UCLI", "--multisig-owners", governance.address, "--performance-fee", "0", "--management-fee", "0"], standalone_mode=False)
        deployment = json.loads((root / "upshift-cli.deployment.json").read_text())["deployments"]["ethereum"]
        environment.update(VAULT_ADDRESS=deployment["vault_address"], VAULT_ADAPTER_ADDRESS=deployment["module_address"])
        patch.setattr(os, "environ", environment.copy())
        cli.main(args=["init"], standalone_mode=False)
        cli.main(args=["lagoon-first-deposit", "--deposit-amount", "400"], standalone_mode=False)
        cli.main(args=["lagoon-settle"], standalone_mode=False)
    return environment, web3


@pytest.fixture()
def environment(runtime: tuple[dict[str, str], Web3], anvil_fork_pool: AnvilForkPool, monkeypatch: MonkeyPatch) -> Iterator[dict[str, str]]:
    """Restore the fork and the authoritative executor state after each case."""
    env, web3 = runtime
    state_path = Path(env["STATE_FILE"])
    saved = state_path.read_bytes()
    monkeypatch.setattr(os, "environ", env.copy())
    with_snapshot = evm_snapshot_revert(anvil_fork_pool.get_launch(JSON_RPC_ETHEREUM, ETHEREUM_MIDNIGHT_BLOCK))
    next(with_snapshot)
    try:
        yield env
    finally:
        state_path.write_bytes(saved)
        next(with_snapshot, None)


@pytest.mark.parametrize("address", [SENTORA, CORE])
def test_upshift_perform_test_trade(environment: dict[str, str], runtime: tuple[dict[str, str], Web3], address: str) -> None:
    """Exercise both contract families through the guarded public command.

    1. Invoke perform-test-trade against a real deployed Lagoon Safe.
    2. Reload state and verify exact request/payout accounting and no escrow.
    """
    # 1. Invoke perform-test-trade against a real deployed Lagoon Safe.
    cli = get_command(app)
    cli.main(args=["perform-test-trade", "--pair", f"(ethereum, {address})", "--amount", "10", "--no-test-short"], standalone_mode=False)
    # 2. Reload state and verify exact request/payout accounting and no escrow.
    state = State.read_json_file(Path(environment["STATE_FILE"]))
    assert not state.portfolio.open_positions and not state.portfolio.frozen_positions
    trades = list(state.portfolio.get_all_trades())
    assert len(trades) == 2 and all(trade.is_success() for trade in trades)
    assert state.portfolio.get_cash() == pytest.approx(399.999999, rel=0, abs=0.0000001)
    web3 = runtime[1]
    external = create_vault_instance_autodetect(web3, address)
    lagoon = LagoonVault(web3, VaultSpec(1, environment["VAULT_ADDRESS"]))
    assert external.share_token.fetch_raw_balance_of(lagoon.safe_address) == 0
    assert external.denomination_token.fetch_balance_of(lagoon.safe_address) == Decimal("399.999999")


def test_upshift_pending_resume(environment: dict[str, str], runtime: tuple[dict[str, str], Web3], monkeypatch: MonkeyPatch) -> None:
    """Resume one saved queue request without submitting a duplicate withdrawal.

    1. Deposit and request through the public CLI with time advancement disabled.
    2. Repeat close-only and verify the same request remains pending.
    3. Advance real fork time and resume twice, applying the payout once.
    """
    # 1. Deposit and request through the public CLI with time advancement disabled.
    cli = get_command(app)
    args = ["perform-test-trade", "--pair", f"(ethereum, {SENTORA})", "--no-test-short"]
    cli.main(args=args + ["--amount", "10", "--buy-only"], standalone_mode=False)
    monkeypatch.setattr(perform_test_trade_command, "make_test_trade", partial(make_test_trade, force_async_settlement_on_anvil=False))
    cli.main(args=args + ["--close-only"], standalone_mode=False)
    path = Path(environment["STATE_FILE"])
    pending = State.read_json_file(path)
    sell = next(t for t in pending.portfolio.get_all_trades() if t.is_sell())
    assert sell.get_status() is TradeStatus.vault_settlement_pending
    request = sell.other_data["vault_request_tx_hash"]
    # 2. Repeat close-only and verify the same request remains pending.
    cli.main(args=args + ["--close-only"], standalone_mode=False)
    assert len(list(State.read_json_file(path).portfolio.get_all_trades())) == 2
    # 3. Advance real fork time and resume twice, applying the payout once.
    manager = create_vault_instance_autodetect(runtime[1], SENTORA).get_deposit_manager()
    ticket = manager.reconstruct_redemption_ticket(sell.other_data)
    manager.force_settle(ticket)
    batch = manager.vault.upshift_contract.functions.processAllClaimsByDate(ticket.year, ticket.month, ticket.day, 1).transact({"from": runtime[1].eth.accounts[0]})
    assert wait_for_transaction_receipt_robust(runtime[1], batch)["status"] == 1
    # Remote Hypersync cannot index new fork transactions. Substitute only its
    # lookup result; real batch execution and native receipt validation remain.
    monkeypatch.setattr(type(manager), "fetch_completed_redemption_tx_hash", lambda self, queued: batch)
    cli.main(args=args + ["--close-only"], standalone_mode=False)
    cli.main(args=args + ["--close-only"], standalone_mode=False)
    state = State.read_json_file(path)
    trades = list(state.portfolio.get_all_trades())
    assert len(trades) == 2 and all(t.is_success() for t in trades)
    assert trades[-1].other_data["vault_request_tx_hash"] == request
    assert state.portfolio.get_cash() == pytest.approx(399.999999, rel=0, abs=0.0000001)


@pytest.mark.skipif(not all(DATA_KEYS.values()), reason="Vault metadata API credentials required")
@pytest.mark.parametrize("vault_address", [SENTORA, NEMO])
def test_upshift_vault_test_trade(environment: dict[str, str], tmp_path: Path, monkeypatch: MonkeyPatch, vault_address: str) -> None:
    """Run the standalone command with real metadata and guarded fork execution.

    1. Load the actual licensed vault catalogue and select Sentora.
    2. Execute the public simulation command with real contract settlement.
    3. Check the machine-readable report proves a completed round trip.
    """
    # 1. Load the actual licensed vault catalogue and select Sentora.
    for name, value in DATA_KEYS.items():
        monkeypatch.setenv(name, value)
    report = tmp_path / "report.json"
    # 2. Execute the public simulation command with real contract settlement.
    get_command(app).main(args=["vault-test-trade", "--state-file", str(tmp_path / "standalone-state.json"), "--auto-simulated", "--settle-async-on-anvil", "--vault-id", f"1-{vault_address}", "--amount", "10", "--simulation-fork-block", f"1:{ETHEREUM_MIDNIGHT_BLOCK}", "--simulation-fork-block-reference", "docs/superpowers/plans/2026-10-30-upshift-trade-executor-support.md", "--report-json", str(report)], standalone_mode=False)
    # 3. Check the machine-readable report proves a completed round trip.
    result = json.loads(report.read_text())
    rows = result["results"] if isinstance(result, dict) else result
    assert len(rows) == 1
    assert rows[0]["result"] == "success_simulated"
