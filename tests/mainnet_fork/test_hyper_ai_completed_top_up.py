"""Exercise the October Hyper-AI top-up recovery through the real Typer CLI.

The fixture retains the six held positions from the 8 October production state,
including their trades and the pending slot. Closed history, charts and statistics
are removed mechanically. Set ``HYPER_AI_TOP_UP_STATE`` to the untouched downloaded
snapshot to repeat the same CLI test against the full production ledger.
"""

import datetime
import os
import shutil
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from eth_defi.provider.anvil import AnvilLaunch, fund_erc20_on_anvil, launch_anvil
from eth_defi.hyperliquid.session import HyperliquidSession
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.token import fetch_erc20_details

from tradeexecutor.cli.main import app
from tradeexecutor.ethereum import multichain_balance
from tradeexecutor.ethereum.vault import hypercore_transit_recovery, hypercore_vault
from tradeexecutor.state.state import State
from tradeexecutor.state.store import JSONFileStore
from tradeexecutor.state.trade import TradeStatus, has_unresolved_hypercore_accounting


BLOCK = 47_989_499
SAFE = "0xa8F8DEbb722c6174B814b432169BF569603F673F"
VAULT = "0xC723aDd84EE4646044ff28e552808E0a3ac48b54"
MODULE = "0xD58F0D72024ABB64568dC6d653a3172faa8B49FB"
DEPOSIT = Decimal("28.175784")
LEDGER_EVENT = {
    "time": 1791427830047,
    "hash": "0x31706300b7f4fe2732ea044614c9e30000ab7ae652f81cf9d5390e5376f8d811",
    "delta": {"type": "vaultDeposit", "vault": "0xce71894b42f5c31a6ae6ca4fb77441c68ab15086", "usdc": "28.175784"},
}

pytestmark = [pytest.mark.warm_rpc_test_group, pytest.mark.xdist_group("fork:hyperliquid:47989499:top-up")]


@pytest.fixture()
def anvil() -> Iterator[AnvilLaunch]:
    """Fork the production maintenance block; no test can transact on mainnet."""
    fork = launch_anvil(os.environ.get("JSON_RPC_HYPERLIQUID") or "https://rpc.hyperliquid.xyz/evm", fork_block_number=BLOCK)
    try:
        yield fork
    finally:
        fork.close()


@pytest.fixture()
def top_up_cli(anvil: AnvilLaunch, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict, list[dict]]:
    """Copy production accounting and mock only HyperCore's unavailable fork state."""
    source = Path(os.environ.get("HYPER_AI_TOP_UP_STATE", Path(__file__).parents[1] / "hyperliquid/state/hyper-ai-oct8-top-up.json"))
    state_file = tmp_path / "hyper-ai.json"
    shutil.copy2(source, state_file)
    state = State.read_json_file(state_file)
    web3 = create_multi_provider_web3(anvil.json_rpc_url)
    reserve = state.portfolio.get_default_reserve_position()
    token = fetch_erc20_details(web3, reserve.asset.address)
    # Isolate this deposit test from unrelated reserve movements on the fork.
    fund_erc20_on_anvil(web3, token.address, SAFE, token.convert_to_raw(reserve.quantity))
    equity = {p.pair.pool_address.lower(): Decimal(str(p.get_value())) for p in state.portfolio.get_open_positions()}
    # The vault can lose more than the top-up: total-equity growth is not proof.
    equity[LEDGER_EVENT["delta"]["vault"]] -= Decimal("100")
    events = [LEDGER_EVENT]

    def post_info(session: HyperliquidSession, payload: dict, **kwargs) -> SimpleNamespace:
        """Only the ledger is mocked here; EVM receipts still come from Anvil."""
        assert payload["type"] == "userNonFundingLedgerUpdates"
        assert payload["user"].lower() == SAFE.lower()
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: events)

    def vault_equity(session, *, user: str, vault_address: str, **kwargs):
        """Anvil cannot replay HyperCore equity; expose falling incident equity."""
        return SimpleNamespace(equity=equity.get(vault_address.lower(), Decimal(0)))

    def no_transfer(**kwargs):
        """A completed deposit must never initiate a second external transfer."""
        pytest.fail("Completed top-up attempted a custody transfer")

    monkeypatch.setattr(HyperliquidSession, "post_info", post_info)
    monkeypatch.setattr(hypercore_vault, "fetch_user_vault_equity", vault_equity)
    monkeypatch.setattr(multichain_balance, "fetch_user_vault_equity", vault_equity)
    monkeypatch.setattr(hypercore_transit_recovery, "execute_hypercore_transit_recovery_actions", no_transfer)
    for name in tuple(os.environ):
        if name.startswith("JSON_RPC_"):
            monkeypatch.delenv(name)
    environment = {
        "EXECUTOR_ID": "hyper-ai", "STATE_FILE": str(state_file),
        "STRATEGY_FILE": str(Path(__file__).parents[2] / "strategies/test_only/minimal_hyperliquid_strategy.py"),
        "CACHE_PATH": str(tmp_path / "cache"), "JSON_RPC_HYPERLIQUID": anvil.json_rpc_url,
        "PRIVATE_KEY": "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
        "ASSET_MANAGEMENT_MODE": "lagoon", "VAULT_ADDRESS": VAULT, "VAULT_ADAPTER_ADDRESS": MODULE,
        "UNIT_TESTING": "true", "LOG_LEVEL": "info", "SKIP_HYPERCORE_TRANSIT_RECOVERY": "true",
        "CLEANUP_HYPERCORE_SMALL_POSITIONS": "false",
    }
    return state_file, environment, events


def test_correct_accounts_completed_top_up(top_up_cli: tuple[Path, dict, list[dict]]) -> None:
    """Recover the actual top-up without resending capital or replaying its slot.

    1. Preview against copied production accounting without changing its bytes.
    2. Execute correct-accounts and check principal, reserves and pending slot.
    3. Repeat the command and independent account check without a second fill.
    """
    state_file, env, _events = top_up_cli
    runner = CliRunner()
    before = state_file.read_bytes()
    original = State.read_json_file(state_file)
    quantity = original.portfolio.get_position_by_id(531).get_quantity()
    reserve = original.portfolio.get_default_reserve_position().quantity

    # 1. Dry-run traverses the same evidence path but cannot save or back up.
    preview = runner.invoke(app, ["correct-accounts", "--dry-run"], env=env)
    assert preview.exit_code == 0, repr(preview.exception)
    assert "1765" in preview.output and "28.175784" in preview.output
    assert state_file.read_bytes() == before
    assert not state_file.with_suffix(".backup-1.json").exists()

    # 2. The original trade acquires principal; reserves were allocated already.
    corrected = runner.invoke(app, ["correct-accounts"], env=env)
    assert corrected.exit_code == 0, repr(corrected.exception)
    state = State.read_json_file(state_file)
    trade = state.portfolio.get_trade_by_id(1765)
    assert trade.is_success()
    assert trade.executed_quantity == DEPOSIT
    assert trade.reserve_currency_allocated == 0
    assert state.portfolio.get_position_by_id(531).get_quantity() == quantity + DEPOSIT
    assert float(state.portfolio.get_default_reserve_position().quantity) == pytest.approx(float(reserve), abs=1e-6)
    assert not any(has_unresolved_hypercore_accounting(t) for t in state.portfolio.get_all_trades())
    assert state.pending_data_availability_slot is None
    assert state.last_cycle_at == datetime.datetime(2026, 10, 8)
    assert state.cycle == original.cycle + 1
    assert state_file.with_suffix(".backup-1.json").is_file()
    state.check_if_clean()

    # 3. Both repeat correction and independent checks see settled accounting.
    assert runner.invoke(app, ["check-accounts"], env=env).exit_code == 0
    again = runner.invoke(app, ["correct-accounts"], env=env)
    assert again.exit_code == 0, repr(again.exception)
    repeated = State.read_json_file(state_file)
    assert repeated.cycle == state.cycle
    assert repeated.portfolio.get_position_by_id(531).get_quantity() == quantity + DEPOSIT


def test_correct_accounts_rejects_ambiguous_top_up(
    top_up_cli: tuple[Path, dict, list[dict]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject unmatched or duplicate evidence before changing local accounting.

    1. Supply the wrong amount, the wrong vault, or duplicate deposit evidence.
    2. Invoke the real CLI and verify each refusal preserves the original bytes.
    3. Fail the final account check after a valid match; no settlement is saved.
    4. Refuse a started sibling before entering any generic correction work.
    """
    state_file, env, events = top_up_cli
    before = state_file.read_bytes()
    # 1. A phase-1 receipt or existing positive equity cannot prove this fill.
    cases = [
        [],
        [{**LEDGER_EVENT, "delta": {**LEDGER_EVENT["delta"], "usdc": "29"}}],
        [{**LEDGER_EVENT, "delta": {**LEDGER_EVENT["delta"], "vault": SAFE}}],
        [LEDGER_EVENT, LEDGER_EVENT],
    ]
    # 2. None of these cases may produce a reserve refund or settlement write.
    for case in cases:
        events[:] = case
        result = CliRunner().invoke(app, ["correct-accounts"], env=env)
        assert result.exit_code == 1
        assert "matching HyperCore deposit" in str(result.exception)
        assert state_file.read_bytes() == before

    # 3. Proof of the fill must not bypass the command's final accounting gate.
    events[:] = [LEDGER_EVENT]
    monkeypatch.setattr("tradeexecutor.cli.commands.correct_accounts.check_accounts", lambda *args, **kwargs: (False, None))
    failed_check = CliRunner().invoke(app, ["correct-accounts"], env=env)
    assert failed_check.exit_code == 1
    assert "original state was not saved" in str(failed_check.exception)
    assert state_file.read_bytes() == before

    # 4. A started sibling with saved transactions passes the legacy preflight,
    # but must never allow this partially executed decision to be consumed.
    state = State.read_json_file(state_file)
    top_up = state.portfolio.get_trade_by_id(1765)
    sibling = next(
        t for t in state.portfolio.get_all_trades()
        if t.opened_at == top_up.opened_at and t.get_status() == TradeStatus.success and t.repaired_trade_id is None
    )
    sibling.executed_at = sibling.broadcasted_at = None
    sibling.started_at = top_up.started_at
    sibling.blockchain_transactions = list(top_up.blockchain_transactions)
    JSONFileStore(state_file).sync(state)
    before = state_file.read_bytes()

    def no_generic_correction(**kwargs):
        """A non-terminal slot must fail before any correction side effects."""
        pytest.fail("Entered generic correction with an unfinished sibling")

    monkeypatch.setattr("tradeexecutor.cli.commands.correct_accounts._sync_hypercore_vault_positions", no_generic_correction)
    sibling_failure = CliRunner().invoke(app, ["correct-accounts"], env=env)
    assert sibling_failure.exit_code == 1
    assert "non-terminal sibling" in str(sibling_failure.exception)
    assert state_file.read_bytes() == before
