"""Reproduce Hyper-AI's interrupted 28 September deposit through the CLI.

The committed state is a mechanically reduced copy of the actual pre-repair
state. Set ``HYPER_AI_INCIDENT_STATE`` to the untouched 44 MB snapshot for the
required full-state gate before any production repair. The HyperEVM fork is
pinned to the failure block; only HyperCore Info/CoreWriter effects are mocked
because Anvil cannot replay those cross-domain effects.
"""

import os
import shutil
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from eth_defi.provider.anvil import AnvilLaunch, fund_erc20_on_anvil, launch_anvil
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.token import fetch_erc20_details

from tradeexecutor.cli.main import app
from tradeexecutor.ethereum import multichain_balance
from tradeexecutor.ethereum.vault import hypercore_transit_recovery, hypercore_vault
from tradeexecutor.ethereum.vault.hypercore_transit_recovery import HypercoreTransitBalanceSnapshot
from tradeexecutor.state.state import State
from tradeexecutor.state.store import JSONFileStore
from tradeexecutor.state.trade import TradeStatus, has_unresolved_hypercore_accounting
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.hypercore_data_availability import calculate_hypercore_slot_schedule


BLOCK = 47_080_105
PUBLIC_HYPEREVM_RPC = "https://rpc.hyperliquid.xyz/evm"
SAFE = "0xa8F8DEbb722c6174B814b432169BF569603F673F"
USDC = "0xb88339cb7199b77e23db6e890353e22632ba630f"
VAULT = "0xC723aDd84EE4646044ff28e552808E0a3ac48b54"
ADAPTER = "0xf79d5540fA3a6ea738Aa21A562c5AD7224406F84"
PHASE_2 = "0xc2d0aeb0cba1b7bfd6535097b15205acf232fc47570f444510bc806881905e53"
PHASE_3_FAILED = "0x0d5a389f0dd981c38c4e848e808479ce7ce2e1953246e6bd320c636f2c32fb2d"
INITIAL_SAFE = Decimal("10775.588216")
INITIAL_PERP = Decimal("3024.126339")
INITIAL_SPOT = Decimal("0.008197")

pytestmark = [
    pytest.mark.warm_rpc_test_group,
    pytest.mark.xdist_group("fork:hyperliquid:47080105:isolated"),
]


@pytest.fixture()
def anvil() -> Iterator[AnvilLaunch]:
    """Fork the incident block, letting the session fixture install its stored RPC seed."""
    fork = launch_anvil(
        os.environ.get("JSON_RPC_HYPERLIQUID") or PUBLIC_HYPEREVM_RPC,
        fork_block_number=BLOCK,
    )
    try:
        yield fork
    finally:
        fork.close()


@pytest.fixture()
def incident_state_file(tmp_path: Path) -> Path:
    """Use an isolated state copy so neither fixture nor original is mutated."""
    fixture = Path(__file__).resolve().parent.parent / "hyperliquid" / "state" / "hyper-ai-sept28-incident.json"
    source = Path(os.environ.get("HYPER_AI_INCIDENT_STATE", fixture))
    assert source.is_file(), f"Missing incident state: {source}"
    result = tmp_path / "hyper-ai.json"
    shutil.copy2(source, result)
    return result


def test_interrupted_hypercore_deposit_cli(
    anvil: AnvilLaunch,
    incident_state_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repair never-broadcast buys, preview custody, then reconcile the partial slot.

    1. Verify the pinned fork and actual incident receipts and Safe balance.
    2. Enter ``repair`` and ``correct-accounts --dry-run`` through Typer.
    3. Simulate only the HyperCore cross-domain transfer, then enter plain
       ``correct-accounts`` through Typer.
    4. Check cash, trade, position, next-slot state and the independent
       ``check-accounts`` CLI; rerun without double credit.
    5. Inject a crash after external recovery but before the state-file save,
       then verify that a rerun does not transfer or credit the cash twice.
    6. Reject non-zero target vault equity before any transfer or state repair.

    Repeat with ``HYPER_AI_INCIDENT_STATE`` pointing to the original 44 MB
    snapshot before using these commands against production.
    """
    # 1. Confirm the local fork reproduces the incident's on-chain evidence.
    web3 = create_multi_provider_web3(anvil.json_rpc_url)
    token = fetch_erc20_details(web3, USDC)
    assert web3.eth.chain_id == 999
    assert web3.eth.block_number == BLOCK
    state = State.read_json_file(incident_state_file)
    trade = state.portfolio.get_trade_by_id(1728)
    assert state.cycle == 123
    assert state.pending_data_availability_slot.isoformat() == "2026-09-28T00:00:00"
    assert trade.get_status() == TradeStatus.started
    assert trade.other_data["hypercore_deposit_capital_at_risk"]["amount_human"] == "3023.713174"
    assert all(state.portfolio.get_trade_by_id(trade_id).get_status() == TradeStatus.success for trade_id in (1726, 1727, 1730))
    assert all(state.portfolio.get_trade_by_id(trade_id).get_status() == TradeStatus.planned for trade_id in (1729, 1731, 1732, 1733))
    hashes = [tx.tx_hash for tx in trade.blockchain_transactions] + [PHASE_2, PHASE_3_FAILED]
    assert [web3.eth.get_transaction_receipt(tx_hash)["status"] for tx_hash in hashes] == [1, 1, 1, 0]
    assert token.fetch_balance_of(SAFE) == INITIAL_SAFE

    # HyperCore's Info API is not historical on an Anvil fork. Use the saved
    # position values, with Fadorador at zero, for its read-only responses.
    expected_equity = {
        position.pair.pool_address.lower(): Decimal(str(position.calculate_quantity_usd_value(position.get_quantity())))
        for position in state.portfolio.get_open_positions()
        if position.pair.is_hyperliquid_vault()
    }
    expected_equity[trade.pair.pool_address.lower()] = Decimal(0)

    def vault_equity(_session, *, user: str, vault_address: str, **_kwargs):
        """Return incident-time equity at the mocked HyperCore API boundary."""
        assert user.lower() == SAFE.lower()
        amount = expected_equity.get(vault_address.lower(), Decimal(0))
        return SimpleNamespace(equity=amount) if amount > 0 else None

    monkeypatch.setattr(hypercore_vault, "fetch_user_vault_equity", vault_equity)
    monkeypatch.setattr(multichain_balance, "fetch_user_vault_equity", vault_equity)
    hypercore = {"perp": INITIAL_PERP, "spot": INITIAL_SPOT, "transfers": 0}

    def transit_snapshot(*, session, safe_address: str, reserve_token):
        """Read real Anvil Safe USDC and incident-time HyperCore cash classes."""
        assert safe_address.lower() == SAFE.lower()
        return HypercoreTransitBalanceSnapshot(
            safe_address=safe_address,
            evm_usdc_balance=reserve_token.fetch_balance_of(safe_address),
            spot_total_usdc=hypercore["spot"],
            spot_free_usdc=hypercore["spot"],
            perp_withdrawable=hypercore["perp"],
            perp_account_value=hypercore["perp"],
            perp_position_count=0,
        )

    monkeypatch.setattr(hypercore_transit_recovery, "fetch_hypercore_transit_balances", transit_snapshot)

    def settle_hypercore(*, actions, **_kwargs) -> list[str]:
        """Model the unavailable cross-domain transfer; keep real ERC-20 accounting."""
        assert [action.action_kind for action in actions] == ["perp_to_spot", "spot_to_evm"]
        assert actions[0].amount == Decimal("3023.626339")
        assert actions[1].amount == Decimal("3023.624536")
        hypercore["perp"] = Decimal("0.50")
        hypercore["spot"] = Decimal("0.01")
        hypercore["transfers"] += 1
        # fund_erc20_on_anvil overwrites the absolute balance, not adds to it.
        fund_erc20_on_anvil(web3, USDC, SAFE, token.convert_to_raw(INITIAL_SAFE + actions[1].amount))
        return [action.action_kind for action in actions]

    monkeypatch.setattr(hypercore_transit_recovery, "execute_hypercore_transit_recovery_actions", settle_hypercore)
    for name in tuple(os.environ):
        if name.startswith("JSON_RPC_") and name != "JSON_RPC_HYPERLIQUID":
            monkeypatch.delenv(name)
    strategy = Path(__file__).resolve().parents[2] / "strategies" / "test_only" / "minimal_hyperliquid_strategy.py"
    env = {
        "EXECUTOR_ID": "hyper-ai",
        "STRATEGY_FILE": str(strategy),
        "STATE_FILE": str(incident_state_file),
        "CACHE_PATH": str(tmp_path / "cache"),
        "JSON_RPC_HYPERLIQUID": anvil.json_rpc_url,
        "PRIVATE_KEY": "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
        "ASSET_MANAGEMENT_MODE": "lagoon",
        "VAULT_ADDRESS": VAULT,
        "VAULT_ADAPTER_ADDRESS": ADAPTER,
        "UNIT_TESTING": "true",
        "AUTO_APPROVE": "true",
        "LOG_LEVEL": "info",
    }
    runner = CliRunner()

    # 2. Repair records safe partial progress, but leaves deposit #1728 at risk.
    repaired = runner.invoke(app, ["repair"], env=env)
    assert repaired.exit_code == 0, repr(repaired.exception)
    assert "deferred hypercore trade(s) [1728]" in repaired.output.lower()
    assert incident_state_file.with_suffix(".backup-1.json").is_file(), repr(repaired.exception)
    state = State.read_json_file(incident_state_file)
    assert state.portfolio.get_trade_by_id(1728).get_status() == TradeStatus.started
    assert all(state.portfolio.get_trade_by_id(trade_id).get_status() == TradeStatus.repaired for trade_id in (1729, 1731, 1732, 1733))
    assert all(state.portfolio.get_position_by_id(position_id).is_closed() for position_id in (533, 534, 535))
    assert token.fetch_balance_of(SAFE) == INITIAL_SAFE
    before_dry_run = incident_state_file.read_bytes()
    preview = runner.invoke(app, ["correct-accounts", "--dry-run"], env=env)
    assert preview.exit_code == 0, repr(preview.exception)
    assert "not safe to restart" in preview.output.lower()
    assert "3023.626339" in preview.output
    assert "3023.624536" in preview.output
    assert incident_state_file.read_bytes() == before_dry_run
    assert hypercore["transfers"] == 0

    # 3. The real CLI may change state only after the mocked transfer has
    # changed HyperCore custody and the fork's real ERC-20 Safe balance.
    corrected = runner.invoke(app, ["correct-accounts"], env=env)
    assert corrected.exit_code == 0, repr(corrected.exception)
    assert hypercore["transfers"] == 1
    assert token.fetch_balance_of(SAFE) == INITIAL_SAFE + Decimal("3023.624536")

    # 4. Restart scheduling must advance to 30 September without replaying
    # the three successful sibling trades from the 28 September slot.
    state = State.read_json_file(incident_state_file)
    assert state.portfolio.get_trade_by_id(1728).get_status() == TradeStatus.failed
    assert state.portfolio.get_position_by_id(532).is_closed()
    assert not any(has_unresolved_hypercore_accounting(t) for t in state.portfolio.get_all_trades())
    assert state.pending_data_availability_slot is None
    assert state.cycle == 124
    assert state.last_cycle_at.isoformat() == "2026-09-28T00:00:00"
    saved_reserve = state.portfolio.get_default_reserve_position().quantity
    assert saved_reserve == INITIAL_SAFE + Decimal("3023.624536")
    state.check_if_clean()
    assert calculate_hypercore_slot_schedule(state.last_cycle_at, CycleDuration.cycle_2d, state).isoformat() == "2026-09-30T00:00:00"
    checked = runner.invoke(app, ["check-accounts"], env=env)
    assert checked.exit_code == 0, repr(checked.exception)
    assert "all accounts match" in checked.output.lower()
    again = runner.invoke(app, ["correct-accounts"], env=env)
    assert again.exit_code == 0, again.output
    assert hypercore["transfers"] == 1
    after_rerun = State.read_json_file(incident_state_file)
    assert after_rerun.cycle == 124
    assert after_rerun.portfolio.get_default_reserve_position().quantity == saved_reserve

    # 5. A crash after the transfer leaves the old state on disk while cash
    # is already back in the Safe. The rerun must detect that custody pattern.
    source = Path(os.environ.get(
        "HYPER_AI_INCIDENT_STATE",
        Path(__file__).resolve().parent.parent / "hyperliquid" / "state" / "hyper-ai-sept28-incident.json",
    ))
    crash_state_file = tmp_path / "hyper-ai-crash.json"
    shutil.copy2(source, crash_state_file)
    crash_env = {**env, "STATE_FILE": str(crash_state_file)}
    hypercore.update(perp=INITIAL_PERP, spot=INITIAL_SPOT)
    fund_erc20_on_anvil(web3, USDC, SAFE, token.convert_to_raw(INITIAL_SAFE))
    assert runner.invoke(app, ["repair"], env=crash_env).exit_code == 0
    before_crash = crash_state_file.read_bytes()
    original_sync = JSONFileStore.sync
    crashed = False

    def fail_after_recovery(store, state, *args, **kwargs):
        """Fail exactly at the first persisted-state write after external recovery."""
        nonlocal crashed
        if Path(store.path) == crash_state_file and hypercore["transfers"] == 2 and not crashed:
            crashed = True
            raise RuntimeError("Injected state save failure after HyperCore recovery")
        return original_sync(store, state, *args, **kwargs)

    monkeypatch.setattr(JSONFileStore, "sync", fail_after_recovery)
    interrupted = runner.invoke(app, ["correct-accounts"], env=crash_env)
    assert interrupted.exit_code == 1
    assert crashed
    assert crash_state_file.read_bytes() == before_crash
    assert token.fetch_balance_of(SAFE) == INITIAL_SAFE + Decimal("3023.624536")
    monkeypatch.setattr(JSONFileStore, "sync", original_sync)
    resumed = runner.invoke(app, ["correct-accounts"], env=crash_env)
    assert resumed.exit_code == 0, repr(resumed.exception)
    assert hypercore["transfers"] == 2
    resumed_state = State.read_json_file(crash_state_file)
    assert resumed_state.cycle == 124
    assert resumed_state.portfolio.get_default_reserve_position().quantity == state.portfolio.get_default_reserve_position().quantity
    assert runner.invoke(app, ["correct-accounts"], env=crash_env).exit_code == 0
    assert hypercore["transfers"] == 2

    # 6. Non-zero vault equity makes the location of the attempted deposit
    # ambiguous. The CLI must refuse before any HyperCore transfer or state fix.
    negative_state_file = tmp_path / "hyper-ai-negative.json"
    shutil.copy2(source, negative_state_file)
    negative_env = {**env, "STATE_FILE": str(negative_state_file)}
    hypercore.update(perp=INITIAL_PERP, spot=INITIAL_SPOT)
    fund_erc20_on_anvil(web3, USDC, SAFE, token.convert_to_raw(INITIAL_SAFE))
    assert runner.invoke(app, ["repair"], env=negative_env).exit_code == 0
    before_negative = negative_state_file.read_bytes()
    expected_equity[trade.pair.pool_address.lower()] = Decimal(1)
    rejected = runner.invoke(app, ["correct-accounts"], env=negative_env)
    assert rejected.exit_code == 1
    assert "vault equity" in str(rejected.exception)
    assert negative_state_file.read_bytes() == before_negative
    assert hypercore["transfers"] == 2
    assert token.fetch_balance_of(SAFE) == INITIAL_SAFE
