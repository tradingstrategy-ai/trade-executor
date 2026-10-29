"""Guarded public ERC-4626 test trades on a pinned Arbitrum fork.

Only investor gas/USDC are injected. Protocol vault liquidity and admission are
not changed. The test performs separate CLI opens/closes, as the shell tutorial
will, without depending on downloaded market data.
"""

import json
import os
from collections.abc import Iterator
from decimal import Decimal
from functools import partial
from pathlib import Path
from unittest.mock import patch

import pytest
from eth_defi.erc_4626.vault_protocol.lagoon.vault import LagoonVault
from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import fund_erc20_on_anvil
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.testing.anvil_fork_pool import AnvilForkPool
from eth_defi.testing.fork_blocks import ARBITRUM_MIDNIGHT_BLOCK
from eth_defi.token import USDC_NATIVE_TOKEN, fetch_erc20_details
from eth_defi.vault.base import VaultSpec
from pytest import MonkeyPatch, TempPathFactory
from typer.main import get_command
from web3 import Web3

from tradeexecutor.cli.main import app
from tradeexecutor.cli.commands import perform_test_trade as perform_test_trade_command
from tradeexecutor.cli.testtrade import make_test_trade
from tradeexecutor.ethereum.vault.vault_routing import VaultRouting
from tradeexecutor.state.state import State
from tradeexecutor.state.store import JSONFileStore
from tradeexecutor.state.trade import TradeFlag
from tradeexecutor.testing.erc4626_manual import STRATEGY_FILE, load_universe, main as manual_main, position_phase

JSON_RPC_ARBITRUM = os.environ.get("JSON_RPC_ARBITRUM")
pytestmark = [pytest.mark.skipif(not JSON_RPC_ARBITRUM, reason="JSON_RPC_ARBITRUM required"), pytest.mark.xdist_group("fork:arbitrum:erc4626-cli")]


@pytest.fixture(scope="module")
def manual_runtime(anvil_fork_pool: AnvilForkPool, tmp_path_factory: TempPathFactory) -> Iterator[tuple[dict, Web3]]:
    """Deploy and subscribe once; individual tests restore this funded snapshot."""
    launch = anvil_fork_pool.get_launch(JSON_RPC_ARBITRUM, ARBITRUM_MIDNIGHT_BLOCK)
    web3 = create_multi_provider_web3(launch.json_rpc_url)
    root = tmp_path_factory.mktemp("erc4626-cli")
    wallet = HotWallet.create_for_testing(web3, test_account_n=8, eth_amount=10)
    token = fetch_erc20_details(web3, USDC_NATIVE_TOKEN[42161])
    fund_erc20_on_anvil(web3, token.address, wallet.address, token.convert_to_raw(Decimal(20)))
    environment = {
        "PATH": os.environ["PATH"],
        "EXECUTOR_ID": "erc4626-cli",
        "NAME": "ERC4626 manual CLI",
        "STRATEGY_FILE": str(STRATEGY_FILE),
        "STATE_FILE": str(root / "erc4626-cli.json"),
        "VAULT_RECORD_FILE": str(root / "vault-record.txt"),
        "PRIVATE_KEY": Web3.to_hex(wallet.private_key),
        "JSON_RPC_ARBITRUM": launch.json_rpc_url,
        "ASSET_MANAGEMENT_MODE": "lagoon",
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",
        "CONFIRMATION_BLOCK_COUNT": "1",
        "MIN_GAS_BALANCE": "0.001",
        # The static universe makes no API requests, but check-wallet creates a client.
        "TRADING_STRATEGY_API_KEY": "secret-token:unused-static-universe",
        "CACHE_PATH": str(root / "cache"),
    }
    cli = get_command(app)
    with patch.dict(os.environ, environment, clear=True):
        cli.main(args=["lagoon-deploy-vault", "--fund-name", "ERC4626 CLI", "--fund-symbol", "ECLI", "--multisig-owners", wallet.address, "--performance-fee", "0", "--management-fee", "0"], standalone_mode=False)
        deployment = json.loads((root / "erc4626-cli.deployment.json").read_text())["deployments"]["arbitrum"]
        environment.update(VAULT_ADDRESS=deployment["vault_address"], VAULT_ADAPTER_ADDRESS=deployment["module_address"])
    with patch.dict(os.environ, environment, clear=True):
        cli.main(args=["init"], standalone_mode=False)
        cli.main(args=["check-wallet"], standalone_mode=False)
        cli.main(args=["lagoon-first-deposit", "--deposit-amount", "10"], standalone_mode=False)
        # Record the command's external settlement before snapshotting. A nested
        # simulation fork cannot fetch historical receipts mined on its parent.
        cli.main(args=["lagoon-settle"], standalone_mode=False)
    yield environment, web3


@pytest.fixture()
def manual_environment(manual_runtime: tuple[dict, Web3], monkeypatch: MonkeyPatch) -> Iterator[dict]:
    """Restore funded custody and a fresh state file around each CLI scenario."""
    environment, web3 = manual_runtime
    # Keep the real CLI, routing and receipts, but use the helper's existing
    # clock argument to test immediate exits without a public test-only option.
    monkeypatch.setattr(perform_test_trade_command, "make_test_trade", partial(make_test_trade, anvil_time_skip_seconds=0))
    state_path = Path(environment["STATE_FILE"])
    saved = state_path.read_bytes()
    snapshot = web3.provider.make_request("evm_snapshot", [])["result"]
    with patch.dict(os.environ, environment, clear=True):
        try:
            yield environment
        finally:
            state_path.write_bytes(saved)
            web3.provider.make_request("evm_revert", [snapshot])


def test_perform_test_trade_erc4626_round_trip(manual_environment: dict, manual_runtime: tuple[dict, Web3]) -> None:
    """Run the manual tutorial matrix and return the investor capital through the public CLI.

    1. Open and separately close each implementation through the public commands.
    2. Check receipt amounts, immediate-exit timing, exhausted shares and clean accounts.
    3. Verify only forced settlement refreshes a below-tolerance NAV change with an empty queue.
    4. Redeem the investor and reconcile actual returned USDC against protocol cash flows.
    """
    # 1. Open and separately close each implementation through the public commands.
    cli = get_command(app)
    web3 = manual_runtime[1]
    state_path = Path(manual_environment["STATE_FILE"])
    for pair in load_universe().iterate_pairs():
        selector = f"(arbitrum, {pair.pool_address})"
        cli.main(args=["perform-test-trade", "--pair", selector, "--amount", "1", "--buy-only", "--no-test-short"], standalone_mode=False)
        opened = State.read_json_file(state_path)
        position = next(p for p in opened.portfolio.open_positions.values() if p.pair.pool_address == pair.pool_address)
        buy = position.get_first_trade()
        assert buy.is_success() and buy.is_test()
        assert buy.executed_reserve == Decimal(1)
        assert buy.executed_quantity > 0
        assert position_phase(opened, load_universe(), pair.pool_address) == "close"
        cli.main(args=["perform-test-trade", "--pair", selector, "--close-only", "--no-test-short"], standalone_mode=False)

        # 2. Check receipt amounts, immediate-exit timing, exhausted shares and clean accounts.
        closed = State.read_json_file(state_path)
        assert not closed.portfolio.open_positions
        assert not closed.portfolio.frozen_positions
        final = closed.portfolio.closed_positions[position.position_id]
        sell = final.get_last_trade()
        buy_timestamp = web3.eth.get_block(buy.blockchain_transactions[-1].block_number)["timestamp"]
        sell_timestamp = web3.eth.get_block(sell.blockchain_transactions[-1].block_number)["timestamp"]
        # Allow ordinary RPC/test latency, but detect the helper's default 24h jump.
        assert 0 <= sell_timestamp - buy_timestamp < 3600
        assert sell.is_success() and sell.is_test()
        assert sell.executed_quantity == -buy.executed_quantity
        assert sell.executed_reserve > 0
        assert final.get_quantity() == 0
        assert position_phase(closed, load_universe(), pair.pool_address) == "done"
        with pytest.raises(SystemExit) as exit_result:
            cli.main(args=["check-accounts"], standalone_mode=False)
        assert exit_result.value.code == 0

    # 3. Verify only forced settlement refreshes a below-tolerance NAV change with an empty queue.
    wallet = HotWallet.from_private_key(manual_environment["PRIVATE_KEY"])
    token = fetch_erc20_details(web3, USDC_NATIVE_TOKEN[42161])
    assert manual_main("audit") == 0
    protocol_cash_flow = sum((t.executed_reserve * (-1 if t.is_buy() else 1) for t in closed.portfolio.get_all_trades()), Decimal(0))
    vault = LagoonVault(web3, VaultSpec(42161, manual_environment["VAULT_ADDRESS"]), trading_strategy_module_address=manual_environment["VAULT_ADAPTER_ADDRESS"])
    cash = token.fetch_balance_of(vault.safe_address)
    nav_before = vault.fetch_total_assets(web3.eth.block_number)
    assert Decimal(0) < nav_before - cash < Decimal("0.05")
    nonce_before = web3.eth.get_transaction_count(wallet.address)
    cli.main(args=["lagoon-settle"], standalone_mode=False)
    assert web3.eth.get_transaction_count(wallet.address) == nonce_before
    assert vault.fetch_total_assets(web3.eth.block_number) == nav_before
    cli.main(args=["lagoon-settle", "--force-valuation"], standalone_mode=False)
    assert web3.eth.get_transaction_count(wallet.address) == nonce_before + 2
    assert vault.fetch_total_assets(web3.eth.block_number) == pytest.approx(cash, abs=Decimal("0.000001"))

    # 4. Redeem the investor and reconcile actual returned USDC against protocol cash flows.
    cli.main(args=["lagoon-redeem"], standalone_mode=False)
    assert vault.share_token.fetch_raw_balance_of(wallet.address) == 0
    assert manual_main("finished") == 0
    assert token.fetch_balance_of(wallet.address) == pytest.approx(Decimal(20) + protocol_cash_flow, abs=Decimal("0.000002"))


def test_erc4626_interrupted_buy_requires_reconciliation(manual_environment: dict, manual_runtime: tuple[dict, Web3], monkeypatch: MonkeyPatch) -> None:
    """Use normal state persistence and refuse to retry an unresolved deposit.

    Receipt analysis is interrupted after real fork execution to model a crash.

    1. Interrupt receipt settlement after depositing all available reserve cash.
    2. Reload the saved transaction and verify real custody balances.
    3. Refuse another execution until the existing trade has been reconciled.
    """
    # 1. Interrupt receipt settlement after depositing all available reserve cash.
    cli = get_command(app)
    pair = next(load_universe().iterate_pairs())
    selector = f"(arbitrum, {pair.pool_address})"
    def interrupt(*args, **kwargs):
        raise RuntimeError("Interrupted after receipt confirmation")
    monkeypatch.setattr(VaultRouting, "settle_trade", interrupt)
    with pytest.raises(RuntimeError, match="Interrupted after receipt"):
        cli.main(args=["perform-test-trade", "--pair", selector, "--amount", "10", "--buy-only", "--no-test-short"], standalone_mode=False)

    # 2. Reload the saved transaction and verify real custody balances.
    state_path = Path(manual_environment["STATE_FILE"])
    pending = State.read_json_file(state_path)
    trade = next(pending.portfolio.get_all_trades())
    hashes = [t.tx_hash for t in trade.blockchain_transactions]
    assert trade.is_pending() and len(hashes) == 2
    web3 = manual_runtime[1]
    token = fetch_erc20_details(web3, USDC_NATIVE_TOKEN[42161])
    vault = LagoonVault(web3, VaultSpec(42161, manual_environment["VAULT_ADDRESS"]), trading_strategy_module_address=manual_environment["VAULT_ADAPTER_ADDRESS"])
    assert token.fetch_raw_balance_of(vault.safe_address) == 0
    assert fetch_erc20_details(web3, pair.pool_address).fetch_raw_balance_of(vault.safe_address) > 0
    wallet_address = Web3.to_checksum_address(trade.blockchain_transactions[-1].from_address)
    nonce = web3.eth.get_transaction_count(wallet_address)
    # 3. Refuse another execution until the existing trade has been reconciled.
    with pytest.raises(RuntimeError, match="unresolved trades"):
        cli.main(args=["perform-test-trade", "--pair", selector, "--close-only", "--no-test-short"], standalone_mode=False)
    assert web3.eth.get_transaction_count(wallet_address) == nonce
    result = State.read_json_file(state_path)
    assert len(list(result.portfolio.get_all_trades())) == 1
    assert [t.tx_hash for t in next(result.portfolio.get_all_trades()).blockchain_transactions] == hashes


def test_erc4626_refuses_ordinary_position_and_simulation_isolated(manual_environment: dict, manual_runtime: tuple[dict, Web3], monkeypatch: MonkeyPatch) -> None:
    """Refuse ordinary holdings and retain the source state after a failed simulation.

    Receipt analysis is interrupted only in simulation to exercise its error
    persistence path; the simulation still executes real fork transactions.

    1. Open a position and remove its test flag to represent an ordinary holding.
    2. Refuse close-only without submitting another transaction.
    3. Restore a pristine state and interrupt a simulated buy; verify file isolation.
    """
    # 1. Open a position and remove its test flag to represent an ordinary holding.
    cli = get_command(app)
    pair = next(load_universe().iterate_pairs())
    selector = f"(arbitrum, {pair.pool_address})"
    state_path = Path(manual_environment["STATE_FILE"])
    pristine = state_path.read_bytes()
    web3 = manual_runtime[1]
    snapshot = web3.provider.make_request("evm_snapshot", [])["result"]
    cli.main(args=["perform-test-trade", "--pair", selector, "--buy-only", "--no-test-short"], standalone_mode=False)
    state = State.read_json_file(state_path)
    next(state.portfolio.get_all_trades()).flags.remove(TradeFlag.test_trade)
    JSONFileStore(state_path).sync(state)

    # 2. Refuse close-only without submitting another transaction.
    web3 = manual_runtime[1]
    wallet = HotWallet.from_private_key(manual_environment["PRIVATE_KEY"])
    nonce = web3.eth.get_transaction_count(wallet.address)
    with pytest.raises(RuntimeError, match="non-test"):
        cli.main(args=["perform-test-trade", "--pair", selector, "--close-only"], standalone_mode=False)
    assert web3.eth.get_transaction_count(wallet.address) == nonce

    # 3. Restore a pristine state and interrupt a simulated buy; verify file isolation.
    state_path.write_bytes(pristine)
    web3.provider.make_request("evm_revert", [snapshot])
    def interrupt(*args, **kwargs):
        raise RuntimeError("Interrupted simulated receipt analysis")
    monkeypatch.setattr(VaultRouting, "settle_trade", interrupt)
    with pytest.raises(RuntimeError, match="Interrupted simulated"):
        cli.main(args=["perform-test-trade", "--pair", selector, "--buy-only", "--no-test-short", "--simulate"], standalone_mode=False)
    assert state_path.read_bytes() == pristine
