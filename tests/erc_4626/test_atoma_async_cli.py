"""Atoma through the public Typer commands and a real Lagoon Safe on Anvil.

Spoofs are limited to gas funding, a real USDC-holder transfer, advancing time
and impersonating the deployed settlement operator. No executor, receipt or
persistence path is mocked. The dummy strategy never acts as the operator.
See .claude/docs/vault-deposit-redeem.md and the Lagoon reference lifecycle in
tests/erc_4626/test_vault_async_lagoon_erc_7540.py before adding another mechanism.
"""

import calendar
import json
import os
from collections.abc import Iterator
from decimal import Decimal
from functools import partial
from pathlib import Path
from unittest.mock import Mock

import pytest
from eth_defi.erc_4626.vault_protocol.atoma.vault import ATOMA_IMPLEMENTATION, ATOMA_VAULT_ADDRESSES, AtomaVault
from eth_defi.erc_4626.vault_protocol.atoma.deposit_redeem import AtomaDepositManager
from eth_defi.compat import native_datetime_utc_now
from eth_defi.erc_4626.vault_protocol.lagoon.vault import LagoonVault
from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import is_anvil, make_anvil_custom_rpc_request, mine
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.provider.receipt import wait_for_transaction_receipt_robust
from eth_defi.testing.anvil_fork_pool import AnvilForkPool
from eth_defi.token import USDC_NATIVE_TOKEN, USDC_WHALE, fetch_erc20_details
from eth_defi.vault.base import VaultSpec
from eth_defi.vault.deposit_redeem import AsyncVaultRequestStatus, VaultFlowUnavailable
from pytest import MonkeyPatch, TempPathFactory
from typer.main import get_command
from web3 import Web3
from web3.exceptions import ContractLogicError

from tradeexecutor.cli.main import app
from tradeexecutor.cli.commands import perform_test_trade as perform_test_trade_command
from tradeexecutor.cli.testtrade import make_test_trade
from tradeexecutor.ethereum.vault.vault_live_pricing import VaultPricing
from tradeexecutor.ethereum.vault.vault_valuation import VaultValuator
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeStatus
from tradeexecutor.strategy.valuation import InvalidValuationOutput, revalue_state


JSON_RPC_ARBITRUM = os.environ.get("JSON_RPC_ARBITRUM")
FORK_BLOCK = 519_150_162
STRATEGY_FILE = Path(__file__).resolve().parents[2] / "strategies/test_only/atoma_async.py"
pytestmark = [
    pytest.mark.skipif(not JSON_RPC_ARBITRUM, reason="JSON_RPC_ARBITRUM required"),
    pytest.mark.warm_rpc_test_group,
    pytest.mark.xdist_group("fork:arbitrum:atoma:519150162"),
]


@pytest.fixture(scope="module")
def atoma_runtime(anvil_fork_pool: AnvilForkPool, tmp_path_factory: TempPathFactory) -> tuple[dict[str, str], Web3]:
    """Deploy and subscribe once, preserving real protocol state and permissions."""
    launch = anvil_fork_pool.get_launch(JSON_RPC_ARBITRUM, FORK_BLOCK)
    web3 = create_multi_provider_web3(launch.json_rpc_url)
    assert is_anvil(web3)
    root = tmp_path_factory.mktemp("atoma-cli")
    wallet = HotWallet.create_for_testing(web3, test_account_n=8, eth_amount=10)
    governance = HotWallet.create_for_testing(web3, test_account_n=7, eth_amount=1)
    token = fetch_erc20_details(web3, USDC_NATIVE_TOKEN[42161])
    whale = Web3.to_checksum_address(USDC_WHALE[42161])
    make_anvil_custom_rpc_request(web3, "anvil_impersonateAccount", [whale])
    make_anvil_custom_rpc_request(web3, "anvil_setBalance", [whale, hex(10**18)])
    try:
        tx = token.transfer(wallet.address, Decimal(500)).transact({"from": whale})
        assert wait_for_transaction_receipt_robust(web3, tx)["status"] == 1
    finally:
        make_anvil_custom_rpc_request(web3, "anvil_stopImpersonatingAccount", [whale])
    environment = {
        "PATH": os.environ["PATH"],
        "EXECUTOR_ID": "atoma-cli",
        "NAME": "Atoma CLI",
        "STRATEGY_FILE": str(STRATEGY_FILE),
        "STATE_FILE": str(root / "state.json"),
        "VAULT_RECORD_FILE": str(root / "record.txt"),
        "PRIVATE_KEY": Web3.to_hex(wallet.private_key),
        "JSON_RPC_ARBITRUM": launch.json_rpc_url,
        "ASSET_MANAGEMENT_MODE": "lagoon",
        "UNIT_TESTING": "true",
        "SIMULATE": "false",
        "LOG_LEVEL": "disabled",
        "CONFIRMATION_BLOCK_COUNT": "1",
        "MIN_GAS_BALANCE": "0.001",
        "GAS_BALANCE_WARNING_LEVEL": "0",
        "TRADING_STRATEGY_API_KEY": "secret-token:unused-static-universe",
        "CACHE_PATH": str(root / "cache"),
        "MAX_CYCLES": "1",
        "TRADE_IMMEDIATELY": "true",
        "POSITION_TRIGGER_CHECK_MINUTES": "0",
        "STATS_REFRESH_MINUTES": "0",
        "VISUALISATION": "false",
        "CHECK_ACCOUNTS": "true",
        "LAGOON_MAX_SETTLEMENT_AMOUNT": "5000",
    }
    cli = get_command(app)
    with MonkeyPatch.context() as patch:
        patch.setattr(os, "environ", environment.copy())
        cli.main(args=["lagoon-deploy-vault", "--fund-name", "Atoma CLI", "--fund-symbol", "ACLI", "--multisig-owners", governance.address, "--performance-fee", "0", "--management-fee", "0"], standalone_mode=False)
        deployment = json.loads((root / "atoma-cli.deployment.json").read_text())["deployments"]["arbitrum"]
        environment.update(VAULT_ADDRESS=deployment["vault_address"], VAULT_ADAPTER_ADDRESS=deployment["module_address"])
        patch.setattr(os, "environ", environment.copy())
        cli.main(args=["init"], standalone_mode=False)
        cli.main(args=["lagoon-first-deposit", "--deposit-amount", "400"], standalone_mode=False)
        cli.main(args=["lagoon-settle"], standalone_mode=False)
    return environment, web3


@pytest.fixture()
def atoma_environment(atoma_runtime: tuple[dict[str, str], Web3], monkeypatch: MonkeyPatch) -> Iterator[dict[str, str]]:
    """Restore both the fork and the normal executor state between CLI scenarios."""
    environment, web3 = atoma_runtime
    state_path = Path(environment["STATE_FILE"])
    saved = state_path.read_bytes()
    snapshot = make_anvil_custom_rpc_request(web3, "evm_snapshot", [])
    monkeypatch.setattr(os, "environ", environment.copy())
    try:
        yield environment
    finally:
        state_path.write_bytes(saved)
        make_anvil_custom_rpc_request(web3, "evm_revert", [snapshot])


def test_atoma_start_blackbox(atoma_environment: dict[str, str], atoma_runtime: tuple[dict[str, str], Web3]) -> None:
    """Run both deployed ABIs through start, pending restart, settlement and claim.

    1. Start the dummy strategy and check synchronous deposits and locked exits.
    2. Advance only fork time, request exits through start and repeat while pending.
    3. Settle as the real operator, verify burned shares and claim entitlements.
    4. Restart start to claim, reconcile cash and prove repeating cannot trade again.
    5. Post final Lagoon NAV and return the investor's cash through the normal CLI.
    """
    cli = get_command(app)
    web3 = atoma_runtime[1]
    state_path = Path(atoma_environment["STATE_FILE"])
    vaults = [AtomaVault(web3, VaultSpec(42161, address)) for address in sorted(ATOMA_VAULT_ADDRESSES)]

    # 1. Start the dummy strategy and check synchronous deposits and locked exits.
    cli.main(args=["start"], standalone_mode=False)
    state = State.read_json_file(state_path)
    assert len(state.portfolio.open_positions) == 2
    assert len(list(state.portfolio.get_all_trades())) == 2
    assert state.portfolio.get_cash() == pytest.approx(200)
    lagoon = LagoonVault(web3, VaultSpec(42161, atoma_environment["VAULT_ADDRESS"]), trading_strategy_module_address=atoma_environment["VAULT_ADAPTER_ADDRESS"])
    owner = lagoon.safe_address
    for vault in vaults:
        assert vault.fetch_implementation() == ATOMA_IMPLEMENTATION
        assert vault.share_token.decimals == 12
        assert vault.get_deposit_manager().fetch_unlocked_shares(owner) == 0
        module = lagoon.trading_strategy_module
        assert module.functions.isAllowedCallSite(vault.address, Web3.keccak(text="requestWithdrawal(uint256)")[:4]).call()
        for function in (vault.vault_contract.functions.redeem(1, owner, owner), vault.vault_contract.functions.pause()):
            # Real Guard execution rejects depositor bypasses and operator authority.
            with pytest.raises((ContractLogicError, ValueError), match="Selector not allowed"):
                module.functions.performCall(vault.address, function._encode_transaction_data()).call({"from": HotWallet.from_private_key(atoma_environment["PRIVATE_KEY"]).address})
    cli.main(args=["start"], standalone_mode=False)
    assert len(list(State.read_json_file(state_path).portfolio.get_all_trades())) == 2

    # 2. Advance only fork time, request exits through start and repeat while pending.
    lock_ends = [vault.get_deposit_manager().get_redemption_delay_over(owner) for vault in vaults]
    mine(web3, timestamp=max(calendar.timegm(end.timetuple()) for end in lock_ends))
    cli.main(args=["start"], standalone_mode=False)
    pending = State.read_json_file(state_path)
    sells = [t for t in pending.portfolio.get_all_trades() if t.is_sell()]
    assert len(sells) == 2
    assert all(t.get_status() == TradeStatus.vault_settlement_pending for t in sells)
    assert pending.portfolio.get_cash() == pytest.approx(200)
    assert pending.portfolio.get_net_asset_value() == pytest.approx(399, abs=0.02)
    with pytest.raises(SystemExit) as checked:
        cli.main(args=["check-accounts"], standalone_mode=False)
    assert checked.value.code == 0
    request_hashes = [t.other_data["vault_request_tx_hash"] for t in sells]
    cli.main(args=["start"], standalone_mode=False)
    repeated = State.read_json_file(state_path)
    assert [t.other_data["vault_request_tx_hash"] for t in repeated.portfolio.get_all_trades() if t.is_sell()] == request_hashes

    # 3. Settle as the real operator, verify burned shares and claim entitlements.
    expected_payout = Decimal(0)
    for trade in sells:
        vault = next(v for v in vaults if v.address.lower() == trade.pair.pool_address.lower())
        manager = vault.get_deposit_manager()
        ticket = manager.reconstruct_redemption_ticket(trade.other_data)
        result = manager.force_settle(ticket)
        assert result.status_after is AsyncVaultRequestStatus.claimable
        assert vault.share_token.fetch_raw_balance_of(owner) == 0
        assert manager.can_finish_redeem(ticket)
        expected_payout += vault.denomination_token.convert_to_decimals(manager.fetch_claim_amounts(ticket)[2])
    with pytest.raises(SystemExit) as checked:
        cli.main(args=["check-accounts"], standalone_mode=False)
    assert checked.value.code == 0

    # 4. Restart start to claim, reconcile cash and prove repeating cannot trade again.
    cli.main(args=["start"], standalone_mode=False)
    final = State.read_json_file(state_path)
    assert not final.portfolio.open_positions and not final.portfolio.frozen_positions
    assert len(final.portfolio.closed_positions) == 2
    assert all(t.is_success() for t in final.portfolio.get_all_trades())
    assert final.portfolio.get_cash() == pytest.approx(Decimal(200) + expected_payout)
    assert sum((Decimal(t.other_data["vault_withdrawal_fee"]) for t in final.portfolio.get_all_trades() if t.is_sell()), Decimal(0)) == pytest.approx(Decimal(1), abs=Decimal("0.000002"))
    for vault in vaults:
        assert vault.share_token.fetch_raw_balance_of(owner) == 0
    cli.main(args=["start"], standalone_mode=False)
    assert len(list(State.read_json_file(state_path).portfolio.get_all_trades())) == 4

    # 5. Post final Lagoon NAV and return the investor's cash through the normal CLI.
    cli.main(args=["lagoon-settle", "--force-valuation"], standalone_mode=False)
    assert lagoon.fetch_total_assets(web3.eth.block_number) == pytest.approx(Decimal(200) + expected_payout)
    cli.main(args=["lagoon-redeem"], standalone_mode=False)
    wallet = HotWallet.from_private_key(atoma_environment["PRIVATE_KEY"])
    assert lagoon.share_token.fetch_raw_balance_of(wallet.address) == 0
    assert vaults[0].denomination_token.fetch_balance_of(wallet.address) == pytest.approx(Decimal(300) + expected_payout)
    # Diagnostic evidence only; neither the CLI nor this test uses it for recovery.
    evidence = {
        "fork_block": FORK_BLOCK,
        "implementation": ATOMA_IMPLEMENTATION,
        "safe": owner,
        "claims": [
            {"vault": p.pair.pool_address,
             "request": p.get_last_trade().other_data["vault_request_tx_hash"],
             "claim": p.get_last_trade().blockchain_transactions[-1].tx_hash,
             "shares": str(-p.get_last_trade().executed_quantity),
             "net_usdc": str(p.get_last_trade().executed_reserve),
             "fee_usdc": p.get_last_trade().other_data["vault_withdrawal_fee"]}
            for p in final.portfolio.closed_positions.values()
        ],
    }
    (state_path.parent / "atoma-fork-evidence.json").write_text(json.dumps(evidence, indent=2))


def test_atoma_test_trade_resume(atoma_environment: dict[str, str], atoma_runtime: tuple[dict[str, str], Web3], monkeypatch: MonkeyPatch) -> None:
    """Resume a test-owned exit through the public command without a duplicate request.

    1. Deposit through buy-only and refuse an immediate locked exit without a trade.
    2. Unlock and request once, holding only the existing Anvil test-settlement driver.
    3. Settle externally, resume the command and verify repeat safety and fee-net cash.
    """
    cli = get_command(app)
    web3 = atoma_runtime[1]
    state_path = Path(atoma_environment["STATE_FILE"])
    vault = AtomaVault(web3, VaultSpec(42161, sorted(ATOMA_VAULT_ADDRESSES)[0]))
    pair = f"(arbitrum, {vault.address})"
    # 1. Deposit through buy-only and refuse an immediate locked exit without a trade.
    cli.main(args=["perform-test-trade", "--pair", pair, "--amount", "100", "--buy-only", "--no-test-short"], standalone_mode=False)
    with pytest.raises(VaultFlowUnavailable, match="insufficient_unlocked_shares"):
        cli.main(args=["perform-test-trade", "--pair", pair, "--close-only", "--no-test-short"], standalone_mode=False)
    state = State.read_json_file(state_path)
    assert len(list(state.portfolio.get_all_trades())) == 1
    assert not state.portfolio.frozen_positions
    lagoon = LagoonVault(web3, VaultSpec(42161, atoma_environment["VAULT_ADDRESS"]))
    owner = lagoon.safe_address
    # 2. Unlock and request once, holding only the existing Anvil test-settlement driver.
    end = vault.get_deposit_manager().get_redemption_delay_over(owner)
    mine(web3, timestamp=calendar.timegm(end.timetuple()))
    # Only the harness's existing force flag changes; routing and receipts stay real.
    monkeypatch.setattr(perform_test_trade_command, "make_test_trade", partial(make_test_trade, force_async_settlement_on_anvil=False))
    cli.main(args=["perform-test-trade", "--pair", pair, "--close-only", "--no-test-short"], standalone_mode=False)
    pending = State.read_json_file(state_path)
    sell = next(t for t in pending.portfolio.get_all_trades() if t.is_sell())
    assert sell.get_status() is TradeStatus.vault_settlement_pending
    cli.main(args=["perform-test-trade", "--pair", pair, "--close-only", "--no-test-short"], standalone_mode=False)
    assert len(list(State.read_json_file(state_path).portfolio.get_all_trades())) == 2
    # 3. Settle externally, resume the command and verify repeat safety and fee-net cash.
    manager = vault.get_deposit_manager()
    manager.force_settle(manager.reconstruct_redemption_ticket(sell.other_data))
    cli.main(args=["perform-test-trade", "--pair", pair, "--close-only", "--no-test-short"], standalone_mode=False)
    cli.main(args=["perform-test-trade", "--pair", pair, "--close-only", "--no-test-short"], standalone_mode=False)
    final = State.read_json_file(state_path)
    assert not final.portfolio.open_positions and not final.portfolio.frozen_positions
    assert len(list(final.portfolio.get_all_trades())) == 2
    assert final.portfolio.get_cash() == pytest.approx(399.5, abs=0.000002)


def test_atoma_haircut_and_zero_claim(atoma_environment: dict[str, str], atoma_runtime: tuple[dict[str, str], Web3]) -> None:
    """Real owner loss reporting changes settled valuation and closes zero-payout trades.

    1. Create and settle strategy requests through the real start and operator paths.
    2. Induce a haircut then total loss with real privileged calls on this snapshot.
    3. Revalue the same positions, restart to claim zero and verify realised losses/NAV.
    """
    cli = get_command(app)
    web3 = atoma_runtime[1]
    state_path = Path(atoma_environment["STATE_FILE"])
    lagoon = LagoonVault(web3, VaultSpec(42161, atoma_environment["VAULT_ADDRESS"]))
    vaults = [AtomaVault(web3, VaultSpec(42161, address)) for address in sorted(ATOMA_VAULT_ADDRESSES)]
    # 1. Create and settle strategy requests through the real start and operator paths.
    cli.main(args=["start"], standalone_mode=False)
    mine(web3, timestamp=max(calendar.timegm(v.get_deposit_manager().get_redemption_delay_over(lagoon.safe_address).timetuple()) for v in vaults))
    cli.main(args=["start"], standalone_mode=False)
    state = State.read_json_file(state_path)
    for trade in state.portfolio.get_all_trades():
        if trade.is_sell():
            manager = next(v for v in vaults if v.address.lower() == trade.pair.pool_address).get_deposit_manager()
            manager.force_settle(manager.reconstruct_redemption_ticket(trade.other_data))
    valuator = VaultValuator(VaultPricing(web3, owner_address_resolver=lambda pair: lagoon.safe_address))
    revalue_state(state, native_datetime_utc_now(), valuator)
    before = state.portfolio.get_net_asset_value()
    # 2. Induce a haircut then total loss with real privileged calls on this snapshot.
    for vault in vaults:
        f = vault.vault_contract.functions
        for role, function in (("operator", f.pause()), ("owner", f.resyncTotalAssets(f.settledUnclaimedAssets().call() // 2))):
            actor = getattr(f, role)().call()
            make_anvil_custom_rpc_request(web3, "anvil_impersonateAccount", [actor])
            make_anvil_custom_rpc_request(web3, "anvil_setBalance", [actor, hex(10**18)])
            try:
                assert wait_for_transaction_receipt_robust(web3, function.transact({"from": actor}))["status"] == 1
            finally:
                make_anvil_custom_rpc_request(web3, "anvil_stopImpersonatingAccount", [actor])
    revalue_state(state, native_datetime_utc_now(), valuator)
    assert 200 < state.portfolio.get_net_asset_value() < before
    for vault in vaults:
        actor = vault.vault_contract.functions.owner().call()
        make_anvil_custom_rpc_request(web3, "anvil_impersonateAccount", [actor])
        try:
            assert wait_for_transaction_receipt_robust(web3, vault.vault_contract.functions.resyncTotalAssets(0).transact({"from": actor}))["status"] == 1
        finally:
            make_anvil_custom_rpc_request(web3, "anvil_stopImpersonatingAccount", [actor])
    # 3. Revalue the same positions, restart to claim zero and verify realised losses/NAV.
    revalue_state(state, native_datetime_utc_now(), valuator)
    assert state.portfolio.get_net_asset_value() == pytest.approx(200)
    state.write_json_file(state_path)
    cli.main(args=["start"], standalone_mode=False)
    final = State.read_json_file(state_path)
    assert len(final.portfolio.closed_positions) == 2
    assert not final.portfolio.frozen_positions
    assert all(t.is_success() for t in final.portfolio.get_all_trades())
    assert all(t.executed_reserve == 0 for t in final.portfolio.get_all_trades() if t.is_sell())
    assert sum(p.get_realised_profit_usd() for p in final.portfolio.closed_positions.values()) == pytest.approx(-200)
    assert final.portfolio.get_cash() == pytest.approx(200)
    assert lagoon.fetch_total_assets(web3.eth.block_number) == pytest.approx(200)


def test_atoma_claim_restart(atoma_environment: dict[str, str], atoma_runtime: tuple[dict[str, str], Web3], monkeypatch: MonkeyPatch) -> None:
    """Recover real paid claims saved before broadcast but not yet accounted for.

    1. Deposit and request through the dummy strategy, then settle on the fork.
    2. Interrupt receipt analysis after real claims pay, leaving their signed identities saved.
    3. Reload through start and apply those exact receipts once before valuation/NAV.
    """
    cli = get_command(app)
    web3 = atoma_runtime[1]
    state_path = Path(atoma_environment["STATE_FILE"])
    lagoon = LagoonVault(web3, VaultSpec(42161, atoma_environment["VAULT_ADDRESS"]))
    vaults = [AtomaVault(web3, VaultSpec(42161, address)) for address in sorted(ATOMA_VAULT_ADDRESSES)]
    # 1. Deposit and request through the dummy strategy, then settle on the fork.
    cli.main(args=["start"], standalone_mode=False)
    mine(web3, timestamp=max(calendar.timegm(v.get_deposit_manager().get_redemption_delay_over(lagoon.safe_address).timetuple()) for v in vaults))
    cli.main(args=["start"], standalone_mode=False)
    state = State.read_json_file(state_path)
    for trade in state.portfolio.get_all_trades():
        if trade.is_sell():
            manager = next(v for v in vaults if v.address.lower() == trade.pair.pool_address).get_deposit_manager()
            manager.force_settle(manager.reconstruct_redemption_ticket(trade.other_data))
    # 2. Interrupt receipt analysis after real claims pay, leaving their signed identities saved.
    # This is a crash injection, not a fake receipt: the real transaction has already paid.
    with monkeypatch.context() as interruption:
        interruption.setattr(AtomaDepositManager, "analyse_redemption", Mock(side_effect=RuntimeError("Interrupted receipt accounting")))
        with pytest.raises(InvalidValuationOutput, match="Recover"):
            cli.main(args=["start"], standalone_mode=False)
    interrupted = State.read_json_file(state_path)
    sells = [t for t in interrupted.portfolio.get_all_trades() if t.is_sell()]
    claim_hashes = [t.blockchain_transactions[-1].tx_hash for t in sells]
    assert all(t.get_status() is TradeStatus.vault_settlement_pending for t in sells)
    assert all(web3.eth.get_transaction_receipt(tx)["status"] == 1 for tx in claim_hashes)
    assert interrupted.portfolio.get_cash() == pytest.approx(200)
    # 3. Reload through start and apply those exact receipts once before valuation/NAV.
    cli.main(args=["start"], standalone_mode=False)
    recovered = State.read_json_file(state_path)
    assert not recovered.portfolio.open_positions and not recovered.portfolio.frozen_positions
    assert recovered.portfolio.get_cash() == pytest.approx(399, abs=0.000002)
    assert [t.blockchain_transactions[-1].tx_hash for t in recovered.portfolio.get_all_trades() if t.is_sell()] == claim_hashes
    cli.main(args=["start"], standalone_mode=False)
    assert State.read_json_file(state_path).portfolio.get_cash() == pytest.approx(399, abs=0.000002)


def test_atoma_unreconciled_ticket_blocks_nav(atoma_environment: dict[str, str], atoma_runtime: tuple[dict[str, str], Web3]) -> None:
    """Refuse an unreconciled ticket even when its previous valuation is recent.

    1. Create real strategy withdrawal requests with a recent valuation.
    2. Persist the adapter's reconciliation-required marker on one ticket.
    3. Restart start and verify no claim/NAV transaction or timestamp refresh occurs.
    """
    cli = get_command(app)
    web3 = atoma_runtime[1]
    state_path = Path(atoma_environment["STATE_FILE"])
    lagoon = LagoonVault(web3, VaultSpec(42161, atoma_environment["VAULT_ADDRESS"]))
    vaults = [AtomaVault(web3, VaultSpec(42161, address)) for address in sorted(ATOMA_VAULT_ADDRESSES)]
    # 1. Create real strategy withdrawal requests with a recent valuation.
    cli.main(args=["start"], standalone_mode=False)
    mine(web3, timestamp=max(calendar.timegm(v.get_deposit_manager().get_redemption_delay_over(lagoon.safe_address).timetuple()) for v in vaults))
    cli.main(args=["start"], standalone_mode=False)
    state = State.read_json_file(state_path)
    position = next(iter(state.portfolio.open_positions.values()))
    valued_at = position.last_pricing_at
    # 2. Persist the adapter's reconciliation-required marker on one ticket.
    # Quote attribution itself is covered with real manager logic in its unit tests.
    position.get_last_trade().other_data["atoma_reconciliation_required"] = True
    state.write_json_file(state_path)
    wallet = HotWallet.from_private_key(atoma_environment["PRIVATE_KEY"])
    nonce = web3.eth.get_transaction_count(wallet.address)
    # 3. Restart start and verify no claim/NAV transaction or timestamp refresh occurs.
    with pytest.raises(InvalidValuationOutput, match="reconciliation"):
        cli.main(args=["start"], standalone_mode=False)
    refused = State.read_json_file(state_path)
    assert refused.portfolio.open_positions[position.position_id].last_pricing_at == valued_at
    assert all(t.get_status() is TradeStatus.vault_settlement_pending for t in refused.portfolio.get_all_trades() if t.is_sell())
    assert web3.eth.get_transaction_count(wallet.address) == nonce
    assert refused.portfolio.get_cash() == pytest.approx(200)
