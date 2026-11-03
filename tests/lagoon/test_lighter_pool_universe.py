"""Lighter public-pool identity, routing, deployment translation, admission and operation resume."""

import datetime
import json
from contextlib import ExitStack
from decimal import Decimal
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import click
import pytest
from eth_defi.lighter.constants import LIGHTER_USDC_ETHEREUM
from eth_defi.token import USDC_NATIVE_TOKEN
from eth_defi.vault.deposit_redeem import VaultRequestInvariantViolation
from web3 import Web3

from tradeexecutor.cli.commands.pair_mapping import parse_pair_data
from tradeexecutor.ethereum.lagoon.universe_config import (
    _collect_chain_token_addresses,
    _collect_erc4626_vault_addresses,
    translate_trading_universe_to_lagoon_config,
)
from eth_defi.lighter.execution import LighterOperation, LighterOperationStage, LighterPoolExecutor, prepare_pool_deposit

from tradeexecutor.ethereum.lighter.lighter_pool import create_lighter_pool_pair
from tradeexecutor.ethereum.lighter.lighter_pool_routing import LighterPoolContext, LighterPoolRouting
from tradeexecutor.ethereum.lighter.lighter_pool_valuation import LighterPoolPricing
from tradeexecutor.exchange_account.lighter_operator import LighterOperatorRecord
from tradeexecutor.state.identifier import normalise_custody_chain_id
from tradeexecutor.strategy.generic.default_protocols import default_match_router, default_supported_routers
from tradeexecutor.strategy.redemption import DepositBlockReason
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.testing.erc4626_manual import load_universe
from tradeexecutor.testing.lighter_core import FakeLighterAccount, FakeLighterCore, FakeLighterPool
from tradeexecutor.testing import multi_vault_manual


STRATEGY_FILE = Path(__file__).resolve().parents[2] / "strategies/test_only/multi_vault_arbitrum.py"

#: pmalt, the tutorial pool.
PMALT = 281474976552918


@pytest.fixture(scope="module")
def universe() -> TradingStrategyUniverse:
    """The Arbitrum hub tutorial universe, built without market data."""
    return load_universe(STRATEGY_FILE)


def test_lighter_pool_routes_and_deploys_on_ethereum_satellite(universe: TradingStrategyUniverse) -> None:
    """A pool pair uses the pool router and the Ethereum satellite's custody and Guard.

    Lighter pools carry synthetic dataset chain 9998. Routing, funding and
    deployment must use Ethereum (chain 1) without passing the synthetic
    ``lighter-pool-`` identity to any ERC-20 or ERC-4626 code.

    1. Find the pool pair in the tutorial universe and check its identities.
    2. Check router selection for the pool and the other vaults.
    3. Collect guard vaults and tokens: the pool is in neither.
    4. Translate the universe to Lagoon configs: only Ethereum permits Lighter.
    5. Parse the pool identity used by perform-test-trade.
    """
    # 1. Find the pool pair in the tutorial universe and check its identities.
    pool = next(p for p in universe.iterate_pairs() if p.is_lighter_pool())
    assert pool.chain_id == 9998
    assert pool.get_lighter_pool_index() == PMALT
    assert pool.get_funding_chain_id() == 1
    assert normalise_custody_chain_id(pool.chain_id) == 1
    assert normalise_custody_chain_id(9999) == 999
    assert normalise_custody_chain_id(42161) == 42161
    assert pool.has_delayed_vault_redemption()

    # 2. Check router selection for the pool and the other vaults.
    assert default_match_router(universe, pool).router_name == "lighter_pool"
    routers = {routing_id.router_name for routing_id in default_supported_routers(universe)}
    assert routers == {"lighter_pool", "vault", "cctp-bridge"}

    # 3. Collect guard vaults and tokens: the pool is in neither.
    vaults = _collect_erc4626_vault_addresses(universe, {42161, 1})
    assert vaults == {
        42161: [Web3.to_checksum_address("0xcc56410e1a136af0eceb7241c6ae394f4d8b581c")],
        1: [Web3.to_checksum_address("0xE9B725010A9E419412ed67d0fA5f3A5f40159D32")],
    }
    tokens = _collect_chain_token_addresses(universe, {42161, 1}, any_asset=False)
    assert Web3.to_checksum_address(LIGHTER_USDC_ETHEREUM) in tokens[1]
    assert Web3.to_checksum_address(USDC_NATIVE_TOKEN[42161]) in tokens[42161]
    assert not any("lighter-pool" in address for addresses in tokens.values() for address in addresses)

    # 4. Translate the universe to Lagoon configs: only Ethereum permits Lighter.
    # No RPC is read while building configs; the mocks only satisfy the
    # per-chain connection lookup and vault descriptor construction.
    configs = translate_trading_universe_to_lagoon_config(
        universe=universe,
        chain_web3={"arbitrum": MagicMock(), "ethereum": MagicMock()},
        safe_owners=["0x0000000000000000000000000000000000000001"],
        safe_threshold=1,
        safe_salt_nonce=1,
        asset_managers=["0x0000000000000000000000000000000000000001"],
    )
    assert set(configs) == {"arbitrum", "ethereum"}
    assert configs["ethereum"].lighter_deployment is not None and configs["ethereum"].satellite_chain
    assert configs["arbitrum"].lighter_deployment is None and not configs["arbitrum"].satellite_chain

    # 5. Parse the pool identity used by perform-test-trade.
    chain, address = parse_pair_data(f"(lighter, lighter-pool-{PMALT})")
    assert (chain.value, address) == (9998, f"lighter-pool-{PMALT}")


def test_lighter_pool_full_closed_or_malformed_is_refused() -> None:
    """Deposits into a full or closed pool, or a malformed identity, are refused before money moves.

    Steady Wealth Builder's operator held exactly the 5% minimum of supply on
    3 November 2026, so no share can be minted. The admission check reads live
    pool state through the upstream parser, answered here by the fake API.

    1. Serve a full pool and a closed pool from the fake Lighter API.
    2. Check that pricing refuses both with stable reasons and zero capacity.
    3. Check that a malformed pool identity cannot be parsed.
    """
    # 1. Serve a full pool and a closed pool from the fake Lighter API.
    # The fake answers reads only, so it needs no Ethereum fork connection.
    core = FakeLighterCore(None, master_index=127)
    core.add_pool(FakeLighterPool(index=281474976552443, equity=Decimal("75901.64"), total_shares=45_400_960, operator_shares=2_270_048))
    core.add_pool(FakeLighterPool(index=PMALT, equity=Decimal("35301.90"), total_shares=47_656_547, operator_shares=42_136_849, status=1))
    pricing = LighterPoolPricing(core.create_api_session())
    full = create_lighter_pool_pair(281474976552443, internal_id=1, internal_exchange_id=1, name="Steady Wealth Builder")
    closed = create_lighter_pool_pair(PMALT, internal_id=2, internal_exchange_id=1, name="pmalt")

    # 2. Check that pricing refuses both with stable reasons and zero capacity.
    full_result = pricing.check_deposit(None, full)
    assert not full_result.can_deposit
    assert full_result.reason_code == DepositBlockReason.vault_max_deposit_zero
    assert pricing.get_max_deposit(None, full) == 0
    closed_result = pricing.check_deposit(None, closed)
    assert not closed_result.can_deposit
    assert closed_result.reason_code == DepositBlockReason.vault_deposits_closed
    assert pricing.check_redemption(None, closed).can_redeem

    # 3. Check that a malformed pool identity cannot be parsed.
    malformed = create_lighter_pool_pair(1, internal_id=3, internal_exchange_id=1)
    malformed.pool_address = "lighter-pool-abc"
    with pytest.raises(AssertionError, match="Malformed Lighter pool identity"):
        malformed.get_lighter_pool_index()


def test_lighter_pool_operation_resume_never_resubmits() -> None:
    """A restarted pool phase reconciles its saved operation instead of signing another.

    The router saves each upstream ``LighterOperation`` on the trade before
    submission. After a restart it must return a confirmed operation without a
    second sequencer transaction, replace a never-sent prepared operation with a
    fresh one, and halt on a rejected one.

    1. Mint once through the router helper against the fake sequencer and API.
    2. Rerun with the saved confirmed operation: no new transaction is created.
    3. Downgrade the saved stage to prepared: one fresh operation is signed and sent.
    4. Mark the saved operation rejected: the helper halts with an invariant violation.
    """
    core = FakeLighterCore(None, master_index=127)
    core.accounts[127] = FakeLighterAccount(index=127, role="master", collateral=Decimal(10))
    core.add_pool(FakeLighterPool(index=PMALT, equity=Decimal("35301.90"), total_shares=47_656_547, operator_shares=42_136_849))
    operator = LighterOperatorRecord(vault_address="0x" + "1" * 40, safe_address="0x" + "2" * 40, module_address="0x" + "3" * 40, account_index=127, api_key_index=4, api_private_key="0x00")
    # Only the API session and operator are used by the operation helper; the
    # Ethereum connection, builder and contracts belong to the EVM phases.
    ctx = LighterPoolContext(web3=None, tx_builder=None, safe_address=operator.safe_address, usdc=None, zk=None, session=core.create_api_session(), operator=operator)
    router = LighterPoolRouting(session=ctx.session, poll_interval=0.0)
    signer = core.create_pool_signer(operator)
    data = {"operations": {}}

    def prepare(executor: LighterPoolExecutor) -> LighterOperation:
        return prepare_pool_deposit(executor, PMALT, Decimal(5))

    # Upstream submits through its own requests session; the pool patches
    # redirect that submission to the fake sequencer instead of the live API.
    with ExitStack() as patches:
        for target, replacement in core.get_pool_patches():
            patches.enter_context(mock.patch(target, new=replacement))

        # 1. Mint once through the router helper against the fake sequencer and API.
        first = router._run_operation(ctx, data, "mint", signer, prepare)
        assert first.stage == LighterOperationStage.confirmed and first.amount > 0
        assert len(core.api_transactions) == 1
        assert core.accounts[127].pool_shares[PMALT].shares == first.amount

        # 2. Rerun with the saved confirmed operation: no new transaction is created.
        again = router._run_operation(ctx, data, "mint", signer, prepare)
        assert again.tx_hash == first.tx_hash and len(core.api_transactions) == 1

        # 3. Downgrade the saved stage to prepared: one fresh operation is signed and sent.
        saved = json.loads(data["operations"]["mint"])
        saved["stage"] = LighterOperationStage.prepared.value
        data["operations"]["mint"] = json.dumps(saved)
        fresh = router._run_operation(ctx, data, "mint", signer, prepare)
        assert fresh.tx_hash != first.tx_hash and fresh.nonce == first.nonce + 1
        assert len(core.api_transactions) == 2

        # 4. Mark the saved operation rejected: the helper halts with an invariant violation.
        saved = json.loads(data["operations"]["mint"])
        saved["stage"] = LighterOperationStage.rejected.value
        data["operations"]["mint"] = json.dumps(saved)
        with pytest.raises(VaultRequestInvariantViolation, match="was rejected"):
            router._run_operation(ctx, data, "mint", signer, prepare)
        assert len(core.api_transactions) == 2


def test_multi_vault_tutorial_resume_checks(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """A resumed live tutorial reads the vault state before it needs money or a lock.

    Both cases stopped a funded run on 2026-11-03 or would have: simulation
    does not reach them. The chain is mocked because each case needs a vault
    state that a fork only reaches after days of epochs.

    1. A pending subscription needs no money in an empty investor wallet
    2. A new subscription without the money fails before the approval
    3. Unlocked Atoma shares have no deadline and exit normally
    4. Locked Atoma shares exit with the status the script waits on
    """
    usdc = MagicMock()
    usdc.fetch_balance_of.return_value = Decimal(0)
    vault = MagicMock()
    pending = vault.vault_contract.functions.pendingDepositRequest.return_value.call
    claimable = vault.vault_contract.functions.maxDeposit.return_value.call
    monkeypatch.setenv("JSON_RPC_ARBITRUM", "http://localhost")
    monkeypatch.setenv("PRIVATE_KEY", "0x" + "11" * 32)
    monkeypatch.setenv("VAULT_ADDRESS", "0x" + "22" * 20)
    monkeypatch.setattr(multi_vault_manual, "create_multi_provider_web3", MagicMock())
    monkeypatch.setattr(multi_vault_manual, "get_vault", lambda web3: vault)
    monkeypatch.setattr(multi_vault_manual, "fetch_erc20_details", lambda *args: usdc)

    # 1. A pending subscription needs no money in an empty investor wallet
    pending.return_value = 111_000_000
    multi_vault_manual.request_deposit(amount="111")
    assert capsys.readouterr().out.strip() == "pending"
    usdc.approve.assert_not_called()

    # 2. A new subscription without the money fails before the approval
    pending.return_value = 0
    claimable.return_value = 0
    with pytest.raises(RuntimeError, match="needs 111"):
        multi_vault_manual.request_deposit(amount="111")
    usdc.approve.assert_not_called()

    # 3. Unlocked Atoma shares have no deadline and exit normally
    manager = MagicMock()
    atoma = MagicMock()
    atoma.return_value.get_deposit_manager.return_value = manager
    web3 = MagicMock()
    web3.eth.get_block.return_value = {"timestamp": 1_800_000_000}
    monkeypatch.setattr(multi_vault_manual, "create_multi_provider_web3", lambda url: web3)
    monkeypatch.setattr(multi_vault_manual, "load_universe", MagicMock())
    monkeypatch.setattr(multi_vault_manual, "get_target_pair", lambda universe, target: MagicMock(pool_address="0x" + "33" * 20))
    monkeypatch.setattr(multi_vault_manual, "LagoonVault", MagicMock())
    monkeypatch.setattr(multi_vault_manual, "AtomaVault", atoma)
    manager.get_redemption_delay_over.return_value = None
    multi_vault_manual.atoma_unlocked(strategy_file=Path("unused.py"))
    assert "unlocked" in capsys.readouterr().out

    # 4. Locked Atoma shares exit with the status the script waits on
    manager.get_redemption_delay_over.return_value = datetime.datetime(2030, 1, 1)
    with pytest.raises(click.exceptions.Exit) as exit_info:
        multi_vault_manual.atoma_unlocked(strategy_file=Path("unused.py"))
    assert exit_info.value.exit_code == 3
    assert "locked until" in capsys.readouterr().out
