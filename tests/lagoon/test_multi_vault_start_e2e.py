"""Strategy-produced Lighter, Atoma and Upshift trades on persistent dual forks."""

import json
import os
import shutil
from collections.abc import Iterator
from decimal import Decimal

import pytest
from eth_defi.erc_4626.vault_protocol.lagoon.vault import LagoonVault
from eth_defi.vault.base import VaultSpec

from tradeexecutor.testing.multi_vault_start import MultiVaultStartDriver, deployed_multi_vault
from tradeexecutor.testing.lighter_core import FakeLighterCore, FAKE_TX_TYPE_MINT


pytestmark = [
    pytest.mark.skipif(not all(os.environ.get(k) for k in ("JSON_RPC_ARBITRUM", "JSON_RPC_ETHEREUM", "TRADING_STRATEGY_API_KEY")), reason="Dual-fork RPCs and normal API credentials required"),
    pytest.mark.skipif(shutil.which("anvil") is None or shutil.which("jq") is None, reason="Anvil and jq required"),
    pytest.mark.timeout(1800),
]


@pytest.fixture()
def driver() -> Iterator[MultiVaultStartDriver]:
    """Deploy through the shared fixed-fork runbook fixture."""
    with deployed_multi_vault() as run:
        yield run


def test_multi_vault_start_lifecycle(driver: MultiVaultStartDriver) -> None:
    """Normal start cycles invest and exit three vaults without diluting bootstrap shares.

    1. Settle bootstrap NAV before an outside subscription and check its share price.
    2. Queue a full redemption, defer settlement and execute all three ordinary buys.
    3. Advance locks, request exits and act as Atoma's operator on the fork.
    4. Resume settlements and bridge-back, settle NAV and redeem the investor.
    """
    # 1. Settle bootstrap NAV before an outside subscription and check its share price.
    driver.command("lagoon-settle", "--force-valuation")
    before = driver.read_state()
    assert before.portfolio.get_lighter_idle_collateral_value() == pytest.approx(1)
    # Forged attestations charge no CCTP fee: the unused 0.01 USDC
    # allowance remains satellite cash, alongside 18.99 hub + 1 Lighter.
    assert before.portfolio.get_net_asset_value() == pytest.approx(20, abs=0.000001)
    assert before.portfolio.get_net_asset_value() == pytest.approx(float(driver.observed_assets()), abs=0.000001)
    web3 = driver.simulation.get_web3(42161)
    vault = LagoonVault(web3, VaultSpec(42161, driver.environment()["VAULT_ADDRESS"]))
    supply_before = vault.fetch_total_supply("latest")
    driver.command("request-deposit", "--amount", "111", helper=True)
    driver.command("lagoon-settle")
    driver.command("claim-shares", helper=True)
    supply_after = vault.fetch_total_supply("latest")
    assert float(supply_after - supply_before) == pytest.approx(float(supply_before) * 111 / before.portfolio.get_net_asset_value(), rel=0.00001)

    # 2. Queue a full redemption, defer settlement and execute all three ordinary buys.
    driver.request_redemption()
    block_before = web3.eth.block_number
    driver.command("lagoon-settle", "--force-valuation")
    assert web3.eth.block_number == block_before
    assert vault.get_flow_manager().fetch_pending_redemption("latest") == supply_after
    rejected = driver.command("lagoon-redeem", check=False)
    assert rejected.returncode != 0 and "Return idle Lighter pool collateral" in rejected.stdout + rejected.stderr
    assert web3.eth.block_number == block_before
    driver.start()
    state = driver.read_state()
    buys = [t for t in state.portfolio.get_all_trades() if not t.pair.is_cctp_bridge() and t.is_buy()]
    assert len(buys) == 3 and all(t.is_success() and not t.is_test() for t in buys)
    lighter = next(t for t in buys if t.pair.is_lighter_pool())
    assert Decimal(lighter.other_data["lighter_idle_allocated"]) == Decimal(1)
    assert Decimal(lighter.other_data["lighter_pool"]["deposit_usdc"]) == Decimal(4)
    assert float(lighter.executed_reserve) <= 5
    assert state.portfolio.get_net_asset_value() == pytest.approx(float(driver.observed_assets()), abs=0.00001)

    # 3. Advance locks, request exits and act as Atoma's operator on the fork.
    driver.advance(4 * 24 * 3600)
    driver.start()
    driver.settle_atoma()

    # 4. Resume settlements and bridge-back, settle NAV and redeem the investor.
    for _ in range(3):
        driver.advance()
        driver.start()
    state = driver.read_state()
    trades = [t for t in state.portfolio.get_all_trades() if not t.pair.is_cctp_bridge()]
    assert len(trades) == 6 and all(t.is_success() and not t.is_test() for t in trades)
    assert state.portfolio.get_lighter_idle_collateral_value() < 0.000001
    assert state.portfolio.get_net_asset_value() == pytest.approx(float(driver.observed_assets()), abs=0.00001)
    driver.command("lagoon-settle", "--force-valuation")
    driver.command("lagoon-redeem")
    driver.command("finished", helper=True)


def test_multi_vault_start_restart_and_refusal(driver: MultiVaultStartDriver) -> None:
    """Uncertain sequencer submissions and failed NAV reads survive fresh CLI processes.

    1. Refuse an account-read failure before a pending subscription can settle.
    2. Restore the provider and submit a mint whose confirmation is delayed.
    3. Restart and reconcile the same signed operation without another mint.
    4. Finish delayed Atoma settlement and all decision-produced exits.

    The harness delays only sequencer confirmation and public API reads; signed
    operations, state callbacks and execution are the production implementations.
    """
    # 1. Refuse an account-read failure before a pending subscription can settle.
    driver.command("request-deposit", "--amount", "111", helper=True)
    web3 = driver.simulation.get_web3(42161)
    block_before = web3.eth.block_number
    rejected = driver.command("lagoon-settle", controls=("--fail-account-read",), check=False)
    assert rejected.returncode != 0
    assert "Simulated public account read failure" in rejected.stdout + rejected.stderr
    assert web3.eth.block_number == block_before  # Neither NAV nor settlement was broadcast.

    # 2. Restore the provider and submit a mint whose confirmation is delayed.
    driver.command("lagoon-settle")
    driver.command("claim-shares", helper=True)
    driver.start(controls=("--delay-pool-confirmation",))
    state = driver.read_state()
    assert state.portfolio.has_incomplete_lighter_valuation()
    pool = next(t for t in state.portfolio.get_all_trades() if t.pair.is_lighter_pool())
    original = json.loads(pool.other_data["lighter_pool"]["operations"]["mint"])["tx_hash"]
    statistics_before = len(state.stats.portfolio)
    # A further process must remain alive when the same operation is still
    # pending, without publishing partial NAV or new portfolio statistics.
    driver.start(controls=("--delay-pool-confirmation",))
    assert len(driver.read_state().stats.portfolio) == statistics_before
    block_before = web3.eth.block_number
    driver.command("lagoon-settle", controls=("--delay-pool-confirmation",))
    assert web3.eth.block_number == block_before

    # 3. Restart and reconcile the same signed operation without another mint.
    driver.advance()
    driver.start()
    pool = driver.read_state().portfolio.get_trade_by_id(pool.trade_id)
    assert pool.is_success()
    assert json.loads(pool.other_data["lighter_pool"]["operations"]["mint"])["tx_hash"] == original
    core = FakeLighterCore.load(driver.simulation.get_web3(1), driver.simulation.fake_lighter_file)
    assert sum(tx["type"] == FAKE_TX_TYPE_MINT for tx in core.api_transactions.values()) == 1

    # 4. Finish delayed Atoma settlement and all decision-produced exits.
    driver.advance(4 * 24 * 3600)
    driver.start()
    driver.settle_atoma()
    for _ in range(3):
        driver.advance()
        driver.start()
    assert all(t.is_success() for t in driver.read_state().portfolio.get_all_trades())
