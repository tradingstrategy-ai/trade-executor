"""Public account commands against fixed Arbitrum/Ethereum Lagoon deployments.

Arbitrum 520900000 contains the reviewed Atoma upgrade; Ethereum 25598869
is the established July 2026 fixture. Both clocks advance to a shared simulated
UTC before assertions. Only Circle attestations and Lighter's sequencer are
fake: custody, Guard, ERC-20, vault and CCTP transactions execute on Anvil.
"""

import json
import os
import shutil
from collections.abc import Iterator
from decimal import Decimal
from unittest import mock

import pytest
from eth_defi.compat import native_datetime_utc_fromtimestamp
from typer.main import get_command

from tradeexecutor.cli.main import app
from tradeexecutor.cli import account_check
from tradeexecutor.cli.commands import correct_accounts as correction_command
from tradeexecutor.state.trade import TradeStatus
from tradeexecutor.testing.multi_vault_start import MultiVaultStartDriver, deployed_multi_vault
from tradeexecutor.testing.multi_vault_simulation import installed_multi_vault_simulation, multi_vault_simulation_clock
from tradeexecutor.testing.lighter_core import FakeLighterCore, FakeLighterHolding


pytestmark = [
    pytest.mark.skipif(not all(os.environ.get(k) for k in ("JSON_RPC_ARBITRUM", "JSON_RPC_ETHEREUM", "TRADING_STRATEGY_API_KEY")), reason="Dual-fork archive RPCs and API credentials required"),
    pytest.mark.skipif(shutil.which("anvil") is None or shutil.which("jq") is None, reason="Anvil and jq required"),
    pytest.mark.timeout(1800),
]


@pytest.fixture()
def driver() -> Iterator[MultiVaultStartDriver]:
    """Deploy real custody contracts and initialise a valid bootstrap valuation."""
    with deployed_multi_vault() as run:
        run.command("lagoon-settle", "--force-valuation")
        yield run


@pytest.mark.skipif(os.environ.get("GITHUB_ACTIONS") == "true", reason="Takes approximately 4 minutes 59 seconds including fixture setup; skipped on CI for now until optimised")
def test_settled_accounts_and_atomic_correction(driver: MultiVaultStartDriver) -> None:
    """Check settled balances and preserve state until correction is verified.

    1. Inspect both chains without a signer and independently reconcile custody.
    2. Demonstrate a material deficit and fail final verification before saving.
    3. Apply the correction, reload and repeat to prove no duplicate balance event.
    4. Refuse investor flows and unsafe multichain command combinations.
    5. Check held venues, zero-price discrepancies and reconstruct a known holding.
    6. Check held assets and configured cash in a twenty-pair universe.
    """
    # 1. Inspect both chains without a signer and independently reconcile custody.
    before = driver.state_file.read_bytes()
    result = driver.command("check-accounts", environment_overrides={"PRIVATE_KEY": None, "LOG_LEVEL": "info"})
    assert "All accounts match" in result.stdout + result.stderr
    assert "Observed at" in result.stdout + result.stderr
    assert "255988" in result.stdout + result.stderr
    assert "520900" in result.stdout + result.stderr
    assert driver.fork_timestamps[1] < driver.fork_timestamps[42161]
    assert driver.state_file.read_bytes() == before
    assert driver.read_state().portfolio.get_net_asset_value() == pytest.approx(float(driver.observed_assets()))

    # 2. Demonstrate a material deficit and fail final verification before saving.
    state = driver.read_state()
    # The reserve-only case used to skip on-chain checks. Losing the bridge
    # ledger here leaves only a harmless 0.01 USDC satellite surplus, whereas
    # the material hub deficit must still be corrected. This is a state-only
    # interruption; both real fork balances stay unchanged.
    state.portfolio.open_positions.clear()
    state.portfolio.get_default_reserve_position().quantity += Decimal(1000)
    driver.state_file.write_text(state.to_json_safe())
    assert driver.command("check-accounts", check=False).returncode != 0
    driver.command("correct-accounts")
    assert driver.read_state().portfolio.get_default_reserve_position().quantity == Decimal("18.99")
    # Restore the complete original ledger together with its unchanged chain
    # and sequencer state. The separate exact-NAV scenario retains the known
    # 0.01 USDC bridge balance rather than asking dust discovery to recreate it.
    driver.state_file.write_bytes(before)
    state = driver.read_state()
    state.portfolio.get_default_reserve_position().quantity += Decimal(1000)
    driver.state_file.write_text(state.to_json_safe())
    uncorrected = driver.state_file.read_bytes()
    result = driver.command("check-accounts", check=False)
    assert result.returncode != 0
    assert "Accounts do not match" in result.stdout + result.stderr
    simulation = driver.simulation
    timestamp = native_datetime_utc_fromtimestamp(simulation.get_web3(42161).eth.get_block("latest")["timestamp"])
    # Only the final observation fails. Initial balances, custody and correction
    # events run normally; this reproduces losing the provider before commit.
    with mock.patch.dict(os.environ, driver.environment(), clear=True), installed_multi_vault_simulation(simulation), multi_vault_simulation_clock(timestamp):
        with mock.patch.object(correction_command, "check_accounts", side_effect=TimeoutError("final observation unavailable")):
            with pytest.raises(TimeoutError, match="final observation unavailable"):
                get_command(app).main(args=["correct-accounts"], standalone_mode=False)
    assert driver.state_file.read_bytes() == uncorrected

    # 3. Apply the correction, reload and repeat to prove no duplicate balance event.
    driver.command("correct-accounts")
    corrected = driver.read_state()
    events = len(corrected.sync.accounting.balance_update_refs)
    assert corrected.portfolio.get_net_asset_value() == pytest.approx(float(driver.observed_assets()))
    driver.command("correct-accounts")
    assert len(driver.read_state().sync.accounting.balance_update_refs) == events
    driver.command("check-accounts")

    # 4. Refuse investor flows and unsafe multichain command combinations.
    for flag in ("--transfer-away", "--process-redemption"):
        saved = driver.state_file.read_bytes()
        blocks = {chain: driver.simulation.get_web3(chain).eth.block_number for chain in (1, 42161)}
        result = driver.command("correct-accounts", flag, check=False)
        assert result.returncode != 0 and "refuses" in result.stdout + result.stderr
        assert driver.state_file.read_bytes() == saved
        assert {chain: driver.simulation.get_web3(chain).eth.block_number for chain in blocks} == blocks
    driver.command("request-deposit", "--amount", "1", helper=True)
    saved = driver.state_file.read_bytes()
    cursor = driver.read_state().sync.treasury.last_lagoon_settlement_block_scanned
    result = driver.command("correct-accounts", check=False)
    assert result.returncode != 0 and "investor flows" in result.stdout + result.stderr
    assert driver.state_file.read_bytes() == saved
    driver.command("lagoon-settle")
    # Reproduce settlement mined before persistence: the queue is now empty,
    # but its real event must still be booked by Lagoon's dedicated discovery
    # cursor rather than imported as an unknown reserve correction.
    driver.state_file.write_bytes(saved)
    result = driver.command("correct-accounts", check=False)
    assert result.returncode != 0 and "investor flows" in result.stdout + result.stderr
    assert driver.state_file.read_bytes() == saved
    driver.command("lagoon-settle")
    driver.command("claim-shares", helper=True)
    driver.command("correct-accounts")
    assert driver.read_state().sync.treasury.last_lagoon_settlement_block_scanned >= cursor

    # 5. Check held venues, zero-price discrepancies and reconstruct a known holding.
    driver.command("request-deposit", "--amount", "110", helper=True)
    driver.command("lagoon-settle")
    driver.command("claim-shares", helper=True)
    driver.start()
    driver.command("check-accounts", environment_overrides={"PRIVATE_KEY": None})
    held = driver.state_file.read_bytes()
    state = driver.read_state()
    atoma = next(p for p in state.portfolio.open_positions.values() if p.pair.get_vault_protocol() == "atoma")
    atoma.last_token_price = 0
    next(t for t in atoma.trades.values() if t.is_success()).executed_quantity += Decimal(10)
    driver.state_file.write_text(state.to_json_safe())
    assert driver.command("check-accounts", check=False).returncode != 0
    driver.command("correct-accounts")
    driver.command("check-accounts")
    driver.state_file.write_bytes(held)
    state = driver.read_state()
    upshift = next(p for p in state.portfolio.open_positions.values() if p.pair.get_vault_protocol() == "upshift")
    # An accidentally omitted position must be reconstructed using a real
    # share quote. Its underlying USDC is already allocated to the vault, so
    # rebuilding shares must not change the bridge's free cash projection.
    state.portfolio.open_positions.pop(upshift.position_id)
    allocated = next(p.bridge_capital_allocated for p in state.portfolio.open_positions.values() if p.pair.is_cctp_bridge())
    driver.state_file.write_text(state.to_json_safe())
    assert driver.command("check-accounts", check=False).returncode != 0
    driver.command("correct-accounts")
    driver.command("check-accounts")
    state = driver.read_state()
    bridge = next(p for p in state.portfolio.open_positions.values() if p.pair.is_cctp_bridge())
    assert bridge.bridge_capital_allocated == allocated
    assert state.portfolio.get_net_asset_value() == pytest.approx(float(driver.observed_assets()), abs=0.00001)

    # 6. Check held assets and configured cash in a twenty-pair universe.
    strategy = driver.directory / "strategy.py"
    original_strategy = strategy.read_text()
    # Use a static inspection fixture so changes to the base function's spelling
    # cannot silently break the twenty-pair coverage setup.
    shutil.copyfile(driver.repository / "strategies/test_only/multi_vault_account_checks.py", strategy)
    result = driver.command("check-accounts", environment_overrides={"LOG_LEVEL": "info"})
    assert "20 pairs" in result.stdout + result.stderr
    assert "arbitrary untracked tokens are not discovered" in result.stdout + result.stderr
    assert "Atoma-Index" in result.stdout + result.stderr
    state.portfolio.get_default_reserve_position().quantity -= Decimal(10)
    driver.state_file.write_text(state.to_json_safe())
    assert driver.command("check-accounts", check=False).returncode != 0
    driver.command("correct-accounts")
    driver.command("check-accounts")
    saved = driver.state_file.read_bytes()
    state = driver.read_state()
    bridge = next(p for p in state.portfolio.open_positions.values() if p.pair.is_cctp_bridge())
    bridge.bridge_capital_allocated = bridge.get_quantity()
    driver.state_file.write_text(state.to_json_safe())
    # Even zero free cash must retain its bridge custody row in the large
    # universe. The 0.01 actual residual is within the existing cash tolerance;
    # this asserts coverage, not a new dust policy or an economic write-off.
    result = driver.command("check-accounts", environment_overrides={"LOG_LEVEL": "info"})
    assert "USDC (CCTP bridged to Ethereum)" in result.stdout + result.stderr
    driver.state_file.write_bytes(saved)
    strategy.write_text(original_strategy)


def test_incomplete_account_observations(driver: MultiVaultStartDriver) -> None:
    """Refuse incomplete custody observations without altering financial state.

    1. Remove or misroute the satellite provider and conflict its Safe metadata.
    2. Refuse missing metadata, invalid snapshots and failed public pool reads.
    3. Check collateral rounding and refuse an unexplained pool cash movement.
    4. Refuse changed ownership, invalid cash and unexpected raw pool shares.
    """
    # 1. Remove or misroute the satellite provider and conflict its Safe metadata.
    saved = driver.state_file.read_bytes()
    blocks = {chain: driver.simulation.get_web3(chain).eth.block_number for chain in (1, 42161)}
    for overrides in ({"JSON_RPC_ETHEREUM": None}, {"JSON_RPC_ETHEREUM": driver.simulation.arbitrum_rpc}):
        for command in ("check-accounts", "correct-accounts"):
            result = driver.command(command, check=False, environment_overrides=overrides)
            assert result.returncode != 0
            assert "RPC" in result.stdout + result.stderr
            assert driver.state_file.read_bytes() == saved
    artefact = driver.state_file.with_suffix(".deployment.json")
    original = artefact.read_bytes()
    # Capture deployment-derived CLI settings before removing the artefact;
    # the subprocess driver also reads it to construct those settings.
    environment = driver.environment()
    artefact.unlink()
    with mock.patch.dict(os.environ, environment, clear=True):
        for command in ("check-accounts", "correct-accounts"):
            with pytest.raises(RuntimeError, match="deployment metadata"):
                get_command(app).main(args=[command], standalone_mode=False)
            assert driver.state_file.read_bytes() == saved
    artefact.write_bytes(original)
    for issue, message in (("missing_chain", "Incomplete multichain deployment metadata"), ("unknown_chain", "Unknown custody chain")):
        payload = json.loads(original)
        ethereum = payload["deployments"].pop("ethereum")
        if issue == "unknown_chain":
            payload["deployments"]["unknown_chain"] = ethereum
        artefact.write_text(json.dumps(payload))
        for command in ("check-accounts", "correct-accounts"):
            result = driver.command(command, check=False)
            assert result.returncode != 0 and message in result.stdout + result.stderr
            assert driver.state_file.read_bytes() == saved
        artefact.write_bytes(original)
    payload = json.loads(original)
    payload["deployments"]["ethereum"]["safe_address"] = "0x0000000000000000000000000000000000000001"
    artefact.write_text(json.dumps(payload))
    result = driver.command("check-accounts", check=False)
    assert result.returncode != 0 and "Safe" in result.stdout + result.stderr
    artefact.write_bytes(original)

    # 2. Refuse missing metadata, invalid snapshots and failed public pool reads.
    payload = json.loads(original)
    payload["deployments"]["ethereum"].pop("lighter_account_setup")
    artefact.write_text(json.dumps(payload))
    assert driver.command("check-accounts", check=False).returncode != 0
    artefact.write_bytes(original)
    state = driver.read_state()
    next(iter(state.portfolio.lighter_idle_collateral.values())).valid = False
    driver.state_file.write_text(state.to_json_safe())
    invalid = driver.state_file.read_bytes()
    for command in ("check-accounts", "correct-accounts"):
        assert driver.command(command, check=False).returncode != 0
        assert driver.state_file.read_bytes() == invalid
    driver.state_file.write_bytes(saved)
    result = driver.command("check-accounts", controls=("--fail-account-read",), check=False)
    assert result.returncode != 0 and "account read failure" in result.stdout + result.stderr
    assert driver.state_file.read_bytes() == saved

    for command in ("check-accounts", "correct-accounts"):
        result = driver.command(command, controls=("--fail-satellite-read",), check=False)
        assert result.returncode != 0 and "satellite balance read failure" in result.stdout + result.stderr
        assert driver.state_file.read_bytes() == saved

    # 3. Check collateral rounding and refuse an unexplained pool cash movement.
    core_path = driver.simulation.fake_lighter_file
    original_core = core_path.read_bytes()
    core = FakeLighterCore.load(driver.simulation.get_web3(1), core_path)
    core.accounts[core.master_index].collateral += Decimal("0.000001")
    core.save(core_path)
    driver.command("check-accounts")
    core.accounts[core.master_index].collateral += Decimal("0.000001")
    core.save(core_path)
    for command in ("check-accounts", "correct-accounts"):
        assert driver.command(command, check=False).returncode != 0
        assert driver.state_file.read_bytes() == saved
    core_path.write_bytes(original_core)

    # 4. Refuse changed ownership, invalid cash and unexpected raw pool shares.
    # Only sequencer responses are altered; all EVM balances and executor
    # state stay at the same settled snapshot throughout these observations.
    for issue in ("owner", "negative_cash", "unexpected_shares"):
        core = FakeLighterCore.load(driver.simulation.get_web3(1), core_path)
        if issue == "owner":
            core.owner = "0x0000000000000000000000000000000000000001"
        elif issue == "negative_cash":
            core.accounts[core.master_index].collateral = Decimal(-1)
        else:
            core.accounts[core.master_index].pool_shares[next(iter(core.pools))] = FakeLighterHolding(shares=1, principal=Decimal("0.001"))
        core.save(core_path)
        for command in ("check-accounts", "correct-accounts"):
            assert driver.command(command, check=False).returncode != 0
            assert driver.state_file.read_bytes() == saved
        core_path.write_bytes(original_core)
    assert {chain: driver.simulation.get_web3(chain).eth.block_number for chain in blocks} == blocks


@pytest.mark.skipif(os.environ.get("GITHUB_ACTIONS") == "true", reason="Takes approximately 4 minutes 59 seconds including fixture setup; skipped on CI for now until optimised")
def test_pending_bridge_account_commands(driver: MultiVaultStartDriver) -> None:
    """Keep actual CCTP entitlement through delayed receive and process restart.

    1. Subscribe and burn real hub USDC while the simulated attester is delayed.
    2. Inspect pending capital and refuse generic corrections without state writes.
    3. Mine the real receive but interrupt booking, then restart without a second mint.
    4. Inspect pending pool and queued-vault operations without blocking recovery.
    """
    # 1. Subscribe and burn real hub USDC while the simulated attester is delayed.
    driver.command("request-deposit", "--amount", "111", helper=True)
    driver.command("lagoon-settle")
    driver.command("claim-shares", helper=True)
    driver.start(controls=("--delay-cctp-attestation",), check=False)
    state = driver.read_state()
    bridges = [t for t in state.portfolio.get_all_trades() if t.get_status() == TradeStatus.cctp_in_transit]
    assert len(bridges) == 1
    bridge = bridges[0]
    burn_hash = bridge.other_data["cctp_burn_tx_hash"]
    assert driver.simulation.get_web3(42161).eth.get_transaction_receipt(burn_hash)["status"] == 1
    assert state.portfolio.get_in_transit_value() == pytest.approx(float(bridge.planned_reserve))

    # 2. Inspect pending capital and refuse generic corrections without state writes.
    saved = driver.state_file.read_bytes()
    for command in ("check-accounts", "correct-accounts"):
        result = driver.command(command, check=False)
        assert result.returncode != 0 and "Pending" in result.stdout + result.stderr
        if command == "check-accounts":
            assert "Unchecked pending asset" in result.stdout + result.stderr
            assert "Atoma-Index" in result.stdout + result.stderr
        assert driver.state_file.read_bytes() == saved
    assert driver.start(controls=("--delay-cctp-attestation",), check=False).returncode != 0

    # 3. Mine the real receive but interrupt booking, then restart without a second mint.
    interrupted = driver.start(controls=("--interrupt-cctp-booking",), check=False)
    pending = driver.read_state().portfolio.get_trade_by_id(bridge.trade_id)
    assert pending.get_status() == TradeStatus.cctp_in_transit
    assert len(pending.blockchain_transactions) > len(bridge.blockchain_transactions), interrupted.stdout + interrupted.stderr
    receive_hash = pending.blockchain_transactions[-1].tx_hash
    assert driver.simulation.get_web3(1).eth.get_transaction_receipt(receive_hash)["status"] == 1
    driver.start()
    settled = driver.read_state().portfolio.get_trade_by_id(bridge.trade_id)
    assert settled.is_success()
    assert settled.blockchain_transactions[-1].tx_hash == receive_hash
    assert settled.other_data["cctp_burn_tx_hash"] == burn_hash
    assert driver.read_state().portfolio.get_in_transit_value() == 0
    assert driver.read_state().portfolio.get_net_asset_value() == pytest.approx(float(driver.observed_assets()), abs=0.00001)

    # 4. Inspect pending pool and queued-vault operations without blocking recovery.
    driver.advance(4 * 24 * 3600)
    driver.start(controls=("--delay-pool-confirmation",), check=False)
    pending_state = driver.read_state()
    pending_trades = account_check.pending_account_trades(pending_state)
    assert pending_trades
    saved = driver.state_file.read_bytes()
    for command in ("check-accounts", "correct-accounts"):
        assert driver.command(command, check=False).returncode != 0
        assert driver.state_file.read_bytes() == saved
    driver.advance(8 * 24 * 3600)
    driver.settle_atoma()
    driver.start()
    driver.start()
    driver.command("check-accounts")
