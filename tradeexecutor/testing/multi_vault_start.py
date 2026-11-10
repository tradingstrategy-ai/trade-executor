"""Process-level driver for the mixed-vault strategy on persistent forks."""

import datetime
import calendar
import copy
import json
import os
import subprocess
import sys
import tempfile
import traceback
from contextlib import contextmanager
from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from unittest import mock

from typer.testing import CliRunner

from eth_defi.provider.anvil import mine
from eth_defi.hotwallet import HotWallet
from eth_defi.erc_4626.vault_protocol.lagoon.vault import LagoonVault
from eth_defi.compat import native_datetime_utc_fromtimestamp, native_datetime_utc_now
from eth_defi.token import USDC_NATIVE_TOKEN, fetch_erc20_details
from eth_defi.vault.base import VaultSpec
from eth_defi.erc_4626.vault_protocol.atoma.vault import AtomaVault
from eth_defi.erc_4626.classification import create_vault_instance_autodetect

from tradeexecutor.state.state import State
from tradeexecutor.testing.multi_vault_simulation import MultiVaultSimulation, stop_multi_vault_simulation, ARBITRUM_ATOMA_FORK_BLOCK, ETHEREUM_MULTI_VAULT_FORK_BLOCK
from tradeexecutor.state.identifier import TradingPairKind
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, create_pair_universe_from_code
from tradeexecutor.testing.lighter_core import FakeLighterCore
from tradeexecutor.testing.lighter_manual import broadcast_and_wait
from tradeexecutor.testing.multi_vault_manual import app as manual_app


@contextmanager
def deployed_multi_vault() -> Iterator["MultiVaultStartDriver"]:
    """Reuse the public deployment runbook on the established fixed dual forks.

    Each caller owns its forks, state and fake venue together. No test restores
    an EVM snapshot while accidentally retaining a later executor or API state.
    """
    repository = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="multi-vault-start-") as directory:
        run = MultiVaultStartDriver(repository, Path(directory) / "run", Path(directory) / "simulation")
        environment = {k: v for k, v in os.environ.items() if k not in {"HYPERSYNC_API_KEY", "TELEGRAM_API_KEY", "TELEGRAM_CHAT_ID", "DERIVE_OWNER_PRIVATE_KEY", "DERIVE_SESSION_PRIVATE_KEY"}}
        environment.update({"SIMULATE": "true", "MULTI_VAULT_RUN_DIR": str(run.directory), "MULTI_VAULT_SIM_DIR": str(run.simulation_directory), "MULTI_VAULT_PYTHON": sys.executable, "LOG_LEVEL": "warning"})
        try:
            with (Path(directory) / "deployment.log").open("w+") as output:
                result = subprocess.run(["bash", "scripts/lagoon/manual-multi-vault-cli.sh", "--deploy", "--targets", ""], cwd=repository, env=environment, stdout=output, stderr=subprocess.STDOUT, text=True, timeout=1200)
                output.seek(0)
                assert result.returncode == 0, output.read()[-16000:]
            # Source state is historical; the existing harness advances both
            # clocks together so claims and freshness use one simulated UTC.
            run.fork_timestamps = {
                chain: run.simulation.get_web3(chain).eth.get_block(block)["timestamp"]
                for chain, block in ((42161, ARBITRUM_ATOMA_FORK_BLOCK), (1, ETHEREUM_MULTI_VAULT_FORK_BLOCK))
            }
            tip = max(run.simulation.get_web3(chain).eth.get_block("latest")["timestamp"] for chain in (1, 42161))
            run.advance(max(1, calendar.timegm(native_datetime_utc_now().timetuple()) - tip + 1))
            yield run
        finally:
            if (run.simulation_directory / "simulation.json").exists():
                stop_multi_vault_simulation(run.simulation_directory)


@dataclass
class MultiVaultStartDriver:
    """One writer operating the same state and simulated venue across CLI processes."""

    repository: Path
    directory: Path
    simulation_directory: Path

    #: Original historical times before the existing driver synchronises clocks.
    fork_timestamps: dict[int, int] | None = None

    @property
    def state_file(self) -> Path:
        """The normal executor state, shared across commands."""
        return self.directory / "state/multi-vault-cli.json"

    @property
    def simulation(self) -> MultiVaultSimulation:
        """Load existing fork descriptions, never launch a second venue."""
        return MultiVaultSimulation.load(self.simulation_directory)

    def read_state(self) -> State:
        """Reload state exactly as the next executor process does."""
        return State.read_json_file(self.state_file)

    def observed_assets(self) -> Decimal:
        """Independently total settled custody balances and actual venue holdings.

        Call only at completed deposit/exit boundaries. Queued claims and CCTP
        transit need their own entitlement assertions rather than this total.
        """
        simulation = self.simulation
        state = self.read_state()
        assert not any(t.is_vault_settlement_in_flight() for t in state.portfolio.get_all_trades())
        core = FakeLighterCore.load(simulation.get_web3(1), simulation.fake_lighter_file)
        safe = core.owner
        total = Decimal(0)
        for chain in (1, 42161):
            token = fetch_erc20_details(simulation.get_web3(chain), USDC_NATIVE_TOKEN[chain])
            total += token.fetch_balance_of(safe)
        account = core.accounts[core.master_index]
        total += account.collateral
        for index, holding in account.pool_shares.items():
            pool = core.pools[index]
            gross = Decimal(holding.shares) * pool.get_share_value()
            total += gross - max(gross - holding.principal, Decimal(0)) * pool.operator_fee / 100
        for chain, address in (
            (42161, "0xcc56410e1a136af0eceb7241c6ae394f4d8b581c"),
            (1, "0xE9B725010A9E419412ed67d0fA5f3A5f40159D32"),
        ):
            vault = create_vault_instance_autodetect(simulation.get_web3(chain), address)
            total += vault.share_token.fetch_balance_of(safe) * vault.fetch_share_price("latest")
        return total

    def environment(self) -> dict[str, str]:
        """Supply explicit fork settings and public test keys to child commands."""
        environment = {k: v for k, v in os.environ.items() if not k.startswith("JSON_RPC_") and k not in {"HYPERSYNC_API_KEY", "TELEGRAM_API_KEY", "TELEGRAM_CHAT_ID", "DERIVE_OWNER_PRIVATE_KEY", "DERIVE_SESSION_PRIVATE_KEY"}}
        simulation = self.simulation
        artifact = json.loads(self.state_file.with_suffix(".deployment.json").read_text())
        deployment = artifact["deployments"]["arbitrum"]
        environment.update({
            "JSON_RPC_ARBITRUM": simulation.arbitrum_rpc,
            "JSON_RPC_ETHEREUM": simulation.ethereum_rpc,
            "MULTI_VAULT_SIM_DIR": str(self.simulation_directory),
            "MULTI_VAULT_SIMULATE": "true",
            "PRIVATE_KEY": "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
            "ID": "multi-vault-cli",
            "EXECUTOR_ID": "multi-vault-cli",
            "STATE_FILE": str(self.state_file),
            "STRATEGY_FILE": str(self.directory / "strategy.py"),
            "CACHE_PATH": str(self.directory / "cache"),
            "LIGHTER_OPERATOR_RECORD_FILE": str(self.directory / "vault-record.json"),
            "VAULT_ADDRESS": deployment["vault_address"],
            "VAULT_ADAPTER_ADDRESS": deployment["module_address"],
            "ASSET_MANAGEMENT_MODE": "lagoon",
            "UNIT_TESTING": "true",
            "SIMULATE": "false",
            "CONFIRMATION_BLOCK_COUNT": "1",
            "LOG_LEVEL": "warning",
        })
        return environment

    def command(self, *arguments: str, timestamp: datetime.datetime | None = None, controls: tuple[str, ...] = (), helper: bool = False, check: bool = True, environment_overrides: dict[str, str | None] | None = None, in_process: bool = False) -> subprocess.CompletedProcess:
        """Invoke the registered CLI, retaining subprocesses for restart coverage.

        Validation-only tests can opt into in-process execution to avoid
        repeated interpreter startup. Both modes reload the normal state and
        simulated venue through the same public entry point.
        """
        args = [sys.executable, "-m", "tradeexecutor.testing.multi_vault_manual"]
        if not helper:
            args += ["cli", *controls]
            if timestamp is None:
                tip = max(self.simulation.get_web3(chain).eth.get_block("latest")["timestamp"] for chain in (1, 42161))
                timestamp = native_datetime_utc_fromtimestamp(tip)
            args += ["--simulation-time", timestamp.isoformat()]
            args += ["--"]
        environment = self.environment()
        for key, value in (environment_overrides or {}).items():
            if value is None:
                environment.pop(key, None)
            else:
                environment[key] = value
        if in_process:
            with mock.patch.dict(os.environ, environment, clear=True):
                invocation = CliRunner().invoke(manual_app, args[3:] + list(arguments))
            # CliRunner combines terminal output on the project's Click version.
            stderr = ""
            if invocation.exception and not isinstance(invocation.exception, SystemExit):
                stderr += "".join(traceback.format_exception(invocation.exception))
            result = subprocess.CompletedProcess(args + list(arguments), invocation.exit_code, invocation.output, stderr)
        else:
            result = subprocess.run(args + list(arguments), env=environment, cwd=self.repository, capture_output=True, text=True, timeout=900)
        if check and result.returncode:
            raise AssertionError(result.stdout[-8000:] + result.stderr[-8000:])
        return result


    def start(self, *, controls: tuple[str, ...] = (), check: bool = True) -> subprocess.CompletedProcess:
        """Run one decision cycle using the current fork time."""
        tip = max(self.simulation.get_web3(chain).eth.get_block("latest")["timestamp"] for chain in (1, 42161))
        timestamp = native_datetime_utc_fromtimestamp(tip)
        return self.command("start", "--run-single-cycle", timestamp=timestamp, controls=controls, check=check)

    def advance(self, seconds: int = 600) -> None:
        """Advance both fork clocks together, retaining real monotonic timeouts."""
        simulation = self.simulation
        tip = max(simulation.get_web3(chain).eth.get_block("latest")["timestamp"] for chain in (1, 42161))
        for chain in (1, 42161):
            mine(simulation.get_web3(chain), timestamp=tip + seconds)

    def settle_atoma(self) -> None:
        """Act as Atoma's operator on the fork for the saved redemption ticket."""
        web3 = self.simulation.get_web3(42161)
        vault = AtomaVault(web3, VaultSpec(42161, "0xcc56410e1a136af0eceb7241c6ae394f4d8b581c"))
        manager = vault.get_deposit_manager()
        for position in self.read_state().portfolio.open_positions.values():
            if position.pair.get_vault_protocol() == "atoma":
                for trade in position.trades.values():
                    if trade.is_sell() and trade.is_vault_settlement_in_flight():
                        manager.force_settle(manager.reconstruct_redemption_ticket(trade.other_data))

    def request_redemption(self) -> None:
        """Queue all simulated investor shares without settling or claiming them."""
        environment = self.environment()
        web3 = self.simulation.get_web3(42161)
        wallet = HotWallet.from_private_key(environment["PRIVATE_KEY"])
        vault = LagoonVault(web3, VaultSpec(42161, environment["VAULT_ADDRESS"]))
        shares = vault.share_token.fetch_raw_balance_of(wallet.address)
        broadcast_and_wait(web3, wallet, vault.request_redeem(wallet.address, shares), "queue simulated investor redemption")


def expand_account_check_universe(universe: TradingStrategyUniverse) -> TradingStrategyUniverse:
    """Exercise the discovery limit with twenty pairs and real tracked custody.

    Extra untracked spot tokens deliberately have no contracts. A checker that
    wrongly enumerates them will fail its RPC reads, while held tokens and
    configured cash must still be read from the actual fork deployments.
    This helper is only for account inspection, never strategy execution.
    """
    pairs = list(universe.iterate_pairs())
    template = next(pair for pair in pairs if pair.is_vault() and not pair.is_lighter_pool())
    for index in range(len(pairs) + 1, 21):
        pair = copy.deepcopy(template)
        pair.internal_id = index
        pair.kind = TradingPairKind.spot_market_hold
        pair.pool_address = pair.base.address = f"0x{index:040x}"
        pair.base.internal_id = index
        pair.base.token_symbol = f"UNTRACKED{index}"
        pair.other_data = {}
        pairs.append(pair)
    universe.data_universe.pairs = create_pair_universe_from_code(universe.primary_chain, pairs)
    return universe
