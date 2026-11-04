"""Simulated Arbitrum hub with an Ethereum satellite: two Anvil forks, a fake Lighter sequencer and forged CCTP.

``scripts/lagoon/manual-multi-vault-cli.sh`` runs each ``trade-executor``
command as its own process. In simulated mode those processes share:

- two Anvil forks, Arbitrum at the reviewed Atoma block and Ethereum at the
  shared midnight block, owned by a detached host process that outlives every
  command (:py:func:`run_simulation_host`);
- one :py:class:`~tradeexecutor.testing.lighter_core.FakeLighterCore`, saved to
  ``lighter-core.json`` after each command and reloaded before the next;
- a test CCTP attester on each fork, so a real burn's ``MessageSent`` message
  is relayed by a real ``receiveMessage`` with a forged attestation.

All simulation files live in a named directory under ``/tmp``, never in the
executor run directory, so the run directory holds only what a live run holds.
Every Safe, Guard, Lagoon, Atoma, Upshift, CCTP and ZkLighter transaction is
real on the forks. Only Lighter's sequencer, Circle's attestation service and
the clock (Atoma's lock) are simulated.

Use :py:func:`installed_multi_vault_simulation` around an in-process
command, as :py:mod:`tradeexecutor.testing.multi_vault_manual` does.
"""

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from unittest import mock

from eth_account import Account
from hexbytes import HexBytes
from eth_defi.cctp.attestation import CCTPAttestation
from eth_defi.cctp.constants import CHAIN_ID_TO_CCTP_DOMAIN
from eth_defi.cctp.testing import forge_attestation, replace_attester_on_fork
from eth_defi.cctp.transfer import get_message_transmitter_v2
from eth_defi.provider.anvil import fork_network_anvil, fund_erc20_on_anvil
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.testing.fork_blocks import ETHEREUM_MIDNIGHT_BLOCK
from eth_defi.token import USDC_NATIVE_TOKEN, fetch_erc20_details
from web3 import Web3
from web3.logs import DISCARD

from tradeexecutor.ethereum import ethereum_protocol_adapters
from tradeexecutor.testing.lighter_core import FakeLighterCore, FakeLighterPool
from tradeexecutor.testing.lighter_simulation import ANVIL_DEPLOYER_PRIVATE_KEY, SIMULATED_MASTER_INDEX


logger = logging.getLogger(__name__)

#: Arbitrum fork block after Atoma's 2026-10-29 fee-split upgrade, so the
#: simulation exercises the implementation live vaults use. Shared with
#: eth-defi's Atoma tests, whose warmed RPC cache covers it.
ARBITRUM_ATOMA_FORK_BLOCK = 520_900_000

#: Simulation description file inside the simulation directory.
SIMULATION_FILE = "simulation.json"

#: Fake Lighter sequencer state inside the simulation directory.
FAKE_LIGHTER_FILE = "lighter-core.json"

#: Fork URLs published by the simulation host.
FORKS_FILE = "forks.json"

#: Simulation host and Anvil log.
HOST_LOG_FILE = "simulation-host.log"

#: The description file of a stopped simulation, kept for inspection.
STOPPED_SIMULATION_FILE = "simulation.stopped.json"

#: pmalt pool numbers observed on 3 November 2026, used as the fake starting state.
PMALT_POOL_INDEX = 281474976552918

#: CCTP header offsets of the attester-filled fields, see ``craft_cctp_message()``.
_NONCE_SLICE = slice(12, 44)
_FINALITY_EXECUTED_SLICE = slice(144, 148)
_DESTINATION_DOMAIN_SLICE = slice(8, 12)
#: Finalised transfer threshold, as Circle reports for a standard transfer.
_FINALITY_FINALISED = 2000


@dataclass(slots=True)
class MultiVaultSimulation:
    """A running simulated environment, described by its directory."""

    #: Directory holding the description, fake state and fork logs.
    directory: Path
    #: Local Arbitrum fork RPC.
    arbitrum_rpc: str
    #: Local Ethereum fork RPC.
    ethereum_rpc: str
    #: Test attester private keys by chain id; fork-only test keys.
    attesters: dict[int, str]
    #: Process id of the detached simulation host owning both forks.
    pids: dict[str, int]

    @property
    def fake_lighter_file(self) -> Path:
        """Return the fake sequencer state file."""
        return self.directory / FAKE_LIGHTER_FILE

    def get_web3(self, chain_id: int) -> Web3:
        """Connect to one fork by chain id."""
        url = {42161: self.arbitrum_rpc, 1: self.ethereum_rpc}[chain_id]
        return create_multi_provider_web3(url, default_http_timeout=(3.0, 250.0))

    def save(self) -> None:
        """Write the description file."""
        data = {"arbitrum_rpc": self.arbitrum_rpc, "ethereum_rpc": self.ethereum_rpc, "attesters": {str(k): v for k, v in self.attesters.items()}, "pids": self.pids}
        path = self.directory / SIMULATION_FILE
        path.write_text(json.dumps(data, indent=2))
        os.chmod(path, 0o600)

    @classmethod
    def load(cls, directory: Path) -> "MultiVaultSimulation":
        """Read the description file of a running simulation."""
        data = json.loads((directory / SIMULATION_FILE).read_text())
        return cls(directory, data["arbitrum_rpc"], data["ethereum_rpc"], {int(k): v for k, v in data["attesters"].items()}, data["pids"])


def run_simulation_host(directory: Path) -> None:
    """Own both forks until ``SIGTERM``, then close them.

    Runs as a detached process started by :py:func:`start_multi_vault_simulation`,
    so the forks outlive each command process. eth-defi's
    :py:func:`~eth_defi.provider.anvil.fork_network_anvil` fronts the upstream
    RPCs with a local proxy, so their credentials never appear on an Anvil
    command line.

    :param directory:
        Simulation directory; the fork URLs are written to ``forks.json``.
    """
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_args: stop.set())
    launches = {}
    try:
        launches["arbitrum"] = fork_network_anvil(os.environ["JSON_RPC_ARBITRUM"], fork_block_number=ARBITRUM_ATOMA_FORK_BLOCK)
        launches["ethereum"] = fork_network_anvil(os.environ["JSON_RPC_ETHEREUM"], fork_block_number=ETHEREUM_MIDNIGHT_BLOCK)
        (directory / FORKS_FILE).write_text(json.dumps({slug: launch.json_rpc_url for slug, launch in launches.items()}))
        logger.info("Simulation forks ready: %s", {slug: launch.json_rpc_url for slug, launch in launches.items()})
        stop.wait()
    finally:
        for launch in launches.values():
            launch.close()
        logger.info("Simulation forks closed")


def _wait_for_forks(directory: Path, host: subprocess.Popen, timeout: float = 300) -> dict[str, str]:
    """Wait for the host to publish its fork URLs.

    :return:
        Fork RPC URLs by chain slug.
    """
    deadline = time.monotonic() + timeout
    forks_file = directory / FORKS_FILE
    while not forks_file.exists():
        if host.poll() is not None or time.monotonic() > deadline:
            raise RuntimeError(f"Simulation forks did not start; see {directory / HOST_LOG_FILE}")
        time.sleep(1)
    return json.loads(forks_file.read_text())


def start_multi_vault_simulation(directory: Path, *, deployer_usdc: Decimal) -> MultiVaultSimulation:
    """Fork both chains, fund the test deployer, replace CCTP attesters and seed the fake sequencer.

    :param directory:
        New simulation directory under ``/tmp``.
    :param deployer_usdc:
        Native Arbitrum USDC for the deployer.
    :return:
        Running simulation; stop it with :py:func:`stop_multi_vault_simulation`.
    """
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    assert os.environ.get("JSON_RPC_ARBITRUM") and os.environ.get("JSON_RPC_ETHEREUM"), "JSON_RPC_ARBITRUM and JSON_RPC_ETHEREUM are required to fork both chains"
    log = open(directory / HOST_LOG_FILE, "w")  # noqa: SIM115 - the detached host owns it
    host = subprocess.Popen([sys.executable, "-m", "tradeexecutor.testing.multi_vault_simulation", str(directory)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    forks = _wait_for_forks(directory, host)
    simulation = MultiVaultSimulation(directory, forks["arbitrum"], forks["ethereum"], {}, {"host": host.pid})
    deployer = Account.from_key(ANVIL_DEPLOYER_PRIVATE_KEY)
    for chain_id in (42161, 1):
        web3 = simulation.get_web3(chain_id)
        web3.provider.make_request("anvil_setBalance", [deployer.address, hex(10 * 10**18)])
        simulation.attesters[chain_id] = Web3.to_hex(replace_attester_on_fork(web3).key)
    arbitrum = simulation.get_web3(42161)
    usdc = fetch_erc20_details(arbitrum, USDC_NATIVE_TOKEN[42161])
    fund_erc20_on_anvil(arbitrum, usdc.address, deployer.address, usdc.convert_to_raw(deployer_usdc))
    simulation.save()

    core = FakeLighterCore(simulation.get_web3(1), SIMULATED_MASTER_INDEX)
    core.add_pool(FakeLighterPool(
        index=PMALT_POOL_INDEX,
        equity=Decimal("35301.906501"),
        total_shares=47_656_547,
        operator_shares=42_136_849,
    ))
    core.save(simulation.fake_lighter_file)
    return simulation


def stop_multi_vault_simulation(directory: Path) -> None:
    """Stop the forks of a simulation and mark it stopped.

    The description file is removed, so a later resume is refused at once;
    the logs and fake sequencer state stay for inspection.

    :param directory:
        Simulation directory.
    """
    simulation = MultiVaultSimulation.load(directory)
    host = simulation.pids["host"]
    try:
        os.killpg(host, signal.SIGTERM)
    except ProcessLookupError:
        logger.info("Simulation host %d was not running", host)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            os.kill(host, 0)
        except ProcessLookupError:
            break
        time.sleep(1)
    (directory / SIMULATION_FILE).rename(directory / STOPPED_SIMULATION_FILE)
    logger.info("Simulation in %s stopped", directory)


def forge_cctp_attestation(simulation: MultiVaultSimulation, source_domain: int, transaction_hash: str, **kwargs) -> CCTPAttestation:
    """Stand-in for Circle's ``fetch_attestation()`` on the simulated forks.

    Reads the burn's real ``MessageSent`` message, fills the fields Circle's
    attester fills (a unique nonce from the burn hash and the finalised
    threshold), and signs it with the destination fork's test attester. The
    burn amount is minted unchanged, so the simulation charges no CCTP fee.

    :param simulation:
        Running simulation.
    :param source_domain:
        CCTP domain of the burn chain.
    :param transaction_hash:
        Burn transaction hash.
    :param kwargs:
        Unused polling options of the replaced function.
    :return:
        Attestation accepted by the destination ``MessageTransmitterV2``.
    """
    chain_by_domain = {domain: chain_id for chain_id, domain in CHAIN_ID_TO_CCTP_DOMAIN.items() if chain_id in simulation.attesters}
    web3 = simulation.get_web3(chain_by_domain[source_domain])
    receipt = web3.eth.get_transaction_receipt(transaction_hash)
    events = get_message_transmitter_v2(web3).events.MessageSent().process_receipt(receipt, errors=DISCARD)
    assert len(events) == 1, f"Expected one CCTP message in {transaction_hash}, got {len(events)}"
    message = bytearray(events[0]["args"]["message"])
    message[_NONCE_SLICE] = Web3.keccak(HexBytes(transaction_hash))
    message[_FINALITY_EXECUTED_SLICE] = _FINALITY_FINALISED.to_bytes(4, "big")
    destination_chain_id = chain_by_domain[int.from_bytes(message[_DESTINATION_DOMAIN_SLICE], "big")]
    attester = Account.from_key(simulation.attesters[destination_chain_id])
    logger.info("Forged CCTP attestation for burn %s to chain %d", transaction_hash, destination_chain_id)
    return CCTPAttestation(message=bytes(message), attestation=forge_attestation(bytes(message), attester), status="complete")


@contextmanager
def installed_multi_vault_simulation(simulation: MultiVaultSimulation) -> Iterator[FakeLighterCore]:
    """Install the fake sequencer and forged CCTP for one in-process command.

    The fake sequencer state is saved back afterwards, also after a failure,
    so the next command process continues from the same simulated venue.

    :param simulation:
        Running simulation.
    :return:
        The loaded fake core.
    """
    core = FakeLighterCore.load(simulation.get_web3(1), simulation.fake_lighter_file)

    def forge(source_domain: int, transaction_hash: str, **kwargs) -> CCTPAttestation:
        return forge_cctp_attestation(simulation, source_domain, transaction_hash, **kwargs)

    original_create_cctp_bridge_adapter = ethereum_protocol_adapters.create_cctp_bridge_adapter

    def create_cctp_bridge_adapter(*args, **kwargs):
        # Relay real messages with forged attestations instead of crediting
        # the destination directly, which Anvil detection would otherwise select.
        kwargs["skip_attestation"] = False
        return original_create_cctp_bridge_adapter(*args, **kwargs)

    with ExitStack() as stack:
        for target, replacement in core.get_patches() + core.get_pool_patches():
            stack.enter_context(mock.patch(target, new=replacement))
        for target in (
            "eth_defi.cctp.attestation.fetch_attestation",
            "eth_defi.erc_4626.vault_protocol.lagoon.lighter.fetch_attestation",
            "tradeexecutor.cli.commands.reclaim_satellites.fetch_attestation",
        ):
            stack.enter_context(mock.patch(target, new=forge))
        stack.enter_context(mock.patch.object(ethereum_protocol_adapters, "create_cctp_bridge_adapter", new=create_cctp_bridge_adapter))
        try:
            yield core
        finally:
            core.save(simulation.fake_lighter_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run_simulation_host(Path(sys.argv[1]))
