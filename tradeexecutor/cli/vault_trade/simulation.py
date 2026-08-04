"""Disposable Anvil infrastructure for ``vault-test-trade --auto-simulated``."""

import calendar
import datetime
import json
import logging
import tempfile
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from eth_defi.compat import native_datetime_utc_now
from eth_defi.middleware import ProbablyNodeHasNoBlock
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.provider.anvil import (
    ArchiveNodeRequired,
    RPCRequestError,
    make_anvil_custom_rpc_request,
)
from eth_defi.testing.rpc_cache import (
    seed_default_foundry_rpc_cache,
    seed_foundry_rpc_cache,
)
from eth_defi.testing.fork_blocks import MIDNIGHT_BLOCKS
from eth_defi.vault.base import VaultSpec
from requests.exceptions import RequestException
from tradingstrategy.chain import ChainId
from web3.exceptions import (
    BadFunctionCallOutput,
    CannotHandleRequest,
    MultipleFailedRequests,
    ProviderConnectionError,
    RequestTimedOut,
)

from tradeexecutor.cli.bootstrap import (
    create_execution_and_sync_model,
    create_web3_config,
)
from tradeexecutor.cli.vault_trade.core import deploy_simulated_lagoon_multichain
from tradeexecutor.ethereum.web3config import get_chain_slug
from tradeexecutor.ethereum.token import translate_token_details
from tradeexecutor.strategy.execution_model import AssetManagementMode

logger = logging.getLogger(__name__)


SIMULATED_VAULT_ATTEMPT_TIMEOUT = 60
SIMULATED_VAULT_INFRASTRUCTURE_RESTARTS = 1


#: Project cache seeds supplement eth-defi's generic test seeds.
PROJECT_RPC_CACHE_SEED_DIR = Path(__file__).parents[3] / "tests" / "rpc_cache_seed"

#: Persistent Foundry fork cache for production vault-test-trade experiments.
VAULT_TEST_RPC_CACHE_DIR = Path("~/.tradingstrategy/vaults/rpc-cache").expanduser()

#: Anvil 1.7.x stores fork ``storage.json`` here regardless of
#: ``FOUNDRY_RPC_CACHE_DIR``. Keep this explicit until Foundry exposes a
#: working fork-RPC-cache override.
DEFAULT_FOUNDRY_RPC_CACHE_DIR = Path.home() / ".foundry" / "cache" / "rpc"

#: These providers cannot supply the historic state required by the simulated
#: Lagoon balance path. Capture their live tip once per command and retain that
#: value only for whole-generation replacements.
LIVE_TIP_VAULT_TEST_CHAINS = frozenset({ChainId.base, ChainId.monad})


SIMULATED_INFRASTRUCTURE_EXCEPTIONS = (
    ArchiveNodeRequired,
    BrokenPipeError,
    CannotHandleRequest,
    ConnectionRefusedError,
    ConnectionResetError,
    MultipleFailedRequests,
    ProbablyNodeHasNoBlock,
    ProviderConnectionError,
    RequestException,
    RequestTimedOut,
    RPCRequestError,
    TimeoutError,
)


class SimulatedVaultAttemptTimeout(BaseException):
    """A fork-only vault attempt exceeded its diagnostic time budget.

    This intentionally bypasses broad ``except Exception`` blocks in third-party
    vault adapters. The command catches it at the outer per-vault boundary.
    """


@dataclass(slots=True)
class SimulatedVaultRuntime:
    """One disposable generation of the multichain Anvil simulation.

    All fields belong to the same fork generation and must be replaced
    together.  ``temporary_deployment_dir`` owns the generated deployment
    artefact consumed by normal Lagoon execution-model bootstrap.
    """

    generation: int
    web3config: Any
    deployment: Any
    deployment_file: Path
    execution_model: Any
    sync_model: Any
    reserve_asset: Any
    temporary_deployment_dir: tempfile.TemporaryDirectory
    #: Immutable upstream heights immediately after Anvil forks start, before
    #: the test Lagoon topology deploys any contracts.
    fork_blocks: dict[str, int] = field(default_factory=dict)

    #: Fork blocks requested for this generation. The first generation captures
    #: them, and each replacement uses exactly the same heights so Foundry can
    #: reuse its disk cache and diagnostics remain reproducible.
    pinned_fork_blocks: dict[ChainId, int] = field(default_factory=dict)

    def close(self) -> None:
        """Hard-stop all forks and remove this generation's artefact."""

        try:
            try:
                self.web3config.close(log_level=logging.ERROR, block_timeout=5)
            except Exception:
                logger.exception(
                    "One or more Anvil processes did not close cleanly for simulation generation %d",
                    self.generation,
                )
        finally:
            try:
                sync_foundry_rpc_cache_to_vault_cache()
            except Exception:
                logger.exception(
                    "Could not copy Foundry RPC cache after simulation generation %d",
                    self.generation,
                )
            finally:
                self.temporary_deployment_dir.cleanup()


def is_simulated_infrastructure_failure(error: BaseException) -> bool:
    """Check if a failed simulated attempt needs a fresh Anvil generation.

    Only transport/process failures qualify.  Reverts and adapter errors are
    vault results and must be persisted without an automatic retry.
    """

    if isinstance(error, SimulatedVaultAttemptTimeout):
        return True
    if isinstance(error, SIMULATED_INFRASTRUCTURE_EXCEPTIONS):
        return True

    # eth-defi detects an empty Anvil ``eth_call`` as
    # ``ProbablyNodeHasNoBlock``. Web3 can then wrap that in its ABI decode
    # error, keeping the original exception only in the wrapper's context.
    # This specific wrapper is still an infrastructure error; do not apply the
    # same context rule to arbitrary adapter exceptions.
    if isinstance(error, BadFunctionCallOutput):
        nested = error.__cause__ or error.__context__
        while nested is not None and nested is not error:
            if isinstance(nested, ProbablyNodeHasNoBlock):
                return True
            nested = nested.__cause__ or nested.__context__

    message = str(error).lower()
    if any(
        clue in message
        for clue in (
            "anvil did not start",
            "could not read block number from anvil",
            "could not restore simulated",
            "failed to create genesis",
            "rpc smoke test failed",
        )
    ):
        return True

    # Only follow explicit exception chaining. ``__context__`` merely means
    # this exception was raised while another was being handled: treating that
    # as causation can turn a deterministic adapter failure into an Anvil retry.
    nested = error.__cause__
    return (
        nested is not None
        and nested is not error
        and is_simulated_infrastructure_failure(nested)
    )


def queue_simulated_infrastructure_retry(
    spec: VaultSpec,
    pending_specs: deque,
    restart_counts: dict[str, int],
) -> bool:
    """Queue one clean rerun of a vault after replacing all Anvil forks.

    :return:
        ``True`` when the id was put back at the front of the sequential queue,
        or ``False`` after its single infrastructure retry was already used.
    """

    vault_id = spec.as_string_id()
    if restart_counts[vault_id] >= SIMULATED_VAULT_INFRASTRUCTURE_RESTARTS:
        return False
    restart_counts[vault_id] += 1
    pending_specs.appendleft(spec)
    return True


def raise_simulated_vault_attempt_timeout(signum, frame) -> None:
    """Interrupt a stuck fork-only adapter call so the next vault can run."""

    raise SimulatedVaultAttemptTimeout(
        f"Simulated vault attempt exceeded {SIMULATED_VAULT_ATTEMPT_TIMEOUT} seconds"
    )


def get_simulation_chain_ids(
    vault_specs: list[VaultSpec],
    *,
    primary_chain_id: ChainId,
) -> set[ChainId]:
    """Return every chain a simulated batch needs to fork."""

    assert vault_specs, "A simulated vault test needs at least one vault"
    return {primary_chain_id, *(ChainId(spec.chain_id) for spec in vault_specs)}


def parse_simulation_fork_blocks(
    values: list[str] | None,
) -> dict[ChainId, int]:
    """Parse repeatable ``CHAIN_ID:BLOCK_NUMBER`` command-line values."""

    blocks: dict[ChainId, int] = {}
    for value in values or []:
        chain_id_text, separator, block_number_text = value.partition(":")
        if not separator or not chain_id_text or not block_number_text:
            raise ValueError(
                "--simulation-fork-block must use CHAIN_ID:BLOCK_NUMBER, "
                f"got {value!r}"
            )
        try:
            chain_id = ChainId(int(chain_id_text))
        except ValueError as error:
            raise ValueError(
                f"Unknown chain id in --simulation-fork-block: {chain_id_text!r}"
            ) from error
        try:
            block_number = int(block_number_text)
        except ValueError as error:
            raise ValueError(
                f"Invalid block number in --simulation-fork-block: {block_number_text!r}"
            ) from error
        if block_number <= 0:
            raise ValueError(
                f"--simulation-fork-block must be positive, got {value!r}"
            )
        if chain_id in blocks:
            raise ValueError(
                f"Duplicate --simulation-fork-block for chain id {chain_id.value}"
            )
        blocks[chain_id] = block_number
    return blocks


def validate_simulation_fork_blocks(
    blocks: dict[ChainId, int],
    vault_specs: list[VaultSpec],
    *,
    primary_chain_id: ChainId,
) -> None:
    """Ensure fixed mode defines exactly every chain needed by the batch."""

    selected_chain_ids = get_simulation_chain_ids(
        vault_specs,
        primary_chain_id=primary_chain_id,
    )
    missing = sorted(selected_chain_ids - blocks.keys(), key=lambda chain_id: chain_id.value)
    unrelated = sorted(blocks.keys() - selected_chain_ids, key=lambda chain_id: chain_id.value)
    if missing or unrelated:
        details = []
        if missing:
            details.append(
                "missing " + ", ".join(str(chain_id.value) for chain_id in missing)
            )
        if unrelated:
            details.append(
                "unrelated " + ", ".join(str(chain_id.value) for chain_id in unrelated)
            )
        raise ValueError(
            "--simulation-fork-block values must cover exactly the simulated chains: "
            + "; ".join(details)
        )


def seed_vault_foundry_rpc_cache(cache_dir: Path = VAULT_TEST_RPC_CACHE_DIR) -> None:
    """Seed both the requested vault cache and Anvil's actual fork cache."""

    seed_default_foundry_rpc_cache(cache_dir)
    seed_foundry_rpc_cache(PROJECT_RPC_CACHE_SEED_DIR, cache_dir)
    seed_foundry_rpc_cache(cache_dir, DEFAULT_FOUNDRY_RPC_CACHE_DIR)
    seed_default_foundry_rpc_cache(DEFAULT_FOUNDRY_RPC_CACHE_DIR)
    seed_foundry_rpc_cache(PROJECT_RPC_CACHE_SEED_DIR, DEFAULT_FOUNDRY_RPC_CACHE_DIR)


def sync_foundry_rpc_cache_to_vault_cache(
    cache_dir: Path = VAULT_TEST_RPC_CACHE_DIR,
) -> int:
    """Persist Anvil's actual fork RPC cache into the vault cache directory."""

    if not DEFAULT_FOUNDRY_RPC_CACHE_DIR.exists():
        return 0
    return seed_foundry_rpc_cache(
        DEFAULT_FOUNDRY_RPC_CACHE_DIR,
        cache_dir,
        overwrite=True,
    )


def get_last_midnight_utc(now: datetime.datetime | None = None) -> datetime.datetime:
    """Return the most recent midnight as a naive UTC datetime."""

    now = now or native_datetime_utc_now()
    return datetime.datetime.combine(now.date(), datetime.time())


def resolve_block_number_at_or_before_timestamp(web3, target_at: datetime.datetime) -> int:
    """Resolve the latest block whose timestamp is not after ``target_at``."""

    target_timestamp = calendar.timegm(target_at.timetuple())
    latest = int(web3.eth.block_number)
    latest_timestamp = int(web3.eth.get_block(latest)["timestamp"])
    if latest_timestamp <= target_timestamp:
        return latest

    low = 0
    high = latest
    while low < high:
        middle = (low + high + 1) // 2
        block_timestamp = int(web3.eth.get_block(middle)["timestamp"])
        if block_timestamp <= target_timestamp:
            low = middle
        else:
            high = middle - 1
    return low


def get_shared_simulation_fork_blocks(
    vault_specs: list[VaultSpec],
    rpc_kwargs: dict | None = None,
    target_at: datetime.datetime | None = None,
    primary_chain_id: ChainId | None = None,
) -> dict[ChainId, int]:
    """Choose one fork block per vault chain at the last UTC midnight.

    Unit tests may omit RPC configuration and use eth-defi's static cached
    blocks. Command execution always resolves every selected chain dynamically.
    """

    fallback_primary_chain_id = primary_chain_id or ChainId(vault_specs[0].chain_id)
    selected_chain_ids = get_simulation_chain_ids(
        vault_specs,
        primary_chain_id=fallback_primary_chain_id,
    )
    if rpc_kwargs is None:
        return {
            chain_id: MIDNIGHT_BLOCKS[chain_id.value]
            for chain_id in selected_chain_ids
            if (
                chain_id not in LIVE_TIP_VAULT_TEST_CHAINS
                and chain_id.value in MIDNIGHT_BLOCKS
            )
        }

    target_at = target_at or get_last_midnight_utc()
    selected_fork_blocks: dict[ChainId, int] = {}
    for chain_id in sorted(selected_chain_ids, key=lambda candidate: candidate.value):
        rpc_key = f"json_rpc_{get_chain_slug(chain_id)}"
        configuration_line = rpc_kwargs.get(rpc_key)
        if not configuration_line:
            raise RuntimeError(
                f"Cannot resolve simulated {chain_id.name} fork block for "
                f"{target_at.isoformat()}: {rpc_key} is not configured"
            )
        web3 = create_multi_provider_web3(
            configuration_line,
            default_http_timeout=(3.0, 30.0),
            retries=2,
        )
        selected_fork_blocks[chain_id] = resolve_block_number_at_or_before_timestamp(
            web3,
            target_at,
        )
    return selected_fork_blocks


def rotate_simulated_rpc_upstreams(rpc_kwargs: dict) -> None:
    """Move each configured upstream's first provider behind its fallbacks.

    An upstream returning HTTP 200 with an empty ``eth_call`` result is not a
    transport error, so eth-defi's proxy cannot safely fail over by itself. A
    complete Anvil-generation retry is the right recovery boundary; rotate the
    provider order at that boundary so the new proxy starts with another
    archive provider instead of reproducing the same faulty response.
    """

    for key, value in rpc_kwargs.items():
        if not key.startswith("json_rpc_") or not isinstance(value, str):
            continue
        upstreams = value.split()
        if len(upstreams) > 1:
            rpc_kwargs[key] = " ".join(upstreams[1:] + upstreams[:1])


def start_simulated_vault_runtime(  # noqa: PLR0917
    *,
    generation: int,
    executor_id: str,
    rpc_kwargs: dict,
    unit_testing: bool,
    vault_specs: list[VaultSpec],
    primary_chain_id: ChainId,
    vault_universe,
    private_key: str,
    amount: Decimal,
    asset_management_mode: AssetManagementMode,
    confirmation_timeout: int,
    confirmation_block_count: int,
    min_gas_balance: float | None,
    max_slippage: float,
    token_cache,
    pinned_fork_blocks: dict[ChainId, int] | None = None,
) -> SimulatedVaultRuntime:
    """Create a complete disposable Anvil and Lagoon simulation generation.

    Setup uses normal command bootstrap after writing an ephemeral deployment
    artefact.  Any failure closes every Anvil already started for this
    generation before propagating to the bounded replacement loop.
    """

    web3config = None
    temporary_deployment_dir = None
    try:
        # Replacements use the same fixed blocks and share this persistent cache.
        seed_vault_foundry_rpc_cache()

        initial_pinned_fork_blocks = (
            pinned_fork_blocks
            or get_shared_simulation_fork_blocks(
                vault_specs,
                rpc_kwargs=rpc_kwargs,
                primary_chain_id=primary_chain_id,
            )
        )

        # Web3Config launches one local Anvil proxy for every selected upstream
        # RPC.  Local RPC retries are disabled; a dead process is replaced by the
        # generation-level retry outside this function.
        web3config = create_web3_config(
            **rpc_kwargs,
            unit_testing=unit_testing,
            simulate=True,
            simulate_http_timeout=(3.0, 40.0),
            simulate_fork_blocks=initial_pinned_fork_blocks,
            anvil_warm_up_block=True,
        )
        if not web3config.has_any_connection():
            raise RuntimeError("vault-test-trade requires JSON-RPC connections")
        web3config.set_default_chain(primary_chain_id)
        web3config.check_default_chain_id()
        fork_blocks = {
            str(chain_id.value): int(web3.eth.block_number)
            for chain_id, web3 in web3config.connections.items()
        }
        effective_pinned_fork_blocks = initial_pinned_fork_blocks | {
            chain_id: int(web3.eth.block_number)
            for chain_id, web3 in web3config.connections.items()
            if chain_id not in initial_pinned_fork_blocks
        }

        # Deploy the same hub/satellite Lagoon contracts used by integration
        # tests before constructing trade-executor models around them.
        deployment, artifact = deploy_simulated_lagoon_multichain(
            web3config=web3config,
            vault_specs=vault_specs,
            vault_universe=vault_universe,
            private_key=private_key,
            amount=amount,
            primary_chain_id=primary_chain_id,
        )
        # Normal bootstrap consumes a deployment file, so write the ephemeral
        # topology in its standard JSON shape rather than adding a special path.
        artifact["simulation_generation"] = generation
        temporary_deployment_dir = tempfile.TemporaryDirectory(
            prefix=f"vault-test-trade-generation-{generation}-"
        )
        deployment_file = (
            Path(temporary_deployment_dir.name) / f"{executor_id}.deployment.json"
        )
        deployment_file.write_text(json.dumps(artifact, indent=2))

        # Reuse production Lagoon transaction builders and sync models.  This is
        # what makes simulation diagnose real adapter/routing compatibility.
        execution_model, sync_model, _, _ = create_execution_and_sync_model(
            asset_management_mode=asset_management_mode,
            private_key=private_key,
            web3config=web3config,
            confirmation_timeout=datetime.timedelta(seconds=confirmation_timeout),
            confirmation_block_count=confirmation_block_count,
            min_gas_balance=min_gas_balance,
            max_slippage=max_slippage,
            vault_address=deployment.vault_address,
            vault_adapter_address=deployment.module_address,
            vault_payment_forwarder_address=None,
            token_cache=token_cache,
            deployment_file=deployment_file,
        )
        reserve_asset = translate_token_details(sync_model.vault.denomination_token)
        logger.info(
            "Started simulated vault runtime generation %d using Anvil processes %s",
            generation,
            {
                chain_id.name: anvil.process.pid
                for chain_id, anvil in web3config.anvils.items()
            },
        )
        return SimulatedVaultRuntime(
            generation=generation,
            web3config=web3config,
            deployment=deployment,
            deployment_file=deployment_file,
            execution_model=execution_model,
            sync_model=sync_model,
            reserve_asset=reserve_asset,
            temporary_deployment_dir=temporary_deployment_dir,
            fork_blocks=fork_blocks,
            pinned_fork_blocks=effective_pinned_fork_blocks,
        )
    except BaseException:
        # Multichain setup can fail after earlier forks and contracts exist.
        # Always tear down the partial generation before retrying from scratch.
        if web3config is not None:
            try:
                web3config.close(log_level=logging.ERROR, block_timeout=5)
            except Exception:
                logger.exception(
                    "Could not fully clean up a failed simulated vault runtime"
                )
        try:
            sync_foundry_rpc_cache_to_vault_cache()
        except Exception:
            logger.exception("Could not copy Foundry RPC cache after setup failure")
        if temporary_deployment_dir is not None:
            temporary_deployment_dir.cleanup()
        raise


def start_simulated_vault_runtime_with_replacement(**kwargs) -> SimulatedVaultRuntime:
    """Start a generation with one bounded whole-generation replacement.

    Deterministic deployment, adapter and contract errors escape immediately.
    Only failures classified as infrastructure consume the replacement budget.
    """

    generation = kwargs.pop("generation")
    last_error = None
    for offset in range(SIMULATED_VAULT_INFRASTRUCTURE_RESTARTS + 1):
        current_generation = generation + offset
        try:
            return start_simulated_vault_runtime(
                generation=current_generation,
                **kwargs,
            )
        except BaseException as e:
            if not is_simulated_infrastructure_failure(e):
                raise
            last_error = e
            logger.warning(
                "Discarding failed simulated vault runtime generation %d: %s",
                current_generation,
                e,
            )

    assert last_error is not None
    raise last_error


def take_simulated_snapshots(
    web3config, deployment, spec: VaultSpec
) -> dict[ChainId, str]:
    """Snapshot only chains that the selected vault attempt can mutate.

    A home-chain vault touches one fork.  A satellite vault touches the hub for
    CCTP and its destination fork for the vault transaction.
    """

    affected_chains = {
        deployment.primary_chain_id,
        ChainId(spec.chain_id),
    }
    return {
        chain_id: make_anvil_custom_rpc_request(
            web3config.get_connection(chain_id),
            "evm_snapshot",
        )
        for chain_id in affected_chains
    }


def restore_simulated_snapshots(web3config, fork_snapshots: dict[ChainId, str]) -> None:
    """Restore and health-check a still-responsive Anvil generation.

    ``evm_revert`` returning false means the snapshot is unusable.  A following
    block-number request catches a process that accepted the revert but became
    unresponsive immediately afterwards.
    """

    for chain_id, snapshot in fork_snapshots.items():
        web3 = web3config.get_connection(chain_id)
        reverted = make_anvil_custom_rpc_request(web3, "evm_revert", [snapshot])
        if reverted is not True:
            raise RPCRequestError(
                f"Could not restore simulated {chain_id.name} fork snapshot {snapshot}"
            )
        make_anvil_custom_rpc_request(web3, "eth_blockNumber")
