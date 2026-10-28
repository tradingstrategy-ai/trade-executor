"""Shared building blocks of the Lighter Lagoon vault tutorials.

The manual tutorials under ``scripts/lagoon/`` deploy a temporary Lagoon
vault whose Safe owns a Lighter master account and sub-accounts, drive it
with real ``trade-executor`` commands, wind it up and return every USDC to
the deployer. This module holds the parts they share: configuration, the
in-process command runner, deployment, state initialisation, subscription,
redemption and the final sweep and reconciliation.

Two modes exist:

- **Live**: Ethereum mainnet and Lighter's real sequencer. Needs a funded
  deployer key. Lighter withdrawals take 20 to 30 minutes each.
- **Simulate** (``SIMULATE=true``): the fork and fake sequencer of
  :py:mod:`tradeexecutor.testing.lighter_simulation`, so the executor's
  accounting is exercised end to end in minutes with no real money.

Secrets: the master API key stays in the mode-0600 operator record written
by deployment and each sub-account key in its mode-0600 hand-over file.
Nothing here prints or logs a key.
"""

import json
import logging
import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest import mock

from eth_defi.erc_4626.vault_protocol.lagoon.deployment import LIGHTER_BOOTSTRAP_SUBSCRIPTION
from eth_defi.erc_4626.vault_protocol.lagoon.vault import LagoonVault
from eth_defi.hotwallet import HotWallet
from eth_defi.lighter.constants import LIGHTER_ETHEREUM_DEPLOYMENT_CHAIN_ID, LIGHTER_USDC_ETHEREUM
from eth_defi.lighter.pubkey import MIN_API_KEY_INDEX
from eth_defi.lighter.session import create_lighter_session
from eth_defi.provider.broken_provider import _latest_delayed_block_number_cache
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.provider.receipt import wait_for_transaction_receipt_robust
from eth_defi.token import TokenDetails, fetch_erc20_details
from eth_defi.utils import setup_console_logging
from eth_defi.vault.base import VaultSpec
from web3 import Web3
from web3.contract.contract import ContractFunction

from tradeexecutor.cli.main import app
from tradeexecutor.ethereum.lighter.lighter_routing import DEFAULT_LIGHTER_TRANSFER_TIMEOUT_SECONDS, DEFAULT_LIGHTER_WITHDRAWAL_TIMEOUT_SECONDS
from tradeexecutor.exchange_account import lighter as lighter_reads
from tradeexecutor.exchange_account.lighter_operator import get_lighter_handover_path
from tradeexecutor.exchange_account.sub_account import get_lighter_account_resolution
from tradeexecutor.state.state import State
from tradeexecutor.strategy.strategy_module import read_strategy_module
from tradeexecutor.testing.lighter_manual import fetch_lighter_equity, reconcile_deployer, sweep_safe_to_deployer
from tradeexecutor.testing.lighter_report import get_lighter_funding_fees
from tradeexecutor.testing.lighter_simulation import ANVIL_DEPLOYER_PRIVATE_KEY, SimulatedLighter, launch_simulated_lighter

logger = logging.getLogger(__name__)

#: Directory of the test-only strategy modules.
STRATEGY_DIR = Path(__file__).resolve().parents[2] / "strategies" / "test_only"


#: API-key slot registered on the master and on every sub-account.
API_KEY_INDEX = MIN_API_KEY_INDEX

#: Maximum wait for a simulated withdrawal to become claimable; live runs use the executor default of an hour.
SIMULATED_WITHDRAW_TIMEOUT_SECONDS = 60

#: Maximum wait for a simulated transfer to appear in history; live runs use the executor default.
SIMULATED_TRANSFER_TIMEOUT_SECONDS = 5

#: Fee the fake sequencer charges per transfer, so simulated runs exercise the fee accounting.
SIMULATED_TRANSFER_FEE_USDC = Decimal("0.5")

#: Phase callables take the tutorial runtime.
Phase = Callable[["Tutorial"], None]


@dataclass(frozen=True, slots=True)
class TutorialConfig:
    """Operator configuration, with secrets excluded from its representation."""

    #: Ethereum RPC endpoint; hidden because a URL may contain credentials.
    rpc_url: str = field(repr=False)
    #: Funded deployer key used for the temporary vault lifecycle.
    deployer_private_key: str = field(repr=False)
    #: Private directory holding this run's artefacts and progress.
    run_dir: Path
    #: Executor id, also the prefix of the deployment artefacts.
    executor_id: str
    #: Strategy module driving the ``start`` cycles.
    strategy_file: Path
    #: USDC subscribed after deployment.
    subscription_usdc: Decimal
    #: Number of sub-accounts the strategy declares and deployment creates.
    sub_account_count: int
    #: Run against an Anvil fork and a fake Lighter sequencer.
    simulate: bool = False
    #: Existing Anvil fork to reuse in simulate mode, so a test can share its fork.
    simulate_rpc_url: str | None = field(default=None, repr=False)

    @property
    def withdraw_timeout(self) -> int:
        """Return the maximum wait for a withdrawal to become claimable."""
        return SIMULATED_WITHDRAW_TIMEOUT_SECONDS if self.simulate else DEFAULT_LIGHTER_WITHDRAWAL_TIMEOUT_SECONDS

    @property
    def transfer_timeout(self) -> int:
        """Return the maximum wait for a transfer between accounts to appear in history."""
        return SIMULATED_TRANSFER_TIMEOUT_SECONDS if self.simulate else DEFAULT_LIGHTER_TRANSFER_TIMEOUT_SECONDS


@dataclass(slots=True)
class Tutorial:
    """Runtime objects and public progress shared by tutorial phases."""

    #: Operator configuration.
    config: TutorialConfig
    #: RPC the commands use: the configured endpoint, or the fork in simulate mode.
    rpc_url: str = field(repr=False)
    #: Ethereum connection, excluded from repr because its URL may be sensitive.
    web3: Web3 = field(repr=False)
    #: Funded deployer wallet, excluded from repr because it holds a key.
    deployer: HotWallet = field(repr=False)
    #: Native Ethereum USDC.
    usdc: TokenDetails
    #: Public progress persisted in ``progress.json``.
    progress: dict[str, Any]
    #: Fork and fake sequencer in simulate mode.
    simulation: SimulatedLighter | None = None

    @property
    def progress_file(self) -> Path:
        """Return the progress manifest path."""
        return self.config.run_dir / "progress.json"

    @property
    def state_file(self) -> Path:
        """Return the executor state path."""
        return self.config.run_dir / "state.json"

    @property
    def record_file(self) -> Path:
        """Return the public vault record path."""
        return self.config.run_dir / "vault-record.txt"

    @property
    def key_backup_dir(self) -> Path:
        """Return where deployment saves each generated key until its records are written."""
        return self.config.run_dir / "key-backup"

    @property
    def operator_record_file(self) -> Path:
        """Return the mode-0600 operator record holding the master key."""
        return self.record_file.with_suffix(".json")

    @property
    def safe_address(self) -> str:
        """Return the deployed vault's Safe."""
        return self.progress["safe_address"]

    @property
    def account_indices(self) -> list[int]:
        """Return the master index followed by the sub-account indices."""
        return [int(index) for index in self.progress["account_indices"]]

    @property
    def sub_account_ids(self) -> list[str]:
        """Return the deployed sub-account ids in creation order."""
        return [str(account_id) for account_id in self.progress["sub_account_ids"]]

    def get_accounts(self) -> dict[str, int]:
        """Return the Lighter account index of the master and of each sub-account id, as recorded in the state."""
        return get_lighter_account_resolution(State.read_json_file(self.state_file))

    def get_handover_file(self, account_id: str) -> Path:
        """Return the mode-0600 hand-over file holding one sub-account's key."""
        return get_lighter_handover_path(self.record_file, account_id)

    def write_progress(self) -> None:
        """Persist progress, including values saved within a phase."""
        self.progress_file.write_text(json.dumps(self.progress, indent=2))

    def is_done(self, phase: int) -> bool:
        """Check whether a phase completed in an earlier run."""
        return self.progress.get("phase", 0) >= phase

    def close(self) -> None:
        """Release the simulated environment, if any."""
        if self.simulation is not None:
            self.simulation.close()
            self.simulation = None


def configure_logging() -> None:
    """Enable tutorial progress while suppressing credential-bearing provider logs."""
    setup_console_logging("INFO")
    logging.getLogger("tradeexecutor.testing").setLevel(logging.INFO)
    logging.getLogger("__main__").setLevel(logging.INFO)
    for namespace in ("web3", "urllib3", "eth_defi.provider"):
        logging.getLogger(namespace).setLevel(logging.WARNING)


def require_env(name: str, *fallback_names: str) -> str:
    """Read required configuration without echoing its value."""
    for key in (name, *fallback_names):
        if value := os.environ.get(key):
            return value
    raise RuntimeError(f"{name} is required")


def is_simulate_requested() -> bool:
    """Return whether ``SIMULATE`` asks for the Anvil and fake sequencer mode."""
    return os.environ.get("SIMULATE", "").lower() in {"1", "true", "yes"}


def get_default_run_dir(executor_id: str, simulate: bool) -> Path:
    """Return a fresh run directory: temporary when simulating, otherwise under the user's examples."""
    if simulate:
        return Path(tempfile.gettempdir()) / f"{executor_id}-simulate-{int(time.time())}"
    return Path("~/.tradingstrategy/examples").expanduser() / f"{executor_id}-{int(time.time())}"


def load_tutorial_config(*, executor_id: str, strategy_file: Path, subscription_usdc: Decimal) -> TutorialConfig:
    """Read a tutorial's environment: the RPC, the deployer key, the mode and the run directory.

    ``JSON_RPC_ETHEREUM`` is required in both modes: simulate mode forks it.
    The deployer key, ``LIGHTER_TEST_PRIVATE_KEY`` or ``PRIVATE_KEY``, is
    required only live; simulate mode uses a public Anvil test key.
    ``SIMULATE`` selects the mode, ``LIGHTER_RUN_DIR`` the run directory to
    resume, and ``LIGHTER_SIMULATE_RPC_URL`` lets a test share its fork.

    :param executor_id:
        Executor id of the tutorial.
    :param strategy_file:
        Strategy module driving the ``start`` cycles; its declared
        sub-accounts are created at deployment.
    :param subscription_usdc:
        USDC subscribed after deployment.
    :return:
        Validated configuration.
    """
    simulate = is_simulate_requested()
    sub_accounts = read_strategy_module(strategy_file).parameters.get("lighter_sub_accounts") or []
    return TutorialConfig(
        rpc_url=require_env("JSON_RPC_ETHEREUM"),
        deployer_private_key=ANVIL_DEPLOYER_PRIVATE_KEY if simulate else require_env("LIGHTER_TEST_PRIVATE_KEY", "PRIVATE_KEY"),
        run_dir=Path(os.environ.get("LIGHTER_RUN_DIR", str(get_default_run_dir(executor_id, simulate)))),
        executor_id=executor_id,
        strategy_file=strategy_file,
        subscription_usdc=subscription_usdc,
        sub_account_count=len(sub_accounts),
        simulate=simulate,
        simulate_rpc_url=os.environ.get("LIGHTER_SIMULATE_RPC_URL"),
    )


def prepare_tutorial(config: TutorialConfig) -> Tutorial:
    """Connect to Ethereum, then load earlier progress or check the deployer's funds.

    Simulate mode never resumes: every simulated run deploys a fresh vault.
    """
    config.run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    required_usdc = LIGHTER_BOOTSTRAP_SUBSCRIPTION + config.subscription_usdc
    simulation = None
    if config.simulate:
        simulation = launch_simulated_lighter(
            config.rpc_url,
            deployer_usdc=required_usdc,
            existing_rpc_url=config.simulate_rpc_url,
            transfer_fee_usdc=SIMULATED_TRANSFER_FEE_USDC,
        )
        web3, rpc_url, deployer, usdc = simulation.web3, simulation.rpc_url, simulation.deployer, simulation.usdc
        (config.run_dir / "progress.json").unlink(missing_ok=True)
    else:
        rpc_url = config.rpc_url
        web3 = create_multi_provider_web3(rpc_url, default_http_timeout=(3.0, 180.0))
        deployer = HotWallet.from_private_key(config.deployer_private_key)
        usdc = fetch_erc20_details(web3, LIGHTER_USDC_ETHEREUM)
    assert web3.eth.chain_id == LIGHTER_ETHEREUM_DEPLOYMENT_CHAIN_ID, "Ethereum mainnet RPC required"

    tutorial = Tutorial(config=config, rpc_url=rpc_url, web3=web3, deployer=deployer, usdc=usdc, progress={}, simulation=simulation)
    if tutorial.progress_file.exists():
        tutorial.progress = json.loads(tutorial.progress_file.read_text())
    else:
        start_usdc = usdc.fetch_balance_of(deployer.address)
        if start_usdc < required_usdc:
            raise RuntimeError(f"Deployer {deployer.address} has {start_usdc} USDC but the tutorial needs {required_usdc} USDC")
        tutorial.progress = {
            "start_usdc": str(start_usdc),
            "start_eth_wei": web3.eth.get_balance(deployer.address),
        }
    logger.info("Deployer: %s", deployer.address)
    logger.info("Run artefacts: %s", config.run_dir)
    logger.info("Resuming after step %d", tutorial.progress.get("phase", 0))
    return tutorial


def run_cli(args: list[str], environment: dict[str, str], check: bool = True) -> int:
    """Invoke a real Typer command without printing its environment.

    :param args:
        Command and its options.
    :param environment:
        The command's whole environment.
    :param check:
        Raise ``SystemExit`` on a non-zero exit code instead of returning it.
    :return:
        The exit code.
    """
    _latest_delayed_block_number_cache.clear()
    logger.info("Running: trade-executor %s", " ".join(args))
    patched_environment = dict(environment)
    for key in ("PATH", "HOME", "USER", "TMPDIR", "SHELL"):
        if key not in patched_environment and key in os.environ:
            patched_environment[key] = os.environ[key]
    try:
        with mock.patch.dict(os.environ, patched_environment, clear=True):
            try:
                app(args, standalone_mode=False)
            except SystemExit as error:
                if error.code not in (None, 0):
                    if check:
                        raise
                    return error.code
    finally:
        # Typer commands replace the root logger with their own LOG_LEVEL.
        configure_logging()
    return 0


def broadcast_and_wait(tutorial: Tutorial, function: ContractFunction) -> None:
    """Broadcast one deployer transaction and wait for its receipt."""
    tutorial.deployer.sync_nonce(tutorial.web3)
    tx_hash = tutorial.deployer.transact_and_broadcast_with_contract(function)
    receipt = wait_for_transaction_receipt_robust(tutorial.web3, tx_hash)
    assert receipt["status"] == 1, f"Transaction reverted: {Web3.to_hex(tx_hash)}"


def get_vault(tutorial: Tutorial) -> LagoonVault:
    """Create the vault wrapper from public progress fields."""
    return LagoonVault(
        tutorial.web3,
        VaultSpec(tutorial.web3.eth.chain_id, tutorial.progress["vault_address"]),
        trading_strategy_module_address=tutorial.progress["module_address"],
        default_block_identifier="latest",
        require_denomination_token=True,
    )


def get_base_environment(tutorial: Tutorial) -> dict[str, str]:
    """Return the executor environment shared by deployment and every later command."""
    return {
        "EXECUTOR_ID": tutorial.config.executor_id,
        "STRATEGY_FILE": str(tutorial.config.strategy_file),
        "STATE_FILE": str(tutorial.state_file),
        "PRIVATE_KEY": tutorial.config.deployer_private_key,
        "JSON_RPC_ETHEREUM": tutorial.rpc_url,
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "warning",
    }


def get_cli_environment(tutorial: Tutorial, **overrides: str) -> dict[str, str]:
    """Return the executor environment used for every post-deployment command."""
    environment = {
        **get_base_environment(tutorial),
        "NAME": f"{tutorial.config.executor_id} tutorial",
        "ASSET_MANAGEMENT_MODE": "lagoon",
        "VAULT_ADDRESS": tutorial.progress["vault_address"],
        "VAULT_ADAPTER_ADDRESS": tutorial.progress["module_address"],
        "LIGHTER_OPERATOR_RECORD_FILE": str(tutorial.operator_record_file),
        "LIGHTER_WITHDRAWAL_TIMEOUT": str(tutorial.config.withdraw_timeout),
        "LIGHTER_TRANSFER_TIMEOUT": str(tutorial.config.transfer_timeout),
        "RUN_SINGLE_CYCLE": "true",
        "CACHE_PATH": str(tutorial.config.run_dir / "cache"),
        "MIN_GAS_BALANCE": "0",
    }
    if trading_strategy_api_key := os.environ.get("TRADING_STRATEGY_API_KEY"):
        environment["TRADING_STRATEGY_API_KEY"] = trading_strategy_api_key
    environment.update(overrides)
    return environment


def fetch_lighter_balances(tutorial: Tutorial) -> dict[int, Decimal]:
    """Read every Safe-owned Lighter account's equity from one public snapshot."""
    equity = fetch_lighter_equity(tutorial.safe_address)
    return {index: equity[index].get_total() for index in tutorial.account_indices}


def run_start(tutorial: Tutorial, **overrides: str) -> None:
    """Run one ``start`` cycle and log the resulting Safe and Lighter balances."""
    run_cli(["start"], get_cli_environment(tutorial, **overrides))
    logger.info("Safe USDC: %s", tutorial.usdc.fetch_balance_of(tutorial.safe_address))
    for index, value in fetch_lighter_balances(tutorial).items():
        logger.info("Lighter account %d equity: %s", index, value)


def deploy(tutorial: Tutorial, fund_name: str, fund_symbol: str) -> None:
    """Deploy the vault with keyed sub-accounts and record its public details.

    Deployment writes the generated keys to their records only when it
    completes, so ``LIGHTER_KEY_BACKUP_DIR`` saves each key in the run
    directory before it is registered; the command deletes the backups after
    a successful deployment and keeps them for recovery after a failure.
    """
    environment = {
        **get_base_environment(tutorial),
        "VAULT_RECORD_FILE": str(tutorial.record_file),
        "FUND_NAME": fund_name,
        "FUND_SYMBOL": fund_symbol,
        "MULTISIG_OWNERS": tutorial.deployer.address,
        "ANY_ASSET": "true",
        "PERFORMANCE_FEE": "0",
        "MANAGEMENT_FEE": "0",
        "GENERATE_LIGHTER_API_KEY": "true",
        "LIGHTER_API_KEY_INDEX": str(API_KEY_INDEX),
        "LIGHTER_SUB_ACCOUNT_API_KEY_INDEX": str(API_KEY_INDEX),
        "LIGHTER_KEY_BACKUP_DIR": str(tutorial.key_backup_dir),
    }
    run_cli(["lagoon-deploy-vault"], environment)
    record_deployment(tutorial)


def record_deployment(tutorial: Tutorial) -> None:
    """Check the deployed accounts against Lighter and record their public details."""
    # Read only public fields; the private record is passed to commands by path.
    public = json.loads(tutorial.state_file.with_name(f"{tutorial.config.executor_id}.deployment.json").read_text())["deployments"]["ethereum"]
    setup = public["lighter_account_setup"]
    sub_indices = [int(sub["account_index"]) for sub in setup["sub_accounts"]]
    sub_ids = [str(sub["label"]) for sub in setup["sub_accounts"]]
    assert len(sub_indices) == tutorial.config.sub_account_count
    tutorial.progress.update({"vault_address": public["vault_address"], "module_address": public["module_address"], "sub_account_ids": sub_ids})
    safe_address = get_vault(tutorial).safe_address
    session = create_lighter_session()
    try:
        accounts = lighter_reads.fetch_lighter_accounts(session, safe_address)
    finally:
        session.close()
    assert accounts is not None and accounts.master_index == int(setup["account_index"])
    assert set(sub_indices) <= set(accounts.sub_account_indices), "Lighter does not list the created sub-accounts"
    for account_id in sub_ids:
        handover = tutorial.get_handover_file(account_id)
        assert handover.exists() and handover.stat().st_mode & 0o077 == 0, f"Missing or unsafe hand-over file {handover}"
    tutorial.progress.update({
        "safe_address": safe_address,
        "account_indices": [accounts.master_index, *sub_indices],
    })
    logger.info("Master account %d, sub-accounts %s", accounts.master_index, dict(zip(sub_ids, sub_indices)))


def initialise_state(tutorial: Tutorial) -> None:
    """Create executor state, account for the bootstrap subscription and check accounts."""
    environment = get_cli_environment(tutorial)
    run_cli(["init"], environment)
    run_cli(["correct-accounts", "--process-redemption"], environment)
    run_cli(["check-accounts"], environment)
    state = State.read_json_file(tutorial.state_file)
    lighter_positions = [position for position in state.portfolio.open_positions.values() if position.is_exchange_account()]
    assert len(lighter_positions) == 1 + tutorial.config.sub_account_count, "Expected one position per Lighter account"
    assert list(get_lighter_account_resolution(state).values()) == tutorial.account_indices, "init did not record the Lighter accounts in the state"


def subscribe(tutorial: Tutorial, settle_with_start: bool = True, **start_overrides: str) -> None:
    """Subscribe to Lagoon and settle the subscription, by default with one ``start`` cycle.

    The deposit request is recorded in ``progress.json`` and checked onchain
    first, so a resumed run settles and claims the earlier request instead of
    subscribing twice.

    :param settle_with_start:
        Settle with a ``start`` cycle, which also runs the strategy. With
        ``False`` the subscription is settled with
        ``correct-accounts --process-redemption`` and stays in the Safe.
    :param start_overrides:
        Extra environment for the ``start`` cycle.
    """
    vault = get_vault(tutorial)
    deployer = tutorial.deployer.address
    amount = tutorial.config.subscription_usdc
    pending_raw = vault.vault_contract.functions.pendingDepositRequest(0, deployer).call()
    if not tutorial.progress.get("subscription_requested") and pending_raw == 0:
        assert tutorial.usdc.fetch_balance_of(deployer) >= amount, "Deployer lacks USDC"
        broadcast_and_wait(tutorial, tutorial.usdc.approve(vault.address, amount))
        broadcast_and_wait(tutorial, vault.request_deposit(deployer, tutorial.usdc.convert_to_raw(amount)))
        pending_raw = tutorial.usdc.convert_to_raw(amount)
    tutorial.progress["subscription_requested"] = True
    tutorial.write_progress()
    if pending_raw > 0 and settle_with_start:
        run_start(tutorial, **start_overrides)
    elif pending_raw > 0:
        run_cli(["correct-accounts", "--process-redemption"], get_cli_environment(tutorial))
    if vault.vault_contract.functions.maxDeposit(deployer).call() > 0:
        broadcast_and_wait(tutorial, vault.finalise_deposit(deployer))


def redeem_all_shares(tutorial: Tutorial, drain_cycles: int, **start_overrides: str) -> None:
    """Request a full redemption, let ``start`` settle it and claim with ``lagoon-redeem``.

    :param tutorial:
        Tutorial runtime.
    :param drain_cycles:
        ``start`` cycles to run after the request; the strategy drains the
        Lighter accounts and Lagoon settles the redemption during them.
    :param start_overrides:
        Extra environment for the ``start`` cycles, such as sub-account signer files.
    """
    vault = get_vault(tutorial)
    deployer = tutorial.deployer.address
    requested = tutorial.progress.get("redemption_requested") or vault.vault_contract.functions.pendingRedeemRequest(0, deployer).call() > 0 or vault.vault_contract.functions.maxRedeem(deployer).call() > 0
    if not requested:
        shares = vault.share_token.fetch_raw_balance_of(deployer)
        assert shares > 0, "The deployer holds no shares to redeem"
        broadcast_and_wait(tutorial, vault.request_redeem(deployer, shares))
    tutorial.progress["redemption_requested"] = True
    tutorial.write_progress()
    for _ in range(drain_cycles):
        if vault.vault_contract.functions.maxRedeem(deployer).call() > 0:
            break
        run_start(tutorial, **start_overrides)
    run_cli(["lagoon-redeem"], get_cli_environment(tutorial, **start_overrides))
    assert vault.share_token.fetch_raw_balance_of(deployer) == 0, "Shares remain after lagoon-redeem"


def sweep_safe(tutorial: Tutorial) -> None:
    """Check nothing backs investor claims, then move the Safe's USDC and ETH to the deployer."""
    sweep_safe_to_deployer(tutorial.web3, get_vault(tutorial), tutorial.deployer, tutorial.usdc)


def reconcile_deployer_balances(tutorial: Tutorial, extra_fees_usdc: Decimal = Decimal(0)) -> list[tuple[str, Decimal]]:
    """Explain the deployer's USDC and ETH change over the run, failing on an unexplained difference.

    :param tutorial:
        Tutorial runtime.
    :param extra_fees_usdc:
        Fees paid outside the executor's trades, for example by a trader-made transfer.
    :return:
        The reconciliation rows.
    """
    state = State.read_json_file(tutorial.state_file)
    return reconcile_deployer(
        tutorial.web3,
        tutorial.deployer,
        tutorial.usdc,
        start_usdc=Decimal(tutorial.progress["start_usdc"]),
        start_eth_wei=tutorial.progress["start_eth_wei"],
        fees=get_lighter_funding_fees(state) + extra_fees_usdc,
        lighter_dust=sum(fetch_lighter_balances(tutorial).values(), Decimal(0)),
    )


def refund(tutorial: Tutorial, extra_fees_usdc: Decimal = Decimal(0)) -> None:
    """Sweep the Safe back to the deployer and reconcile the deployer's balances."""
    sweep_safe(tutorial)
    reconcile_deployer_balances(tutorial, extra_fees_usdc)


def run_phases(tutorial: Tutorial, phases: list[tuple[int, str, Phase]]) -> None:
    """Run numbered phases, skipping those an earlier run completed.

    :param tutorial:
        Tutorial runtime.
    :param phases:
        ``(number, title, callable)`` in order.
    """
    for number, title, phase in phases:
        if tutorial.is_done(number):
            logger.info("Step %d already completed: %s", number, title)
            continue
        logger.info("%s", "=" * 72)
        logger.info("Step %d: %s", number, title)
        logger.info("%s", "=" * 72)
        phase(tutorial)
        tutorial.progress["phase"] = number
        tutorial.write_progress()
    logger.info("Tutorial complete. Public artefacts and protected records are in %s", tutorial.config.run_dir)
