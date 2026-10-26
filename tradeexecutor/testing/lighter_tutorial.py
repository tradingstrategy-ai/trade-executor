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
- **Simulate** (``SIMULATE=true``): a fixed-block Anvil fork of Ethereum
  with :py:class:`~tradeexecutor.testing.lighter_core.FakeLighterCore`
  standing in for the sequencer. Every Safe, guard, Lagoon and ZkLighter
  transaction still runs, so the executor's accounting is exercised end to
  end in minutes with no real money.

Secrets: the master API key stays in the mode-0600 operator record written
by deployment and each sub-account key in its mode-0600 hand-over file.
Nothing here prints or logs a key.
"""

import json
import logging
import os
import tempfile
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest import mock

import eth_defi.erc_4626.vault_protocol.lagoon.deployment as lagoon_deployment
from eth_defi.compat import native_datetime_utc_now
from eth_defi.erc_4626.vault_protocol.lagoon.deployment import LIGHTER_BOOTSTRAP_SUBSCRIPTION
from eth_defi.erc_4626.vault_protocol.lagoon.vault import LagoonVault
from eth_defi.hotwallet import HotWallet
from eth_defi.lighter.api_key import LighterApiKey
from eth_defi.lighter.constants import LIGHTER_ETHEREUM_DEPLOYMENT_CHAIN_ID, LIGHTER_USDC_ETHEREUM
from eth_defi.lighter.pubkey import MIN_API_KEY_INDEX
from eth_defi.lighter.session import create_lighter_session
from eth_defi.provider.anvil import AnvilLaunch, fork_network_anvil
from eth_defi.provider.broken_provider import _latest_delayed_block_number_cache
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.provider.receipt import wait_for_transaction_receipt_robust
from eth_defi.safe.sweep import sweep_safe_to_owner
from eth_defi.testing.fork_blocks import ETHEREUM_MIDNIGHT_BLOCK
from eth_defi.token import USDC_WHALE, TokenDetails, fetch_erc20_details
from eth_defi.trace import assert_transaction_success_with_explanation
from eth_defi.utils import setup_console_logging
from eth_defi.vault.base import VaultSpec
from tabulate import tabulate
from web3 import Web3
from web3.contract.contract import ContractFunction

from tradeexecutor.cli.main import app
from tradeexecutor.exchange_account import lighter as lighter_reads
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeFlag
from tradeexecutor.testing.lighter_core import FakeLighterCore

logger = logging.getLogger(__name__)

#: Directory of the test-only strategy modules.
STRATEGY_DIR = Path(__file__).resolve().parents[2] / "strategies" / "test_only"

#: Lagoon cannot return sub-unit rounding residue; tolerate two raw units.
MAX_UNEXPLAINED_RAW = 2

#: Public deterministic Anvil account zero key used as the simulated deployer; never a production secret.
ANVIL_DEPLOYER_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

#: Master account index the fake sequencer assigns in simulate mode.
SIMULATED_MASTER_INDEX = 127


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
    #: API-key slot for the master and sub-account keys.
    api_key_index: int
    #: Maximum wait for a withdrawal to become claimable.
    withdraw_timeout: int
    #: Maximum wait for a sub-account transfer to appear in history.
    transfer_timeout: int
    #: Number of sub-accounts created at deployment.
    sub_account_count: int = 2
    #: Run against an Anvil fork and a fake Lighter sequencer.
    simulate: bool = False
    #: Existing Anvil fork to reuse in simulate mode, instead of launching one.
    simulate_rpc_url: str | None = field(default=None, repr=False)
    #: Fee the fake sequencer charges per transfer in simulate mode.
    simulated_transfer_fee_usdc: Decimal = Decimal(0)


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
    #: Fake sequencer in simulate mode.
    fake_core: FakeLighterCore | None = None
    #: Anvil fork launched for simulate mode.
    anvil: AnvilLaunch | None = None
    #: Installed fake sequencer patches and other resources to release.
    resources: ExitStack = field(default_factory=ExitStack)

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
    def operator_record_file(self) -> Path:
        """Return the mode-0600 operator record holding the master key."""
        return self.record_file.with_suffix(".json")

    @property
    def account_indices(self) -> list[int]:
        """Return the master index followed by the sub-account indices."""
        return [int(index) for index in self.progress["account_indices"]]

    def get_handover_file(self, account_index: int) -> Path:
        """Return the mode-0600 hand-over file holding one sub-account's key."""
        return self.record_file.with_name(f"{self.record_file.stem}.nautilus.{account_index}.json")

    def write_progress(self) -> None:
        """Persist progress, including values saved within a phase."""
        self.progress_file.write_text(json.dumps(self.progress, indent=2))

    def is_done(self, phase: int) -> bool:
        """Check whether a phase completed in an earlier run."""
        return self.progress.get("phase", 0) >= phase

    def close(self) -> None:
        """Release the fake sequencer patches and any launched fork."""
        self.resources.close()
        if self.anvil is not None:
            self.anvil.close()
            self.anvil = None


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


def load_tutorial_config(
    *,
    executor_id: str,
    strategy_file: Path,
    subscription_usdc: Decimal,
    sub_account_count: int = 2,
) -> TutorialConfig:
    """Read the shared environment variables of a tutorial.

    ``JSON_RPC_ETHEREUM`` is required in both modes: simulate mode forks it.
    The deployer key, ``LIGHTER_TEST_PRIVATE_KEY`` or ``PRIVATE_KEY``, is
    required only live; simulate mode uses a public Anvil test key.

    :param executor_id:
        Executor id of the tutorial.
    :param strategy_file:
        Strategy module driving the ``start`` cycles.
    :param subscription_usdc:
        USDC subscribed after deployment.
    :param sub_account_count:
        Sub-accounts created at deployment.
    :return:
        Validated configuration.
    """
    simulate = is_simulate_requested()
    default_run_dir = Path(tempfile.gettempdir()) / f"{executor_id}-simulate-{int(time.time())}" if simulate else Path("~/.tradingstrategy/examples").expanduser() / f"{executor_id}-{int(time.time())}"
    return TutorialConfig(
        rpc_url=require_env("JSON_RPC_ETHEREUM"),
        deployer_private_key=ANVIL_DEPLOYER_PRIVATE_KEY if simulate else require_env("LIGHTER_TEST_PRIVATE_KEY", "PRIVATE_KEY"),
        run_dir=Path(os.environ.get("LIGHTER_RUN_DIR", str(default_run_dir))),
        executor_id=executor_id,
        strategy_file=strategy_file,
        subscription_usdc=subscription_usdc,
        api_key_index=int(os.environ.get("LIGHTER_API_KEY_INDEX", str(MIN_API_KEY_INDEX))),
        withdraw_timeout=int(os.environ.get("LIGHTER_WITHDRAW_TIMEOUT", "60" if simulate else "3600")),
        transfer_timeout=int(os.environ.get("LIGHTER_TRANSFER_TIMEOUT", "5" if simulate else "300")),
        sub_account_count=sub_account_count,
        simulate=simulate,
        simulate_rpc_url=os.environ.get("LIGHTER_SIMULATE_RPC_URL"),
        simulated_transfer_fee_usdc=Decimal(os.environ.get("LIGHTER_SIMULATED_TRANSFER_FEE_USDC", "0.5")),
    )


def prepare_tutorial(config: TutorialConfig) -> Tutorial:
    """Connect to Ethereum, load earlier progress or check the deployer's funds.

    Simulate mode forks Ethereum at a fixed block, funds the Anvil deployer
    with ETH and USDC from a whale, and installs the fake sequencer. It never
    resumes: every simulated run deploys a fresh vault.
    """
    config.run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    anvil = None
    fake_core = None
    resources = ExitStack()
    if config.simulate:
        if config.simulate_rpc_url:
            rpc_url = config.simulate_rpc_url
        else:
            logger.info("SIMULATE=true: forking Ethereum at block %d", ETHEREUM_MIDNIGHT_BLOCK)
            anvil = fork_network_anvil(config.rpc_url, fork_block_number=ETHEREUM_MIDNIGHT_BLOCK, unlocked_addresses=[USDC_WHALE[1]])
            rpc_url = anvil.json_rpc_url
        web3 = create_multi_provider_web3(rpc_url, default_http_timeout=(3.0, 250.0))
        deployer = HotWallet.from_private_key(config.deployer_private_key)
        usdc = fetch_erc20_details(web3, LIGHTER_USDC_ETHEREUM)
        web3.provider.make_request("anvil_setBalance", [deployer.address, hex(10 * 10**18)])
        funding = usdc.contract.functions.transfer(deployer.address, usdc.convert_to_raw(LIGHTER_BOOTSTRAP_SUBSCRIPTION + config.subscription_usdc)).transact({"from": USDC_WHALE[1]})
        assert_transaction_success_with_explanation(web3, funding)
        fake_core = FakeLighterCore(web3, SIMULATED_MASTER_INDEX, transfer_fee=config.simulated_transfer_fee_usdc)
        for target, replacement in fake_core.get_patches():
            resources.enter_context(mock.patch(target, new=replacement))
        if (config.run_dir / "progress.json").exists():
            (config.run_dir / "progress.json").unlink()
    else:
        rpc_url = config.rpc_url
        web3 = create_multi_provider_web3(rpc_url, default_http_timeout=(3.0, 180.0))
        deployer = HotWallet.from_private_key(config.deployer_private_key)
        usdc = fetch_erc20_details(web3, LIGHTER_USDC_ETHEREUM)
    assert web3.eth.chain_id == LIGHTER_ETHEREUM_DEPLOYMENT_CHAIN_ID, "Ethereum mainnet RPC required"

    tutorial = Tutorial(config=config, rpc_url=rpc_url, web3=web3, deployer=deployer, usdc=usdc, progress={}, fake_core=fake_core, anvil=anvil, resources=resources)
    if tutorial.progress_file.exists():
        tutorial.progress = json.loads(tutorial.progress_file.read_text())
    else:
        start_usdc = usdc.fetch_balance_of(deployer.address)
        required_usdc = LIGHTER_BOOTSTRAP_SUBSCRIPTION + config.subscription_usdc
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


def run_cli(args: list[str], environment: dict[str, str]) -> None:
    """Invoke a real Typer command without printing its environment."""
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
                    raise
    finally:
        # Typer commands replace the root logger with their own LOG_LEVEL.
        configure_logging()


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


def get_cli_environment(tutorial: Tutorial, **overrides: str) -> dict[str, str]:
    """Return the executor environment used for every post-deployment command."""
    environment = {
        "ID": tutorial.config.executor_id,
        "NAME": f"{tutorial.config.executor_id} tutorial",
        "STRATEGY_FILE": str(tutorial.config.strategy_file),
        "STATE_FILE": str(tutorial.state_file),
        "PRIVATE_KEY": tutorial.config.deployer_private_key,
        "JSON_RPC_ETHEREUM": tutorial.rpc_url,
        "ASSET_MANAGEMENT_MODE": "lagoon",
        "VAULT_ADDRESS": tutorial.progress["vault_address"],
        "VAULT_ADAPTER_ADDRESS": tutorial.progress["module_address"],
        "LIGHTER_ACCOUNT_INDEXES": ",".join(str(index) for index in tutorial.account_indices),
        "LIGHTER_OPERATOR_RECORD_FILE": str(tutorial.operator_record_file),
        "LIGHTER_WITHDRAWAL_TIMEOUT": str(tutorial.config.withdraw_timeout),
        "LIGHTER_TRANSFER_TIMEOUT": str(tutorial.config.transfer_timeout),
        "RUN_SINGLE_CYCLE": "true",
        "UNIT_TESTING": "true",
        "CACHE_PATH": str(tutorial.config.run_dir / "cache"),
        "MIN_GAS_BALANCE": "0",
        "LOG_LEVEL": "warning",
    }
    if trading_strategy_api_key := os.environ.get("TRADING_STRATEGY_API_KEY"):
        environment["TRADING_STRATEGY_API_KEY"] = trading_strategy_api_key
    environment.update(overrides)
    return environment


def fetch_lighter_balances(tutorial: Tutorial) -> dict[int, Decimal]:
    """Read every Safe-owned Lighter account's equity from one public snapshot."""
    session = create_lighter_session()
    try:
        equity = lighter_reads.fetch_lighter_accounts_equity(session, tutorial.progress["safe_address"])
    finally:
        session.close()
    return {index: equity[index].get_total() for index in tutorial.account_indices}


def run_start(tutorial: Tutorial, **overrides: str) -> None:
    """Run one ``start`` cycle and log the resulting Safe and Lighter balances."""
    run_cli(["start"], get_cli_environment(tutorial, **overrides))
    logger.info("Safe USDC: %s", tutorial.usdc.fetch_balance_of(tutorial.progress["safe_address"]))
    for index, value in fetch_lighter_balances(tutorial).items():
        logger.info("Lighter account %d equity: %s", index, value)


def save_generated_key(key: LighterApiKey, number: int, deployer: str) -> Path:
    """Write one newly generated Lighter API key to a private temporary file.

    :param key:
        Key pair just generated by the deployment.
    :param number:
        Generation order: 1 is the master account key, then one per sub-account
        in creation order.
    :param deployer:
        Deployer address, to identify the deployment the key belongs to.
    :return:
        Path of the file, created with mode 0600 by :py:func:`tempfile.mkstemp`.
    """
    descriptor, name = tempfile.mkstemp(prefix=f"lighter-api-key-{number}-", suffix=".json")
    with os.fdopen(descriptor, "w") as out:
        json.dump(
            {
                "role": "master" if number == 1 else f"sub-account {number - 1}",
                "deployer": deployer,
                "created_at": native_datetime_utc_now().isoformat(),
                "api_key_index": key.api_key_index,
                "public_key": key.public_key,
                "private_key": key.private_key,
            },
            out,
            indent=2,
        )
    return Path(name)


def deploy(tutorial: Tutorial, fund_name: str, fund_symbol: str) -> None:
    """Deploy the vault with keyed sub-accounts, backing up each key as it is generated.

    Deployment writes its records only when it completes, so each generated
    key is first saved to its own mode-0600 temporary file. A failed
    deployment keeps those files for recovery; a successful one deletes them.
    """
    generate_key = lagoon_deployment.generate_lighter_api_key_pair
    saved_keys: list[Path] = []

    def generate_and_save_key(api_key_index: int) -> LighterApiKey:
        key = generate_key(api_key_index)
        saved_keys.append(save_generated_key(key, len(saved_keys) + 1, tutorial.deployer.address))
        logger.warning("Saved generated Lighter API key %d to %s until deployment completes", len(saved_keys), saved_keys[-1])
        return key

    environment = {
        "EXECUTOR_ID": tutorial.config.executor_id,
        "STRATEGY_FILE": str(STRATEGY_DIR / "minimal_lighter_strategy.py"),
        "LIGHTER_ACCOUNT_INDEX": "0",
        "STATE_FILE": str(tutorial.state_file),
        "PRIVATE_KEY": tutorial.config.deployer_private_key,
        "JSON_RPC_ETHEREUM": tutorial.rpc_url,
        "VAULT_RECORD_FILE": str(tutorial.record_file),
        "FUND_NAME": fund_name,
        "FUND_SYMBOL": fund_symbol,
        "MULTISIG_OWNERS": tutorial.deployer.address,
        "ANY_ASSET": "true",
        "PERFORMANCE_FEE": "0",
        "MANAGEMENT_FEE": "0",
        "UNIT_TESTING": "true",
        "GENERATE_LIGHTER_API_KEY": "true",
        "LIGHTER_API_KEY_INDEX": str(tutorial.config.api_key_index),
        "LIGHTER_SUB_ACCOUNT_COUNT": str(tutorial.config.sub_account_count),
        "LIGHTER_SUB_ACCOUNT_API_KEY_INDEX": str(tutorial.config.api_key_index),
        "LOG_LEVEL": "warning",
    }
    with mock.patch.object(lagoon_deployment, "generate_lighter_api_key_pair", generate_and_save_key):
        run_cli(["lagoon-deploy-vault"], environment)

    # The operator record and hand-over files now hold every key.
    for path in saved_keys:
        path.unlink()
    logger.info("Deployment complete; deleted %d temporary key files", len(saved_keys))
    record_deployment(tutorial)


def record_deployment(tutorial: Tutorial) -> None:
    """Check the deployed accounts against Lighter and record their public details."""
    # Read only public fields; the private record is passed to commands by path.
    public = json.loads(tutorial.state_file.with_name(f"{tutorial.config.executor_id}.deployment.json").read_text())["deployments"]["ethereum"]
    setup = public["lighter_account_setup"]
    sub_indices = [int(sub["account_index"]) for sub in setup["sub_accounts"]]
    assert len(sub_indices) == tutorial.config.sub_account_count
    tutorial.progress.update({"vault_address": public["vault_address"], "module_address": public["module_address"]})
    safe_address = get_vault(tutorial).safe_address
    session = create_lighter_session()
    try:
        accounts = lighter_reads.fetch_lighter_accounts(session, safe_address)
    finally:
        session.close()
    assert accounts is not None and accounts.master_index == int(setup["account_index"])
    assert set(sub_indices) <= set(accounts.sub_account_indices), "Lighter does not list the created sub-accounts"
    for index in sub_indices:
        handover = tutorial.get_handover_file(index)
        assert handover.exists() and handover.stat().st_mode & 0o077 == 0, f"Missing or unsafe hand-over file {handover}"
    tutorial.progress.update({
        "safe_address": safe_address,
        "account_indices": [accounts.master_index, *sub_indices],
    })
    logger.info("Master account %d, sub-accounts %s", accounts.master_index, sub_indices)


def initialise_state(tutorial: Tutorial) -> None:
    """Create executor state, account for the bootstrap subscription and check accounts."""
    environment = get_cli_environment(tutorial)
    run_cli(["init"], environment)
    run_cli(["correct-accounts", "--process-redemption"], environment)
    run_cli(["check-accounts"], environment)
    state = State.read_json_file(tutorial.state_file)
    lighter_positions = [position for position in state.portfolio.open_positions.values() if position.is_exchange_account()]
    assert len(lighter_positions) == 1 + tutorial.config.sub_account_count, "Expected one position per Lighter account"


def subscribe(tutorial: Tutorial, **start_overrides: str) -> None:
    """Subscribe to Lagoon and run one ``start`` cycle, which settles the subscription.

    The deposit request is recorded in ``progress.json`` and checked onchain
    first, so a resumed run settles and claims the earlier request instead of
    subscribing twice.
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
    if pending_raw > 0:
        run_start(tutorial, **start_overrides)
    if vault.vault_contract.functions.maxDeposit(deployer).call() > 0:
        broadcast_and_wait(tutorial, vault.finalise_deposit(deployer))


def redeem_all_shares(tutorial: Tutorial, drain_cycles: int, **start_overrides: str) -> None:
    """Request a full redemption, let ``start`` settle it and claim with ``lagoon-redeem``.

    :param tutorial:
        Tutorial runtime.
    :param drain_cycles:
        ``start`` cycles to run after the request; the strategy drains the
        Lighter accounts and Lagoon settles the redemption during them.
    """
    vault = get_vault(tutorial)
    deployer = tutorial.deployer.address
    if not tutorial.progress.get("redemption_requested") and vault.vault_contract.functions.pendingRedeemRequest(0, deployer).call() == 0 and vault.vault_contract.functions.maxRedeem(deployer).call() == 0:
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


def refund(tutorial: Tutorial, extra_fees_usdc: Decimal = Decimal(0)) -> None:
    """Check nothing backs investor claims, sweep the Safe and reconcile balances.

    :param tutorial:
        Tutorial runtime.
    :param extra_fees_usdc:
        Fees paid outside the executor's trades, for example by a trader-made transfer.
    """
    vault = get_vault(tutorial)
    assert vault.share_token.fetch_raw_balance_of(tutorial.deployer.address) == 0
    assert vault.vault_contract.functions.totalSupply().call() == 0, "Vault shares are outstanding; do not sweep"
    assert vault.vault_contract.functions.maxRedeem(tutorial.deployer.address).call() == 0, "Claimable redemption left"
    assert vault.vault_contract.functions.maxDeposit(tutorial.deployer.address).call() == 0, "Claimable deposit left"
    for label, address in (("vault", vault.address), ("silo", vault.silo_address)):
        logger.info("%s USDC balance: %s", label, tutorial.usdc.fetch_balance_of(address))
    for transfer in sweep_safe_to_owner(tutorial.web3, tutorial.progress["safe_address"], tutorial.deployer, [tutorial.usdc]):
        logger.info("Swept %s %s to the deployer: %s", transfer.amount, transfer.symbol, transfer.tx_hash)

    state = State.read_json_file(tutorial.state_file)
    fees = sum((Decimal(trade.other_data.get("funding_fee_usdc", "0")) for trade in state.portfolio.get_all_trades() if trade.is_internal_account_transfer()), Decimal(0))
    fees += extra_fees_usdc
    lighter_dust = sum(fetch_lighter_balances(tutorial).values(), Decimal(0))
    start_usdc = Decimal(tutorial.progress["start_usdc"])
    end_usdc = tutorial.usdc.fetch_balance_of(tutorial.deployer.address)
    gas_eth = Decimal(tutorial.progress["start_eth_wei"] - tutorial.web3.eth.get_balance(tutorial.deployer.address)) / Decimal(10**18)
    unexplained = start_usdc - end_usdc - fees - lighter_dust
    rows = [
        ("Deployer USDC at start", start_usdc),
        ("Deployer USDC at end", end_usdc),
        ("Lighter transfer fees", fees),
        ("USDC left on Lighter (below withdrawal minimum)", lighter_dust),
        ("Unexplained USDC difference", unexplained),
        ("ETH spent on gas", gas_eth),
    ]
    print(tabulate(rows, headers=["Item", "Amount"], tablefmt="simple"))
    assert abs(unexplained) <= tutorial.usdc.convert_to_decimals(MAX_UNEXPLAINED_RAW), f"Unexplained USDC difference {unexplained}"


def describe_lighter_trade(trade: TradeExecution) -> str:
    """Return a one-line human description of a Lighter custody trade.

    :param trade:
        Any trade on an exchange-account position.
    :return:
        What moved, how much and between which accounts.
    """
    account = trade.pair.get_exchange_account_id()
    amount = abs(trade.executed_quantity if trade.is_success() else trade.planned_quantity)
    if trade.is_internal_account_transfer():
        source = trade.other_data.get("funding_source_account_id")
        return f"Transfer {amount} USDC from Lighter account {source} to account {account}"
    if TradeFlag.external_account_transfer in (trade.flags or set()):
        if trade.is_buy():
            return f"Deposit {amount} USDC from the Safe to Lighter account {account}"
        return f"Withdraw {amount} USDC from Lighter account {account} to the Safe"
    if amount == 0:
        return f"Open the position of Lighter account {account}"
    return trade.get_human_description()


def get_lighter_trade_hashes(trade: TradeExecution) -> list[str]:
    """Return the Ethereum transaction hashes and Lighter identifiers of a trade.

    :param trade:
        Any trade on an exchange-account position.
    :return:
        ``label: hash`` lines, Ethereum transactions first.
    """
    lines = [f"ethereum: {tx.tx_hash}" for tx in trade.blockchain_transactions if tx.tx_hash]
    other_data = trade.other_data or {}
    for key, label in (
        ("lighter_deposit_tx_hash", "ethereum deposit"),
        ("lighter_withdrawal_request_id", "withdrawal request"),
        ("lighter_claim_tx_hash", "ethereum claim"),
        ("lighter_transfer_tx_hash", "lighter transfer"),
    ):
        if other_data.get(key):
            lines.append(f"{label}: {other_data[key]}")
    return lines


def print_trade_summary(state: State) -> list[dict[str, Any]]:
    """Print every exchange-account trade with its description, status and hashes.

    :param state:
        Executor state at the end of a tutorial.
    :return:
        The printed rows, in trade id order.
    """
    rows = []
    for trade in sorted(state.portfolio.get_all_trades(), key=lambda trade: trade.trade_id):
        if not trade.pair.is_exchange_account():
            continue
        rows.append({
            "trade": trade.trade_id,
            "position": trade.pair.get_ticker(),
            "description": describe_lighter_trade(trade),
            "status": trade.get_status().value,
            "notes": trade.notes or "",
            "hashes": "\n".join(get_lighter_trade_hashes(trade)) or "-",
        })
    print(tabulate(rows, headers="keys", tablefmt="grid"))
    return rows


def run_phases(tutorial: Tutorial, phases: list[tuple[int, str, Any]]) -> None:
    """Run numbered phases, skipping those an earlier run completed.

    :param tutorial:
        Tutorial runtime.
    :param phases:
        ``(number, title, callable)`` in order; each callable takes the tutorial.
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
