"""Tutorial: run a Lagoon vault with a Lighter master account and two sub-accounts.

This live Ethereum mainnet tutorial exercises the multi-account custody
lifecycle with real money:

1. Deploy a Lagoon vault whose Safe owns a Lighter master account and two
   sub-accounts, each sub-account with its own API key for an external trader
2. Initialise executor state with one position per Lighter account
3. Subscribe to the vault and let ``start`` deposit the idle Safe cash to the master
4. Fund both sub-accounts from the master with ``start``
5. Rebalance: withdraw a sub-account to the Safe with the key-free L1
   ``withdraw`` path, redeposit, and fund the other sub-account
6. Request a full redemption, let ``start`` drain every account to the Safe,
   and settle it
7. Redeem all shares with ``lagoon-redeem``
8. Sweep the remaining Safe USDC and ETH back to the deployer and print a
   reconciliation table

Every custody movement runs through the real ``start`` command and the
``decide_trades()`` helper used in production, with
``strategies/test_only/lighter_multi_subaccount_strategy.py``. No orders are
placed: the external trader's role is left empty.

Lighter's sequencer and withdrawal proofs cannot be simulated by Anvil, so
this script supports Ethereum mainnet only. Each secure withdrawal takes
about 20 minutes. Use a funded test account and small values.

Secrets: the master API key stays in the mode-0600 operator record written
by deployment. The sub-account keys stay in their mode-0600 hand-over files.
This script never prints, logs or copies any of them.

Resuming: the run directory holds ``progress.json`` with the public
deployment details and the last completed phase. Rerun with the same
``LIGHTER_RUN_DIR`` to continue. If a ``start`` cycle was interrupted during a
transfer, the next ``start`` refuses to run; recover it with
``trade-executor check-accounts`` and ``trade-executor correct-accounts``
using the environment printed in the log, then rerun the tutorial.

Example
-------

.. code-block:: shell

    source .local-test.env
    export LIGHTER_TEST_PRIVATE_KEY="0x..."
    export JSON_RPC_ETHEREUM="https://..."
    poetry run python scripts/lagoon/manual-trade-executor-lighter-subaccounts.py

Optional environment variables
------------------------------

``LIGHTER_SUBSCRIPTION_USDC``
    USDC subscribed after deployment. Defaults to 40.

``LIGHTER_API_KEY_INDEX``
    API-key slot for the master and the sub-account keys. Defaults to 4.

``LIGHTER_RUN_DIR``
    Private working directory. Defaults below ``~/.tradingstrategy/examples``.

``LIGHTER_WITHDRAW_TIMEOUT`` / ``LIGHTER_TRANSFER_TIMEOUT``
    Maximum seconds to wait for a secure withdrawal and a sub-account transfer.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest import mock

from eth_defi.erc_4626.vault_protocol.lagoon.vault import LagoonVault
from eth_defi.hotwallet import HotWallet
from eth_defi.lighter.api import fetch_lighter_accounts
from eth_defi.lighter.constants import LIGHTER_ETHEREUM_DEPLOYMENT_CHAIN_ID, LIGHTER_USDC_ETHEREUM
from eth_defi.lighter.pubkey import MIN_API_KEY_INDEX
from eth_defi.lighter.session import create_lighter_session
from eth_defi.lighter.valuation import fetch_lighter_accounts_equity
from eth_defi.provider.broken_provider import _latest_delayed_block_number_cache
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.provider.receipt import wait_for_transaction_receipt_robust
from eth_defi.safe.sweep import sweep_safe_to_owner
from eth_defi.token import TokenDetails, fetch_erc20_details
from eth_defi.utils import setup_console_logging
from eth_defi.vault.base import VaultSpec
from tabulate import tabulate
from web3 import Web3
from web3.contract.contract import ContractFunction

from tradeexecutor.cli.main import app
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeFlag

logger = logging.getLogger(__name__)

#: Number of sub-accounts the tutorial creates.
SUB_ACCOUNT_COUNT = 2
#: USDC the strategy keeps in the Safe until the final redemption.
SAFE_BUFFER_USDC = Decimal(20)
#: Lagoon cannot return sub-unit rounding residue; tolerate two raw units.
MAX_UNEXPLAINED_RAW = 2

# ---------------------------------------------------------------------------
# Configuration and helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TutorialConfig:
    """Operator configuration, with secrets excluded from its representation."""

    #: Ethereum RPC endpoint; hidden because a URL may contain credentials.
    rpc_url: str = field(repr=False)
    #: Funded deployer key used for the temporary vault lifecycle.
    deployer_private_key: str = field(repr=False)
    #: Private directory holding this run's artefacts and progress.
    run_dir: Path
    #: USDC subscribed after deployment.
    subscription_usdc: Decimal
    #: API-key slot for the master and sub-account keys.
    api_key_index: int
    #: Maximum wait for a secure withdrawal to become claimable.
    withdraw_timeout: int
    #: Maximum wait for a sub-account transfer to appear in history.
    transfer_timeout: int


@dataclass(slots=True)
class Tutorial:
    """Runtime objects and public progress shared by tutorial phases."""

    #: Operator configuration.
    config: TutorialConfig
    #: Ethereum connection, excluded from repr because its URL may be sensitive.
    web3: Web3 = field(repr=False)
    #: Funded deployer wallet, excluded from repr because it holds a key.
    deployer: HotWallet = field(repr=False)
    #: Native Ethereum USDC.
    usdc: TokenDetails
    #: Public progress persisted in ``progress.json``.
    progress: dict[str, Any]

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
        """Return the public vault record path; its JSON sibling is private."""
        return self.config.run_dir / "vault-record.txt"

    def save_progress(self, phase: int) -> None:
        """Record a completed phase."""
        self.progress["phase"] = phase
        self.progress_file.write_text(json.dumps(self.progress, indent=2))

    def is_done(self, phase: int) -> bool:
        """Check whether a phase completed in an earlier run."""
        return self.progress.get("phase", 0) >= phase


def configure_logging() -> None:
    """Enable tutorial progress while suppressing credential-bearing provider logs."""
    setup_console_logging("INFO")
    logger.setLevel(logging.INFO)
    for namespace in ("web3", "urllib3", "eth_defi.provider"):
        logging.getLogger(namespace).setLevel(logging.WARNING)


def log_step(number: int, title: str) -> None:
    """Print one visible tutorial phase heading."""
    logger.info("%s", "=" * 72)
    logger.info("Step %d: %s", number, title)
    logger.info("%s", "=" * 72)


def require_env(name: str, *fallback_names: str) -> str:
    """Read required configuration without echoing its value."""
    for key in (name, *fallback_names):
        if value := os.environ.get(key):
            return value
    raise RuntimeError(f"{name} is required")


def load_config() -> TutorialConfig:
    """Validate environment variables before creating any onchain state."""
    if os.environ.get("SIMULATE", "").lower() in {"1", "true", "yes"}:
        raise RuntimeError("This tutorial is live Ethereum/Lighter only; SIMULATE is unsupported")
    run_dir = Path(os.environ.get("LIGHTER_RUN_DIR", str(Path("~/.tradingstrategy/examples").expanduser() / f"lighter-subaccounts-{int(time.time())}")))
    return TutorialConfig(
        rpc_url=require_env("JSON_RPC_ETHEREUM"),
        deployer_private_key=require_env("LIGHTER_TEST_PRIVATE_KEY", "PRIVATE_KEY"),
        run_dir=run_dir,
        subscription_usdc=Decimal(os.environ.get("LIGHTER_SUBSCRIPTION_USDC", "40")),
        api_key_index=int(os.environ.get("LIGHTER_API_KEY_INDEX", str(MIN_API_KEY_INDEX))),
        withdraw_timeout=int(os.environ.get("LIGHTER_WITHDRAW_TIMEOUT", "3600")),
        transfer_timeout=int(os.environ.get("LIGHTER_TRANSFER_TIMEOUT", "300")),
    )


def prepare_tutorial(config: TutorialConfig) -> Tutorial:
    """Connect to Ethereum, check deployer funds and load any earlier progress."""
    web3 = create_multi_provider_web3(config.rpc_url, default_http_timeout=(3.0, 180.0))
    assert web3.eth.chain_id == LIGHTER_ETHEREUM_DEPLOYMENT_CHAIN_ID, "Ethereum mainnet RPC required"
    deployer = HotWallet.from_private_key(config.deployer_private_key)
    usdc = fetch_erc20_details(web3, LIGHTER_USDC_ETHEREUM)
    config.run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    progress_file = config.run_dir / "progress.json"
    progress = json.loads(progress_file.read_text()) if progress_file.exists() else {}
    if not progress:
        progress = {
            "start_usdc": str(usdc.fetch_balance_of(deployer.address)),
            "start_eth_wei": web3.eth.get_balance(deployer.address),
        }
    logger.info("Deployer: %s", deployer.address)
    logger.info("Run artefacts: %s", config.run_dir)
    logger.info("Resuming after phase %d", progress.get("phase", 0))
    return Tutorial(config=config, web3=web3, deployer=deployer, usdc=usdc, progress=progress)


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
    strategy_file = Path(__file__).resolve().parents[2] / "strategies/test_only/lighter_multi_subaccount_strategy.py"
    environment = {
        "ID": "lighter-subaccounts",
        "NAME": "Lighter sub-accounts tutorial",
        "STRATEGY_FILE": str(strategy_file),
        "STATE_FILE": str(tutorial.state_file),
        "PRIVATE_KEY": tutorial.config.deployer_private_key,
        "JSON_RPC_ETHEREUM": tutorial.config.rpc_url,
        "ASSET_MANAGEMENT_MODE": "lagoon",
        "VAULT_ADDRESS": tutorial.progress["vault_address"],
        "VAULT_ADAPTER_ADDRESS": tutorial.progress["module_address"],
        "LIGHTER_ACCOUNT_INDEXES": ",".join(str(index) for index in tutorial.progress["account_indices"]),
        "LIGHTER_OPERATOR_RECORD_FILE": str(tutorial.record_file.with_suffix(".json")),
        "LIGHTER_WITHDRAWAL_TIMEOUT": str(tutorial.config.withdraw_timeout),
        "LIGHTER_TRANSFER_TIMEOUT": str(tutorial.config.transfer_timeout),
        "LIGHTER_SUB_ACCOUNT_WEIGHTS": "",
        "LIGHTER_SAFE_CASH_BUFFER_USD": str(SAFE_BUFFER_USDC),
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
        equity = fetch_lighter_accounts_equity(session, tutorial.progress["safe_address"])
    finally:
        session.close()
    return {index: equity[index].get_total() for index in tutorial.progress["account_indices"]}


def run_start(tutorial: Tutorial, **overrides: str) -> None:
    """Run one ``start`` cycle and log the resulting Safe and Lighter balances."""
    run_cli(["start"], get_cli_environment(tutorial, **overrides))
    balances = fetch_lighter_balances(tutorial)
    logger.info("Safe USDC: %s", tutorial.usdc.fetch_balance_of(tutorial.progress["safe_address"]))
    for index, value in balances.items():
        logger.info("Lighter account %d equity: %s", index, value)


# ---------------------------------------------------------------------------
# Tutorial phases
# ---------------------------------------------------------------------------


def deploy(tutorial: Tutorial) -> None:
    """Deploy the vault with two keyed sub-accounts and record public details."""
    minimal_strategy = Path(__file__).resolve().parents[2] / "strategies/test_only/minimal_lighter_strategy.py"
    run_cli(
        ["lagoon-deploy-vault"],
        {
            "EXECUTOR_ID": "lighter-subaccounts",
            "STRATEGY_FILE": str(minimal_strategy),
            "LIGHTER_ACCOUNT_INDEX": "0",
            "STATE_FILE": str(tutorial.state_file),
            "PRIVATE_KEY": tutorial.config.deployer_private_key,
            "JSON_RPC_ETHEREUM": tutorial.config.rpc_url,
            "VAULT_RECORD_FILE": str(tutorial.record_file),
            "FUND_NAME": "Lighter sub-accounts test",
            "FUND_SYMBOL": "LSA",
            "MULTISIG_OWNERS": tutorial.deployer.address,
            "ANY_ASSET": "true",
            "PERFORMANCE_FEE": "0",
            "MANAGEMENT_FEE": "0",
            "UNIT_TESTING": "true",
            "GENERATE_LIGHTER_API_KEY": "true",
            "LIGHTER_API_KEY_INDEX": str(tutorial.config.api_key_index),
            "LIGHTER_SUB_ACCOUNT_COUNT": str(SUB_ACCOUNT_COUNT),
            "LIGHTER_SUB_ACCOUNT_API_KEY_INDEX": str(tutorial.config.api_key_index),
            "LOG_LEVEL": "warning",
        },
    )
    # Read only public fields; the private record is passed to commands by path.
    public = json.loads(tutorial.state_file.with_name("lighter-subaccounts.deployment.json").read_text())["deployments"]["ethereum"]
    setup = public["lighter_account_setup"]
    sub_indices = [int(sub["account_index"]) for sub in setup["sub_accounts"]]
    assert len(sub_indices) == SUB_ACCOUNT_COUNT
    tutorial.progress.update({"vault_address": public["vault_address"], "module_address": public["module_address"]})
    safe_address = get_vault(tutorial).safe_address
    session = create_lighter_session()
    try:
        accounts = fetch_lighter_accounts(session, safe_address)
    finally:
        session.close()
    assert accounts is not None and accounts.master_index == int(setup["account_index"])
    assert set(sub_indices) <= set(accounts.sub_account_indices), "Lighter does not list the created sub-accounts"
    for index in sub_indices:
        handover = tutorial.record_file.with_suffix(".json").with_name(f"{tutorial.record_file.stem}.nautilus.{index}.json")
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
    assert len(lighter_positions) == 1 + SUB_ACCOUNT_COUNT, "Expected one position per Lighter account"


def subscribe(tutorial: Tutorial) -> None:
    """Subscribe to Lagoon; ``start`` settles it and deposits idle Safe cash to the master."""
    vault = get_vault(tutorial)
    amount = tutorial.config.subscription_usdc
    assert tutorial.usdc.fetch_balance_of(tutorial.deployer.address) >= amount, "Deployer lacks USDC"
    broadcast_and_wait(tutorial, tutorial.usdc.approve(vault.address, amount))
    broadcast_and_wait(tutorial, vault.request_deposit(tutorial.deployer.address, tutorial.usdc.convert_to_raw(amount)))
    run_start(tutorial)
    broadcast_and_wait(tutorial, vault.finalise_deposit(tutorial.deployer.address))
    assert tutorial.usdc.fetch_balance_of(tutorial.progress["safe_address"]) == SAFE_BUFFER_USDC


def redeem_everything(tutorial: Tutorial) -> None:
    """Drain every account for a full redemption, then redeem with ``lagoon-redeem``."""
    vault = get_vault(tutorial)
    shares = vault.share_token.fetch_raw_balance_of(tutorial.deployer.address)
    broadcast_and_wait(tutorial, vault.request_redeem(tutorial.deployer.address, shares))
    # The redemption needs every account's capital, so the buffer is released too.
    run_start(tutorial, LIGHTER_SAFE_CASH_BUFFER_USD="0")
    run_start(tutorial, LIGHTER_SAFE_CASH_BUFFER_USD="0")
    run_cli(["lagoon-redeem"], get_cli_environment(tutorial))
    assert vault.share_token.fetch_raw_balance_of(tutorial.deployer.address) == 0


def refund(tutorial: Tutorial) -> None:
    """Check nothing backs investor claims, sweep the Safe and reconcile balances."""
    vault = get_vault(tutorial)
    safe_address = tutorial.progress["safe_address"]
    assert vault.share_token.fetch_raw_balance_of(tutorial.deployer.address) == 0
    assert vault.vault_contract.functions.totalSupply().call() == 0, "Vault shares are outstanding; do not sweep"
    assert vault.vault_contract.functions.maxRedeem(tutorial.deployer.address).call() == 0, "Claimable redemption left"
    assert vault.vault_contract.functions.maxDeposit(tutorial.deployer.address).call() == 0, "Claimable deposit left"
    for label, address in (("vault", vault.address), ("silo", vault.silo_address)):
        logger.info("%s USDC balance: %s", label, tutorial.usdc.fetch_balance_of(address))
    swept = sweep_safe_to_owner(tutorial.web3, safe_address, tutorial.deployer, [tutorial.usdc])
    for transfer in swept:
        logger.info("Swept %s %s to the deployer: %s", transfer.amount, transfer.symbol, transfer.tx_hash)

    state = State.read_json_file(tutorial.state_file)
    fees = sum((Decimal(trade.other_data.get("funding_fee_usdc", "0")) for trade in state.portfolio.get_all_trades() if TradeFlag.internal_account_transfer in (trade.flags or set())), Decimal(0))
    lighter_dust = sum(fetch_lighter_balances(tutorial).values(), Decimal(0))
    start_usdc = Decimal(tutorial.progress["start_usdc"])
    end_usdc = tutorial.usdc.fetch_balance_of(tutorial.deployer.address)
    end_eth_wei = tutorial.web3.eth.get_balance(tutorial.deployer.address)
    gas_eth = Decimal(tutorial.progress["start_eth_wei"] - end_eth_wei) / Decimal(10**18)
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


def main() -> None:
    """Run the tutorial phases, skipping those completed in an earlier run."""
    configure_logging()
    tutorial = prepare_tutorial(load_config())
    phases = [
        (1, "Deploy the vault with two Lighter sub-accounts", deploy),
        (2, "Initialise executor state with one position per account", initialise_state),
        (3, "Subscribe and deposit idle Safe cash to the master", subscribe),
        (4, "Fund both sub-accounts from the master", lambda t: run_start(t, LIGHTER_SUB_ACCOUNT_WEIGHTS="0.5,0.5")),
        (5, "Withdraw sub-account 2 to the Safe with the key-free L1 path", lambda t: run_start(t, LIGHTER_SUB_ACCOUNT_WEIGHTS="0.75,0.25")),
        (6, "Redeposit the withdrawal to the master", lambda t: run_start(t, LIGHTER_SUB_ACCOUNT_WEIGHTS="0.75,0.25")),
        (7, "Fund sub-account 1 from the master", lambda t: run_start(t, LIGHTER_SUB_ACCOUNT_WEIGHTS="0.75,0.25")),
        (8, "Drain every account and redeem all shares", redeem_everything),
        (9, "Sweep the Safe and reconcile the deployer's balances", refund),
    ]
    for number, title, phase in phases:
        if tutorial.is_done(number):
            logger.info("Step %d already completed: %s", number, title)
            continue
        log_step(number, title)
        phase(tutorial)
        tutorial.save_progress(number)
    logger.info("Tutorial complete. Public artefacts and protected records are in %s", tutorial.config.run_dir)


if __name__ == "__main__":
    main()
