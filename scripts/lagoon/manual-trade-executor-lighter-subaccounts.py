"""Tutorial: run a Lagoon vault with a Lighter master account and two sub-accounts.

This live Ethereum mainnet tutorial exercises the multi-account custody
lifecycle with real money:

1. Deploy a Lagoon vault whose Safe owns a Lighter master account and two
   sub-accounts, each sub-account with its own API key for an external trader
2. Initialise executor state with one position per Lighter account
3. Subscribe to the vault and let ``start`` deposit the idle Safe cash to the master
4. Fund both sub-accounts from the master
5. Withdraw sub-account 2 to the Safe with the key-free L1 ``withdraw``
6. Redeposit the withdrawal to the master
7. Fund sub-account 1 from the master
8. Act as the external trader: transfer USDC from sub-account 1 to
   sub-account 2 with sub-account 1's own API key, and check that the next
   ``start`` cycle revalues both positions
9. Request a full redemption, let ``start`` drain every account to the Safe,
   and redeem all shares with ``lagoon-redeem``
10. Sweep the remaining Safe USDC and ETH back to the deployer and print a
    reconciliation table

Every executor custody movement runs through the real ``start`` command and
the production ``decide_trades()`` helper, with
``strategies/test_only/lighter_multi_subaccount_strategy.py``. No orders are
placed. The executor holds only the master key, so it cannot make the
trader's sub-account to sub-account transfer; it only revalues the positions.

The tutorial ran end to end on 2026-10-26 before step 8 was added; step 8
has not yet run live. Lighter's sequencer and withdrawal proofs cannot be
simulated by Anvil, so this script supports Ethereum mainnet only. A master
secure withdrawal takes about 20 minutes and a sub-account L1 withdrawal took
about 29 minutes.

Funding: the deployer needs 5 USDC plus mainnet gas ETH (about 0.011 ETH in
the 2026-10-26 run). Deployment subscribes 1 USDC and moves it to the master
account; the tutorial subscribes 4 USDC more. Lighter's transfer, secure
withdrawal and direct deposit minimums are 1 USDC. The strategy budgets the
1 USDC fee that Lighter's public endpoint quotes on top of each master to
sub-account transfer, so every movement still reaches the minimum if it is
charged. Lighter charged nothing for same-owner transfers in the 2026-10-26
run and all 5 USDC returned to the deployer.

Secrets: the master API key stays in the mode-0600 operator record written by
deployment, and each sub-account key in its mode-0600 hand-over file.
Deployment writes these records only after it completes, so a failure after
key registration would lose the keys. The deploy step therefore also saves
each key to its own mode-0600 temporary file as soon as it is generated. A
failed deployment keeps these files so the operator can recover the
registered keys; a successful one deletes them. Paths are logged, never the
keys.

Resuming: the run directory holds ``progress.json`` with the public
deployment details and the last completed step. Rerun with the same
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
    USDC subscribed after deployment. Defaults to and must be at least 4.

``LIGHTER_API_KEY_INDEX``
    API-key slot for the master and the sub-account keys. Defaults to 4.

``LIGHTER_RUN_DIR``
    Private working directory. Defaults below ``~/.tradingstrategy/examples``.

``LIGHTER_WITHDRAW_TIMEOUT`` / ``LIGHTER_TRANSFER_TIMEOUT``
    Maximum seconds to wait for a withdrawal and a sub-account transfer.
    Default 3600 and 300.
"""

import asyncio
import json
import logging
import os
import tempfile
import time
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
from eth_defi.lighter.api import fetch_lighter_accounts
from eth_defi.lighter.api_key import LighterApiKey
from eth_defi.lighter.constants import LIGHTER_ETHEREUM_DEPLOYMENT_CHAIN_ID, LIGHTER_USDC_ETHEREUM
from eth_defi.lighter.pubkey import MIN_API_KEY_INDEX
from eth_defi.lighter.sdk import create_lighter_signer_client
from eth_defi.lighter.transfer import create_lighter_auth_token_manager, fetch_lighter_transfer_fee, request_lighter_transfer, wait_for_lighter_transfer
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

#: Test strategies used by the deployment and by every later command.
STRATEGY_DIR = Path(__file__).resolve().parents[2] / "strategies" / "test_only"
#: Number of sub-accounts the tutorial creates.
SUB_ACCOUNT_COUNT = 2
#: USDC the strategy keeps in the Safe; none, so all capital moves to Lighter.
SAFE_BUFFER_USDC = Decimal(0)
#: Fee budgeted on top of each master to sub-account transfer: the public
#: ``transferFeeInfo`` quote. The authenticated quote, and the fee charged
#: between same-owner accounts, was 0 on 2026-10-26.
TRANSFER_FEE_USDC = Decimal(1)
#: Smallest subscription for which every movement reaches Lighter's 1 USDC
#: minimum even when the budgeted fee is charged. With the 1 USDC bootstrap
#: the master holds 5 USDC: two 1 USDC fundings and their fees leave 1 USDC,
#: the redeposited 1 USDC withdrawal makes 2 USDC, and the last funding moves
#: 1 USDC plus its fee.
MIN_SUBSCRIPTION_USDC = Decimal(4)
#: Funding weights: a fifth of the capital, 1 USDC at the minimum, per sub-account.
FUNDING_WEIGHTS = "0.2,0.2"
#: Rebalance weights: drain sub-account 2 and move its capital to sub-account 1.
REBALANCE_WEIGHTS = "1,0"
#: USDC the trader moves from sub-account 1 to sub-account 2, Lighter's transfer minimum.
TRADER_TRANSFER_USDC = Decimal(1)
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
    #: Maximum wait for a withdrawal to become claimable.
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
        """Return the public vault record path."""
        return self.config.run_dir / "vault-record.txt"

    @property
    def operator_record_file(self) -> Path:
        """Return the mode-0600 operator record holding the master key."""
        return self.record_file.with_suffix(".json")

    def get_handover_file(self, account_index: int) -> Path:
        """Return the mode-0600 hand-over file holding one sub-account's key."""
        return self.record_file.with_name(f"{self.record_file.stem}.nautilus.{account_index}.json")

    def write_progress(self) -> None:
        """Persist progress, including values saved within a phase."""
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
    config = TutorialConfig(
        rpc_url=require_env("JSON_RPC_ETHEREUM"),
        deployer_private_key=require_env("LIGHTER_TEST_PRIVATE_KEY", "PRIVATE_KEY"),
        run_dir=run_dir,
        subscription_usdc=Decimal(os.environ.get("LIGHTER_SUBSCRIPTION_USDC", str(MIN_SUBSCRIPTION_USDC))),
        api_key_index=int(os.environ.get("LIGHTER_API_KEY_INDEX", str(MIN_API_KEY_INDEX))),
        withdraw_timeout=int(os.environ.get("LIGHTER_WITHDRAW_TIMEOUT", "3600")),
        transfer_timeout=int(os.environ.get("LIGHTER_TRANSFER_TIMEOUT", "300")),
    )
    if config.subscription_usdc < MIN_SUBSCRIPTION_USDC:
        raise RuntimeError(f"LIGHTER_SUBSCRIPTION_USDC must be at least {MIN_SUBSCRIPTION_USDC} for every movement to reach Lighter's 1 USDC minimum")
    return config


def prepare_tutorial(config: TutorialConfig) -> Tutorial:
    """Connect to Ethereum, load earlier progress or check the deployer's funds."""
    web3 = create_multi_provider_web3(config.rpc_url, default_http_timeout=(3.0, 180.0))
    assert web3.eth.chain_id == LIGHTER_ETHEREUM_DEPLOYMENT_CHAIN_ID, "Ethereum mainnet RPC required"
    deployer = HotWallet.from_private_key(config.deployer_private_key)
    usdc = fetch_erc20_details(web3, LIGHTER_USDC_ETHEREUM)
    config.run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    tutorial = Tutorial(config=config, web3=web3, deployer=deployer, usdc=usdc, progress={})
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
        "ID": "lighter-subaccounts",
        "NAME": "Lighter sub-accounts tutorial",
        "STRATEGY_FILE": str(STRATEGY_DIR / "lighter_multi_subaccount_strategy.py"),
        "STATE_FILE": str(tutorial.state_file),
        "PRIVATE_KEY": tutorial.config.deployer_private_key,
        "JSON_RPC_ETHEREUM": tutorial.config.rpc_url,
        "ASSET_MANAGEMENT_MODE": "lagoon",
        "VAULT_ADDRESS": tutorial.progress["vault_address"],
        "VAULT_ADAPTER_ADDRESS": tutorial.progress["module_address"],
        "LIGHTER_ACCOUNT_INDEXES": ",".join(str(index) for index in tutorial.progress["account_indices"]),
        "LIGHTER_OPERATOR_RECORD_FILE": str(tutorial.operator_record_file),
        "LIGHTER_WITHDRAWAL_TIMEOUT": str(tutorial.config.withdraw_timeout),
        "LIGHTER_TRANSFER_TIMEOUT": str(tutorial.config.transfer_timeout),
        "LIGHTER_SUB_ACCOUNT_WEIGHTS": "",
        "LIGHTER_SAFE_CASH_BUFFER_USD": str(SAFE_BUFFER_USDC),
        "LIGHTER_TRANSFER_FEE_USD": str(TRANSFER_FEE_USDC),
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


# ---------------------------------------------------------------------------
# Tutorial phases
# ---------------------------------------------------------------------------


def deploy(tutorial: Tutorial) -> None:
    """Deploy the vault with keyed sub-accounts, backing up each key as it is generated."""
    generate_key = lagoon_deployment.generate_lighter_api_key_pair
    saved_keys: list[Path] = []

    def generate_and_save_key(api_key_index: int) -> LighterApiKey:
        key = generate_key(api_key_index)
        saved_keys.append(save_generated_key(key, len(saved_keys) + 1, tutorial.deployer.address))
        logger.warning("Saved generated Lighter API key %d to %s until deployment completes", len(saved_keys), saved_keys[-1])
        return key

    environment = {
        "EXECUTOR_ID": "lighter-subaccounts",
        "STRATEGY_FILE": str(STRATEGY_DIR / "minimal_lighter_strategy.py"),
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
    assert len(lighter_positions) == 1 + SUB_ACCOUNT_COUNT, "Expected one position per Lighter account"


def subscribe(tutorial: Tutorial) -> None:
    """Subscribe to Lagoon; ``start`` settles it and deposits idle Safe cash to the master.

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
        run_start(tutorial)
    if vault.vault_contract.functions.maxDeposit(deployer).call() > 0:
        broadcast_and_wait(tutorial, vault.finalise_deposit(deployer))
    assert tutorial.usdc.fetch_balance_of(tutorial.progress["safe_address"]) == SAFE_BUFFER_USDC


async def submit_trader_transfer(tutorial: Tutorial, handover: dict[str, Any], to_account_index: int) -> None:
    """Sign a sub-account to sub-account transfer with the sending sub-account's own key.

    A submission is not idempotent, so its hash is saved to ``progress.json``
    before waiting; a resumed run confirms the saved hash instead of sending
    the transfer again.
    """
    from_account_index = int(handover["account_index"])
    api_key_index = int(handover["api_key_index"])
    client = create_lighter_signer_client(from_account_index, api_key_index, handover["private_key"])
    try:
        auth = create_lighter_auth_token_manager(client, api_key_index)
        tx_hash = tutorial.progress.get("trader_transfer_tx_hash")
        if tx_hash is None and tutorial.progress.get("trader_transfer_submitting"):
            raise RuntimeError(
                f"A transfer from sub-account {from_account_index} may have been submitted without a recorded hash; "
                f"check its transfer history before rerunning"
            )
        if tx_hash is None:
            tutorial.progress["trader_transfer_submitting"] = True
            tutorial.write_progress()
            fee = await fetch_lighter_transfer_fee(client, auth, to_account_index=to_account_index)
            logger.info("Lighter quotes a %s USDC fee from sub-account %d to sub-account %d", fee, from_account_index, to_account_index)
            tx_hash = await request_lighter_transfer(client, to_account_index=to_account_index, amount=TRADER_TRANSFER_USDC, fee=fee, api_key_index=api_key_index)
            tutorial.progress["trader_transfer_tx_hash"] = tx_hash
            tutorial.write_progress()
        record = await wait_for_lighter_transfer(client, auth, account_index=from_account_index, tx_hash=tx_hash, timeout=tutorial.config.transfer_timeout)
    finally:
        await client.close()
    assert (record.from_account_index, record.to_account_index, record.amount) == (from_account_index, to_account_index, TRADER_TRANSFER_USDC), f"Unexpected transfer record {record}"
    tutorial.progress["trader_transfer_fee_usdc"] = str(record.fee)
    logger.info("Trader transfer %s executed: %s USDC, fee %s USDC", tx_hash, record.amount, record.fee)


def transfer_between_sub_accounts(tutorial: Tutorial) -> None:
    """Move USDC from sub-account 1 to sub-account 2 as the external trader would.

    The executor's positions are stale until the next ``start`` cycle
    revalues each account from Lighter. That cycle has no target weights, so
    it moves no capital.
    """
    _master_index, sub_1, sub_2 = tutorial.progress["account_indices"]
    # The key is read into memory only and never logged.
    handover = json.loads(tutorial.get_handover_file(sub_1).read_text())
    assert int(handover["account_index"]) == sub_1, f"Hand-over file is not for sub-account {sub_1}"

    # Save the balances before submitting, so a resumed run compares against them.
    if "trader_transfer_before" not in tutorial.progress:
        tutorial.progress["trader_transfer_before"] = {str(index): str(value) for index, value in fetch_lighter_balances(tutorial).items()}
        tutorial.write_progress()
    before = {int(index): Decimal(value) for index, value in tutorial.progress["trader_transfer_before"].items()}
    asyncio.run(submit_trader_transfer(tutorial, handover, sub_2))

    run_start(tutorial)
    after = fetch_lighter_balances(tutorial)
    fee = Decimal(tutorial.progress["trader_transfer_fee_usdc"])
    assert after[sub_2] == before[sub_2] + TRADER_TRANSFER_USDC, f"Sub-account {sub_2} did not receive the transfer"
    assert sum(after.values(), Decimal(0)) == sum(before.values(), Decimal(0)) - fee, "The transfer changed total Lighter equity by more than its fee"

    # The executor must now track each account at its Lighter equity.
    state = State.read_json_file(tutorial.state_file)
    tracked = {
        int(position.pair.get_exchange_account_id()): Decimal(str(position.get_quantity()))
        for position in state.portfolio.open_positions.values()
        if position.is_exchange_account()
    }
    for index, value in after.items():
        assert tracked[index] == value, f"Executor tracks account {index} at {tracked[index]}, Lighter reports {value}"


def redeem_everything(tutorial: Tutorial) -> None:
    """Drain every account for a full redemption, then redeem with ``lagoon-redeem``."""
    vault = get_vault(tutorial)
    shares = vault.share_token.fetch_raw_balance_of(tutorial.deployer.address)
    broadcast_and_wait(tutorial, vault.request_redeem(tutorial.deployer.address, shares))
    # Draining every account and settling the redemption can span two cycles.
    run_start(tutorial)
    run_start(tutorial)
    run_cli(["lagoon-redeem"], get_cli_environment(tutorial))
    assert vault.share_token.fetch_raw_balance_of(tutorial.deployer.address) == 0


def refund(tutorial: Tutorial) -> None:
    """Check nothing backs investor claims, sweep the Safe and reconcile balances."""
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
    fees = sum((Decimal(trade.other_data.get("funding_fee_usdc", "0")) for trade in state.portfolio.get_all_trades() if TradeFlag.internal_account_transfer in (trade.flags or set())), Decimal(0))
    fees += Decimal(tutorial.progress.get("trader_transfer_fee_usdc", "0"))
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


def main() -> None:
    """Run the tutorial phases, skipping those completed in an earlier run."""
    configure_logging()
    tutorial = prepare_tutorial(load_config())
    phases = [
        (1, "Deploy the vault with two Lighter sub-accounts", deploy),
        (2, "Initialise executor state with one position per account", initialise_state),
        (3, "Subscribe and deposit idle Safe cash to the master", subscribe),
        (4, "Fund both sub-accounts from the master", lambda t: run_start(t, LIGHTER_SUB_ACCOUNT_WEIGHTS=FUNDING_WEIGHTS)),
        (5, "Withdraw sub-account 2 to the Safe with the key-free L1 path", lambda t: run_start(t, LIGHTER_SUB_ACCOUNT_WEIGHTS=REBALANCE_WEIGHTS)),
        (6, "Redeposit the withdrawal to the master", lambda t: run_start(t, LIGHTER_SUB_ACCOUNT_WEIGHTS=REBALANCE_WEIGHTS)),
        (7, "Fund sub-account 1 from the master", lambda t: run_start(t, LIGHTER_SUB_ACCOUNT_WEIGHTS=REBALANCE_WEIGHTS)),
        (8, "Transfer from sub-account 1 to sub-account 2 as the external trader", transfer_between_sub_accounts),
        (9, "Drain every account and redeem all shares", redeem_everything),
        (10, "Sweep the Safe and reconcile the deployer's balances", refund),
    ]
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


if __name__ == "__main__":
    main()
