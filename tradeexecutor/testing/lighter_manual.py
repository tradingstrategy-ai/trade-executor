"""Shell helpers for the command-line Lighter sub-account manual test.

``scripts/lagoon/manual-lighter-subaccounts-cli.sh`` drives a Lagoon vault
with a Safe-owned Lighter master account and sub-accounts through
``trade-executor`` commands. This module holds the few actions no command
does, each a subcommand:

.. code-block:: shell

    poetry run python -m tradeexecutor.testing.lighter_manual balances

Every subcommand reads the same environment as the executor commands:
``JSON_RPC_ETHEREUM``, ``PRIVATE_KEY`` (the deployer), and after deployment
``VAULT_ADDRESS``, ``VAULT_ADAPTER_ADDRESS`` and ``STATE_FILE``. A
subcommand writes its result, the value the shell script acts on, to
standard output as a single line, and its human-readable progress and
tables to standard error through logging, so ``$(...)`` in the script
captures only the result while the operator still sees everything.

Nothing here prints a private key or an RPC URL.
"""

import argparse
import json
import logging
import os
import re
import sys
from decimal import ROUND_DOWN, Decimal, InvalidOperation
from pathlib import Path

from eth_defi.erc_4626.vault_protocol.lagoon.vault import LagoonVault
from eth_defi.hotwallet import HotWallet
from eth_defi.lighter.constants import LIGHTER_USDC_ETHEREUM
from eth_defi.lighter.session import create_lighter_session
from eth_defi.lighter.valuation import LighterEquity
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.provider.receipt import wait_for_transaction_receipt_robust
from eth_defi.safe.sweep import SafeSweepTransfer, sweep_safe_to_owner
from eth_defi.token import TokenDetails, fetch_erc20_details
from eth_defi.vault.base import VaultSpec
from tabulate import tabulate
from web3 import Web3
from web3.contract.contract import ContractFunction

from tradeexecutor.exchange_account import lighter as lighter_reads
from tradeexecutor.exchange_account.lighter import LIGHTER_DUST_USDC
from tradeexecutor.exchange_account.sub_account import get_lighter_account_resolution
from tradeexecutor.state.state import State
from tradeexecutor.testing.lighter_report import get_lighter_funding_fees

#: Fixed name: run with ``python -m`` the module is ``__main__``, which would drop its INFO progress.
logger = logging.getLogger("tradeexecutor.testing.lighter_manual")

#: Lagoon ERC-7540 deposit-request identifier.
LAGOON_DEPOSIT_REQUEST_ID = 0

#: Gas limit for the deployer's approval, deposit request and claim.
DEPLOYER_TRANSACTION_GAS_LIMIT = 1_000_000

#: Largest unexplained change of the deployer's USDC the reconciliation accepts, in raw units:
#: Lagoon cannot return sub-unit rounding residue.
MAX_UNEXPLAINED_RAW = 2

#: Default tolerance of ``expect-balances``, in USDC.
DEFAULT_TOLERANCE_USDC = Decimal("0.01")

#: ``name=value`` or ``name<=value`` in ``expect-balances``.
EXPECTATION_PATTERN = re.compile(r"^(?P<name>[A-Za-z0-9_-]+)(?P<operator><=|=)(?P<value>[0-9.]+)$")

#: USDC precision used for amounts passed back to the shell.
USDC_QUANTUM = Decimal("0.000001")


class ManualTestError(RuntimeError):
    """A check of the manual test failed; the message says what to do."""


def configure_logging() -> None:
    """Log progress to standard error, keeping credential-bearing provider logs quiet.

    Standard output is reserved for subcommand results the shell captures.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.WARNING)
    logging.getLogger("tradeexecutor.testing").setLevel(logging.INFO)
    for namespace in ("web3", "urllib3", "eth_defi.provider"):
        logging.getLogger(namespace).setLevel(logging.WARNING)


def require_env(name: str) -> str:
    """Read required configuration without echoing its value."""
    value = os.environ.get(name)
    if not value:
        raise ManualTestError(f"{name} is not set")
    return value


def parse_usdc(value: str) -> Decimal:
    """Parse a non-negative, finite USDC amount given on the command line."""
    try:
        amount = Decimal(value)
    except InvalidOperation as error:
        raise ManualTestError(f"Not a USDC amount: {value!r}") from error
    if not amount.is_finite() or amount < 0:
        raise ManualTestError(f"USDC amount must be non-negative and finite: {value!r}")
    return amount


def format_usdc(amount: Decimal) -> str:
    """Format USDC for the shell, rounded down to 6 decimals, without exponent notation."""
    return str(amount.quantize(USDC_QUANTUM, rounding=ROUND_DOWN))


def connect() -> tuple[Web3, HotWallet, TokenDetails]:
    """Connect to Ethereum and load the deployer wallet and native USDC."""
    web3 = create_multi_provider_web3(require_env("JSON_RPC_ETHEREUM"), default_http_timeout=(3.0, 180.0))
    deployer = HotWallet.from_private_key(require_env("PRIVATE_KEY"))
    usdc = fetch_erc20_details(web3, LIGHTER_USDC_ETHEREUM)
    return web3, deployer, usdc


def get_vault(web3: Web3) -> LagoonVault:
    """Load the deployed vault from ``VAULT_ADDRESS`` and ``VAULT_ADAPTER_ADDRESS``."""
    return LagoonVault(
        web3,
        VaultSpec(web3.eth.chain_id, require_env("VAULT_ADDRESS")),
        trading_strategy_module_address=require_env("VAULT_ADAPTER_ADDRESS"),
        default_block_identifier="latest",
        require_denomination_token=True,
    )


def broadcast_and_wait(web3: Web3, deployer: HotWallet, function: ContractFunction, description: str) -> None:
    """Broadcast one deployer transaction and wait until every RPC node sees it."""
    deployer.sync_nonce(web3)
    logger.info("Broadcasting: %s", description)
    tx_hash = deployer.transact_and_broadcast_with_contract(function, gas_limit=DEPLOYER_TRANSACTION_GAS_LIMIT)
    receipt = wait_for_transaction_receipt_robust(web3, tx_hash)
    if receipt["status"] != 1:
        raise ManualTestError(f"Transaction reverted: {description} {Web3.to_hex(tx_hash)}")
    logger.info("Confirmed: %s %s", description, Web3.to_hex(tx_hash))


def get_account_labels() -> dict[str, int]:
    """Return each Lighter account id, ``master`` and the sub-account ids, mapped to its index from the state."""
    state = State.read_json_file(Path(require_env("STATE_FILE")))
    resolution = get_lighter_account_resolution(state)
    if not resolution:
        raise ManualTestError("The state records no Lighter accounts; run trade-executor init first")
    return resolution


def fetch_lighter_equity(safe_address: str) -> dict[int, LighterEquity]:
    """Read every Lighter account a Safe owns from one public snapshot, keyed by account index."""
    session = create_lighter_session()
    try:
        return lighter_reads.fetch_lighter_accounts_equity(session, safe_address)
    finally:
        session.close()


def fetch_balances(web3: Web3, deployer: HotWallet, usdc: TokenDetails, vault: LagoonVault, accounts: dict[str, int]) -> dict[str, dict[str, Decimal]]:
    """Read the deployer's, the Safe's and every Lighter account's balances.

    :return:
        Name to ``{"equity": ..., "free": ...}`` for the Lighter accounts and
        ``{"usdc": ...}`` for the Safe and the deployer, plus the deployer's ``eth``.
    """
    equity = fetch_lighter_equity(vault.safe_address)
    balances: dict[str, dict[str, Decimal]] = {
        "deployer": {"usdc": usdc.fetch_balance_of(deployer.address), "eth": Decimal(web3.eth.get_balance(deployer.address)) / Decimal(10**18)},
        "safe": {"usdc": usdc.fetch_balance_of(vault.safe_address)},
    }
    for name, index in accounts.items():
        account = equity.get(index)
        balances[name] = {
            "equity": account.get_total() if account is not None else Decimal(0),
            "free": account.available_balance if account is not None else Decimal(0),
        }
    return balances


def log_balances(balances: dict[str, dict[str, Decimal]], accounts: dict[str, int]) -> None:
    """Log the balances as a table."""
    rows = [
        ("deployer", "", balances["deployer"]["usdc"], f"{balances['deployer']['eth']:.6f} ETH"),
        ("safe", "", balances["safe"]["usdc"], ""),
    ]
    for name, index in accounts.items():
        rows.append((name, index, balances[name]["equity"], f"free {balances[name]['free']}"))
    logger.info("Balances (USDC):\n%s", tabulate(rows, headers=["Holder", "Lighter index", "USDC", "Notes"], tablefmt="simple"))


def command_check_funds(args: argparse.Namespace) -> int:
    """Check the deployer can pay for the run and record its starting balances."""
    web3, deployer, usdc = connect()
    start_file: Path = args.start_file
    if start_file.exists():
        logger.info("%s already records the starting balances of this run; not overwriting it", start_file)
        print("recorded")
        return 0
    usdc_balance = usdc.fetch_balance_of(deployer.address)
    eth_wei = web3.eth.get_balance(deployer.address)
    logger.info("Deployer %s holds %s USDC and %s ETH", deployer.address, usdc_balance, Decimal(eth_wei) / Decimal(10**18))
    if eth_wei == 0:
        raise ManualTestError(f"Deployer {deployer.address} has no ETH for gas")
    if usdc_balance < args.usdc:
        raise ManualTestError(f"Deployer {deployer.address} has {usdc_balance} USDC but the run needs {args.usdc} USDC")
    start_file.write_text(f"START_USDC={usdc_balance}\nSTART_ETH_WEI={eth_wei}\nDEPLOYER_ADDRESS={deployer.address}\n")
    logger.info("Recorded the starting balances in %s", start_file)
    print("ok")
    return 0


def command_request_deposit(args: argparse.Namespace) -> int:
    """Request a Lagoon deposit once per run, and report where the subscription stands.

    The marker file records that the request was made, so a rerun never
    subscribes twice, even after the request has been settled and stopped
    being pending.
    """
    web3, deployer, usdc = connect()
    vault = get_vault(web3)
    marker: Path = args.marker
    pending = vault.vault_contract.functions.pendingDepositRequest(LAGOON_DEPOSIT_REQUEST_ID, deployer.address).call()
    claimable = vault.vault_contract.functions.maxDeposit(deployer.address).call()
    if not marker.exists():
        if pending == 0:
            amount_raw = usdc.convert_to_raw(args.amount)
            if usdc.fetch_raw_balance_of(deployer.address) < amount_raw:
                raise ManualTestError(f"Deployer {deployer.address} holds less than {args.amount} USDC")
            broadcast_and_wait(web3, deployer, usdc.approve(vault.address, args.amount), f"approve {args.amount} USDC to the vault")
            broadcast_and_wait(web3, deployer, vault.request_deposit(deployer.address, amount_raw), f"request a {args.amount} USDC deposit")
            marker.write_text(f"{args.amount}\n")
            logger.info("Requested a %s USDC deposit; it waits in the vault's silo until settled", args.amount)
            print("requested")
            return 0
        # A request made by an interrupted earlier run; record it, do not repeat it.
        marker.write_text(f"{args.amount}\n")
    if pending > 0:
        logger.info("A %s USDC deposit request is pending settlement", usdc.convert_to_decimals(pending))
        print("pending")
    elif claimable > 0:
        logger.info("A settled deposit of %s USDC is claimable", usdc.convert_to_decimals(claimable))
        print("claimable")
    else:
        logger.info("The deposit was requested, settled and claimed earlier")
        print("settled")
    return 0


def command_claim_shares(args: argparse.Namespace) -> int:
    """Claim the deployer's settled deposit as shares; a no-op when nothing is claimable."""
    web3, deployer, _usdc = connect()
    vault = get_vault(web3)
    claimable = vault.vault_contract.functions.maxDeposit(deployer.address).call()
    if claimable > 0:
        broadcast_and_wait(web3, deployer, vault.finalise_deposit(deployer.address), "claim the settled deposit as shares")
    elif vault.vault_contract.functions.pendingDepositRequest(LAGOON_DEPOSIT_REQUEST_ID, deployer.address).call() > 0:
        raise ManualTestError("The deposit request is not settled yet; run trade-executor lagoon-settle first")
    shares = vault.share_token.fetch_balance_of(deployer.address)
    logger.info("Deployer holds %s %s shares", shares, vault.share_token.symbol)
    if shares <= 0:
        raise ManualTestError("The deployer holds no shares after the claim")
    print(f"held {shares}")
    return 0


def command_balances(args: argparse.Namespace) -> int:
    """Log the deployer's, the Safe's and every Lighter account's balances."""
    web3, deployer, usdc = connect()
    vault = get_vault(web3)
    accounts = get_account_labels()
    balances = fetch_balances(web3, deployer, usdc, vault, accounts)
    log_balances(balances, accounts)
    if args.json:
        print(json.dumps({name: {key: str(value) for key, value in values.items()} for name, values in balances.items()}))
    return 0


def parse_expectations(values: list[str]) -> list[tuple[str, str, Decimal]]:
    """Parse ``name=value`` and ``name<=value`` balance expectations.

    :return:
        ``(name, operator, value)`` in the order given.
    """
    expectations = []
    for value in values:
        match = EXPECTATION_PATTERN.match(value)
        if match is None:
            raise ManualTestError(f"Expected name=USDC or name<=USDC, got {value!r}")
        expectations.append((match["name"], match["operator"], parse_usdc(match["value"])))
    return expectations


def check_expectations(expectations: list[tuple[str, str, Decimal]], observed: dict[str, Decimal], tolerance: Decimal) -> list[tuple[str, str, Decimal, Decimal, bool]]:
    """Compare observed balances to expectations.

    :param expectations:
        From :py:func:`parse_expectations`.
    :param observed:
        Name to observed USDC: the Safe's balance or a Lighter account's equity.
    :param tolerance:
        Allowed difference for ``=``.
    :return:
        ``(name, operator, expected, observed, ok)`` per expectation.
    """
    rows = []
    for name, operator, expected in expectations:
        if name not in observed:
            raise ManualTestError(f"Unknown balance {name!r}; expected one of {sorted(observed)}")
        value = observed[name]
        ok = value <= expected + tolerance if operator == "<=" else abs(value - expected) <= tolerance
        rows.append((name, operator, expected, value, ok))
    return rows


def command_expect_balances(args: argparse.Namespace) -> int:
    """Check the Safe's USDC and the Lighter accounts' equity against expectations."""
    expectations = parse_expectations(args.expectations)
    web3, deployer, usdc = connect()
    vault = get_vault(web3)
    accounts = get_account_labels()
    balances = fetch_balances(web3, deployer, usdc, vault, accounts)
    observed = {"safe": balances["safe"]["usdc"], **{name: balances[name]["equity"] for name in accounts}}
    rows = check_expectations(expectations, observed, args.tolerance)
    ok = all(row[4] for row in rows)
    table = tabulate([(name, f"{operator} {expected}", value, "ok" if row_ok else "DIFFERS") for name, operator, expected, value, row_ok in rows], headers=["Holder", "Expected", "Observed", ""], tablefmt="simple")
    logger.log(logging.INFO if ok or args.quiet else logging.WARNING, "Balance check %s:\n%s", "passed" if ok else "failed", table)
    print("ok" if ok else "mismatch")
    return 0 if ok else 1


def has_pending_lighter_transfer(state: State, account_index: int) -> bool:
    """Check whether the state has an unfinished custody transfer on one Lighter account."""
    for position in state.portfolio.get_all_positions():
        if position.pair.is_exchange_account() and int(position.pair.get_exchange_account_id()) == account_index:
            if any(trade.is_exchange_account_transfer_pending() for trade in position.trades.values()):
                return True
    return False


def command_withdrawal_plan(args: argparse.Namespace) -> int:
    """Decide how to withdraw one Lighter account: resume, withdraw an amount, or skip.

    A withdrawal ``lighter-move-funds`` submitted earlier is persisted as a
    pending trade, and the account's equity already reads zero while the
    USDC waits on the ZkLighter contract. Equity alone would therefore skip
    it; the pending trade decides first.

    Prints ``resume``, ``withdraw <amount>`` or ``skip``.
    """
    state = State.read_json_file(Path(require_env("STATE_FILE")))
    if has_pending_lighter_transfer(state, args.account_index):
        logger.info("Lighter account %d has a withdrawal in flight; lighter-move-funds resumes and claims it", args.account_index)
        print("resume")
        return 0
    web3, _deployer, _usdc = connect()
    account = fetch_lighter_equity(get_vault(web3).safe_address).get(args.account_index)
    available = account.available_balance if account is not None else Decimal(0)
    if available < LIGHTER_DUST_USDC:
        logger.info("Lighter account %d has %s USDC free, below the %s USDC withdrawal minimum; nothing to withdraw", args.account_index, available, LIGHTER_DUST_USDC)
        print("skip")
        return 0
    logger.info("Lighter account %d has %s USDC free to withdraw", args.account_index, available)
    print(f"withdraw {format_usdc(available)}")
    return 0


def command_redeem_status(args: argparse.Namespace) -> int:
    """Print ``redeem`` while the deployer has shares or a redemption in progress, ``done`` otherwise."""
    web3, deployer, _usdc = connect()
    vault = get_vault(web3)
    shares = vault.share_token.fetch_raw_balance_of(deployer.address)
    pending = vault.vault_contract.functions.pendingRedeemRequest(LAGOON_DEPOSIT_REQUEST_ID, deployer.address).call()
    claimable = vault.vault_contract.functions.maxRedeem(deployer.address).call()
    logger.info("Deployer shares %s raw, redemption pending %s raw, claimable %s raw", shares, pending, claimable)
    print("redeem" if shares or pending or claimable else "done")
    return 0


def sweep_safe_to_deployer(web3: Web3, vault: LagoonVault, deployer: HotWallet, usdc: TokenDetails) -> list[SafeSweepTransfer]:
    """Move everything left in the Safe to the deployer, once no investor claim remains.

    The deployer must be the Safe's sole owner, as at deployment.

    :return:
        One transfer per swept token, empty when the Safe held nothing.
    :raise ManualTestError:
        If shares are outstanding or a deposit or redemption is still claimable.
    """
    if vault.vault_contract.functions.totalSupply().call() != 0:
        raise ManualTestError("Vault shares are outstanding; redeem them with trade-executor lagoon-redeem before sweeping")
    if vault.vault_contract.functions.maxRedeem(deployer.address).call() != 0:
        raise ManualTestError("A redemption is claimable; claim it with trade-executor lagoon-redeem before sweeping")
    if vault.vault_contract.functions.maxDeposit(deployer.address).call() != 0:
        raise ManualTestError("A deposit is claimable; claim it before sweeping")
    transfers = sweep_safe_to_owner(web3, vault.safe_address, deployer, [usdc])
    for transfer in transfers:
        logger.info("Swept %s %s from the Safe to the deployer: %s", transfer.amount, transfer.symbol, transfer.tx_hash)
    if not transfers:
        logger.info("The Safe holds nothing to sweep")
    return transfers


def command_sweep_safe(args: argparse.Namespace) -> int:
    """Move everything left in the Safe to the deployer."""
    web3, deployer, usdc = connect()
    transfers = sweep_safe_to_deployer(web3, get_vault(web3), deployer, usdc)
    print(f"swept {len(transfers)}")
    return 0


def read_start_file(path: Path) -> dict[str, str]:
    """Read the ``KEY=value`` lines written by ``check-funds``."""
    if not path.exists():
        raise ManualTestError(f"{path} is missing; it is written by check-funds in step 0")
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def reconcile_deployer(
    web3: Web3,
    deployer: HotWallet,
    usdc: TokenDetails,
    start_usdc: Decimal,
    start_eth_wei: int,
    fees: Decimal,
    lighter_dust: Decimal,
) -> list[tuple[str, Decimal]]:
    """Explain the deployer's USDC and ETH change over a run, failing on anything unexplained.

    After a full unwind the deployer has paid only Lighter's transfer fees,
    left only Lighter dust below the withdrawal minimum, and spent gas.

    :param start_usdc:
        Deployer USDC before the run.
    :param start_eth_wei:
        Deployer ETH before the run, in wei.
    :param fees:
        Lighter transfer fees paid during the run.
    :param lighter_dust:
        USDC still on the Safe's Lighter accounts.
    :return:
        The reconciliation rows, also logged as a table.
    :raise ManualTestError:
        If the unexplained USDC difference is larger than :py:data:`MAX_UNEXPLAINED_RAW`.
    """
    end_usdc = usdc.fetch_balance_of(deployer.address)
    gas_eth = Decimal(start_eth_wei - web3.eth.get_balance(deployer.address)) / Decimal(10**18)
    unexplained = start_usdc - end_usdc - fees - lighter_dust
    rows = [
        ("Deployer USDC at start", start_usdc),
        ("Deployer USDC at end", end_usdc),
        ("Lighter transfer fees", fees),
        ("USDC left on Lighter (below withdrawal minimum)", lighter_dust),
        ("Unexplained USDC difference", unexplained),
        ("ETH spent on gas", gas_eth),
    ]
    logger.info("Reconciliation:\n%s", tabulate(rows, headers=["Item", "Amount"], tablefmt="simple"))
    if abs(unexplained) > usdc.convert_to_decimals(MAX_UNEXPLAINED_RAW):
        raise ManualTestError(f"Unexplained USDC difference {unexplained}; inspect the state's trades and balance updates")
    return rows


def command_reconcile(args: argparse.Namespace) -> int:
    """Explain the deployer's USDC and ETH change since ``check-funds``."""
    web3, deployer, usdc = connect()
    vault = get_vault(web3)
    start = read_start_file(args.start_file)
    accounts = get_account_labels()
    state = State.read_json_file(Path(require_env("STATE_FILE")))
    equity = fetch_lighter_equity(vault.safe_address)
    lighter_dust = sum((equity[index].get_total() for index in accounts.values() if index in equity), Decimal(0))
    reconcile_deployer(web3, deployer, usdc, Decimal(start["START_USDC"]), int(start["START_ETH_WEI"]), get_lighter_funding_fees(state), lighter_dust)
    print("ok")
    return 0


def create_parser() -> argparse.ArgumentParser:
    """Build the command-line parser of the subcommands."""
    parser = argparse.ArgumentParser(prog="python -m tradeexecutor.testing.lighter_manual", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    check_funds = commands.add_parser("check-funds", help="Check the deployer's ETH and USDC and record the starting balances.")
    check_funds.add_argument("--usdc", type=parse_usdc, required=True, help="USDC the run needs.")
    check_funds.add_argument("--start-file", type=Path, required=True, help="File to record the starting balances in.")
    check_funds.set_defaults(handler=command_check_funds)

    request_deposit = commands.add_parser("request-deposit", help="Request a Lagoon deposit once per run.")
    request_deposit.add_argument("amount", type=parse_usdc, help="USDC to subscribe.")
    request_deposit.add_argument("--marker", type=Path, required=True, help="File recording that the request was made.")
    request_deposit.set_defaults(handler=command_request_deposit)

    claim_shares = commands.add_parser("claim-shares", help="Claim the settled deposit as shares.")
    claim_shares.set_defaults(handler=command_claim_shares)

    balances = commands.add_parser("balances", help="Show the deployer's, the Safe's and every Lighter account's balances.")
    balances.add_argument("--json", action="store_true", help="Also print the balances as JSON on standard output.")
    balances.set_defaults(handler=command_balances)

    expect_balances = commands.add_parser("expect-balances", help="Check balances, for example safe=6 master=1 trend<=1.")
    expect_balances.add_argument("expectations", nargs="+", help="name=USDC or name<=USDC; names are safe, master and the sub-account ids.")
    expect_balances.add_argument("--tolerance", type=parse_usdc, default=DEFAULT_TOLERANCE_USDC, help="Allowed difference in USDC.")
    expect_balances.add_argument("--quiet", action="store_true", help="Log a mismatch at INFO, for probing checks.")
    expect_balances.set_defaults(handler=command_expect_balances)

    withdrawal_plan = commands.add_parser("withdrawal-plan", help="Print resume, withdraw <amount> or skip for one Lighter account.")
    withdrawal_plan.add_argument("account_index", type=int, help="Lighter account index.")
    withdrawal_plan.set_defaults(handler=command_withdrawal_plan)

    redeem_status = commands.add_parser("redeem-status", help="Print redeem while the deployer has shares to redeem, done otherwise.")
    redeem_status.set_defaults(handler=command_redeem_status)

    sweep_safe = commands.add_parser("sweep-safe", help="Move the Safe's USDC and ETH to the deployer.")
    sweep_safe.set_defaults(handler=command_sweep_safe)

    reconcile = commands.add_parser("reconcile", help="Explain the deployer's balance change over the run.")
    reconcile.add_argument("--start-file", type=Path, required=True, help="File written by check-funds.")
    reconcile.set_defaults(handler=command_reconcile)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one subcommand.

    :param argv:
        Arguments without the program name; the process arguments by default.
    :return:
        Process exit code: 0 on success, 1 on a failed check.
    """
    configure_logging()
    args = create_parser().parse_args(argv)
    try:
        return args.handler(args)
    except ManualTestError as error:
        logger.error("%s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
