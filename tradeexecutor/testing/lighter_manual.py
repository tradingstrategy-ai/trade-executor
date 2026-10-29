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
import asyncio
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
from tradeexecutor.exchange_account.lighter_operator import load_lighter_sub_account_operator_records, request_lighter_funding_transfer, wait_for_lighter_funding_transfer
from tradeexecutor.exchange_account.sub_account import get_lighter_account_resolution
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
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

#: Default least ETH the deployer needs: the comparable tutorial run on
#: 2026-10-26 spent about 0.011 ETH of gas, and lagoon-redeem refuses to
#: start below 0.001 ETH.
DEFAULT_MIN_ETH = Decimal("0.02")

#: USDC precision used for amounts passed back to the shell.
USDC_QUANTUM = Decimal("0.000001")

#: Default wait for a transfer between Lighter accounts to show in the sender's history, in seconds.
DEFAULT_TRANSFER_TIMEOUT = 600


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
    eth = Decimal(eth_wei) / Decimal(10**18)
    logger.info("Deployer %s holds %s USDC and %s ETH", deployer.address, usdc_balance, eth)
    if eth < args.min_eth:
        raise ManualTestError(f"Deployer {deployer.address} has {eth} ETH but the run needs at least {args.min_eth} ETH for gas")
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


def find_pending_lighter_transfer(state: State, account_index: int) -> TradeExecution | None:
    """Return the unfinished custody transfer on one Lighter account, if any."""
    for position in state.portfolio.get_all_positions():
        if position.pair.is_exchange_account() and int(position.pair.get_exchange_account_id()) == account_index:
            for trade in position.trades.values():
                if trade.is_exchange_account_transfer_pending():
                    return trade
    return None


def command_withdrawal_plan(args: argparse.Namespace) -> int:
    """Decide how to withdraw one Lighter account: resume, withdraw an amount, or skip.

    ``lighter-move-funds`` saves a withdrawal as a pending trade before it
    submits the request and records the request id after. A submitted
    withdrawal already reads as zero equity while the USDC waits on the
    ZkLighter contract, so the pending trade decides first:

    - with a recorded request id, ``lighter-move-funds`` resumes it: it waits,
      claims and books it without sending a second request;
    - without one, the command stopped between saving the trade and
      recording the request, and only Lighter's or Ethereum's history can
      tell whether the request was sent. Nothing here may send another.

    Prints ``resume``, ``unconfirmed <trade id>``, ``withdraw <amount>`` or ``skip``.
    """
    state = State.read_json_file(Path(require_env("STATE_FILE")))
    pending = find_pending_lighter_transfer(state, args.account_index)
    if pending is not None:
        if (pending.other_data or {}).get("lighter_withdrawal_request_id"):
            logger.info("Lighter account %d has withdrawal trade #%d in flight; lighter-move-funds resumes and claims it", args.account_index, pending.trade_id)
            print("resume")
        else:
            logger.warning("Lighter account %d has transfer trade #%d started without a recorded request; check the history before anything else", args.account_index, pending.trade_id)
            print(f"unconfirmed {pending.trade_id}")
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
        Lighter transfer fees paid during the run, by the executor and by a trader.
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
    fees = get_lighter_funding_fees(state)
    if args.trader_transfer_marker is not None and args.trader_transfer_marker.exists():
        fields = args.trader_transfer_marker.read_text().split()
        if len(fields) != 2:
            raise ManualTestError(
                f"{args.trader_transfer_marker} records a transfer request without its result. Check the sub-account's transfer history on Lighter: "
                "if the transfer executed, write its hash and fee into the marker as '<hash> <fee>'; if not, delete the marker. Then rerun."
            )
        fees += Decimal(fields[1])
    reconcile_deployer(web3, deployer, usdc, Decimal(start["START_USDC"]), int(start["START_ETH_WEI"]), fees, lighter_dust)
    print("ok")
    return 0


def command_trader_transfer(args: argparse.Namespace) -> int:
    """Move USDC from a sub-account to another account of the Safe, as its trader would, outside the executor.

    A trader holds its sub-account's API key and may move USDC to another
    account of the same Safe, a sub-account or the master, at any time. The executor does not see such a
    transfer: it is booked nowhere, so ``check-accounts`` must report both
    accounts differing from the books and ``correct-accounts`` must accept
    the new equity. The transfer is signed with the sending sub-account's
    hand-over record from ``LIGHTER_SUB_ACCOUNT_OPERATOR_RECORD_FILES``.

    The marker file is written before the request and records its hash and
    Lighter's fee after, so a rerun never sends a second transfer: it prints
    ``sent-earlier`` and the caller checks the balances instead. ``reconcile``
    reads the fee from it, as no executor trade records it.

    Prints ``sent`` or ``sent-earlier``.
    """
    marker: Path = args.marker
    if marker.exists():
        logger.info("%s records a transfer an earlier run started: %s; not sending another", marker, marker.read_text().strip() or "no hash recorded")
        print("sent-earlier")
        return 0
    accounts = get_account_labels()
    for name in (args.source, args.destination):
        if name not in accounts:
            raise ManualTestError(f"Unknown Lighter account {name!r}; expected one of {sorted(accounts)}")
    signers = load_lighter_sub_account_operator_records(require_env("LIGHTER_SUB_ACCOUNT_OPERATOR_RECORD_FILES"))
    signer = signers.get(accounts[args.source])
    if signer is None:
        raise ManualTestError(f"No hand-over record for sub-account {args.source} in LIGHTER_SUB_ACCOUNT_OPERATOR_RECORD_FILES")
    marker.write_text("requested\n")
    request = asyncio.run(request_lighter_funding_transfer(signer, accounts[args.destination], args.amount))
    marker.write_text(f"{request.tx_hash} {request.fee}\n")
    logger.info("Sent %s USDC from %s to %s with %s's own key, fee %s: %s", args.amount, args.source, args.destination, args.source, request.fee, request.tx_hash)
    asyncio.run(wait_for_lighter_funding_transfer(signer, request.tx_hash, args.timeout))
    logger.info("Lighter executed the transfer; the executor has not booked it")
    print("sent")
    return 0


def command_donate(args: argparse.Namespace) -> int:
    """Send USDC from the deployer straight to the Safe, bypassing the vault.

    USDC that reaches the Safe without a deposit request is income the
    executor did not book: ``check-accounts`` must report the reserve
    differing from the books, and once ``correct-accounts`` books it, the
    next settlement raises the share price. The marker file is written
    before the transfer, so a rerun never sends twice.

    Prints ``sent`` or ``sent-earlier``.
    """
    marker: Path = args.marker
    if marker.exists():
        logger.info("%s records a donation an earlier run started; not sending another", marker)
        print("sent-earlier")
        return 0
    web3, deployer, usdc = connect()
    vault = get_vault(web3)
    marker.write_text(f"{args.amount}\n")
    broadcast_and_wait(web3, deployer, usdc.transfer(vault.safe_address, args.amount), f"send {args.amount} USDC straight to the Safe {vault.safe_address}")
    print("sent")
    return 0


def command_expect_nav(args: argparse.Namespace) -> int:
    """Check the vault's settled NAV against the live balances and the books.

    Run after ``lagoon-settle`` with no deposit or redemption waiting: the
    vault's total assets are then the NAV the executor posted, which must
    equal both the Safe's USDC plus every Lighter account's equity and the
    state's total equity. With ``--nav`` the NAV must also be that amount.

    Prints ``ok`` or ``mismatch``.
    """
    web3, deployer, usdc = connect()
    vault = get_vault(web3)
    accounts = get_account_labels()
    balances = fetch_balances(web3, deployer, usdc, vault, accounts)
    held = balances["safe"]["usdc"] + sum((balances[name]["equity"] for name in accounts), Decimal(0))
    nav = vault.fetch_total_assets("latest")
    state = State.read_json_file(Path(require_env("STATE_FILE")))
    booked = Decimal(str(state.portfolio.get_total_equity()))
    rows = [
        ("Vault total assets (settled NAV)", nav, ""),
        ("Safe and Lighter accounts", held, "ok" if abs(held - nav) <= args.tolerance else "DIFFERS"),
        ("State total equity", booked, "ok" if abs(booked - nav) <= args.tolerance else "DIFFERS"),
    ]
    if args.nav is not None:
        rows.append(("Expected NAV", args.nav, "ok" if abs(args.nav - nav) <= args.tolerance else "DIFFERS"))
    ok = all(row[2] != "DIFFERS" for row in rows)
    share_price = vault.fetch_share_price("latest")
    logger.log(logging.INFO if ok else logging.WARNING, "NAV check %s, share price %s:\n%s", "passed" if ok else "failed", share_price, tabulate(rows, headers=["Source", "USDC", ""], tablefmt="simple"))
    print("ok" if ok else "mismatch")
    return 0 if ok else 1


def command_books(args: argparse.Namespace) -> int:
    """Print how many balance updates the state holds, to tell whether a command booked anything.

    Prints ``balance-updates <count>``.
    """
    state = State.read_json_file(Path(require_env("STATE_FILE")))
    positions = [*state.portfolio.get_all_positions(), *state.portfolio.reserves.values()]
    count = sum(len(position.balance_updates) for position in positions)
    logger.info("The state holds %d balance updates, total equity %s USDC", count, state.portfolio.get_total_equity())
    print(f"balance-updates {count}")
    return 0


def create_parser() -> argparse.ArgumentParser:
    """Build the command-line parser of the subcommands."""
    parser = argparse.ArgumentParser(prog="python -m tradeexecutor.testing.lighter_manual", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    check_funds = commands.add_parser("check-funds", help="Check the deployer's ETH and USDC and record the starting balances.")
    check_funds.add_argument("--usdc", type=parse_usdc, required=True, help="USDC the run needs.")
    check_funds.add_argument("--min-eth", type=parse_usdc, default=DEFAULT_MIN_ETH, help="Least ETH the deployer needs for gas.")
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

    withdrawal_plan = commands.add_parser("withdrawal-plan", help="Print resume, unconfirmed <trade id>, withdraw <amount> or skip for one Lighter account.")
    withdrawal_plan.add_argument("account_index", type=int, help="Lighter account index.")
    withdrawal_plan.set_defaults(handler=command_withdrawal_plan)

    redeem_status = commands.add_parser("redeem-status", help="Print redeem while the deployer has shares to redeem, done otherwise.")
    redeem_status.set_defaults(handler=command_redeem_status)

    sweep_safe = commands.add_parser("sweep-safe", help="Move the Safe's USDC and ETH to the deployer.")
    sweep_safe.set_defaults(handler=command_sweep_safe)

    reconcile = commands.add_parser("reconcile", help="Explain the deployer's balance change over the run.")
    reconcile.add_argument("--start-file", type=Path, required=True, help="File written by check-funds.")
    reconcile.add_argument("--trader-transfer-marker", type=Path, help="Marker written by trader-transfer, for the fee of a transfer the executor did not make.")
    reconcile.set_defaults(handler=command_reconcile)

    trader_transfer = commands.add_parser("trader-transfer", help="Move USDC between sub-accounts with the sender's own key, outside the executor.")
    trader_transfer.add_argument("source", help="Sending sub-account id.")
    trader_transfer.add_argument("destination", help="Receiving account id, master or a sub-account.")
    trader_transfer.add_argument("amount", type=parse_usdc, help="USDC to move.")
    trader_transfer.add_argument("--marker", type=Path, required=True, help="File recording that the transfer was sent.")
    trader_transfer.add_argument("--timeout", type=int, default=DEFAULT_TRANSFER_TIMEOUT, help="Seconds to wait for the transfer to show in the sender's history.")
    trader_transfer.set_defaults(handler=command_trader_transfer)

    donate = commands.add_parser("donate", help="Send USDC from the deployer straight to the Safe, outside the vault.")
    donate.add_argument("amount", type=parse_usdc, help="USDC to send.")
    donate.add_argument("--marker", type=Path, required=True, help="File recording that the USDC was sent.")
    donate.set_defaults(handler=command_donate)

    expect_nav = commands.add_parser("expect-nav", help="Check the settled NAV against the live balances and the books.")
    expect_nav.add_argument("--nav", type=parse_usdc, help="Expected NAV in USDC.")
    expect_nav.add_argument("--tolerance", type=parse_usdc, default=DEFAULT_TOLERANCE_USDC, help="Allowed difference in USDC.")
    expect_nav.set_defaults(handler=command_expect_nav)

    books = commands.add_parser("books", help="Print the number of balance updates in the state.")
    books.set_defaults(handler=command_books)
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
