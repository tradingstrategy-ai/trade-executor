"""Tutorial: Trade a Lighter ETH perpetual through a Lagoon vault.

This live Ethereum tutorial demonstrates the complete vault lifecycle:

1. Deploy a Lagoon vault and create its Safe-owned Lighter account
2. Subscribe to the vault and deposit USDC from the Safe to Lighter
3. Synchronise and verify NAV with trade-executor Typer commands
4. Open a small ETH/USD perpetual long and verify NAV again
5. Close the long and verify NAV again
6. Withdraw Lighter collateral to the Safe
7. Redeem all vault shares back to the deployer

Lighter's sequencer, order fills and secure withdrawals cannot be simulated by
Anvil, so this script deliberately supports Ethereum mainnet only. Use a funded
test account and small values.

Withdrawal submission happens once. The claimability poll uses
``eth_defi.lighter.sdk.LighterAuthTokenManager`` so an expired auth token is
refreshed without replaying the withdrawal request.

The deployment command creates a Lighter API key. The key is read from its
mode-0600 operator record and is never printed, logged, passed as a command-line
argument or copied to strategy state and public reports.

Example
-------

.. code-block:: shell

    source .local-test.env
    export LIGHTER_TEST_PRIVATE_KEY="0x..."
    export JSON_RPC_ETHEREUM="https://..."
    poetry run pip install lighter-sdk==1.1.2
    poetry run python scripts/lagoon/manual-trade-executor-lighter.py

Optional environment variables
------------------------------

``LIGHTER_DEPOSIT_USDC``
    Target Lighter equity. Defaults to the order notional plus a 5 USDC buffer.

``LIGHTER_POSITION_USDC``
    ETH/USD long notional. Defaults to Lighter's minimum plus 1 USDC.

``LIGHTER_MAX_SLIPPAGE``
    Maximum open/close order slippage. Defaults to 2%.

``LIGHTER_API_KEY_INDEX``
    API-key slot created during deployment. Defaults to 4.

``LIGHTER_RUN_DIR``
    New private working directory. Defaults below ``~/.tradingstrategy/examples``.

``LIGHTER_DEPOSIT_TIMEOUT`` / ``LIGHTER_WITHDRAW_TIMEOUT``
    Maximum seconds to wait for Lighter state transitions.

``LIGHTER_AUTH_TOKEN_TIMEOUT``
    Deliberately short authenticated SDK token lifetime used to exercise token
    rotation. Defaults to 30 seconds in this tutorial; production callers
    should use :class:`eth_defi.lighter.sdk.LighterAuthTokenManager` defaults.
"""

import asyncio
import json
import logging
import os
import stat
import time
from dataclasses import dataclass, field
from decimal import ROUND_CEILING, ROUND_DOWN, Decimal
from pathlib import Path
from typing import Any
from unittest import mock

import lighter
from eth_defi.erc_4626.settlement_events import fetch_vault_settlement_logs
from eth_defi.erc_4626.vault_protocol.lagoon.vault import LagoonVault
from eth_defi.hotwallet import HotWallet
from eth_defi.lighter.api import LIGHTER_MIN_MAINNET_USDC, wait_for_lighter_collateral
from eth_defi.lighter.constants import (
    LIGHTER_API_URL,
    LIGHTER_ETHEREUM_DEPLOYMENT_CHAIN_ID,
    LIGHTER_USDC_ETHEREUM,
)
from eth_defi.lighter.lagoon import (
    claim_usdc_to_lagoon_safe_from_lighter,
    deposit_usdc_from_lagoon_safe_into_lighter,
)
from eth_defi.lighter.pubkey import MIN_API_KEY_INDEX
from eth_defi.lighter.sdk import LighterAuthTokenManager
from eth_defi.lighter.session import LighterSession, create_lighter_session
from eth_defi.lighter.valuation import (
    LighterEquity,
    fetch_lighter_account_by_index,
    fetch_lighter_total_equity,
)
from eth_defi.provider.broken_provider import _latest_delayed_block_number_cache
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.provider.receipt import wait_for_transaction_receipt_robust
from eth_defi.token import TokenDetails, fetch_erc20_details
from eth_defi.utils import setup_console_logging
from eth_defi.vault.base import VaultSpec
from web3 import Web3
from web3.contract.contract import ContractFunction

from tradeexecutor.cli.main import app
from tradeexecutor.state.state import State

logger = logging.getLogger(__name__)

#: Lighter mainnet market index for the ETH/USD perpetual used by the tutorial.
ETH_PERP_MARKET_INDEX = 0
#: Interval for public position and withdrawal-history polling.
POLL_SECONDS = 5
#: Deliberately short lifetime so a normal withdrawal wait demonstrates rotation.
TUTORIAL_LIGHTER_AUTH_TOKEN_TIMEOUT = 30
#: Ten native-USDC base units, allowing only token-quantisation dust.
USDC_TOLERANCE = Decimal("0.000010")
#: One cent accommodates float serialisation and live public-equity timing.
NAV_SYNC_TOLERANCE = Decimal("0.01")
#: Maximum two-base-unit Safe residue accepted after full redemption.
MAX_SAFE_USDC_DUST_RAW = 2

# ---------------------------------------------------------------------------
# Tutorial configuration and state
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TutorialConfig:
    """Operator configuration, with secrets excluded from its representation."""

    #: Ethereum RPC endpoint; hidden because a URL may contain credentials.
    rpc_url: str = field(repr=False)
    #: Funded deployer key used for the temporary vault lifecycle.
    deployer_private_key: str = field(repr=False)
    #: Fresh directory that receives this tutorial run's public artefacts.
    run_dir: Path
    #: Optional ETH/USD perpetual notional requested by the operator.
    position_notional: Decimal | None
    #: Target Lighter collateral after the Safe-to-Lighter deposit.
    target_lighter_equity: Decimal
    #: Maximum order-price movement accepted during market-order submission.
    max_slippage: Decimal
    #: Delegated Lighter key slot generated for this one tutorial run.
    api_key_index: int
    #: Maximum wait for the Lighter collateral deposit to become visible.
    deposit_timeout: int
    #: Maximum wait for an accepted Lighter withdrawal to become claimable.
    withdraw_timeout: int
    #: Short-lived SDK bearer-token timeout used to demonstrate key rotation.
    auth_token_timeout: int = TUTORIAL_LIGHTER_AUTH_TOKEN_TIMEOUT


@dataclass(frozen=True, slots=True)
class TutorialContext:
    """Runtime objects shared by tutorial phases, omitting sensitive repr data."""

    #: Immutable operator configuration for the current run.
    config: TutorialConfig
    #: Connected Ethereum client; excluded from repr because its provider may
    #: contain sensitive endpoint data.
    web3: Web3 = field(repr=False)
    #: Funded local wallet; excluded from repr because it owns private-key material.
    deployer: HotWallet = field(repr=False)
    #: Native Ethereum USDC token helper used for all balance assertions.
    usdc: TokenDetails
    #: Static Lighter strategy passed to Typer lifecycle commands.
    strategy_file: Path
    #: Executor state file created inside the isolated run directory.
    state_file: Path
    #: Vault record text file whose paired JSON contains the generated signer.
    record_file: Path


@dataclass(slots=True)
class VaultDeployment:
    """Public vault metadata plus the secret Lighter signer."""

    #: Lagoon vault and Safe wrapper created by the deployment command.
    vault: LagoonVault
    #: Public Lighter account index owned by the deployed Safe.
    account_index: int
    #: Public delegated Lighter key slot used for order submission.
    api_key_index: int
    #: Generated delegated Lighter signer, hidden from repr and cleared after use.
    api_private_key: str = field(repr=False)
    #: Typer environment including deployer credentials, hidden from repr.
    cli_environment: dict[str, str] = field(repr=False)

    def clear_api_private_key(self) -> None:
        """Discard the delegated signer after all authenticated work is complete."""
        self.api_private_key = ""


@dataclass(frozen=True, slots=True)
class NavCheckpoint:
    """Values verified after one Lagoon NAV synchronisation."""

    #: Current native-USDC balance held by the Safe.
    safe_usdc: Decimal
    #: Current public total equity reported by Lighter.
    lighter_equity: Decimal
    #: Signed ETH/USD perpetual base position reported by Lighter.
    eth_position: Decimal
    #: Newly emitted Lagoon total-assets value after settlement.
    posted_nav: Decimal


# ---------------------------------------------------------------------------
# Configuration, logging and CLI helpers
# ---------------------------------------------------------------------------


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
    """Validate environment variables before creating any on-chain state."""
    if os.environ.get("SIMULATE", "").lower() in {"1", "true", "yes"}:
        raise RuntimeError(
            "This tutorial is live Ethereum/Lighter only; SIMULATE is unsupported"
        )

    position_notional = (
        Decimal(os.environ["LIGHTER_POSITION_USDC"])
        if os.environ.get("LIGHTER_POSITION_USDC")
        else None
    )
    run_dir = Path(
        os.environ.get(
            "LIGHTER_RUN_DIR",
            str(
                Path("~/.tradingstrategy/examples").expanduser()
                / f"lighter-manual-{int(time.time())}"
            ),
        )
    )
    auth_token_timeout = int(
        os.environ.get(
            "LIGHTER_AUTH_TOKEN_TIMEOUT",
            str(TUTORIAL_LIGHTER_AUTH_TOKEN_TIMEOUT),
        )
    )
    if auth_token_timeout < 1:
        raise ValueError("LIGHTER_AUTH_TOKEN_TIMEOUT must be at least one second")

    return TutorialConfig(
        rpc_url=require_env("JSON_RPC_ETHEREUM"),
        deployer_private_key=require_env("LIGHTER_TEST_PRIVATE_KEY", "PRIVATE_KEY"),
        run_dir=run_dir,
        position_notional=position_notional,
        target_lighter_equity=Decimal(os.environ.get("LIGHTER_DEPOSIT_USDC", "0")),
        max_slippage=Decimal(os.environ.get("LIGHTER_MAX_SLIPPAGE", "0.02")),
        api_key_index=int(
            os.environ.get("LIGHTER_API_KEY_INDEX", str(MIN_API_KEY_INDEX))
        ),
        deposit_timeout=int(os.environ.get("LIGHTER_DEPOSIT_TIMEOUT", "900")),
        withdraw_timeout=int(os.environ.get("LIGHTER_WITHDRAW_TIMEOUT", "3600")),
        auth_token_timeout=auth_token_timeout,
    )


def prepare_context(config: TutorialConfig) -> TutorialContext:
    """Connect to Ethereum and verify deployer funds."""
    web3 = create_multi_provider_web3(config.rpc_url, default_http_timeout=(3.0, 180.0))
    assert web3.eth.chain_id == LIGHTER_ETHEREUM_DEPLOYMENT_CHAIN_ID, (
        "Ethereum mainnet RPC required"
    )
    deployer = HotWallet.from_private_key(config.deployer_private_key)
    usdc = fetch_erc20_details(web3, LIGHTER_USDC_ETHEREUM)
    usdc_balance = usdc.fetch_balance_of(deployer.address)
    eth_balance = Decimal(
        web3.from_wei(web3.eth.get_balance(deployer.address), "ether")
    )
    assert usdc_balance > 0, "Deployer has no native Ethereum USDC"
    assert eth_balance > 0, "Deployer has no ETH for mainnet gas"

    config.run_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    strategy_file = (
        Path(__file__).resolve().parents[2]
        / "strategies/test_only/minimal_lighter_strategy.py"
    )
    logger.info("Connected to Ethereum mainnet (chain id %d)", web3.eth.chain_id)
    logger.info("Deployer: %s", deployer.address)
    logger.info("Starting balances: %s USDC, %.6f ETH", usdc_balance, eth_balance)
    logger.info(
        "Lighter auth token lifetime for this tutorial: %d seconds",
        config.auth_token_timeout,
    )
    logger.info("Run artefacts: %s", config.run_dir)
    return TutorialContext(
        config=config,
        web3=web3,
        deployer=deployer,
        usdc=usdc,
        strategy_file=strategy_file,
        state_file=config.run_dir / "state.json",
        record_file=config.run_dir / "vault-record.txt",
    )


def public_error(error: Exception) -> RuntimeError:
    """Hide SDK exception text which may contain signed request material."""
    return RuntimeError(f"Lighter operation failed ({type(error).__name__})")


class TutorialError(RuntimeError):
    """Operator-facing tutorial failure containing no SDK payload."""


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
        # Restore tutorial INFO logging before the next lifecycle phase.
        configure_logging()
        logger.setLevel(logging.INFO)


def load_secret_record(path: Path) -> dict[str, Any]:
    """Load the operator record only when its ownership and mode are safe."""
    info = path.stat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_mode & 0o077
    ):
        raise PermissionError(f"Refusing insecure Lighter operator record: {path}")
    return json.loads(path.read_text())


def assert_secret_not_in_public_artifacts(
    run_dir: Path, secret_record: Path, secret: str
) -> None:
    """Verify the generated signer is confined to the operator record."""
    for path in run_dir.iterdir():
        if path == secret_record or not path.is_file() or path.suffix == ".br":
            continue
        if secret in path.read_text(errors="ignore"):
            raise RuntimeError(f"Lighter API key leaked to public artefact {path.name}")


# ---------------------------------------------------------------------------
# Lighter market and API helpers
# ---------------------------------------------------------------------------


def signed_eth_position(account: dict[str, Any]) -> Decimal:
    """Return signed ETH perpetual size from a public account response."""
    for position in account.get("positions") or ():
        if int(position.get("market_id", -1)) == ETH_PERP_MARKET_INDEX:
            size = Decimal(str(position["position"]))
            return size if int(position.get("sign", 1)) >= 0 else -size
    return Decimal(0)


async def wait_for_position(
    session: LighterSession,
    account_index: int,
    expected: str,
    timeout: int = 300,
) -> Decimal:
    """Wait until the public account shows the requested position state."""
    deadline = time.monotonic() + timeout
    while True:
        value = signed_eth_position(
            fetch_lighter_account_by_index(session, account_index)
        )
        if (expected == "long" and value > 0) or (expected == "flat" and value == 0):
            return value
        if value < 0:
            raise RuntimeError("Lighter account unexpectedly has an ETH short")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"ETH position did not become {expected}")
        await asyncio.sleep(POLL_SECONDS)


async def resolve_eth_order(
    requested_notional: Decimal | None,
) -> tuple[int, Decimal, int, Decimal]:
    """Resolve a valid ETH order size from Lighter's public market metadata."""
    api_client = lighter.ApiClient(
        configuration=lighter.Configuration(host=LIGHTER_API_URL)
    )
    try:
        response = await lighter.OrderApi(api_client).order_book_details(
            market_id=ETH_PERP_MARKET_INDEX
        )
        market = response.order_book_details[0]
        price = Decimal(str(market.last_trade_price))
        min_quote = Decimal(str(market.min_quote_amount))
        min_base = Decimal(str(market.min_base_amount))
        size_decimals = int(market.size_decimals)
        step = Decimal(1).scaleb(-int(market.supported_size_decimals))
        target = max(requested_notional or min_quote + 1, min_quote, min_base * price)
        base = (max(min_base, target / price) / step).to_integral_value(
            rounding=ROUND_CEILING
        ) * step
        return int(base * Decimal(10**size_decimals)), base, size_decimals, base * price
    finally:
        await api_client.close()


async def submit_market_order(
    deployment: VaultDeployment,
    base_amount: int,
    max_slippage: Decimal,
    *,
    is_ask: bool,
    reduce_only: bool,
) -> None:
    """Submit one bounded-slippage order without logging signed SDK objects."""
    client = None
    try:
        client = lighter.SignerClient(
            url=LIGHTER_API_URL,
            account_index=deployment.account_index,
            api_private_keys={deployment.api_key_index: deployment.api_private_key},
        )
        if client.check_client():
            raise TutorialError("Lighter API key was rejected")
        (
            _transaction,
            _response,
            error,
        ) = await client.create_market_order_limited_slippage(
            market_index=ETH_PERP_MARKET_INDEX,
            client_order_index=int(time.time() * 1_000),
            base_amount=base_amount,
            max_slippage=float(max_slippage),
            is_ask=is_ask,
            reduce_only=reduce_only,
            api_key_index=deployment.api_key_index,
        )
        if error:
            raise TutorialError("Lighter order was rejected")
    except TutorialError:
        raise
    # SDK exceptions may contain signed request details, so normalise every
    # failure at this public tutorial boundary.
    except Exception as error:  # noqa: BLE001
        raise public_error(error) from None
    finally:
        if client is not None:
            await client.close()


async def request_secure_withdrawal(
    deployment: VaultDeployment,
    amount: Decimal,
    timeout: int,
    auth_token_timeout: int,
) -> Decimal:
    """Request a secure USDC withdrawal and wait for its claimable amount.

    Withdrawal submission is performed once. Only the idempotent
    ``withdraw_history`` follow-up is wrapped in the generic auth-token retry
    helper, so an expired token can never resubmit the withdrawal.
    """
    client = None
    api_client = None
    try:
        client = lighter.SignerClient(
            url=LIGHTER_API_URL,
            account_index=deployment.account_index,
            api_private_keys={deployment.api_key_index: deployment.api_private_key},
        )
        _transaction, response, error = await client.withdraw(
            asset_id=client.ASSET_ID_USDC,
            route_type=client.ROUTE_PERP,
            amount=float(amount),
            api_key_index=deployment.api_key_index,
        )
        if error or response is None:
            raise TutorialError("Lighter secure withdrawal was rejected")
        logger.info("Secure withdrawal accepted: %s", response.tx_hash)

        api_client = lighter.ApiClient(
            configuration=lighter.Configuration(host=LIGHTER_API_URL)
        )
        transaction_api = lighter.TransactionApi(api_client)

        auth_token_generations = 0

        def create_auth_token() -> tuple[str | None, object | None]:
            nonlocal auth_token_generations
            auth_token_generations += 1
            action = "Creating" if auth_token_generations == 1 else "Rotating"
            logger.info(
                "%s Lighter auth token for withdrawal history (generation %d)",
                action,
                auth_token_generations,
            )
            return client.create_auth_token_with_expiry(
                deadline=auth_token_timeout,
                api_key_index=deployment.api_key_index,
            )

        auth_manager = LighterAuthTokenManager(
            token_factory=create_auth_token,
            token_lifetime=auth_token_timeout,
            refresh_margin=0.0,
        )

        deadline = time.monotonic() + timeout
        while True:
            history = await auth_manager.call(
                lambda auth: transaction_api.withdraw_history(
                    authorization=auth,
                    account_index=deployment.account_index,
                ),
                operation_name="withdrawal history",
            )
            rows = [
                row
                for row in history.withdraws
                if row.asset_id == client.ASSET_ID_USDC and Decimal(row.amount) > 0
            ]
            matches = [
                row
                for row in rows
                if abs(Decimal(row.amount) - amount) <= USDC_TOLERANCE
            ]
            if len(matches) > 1:
                raise TutorialError("Ambiguous Lighter withdrawal history")
            if len(matches) == 1 and matches[0].status.lower() == "claimable":
                return Decimal(matches[0].amount)
            if time.monotonic() >= deadline:
                raise TutorialError("Lighter withdrawal did not become claimable in time")
            await asyncio.sleep(POLL_SECONDS)
    except TutorialError:
        raise
    # SDK exceptions may contain signed request details, so normalise every
    # failure at this public tutorial boundary.
    except Exception as error:  # noqa: BLE001
        raise public_error(error) from None
    finally:
        if client is not None:
            await client.close()
        if api_client is not None:
            await api_client.close()


# ---------------------------------------------------------------------------
# Lagoon lifecycle phases
# ---------------------------------------------------------------------------


def broadcast_and_wait(context: TutorialContext, function: ContractFunction) -> None:
    """Broadcast one deployer transaction and wait for its receipt."""
    context.deployer.sync_nonce(context.web3)
    tx_hash = context.deployer.transact_and_broadcast_with_contract(function)
    wait_for_transaction_receipt_robust(context.web3, tx_hash)


def make_vault(
    context: TutorialContext, deployment_data: dict[str, Any]
) -> LagoonVault:
    """Create the vault wrapper from public deployment fields."""
    return LagoonVault(
        context.web3,
        VaultSpec(context.web3.eth.chain_id, deployment_data["vault_address"]),
        trading_strategy_module_address=deployment_data["module_address"],
        default_block_identifier="latest",
        require_denomination_token=True,
    )


def deploy_vault(context: TutorialContext) -> VaultDeployment:
    """Deploy Lagoon, load its protected signer and initialise executor state."""
    deployment_environment = {
        "EXECUTOR_ID": "lighter-manual",
        "STRATEGY_FILE": str(context.strategy_file),
        "STATE_FILE": str(context.state_file),
        "PRIVATE_KEY": context.config.deployer_private_key,
        "JSON_RPC_ETHEREUM": context.config.rpc_url,
        "VAULT_RECORD_FILE": str(context.record_file),
        "FUND_NAME": "Lighter manual test",
        "FUND_SYMBOL": "LTM",
        "ANY_ASSET": "true",
        "PERFORMANCE_FEE": "0",
        "MANAGEMENT_FEE": "0",
        "UNIT_TESTING": "true",
        "GENERATE_LIGHTER_API_KEY": "true",
        "LIGHTER_API_KEY_INDEX": str(context.config.api_key_index),
        "LOG_LEVEL": "warning",
    }
    run_cli(["lagoon-deploy-vault"], deployment_environment)

    secret_record = context.record_file.with_suffix(".json")
    operator_data = load_secret_record(secret_record)
    deployment_data = operator_data["deployments"]["ethereum"]
    lighter_setup = deployment_data["lighter_account_setup"]
    api_private_key = lighter_setup["private_key"]
    assert_secret_not_in_public_artifacts(
        context.config.run_dir, secret_record, api_private_key
    )

    common_environment = {
        "ID": "lighter-manual",
        "NAME": "Lighter manual test",
        "STRATEGY_FILE": str(context.strategy_file),
        "STATE_FILE": str(context.state_file),
        "PRIVATE_KEY": context.config.deployer_private_key,
        "JSON_RPC_ETHEREUM": context.config.rpc_url,
        "ASSET_MANAGEMENT_MODE": "lagoon",
        "VAULT_ADDRESS": deployment_data["vault_address"],
        "VAULT_ADAPTER_ADDRESS": deployment_data["module_address"],
        "LIGHTER_ACCOUNT_INDEX": str(lighter_setup["account_index"]),
        "UNIT_TESTING": "true",
        "CACHE_PATH": str(context.config.run_dir / "cache"),
        "MIN_GAS_BALANCE": "0",
        "LOG_LEVEL": "warning",
    }
    run_cli(["init"], common_environment)
    run_cli(["correct-accounts"], common_environment)

    vault = make_vault(context, deployment_data)
    logger.info("Vault: %s", vault.address)
    logger.info("Safe: %s", vault.safe_address)
    logger.info("Lighter account index: %s", lighter_setup["account_index"])
    logger.info("Lighter API-key slot: %s", lighter_setup["api_key_index"])
    return VaultDeployment(
        vault=vault,
        account_index=int(lighter_setup["account_index"]),
        api_key_index=int(lighter_setup["api_key_index"]),
        api_private_key=api_private_key,
        cli_environment=common_environment,
    )


def fetch_lighter_equity(account_index: int) -> LighterEquity:
    """Fetch one public Lighter equity snapshot."""
    session = create_lighter_session()
    try:
        return fetch_lighter_total_equity(session, account_index)
    finally:
        session.close()


def deposit_to_lighter(
    context: TutorialContext, deployment: VaultDeployment, target_equity: Decimal
) -> None:
    """Subscribe to Lagoon and move the subscribed USDC from Safe to Lighter."""
    current_equity = fetch_lighter_equity(deployment.account_index).get_total()
    assert current_equity.is_finite() and current_equity >= 0, (
        "Invalid Lighter equity before funding"
    )
    deposit_amount = target_equity - current_equity
    assert deposit_amount >= LIGHTER_MIN_MAINNET_USDC, (
        "Additional deposit is below Lighter's minimum"
    )
    assert context.usdc.fetch_balance_of(context.deployer.address) >= deposit_amount, (
        "Deployer lacks USDC"
    )

    logger.info("Subscribing %s USDC to the Lagoon vault", deposit_amount)
    broadcast_and_wait(
        context, context.usdc.approve(deployment.vault.address, deposit_amount)
    )
    broadcast_and_wait(
        context,
        deployment.vault.request_deposit(
            context.deployer.address,
            context.usdc.convert_to_raw(deposit_amount),
        ),
    )
    run_cli(["lagoon-settle"], deployment.cli_environment)
    claimable_assets = deployment.vault.vault_contract.functions.maxDeposit(
        context.deployer.address
    ).call()
    assert claimable_assets > 0, "Lagoon subscription did not produce claimable assets"
    broadcast_and_wait(
        context,
        deployment.vault.finalise_deposit(
            context.deployer.address, raw_amount=claimable_assets
        ),
    )

    safe_balance = context.usdc.fetch_balance_of(deployment.vault.safe_address)
    assert safe_balance >= deposit_amount, "Safe did not receive subscribed USDC"
    logger.info("Depositing %s USDC from Safe to Lighter", deposit_amount)
    deposit_usdc_from_lagoon_safe_into_lighter(
        context.web3,
        context.deployer,
        vault=deployment.vault,
        usdc=context.usdc,
        deposit_usdc=deposit_amount,
    )
    session = create_lighter_session()
    try:
        wait_for_lighter_collateral(
            session,
            deployment.account_index,
            target_equity,
            timeout=context.config.deposit_timeout,
        )
    finally:
        session.close()
    logger.info("Lighter now reports at least %s USDC collateral", target_equity)


def assert_nav_checkpoint(
    context: TutorialContext,
    deployment: VaultDeployment,
    expected_position: str,
    start_block: int,
) -> NavCheckpoint:
    """Verify Safe, Lighter, executor state and newly posted Lagoon NAV agree."""
    session = create_lighter_session()
    try:
        equity = fetch_lighter_total_equity(
            session, deployment.account_index
        ).get_total()
        position = signed_eth_position(
            fetch_lighter_account_by_index(session, deployment.account_index)
        )
    finally:
        session.close()
    assert equity.is_finite() and equity >= 0, "Negative or non-finite Lighter equity"
    assert (expected_position == "flat" and position == 0) or (
        expected_position == "long" and position > 0
    )

    safe_usdc = context.usdc.fetch_balance_of(deployment.vault.safe_address)
    state = State.read_json_file(context.state_file)
    assert state.portfolio.get_net_asset_value() >= 0, "Negative portfolio NAV"
    reserve = state.portfolio.get_default_reserve_position()
    assert reserve.quantity >= 0
    assert abs(reserve.quantity - safe_usdc) <= USDC_TOLERANCE, (
        "State reserve does not match Safe USDC"
    )
    positions = [
        position
        for position in state.portfolio.get_open_and_frozen_positions()
        if position.is_exchange_account()
        and position.pair.get_exchange_account_protocol() == "lighter"
    ]
    assert len(positions) == 1 and positions[0].get_value() >= 0
    assert abs(Decimal(str(positions[0].get_value())) - equity) <= NAV_SYNC_TOLERANCE

    logs = fetch_vault_settlement_logs(
        web3=context.web3,
        address=deployment.vault.address,
        topic0_list=[Web3.to_hex(Web3.keccak(text="NewTotalAssetsUpdated(uint256)"))],
        start_block=start_block,
        end_block=context.web3.eth.block_number,
        use_hypersync=False,
    )
    assert logs, "lagoon-settle did not post a new NAV"
    posted_nav = context.usdc.convert_to_decimals(
        int.from_bytes(bytes(logs[-1]["data"]), byteorder="big")
    )
    assert abs(posted_nav - (safe_usdc + equity)) <= NAV_SYNC_TOLERANCE
    return NavCheckpoint(
        safe_usdc=safe_usdc,
        lighter_equity=equity,
        eth_position=position,
        posted_nav=posted_nav,
    )


def sync_and_report_nav(
    context: TutorialContext,
    deployment: VaultDeployment,
    expected_position: str,
) -> NavCheckpoint:
    """Run Typer NAV commands, assert their result and log a concise summary."""
    start_block = context.web3.eth.block_number
    run_cli(["lagoon-settle"], deployment.cli_environment)
    run_cli(["show-valuation"], deployment.cli_environment)
    checkpoint = assert_nav_checkpoint(
        context, deployment, expected_position, start_block
    )
    logger.info(
        "NAV verified: Safe=%s USDC, Lighter=%s USDC, ETH position=%s, posted NAV=%s USDC",
        checkpoint.safe_usdc,
        checkpoint.lighter_equity,
        checkpoint.eth_position,
        checkpoint.posted_nav,
    )
    return checkpoint


async def open_eth_long(
    context: TutorialContext,
    deployment: VaultDeployment,
    base_amount: int,
    base_size: Decimal,
    notional: Decimal,
) -> Decimal:
    """Open the tutorial ETH/USD long and return its filled size."""
    logger.info(
        "Opening ETH/USD long: %s ETH, approximately %s USDC notional",
        base_size,
        notional,
    )
    await submit_market_order(
        deployment,
        base_amount,
        context.config.max_slippage,
        is_ask=False,
        reduce_only=False,
    )
    session = create_lighter_session()
    try:
        opened = await wait_for_position(session, deployment.account_index, "long")
    finally:
        session.close()
    logger.info("ETH/USD long filled: %s ETH", opened)
    return opened


async def close_eth_long(
    context: TutorialContext,
    deployment: VaultDeployment,
    opened_size: Decimal,
    size_decimals: int,
) -> None:
    """Close the whole ETH/USD long with a reduce-only order."""
    close_amount = int(
        (opened_size * Decimal(10**size_decimals)).to_integral_value(
            rounding=ROUND_CEILING
        )
    )
    logger.info("Closing ETH/USD long: %s ETH", opened_size)
    await submit_market_order(
        deployment,
        close_amount,
        context.config.max_slippage,
        is_ask=True,
        reduce_only=True,
    )
    session = create_lighter_session()
    try:
        await wait_for_position(session, deployment.account_index, "flat")
    finally:
        session.close()
    logger.info("ETH/USD position is flat")


async def withdraw_to_safe(
    context: TutorialContext, deployment: VaultDeployment
) -> Decimal:
    """Withdraw all usable Lighter USDC and claim it to the Lagoon Safe."""
    available = fetch_lighter_equity(deployment.account_index).available_balance
    assert available.is_finite() and available >= 0, "Invalid Lighter available balance"
    withdraw_amount = available.quantize(Decimal("0.000001"), rounding=ROUND_DOWN)
    assert withdraw_amount >= LIGHTER_MIN_MAINNET_USDC, (
        "Available balance is below withdrawal minimum"
    )
    safe_before = context.usdc.fetch_balance_of(deployment.vault.safe_address)

    logger.info("Requesting secure withdrawal of %s USDC", withdraw_amount)
    claimable = await request_secure_withdrawal(
        deployment,
        withdraw_amount,
        context.config.withdraw_timeout,
        context.config.auth_token_timeout,
    )
    logger.info("Claiming %s USDC from Lighter to Safe", claimable)
    # ``lagoon-settle`` runs through Typer and may consume a
    # deployer nonce unknown to this in-process wallet.
    context.deployer.sync_nonce(context.web3)
    claim_usdc_to_lagoon_safe_from_lighter(
        context.web3,
        context.deployer,
        vault=deployment.vault,
        usdc=context.usdc,
        claimable_usdc=claimable,
    )
    safe_after = context.usdc.fetch_balance_of(deployment.vault.safe_address)
    assert context.usdc.convert_to_raw(
        safe_after - safe_before
    ) == context.usdc.convert_to_raw(claimable)
    logger.info("Safe received the claimed %s USDC", claimable)
    return claimable


def fund_redemption_shortfall(
    context: TutorialContext,
    deployment: VaultDeployment,
    share_balance_raw: int,
) -> None:
    """Give the Safe enough temporary liquidity to redeem all shares.

    A NAV-only ``lagoon-settle`` cycle does not update the settled share price.
    After trading fees, the old ``convertToAssets()`` value may therefore be
    slightly greater than the USDC claimed from Lighter. The subsequent full
    redemption returns this small top-up to the deployer.
    """
    required_raw = deployment.vault.vault_contract.functions.convertToAssets(
        share_balance_raw
    ).call()
    safe_raw = context.usdc.fetch_raw_balance_of(deployment.vault.safe_address)
    if required_raw <= safe_raw:
        return

    deficit_raw = required_raw - safe_raw + 1
    deficit = context.usdc.convert_to_decimals(deficit_raw)
    assert context.usdc.fetch_raw_balance_of(context.deployer.address) >= deficit_raw
    logger.info("Temporarily funding Safe redemption shortfall: %s USDC", deficit)
    broadcast_and_wait(
        context,
        context.usdc.transfer(deployment.vault.safe_address, deficit),
    )
    assert (
        context.usdc.fetch_raw_balance_of(deployment.vault.safe_address)
        >= required_raw
    )


def redeem_all_shares(context: TutorialContext, deployment: VaultDeployment) -> None:
    """Redeem all shares and verify excess USDC returned to the deployer."""
    share_balance_raw = deployment.vault.share_token.fetch_raw_balance_of(
        context.deployer.address
    )
    assert share_balance_raw > 0, "Deployer has no vault shares to redeem"
    fund_redemption_shortfall(context, deployment, share_balance_raw)

    usdc_before = context.usdc.fetch_balance_of(context.deployer.address)
    run_cli(["lagoon-redeem"], deployment.cli_environment)
    assert (
        deployment.vault.share_token.fetch_raw_balance_of(context.deployer.address) == 0
    )
    usdc_after = context.usdc.fetch_balance_of(context.deployer.address)
    assert usdc_after > usdc_before, "Redemption did not credit deployer USDC"
    assert (
        context.usdc.fetch_raw_balance_of(deployment.vault.safe_address)
        <= MAX_SAFE_USDC_DUST_RAW
    )
    logger.info(
        "Redeemed all shares; deployer received %s USDC", usdc_after - usdc_before
    )


# ---------------------------------------------------------------------------
# Tutorial entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    """Run the complete tutorial lifecycle in readable phases."""
    configure_logging()

    log_step(1, "Validate configuration and connect to Ethereum")
    context = prepare_context(load_config())
    base_amount, base_size, size_decimals, notional = await resolve_eth_order(
        context.config.position_notional
    )
    target_equity = max(context.config.target_lighter_equity, notional + 5, Decimal(5))
    assert context.usdc.fetch_balance_of(context.deployer.address) >= target_equity
    logger.info("Planned Lighter collateral: %s USDC", target_equity)
    logger.info(
        "Planned ETH/USD position: %s ETH / approximately %s USDC", base_size, notional
    )

    log_step(2, "Deploy Lagoon vault and initialise executor state")
    deployment = deploy_vault(context)

    log_step(3, "Subscribe to Lagoon and deposit USDC to Lighter")
    deposit_to_lighter(context, deployment, target_equity)

    log_step(4, "Synchronise NAV before trading")
    sync_and_report_nav(context, deployment, "flat")

    log_step(5, "Open ETH/USD long")
    opened_size = await open_eth_long(
        context, deployment, base_amount, base_size, notional
    )

    log_step(6, "Synchronise NAV with the long open")
    sync_and_report_nav(context, deployment, "long")

    log_step(7, "Close ETH/USD long")
    await close_eth_long(context, deployment, opened_size, size_decimals)

    log_step(8, "Synchronise NAV after closing")
    sync_and_report_nav(context, deployment, "flat")

    log_step(9, "Withdraw Lighter balance to the Safe")
    await withdraw_to_safe(context, deployment)

    log_step(10, "Synchronise final vault NAV")
    sync_and_report_nav(context, deployment, "flat")

    log_step(11, "Redeem all shares to the deployer")
    redeem_all_shares(context, deployment)

    deployment.clear_api_private_key()
    logger.info("Tutorial complete. Public artefacts are in %s", context.config.run_dir)


if __name__ == "__main__":
    asyncio.run(main())
