"""Run a manual Lighter trade round trip for an existing Lagoon vault.

The command is deliberately an operator test instead of a strategy execution
path.  It moves a bounded amount of Safe-owned USDC to Lighter, opens and
closes a small ETH perpetual position, claims the L2 withdrawal to the Safe,
and posts Lagoon NAV checkpoints throughout the lifecycle.

The Safe must already hold the test collateral. For a newly activated Lighter
vault, use ``scripts/lagoon/deposit-and-settle.py`` first; the activation's
accounted 1 USDC deposit leaves no Safe USDC for this command.
"""

import asyncio
import fcntl
import json
import logging
import os
import stat
import tempfile
import time
from dataclasses import dataclass, field
from decimal import ROUND_CEILING, ROUND_DOWN, Decimal
from pathlib import Path
from typing import Any

import lighter
import typer
from eth_defi.compat import native_datetime_utc_now
from eth_defi.erc_4626.vault_protocol.lagoon.vault import LagoonVault
from eth_defi.lighter.api import (
    LIGHTER_MIN_MAINNET_USDC,
    fetch_lighter_withdrawal_delay,
    wait_for_lighter_collateral,
)
from eth_defi.lighter.constants import (
    LIGHTER_API_URL,
    LIGHTER_ETHEREUM_DEPLOYMENT_CHAIN_ID,
    LIGHTER_USDC_ETHEREUM,
)
from eth_defi.lighter.lagoon import (
    claim_usdc_to_lagoon_safe_from_lighter,
    deposit_usdc_from_lagoon_safe_into_lighter,
)
from eth_defi.lighter.sdk import LighterAuthTokenManager
from eth_defi.lighter.session import LighterSession, create_lighter_session
from eth_defi.lighter.valuation import (
    LighterEquity,
    fetch_lighter_account_by_index,
    fetch_lighter_total_equity,
)
from eth_defi.token import fetch_erc20_details
from eth_defi.vault.base import VaultSpec
from typer import Option

from tradeexecutor.cli.bootstrap import prepare_executor_id
from tradeexecutor.cli.commands import shared_options
from tradeexecutor.cli.commands.app import app
from tradeexecutor.cli.commands.lagoon_utils import (
    create_hot_wallet,
    create_single_chain_web3_config,
    resolve_state_store,
)
from tradeexecutor.cli.log import setup_logging
from tradeexecutor.ethereum.lagoon.vault import LagoonVaultSyncModel
from tradeexecutor.exchange_account.lighter import (
    LIGHTER_PROTOCOL,
    create_lighter_exchange_account_pair,
    create_lighter_vault_valuation_func,
)
from tradeexecutor.exchange_account.pricing import ExchangeAccountPricingModel
from tradeexecutor.exchange_account.state import open_exchange_account_position
from tradeexecutor.exchange_account.valuation import ExchangeAccountValuator
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradeexecutor.state.store import JSONFileStore
from tradeexecutor.strategy.execution_model import AssetManagementMode
from tradeexecutor.strategy.strategy_module import read_strategy_module


logger = logging.getLogger(__name__)

#: Lighter mainnet market index for the ETH/USD perpetual used by this command.
ETH_PERP_MARKET_INDEX = 0
#: Journal format version for the private, non-secret recovery record.
JOURNAL_SCHEMA_VERSION = 1
#: Milliseconds used to make Lighter client order identifiers time based.
MILLISECONDS_PER_SECOND = 1_000
#: Grace period for Lighter and local wall-clock skew when locating a withdrawal.
LIGHTER_WITHDRAWAL_HISTORY_CLOCK_SKEW_SECONDS = 60
#: Public Lighter position polling interval.
LIGHTER_POSITION_POLL_SECONDS = 5
#: Default public deposit-observation timeout.
DEFAULT_LIGHTER_DEPOSIT_TIMEOUT = 900
#: Default additional test deposit with headroom over the ETH/USD minimum order size.
DEFAULT_LIGHTER_TEST_DEPOSIT_USDC = "20"
#: Default public position-observation timeout.
DEFAULT_LIGHTER_POSITION_TIMEOUT = 300
#: Default secure-withdrawal claimability timeout.
DEFAULT_LIGHTER_WITHDRAW_TIMEOUT = 3_600
#: Lighter USDC uses six decimal places for secure withdrawals.
USDC_DECIMALS = Decimal("0.000001")
#: Private operator records and non-secret journals must be owner readable only.
OWNER_ONLY_FILE_MODE = 0o600
#: Group and other permission bits which make a secret record unsafe.
UNSAFE_FILE_PERMISSION_MASK = 0o077


class LighterTestTradeError(RuntimeError):
    """Public operator error for the manual Lighter lifecycle."""


class LighterWithdrawalRejected(LighterTestTradeError):
    """Lighter definitively rejected a secure withdrawal before submission."""


@dataclass(slots=True)
class LighterOperatorRecord:
    """Public Lighter deployment metadata plus the delegated API signer."""

    #: Lagoon vault whose Safe owns the Lighter account.
    vault_address: str

    #: Guarded Safe address recorded during deployment.
    safe_address: str

    #: TradingStrategyModuleV0 address recorded during deployment.
    module_address: str

    #: Public Lighter account index.
    account_index: int

    #: Public delegated API-key slot.
    api_key_index: int

    #: Delegated signing key, deliberately excluded from representations.
    api_private_key: str = field(repr=False)

    def clear_private_key(self) -> None:
        """Discard this command's reference to the delegated signer."""
        self.api_private_key = ""


@dataclass(slots=True)
class LighterTestTradeConfig:
    """Resolved non-secret command configuration."""

    #: Executor identifier used for state and journal defaults.
    executor_id: str

    #: Existing executor state path.
    state_file: Path

    #: Owner-only Lighter operator record created by lagoon-deploy-vault.
    operator_record_file: Path

    #: Non-secret recovery journal path.
    journal_file: Path

    #: Additional Safe USDC moved to Lighter by this run.
    deposit_usdc: Decimal

    #: Approximate ETH perpetual notional in USDC.
    position_usdc: Decimal | None

    #: Maximum acceptable market-order slippage.
    max_slippage: Decimal

    #: Maximum Lighter public deposit observation time in seconds.
    deposit_timeout: int

    #: Maximum Lighter public position observation time in seconds.
    position_timeout: int

    #: Maximum secure-withdrawal observation time in seconds.
    withdraw_timeout: int

    #: Whether the interactive live-value confirmation is skipped.
    auto_approve: bool


def _public_sdk_error(operation: str, error: Exception) -> LighterTestTradeError:
    """Return a secret-free Lighter SDK error.

    SDK exceptions can carry request and response payloads, including signed
    data.  Logs and exceptions at this boundary therefore expose only the
    operation name and exception type.
    """
    return LighterTestTradeError(f"Lighter {operation} failed ({type(error).__name__})")


def _require_owner_only_file(path: Path) -> None:
    """Validate that a secret-bearing operator record is private.

    A root-run container may read an owner-only bind mount owned by its host
    user. Non-root processes must own the record themselves.
    """
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_mode & UNSAFE_FILE_PERMISSION_MASK:
        raise PermissionError(f"Refusing insecure Lighter operator record: {path}")
    if os.getuid() != 0 and info.st_uid != os.getuid():
        raise PermissionError(f"Refusing insecure Lighter operator record: {path}")


def load_lighter_operator_record(path: Path) -> LighterOperatorRecord:
    """Load one Ethereum Lighter operator record without logging its key."""
    _require_owner_only_file(path)
    payload = json.loads(path.read_text())
    deployments = payload.get("deployments")
    if isinstance(deployments, dict):
        deployment = deployments.get("ethereum")
    else:
        deployment = payload
    if not isinstance(deployment, dict):
        raise LighterTestTradeError("Lighter operator record has no Ethereum deployment")
    setup = deployment.get("lighter_account_setup")
    if not isinstance(setup, dict):
        raise LighterTestTradeError("Lighter operator record has no delegated API key")
    try:
        return LighterOperatorRecord(
            vault_address=str(deployment["vault_address"]),
            safe_address=str(deployment["safe_address"]),
            module_address=str(deployment["module_address"]),
            account_index=int(setup["account_index"]),
            api_key_index=int(setup["api_key_index"]),
            api_private_key=str(setup["private_key"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise LighterTestTradeError("Lighter operator record is incomplete") from error


def validate_public_deployment_record(
    *,
    executor_id: str,
    state_file: Path,
    operator: LighterOperatorRecord,
) -> None:
    """Cross-check public Lighter metadata when deployment produced its artefact."""
    deployment_file = state_file.with_name(f"{executor_id}.deployment.json")
    if not deployment_file.exists():
        return
    try:
        payload = json.loads(deployment_file.read_text())
        deployment = payload["deployments"]["ethereum"]
        deployment_record = payload.get("deployment_record", deployment)
        setup = deployment_record.get("lighter_account_setup")
        matches = (
            str(deployment["vault_address"]).lower() == operator.vault_address.lower()
            and str(deployment["module_address"]).lower() == operator.module_address.lower()
            and int(setup["account_index"]) == operator.account_index
            and int(setup["api_key_index"]) == operator.api_key_index
        )
        safe_address = deployment.get("safe_address")
        if safe_address is not None:
            matches = matches and str(safe_address).lower() == operator.safe_address.lower()
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise LighterTestTradeError("Public Lighter deployment artefact is incomplete") from error
    if not matches:
        raise LighterTestTradeError("Lighter operator record does not match the public deployment artefact")


def _write_journal(path: Path, journal: dict[str, Any]) -> None:
    """Atomically persist a non-secret owner-only recovery journal."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        os.fchmod(descriptor, OWNER_ONLY_FILE_MODE)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(journal, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
        os.chmod(path, OWNER_ONLY_FILE_MODE)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _load_or_create_journal(
    config: LighterTestTradeConfig,
    operator: LighterOperatorRecord,
    safe_balance: Decimal,
    equity: LighterEquity,
) -> dict[str, Any]:
    """Load a compatible journal or create a fresh lifecycle record."""
    if config.journal_file.exists():
        journal = json.loads(config.journal_file.read_text())
        if journal.get("vault_address", "").lower() != operator.vault_address.lower():
            raise LighterTestTradeError("Recovery journal belongs to another Lagoon vault")
        if int(journal.get("account_index", -1)) != operator.account_index:
            raise LighterTestTradeError("Recovery journal belongs to another Lighter account")
        if journal.get("phase") == "complete":
            raise LighterTestTradeError(
                f"Recovery journal is complete: {config.journal_file}. Choose a new journal path."
            )
        return journal

    journal = {
        "schema_version": JOURNAL_SCHEMA_VERSION,
        "phase": "created",
        "vault_address": operator.vault_address,
        "safe_address": operator.safe_address,
        "module_address": operator.module_address,
        "account_index": operator.account_index,
        "api_key_index": operator.api_key_index,
        "deposit_usdc": str(config.deposit_usdc),
        "baseline_safe_usdc": str(safe_balance),
        "baseline_lighter_collateral": str(equity.collateral),
        "baseline_lighter_equity": str(equity.get_total()),
        "baseline_lighter_available": str(equity.available_balance),
        "created_at": native_datetime_utc_now().isoformat(),
    }
    _write_journal(config.journal_file, journal)
    return journal


def _update_journal(
    config: LighterTestTradeConfig,
    journal: dict[str, Any],
    phase: str,
    **values: str | int,
) -> None:
    """Persist a lifecycle checkpoint before the next externally visible action.

    The withdrawal request is the deliberate exception: persist its intent
    before submitting an irreversible SDK request, so recovery never replays
    that request blindly.
    """
    journal["phase"] = phase
    journal["updated_at"] = native_datetime_utc_now().isoformat()
    journal.update(values)
    _write_journal(config.journal_file, journal)


def _signed_eth_position(account: dict[str, Any]) -> Decimal:
    """Read the signed ETH perpetual base position from a public account row."""
    for position in account.get("positions") or ():
        if int(position.get("market_id", -1)) != ETH_PERP_MARKET_INDEX:
            continue
        amount = Decimal(str(position["position"]))
        return amount if int(position.get("sign", 1)) >= 0 else -amount
    return Decimal(0)


def _assert_non_negative(value: Decimal, label: str) -> None:
    """Reject invalid public monetary observations before a mutation."""
    if not value.is_finite() or value < 0:
        raise LighterTestTradeError(f"Invalid {label}: value must be finite and non-negative")


async def resolve_eth_order(position_usdc: Decimal | None) -> tuple[int, Decimal, int, Decimal]:
    """Resolve an ETH market-order amount using Lighter's SDK market metadata."""
    api_client = lighter.ApiClient(configuration=lighter.Configuration(host=LIGHTER_API_URL))
    try:
        response = await lighter.OrderApi(api_client).order_book_details(
            market_id=ETH_PERP_MARKET_INDEX,
        )
        market = response.order_book_details[0]
        price = Decimal(str(market.last_trade_price))
        min_quote = Decimal(str(market.min_quote_amount))
        min_base = Decimal(str(market.min_base_amount))
        size_decimals = int(market.size_decimals)
        step = Decimal(1).scaleb(-int(market.supported_size_decimals))
        target = max(position_usdc or min_quote + 1, min_quote, min_base * price)
        base_size = (max(min_base, target / price) / step).to_integral_value(
            rounding=ROUND_CEILING,
        ) * step
        return int(base_size * Decimal(10**size_decimals)), base_size, size_decimals, base_size * price
    except Exception as error:  # noqa: BLE001 - SDK exception types vary by release
        raise _public_sdk_error("market metadata request", error) from None
    finally:
        await api_client.close()


async def submit_lighter_order(
    operator: LighterOperatorRecord,
    *,
    base_amount: int,
    max_slippage: Decimal,
    is_ask: bool,
    reduce_only: bool,
) -> None:
    """Submit one bounded Lighter market order without exposing SDK payloads."""
    client = None
    try:
        client = lighter.SignerClient(
            url=LIGHTER_API_URL,
            account_index=operator.account_index,
            api_private_keys={operator.api_key_index: operator.api_private_key},
        )
        if client.check_client():
            raise LighterTestTradeError("Lighter API key was rejected")
        _transaction, _response, error = await client.create_market_order_limited_slippage(
            market_index=ETH_PERP_MARKET_INDEX,
            client_order_index=int(time.time() * MILLISECONDS_PER_SECOND),
            base_amount=base_amount,
            max_slippage=float(max_slippage),
            is_ask=is_ask,
            reduce_only=reduce_only,
            api_key_index=operator.api_key_index,
        )
        if error:
            raise LighterTestTradeError("Lighter market order was rejected")
    except LighterTestTradeError:
        raise
    except Exception as error:  # noqa: BLE001 - do not log SDK payloads
        raise _public_sdk_error("order submission", error) from None
    finally:
        if client is not None:
            await client.close()


async def wait_for_eth_position(
    session: LighterSession,
    account_index: int,
    *,
    expected_long: bool,
    timeout: int,
) -> Decimal:
    """Wait for the public Lighter account to become long or flat."""
    deadline = time.monotonic() + timeout
    while True:
        position = _signed_eth_position(fetch_lighter_account_by_index(session, account_index))
        if position < 0:
            raise LighterTestTradeError("Lighter account unexpectedly has an ETH short")
        if expected_long and position > 0:
            return position
        if not expected_long and position == 0:
            return position
        if time.monotonic() >= deadline:
            state = "long" if expected_long else "flat"
            raise TimeoutError(f"ETH position did not become {state} within {timeout} seconds")
        await asyncio.sleep(LIGHTER_POSITION_POLL_SECONDS)


async def request_lighter_withdrawal(
    operator: LighterOperatorRecord,
    amount: Decimal,
) -> str:
    """Request one secure Lighter withdrawal without replaying it on recovery.

    The caller logs Lighter's current dynamic ``withdrawalDelay`` before this
    irreversible request. Treat that value as an operator estimate only; poll
    withdrawal history until the exact request becomes ``claimable``. Fast
    withdrawal requires the L1 account's EOA private key, which a Safe-owned
    Lighter account does not have.
    """
    client = None
    try:
        client = lighter.SignerClient(
            url=LIGHTER_API_URL,
            account_index=operator.account_index,
            api_private_keys={operator.api_key_index: operator.api_private_key},
        )
        _transaction, response, error = await client.withdraw(
            asset_id=client.ASSET_ID_USDC,
            route_type=client.ROUTE_PERP,
            amount=float(amount),
            api_key_index=operator.api_key_index,
        )
        if error or response is None:
            raise LighterWithdrawalRejected("Lighter secure withdrawal was rejected")
        transaction_id = str(response.tx_hash)
        logger.info("Secure Lighter withdrawal accepted: %s", transaction_id)
        return transaction_id
    except LighterTestTradeError:
        raise
    except Exception as error:  # noqa: BLE001 - SDK errors can contain signed content
        raise _public_sdk_error("secure withdrawal request", error) from None
    finally:
        if client is not None:
            await client.close()


async def wait_for_lighter_withdrawal_claimable(
    operator: LighterOperatorRecord,
    amount: Decimal,
    timeout: int,
    requested_at: int,
) -> Decimal:
    """Wait for an already-submitted secure withdrawal to become claimable.

    The live ``withdrawalDelay`` value does not replace this status check. The
    recorded request time excludes an older withdrawal of the same amount.
    A Safe-owned Lighter account uses this secure path because fast withdrawal
    requires an EOA private key that the Safe contract cannot provide.
    """
    client = None
    api_client = None
    try:
        client = lighter.SignerClient(
            url=LIGHTER_API_URL,
            account_index=operator.account_index,
            api_private_keys={operator.api_key_index: operator.api_private_key},
        )

        api_client = lighter.ApiClient(configuration=lighter.Configuration(host=LIGHTER_API_URL))
        transaction_api = lighter.TransactionApi(api_client)
        token_generation = 0

        def create_auth_token() -> tuple[str | None, object | None]:
            """Create a short-lived token without exposing its value."""
            nonlocal token_generation
            token_generation += 1
            action = "Creating" if token_generation == 1 else "Rotating"
            logger.info(
                "%s Lighter auth token for withdrawal history (generation %d)",
                action,
                token_generation,
            )
            return client.create_auth_token_with_expiry(
                api_key_index=operator.api_key_index,
            )

        auth_manager = LighterAuthTokenManager(token_factory=create_auth_token)
        deadline = time.monotonic() + timeout
        while True:
            history = await auth_manager.call(
                lambda auth: transaction_api.withdraw_history(
                    authorization=auth,
                    account_index=operator.account_index,
                ),
                operation_name="withdrawal history",
            )
            matches = [
                row
                for row in history.withdraws
                if row.asset_id == client.ASSET_ID_USDC
                and abs(Decimal(str(row.amount)) - amount) <= USDC_DECIMALS
                and row.timestamp >= requested_at - LIGHTER_WITHDRAWAL_HISTORY_CLOCK_SKEW_SECONDS
            ]
            if len(matches) > 1:
                raise LighterTestTradeError("Ambiguous Lighter secure-withdrawal history")
            if len(matches) == 1 and str(matches[0].status).lower() == "claimable":
                return Decimal(str(matches[0].amount))
            if time.monotonic() >= deadline:
                raise TimeoutError("Lighter withdrawal did not become claimable")
            await asyncio.sleep(LIGHTER_POSITION_POLL_SECONDS)
    except LighterTestTradeError:
        raise
    except Exception as error:  # noqa: BLE001 - SDK errors can contain signed content
        raise _public_sdk_error("secure withdrawal history", error) from None
    finally:
        if client is not None:
            await client.close()
        if api_client is not None:
            await api_client.close()


def _get_lighter_position(state: State) -> TradingPosition:
    """Return the single configured Lighter exchange-account position."""
    positions = [
        position
        for position in state.portfolio.get_open_and_frozen_positions()
        if position.is_exchange_account()
        and position.pair.get_exchange_account_protocol() == LIGHTER_PROTOCOL
    ]
    if len(positions) != 1:
        raise LighterTestTradeError(f"Expected exactly one Lighter position, got {len(positions)}")
    return positions[0]


def checkpoint_lagoon_nav(
    *,
    state: State,
    store: JSONFileStore,
    sync_model: LagoonVaultSyncModel,
    reserve_asset: AssetIdentifier,
    session: LighterSession,
    operator: LighterOperatorRecord,
    expected_long: bool,
) -> None:
    """Synchronise external equity, Safe reserve and on-chain Lagoon NAV."""
    position = _get_lighter_position(state)
    equity = fetch_lighter_total_equity(session, operator.account_index)
    _assert_non_negative(equity.get_total(), "Lighter equity")
    _assert_non_negative(equity.available_balance, "Lighter available balance")
    observed_position = _signed_eth_position(
        fetch_lighter_account_by_index(session, operator.account_index),
    )
    if expected_long and observed_position <= 0:
        raise LighterTestTradeError("Expected an open ETH long before NAV synchronisation")
    if not expected_long and observed_position != 0:
        raise LighterTestTradeError("Expected a flat ETH position before NAV synchronisation")

    valuator = ExchangeAccountValuator(
        ExchangeAccountPricingModel(
            lambda pair, **_kwargs: fetch_lighter_total_equity(
                session,
                int(pair.get_exchange_account_id()),
            ).get_total(),
        ),
        web3=sync_model.web3,
    )
    valuator(native_datetime_utc_now(), position)
    sync_model.sync_treasury(
        native_datetime_utc_now(),
        state,
        supported_reserves=[reserve_asset],
        post_valuation=True,
    )
    nav = Decimal(str(sync_model.calculate_valuation(state)))
    _assert_non_negative(nav, "Lagoon NAV")
    store.sync(state)
    logger.info(
        "NAV checkpoint complete: Safe USDC=%s, Lighter equity=%s, Lagoon NAV=%s",
        sync_model.vault.denomination_token.fetch_balance_of(sync_model.vault.safe_address),
        equity.get_total(),
        nav,
    )


def _resolve_config(
    *,
    executor_id: str,
    state_file: Path | None,
    operator_record_file: Path | None,
    journal_file: Path | None,
    deposit_usdc: str,
    position_usdc: str | None,
    max_slippage: str,
    deposit_timeout: int,
    position_timeout: int,
    withdraw_timeout: int,
    auto_approve: bool,
) -> LighterTestTradeConfig:
    """Parse command options without using binary floats for money values."""
    resolved_state_file = state_file or Path(f"state/{executor_id}.json")
    resolved_operator_record = operator_record_file
    if resolved_operator_record is None:
        raise LighterTestTradeError("LIGHTER_OPERATOR_RECORD_FILE is required")
    try:
        resolved_deposit = Decimal(deposit_usdc)
        resolved_position = Decimal(position_usdc) if position_usdc else None
        resolved_slippage = Decimal(max_slippage)
    except Exception as error:  # noqa: BLE001 - Decimal exposes several public parsing errors
        raise LighterTestTradeError("Lighter money options must be decimal values") from error
    if resolved_deposit < LIGHTER_MIN_MAINNET_USDC:
        raise LighterTestTradeError(
            f"LIGHTER_TEST_DEPOSIT_USDC must be at least {LIGHTER_MIN_MAINNET_USDC}",
        )
    if resolved_position is not None and resolved_position <= 0:
        raise LighterTestTradeError("LIGHTER_TEST_POSITION_USDC must be positive")
    if not Decimal(0) < resolved_slippage <= Decimal(1):
        raise LighterTestTradeError("LIGHTER_TEST_MAX_SLIPPAGE must be between zero and one")
    if min(deposit_timeout, position_timeout, withdraw_timeout) <= 0:
        raise LighterTestTradeError("Lighter timeout values must be positive")
    return LighterTestTradeConfig(
        executor_id=executor_id,
        state_file=resolved_state_file,
        operator_record_file=resolved_operator_record,
        journal_file=journal_file or resolved_state_file.with_name(
            f"{executor_id}.lighter-test-trade.json",
        ),
        deposit_usdc=resolved_deposit,
        position_usdc=resolved_position,
        max_slippage=resolved_slippage,
        deposit_timeout=deposit_timeout,
        position_timeout=position_timeout,
        withdraw_timeout=withdraw_timeout,
        auto_approve=auto_approve,
    )


@app.command()
@shared_options.with_json_rpc_options()
def lagoon_lighter_test_trade(
    id: str = shared_options.id,
    strategy_file: Path = shared_options.strategy_file,
    state_file: Path | None = shared_options.state_file,
    log_level: str | None = shared_options.log_level,
    rpc_kwargs: dict | None = None,
    private_key: str | None = shared_options.private_key,
    asset_management_mode: AssetManagementMode | None = shared_options.asset_management_mode,
    vault_address: str | None = shared_options.vault_address,
    vault_adapter_address: str | None = shared_options.vault_adapter_address,
    unit_testing: bool = shared_options.unit_testing,
    simulate: bool = shared_options.simulate,
    lighter_operator_record_file: Path | None = Option(
        None,
        envvar="LIGHTER_OPERATOR_RECORD_FILE",
        help="Mode-0600 JSON record emitted by lagoon-deploy-vault.",
    ),
    lighter_account_index: int | None = Option(
        None,
        envvar="LIGHTER_ACCOUNT_INDEX",
        help="Optional public account index to cross-check the operator record.",
    ),
    lighter_test_deposit_usdc: str = Option(
        DEFAULT_LIGHTER_TEST_DEPOSIT_USDC,
        envvar="LIGHTER_TEST_DEPOSIT_USDC",
        help=(
            "Available Safe USDC to move to Lighter; defaults to 20 USDC. "
            "Fund Safe first with scripts/lagoon/deposit-and-settle.py."
        ),
    ),
    lighter_test_position_usdc: str | None = Option(
        None,
        envvar="LIGHTER_TEST_POSITION_USDC",
        help="Approximate ETH perpetual notional in USDC.",
    ),
    lighter_test_max_slippage: str = Option(
        "0.02",
        envvar="LIGHTER_TEST_MAX_SLIPPAGE",
        help="Maximum Lighter market-order slippage.",
    ),
    lighter_test_journal_file: Path | None = Option(
        None,
        envvar="LIGHTER_TEST_JOURNAL_FILE",
        help="Owner-only non-secret recovery journal path.",
    ),
    lighter_deposit_timeout: int = Option(
        DEFAULT_LIGHTER_DEPOSIT_TIMEOUT,
        envvar="LIGHTER_DEPOSIT_TIMEOUT",
    ),
    lighter_position_timeout: int = Option(
        DEFAULT_LIGHTER_POSITION_TIMEOUT,
        envvar="LIGHTER_POSITION_TIMEOUT",
    ),
    lighter_withdraw_timeout: int = Option(
        DEFAULT_LIGHTER_WITHDRAW_TIMEOUT,
        envvar="LIGHTER_WITHDRAW_TIMEOUT",
    ),
    auto_approve: bool = Option(False, envvar="AUTO_APPROVE"),
) -> None:
    """Deposit, trade, withdraw and NAV-sync one funded Lagoon Lighter vault."""
    if simulate:
        raise LighterTestTradeError("lagoon-lighter-test-trade does not support SIMULATE=true")
    if asset_management_mode != AssetManagementMode.lagoon:
        raise LighterTestTradeError("ASSET_MANAGEMENT_MODE must be lagoon")
    if not private_key or not vault_address or not vault_adapter_address:
        raise LighterTestTradeError(
            "PRIVATE_KEY, VAULT_ADDRESS and VAULT_ADAPTER_ADDRESS are required",
        )

    executor_id = prepare_executor_id(id, strategy_file)
    setup_logging(log_level=log_level)
    config = _resolve_config(
        executor_id=executor_id,
        state_file=state_file,
        operator_record_file=lighter_operator_record_file,
        journal_file=lighter_test_journal_file,
        deposit_usdc=lighter_test_deposit_usdc,
        position_usdc=lighter_test_position_usdc,
        max_slippage=lighter_test_max_slippage,
        deposit_timeout=lighter_deposit_timeout,
        position_timeout=lighter_position_timeout,
        withdraw_timeout=lighter_withdraw_timeout,
        auto_approve=auto_approve,
    )
    operator = load_lighter_operator_record(config.operator_record_file)
    validate_public_deployment_record(
        executor_id=config.executor_id,
        state_file=config.state_file,
        operator=operator,
    )
    if operator.vault_address.lower() != vault_address.lower() or operator.module_address.lower() != vault_adapter_address.lower():
        raise LighterTestTradeError("Lighter operator record does not match the configured vault")
    if lighter_account_index is not None:
        if lighter_account_index != operator.account_index:
            raise LighterTestTradeError("LIGHTER_ACCOUNT_INDEX does not match the Lighter operator record")

    module = read_strategy_module(strategy_file)
    web3config = create_single_chain_web3_config(mod=module, **(rpc_kwargs or {}))
    web3 = web3config.get_default()
    if web3.eth.chain_id != LIGHTER_ETHEREUM_DEPLOYMENT_CHAIN_ID:
        raise LighterTestTradeError("lagoon-lighter-test-trade is Ethereum mainnet only")
    hot_wallet = create_hot_wallet(web3, private_key)
    vault = LagoonVault(
        web3,
        VaultSpec(web3.eth.chain_id, vault_address),
        trading_strategy_module_address=vault_adapter_address,
        default_block_identifier="latest",
        require_denomination_token=True,
    )
    if vault.safe_address.lower() != operator.safe_address.lower():
        raise LighterTestTradeError("Lighter operator record does not match the deployed Safe")
    usdc = fetch_erc20_details(web3, LIGHTER_USDC_ETHEREUM, chain_id=web3.eth.chain_id)
    if vault.denomination_token.address.lower() != usdc.address.lower():
        raise LighterTestTradeError("Lagoon vault denomination token must be native Ethereum USDC")

    reserve_asset = AssetIdentifier(
        chain_id=web3.eth.chain_id,
        address=usdc.address,
        token_symbol=usdc.symbol,
        decimals=usdc.decimals,
    )
    session = create_lighter_session()
    lock_file = config.journal_file.with_suffix(config.journal_file.suffix + ".lock")
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with lock_file.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception:
            session.close()
            operator.clear_private_key()
            web3config.close()
            raise
        try:
            safe_balance = usdc.fetch_balance_of(vault.safe_address)
            equity = fetch_lighter_total_equity(session, operator.account_index)
            _assert_non_negative(safe_balance, "Safe USDC balance")
            _assert_non_negative(equity.get_total(), "Lighter equity")
            _assert_non_negative(equity.available_balance, "Lighter available balance")
            journal = _load_or_create_journal(config, operator, safe_balance, equity)
            phase = str(journal["phase"])
            public_position = _signed_eth_position(
                fetch_lighter_account_by_index(session, operator.account_index),
            )
            if phase not in {"created", "deposited", "opened", "closed", "withdrawal_requested", "withdrawn", "complete"}:
                raise LighterTestTradeError(f"Unknown Lighter recovery phase: {phase}")
            if phase == "created":
                baseline_collateral = Decimal(str(journal["baseline_lighter_collateral"]))
                baseline_safe_balance = Decimal(str(journal["baseline_safe_usdc"]))
                if public_position != 0:
                    raise LighterTestTradeError("Fresh test run refuses an existing Lighter ETH position")
                if safe_balance != baseline_safe_balance:
                    raise LighterTestTradeError(
                        "Safe USDC changed before the deposit was journalled; refusing to retry it",
                    )
                if equity.collateral != baseline_collateral:
                    raise LighterTestTradeError(
                        "Lighter collateral changed before the deposit was journalled; refusing to retry it",
                    )
            elif phase == "opened" and public_position <= 0:
                raise LighterTestTradeError("Journal says the ETH long is open, but Lighter reports no long")
            elif phase != "opened" and public_position != 0:
                raise LighterTestTradeError(
                    f"Journal phase {phase} requires a flat ETH position; inspect Lighter before resuming",
                )
            if safe_balance < config.deposit_usdc and phase == "created":
                raise LighterTestTradeError(
                    f"Lagoon Safe has {safe_balance} USDC, but the requested Lighter deposit requires "
                    f"{config.deposit_usdc} USDC",
                )
            logger.info("Step 1: inspected vault %s and Lighter account %d", vault.address, operator.account_index)
            logger.info("Safe USDC=%s, Lighter equity=%s, ETH position=%s", safe_balance, equity.get_total(), public_position)
            if not config.auto_approve and not unit_testing:
                typer.confirm("Continue with the live Lighter test trade?", abort=True)

            _state_path, store = resolve_state_store(executor_id, config.state_file)
            valuation_func = create_lighter_vault_valuation_func(
                web3,
                vault.safe_address,
                reserve_asset,
                operator.account_index,
                session,
            )
            sync_model = LagoonVaultSyncModel(
                vault=vault,
                hot_wallet=hot_wallet,
                unit_testing=unit_testing,
                calculate_valuation_func=valuation_func,
            )
            if store.is_pristine():
                state = State()
                sync_model.sync_initial(state, reserve_asset=reserve_asset, reserve_token_price=1.0)
            else:
                state = store.load()
            lighter_positions = [
                position
                for position in state.portfolio.get_open_and_frozen_positions()
                if position.is_exchange_account()
                and position.pair.get_exchange_account_protocol() == LIGHTER_PROTOCOL
            ]
            if len(lighter_positions) == 0:
                pair = create_lighter_exchange_account_pair(
                    quote=reserve_asset,
                    account_index=operator.account_index,
                )
                open_exchange_account_position(
                    state=state,
                    strategy_cycle_at=native_datetime_utc_now(),
                    pair=pair,
                    reserve_currency=reserve_asset,
                    notes="Manual Lagoon Lighter test trade account",
                )
                store.sync(state)
            elif len(lighter_positions) != 1:
                raise LighterTestTradeError("Executor state has conflicting Lighter exchange positions")
            elif int(lighter_positions[0].pair.get_exchange_account_id()) != operator.account_index:
                raise LighterTestTradeError("Executor state Lighter account does not match the operator record")

            if phase == "created":
                logger.info("Step 2: depositing %s USDC from Safe to Lighter", config.deposit_usdc)
                deposit_tx_hash = deposit_usdc_from_lagoon_safe_into_lighter(
                    web3,
                    hot_wallet,
                    vault=vault,
                    usdc=usdc,
                    deposit_usdc=config.deposit_usdc,
                )
                wait_for_lighter_collateral(
                    session,
                    operator.account_index,
                    equity.collateral + config.deposit_usdc,
                    timeout=config.deposit_timeout,
                )
                _update_journal(config, journal, "deposited", deposit_tx_hash=deposit_tx_hash)
                phase = "deposited"
                logger.info("Step 3: synchronising NAV after deposit")
                checkpoint_lagoon_nav(
                    state=state,
                    store=store,
                    sync_model=sync_model,
                    reserve_asset=reserve_asset,
                    session=session,
                    operator=operator,
                    expected_long=False,
                )

            if phase == "deposited":
                base_amount, base_size, size_decimals, notional = asyncio.run(
                    resolve_eth_order(config.position_usdc),
                )
                logger.info("Step 4: opening ETH/USD long: %s ETH (~%s USDC)", base_size, notional)
                asyncio.run(
                    submit_lighter_order(
                        operator,
                        base_amount=base_amount,
                        max_slippage=config.max_slippage,
                        is_ask=False,
                        reduce_only=False,
                    ),
                )
                filled_size = asyncio.run(
                    wait_for_eth_position(
                        session,
                        operator.account_index,
                        expected_long=True,
                        timeout=config.position_timeout,
                    ),
                )
                _update_journal(
                    config,
                    journal,
                    "opened",
                    filled_eth_size=str(filled_size),
                    size_decimals=size_decimals,
                )
                phase = "opened"
                logger.info("Step 5: synchronising NAV with ETH long open")
                checkpoint_lagoon_nav(
                    state=state,
                    store=store,
                    sync_model=sync_model,
                    reserve_asset=reserve_asset,
                    session=session,
                    operator=operator,
                    expected_long=True,
                )

            if phase == "opened":
                filled_size = Decimal(str(journal["filled_eth_size"]))
                size_decimals = int(journal["size_decimals"])
                close_amount = int(
                    (filled_size * Decimal(10**size_decimals)).to_integral_value(
                        rounding=ROUND_CEILING,
                    ),
                )
                logger.info("Step 6: closing ETH/USD long: %s ETH", filled_size)
                asyncio.run(
                    submit_lighter_order(
                        operator,
                        base_amount=close_amount,
                        max_slippage=config.max_slippage,
                        is_ask=True,
                        reduce_only=True,
                    ),
                )
                asyncio.run(
                    wait_for_eth_position(
                        session,
                        operator.account_index,
                        expected_long=False,
                        timeout=config.position_timeout,
                    ),
                )
                _update_journal(config, journal, "closed")
                phase = "closed"
                logger.info("Step 7: synchronising NAV after closing ETH/USD long")
                checkpoint_lagoon_nav(
                    state=state,
                    store=store,
                    sync_model=sync_model,
                    reserve_asset=reserve_asset,
                    session=session,
                    operator=operator,
                    expected_long=False,
                )

            if phase == "closed":
                logger.info("Step 8: requesting Lighter withdrawal to the Lagoon Safe")
                current_equity = fetch_lighter_total_equity(session, operator.account_index)
                baseline_equity = Decimal(str(journal["baseline_lighter_equity"]))
                withdraw_amount = max(
                    current_equity.available_balance - baseline_equity,
                    Decimal(0),
                ).quantize(USDC_DECIMALS, rounding=ROUND_DOWN)
                if withdraw_amount < LIGHTER_MIN_MAINNET_USDC:
                    raise LighterTestTradeError("Lighter withdrawal amount is below the 1 USDC minimum")
                try:
                    withdrawal_delay = fetch_lighter_withdrawal_delay(session)
                except Exception as error:  # noqa: BLE001 - informational data must not block safe recovery
                    logger.warning(
                        "Could not read the current Lighter secure-withdrawal delay (%s); continuing with the withdrawal request",
                        type(error).__name__,
                    )
                else:
                    logger.info(
                        "Lighter reports a current secure-withdrawal delay of %d seconds before the withdrawal request",
                        withdrawal_delay,
                    )
                withdrawal_requested_at = int(time.time())
                _update_journal(
                    config,
                    journal,
                    "withdrawal_requested",
                    withdrawal_usdc=str(withdraw_amount),
                    withdrawal_requested_at=withdrawal_requested_at,
                )
                phase = "withdrawal_requested"
                asyncio.run(request_lighter_withdrawal(operator, withdraw_amount))

            if phase == "withdrawal_requested":
                withdraw_amount = Decimal(str(journal["withdrawal_usdc"]))
                logger.info("Step 9: waiting for %s USDC to become claimable", withdraw_amount)
                safe_before = usdc.fetch_balance_of(vault.safe_address)
                claimable = asyncio.run(
                    wait_for_lighter_withdrawal_claimable(
                        operator,
                        withdraw_amount,
                        config.withdraw_timeout,
                        int(journal["withdrawal_requested_at"]),
                    ),
                )
                hot_wallet.sync_nonce(web3)
                claim_tx_hash = claim_usdc_to_lagoon_safe_from_lighter(
                    web3,
                    hot_wallet,
                    vault=vault,
                    usdc=usdc,
                    claimable_usdc=claimable,
                )
                safe_after = usdc.fetch_balance_of(vault.safe_address)
                if usdc.convert_to_raw(safe_after - safe_before) != usdc.convert_to_raw(claimable):
                    raise LighterTestTradeError("Safe did not receive the claimed Lighter withdrawal")
                _update_journal(
                    config,
                    journal,
                    "withdrawn",
                    withdrawal_usdc=str(claimable),
                    claim_tx_hash=claim_tx_hash,
                )
                phase = "withdrawn"

            if phase == "withdrawn":
                logger.info("Step 10: synchronising final Lagoon NAV")
                checkpoint_lagoon_nav(
                    state=state,
                    store=store,
                    sync_model=sync_model,
                    reserve_asset=reserve_asset,
                    session=session,
                    operator=operator,
                    expected_long=False,
                )
                _update_journal(config, journal, "complete")
                logger.info("Step 11: Lighter Lagoon test trade completed successfully")
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            session.close()
            operator.clear_private_key()
            web3config.close()
