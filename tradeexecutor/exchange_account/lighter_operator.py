"""Secret-safe Lighter operator helpers used by routed custody transfers."""

import asyncio
import json
import logging
import os
import stat
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

import lighter

from eth_defi.lighter.api import LIGHTER_STATE_POLL_SECONDS
from eth_defi.lighter.constants import LIGHTER_API_URL
from eth_defi.lighter.sdk import LighterAuthTokenManager, create_lighter_signer_client
from eth_defi.lighter.transfer import (
    LighterTransferError,
    LighterTransferRecord,
    create_lighter_auth_token_manager,
    fetch_lighter_transfer_fee,
    fetch_lighter_transfer_record,
    request_lighter_transfer,
    wait_for_lighter_transfer,
)

logger = logging.getLogger(__name__)

#: File mode required for records containing a delegated API private key.
OWNER_ONLY_FILE_MODE = 0o600
#: Group and other permission bits make a private operator record unsafe.
UNSAFE_FILE_PERMISSION_MASK = 0o077
#: Lighter history timestamps are allowed this much clock skew.
WITHDRAWAL_HISTORY_CLOCK_SKEW_SECONDS = 60
#: Lighter USDC precision used when matching withdrawal history amounts.
USDC_PRECISION = Decimal("0.000001")


@dataclass(slots=True)
class LighterOperatorRecord:
    """Public Lighter deployment data and a delegated API signer."""

    #: Lagoon vault address associated with the delegated Lighter key.
    vault_address: str

    #: Safe that owns Lighter collateral and receives withdrawals.
    safe_address: str

    #: Safe trading module authorised to claim Lighter withdrawals.
    module_address: str

    #: Public Lighter account identifier.
    account_index: int

    #: Delegated Lighter API-key slot.
    api_key_index: int

    #: Delegated Lighter API private key, excluded from representations.
    api_private_key: str = field(repr=False)


def load_lighter_operator_record(path: Path) -> LighterOperatorRecord:
    """Load an owner-only operator record without logging its private key.

    :param path:
        Private JSON record created during Lighter Lagoon deployment.
    :return:
        Validated public deployment details and delegated API signer.
    """
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_mode & UNSAFE_FILE_PERMISSION_MASK:
        raise PermissionError(f"Refusing insecure Lighter operator record: {path}")
    if os.getuid() != 0 and info.st_uid != os.getuid():
        raise PermissionError(f"Refusing operator record owned by another user: {path}")

    payload = json.loads(path.read_text())
    deployment = payload.get("deployments", {}).get("ethereum", payload)
    setup = deployment.get("lighter_account_setup")
    if not isinstance(setup, dict):
        raise ValueError("Lighter operator record has no delegated API key")
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
        raise ValueError("Lighter operator record is incomplete") from error


def _public_sdk_error(operation: str, error: Exception) -> RuntimeError:
    """Convert an SDK exception to a secret-free error."""
    return RuntimeError(f"Lighter {operation} failed ({type(error).__name__})")


async def request_lighter_withdrawal(
    operator: LighterOperatorRecord,
    amount: Decimal,
) -> str:
    """Request one secure withdrawal and return its public request id.

    :param operator:
        Delegated signer and Safe deployment details.
    :param amount:
        Positive USDC amount to request from Lighter.
    :return:
        Public Lighter withdrawal request identifier.
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
            raise RuntimeError("Lighter secure withdrawal was rejected")
        request_id = str(response.tx_hash)
        logger.info("Lighter secure withdrawal accepted, request id %s", request_id)
        return request_id
    except RuntimeError:
        raise
    except Exception as error:  # noqa: BLE001 - SDK exception types vary by release
        raise _public_sdk_error("secure withdrawal request", error) from None
    finally:
        if client is not None:
            await client.close()


async def wait_for_lighter_withdrawal_claimable(
    operator: LighterOperatorRecord,
    amount: Decimal,
    timeout: int,
    requested_at: int,
    request_id: str,
) -> Decimal:
    """Poll authenticated withdrawal history until one request is claimable.

    :param operator:
        Delegated signer and Lighter account details.
    :param amount:
        Requested USDC amount used to verify the returned history entry.
    :param timeout:
        Maximum seconds to wait for the withdrawal delay.
    :param requested_at:
        UTC Unix timestamp at which the withdrawal was requested.
    :param request_id:
        Public identifier returned when the withdrawal was requested.
    :return:
        USDC amount that Lighter made claimable.
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
            nonlocal token_generation
            token_generation += 1
            logger.info("Creating Lighter withdrawal-history auth token generation %d", token_generation)
            return client.create_auth_token_with_expiry(api_key_index=operator.api_key_index)

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
                if str(getattr(row, "tx_hash", "")) == request_id
                or (
                    row.asset_id == client.ASSET_ID_USDC
                    and abs(Decimal(str(row.amount)) - amount) <= USDC_PRECISION
                    and row.timestamp >= requested_at - WITHDRAWAL_HISTORY_CLOCK_SKEW_SECONDS
                )
            ]
            if len(matches) > 1:
                raise RuntimeError("Ambiguous Lighter secure-withdrawal history")
            if len(matches) == 1:
                row = matches[0]
                status = str(row.status).lower()
                if status == "claimable":
                    return Decimal(str(row.amount))
                if status in {"failed", "rejected", "cancelled"}:
                    raise RuntimeError("Lighter secure withdrawal was rejected")

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Lighter withdrawal request {request_id} did not become claimable within {timeout} seconds"
                )
            logger.info(
                "Waiting for Lighter withdrawal request %s to become claimable, %.0f seconds remaining",
                request_id,
                remaining,
            )
            await asyncio.sleep(min(LIGHTER_STATE_POLL_SECONDS, remaining))
    except (RuntimeError, TimeoutError):
        raise
    except Exception as error:  # noqa: BLE001 - SDK exception types vary by release
        raise _public_sdk_error("withdrawal history", error) from None
    finally:
        if api_client is not None:
            await api_client.close()
        if client is not None:
            await client.close()


@dataclass(slots=True, frozen=True)
class LighterFundingRequest:
    """Public result of submitting a master to sub-account transfer."""

    #: L2 transaction hash used to confirm the transfer from history.
    tx_hash: str

    #: Fee charged to the master account on top of the amount.
    fee: Decimal


async def request_lighter_funding_transfer(
    operator: LighterOperatorRecord,
    to_account_index: int,
    amount: Decimal,
) -> LighterFundingRequest:
    """Fund a sub-account from the master account with the master API key.

    The submission is not idempotent. Persist the returned hash before
    waiting for confirmation.

    :param operator:
        Master account signer.
    :param to_account_index:
        Sub-account owned by the same Safe.
    :param amount:
        USDC credited to the sub-account.
    :return:
        Transaction hash and the fee charged on top of the amount.
    """
    client = None
    try:
        client = create_lighter_signer_client(operator.account_index, operator.api_key_index, operator.api_private_key)
        auth = create_lighter_auth_token_manager(client, operator.api_key_index)
        fee = await fetch_lighter_transfer_fee(client, auth, to_account_index=to_account_index)
        tx_hash = await request_lighter_transfer(
            client,
            to_account_index=to_account_index,
            amount=amount,
            fee=fee,
            api_key_index=operator.api_key_index,
        )
        return LighterFundingRequest(tx_hash=tx_hash, fee=fee)
    except LighterTransferError:
        raise
    except Exception as error:  # noqa: BLE001 - SDK exception types vary by release
        raise _public_sdk_error("account transfer", error) from None
    finally:
        if client is not None:
            await client.close()


async def wait_for_lighter_funding_transfer(
    operator: LighterOperatorRecord,
    tx_hash: str,
    timeout: int,
) -> LighterTransferRecord:
    """Wait until a funding transfer appears in the master's transfer history.

    :param operator:
        Master account signer, used for the authenticated history read.
    :param tx_hash:
        Hash returned by :py:func:`request_lighter_funding_transfer`.
    :param timeout:
        Maximum seconds to wait.
    :return:
        Executed transfer record.
    """
    client = None
    try:
        client = create_lighter_signer_client(operator.account_index, operator.api_key_index, operator.api_private_key)
        auth = create_lighter_auth_token_manager(client, operator.api_key_index)
        return await wait_for_lighter_transfer(client, auth, account_index=operator.account_index, tx_hash=tx_hash, timeout=timeout)
    except (LighterTransferError, TimeoutError):
        raise
    except Exception as error:  # noqa: BLE001 - SDK exception types vary by release
        raise _public_sdk_error("transfer history", error) from None
    finally:
        if client is not None:
            await client.close()


async def fetch_lighter_funding_transfer(
    operator: LighterOperatorRecord,
    tx_hash: str,
) -> LighterTransferRecord | None:
    """Read one funding transfer from history without waiting.

    Used by reconciliation after an interrupted run.

    :param operator:
        Master account signer, used for the authenticated history read.
    :param tx_hash:
        Hash recorded when the transfer was submitted.
    :return:
        Executed transfer, or ``None`` if it is not in history.
    """
    client = None
    try:
        client = create_lighter_signer_client(operator.account_index, operator.api_key_index, operator.api_private_key)
        auth = create_lighter_auth_token_manager(client, operator.api_key_index)
        return await fetch_lighter_transfer_record(client, auth, account_index=operator.account_index, tx_hash=tx_hash)
    except LighterTransferError:
        raise
    except Exception as error:  # noqa: BLE001 - SDK exception types vary by release
        raise _public_sdk_error("transfer history", error) from None
    finally:
        if client is not None:
            await client.close()
