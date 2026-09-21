"""Lighter exchange-account support.

Lighter accounts are external to the EVM portfolio. The account index is
stored on a synthetic exchange-account pair and the public Lighter REST API is
used for valuation without credentials. Automatic withdrawals use a separate
owner-only operator record for authenticated requests.
"""

import logging
import math
from collections.abc import Iterable
from decimal import Decimal, InvalidOperation
from typing import Any, Callable

from eth_defi.lighter.constants import LIGHTER_L1_CONTRACT
from eth_defi.lighter.session import LighterSession, create_lighter_session
from eth_defi.lighter.valuation import fetch_lighter_total_equity
from eth_defi.token import fetch_erc20_details
from web3 import Web3

from tradeexecutor.state.identifier import (
    AssetIdentifier,
    TradingPairIdentifier,
    TradingPairKind,
)

logger = logging.getLogger(__name__)


#: Stable synthetic asset symbol persisted in exchange-account state.
LIGHTER_ACCOUNT_SYMBOL = "LIGHTER-ACCOUNT"
#: Dispatcher identifier shared by accounting, valuation and runtime discovery.
LIGHTER_PROTOCOL = "lighter"
#: Only Lighter deployment supported by the initial integration.
LIGHTER_DEPLOYMENT = "ethereum"
#: Explicit allowlist for public deployment artefacts and reports.
LIGHTER_PUBLIC_METADATA_LABELS = {
    "account_index": "Account index",
    "api_key_index": "API-key index",
    "public_key": "Public key",
    "activation_amount": "Activation amount",
    "deposit_tx_hash": "Deposit transaction hash",
    "change_pubkey_tx_hash": "Change public-key transaction hash",
    "observed_collateral": "Observed collateral",
}


def get_public_lighter_metadata(metadata: dict[Any, Any]) -> dict[str, Any]:
    """Copy only fields approved for public Lighter deployment reports."""
    return {
        key: metadata[key]
        for key in LIGHTER_PUBLIC_METADATA_LABELS
        if key in metadata
    }


class LighterEquityInvariantError(ValueError):
    """Raised when the public Lighter API reports an invalid equity value."""


class NegativeLighterEquityError(LighterEquityInvariantError):
    """Raised when the public Lighter API reports impossible negative equity."""


def create_lighter_exchange_account_pair(
    quote: AssetIdentifier,
    account_index: int,
    is_testnet: bool = False,
) -> TradingPairIdentifier:
    """Create a synthetic pair representing one Lighter account.

    :param quote:
        Reserve/quote asset (normally Ethereum USDC).
    :param account_index:
        Public Lighter account index.
    :param is_testnet:
        Retained for parity with the other exchange-account pair factories.
        The first integration supports the canonical Ethereum deployment.
    :return:
        Exchange-account pair with public Lighter metadata only.
    """
    if type(account_index) is not int or account_index < 0:
        raise ValueError(f"Lighter account index must be a non-negative integer: {account_index!r}")

    # Lighter's L1 contract is the stable, canonical identity for this
    # synthetic asset.  It is not used as an ERC-20 contract by the reader.
    base = AssetIdentifier(
        chain_id=quote.chain_id,
        address=LIGHTER_L1_CONTRACT,
        token_symbol=LIGHTER_ACCOUNT_SYMBOL,
        decimals=6,
    )
    # Keep pair addresses valid and deployment-stable, as GMX does for its
    # exchange-account pair. The account index belongs in ``other_data`` and
    # must not be encoded as a short, non-address string here.
    return TradingPairIdentifier(
        base=base,
        quote=quote,
        pool_address=LIGHTER_L1_CONTRACT,
        exchange_address=LIGHTER_L1_CONTRACT,
        internal_id=1,
        internal_exchange_id=1,
        fee=0.0,
        kind=TradingPairKind.exchange_account,
        exchange_name="Lighter",
        other_data={
            "exchange_protocol": LIGHTER_PROTOCOL,
            "exchange_subaccount_id": account_index,
            "exchange_is_testnet": is_testnet,
            "lighter_deployment": LIGHTER_DEPLOYMENT,
        },
    )


def has_lighter_exchange_account_pairs(strategy_universe: Any) -> bool:
    """Return whether a strategy universe contains a Lighter account pair."""
    iterate_pairs = getattr(strategy_universe, "iterate_pairs", None)
    if iterate_pairs is None:
        return False
    return any(
        pair.is_exchange_account()
        and pair.get_exchange_account_protocol() == LIGHTER_PROTOCOL
        for pair in iterate_pairs()
    )


def validate_lighter_exchange_account_pairs(
    pairs: Iterable[TradingPairIdentifier],
) -> list[TradingPairIdentifier]:
    """Return Lighter pairs after validating the external-account topology."""
    exchange_account_pairs = [pair for pair in pairs if pair.is_exchange_account()]
    lighter_pairs = [
        pair
        for pair in exchange_account_pairs
        if pair.get_exchange_account_protocol() == LIGHTER_PROTOCOL
    ]
    if not lighter_pairs:
        return []
    if len(lighter_pairs) != 1:
        raise ValueError(
            "Lighter strategies must contain exactly one exchange-account pair"
        )
    if len(exchange_account_pairs) != 1:
        incompatible = sorted({
            pair.get_exchange_account_protocol() or "missing protocol"
            for pair in exchange_account_pairs
            if pair.get_exchange_account_protocol() != LIGHTER_PROTOCOL
        })
        raise ValueError(
            "Lighter exchange-account strategies cannot mix protocols: "
            + ", ".join(incompatible)
        )
    return lighter_pairs


def validate_lighter_account_value(
    pair: TradingPairIdentifier,
    value: Decimal,
) -> Decimal:
    """Enforce the non-negative Lighter equity invariant."""
    if not pair.is_exchange_account() or pair.get_exchange_account_protocol() != LIGHTER_PROTOCOL:
        raise ValueError("Expected a Lighter exchange-account pair")
    account_index = pair.get_exchange_account_id()
    if account_index is None:
        raise ValueError("Lighter exchange account pair has no account index")
    account_index = int(account_index)
    try:
        if not isinstance(value, Decimal):
            value = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        # Convert malformed upstream values into the same hard, secret-free
        # invariant error used for negative/non-finite equity.  This prevents
        # callers such as ExchangeAccountValuator from treating bad Lighter
        # data as a transient outage and writing a cached valuation instead.
        raise LighterEquityInvariantError(
            f"Invalid equity for Lighter account {account_index}"
        ) from error
    if not value.is_finite():
        raise LighterEquityInvariantError(
            f"Invalid equity for Lighter account {account_index}"
        )
    # The shared state/valuation interfaces eventually carry a Python float.
    # Reject Decimal values which would overflow to infinity at that boundary.
    if not math.isfinite(float(value)):
        raise LighterEquityInvariantError(
            f"Invalid equity for Lighter account {account_index}"
        )
    if value < 0:
        # Keep this error deliberately public and stable: never include API
        # payloads, credentials or request details in an exception.
        raise NegativeLighterEquityError(
            f"Negative equity for Lighter account {account_index}"
        )
    return value


def create_lighter_account_value_func(
    session: LighterSession | None = None,
) -> Callable[..., Decimal]:
    """Create a public Lighter account-value reader.

    One unauthenticated HTTP session is created and reused by the returned
    callable.  ``block_identifier`` is accepted for exchange-account API
    compatibility, but Lighter returns a current observation rather than a
    historical Ethereum-block value.
    """
    lighter_session = session if session is not None else create_lighter_session()

    def get_lighter_account_value(
        pair: TradingPairIdentifier,
        block_identifier: Any = None,
        **kwargs: Any,
    ) -> Decimal:
        # The public Lighter endpoint cannot be block-pinned.  Keep accepting
        # the shared block keyword so exchange-account callers do not need a
        # protocol-specific branch.
        del block_identifier, kwargs
        if not pair.is_exchange_account():
            raise AssertionError(f"Not an exchange account pair: {pair}")
        if pair.get_exchange_account_protocol() != LIGHTER_PROTOCOL:
            raise AssertionError(
                f"Not a Lighter pair: {pair.get_exchange_account_protocol()}"
            )

        account_index = pair.get_exchange_account_id()
        if account_index is None:
            raise ValueError("Lighter exchange account pair has no account index")
        equity = fetch_lighter_total_equity(lighter_session, int(account_index))
        total = validate_lighter_account_value(pair, equity.get_total())
        logger.debug("Lighter account %d total equity: %s", account_index, total)
        return total

    return get_lighter_account_value


def create_lighter_available_balance_func(
    session: LighterSession | None = None,
) -> Callable[..., Decimal]:
    """Create a public reader for collateral currently withdrawable from Lighter.

    :param session:
        Reusable unauthenticated Lighter HTTP session.
    :return:
        Callable accepting an exchange-account pair and returning free USDC.
    """
    lighter_session = session if session is not None else create_lighter_session()

    def get_lighter_available_balance(
        pair: TradingPairIdentifier,
        block_identifier: Any = None,
        **kwargs: Any,
    ) -> Decimal:
        """Read validated Lighter free collateral for one account pair."""
        del block_identifier, kwargs
        if not pair.is_exchange_account() or pair.get_exchange_account_protocol() != LIGHTER_PROTOCOL:
            raise ValueError("Expected a Lighter exchange-account pair")
        account_index = pair.get_exchange_account_id()
        if account_index is None:
            raise ValueError("Lighter exchange account pair has no account index")
        equity = fetch_lighter_total_equity(lighter_session, int(account_index))
        total = validate_lighter_account_value(pair, equity.get_total())
        available = validate_lighter_account_value(pair, equity.available_balance)
        if available > total:
            raise LighterEquityInvariantError(
                f"Invalid available balance for Lighter account {account_index}"
            )
        return available

    return get_lighter_available_balance


def create_lighter_vault_valuation_func(
    web3: Web3,
    safe_address: str,
    reserve_asset: AssetIdentifier,
    account_index: int,
    session: LighterSession | None = None,
) -> Callable[..., float]:
    """Create the custom Lagoon NAV function for a Lighter account.

    The base value is the Safe's native reserve balance at the requested
    treasury-sync block plus the current public Lighter account equity.  The
    Lagoon sync model adds pending settlement value after this function
    returns, so the function deliberately returns ``float`` at its public
    interface boundary.
    """
    lighter_session = session if session is not None else create_lighter_session()
    account_value_func = create_lighter_account_value_func(lighter_session)
    reserve_token = fetch_erc20_details(
        web3,
        reserve_asset.address,
        chain_id=reserve_asset.chain_id,
    )
    pair = create_lighter_exchange_account_pair(
        quote=reserve_asset,
        account_index=account_index,
    )

    def calculate_nav(state: Any, *, block_number: int | None = None) -> float:
        del state
        safe_usdc = Decimal(str(reserve_token.fetch_balance_of(
            safe_address,
            block_identifier=block_number if block_number is not None else "latest",
        )))
        lighter_equity = account_value_func(pair)
        nav = safe_usdc + lighter_equity
        if not nav.is_finite() or nav < 0:
            raise ValueError(f"Invalid NAV for Lighter account {account_index}")
        nav_float = float(nav)
        if not math.isfinite(nav_float):
            raise ValueError(f"Invalid NAV for Lighter account {account_index}")
        logger.info(
            "Lighter vault valuation: Safe reserve=%s, account %d equity=%s, NAV=%s (block=%s)",
            safe_usdc,
            account_index,
            lighter_equity,
            nav,
            block_number,
        )
        return nav_float

    return calculate_nav
