"""Lighter exchange-account support.

Lighter accounts are external to the EVM portfolio. The account index is
stored on a synthetic exchange-account pair and the public Lighter REST API is
used for valuation without credentials. Automatic withdrawals use a separate
owner-only operator record for authenticated requests.
"""

import logging
import math
from collections.abc import Collection, Iterable, Mapping
from decimal import Decimal, InvalidOperation
from typing import Any, Callable

from eth_defi.lighter.api import fetch_lighter_accounts
from eth_defi.lighter.constants import LIGHTER_L1_CONTRACT
from eth_defi.lighter.session import LighterSession, create_lighter_session
from eth_defi.lighter.valuation import fetch_lighter_accounts_equity, fetch_lighter_total_equity
from eth_defi.token import fetch_erc20_details
from requests.exceptions import RequestException
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
#: ``other_data`` role of the Safe-owned master account pair.
#: Id a strategy uses for the Safe-owned master account.
LIGHTER_MASTER_LABEL = "master"

LIGHTER_ROLE_MASTER = "master"
#: ``other_data`` role of a sub-account pair.
LIGHTER_ROLE_SUB_ACCOUNT = "sub_account"
#: Synthetic address prefix for sub-account pairs, ``"LI"`` in ASCII.
#:
#: A sub-account pair needs its own base asset: pair equality and hashing
#: compare base address, quote address and fee only, so pairs sharing the
#: Lighter L1 contract as base would collide in every portfolio lookup.
LIGHTER_SUB_ACCOUNT_ADDRESS_PREFIX = "4c49"
#: Offset keeping sub-account ``internal_id`` values unique in mixed universes.
#:
#: Above every Trading Strategy dataset pair id and above the ``2**48``
#: Lighter account-index range.
LIGHTER_SUB_ACCOUNT_INTERNAL_ID_OFFSET = 10**15
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
#: Explicit allowlist for public sub-account fields in deployment reports.
LIGHTER_PUBLIC_SUB_ACCOUNT_FIELDS = (
    "account_index",
    "api_key_index",
    "public_key",
    "change_pubkey_tx_hash",
    "label",
)


def get_public_lighter_metadata(metadata: dict[Any, Any]) -> dict[str, Any]:
    """Copy only fields approved for public Lighter deployment reports."""
    public = {
        key: metadata[key]
        for key in LIGHTER_PUBLIC_METADATA_LABELS
        if key in metadata
    }
    sub_accounts = metadata.get("sub_accounts")
    if sub_accounts:
        public["sub_accounts"] = [
            {key: sub[key] for key in LIGHTER_PUBLIC_SUB_ACCOUNT_FIELDS if key in sub}
            for sub in sub_accounts
        ]
    return public


def check_lighter_account_set(
    safe_address: str,
    pairs: Iterable[TradingPairIdentifier],
    session: LighterSession | None = None,
) -> tuple[bool, str]:
    """Compare the Lighter accounts a Safe owns with the tracked positions.

    An owned account that has no position is left out of the posted NAV, so
    a mismatch is an accounting error. When Lighter does not list the Safe
    at all, or cannot be read, the check cannot conclude and passes with a
    message saying so.

    :param safe_address:
        Safe owning the Lighter accounts.
    :param pairs:
        Lighter exchange-account pairs of the tracked positions.
    :param session:
        Public Lighter HTTP session.
    :return:
        ``(clean, message)``.
    """
    tracked = {int(pair.get_exchange_account_id()) for pair in pairs}
    lighter_session = session if session is not None else create_lighter_session()
    try:
        accounts = fetch_lighter_accounts(lighter_session, safe_address)
    except (RequestException, ValueError) as error:
        return True, f"Could not read Lighter accounts for {safe_address} ({type(error).__name__})"
    finally:
        if session is None:
            lighter_session.close()
    if accounts is None:
        return True, f"Lighter lists no accounts for {safe_address}"
    owned = set(accounts.get_all_indices())
    if owned == tracked:
        return True, f"Lighter accounts match the tracked positions: {sorted(owned)}"
    return False, f"Lighter accounts {sorted(owned - tracked)} have no position and positions {sorted(tracked - owned)} have no Lighter account"


def format_lighter_sub_account_lines(
    metadata: dict[str, Any] | None,
    *,
    markdown: bool = False,
    indent: str = "  ",
) -> list[str]:
    """Format public sub-account metadata for deployment reports.

    :param metadata:
        Output of :py:func:`get_public_lighter_metadata`.
    :param markdown:
        Produce Markdown bullets instead of indented text.
    :param indent:
        Text indent for non-Markdown output.
    :return:
        One line per sub-account, empty without sub-accounts.
    """
    lines = []
    for sub in (metadata or {}).get("sub_accounts") or []:
        description = f"account {sub.get('account_index')}, API-key index {sub.get('api_key_index')}, public key {sub.get('public_key')}, change public-key transaction {sub.get('change_pubkey_tx_hash')}"
        lines.append(f"- **Sub-account**: `{description}`" if markdown else f"{indent}Sub-account: {description}")
    return lines


class LighterEquityInvariantError(ValueError):
    """Raised when the public Lighter API reports an invalid equity value."""


class NegativeLighterEquityError(LighterEquityInvariantError):
    """Raised when the public Lighter API reports impossible negative equity."""


def get_lighter_sub_account_address(account_index: int) -> str:
    """Return the synthetic address identifying one Lighter sub-account pair.

    :param account_index:
        Lighter sub-account index.
    :return:
        A 20-byte address, ``0x4c49`` followed by the index in hexadecimal.
    """
    return "0x" + LIGHTER_SUB_ACCOUNT_ADDRESS_PREFIX + f"{account_index:036x}"


def create_lighter_exchange_account_pair(
    quote: AssetIdentifier,
    account_index: int,
    is_testnet: bool = False,
    role: str = LIGHTER_ROLE_MASTER,
    master_account_index: int | None = None,
    strategy_label: str | None = None,
) -> TradingPairIdentifier:
    """Create a synthetic pair representing one Lighter account.

    The Safe-owned master account keeps the identity used since the first
    integration, so existing executor state needs no migration. Each
    sub-account gets a unique synthetic base asset, pool address and
    ``internal_id``, so every account is a separate trading position.

    :param quote:
        Reserve/quote asset (normally Ethereum USDC).
    :param account_index:
        Public Lighter account index.
    :param is_testnet:
        Retained for parity with the other exchange-account pair factories.
        The first integration supports the canonical Ethereum deployment.
    :param role:
        :data:`LIGHTER_ROLE_MASTER` or :data:`LIGHTER_ROLE_SUB_ACCOUNT`.
    :param master_account_index:
        Master account owning a sub-account. Defaults to ``account_index``
        for the master itself and is required for a sub-account.
    :param strategy_label:
        Optional name of the external strategy trading a sub-account.
    :return:
        Exchange-account pair with public Lighter metadata only.
    """
    if type(account_index) is not int or account_index < 0:
        raise ValueError(f"Lighter account index must be a non-negative integer: {account_index!r}")
    if role not in (LIGHTER_ROLE_MASTER, LIGHTER_ROLE_SUB_ACCOUNT):
        raise ValueError(f"Unknown Lighter account role: {role!r}")
    if role == LIGHTER_ROLE_MASTER:
        master_account_index = account_index
    elif type(master_account_index) is not int or master_account_index < 0 or master_account_index == account_index:
        raise ValueError("A Lighter sub-account pair needs its master account index")

    other_data = {
        "exchange_protocol": LIGHTER_PROTOCOL,
        "exchange_subaccount_id": account_index,
        "exchange_is_testnet": is_testnet,
        "lighter_deployment": LIGHTER_DEPLOYMENT,
        "lighter_account_role": role,
        "lighter_master_account_index": master_account_index,
    }
    if strategy_label:
        other_data["lighter_sub_account_id"] = strategy_label

    if role == LIGHTER_ROLE_MASTER:
        # Lighter's L1 contract is the stable, canonical identity for the
        # master account. It is not used as an ERC-20 contract by the reader.
        address = LIGHTER_L1_CONTRACT
        symbol = LIGHTER_ACCOUNT_SYMBOL
        internal_id = 1
    else:
        address = get_lighter_sub_account_address(account_index)
        symbol = f"LIGHTER-{strategy_label or account_index}"
        internal_id = LIGHTER_SUB_ACCOUNT_INTERNAL_ID_OFFSET + account_index

    base = AssetIdentifier(
        chain_id=quote.chain_id,
        address=address,
        token_symbol=symbol,
        decimals=6,
    )
    return TradingPairIdentifier(
        base=base,
        quote=quote,
        pool_address=address,
        exchange_address=address,
        internal_id=internal_id,
        internal_exchange_id=1,
        fee=0.0,
        kind=TradingPairKind.exchange_account,
        exchange_name="Lighter",
        other_data=other_data,
    )


def create_lighter_exchange_account_pairs(
    quote: AssetIdentifier,
    master_account_index: int,
    sub_account_indices: Mapping[int, str] | Iterable[int] = (),
    strategy_labels: Mapping[int, str] | None = None,
) -> list[TradingPairIdentifier]:
    """Create pairs for a master account and its sub-accounts.

    :param quote:
        Reserve/quote asset (normally Ethereum USDC).
    :param master_account_index:
        Safe-owned master account.
    :param sub_account_indices:
        Sub-accounts in creation order, either as indices or as a mapping of
        index to label, which is how a strategy's ``Parameters`` declares them.
    :param strategy_labels:
        Optional sub-account id per index when ``sub_account_indices`` is a plain sequence.
    :return:
        Master pair followed by one pair per sub-account.
    """
    if isinstance(sub_account_indices, Mapping):
        labels = dict(sub_account_indices)
        sub_account_indices = list(labels)
    else:
        labels = dict(strategy_labels or {})
    if len(set(labels.values())) != len(labels) or LIGHTER_MASTER_LABEL in labels.values():
        raise ValueError(f"Lighter sub-account ids must be unique and not {LIGHTER_MASTER_LABEL!r}: {labels}")
    pairs = [create_lighter_exchange_account_pair(quote, master_account_index)]
    pairs += [
        create_lighter_exchange_account_pair(
            quote,
            index,
            role=LIGHTER_ROLE_SUB_ACCOUNT,
            master_account_index=master_account_index,
            strategy_label=labels.get(index),
        )
        for index in sub_account_indices
    ]
    return pairs


def get_lighter_account_id(pair: TradingPairIdentifier) -> str:
    """Return the id a strategy uses for a Lighter account pair.

    :param pair:
        Lighter exchange-account pair.
    :return:
        ``master`` for the master account, otherwise the sub-account's id or
        its index when it has no id.
    """
    if get_lighter_account_role(pair) == LIGHTER_ROLE_MASTER:
        return LIGHTER_MASTER_LABEL
    return str((pair.other_data or {}).get("lighter_sub_account_id") or pair.get_exchange_account_id())


def find_lighter_account_pair(pairs: Iterable[TradingPairIdentifier], account: str | int) -> TradingPairIdentifier:
    """Return the Lighter pair a strategy refers to by id or index.

    :param pairs:
        Pairs of a strategy universe.
    :param account:
        ``master``, a sub-account id, or an account index.
    :return:
        The matching pair.
    :raise ValueError:
        If no Lighter pair matches.
    """
    for pair in pairs:
        if not is_lighter_exchange_account_pair(pair):
            continue
        if get_lighter_account_id(pair) == str(account) or str(pair.get_exchange_account_id()) == str(account):
            return pair
    raise ValueError(f"No Lighter account {account!r} in the strategy universe; declare it in Parameters.lighter_sub_accounts")


def parse_lighter_account_indices(raw: str) -> list[int]:
    """Parse comma-separated Lighter account indices, the master account first.

    :param raw:
        For example ``"700,281474976709656"``; an empty string means no accounts.
    :return:
        Account indices in the given order.
    :raise ValueError:
        If a part is not a non-negative integer.
    """
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not all(part.isascii() and part.isdecimal() for part in parts):
        raise ValueError(f"Lighter account indices must be comma-separated non-negative integers: {raw!r}")
    return [int(part) for part in parts]


def is_lighter_exchange_account_pair(pair: TradingPairIdentifier) -> bool:
    """Check whether a pair is a Lighter master or sub-account."""
    return pair.is_exchange_account() and pair.get_exchange_account_protocol() == LIGHTER_PROTOCOL


def get_lighter_account_role(pair: TradingPairIdentifier) -> str:
    """Return the role of a Lighter exchange-account pair.

    Pairs created before sub-account support carry no role and are the
    master account.

    :param pair:
        Lighter exchange-account pair.
    :return:
        :data:`LIGHTER_ROLE_MASTER` or :data:`LIGHTER_ROLE_SUB_ACCOUNT`.
    """
    return (pair.other_data or {}).get("lighter_account_role", LIGHTER_ROLE_MASTER)


def is_lighter_sub_account_pair(pair: TradingPairIdentifier) -> bool:
    """Check whether a pair is a Lighter sub-account."""
    return is_lighter_exchange_account_pair(pair) and get_lighter_account_role(pair) == LIGHTER_ROLE_SUB_ACCOUNT


def get_lighter_master_account_index(pair: TradingPairIdentifier) -> int:
    """Return the master account owning a Lighter account pair.

    :param pair:
        Lighter exchange-account pair.
    :return:
        Master account index; the pair's own index for a master pair.
    """
    other_data = pair.other_data or {}
    master_index = other_data.get("lighter_master_account_index")
    if master_index is None:
        master_index = pair.get_exchange_account_id()
    if master_index is None:
        raise ValueError("Lighter exchange account pair has no account index")
    return int(master_index)


def get_lighter_master_pair(pairs: Iterable[TradingPairIdentifier]) -> TradingPairIdentifier:
    """Return the single master account pair among Lighter pairs.

    :param pairs:
        Pairs of a strategy universe.
    :return:
        The Lighter master account pair.
    """
    masters = [pair for pair in pairs if is_lighter_exchange_account_pair(pair) and get_lighter_account_role(pair) == LIGHTER_ROLE_MASTER]
    if len(masters) != 1:
        raise ValueError(f"Expected one Lighter master account pair, found {len(masters)}")
    return masters[0]


def has_lighter_exchange_account_pairs(strategy_universe: Any) -> bool:
    """Return whether a strategy universe contains a Lighter account pair."""
    iterate_pairs = getattr(strategy_universe, "iterate_pairs", None)
    if iterate_pairs is None:
        return False
    return any(is_lighter_exchange_account_pair(pair) for pair in iterate_pairs())


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
    masters = [pair for pair in lighter_pairs if get_lighter_account_role(pair) == LIGHTER_ROLE_MASTER]
    if len(masters) != 1:
        raise ValueError(
            "Lighter strategies must contain exactly one master account pair"
        )
    master_index = get_lighter_master_account_index(masters[0])
    account_indices = [pair.get_exchange_account_id() for pair in lighter_pairs]
    if None in account_indices or len(set(account_indices)) != len(account_indices):
        raise ValueError("Lighter account pairs must have unique account indices")
    if any(get_lighter_master_account_index(pair) != master_index for pair in lighter_pairs):
        raise ValueError("Every Lighter sub-account must belong to the strategy's master account")
    if len(exchange_account_pairs) != len(lighter_pairs):
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
    if not is_lighter_exchange_account_pair(pair):
        raise ValueError("Expected a Lighter exchange-account pair")
    account_index = pair.get_exchange_account_id()
    if account_index is None:
        raise ValueError("Lighter exchange account pair has no account index")
    return validate_lighter_equity(int(account_index), value)


def validate_lighter_equity(account_index: int, value: Decimal) -> Decimal:
    """Enforce the non-negative, finite Lighter equity invariant for one account.

    :param account_index:
        Lighter account the value belongs to, used in error messages.
    :param value:
        Equity reported by the public Lighter API.
    :return:
        The validated value as a ``Decimal``.
    """
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
        if not is_lighter_exchange_account_pair(pair):
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
    account_indices: Collection[int],
    session: LighterSession | None = None,
) -> Callable[..., float]:
    """Create the custom Lagoon NAV function for the Safe's Lighter accounts.

    The base value is the Safe's native reserve balance at the requested
    treasury-sync block plus the current public equity of every listed
    Lighter account.  The Lagoon sync model adds pending settlement value
    after this function returns, so the function deliberately returns
    ``float`` at its public interface boundary.

    Several accounts are valued from one by-owner snapshot, so a transfer
    between them cannot be counted twice or missed. A single account keeps
    the per-account read used since the first integration.

    :param web3:
        Ethereum connection.
    :param safe_address:
        Safe holding the reserve and owning the Lighter accounts.
    :param reserve_asset:
        Native Ethereum USDC.
    :param account_indices:
        Master account and any sub-accounts to include.
    :param session:
        Reusable public Lighter HTTP session.
    :return:
        NAV callable for the Lagoon sync model.
    """
    indices = tuple(int(index) for index in account_indices)
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("Lighter vault valuation needs unique account indices")
    lighter_session = session if session is not None else create_lighter_session()
    account_value_func = create_lighter_account_value_func(lighter_session)
    reserve_token = fetch_erc20_details(
        web3,
        reserve_asset.address,
        chain_id=reserve_asset.chain_id,
    )
    pair = create_lighter_exchange_account_pair(
        quote=reserve_asset,
        account_index=indices[0],
    )

    def read_lighter_equity() -> Decimal:
        # Fail closed: an owned account without a position would be left out
        # of the posted NAV and understate the share price.
        accounts = fetch_lighter_accounts(lighter_session, safe_address)
        if accounts is not None:
            unconfigured = set(accounts.get_all_indices()) - set(indices)
            if unconfigured:
                raise ValueError(
                    f"Lighter accounts {sorted(unconfigured)} owned by {safe_address} are not in the strategy universe, "
                    f"so the NAV would omit them"
                )
        if len(indices) == 1:
            return account_value_func(pair)
        snapshot = fetch_lighter_accounts_equity(lighter_session, safe_address)
        missing = [index for index in indices if index not in snapshot]
        if missing:
            # Fail closed: a missing account would understate the posted NAV.
            raise ValueError(f"Lighter equity snapshot is missing accounts {missing}")
        return sum((validate_lighter_equity(index, snapshot[index].get_total()) for index in indices), Decimal(0))

    def calculate_nav(state: Any, *, block_number: int | None = None) -> float:
        del state
        safe_usdc = Decimal(str(reserve_token.fetch_balance_of(
            safe_address,
            block_identifier=block_number if block_number is not None else "latest",
        )))
        lighter_equity = read_lighter_equity()
        nav = safe_usdc + lighter_equity
        if not nav.is_finite() or nav < 0:
            raise ValueError(f"Invalid NAV for Lighter accounts {list(indices)}")
        nav_float = float(nav)
        if not math.isfinite(nav_float):
            raise ValueError(f"Invalid NAV for Lighter accounts {list(indices)}")
        logger.info(
            "Lighter vault valuation: Safe reserve=%s, accounts %s equity=%s, NAV=%s (block=%s)",
            safe_usdc,
            list(indices),
            lighter_equity,
            nav,
            block_number,
        )
        return nav_float

    return calculate_nav
