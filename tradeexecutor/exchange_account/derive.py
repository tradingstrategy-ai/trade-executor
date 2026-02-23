"""Derive-specific account value function.

Provides the account value function for Derive.xyz exchange accounts.
"""

import enum
import logging
from decimal import Decimal
from typing import TYPE_CHECKING, Callable

from tradeexecutor.state.identifier import TradingPairIdentifier

# Lazy imports to avoid loading pyrate_limiter at module load time
# (not installed in CI environment)
if TYPE_CHECKING:
    from eth_defi.derive.authentication import DeriveApiClient


class DeriveNetwork(str, enum.Enum):
    """Derive network selection.

    Used by CLI commands to specify mainnet or testnet.
    Inherits from str for Typer/Click compatibility.
    """
    mainnet = "mainnet"
    testnet = "testnet"

logger = logging.getLogger(__name__)


def discover_derive_subaccount_id(
    owner_private_key: str | None = None,
    session_private_key: str | None = None,
    wallet_address: str | None = None,
    network: DeriveNetwork | str = DeriveNetwork.mainnet,
) -> int:
    """Discover the first Derive subaccount ID from the API.

    If ``DERIVE_SUBACCOUNT_ID`` environment variable is set, returns it
    directly without contacting the API. This allows production deployments
    to skip API discovery when the subaccount ID is already known.

    Otherwise authenticates with the Derive API and queries available
    subaccounts. Falls back to environment variables when parameters
    are not provided:

    - ``DERIVE_SUBACCOUNT_ID`` (shortcut, skips API discovery)
    - ``DERIVE_OWNER_PRIVATE_KEY``
    - ``DERIVE_SESSION_PRIVATE_KEY``
    - ``DERIVE_WALLET_ADDRESS`` (optional, derived from owner key if absent)
    - ``DERIVE_NETWORK`` (optional, defaults to ``"mainnet"``)

    :param owner_private_key:
        Derive owner wallet private key hex string.
    :param session_private_key:
        Derive session key private key hex string.
    :param wallet_address:
        Derive wallet address (auto-derived from owner key if absent).
    :param network:
        ``"mainnet"`` or ``"testnet"``.
    :return:
        First subaccount ID.
    :raises RuntimeError:
        If credentials are missing or no subaccounts are found.
    """
    import os

    # Shortcut: if subaccount ID is already known, skip API discovery
    env_subaccount_id = os.environ.get("DERIVE_SUBACCOUNT_ID")
    if env_subaccount_id:
        subaccount_id = int(env_subaccount_id)
        logger.info("Using DERIVE_SUBACCOUNT_ID from environment: %d", subaccount_id)
        return subaccount_id

    from eth_account import Account
    from eth_defi.derive.account import fetch_subaccount_ids
    from eth_defi.derive.authentication import DeriveApiClient
    from eth_defi.derive.onboarding import fetch_derive_wallet_address

    owner_key = owner_private_key or os.environ.get("DERIVE_OWNER_PRIVATE_KEY")
    session_key = session_private_key or os.environ.get("DERIVE_SESSION_PRIVATE_KEY")
    if not owner_key or not session_key:
        raise RuntimeError(
            "DERIVE_OWNER_PRIVATE_KEY and DERIVE_SESSION_PRIVATE_KEY "
            "environment variables are required to discover subaccount IDs. "
            "Alternatively set DERIVE_SUBACCOUNT_ID to skip API discovery."
        )

    if isinstance(network, str):
        network = DeriveNetwork(network)
    is_testnet = (network == DeriveNetwork.testnet)

    # Allow env override for network
    env_network = os.environ.get("DERIVE_NETWORK")
    if env_network:
        is_testnet = (env_network == "testnet")

    owner_account = Account.from_key(owner_key)
    derive_wallet = wallet_address or os.environ.get("DERIVE_WALLET_ADDRESS")
    if not derive_wallet:
        derive_wallet = fetch_derive_wallet_address(
            owner_account.address,
            is_testnet=is_testnet,
        )

    client = DeriveApiClient(
        owner_account=owner_account,
        derive_wallet_address=derive_wallet,
        is_testnet=is_testnet,
        session_key_private=session_key,
    )

    subaccount_ids = fetch_subaccount_ids(client)
    if not subaccount_ids:
        raise RuntimeError(
            f"No Derive subaccounts found for wallet {derive_wallet}"
        )

    logger.info(
        "Discovered %d Derive subaccount(s): %s, using first: %d",
        len(subaccount_ids),
        subaccount_ids,
        subaccount_ids[0],
    )
    return subaccount_ids[0]


def create_derive_exchange_account_pair(
    quote: "AssetIdentifier",
    subaccount_id: int,
    is_testnet: bool = False,
) -> "TradingPairIdentifier":
    """Create a TradingPairIdentifier for a Derive exchange account.

    Builds the pair with correct ``kind``, ``exchange_name``, and ``other_data``
    fields needed by the sync, pricing, and valuation pipeline.

    The base asset is a synthetic ``DERIVE-ACCOUNT`` token created
    automatically with the same ``chain_id`` as the quote asset.
    The subaccount ID is encoded into ``pool_address`` and ``exchange_address``
    for traceability (exchange accounts have no real on-chain pool).

    Example:

    .. code-block:: python

        from tradeexecutor.exchange_account.derive import (
            create_derive_exchange_account_pair,
            discover_derive_subaccount_id,
        )

        subaccount_id = discover_derive_subaccount_id()
        pair = create_derive_exchange_account_pair(
            quote=usdc,
            subaccount_id=subaccount_id,
        )

    :param quote:
        Reserve / quote asset (e.g. ``USDC``).
    :param subaccount_id:
        Derive subaccount ID (integer, from :py:func:`discover_derive_subaccount_id`).
    :param is_testnet:
        Whether this targets the Derive testnet.
    :return:
        Fully configured exchange account pair.
    """
    from tradeexecutor.state.identifier import AssetIdentifier, TradingPairKind

    # Synthetic asset representing the Derive account value
    base = AssetIdentifier(
        chain_id=quote.chain_id,
        address="0x0000000000000000000000000000000000D371E0",
        token_symbol="DERIVE-ACCOUNT",
        decimals=6,
    )

    subaccount_hex = hex(subaccount_id)

    return TradingPairIdentifier(
        base=base,
        quote=quote,
        pool_address=subaccount_hex,
        exchange_address=subaccount_hex,
        internal_id=1,
        internal_exchange_id=1,
        fee=0.0,
        kind=TradingPairKind.exchange_account,
        exchange_name="derive",
        other_data={
            "exchange_protocol": "derive",
            "exchange_subaccount_id": subaccount_id,
            "exchange_is_testnet": is_testnet,
        },
    )


def create_derive_account_value_func(
    clients: "dict[int, DeriveApiClient]",
) -> Callable[[TradingPairIdentifier], Decimal]:
    """Create Derive-specific account value function.

    The returned function queries the Derive API for the total account value
    in USD, which includes collateral and unrealised PnL from all positions.

    Example:

    .. code-block:: python

        from eth_account import Account
        from eth_defi.derive.authentication import DeriveApiClient
        from eth_defi.derive.account import fetch_subaccount_ids

        # Create authenticated client
        owner = Account.from_key(os.environ["DERIVE_OWNER_PRIVATE_KEY"])
        client = DeriveApiClient(
            owner_account=owner,
            derive_wallet_address=wallet_address,
            is_testnet=True,
            session_key_private=os.environ["DERIVE_SESSION_PRIVATE_KEY"],
        )

        # Resolve subaccount
        ids = fetch_subaccount_ids(client)
        client.subaccount_id = ids[0]

        # Create account value function
        clients = {client.subaccount_id: client}
        account_value_func = create_derive_account_value_func(clients)

        # Use with pricing model
        from tradeexecutor.exchange_account.pricing import ExchangeAccountPricingModel
        pricing = ExchangeAccountPricingModel(account_value_func)

    :param clients:
        Dict mapping subaccount_id -> authenticated DeriveApiClient
    :return:
        Function that takes a TradingPairIdentifier and returns account value in USD
    """

    def get_derive_account_value(pair: TradingPairIdentifier) -> Decimal:
        """Get Derive account value for the given pair.

        :param pair:
            Exchange account trading pair with Derive metadata in other_data
        :return:
            Total account value in USD
        :raises KeyError:
            If subaccount_id not in clients dict
        :raises Exception:
            If API call fails
        """
        # Lazy import to avoid loading pyrate_limiter at module load time
        from eth_defi.derive.account import fetch_account_summary

        assert pair.is_exchange_account(), f"Not an exchange account pair: {pair}"
        assert pair.get_exchange_account_protocol() == "derive", \
            f"Not a Derive pair: {pair.get_exchange_account_protocol()}"

        subaccount_id = pair.get_exchange_account_id()
        if subaccount_id is None:
            raise ValueError(f"No exchange_subaccount_id in pair other_data: {pair}")

        client = clients.get(subaccount_id)
        if client is None:
            raise KeyError(f"No client for subaccount_id {subaccount_id}. Available: {list(clients.keys())}")

        try:
            summary = fetch_account_summary(client, subaccount_id)
            logger.debug(
                "Derive subaccount %d value: $%.2f",
                subaccount_id,
                summary.total_value_usd,
            )
            return summary.total_value_usd
        except Exception as e:
            logger.error(
                "Failed to get Derive account value for subaccount %d: %s",
                subaccount_id,
                e,
            )
            raise

    return get_derive_account_value
