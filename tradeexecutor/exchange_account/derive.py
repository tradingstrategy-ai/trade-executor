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
