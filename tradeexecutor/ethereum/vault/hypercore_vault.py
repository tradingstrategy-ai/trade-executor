"""Hypercore native vault support for Lagoon-on-HyperEVM positions.

Provides pair creation and account value function
for Hypercore vault deposits managed through a Lagoon vault on HyperEVM.

Hypercore vault equity is queried via the Hyperliquid info API
(:py:func:`~eth_defi.hyperliquid.api.fetch_user_vault_equity`)
rather than on-chain contracts.

The pair uses ``TradingPairKind.vault`` with
``other_data["vault_protocol"] = "hypercore"`` to distinguish from
ERC-4626 vault positions.

Vault deposits and withdrawals are routed through
:py:class:`~tradeexecutor.ethereum.vault.hypercore_routing.HypercoreVaultRouting`.
"""

import logging
from decimal import Decimal
from typing import Callable

from eth_defi.hyperliquid.api import fetch_user_vault_equity
from eth_defi.hyperliquid.core_writer import CORE_WRITER_ADDRESS
from eth_defi.hyperliquid.session import (
    HYPERLIQUID_API_URL,
    HYPERLIQUID_TESTNET_API_URL,
    create_hyperliquid_session,
)
from eth_typing import HexAddress
from tradingstrategy.chain import ChainId

from tradeexecutor.state.identifier import (
    AssetIdentifier,
    TradingPairIdentifier,
    TradingPairKind,
)

logger = logging.getLogger(__name__)

#: Default Hypercore vault addresses per network (HLP vault).
#:
#: HLP is the Hyperliquid Liquidity Provider vault — the main protocol vault.
#:
#: - Testnet: ``0xa15099a30bbf2e68942d6f4c43d70d04faeab0a0``
#: - Mainnet: ``0xdfc24b077bc1425ad1dea75bcb6f8158e10df303``
HLP_VAULT_ADDRESS: dict[str, HexAddress] = {
    "mainnet": HexAddress("0xdfc24b077bc1425ad1dea75bcb6f8158e10df303"),
    "testnet": HexAddress("0xa15099a30bbf2e68942d6f4c43d70d04faeab0a0"),
}


def create_hypercore_vault_pair(
    quote: AssetIdentifier,
    vault_address: HexAddress | str,
    is_testnet: bool = False,
    internal_id: int = 1,
) -> TradingPairIdentifier:
    """Create a TradingPairIdentifier for a Hypercore vault deposit.

    Builds the pair with ``kind=vault`` and
    ``other_data["vault_protocol"] = "hypercore"``.

    The base asset is a synthetic ``HYPERCORE-VAULT`` token whose address
    is the CoreWriter system contract. ``pool_address`` points to the
    actual Hypercore vault address so pair identity, quarantine checks,
    analytics, and links refer to the real vault. ``exchange_address``
    continues to point to CoreWriter because execution is routed through
    the Hypercore writer contract.

    Example::

        from tradeexecutor.ethereum.vault.hypercore_vault import (
            create_hypercore_vault_pair, HLP_VAULT_ADDRESS,
        )

        pair = create_hypercore_vault_pair(
            quote=usdc,
            vault_address=HLP_VAULT_ADDRESS["mainnet"],
        )

    :param quote:
        Reserve / quote asset (e.g. ``USDC``).

    :param vault_address:
        Hypercore native vault address to deposit into.

    :param is_testnet:
        Whether this targets the HyperEVM testnet (chain 998).

    :param internal_id:
        Internal pair ID. Increment if multiple Hypercore vaults in same universe.

    :return:
        Fully configured vault pair for Hypercore vault.
    """
    # Always use the synthetic Hypercore chain ID (9999) for vault pairs,
    # not the HyperEVM chain ID (999) that the quote token lives on.
    # Hypercore vaults are native Hyperliquid constructs, not EVM contracts.
    # is_hyperliquid_vault() relies on chain_id == 9999 for detection.
    hypercore_chain = ChainId.hypercore.value
    base = AssetIdentifier(
        chain_id=hypercore_chain,
        address=CORE_WRITER_ADDRESS,
        token_symbol="HYPERCORE-VAULT",
        decimals=6,
    )

    # Re-home the quote token to the Hypercore chain so the pair's
    # cross-chain assertion passes. The USDC contract address is the same
    # on both HyperEVM (999) and Hypercore (9999).
    if quote.chain_id != hypercore_chain:
        quote = AssetIdentifier(
            chain_id=hypercore_chain,
            address=quote.address,
            token_symbol=quote.token_symbol,
            decimals=quote.decimals,
        )

    vault_addr = vault_address.lower() if isinstance(vault_address, str) else vault_address

    return TradingPairIdentifier(
        base=base,
        quote=quote,
        pool_address=vault_addr,
        exchange_address=CORE_WRITER_ADDRESS,
        internal_id=internal_id,
        internal_exchange_id=internal_id,
        fee=0.0,
        kind=TradingPairKind.vault,
        exchange_name="Hypercore",
        other_data={
            "vault_protocol": "hypercore",
            "exchange_is_testnet": is_testnet,
        },
    )


def create_hypercore_vault_value_func(
    execution_model=None,
    *,
    session=None,
    safe_address: str | None = None,
    is_testnet: bool = False,
) -> Callable[[TradingPairIdentifier], Decimal]:
    """Create Hypercore vault account value function.

    The returned function queries the Hyperliquid info API for the
    user's vault equity position using the cached
    :py:func:`~eth_defi.hyperliquid.api.fetch_user_vault_equity`,
    returning the USDC equity for the specific vault address stored
    in ``pair.pool_address``.

    Can be called with either an *execution_model* (used by the runner)
    or explicit *session* + *safe_address* (used by CLI commands).

    :param execution_model:
        The execution model (e.g. ``LagoonExecution``) that provides
        Web3 and the transaction builder with the Safe address.

    :param session:
        Explicit Hyperliquid API session.

    :param safe_address:
        Explicit Safe address (used when no execution model available).

    :param is_testnet:
        Whether to use testnet API URL (only used when creating session lazily).

    :return:
        Function that takes a TradingPairIdentifier and returns
        vault equity in USD.
    """
    if execution_model is None:
        assert session is not None and safe_address is not None, \
            "Either execution_model or both session and safe_address must be provided"
        _session = session
        _safe_address = safe_address
    else:
        _session = None
        _safe_address = None

    def get_hypercore_vault_value(pair: TradingPairIdentifier) -> Decimal:
        """Get Hypercore vault equity for the given pair.

        :param pair:
            Vault trading pair with Hypercore metadata in other_data.

        :return:
            Vault equity in USD, or ``Decimal(0)`` if no deposit found.
        """
        assert pair.is_hyperliquid_vault(), f"Not a Hypercore vault pair: {pair}"

        vault_address = pair.pool_address
        assert vault_address, f"No pool_address set for Hypercore vault pair: {pair}"

        if _session is not None:
            session = _session
            safe_address = _safe_address
        else:
            safe_address = execution_model.tx_builder.get_token_delivery_address()
            api_url = HYPERLIQUID_TESTNET_API_URL if is_testnet else HYPERLIQUID_API_URL
            session = create_hyperliquid_session(api_url=api_url)

        try:
            eq = fetch_user_vault_equity(
                session,
                user=safe_address,
                vault_address=vault_address,
            )

            if eq is not None:
                logger.debug(
                    "Hypercore vault %s equity for %s: $%s",
                    vault_address, safe_address, eq.equity,
                )
                return eq.equity

            logger.debug(
                "No Hypercore vault position found for %s in vault %s",
                safe_address, vault_address,
            )
            return Decimal(0)

        except Exception as e:
            logger.error(
                "Failed to get Hypercore vault value for %s (vault %s): %s",
                safe_address, vault_address, e,
            )
            raise

    return get_hypercore_vault_value
