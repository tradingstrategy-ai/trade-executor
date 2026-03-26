"""Multichain balance fetching with off-chain routing.

Routes balance queries to the correct backend based on position type:

- ERC-20 positions: batched on-chain ``balanceOf()`` via :py:func:`fetch_address_balances`
- Hypercore vault positions: Hyperliquid info API via :py:func:`fetch_user_vault_equity`
- Exchange account positions: returns zero (no on-chain representation)

This mirrors the classification in :py:func:`tradeexecutor.strategy.asset.get_onchain_assets`.
"""

import logging
from collections.abc import Collection, Iterable
from decimal import Decimal

from web3 import Web3

from eth_defi.compat import native_datetime_utc_now
from eth_defi.hyperliquid.api import fetch_user_vault_equity
from eth_defi.hyperliquid.session import (
    HyperliquidSession,
    create_hyperliquid_session,
    HYPERLIQUID_API_URL,
    HYPERLIQUID_TESTNET_API_URL,
)
from tradingstrategy.chain import ChainId

from tradeexecutor.ethereum.onchain_balance import fetch_address_balances
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.strategy.sync_model import OnChainBalance


logger = logging.getLogger(__name__)


#: Chain IDs that represent off-chain / API-tracked positions
#: (no ERC-20 ``balanceOf`` available).
_OFFCHAIN_CHAIN_IDS = frozenset({
    ChainId.hypercore.value,  # 9999 — native Hyperliquid vaults
})


def fetch_onchain_balances_multichain(
    web3: Web3,
    address: str,
    assets: list[AssetIdentifier],
    *,
    pairs: list[TradingPairIdentifier] | None = None,
    session: HyperliquidSession | None = None,
    filter_zero: bool = True,
    block_number: int | None = None,
) -> Iterable[OnChainBalance]:
    """Fetch balances for a batch of assets, routing off-chain types automatically.

    Two calling conventions are supported:

    **Asset-only mode** (used by :py:class:`SyncModel` subclasses):
        Pass ``assets`` without ``pairs``.  Off-chain assets (Hypercore chain 9999)
        are silently filtered out, then all remaining ERC-20 assets are fetched
        in a **single** batched ``fetch_address_balances()`` call.

    **Pair-aware mode** (used by CLI commands like ``close-position``):
        Pass both ``assets`` and ``pairs``.  Each pair is classified:

        - **Hypercore vaults** → Hyperliquid info API (equity query)
        - **Exchange accounts** → zero balance
        - **ERC-20** → collected and fetched in one batched call

        Results are yielded in input order: Hypercore first, then exchange
        accounts, then the ERC-20 batch.

    :param web3:
        Web3 connection for ERC-20 ``balanceOf`` calls.

    :param address:
        The token-holding address (Gnosis Safe, hot wallet, vault).

    :param assets:
        Assets to look up.  In pair-aware mode the list should match
        ``[p.base for p in pairs]`` — it is only used as a fallback
        when ``pairs`` is ``None``.

    :param pairs:
        Optional trading pairs for richer routing (Hypercore / exchange account).
        When provided, ``assets`` is ignored and base assets are derived from pairs.

    :param session:
        Optional Hyperliquid session.  Created lazily when a Hypercore
        pair is encountered.

    :param filter_zero:
        Passed through to the ERC-20 ``fetch_address_balances()`` path.

    :param block_number:
        Optional block height for ERC-20 lookups.
    """

    if pairs is not None:
        yield from _fetch_pair_aware(
            web3, address, pairs,
            session=session,
            filter_zero=filter_zero,
            block_number=block_number,
        )
    else:
        yield from _fetch_asset_only(
            web3, address, assets,
            filter_zero=filter_zero,
            block_number=block_number,
        )


# ------------------------------------------------------------------
# Asset-only mode (SyncModel path)
# ------------------------------------------------------------------

def _fetch_asset_only(
    web3: Web3,
    address: str,
    assets: list[AssetIdentifier],
    *,
    filter_zero: bool = True,
    block_number: int | None = None,
) -> Iterable[OnChainBalance]:
    """Filter off-chain assets, then batch-fetch all ERC-20 balances."""
    onchain_assets = filter_onchain_assets(assets)
    if not onchain_assets:
        return
    yield from fetch_address_balances(
        web3,
        address,
        onchain_assets,
        block_number=block_number,
        filter_zero=filter_zero,
    )


# ------------------------------------------------------------------
# Pair-aware mode (close-position / CLI path)
# ------------------------------------------------------------------

def _fetch_pair_aware(
    web3: Web3,
    address: str,
    pairs: list[TradingPairIdentifier],
    *,
    session: HyperliquidSession | None = None,
    filter_zero: bool = True,
    block_number: int | None = None,
) -> Iterable[OnChainBalance]:
    """Route each pair by type, batching all ERC-20 into one call."""

    hypercore_pairs: list[TradingPairIdentifier] = []
    exchange_pairs: list[TradingPairIdentifier] = []
    erc20_assets: list[AssetIdentifier] = []

    for pair in pairs:
        if pair.is_hyperliquid_vault():
            hypercore_pairs.append(pair)
        elif pair.is_exchange_account():
            exchange_pairs.append(pair)
        else:
            erc20_assets.append(pair.base)

    # Hypercore vaults — API lookup
    for pair in hypercore_pairs:
        yield _fetch_hypercore_vault_balance(pair, address, session)

    # Exchange accounts — zero balance
    ts = native_datetime_utc_now()
    for pair in exchange_pairs:
        yield OnChainBalance(
            block_number=None,
            timestamp=ts,
            asset=pair.base,
            amount=Decimal(0),
        )

    # ERC-20 — single batched call
    if erc20_assets:
        yield from fetch_address_balances(
            web3,
            address,
            erc20_assets,
            block_number=block_number,
            filter_zero=filter_zero,
        )


# ------------------------------------------------------------------
# Hypercore vault equity lookup
# ------------------------------------------------------------------

def _fetch_hypercore_vault_balance(
    pair: TradingPairIdentifier,
    safe_address: str,
    session: HyperliquidSession | None,
) -> OnChainBalance:
    """Fetch Hypercore vault equity via the Hyperliquid info API.

    Follows the same pattern as
    :py:func:`~tradeexecutor.ethereum.vault.hypercore_vault.create_hypercore_vault_value_func`
    for session creation and ``None`` handling.
    """

    vault_address = pair.pool_address
    assert vault_address, f"No pool_address set for Hypercore vault pair: {pair}"

    if session is None:
        is_testnet = pair.other_data.get("exchange_is_testnet", False)
        api_url = HYPERLIQUID_TESTNET_API_URL if is_testnet else HYPERLIQUID_API_URL
        session = create_hyperliquid_session(api_url=api_url)

    eq = fetch_user_vault_equity(
        session,
        user=safe_address,
        vault_address=vault_address,
    )

    if eq is not None:
        equity = eq.equity
        logger.debug(
            "Hypercore vault %s equity for %s: $%s",
            vault_address, safe_address, equity,
        )
    else:
        equity = Decimal(0)
        logger.debug(
            "No Hypercore vault position found for %s in vault %s",
            safe_address, vault_address,
        )

    return OnChainBalance(
        block_number=None,
        timestamp=native_datetime_utc_now(),
        asset=pair.base,
        amount=equity,
    )


# ------------------------------------------------------------------
# Asset filtering
# ------------------------------------------------------------------

def filter_onchain_assets(assets: Collection[AssetIdentifier]) -> list[AssetIdentifier]:
    """Remove assets that have no on-chain ERC-20 representation.

    Use this in :py:meth:`SyncModel.fetch_onchain_balances` implementations
    before calling ``fetch_address_balances()`` to avoid ``balanceOf`` calls
    on addresses that are not ERC-20 contracts (e.g. Hypercore vault tokens).

    Mirrors the off-chain classification in
    :py:func:`~tradeexecutor.strategy.asset.get_onchain_assets`.
    """
    return [a for a in assets if a.chain_id not in _OFFCHAIN_CHAIN_IDS]
