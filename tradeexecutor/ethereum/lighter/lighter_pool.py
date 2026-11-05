"""Lighter public-pool identities, signer and API session for pool execution.

A Lighter public pool is a vault position on synthetic dataset chain 9998,
identified as ``lighter-pool-<account index>``. Its custody is the Ethereum
satellite Safe's Lighter master account (chain 1), and its mint, burn and
secure-withdrawal transactions are signed for Lighter's signing chain 304.
Keep these three identities separate.

Position quantity is the protocol's raw share amount and the price is USDC
per raw share, the same unit as
:py:attr:`eth_defi.lighter.execution.LighterPoolState.share_value`. The share
asset keeps the dataset's representational 18 decimals (see
:py:mod:`tradeexecutor.strategy.dex_data_translation`), but pool code never
scales shares by them: dividing raw shares by 10**18 would produce quantities
below ``QUANTITY_EPSILON`` that position accounting snaps to zero. Pool shares
are not ERC-20 tokens, so no on-chain balance conversion uses the field.

See ``deps/web3-ethereum-defi/docs/README-lighter-core-execution.md`` for the
upstream execution contract and :py:mod:`tradeexecutor.ethereum.lighter.lighter_pool_routing`
for the executor lifecycle.
"""

import asyncio
import logging

from eth_defi.abi import ZERO_ADDRESS_STR
from eth_defi.lighter.execution import LighterAccountIdentity
from eth_defi.lighter.constants import LIGHTER_USDC_ETHEREUM
from eth_defi.lighter.sdk import create_lighter_signer_client
from eth_defi.lighter.session import LighterSession
from tradingstrategy.chain import ChainId
from tradingstrategy.vault import VaultMetadata

from tradeexecutor.exchange_account.lighter import create_executor_lighter_session
from tradeexecutor.exchange_account.lighter_operator import LighterOperatorRecord
from tradeexecutor.state.identifier import (
    LIGHTER_POOL_ADDRESS_PREFIX,
    AssetIdentifier,
    TradingPairIdentifier,
    TradingPairKind,
)


logger = logging.getLogger(__name__)

#: Vault protocol slug of Lighter public pools in vault metadata.
LIGHTER_POOL_PROTOCOL = "lighter"

#: Representational share decimals, matching dataset translation; never used to scale raw shares.
LIGHTER_POOL_SHARE_DECIMALS = 18


def create_lighter_pool_pair(
    pool_index: int,
    *,
    internal_id: int,
    internal_exchange_id: int,
    name: str | None = None,
    operator_fee: float | None = None,
) -> TradingPairIdentifier:
    """Create a trading pair for one Lighter public pool.

    The quote is Ethereum native USDC re-homed to dataset chain 9998, as
    :py:func:`~tradeexecutor.ethereum.vault.hypercore_vault.create_hypercore_vault_pair`
    re-homes HyperEVM USDC, because a pair's base and quote share one chain.
    Funding and custody use :py:meth:`TradingPairIdentifier.get_funding_chain_id`
    and :py:func:`~tradeexecutor.state.identifier.normalise_custody_chain_id`.

    :param pool_index:
        Lighter public-pool account index.
    :param internal_id:
        Unique pair id within the strategy universe.
    :param internal_exchange_id:
        Exchange id of the Lighter pool exchange in the universe.
    :param name:
        Human-readable pool name, used as the share-token symbol.
    :param operator_fee:
        Operator fee on positive investor profit, as a fraction.
    :return:
        Vault pair routed by the ``lighter_pool`` router.
    """
    assert pool_index >= 0
    lighter_chain = ChainId.lighter.value
    address = f"{LIGHTER_POOL_ADDRESS_PREFIX}{pool_index}"
    base = AssetIdentifier(
        chain_id=lighter_chain,
        address=address,
        token_symbol=name or f"LIGHTER-{pool_index}",
        decimals=LIGHTER_POOL_SHARE_DECIMALS,
    )
    quote = AssetIdentifier(
        chain_id=lighter_chain,
        address=LIGHTER_USDC_ETHEREUM,
        token_symbol="USDC",
        decimals=6,
    )
    metadata = VaultMetadata(
        vault_name=name or address,
        protocol_name="Lighter",
        protocol_slug=LIGHTER_POOL_PROTOCOL,
        features=[],
        performance_fee=operator_fee,
        management_fee=0,
    )
    return TradingPairIdentifier(
        base=base,
        quote=quote,
        pool_address=address,
        exchange_address=ZERO_ADDRESS_STR,
        internal_id=internal_id,
        internal_exchange_id=internal_exchange_id,
        fee=0.0,
        kind=TradingPairKind.vault,
        exchange_name="Lighter",
        # Token order only matters for AMM pools; a synthetic id is not hex.
        reverse_token_order=False,
        other_data={
            "vault_protocol": LIGHTER_POOL_PROTOCOL,
            "token_metadata": metadata,
        },
    )


def create_lighter_pool_session() -> LighterSession:
    """Create the Ethereum Lighter API session used for pool execution and valuation.

    Reuses the executor's long-retry session for replay-safe reads. Upstream
    submits signed transactions through a separate session that retries only
    Lighter's rate-limit refusals.

    :return:
        Ethereum deployment session.
    """
    return create_executor_lighter_session()


def create_lighter_pool_identity(operator: LighterOperatorRecord) -> LighterAccountIdentity:
    """Build the upstream account identity from the executor's operator record.

    :param operator:
        Owner-only record of the Safe-owned master account and its API key.
    :return:
        Identity with Ethereum custody chain 1 and signing chain 304.
    """
    return LighterAccountIdentity(account_index=operator.account_index, api_key_index=operator.api_key_index)


class LighterPoolSigner:
    """Synchronous official-SDK signer for one account and API key.

    Implements :py:class:`eth_defi.lighter.execution.LighterSyncSigner`. The
    SDK constructor creates an HTTP client and expects an event loop, so the
    client is created and closed on a private loop. Signing itself is
    synchronous native code. Never log this object: it holds the API key.

    :param operator:
        Owner-only operator record.
    """

    def __init__(self, operator: LighterOperatorRecord):
        self.api_key_index = operator.api_key_index
        self._loop = asyncio.new_event_loop()

        async def _create():
            return create_lighter_signer_client(operator.account_index, operator.api_key_index, operator.api_private_key)

        self._client = self._loop.run_until_complete(_create())

    def __repr__(self) -> str:
        return f"<LighterPoolSigner api_key_index={self.api_key_index}>"

    def sign_mint_shares(self, public_pool_index: int, share_amount: int, *, nonce: int, api_key_index: int):
        """Sign a pool mint for an explicit nonce."""
        return self._client.sign_mint_shares(public_pool_index, share_amount, nonce=nonce, api_key_index=api_key_index)

    def sign_burn_shares(self, public_pool_index: int, share_amount: int, *, nonce: int, api_key_index: int):
        """Sign a pool burn for an explicit nonce."""
        return self._client.sign_burn_shares(public_pool_index, share_amount, nonce=nonce, api_key_index=api_key_index)

    def sign_withdraw(self, asset_index: int, route_type: int, amount: int, *, nonce: int, api_key_index: int):
        """Sign a secure collateral withdrawal for an explicit nonce."""
        return self._client.sign_withdraw(asset_index, route_type, amount, nonce=nonce, api_key_index=api_key_index)

    def create_auth_token(self) -> str:
        """Create a short-lived read token for authenticated withdrawal history.

        :return:
            Auth token; never log it.
        """
        token, error = self._client.create_auth_token_with_expiry(api_key_index=self.api_key_index)
        if error is not None or not token:
            raise RuntimeError("Lighter auth-token creation failed")
        return token

    def close(self) -> None:
        """Close the SDK HTTP client and its private loop."""
        try:
            self._loop.run_until_complete(self._client.close())
        finally:
            self._loop.close()


def create_lighter_pool_signer(operator: LighterOperatorRecord) -> LighterPoolSigner:
    """Create the pool signer for an operator record.

    A separate function so the simulated sequencer can substitute its signer.

    :param operator:
        Owner-only operator record.
    :return:
        Synchronous signer; call ``close()`` when done.
    """
    return LighterPoolSigner(operator)
