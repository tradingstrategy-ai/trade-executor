"""Vault metadata utils."""
from web3 import Web3

from eth_defi.abi import ZERO_ADDRESS_STR
from eth_defi.erc_4626.classification import create_vault_instance
from eth_defi.erc_4626.core import get_vault_protocol_name
from eth_defi.erc_4626.vault import ERC4626Vault
from eth_defi.vault.base import VaultBase

from tradeexecutor.ethereum.token import translate_token_details
from tradeexecutor.state.identifier import TradingPairIdentifier, TradingPairKind
from tradingstrategy.vault import VaultMetadata


def translate_vault_to_trading_pair(
    vault: ERC4626Vault,
) -> TradingPairIdentifier:
    """Construct trading pair identifier from a raw onchain vault contract."""

    chain_base = vault.share_token
    chain_quote = vault.denomination_token

    base = translate_token_details(chain_base)
    quote = translate_token_details(chain_quote)

    features = vault.features

    assert features, "vault.features was not for the ERC4626Vault() constructor, needed to use this"

    assert type(features) == set
    assert len(features) >= 1

    protocol_name = get_vault_protocol_name(vault.features)
    protocol_slug = protocol_name.replace(" ", "_").lower()

    try:
        management_fee = vault.get_management_fee("latest")
    except NotImplementedError:
        management_fee = None

    try:
        performance_fee = vault.get_performance_fee("latest")
    except NotImplementedError:
        performance_fee = None

    vault_metadata = VaultMetadata(
        vault_name=vault.name,
        protocol_name=protocol_name,
        protocol_slug=protocol_slug,
        features=list(features),
        performance_fee=performance_fee,
        management_fee=management_fee,  # No reader in ERC4626Vault
    )

    return TradingPairIdentifier(
        base=base,
        quote=quote,
        kind=TradingPairKind.vault,
        pool_address=vault.vault_address,
        exchange_address=ZERO_ADDRESS_STR,
        internal_id=int(vault.vault_address, 16),
        fee=0,
        reverse_token_order=False,
        exchange_name=vault.name,
        other_data={
            "vault_features": features,
            "vault_protocol": protocol_slug,
            "token_metadata": vault_metadata,
        }
    )


def get_vault_from_trading_pair(
    web3: Web3,
    pair: TradingPairIdentifier,
) -> VaultBase:
    """Get a vault instance for this process.

    - In-memory cache for constructed vault objects
    """

    vault = _vault_cache.get(pair)

    if vault is None:
        features = pair.get_vault_features()
        vault = create_vault_instance(
            web3,
            pair.pool_address,
            features=features,
        )

        _vault_cache[pair] = vault

    return vault


_vault_cache: dict[TradingPairIdentifier, VaultBase] = {}

