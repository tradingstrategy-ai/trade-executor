"""Vault metadata utils."""
import hashlib

from web3 import Web3

from eth_defi.abi import ZERO_ADDRESS_STR
from eth_defi.erc_4626.classification import create_vault_instance, create_vault_instance_autodetect
from eth_defi.erc_4626.core import GENERIC_ERC4626_PROTOCOL_NAME, GENERIC_ERC4626_PROTOCOL_SLUG, get_vault_protocol_name, is_generic_erc4626_protocol_slug
from eth_defi.erc_4626.vault import ERC4626Vault
from eth_defi.vault.base import VaultBase

from tradeexecutor.ethereum.token import translate_token_details
from tradeexecutor.state.identifier import TradingPairIdentifier, TradingPairKind
from tradingstrategy.vault import VaultMetadata


def is_explicit_generic_erc4626_pair(pair: TradingPairIdentifier) -> bool:
    """Has this pair opted in to generic synchronous ERC-4626 handling."""
    return is_generic_erc4626_protocol_slug(pair.get_vault_protocol())


def translate_vault_to_trading_pair(
    vault: ERC4626Vault,
) -> TradingPairIdentifier:
    """Construct trading pair identifier from a raw onchain vault contract."""

    chain_base = vault.share_token
    chain_quote = vault.denomination_token

    base = translate_token_details(chain_base)
    quote = translate_token_details(chain_quote)

    features = vault.features
    if features == {}:
        features = set()
    assert features is not None, "vault.features was not set for the ERC4626Vault() constructor, pass set() for a known synchronous vault"

    assert type(features) == set

    if features:
        protocol_name = get_vault_protocol_name(features)
        protocol_slug = protocol_name.replace(" ", "_").lower()
    else:
        protocol_name = GENERIC_ERC4626_PROTOCOL_NAME
        protocol_slug = GENERIC_ERC4626_PROTOCOL_SLUG

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

    # Derive a deterministic pair id from the chain id and vault address,
    # constrained to the JavaScript safe-integer range: the state validator
    # rejects ints above 2^53-1, so the raw 160-bit address value would make
    # the state file unwritable for any position trading this pair. Hashing
    # (rather than masking address bits) keeps the id uniformly distributed
    # even for vanity/CREATE2-mined addresses, and including the chain id
    # separates same-address deployments on different chains (a common
    # CREATE2/Safe pattern). Residual collisions are negligible (birthday
    # bound over 53 bits) and fail loudly: pair universe construction asserts
    # unique ids.
    internal_id = int.from_bytes(
        hashlib.sha256(f"{vault.chain_id}:{vault.vault_address.lower()}".encode("ascii")).digest()[:8],
        "big",
    ) & ((1 << 53) - 1)

    return TradingPairIdentifier(
        base=base,
        quote=quote,
        kind=TradingPairKind.vault,
        pool_address=vault.vault_address,
        exchange_address=ZERO_ADDRESS_STR,
        internal_id=internal_id,
        fee=0,
        reverse_token_order=False,
        exchange_name=vault.name,
        other_data={
            # Store as a list, not a set: the state JSON validator rejects sets,
            # and the dataset-loaded pair path (dex_data_translation) also uses a list.
            "vault_features": list(features),
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
        if features or is_explicit_generic_erc4626_pair(pair):
            vault = create_vault_instance(
                web3,
                pair.pool_address,
                features=features or set(),
            )
        else:
            vault = create_vault_instance_autodetect(
                web3,
                pair.pool_address,
            )

        _vault_cache[pair] = vault

    return vault


_vault_cache: dict[TradingPairIdentifier, VaultBase] = {}
