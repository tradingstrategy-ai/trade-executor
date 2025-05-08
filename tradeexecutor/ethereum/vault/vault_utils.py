from eth_defi.abi import ZERO_ADDRESS_STR
from eth_defi.erc_4626.core import get_vault_protocol_name
from eth_defi.erc_4626.vault import ERC4626Vault
from tradeexecutor.ethereum.token import translate_token_details
from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier, TradingPairKind


def get_vault_trading_pair(
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
        }
    )
