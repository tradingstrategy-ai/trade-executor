"""Test Derive address resolution for strategy metadata.

Verifies that resolve_derive_addresses() correctly extracts public
addresses from credentials and that create_metadata() merges them
into on_chain_data.smart_contracts.

No external services or real credentials required — uses synthetic keys.
"""

import secrets

from eth_account import Account

from tradeexecutor.cli.bootstrap import create_metadata
from tradeexecutor.exchange_account.derive import DeriveNetwork
from tradeexecutor.exchange_account.utils import resolve_derive_addresses
from tradeexecutor.strategy.execution_model import AssetManagementMode
from tradingstrategy.chain import ChainId


def test_resolve_derive_addresses_with_all_credentials():
    """Resolve Derive addresses when owner key and wallet address are both provided.

    1. Generate synthetic owner and session private keys
    2. Call resolve_derive_addresses with explicit wallet address
    3. Verify all public addresses are present and no private keys leak
    """
    # 1. Generate synthetic keys
    owner_key = "0x" + secrets.token_hex(32)
    session_key = "0x" + secrets.token_hex(32)
    wallet_address = "0x" + secrets.token_hex(20)

    owner_account = Account.from_key(owner_key)
    session_account = Account.from_key(session_key)

    # 2. Resolve addresses
    result = resolve_derive_addresses(
        derive_session_private_key=session_key,
        derive_owner_private_key=owner_key,
        derive_wallet_address=wallet_address,
        derive_network=DeriveNetwork.mainnet,
    )

    # 3. Verify all expected keys are present
    assert result["derive_wallet_address"] == wallet_address
    assert result["derive_owner_address"] == owner_account.address
    assert result["derive_session_key_address"] == session_account.address
    assert result["derive_network"] == "mainnet"

    # Verify no private keys in values
    all_values = " ".join(str(v) for v in result.values())
    assert owner_key not in all_values
    assert session_key not in all_values


def test_resolve_derive_addresses_without_owner_key():
    """Resolve Derive addresses when only session key and wallet address are provided.

    1. Generate synthetic session key and wallet address (no owner key)
    2. Call resolve_derive_addresses without owner key
    3. Verify owner address is absent, other fields present
    """
    # 1. Generate synthetic keys
    session_key = "0x" + secrets.token_hex(32)
    wallet_address = "0x" + secrets.token_hex(20)

    session_account = Account.from_key(session_key)

    # 2. Resolve without owner key
    result = resolve_derive_addresses(
        derive_session_private_key=session_key,
        derive_owner_private_key=None,
        derive_wallet_address=wallet_address,
        derive_network=DeriveNetwork.testnet,
    )

    # 3. Verify owner address is absent
    assert "derive_owner_address" not in result
    assert result["derive_wallet_address"] == wallet_address
    assert result["derive_session_key_address"] == session_account.address
    assert result["derive_network"] == "testnet"


def test_derive_addresses_in_metadata():
    """Verify Derive addresses are merged into metadata smart_contracts.

    1. Build a derive_addresses dict
    2. Call create_metadata with derive_addresses
    3. Verify on_chain_data.smart_contracts contains the Derive keys
    """
    # 1. Build addresses
    derive_addresses = {
        "derive_wallet_address": "0xDerive1234",
        "derive_owner_address": "0xOwner5678",
        "derive_session_key_address": "0xSession9abc",
        "derive_network": "mainnet",
    }

    # 2. Create metadata with Derive addresses
    metadata = create_metadata(
        name="Test Derive Strategy",
        short_description="Test",
        long_description=None,
        icon_url=None,
        asset_management_mode=AssetManagementMode.dummy,
        chain_id=ChainId.ethereum,
        vault=None,
        derive_addresses=derive_addresses,
    )

    # 3. Verify all Derive keys present in smart_contracts
    sc = metadata.on_chain_data.smart_contracts
    assert sc["derive_wallet_address"] == "0xDerive1234"
    assert sc["derive_owner_address"] == "0xOwner5678"
    assert sc["derive_session_key_address"] == "0xSession9abc"
    assert sc["derive_network"] == "mainnet"
