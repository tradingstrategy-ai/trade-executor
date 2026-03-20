"""Web3 compatibility helpers."""

from typing import Any

from eth_account.signers.local import LocalAccount
from eth_keys.datatypes import PrivateKey
from eth_typing import HexStr
from web3.middleware.signing import SignAndSendRawMiddlewareBuilder


def construct_sign_and_send_raw_middleware(
    private_key_or_account: LocalAccount | PrivateKey | HexStr | bytes | list[LocalAccount | PrivateKey | HexStr | bytes] | set[LocalAccount | PrivateKey | HexStr | bytes] | tuple[LocalAccount | PrivateKey | HexStr | bytes, ...],
) -> Any:
    """Create a signing middleware compatible with modern Web3 middleware builder API."""
    return SignAndSendRawMiddlewareBuilder.build(private_key_or_account)
