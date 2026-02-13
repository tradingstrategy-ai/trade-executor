"""Hex string conversion utilities for web3.py 7.x compatibility."""


def hexbytes_to_hex_str(value: bytes) -> str:
    """Convert bytes or HexBytes to a 0x-prefixed hex string.

    In web3.py 7.x, HexBytes.hex() no longer includes the 0x prefix.
    This function provides a consistent conversion.
    """
    hex_str = value.hex()
    if hex_str.startswith("0x"):
        return hex_str
    return "0x" + hex_str
