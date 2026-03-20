"""Helpers for secret key handling."""


def ensure_0x_prefixed_private_key(private_key: str) -> str:
    """Normalise a hex private key to the 0x-prefixed format expected by eth-defi."""
    if private_key.startswith("0x"):
        return private_key
    return f"0x{private_key}"
