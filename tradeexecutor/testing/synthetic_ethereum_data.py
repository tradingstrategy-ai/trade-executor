"""Synthetic Ethereum blockchain data generation."""

from eth_account import Account


def generate_random_ethereum_address() -> str:
    """Generate unchecksummed random Ethereum address."""
    account = Account.create()
    return account.address
