import random

from eth_account import Account


def generate_random_ethereum_address() -> str:
    """Get unchecksummed random Ethereum address."""
    account = Account.create()
    return account.address
