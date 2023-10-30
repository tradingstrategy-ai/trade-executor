import logging

from decimal import Decimal
from typing import Dict, Tuple

from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.types import JSONHexAddress


logger = logging.getLogger(__name__)


class OutOfSimulatedBalance(Exception):
    pass


class SimulatedWallet:
    """A wallet that keeps token balances by ERC-20 address.

    - We also simulate incoming and outgoing aToken and vToken amounts with :py:meth:`rebalance`.
    """

    def __init__(self):
        #: Raw balances token address -> balance
        self.balances: Dict[str, Decimal] = {}

        # token address -> asset info for debug logging
        self.tokens: Dict[str, AssetIdentifier] = {}
        self.nonce = 0

    def update_balance(self, token_address: JSONHexAddress, delta: Decimal, reason: str = None):
        """Change the token balance of some delta.

        Check that balance does not go zero.

        :param token_address:
            Token we receive or send

        :param delta:
            The amount of token, human units

        :param reason:
            Reason for this change.

            Only used for backtesting diagnostics.
        """
        assert token_address.lower() == token_address, "No checksummed addresses"
        token_symbol = self.get_token_symbol(token_address)
        assert isinstance(delta, Decimal), f"Expected decimal got: {delta.__class__}: {delta}"
        old_balance = self.balances.get(token_address, Decimal(0))
        new_balance = old_balance + delta

        reason = reason or "unhinted reason"
        logger.info("Wallet balance for %s: %f -> %f (%+f), %s", token_symbol, old_balance, new_balance, delta, reason)

        if new_balance < 0:
            raise OutOfSimulatedBalance(f"Simulated wallet balance went negative {new_balance} for token {token_symbol}")

        self.balances[token_address] = new_balance

    def set_balance(self, token_address: JSONHexAddress, amount: Decimal):
        """Directly set balance."""
        assert token_address.lower() == token_address, "No checksummed addresses"
        assert isinstance(amount, Decimal), f"Expected decimal got: {amount.__class__}: {amount}"
        self.balances[token_address] = amount

    def rebase(self, token_address: JSONHexAddress, new_amount: Decimal):
        """Set rebase token amount."""
        self.set_balance(token_address, new_amount)

    def get_balance(self, token_address: JSONHexAddress) -> Decimal:
        assert token_address.lower() == token_address, "No checksummed addresses"
        return self.balances.get(token_address, Decimal(0))

    def fetch_nonce_and_tx_hash(self) -> Tuple[int, str]:
        """Allocates a dummy nonce for a transaction.

        :return:
            Tuple (nonce, tx_hash)
        """
        nonce = self.nonce
        tx_hash = hex(nonce)
        self.nonce += 1
        return nonce, tx_hash

    def get_token_symbol(self, address: JSONHexAddress) -> str:
        asset = self.tokens.get(address, {})
        if asset:
            return asset.token_symbol
        return f"<token {address}>"

    def update_token_info(self, asset: AssetIdentifier):
        """Set the token info for a particular ERC-20.

        This way the wallet has metadata on what token it has and
        can produce better diagnostics output.
        """
        assert isinstance(asset, AssetIdentifier)
        self.tokens[asset.address] = asset


