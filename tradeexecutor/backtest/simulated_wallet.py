"""On-chain balance simulator for backtesting.

- We simulate ERC-20 token balances as they would appear in live trading

- The simulation will catch bugs like incorrect accounting or rounding
  errors for the trade execution

"""

import logging

from decimal import Decimal
from typing import Dict, Tuple

from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.types import JSONHexAddress


logger = logging.getLogger(__name__)


class OutOfSimulatedBalance(Exception):
    """Backtest detected a situation that we run out of a token balance.

    We simulate both internal ledger and on-chain balances in backtesting.
    This exception usually means that there is a bug an internal accounting of trade execution.
    """


class SimulatedWallet:
    """A wallet that keeps token balances by ERC-20 address.

    - We also simulate incoming and outgoing aToken and vToken amounts with :py:meth:`rebalance`.
    """

    def __init__(self):
        #: Raw balances token address -> balance
        self.balances: Dict[str, Decimal] = {}

        #: token address -> asset info for debug logging
        self.tokens: Dict[str, AssetIdentifier] = {}

        #: Start with zero nonce like Ethereum acconts
        self.nonce = 0

    def update_balance(self, token: JSONHexAddress | AssetIdentifier, delta: Decimal, reason: str = None):
        """Change the token balance of some delta.

        Check that balance does not go zero.

        :param token:
            Token we receive or send.

            Give either raw address or asset definition.

            Any asset definion is automatically added to our internal tracking list for diagnostics.

        :param delta:
            The amount of token, human units

        :param reason:
            Reason for this change.

            Only used for backtesting diagnostics.
        """

        if isinstance(token, AssetIdentifier):
            self.update_token_info(token)
            token = token.address

        assert token.lower() == token, "No checksummed addresses"
        token_symbol = self.get_token_symbol(token)
        assert isinstance(delta, Decimal), f"Expected decimal got: {delta.__class__}: {delta}"
        old_balance = self.balances.get(token, Decimal(0))
        new_balance = old_balance + delta

        reason = reason or "unhinted reason"
        logger.info("Wallet balance for %s: %f -> %f (%+f), %s", token_symbol, old_balance, new_balance, delta, reason)

        if new_balance < 0:
            raise OutOfSimulatedBalance(f"Simulated wallet balance went negative {new_balance} for token {token_symbol}, because of {reason}")

        self.balances[token] = new_balance

    def set_balance(self, token_address: JSONHexAddress, amount: Decimal):
        """Directly set balance."""
        assert token_address.lower() == token_address, "No checksummed addresses"
        assert isinstance(amount, Decimal), f"Expected decimal got: {amount.__class__}: {amount}"
        self.balances[token_address] = amount

    def rebase(self, token_address: JSONHexAddress, new_amount: Decimal):
        """Set rebase token amount."""
        self.set_balance(token_address, new_amount)

    def get_balance(self, token: JSONHexAddress | AssetIdentifier) -> Decimal:
        """Get on-chain balance of one token.

        :return:
            Human-readable token balance
        """
        if isinstance(token, AssetIdentifier):
            token = token.address
        assert token.lower() == token, "No checksummed addresses"
        return self.balances.get(token, Decimal(0))

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
        """Get the human readable name of token for diagnostics output."""
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


