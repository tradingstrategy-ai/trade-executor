"""On-chain balance simulator for backtesting.

- We simulate ERC-20 token balances as they would appear in live trading

- The simulation will catch bugs like incorrect accounting or rounding
  errors for the trade execution

"""

import logging

from decimal import Decimal
from typing import Dict, Tuple

import pandas as pd

from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.utils.accuracy import QUANTITY_EPSILON
from tradeexecutor.state.types import JSONHexAddress


logger = logging.getLogger(__name__)


class OutOfSimulatedBalance(Exception):
    """Backtest detected a situation that we run out of a token balance.

    We simulate both internal ledger and on-chain balances in backtesting.
    This exception usually means that there is a bug an internal accounting of trade execution.
    """


class SimulatedWallet:
    """A wallet that keeps token balances by ERC-20 address.

    - Simulates different incoming and outgoing tokens from a wallet includive, aToken and vToken interest amounts with :py:meth:`rebalance`.

    - If a backtest tries to transfer a token it does not have, or does not have enough of it,
      raise an error

    - Will catch bugs in internal accounting
    """

    def __init__(self):
        #: Raw balances token address -> balance
        self.balances: Dict[str, Decimal] = {}

        #: token address -> asset info for debug logging
        self.tokens: Dict[str, AssetIdentifier] = {}

        #: Start with zero nonce like Ethereum acconts
        self.nonce = 0

    def update_balance(
        self,
        token: JSONHexAddress | AssetIdentifier,
        delta: Decimal,
        reason: str = None,
        epsilon=QUANTITY_EPSILON,
    ):
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

        :param epsilon:
            If the balance goes below this dust threshold, go all the way to zero
        """

        if isinstance(token, AssetIdentifier):
            self.update_token_info(token)

        if abs(delta) <= epsilon:
            # No changes, don't make noise for logging
            return

        token_symbol = self.get_token_symbol(token)
        assert isinstance(delta, Decimal), f"Expected decimal got: {delta.__class__}: {delta}"
        old_balance = self.get_balance(token)
        new_balance = old_balance + delta

        if abs(new_balance) < epsilon:
            # Decimal rounding errors
            new_balance = Decimal(0)

        reason = reason or "unhinted reason"
        logger.info("Wallet balance for %s: %f -> %f (%+f), %s", token_symbol, old_balance, new_balance, delta, reason)

        if new_balance < 0:
            raise OutOfSimulatedBalance(f"Simulated wallet balance went negative {new_balance} for token {token_symbol}, because of {reason}.\nOld balance was {old_balance}")

        if new_balance != 0 and new_balance <= epsilon:
            logger.info("Fixing dust balance to zero, calculated balance %f, epsilon %s", new_balance, epsilon)
            new_balance = Decimal(0)

        self.set_balance(token, new_balance)

    def set_balance(self, token: JSONHexAddress | AssetIdentifier, amount: Decimal):
        """Directly set balance.

        :param token:
            Token we receive or send.

            Give either raw address or asset definition.

            Any asset definion is automatically added to our internal tracking list for diagnostics.

        :param amount:
            New absolute balance.
        """

        if isinstance(token, AssetIdentifier):
            self.update_token_info(token)
            token_address = token.address
        else:
            token_address = token

        assert token_address.lower() == token_address, f"No checksummed addresses: {token_address}"
        assert token_address.startswith("0x")
        assert isinstance(amount, Decimal), f"Expected decimal got: {amount.__class__}: {amount}"
        self.balances[token_address] = amount

    def rebase(self, token: JSONHexAddress | AssetIdentifier, new_amount: Decimal):
        """Set rebase token amount.

        aToken / vToken accrues interest or debt.

        :param new_amount:
            Abs token amount on the chain
        """
        if isinstance(token, AssetIdentifier):
            token_address = token.address
        elif isinstance(token, str):
            token_address = token
        else:
            raise AssertionError(f"Does not understand: {token.__class__}: {token}")

        assert token_address in self.balances, f"Cannot rebase token we do not have: {token}"

        self.set_balance(token, new_amount)

    def get_balance(self, token: JSONHexAddress | AssetIdentifier) -> Decimal:
        """Get on-chain balance of one token.

        :return:
            Human-readable token balance
        """
        if isinstance(token, AssetIdentifier):
            token_address = token.address
        elif isinstance(token, str):
            token_address = token
        else:
            raise AssertionError(f"Does not understand: {token.__class__}: {token}")
        assert token_address.lower() == token_address, f"Non-checksummed address required: {token}"
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

    def get_token_symbol(self, token: JSONHexAddress | AssetIdentifier) -> str:
        """Get the human readable name of token for diagnostics output."""

        if isinstance(token, AssetIdentifier):
            return token.token_symbol

        # Cached symbol
        asset = self.tokens.get(token, {})
        if asset:
            return asset.token_symbol
        return f"<token {token}>"

    def update_token_info(self, asset: AssetIdentifier):
        """Set the token info for a particular ERC-20.

        This way the wallet has metadata on what token it has and
        can produce better diagnostics output.

        Automatically called by :py:meth:`update_balance`.
        """
        assert isinstance(asset, AssetIdentifier)
        self.tokens[asset.address] = asset

    def get_all_balances(self) -> pd.DataFrame:
        """Show the status of the wallet as a printable DataFrame.

        Example:

        .. code-block:: python

            print(wallet.get_all_balances())

        Output:

        .. code-block:: text

                                          Balance
            Token
            USDC                             9500
            aUSDC   998.4999999999999999687749774
            vWETH  0.3003021039165400376391259260

        """
        tokens = sorted([a for a in self.balances.keys()])
        data = [(self.get_token_symbol(a), self.get_balance(a)) for a in tokens]
        df = pd.DataFrame(data, columns=["Token", "Balance"])
        df = df.set_index("Token")
        return df

    def verify_balances(
        self,
        expected: Dict[AssetIdentifier, Decimal],
        epsilon=0.001,
    ) -> Tuple[bool, pd.DataFrame]:
        """Check that our simulated balances are what we expect.

        :return:
            Clean or not, all assets table.
        """
        data = []
        clean = True
        for asset, amount in expected.items():
            on_chain_balance = self.get_balance(asset.address)
            mismatch = False
            diff = abs(on_chain_balance - amount)
            if amount == 0:
                assert diff == 0
                relative_diff = 0
            else:
                relative_diff = diff / amount
                if relative_diff >= epsilon:
                    clean = False
                    mismatch = True

            data.append((asset.token_symbol, amount, on_chain_balance, diff, relative_diff, mismatch, epsilon))

        df = pd.DataFrame(data, columns=["Asset", "Expected", "Actual", "Diff", "Rel diff", "Mismatch", "Epsilon"])
        df.set_index("Asset")
        return clean, df

