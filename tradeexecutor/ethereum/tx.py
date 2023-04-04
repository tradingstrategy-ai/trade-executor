"""EVM transaction construction.

- Base class :py:class:`TransactionBuilder` for different transaction building mechanisms

- The default :py:class:`HotWalletTransactionBuilder`
"""
import datetime
import logging
from abc import abstractmethod, ABC
from typing import List, Optional

from eth_account.datastructures import SignedTransaction
from hexbytes import HexBytes
from web3 import Web3
from web3.contract.contract import ContractFunction

from eth_defi.gas import GasPriceSuggestion, apply_gas, estimate_gas_fees
from eth_defi.hotwallet import HotWallet
from eth_defi.tx import AssetDelta
from eth_defi.revert_reason import fetch_transaction_revert_reason
from eth_defi.confirmation import broadcast_transactions, \
    broadcast_and_wait_transactions_to_complete
from eth_defi.tx import decode_signed_transaction
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction



logger = logging.getLogger(__name__)


class TransactionBuilder(ABC):
    """Base class for different transaction builders.

    We can build different transactions depending if we execute them directly from the hot wallet,
    or through a smart contract:

    - Hot wallet based, see :py:class:`HotWalletTransactionBuilder`

    - Vault based, see :py:class:`tradeexecutor.enzyme.tx.EnzymeTransactionBuilder`

    Life cycle:

    - TransactionBuilder is created once at the application startup

    See also :py:meth:`tradeeexecutor.ethereum.routing_state.EthereumRoutingState.create_signed_transaction`.
    """

    @abstractmethod
    def fetch_gas_price_suggestion(self) -> GasPriceSuggestion:
        """Calculate the suggested gas price based on a policy."""

    @abstractmethod
    def sign_transaction(
            self,
            args_bound_func: ContractFunction,
            gas_limit: int,
            gas_price_suggestion: Optional[GasPriceSuggestion] = None,
            asset_deltas: Optional[List[AssetDelta]] = None,
    ) -> BlockchainTransaction:
        """Createa a signed tranaction and set up tx broadcast parameters.

        :param args_bound_func:
            A Solidity function with its arguments bound to the function instance.

        :param gas_limit:
            Max gas limit per this transaction.

            The transaction will fail if the gas limit is exceeded/

        :param gas_price_suggestion:
            What gas price will be used.

            Support old-style and London style transactions.

        :param asset_deltas:
            Expected assets inbound and outbound.

        :return:
            Prepared BlockchainTransaction instance.

            This transaction object can be stored in the persistent state.
        """

    @staticmethod
    def serialise_to_broadcast_format(tx: "BlockchainTransaction") -> SignedTransaction:
        """Prepare a transaction as a format ready to broadcast."""
        # TODO: Make hash, r, s, v filled up as well
        return SignedTransaction(rawTransaction=tx.signed_bytes, hash=None, r=0, s=0, v=0)

    @staticmethod
    def decode_signed_bytes(tx: "BlockchainTransaction") -> dict:
        """Get raw transaction data out from the signed tx bytes."""
        return decode_signed_transaction(tx.signed_bytes)


class HotWalletTransactionBuilder(TransactionBuilder):
    """Create transactions from the hot wallet and store them in the state.

    Creates trackable transactions. TransactionHelper is initialised
    at the start of the each cycle.

    Transaction builder can prepare multiple transactions in one batch.
    For all tranactions, we use the previously prepared gas price information.
    """

    def __init__(self,
                 web3: Web3,
                 hot_wallet: HotWallet,
                 ):

        self.hot_wallet = hot_wallet
        self.web3 = web3
        # Read once at the start, then cache
        self.chain_id = web3.eth.chain_id

    def fetch_gas_price_suggestion(self) -> GasPriceSuggestion:
        """Calculate the suggested gas price based on a policy."""
        return estimate_gas_fees(self.web3)

    def sign_transaction(
            self,
            args_bound_func: ContractFunction,
            gas_limit: int,
            gas_price_suggestion: Optional[GasPriceSuggestion] = None,
            asset_deltas: Optional[List[AssetDelta]] = None,
    ) -> BlockchainTransaction:
        """Sign a transaction with the hot wallet private key."""

        if gas_price_suggestion is None:
            gas_price_suggestion = self.fetch_gas_price_suggestion()

        logger.info("Signing transactions using gas fee method %s for %s", gas_price_suggestion, args_bound_func)

        tx = args_bound_func.build_transaction({
            "chainId": self.chain_id,
            "from": self.hot_wallet.address,
            "gas": gas_limit,
        })

        apply_gas(tx, gas_price_suggestion)

        signed_tx = self.hot_wallet.sign_transaction_with_new_nonce(tx)
        signed_bytes = signed_tx.rawTransaction.hex()

        return BlockchainTransaction(
            chain_id=self.chain_id,
            from_address=self.hot_wallet.address,
            contract_address=args_bound_func.address,
            function_selector=args_bound_func.fn_name,
            args=args_bound_func.args,
            signed_bytes=signed_bytes,
            tx_hash=signed_tx.hash.hex(),
            nonce=signed_tx.nonce,
            details=tx,
        )

    def broadcast(self, tx: "BlockchainTransaction") -> HexBytes:
        """Broadcast the transaction.

        Push the transaction to the peer-to-peer network / MEV relay
        to be includedin the

        Sets the `tx.broadcasted_at` timestamp.

        :return:
            Transaction hash, or `tx_hash`
        """
        signed_tx = HotWalletTransactionBuilder.serialise_to_broadcast_format(tx)
        tx.broadcasted_at = datetime.datetime.utcnow()
        return broadcast_transactions(self.web3, [signed_tx])[0]

    @staticmethod
    def broadcast_and_wait_transactions_to_complete(
            web3: Web3,
            txs: List[BlockchainTransaction],
            confirmation_block_count=0,
            max_timeout=datetime.timedelta(minutes=5),
            poll_delay=datetime.timedelta(seconds=1),
            revert_reasons=False,
    ):
        """Watch multiple transactions executed at parallel.

        Modifies the given transaction objects in-place
        and updates block inclusion and succeed status.

        .. note ::

            This method is designed to be used only in unit testing
            as a shortcut.

        """

        # Log what we are doing
        for tx in txs:
            logger.info("Broadcasting and executing transaction %s", tx)

        logger.info("Waiting %d txs to confirm", len(txs))
        assert isinstance(confirmation_block_count, int)

        # tx hash -> BlockchainTransaction map
        tx_hashes = {t.tx_hash: t for t in txs}

        signed_txs = [HotWalletTransactionBuilder.serialise_to_broadcast_format(t) for t in txs]

        now_ = datetime.datetime.utcnow()
        for tx in txs:
            tx.broadcasted_at = now_

        receipts = broadcast_and_wait_transactions_to_complete(
            web3,
            signed_txs,
            confirm_ok=False,
            confirmation_block_count=confirmation_block_count,
            max_timeout=max_timeout,
            poll_delay=poll_delay)

        now_ = datetime.datetime.utcnow()

        # Update persistant status of transactions
        # based on the result read from the chain
        for tx_hash, receipt in receipts.items():
            tx = tx_hashes[tx_hash.hex()]
            status = receipt["status"] == 1

            reason = None
            if not status:
                if revert_reasons:
                    reason = fetch_transaction_revert_reason(web3, tx_hash)

            tx.set_confirmation_information(
                now_,
                receipt["blockNumber"],
                receipt["blockHash"].hex(),
                receipt.get("effectiveGasPrice", 0),
                receipt["gasUsed"],
                status,
                reason
            )
