"""EVM transaction construction."""
import datetime
import logging
from typing import List

from eth_account._utils.typed_transactions import TypedTransaction
from eth_account.datastructures import SignedTransaction
from hexbytes import HexBytes
from web3 import Web3
from web3.contract import Contract, ContractFunction

from eth_defi.gas import GasPriceSuggestion, apply_gas
from eth_defi.hotwallet import HotWallet
from eth_defi.revert_reason import fetch_transaction_revert_reason
from eth_defi.confirmation import broadcast_transactions, wait_transactions_to_complete, \
    broadcast_and_wait_transactions_to_complete
from eth_defi.tx import decode_signed_transaction
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction


#: How many gas units we assume ERC-20 approval takes
#: TODO: Move to a better model
APPROVE_GAS_LIMIT = 100_000


logger = logging.getLogger(__name__)


class TransactionBuilder:
    """Create transactions from the hot wallet and store them in the state.

    Creates trackable transactions. TransactionHelper is initialised
    at the start of the each cycle.

    Transaction builder can prepare multiple transactions in one batch.
    For all tranactions, we use the previously prepared gas price information.
    """

    def __init__(self,
                 web3: Web3,
                 hot_wallet: HotWallet,
                 gas_fees: GasPriceSuggestion,
                 ):
        assert isinstance(gas_fees, GasPriceSuggestion)
        self.hot_wallet = hot_wallet
        self.web3 = web3
        self.gas_fees = gas_fees
        # Read once at the start, then cache
        self.chain_id = web3.eth.chain_id

    def sign_transaction(
            self,
            args_bound_func: ContractFunction,
            gas_limit: int
    ) -> BlockchainTransaction:
        """Createa a signed tranaction and set up tx broadcast parameters.

        :param args_bound_func: Web3 function thingy
        :param gas_limit: Max gas per this transaction
        """

        logger.info("Signing transactions using gas gee method %s for %s", self.gas_fees, args_bound_func)

        tx = args_bound_func.build_transaction({
            "chainId": self.chain_id,
            "from": self.hot_wallet.address,
            "gas": gas_limit,
        })

        apply_gas(tx, self.gas_fees)

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

    def create_transaction(
            self,
            contract: Contract,
            function_selector: str,
            args: tuple,
            gas_limit: int,
    ) -> BlockchainTransaction:
        """Create a trackable transaction for the trade executor state.

        - Sets up the state management for the transaction

        - Creates the signed transaction from the hot wallet
        """
        #
        # tx = token.functions.approve(
        #     deployment.router.address,
        #     amount,
        # ).build_transaction({
        #     'chainId': web3.eth.chain_id,
        #     'gas': 100_000,  # Estimate max 100k per approval
        #     'from': hot_wallet.address,
        # })
        contract_func = contract.functions[function_selector]
        args_bound_func = contract_func(*args)
        return self.sign_transaction(args_bound_func, gas_limit)

    def broadcast(self, tx: "BlockchainTransaction") -> HexBytes:
        """Broadcast the transaction.

        :return: tx_hash
        """
        signed_tx = TransactionBuilder.as_signed_tx(tx)
        tx.broadcasted_at = datetime.datetime.utcnow()
        return broadcast_transactions(self.web3, [signed_tx])[0]

    @staticmethod
    def decode_transaction(tx: "BlockchainTransaction") -> dict:
        """Get raw transaction data out from the signed tx bytes."""
        return decode_signed_transaction(tx.signed_bytes)

    @staticmethod
    def as_signed_tx(tx: "BlockchainTransaction") -> SignedTransaction:
        """Get a transaction as a format ready to broadcast."""
        # TODO: Make hash, r, s, v filled up as well
        return SignedTransaction(rawTransaction=tx.signed_bytes, hash=None, r=0, s=0, v=0)

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
        """

        # Log what we are doing
        for tx in txs:
            logger.info("Broadcasting and executing transaction %s", tx)

        logger.info("Waiting %d txs to confirm", len(txs))
        assert isinstance(confirmation_block_count, int)

        # tx hash -> BlockchainTransaction map
        tx_hashes = {t.tx_hash: t for t in txs}

        signed_txs = [TransactionBuilder.as_signed_tx(t) for t in txs]

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
