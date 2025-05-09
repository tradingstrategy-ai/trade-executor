"""EVM transaction construction.

- Base class :py:class:`TransactionBuilder` for different transaction building mechanisms

- The default :py:class:`HotWalletTransactionBuilder`
"""
import datetime
import logging
from decimal import Decimal
from abc import abstractmethod, ABC
from typing import List, Optional

from eth_account.datastructures import SignedTransaction
from hexbytes import HexBytes
from web3 import Web3
from web3.contract.contract import ContractFunction, Contract

from eth_defi.gas import GasPriceSuggestion, apply_gas, estimate_gas_fees
from eth_defi.hotwallet import HotWallet, SignedTransactionWithNonce
from eth_defi.tx import AssetDelta
from eth_defi.revert_reason import fetch_transaction_revert_reason
from eth_defi.confirmation import broadcast_transactions, \
    broadcast_and_wait_transactions_to_complete
from eth_defi.tx import decode_signed_transaction
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction, JSONAssetDelta
from tradeexecutor.state.pickle_over_json import encode_pickle_over_json
from tradeexecutor.state.types import Percent

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

    def __init__(self, web3: Web3):
        self.web3 = web3
        # Read once at the start, then cache
        self.chain_id: int = web3.eth.chain_id

    def get_internal_slippage_tolerance(self) -> Percent | None:
        """Get the slippage tolerance configured for the asset receiver.

        - Vaults have their own security rules against slippage tolerance

        - Any vault slippage tolerance must be higher than trade slippage tolerance

        :return:
            E.g. 0.9995 for 5 BPS slippage tolerance
        """
        return None

    def fetch_gas_price_suggestion(self) -> GasPriceSuggestion:
        """Calculate the suggested gas price based on a policy."""
        return estimate_gas_fees(self.web3)

    def broadcast(self, tx: BlockchainTransaction) -> HexBytes:
        """Broadcast the transaction.

        Push the transaction to the peer-to-peer network / MEV relay
        to be includedin the

        Sets the `tx.broadcasted_at` timestamp.

        :return:
            Transaction hash, or `tx_hash`
        """
        signed_tx = self.serialise_to_broadcast_format(tx)
        tx.broadcasted_at = datetime.datetime.utcnow()
        return broadcast_transactions(self.web3, [signed_tx])[0]

    @staticmethod
    def serialise_to_broadcast_format(tx: "BlockchainTransaction") -> SignedTransactionWithNonce:
        """Prepare a transaction as a format ready to broadcast.

        - We pass this as :py:class:`SignedTransactionWithNonce`
          as the logging output will have more information to
          diagnose broadcasting issues
        """
        signed_tx = tx.get_tx_object()
        return signed_tx

    @staticmethod
    def decode_signed_bytes(tx: "BlockchainTransaction") -> dict:
        """Get raw transaction data out from the signed tx bytes."""
        return decode_signed_transaction(tx.signed_bytes)

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

    @abstractmethod
    def init(self):
        """Initialise the transaction builder.

        Called on application startup.
        """

    @abstractmethod
    def sign_transaction(
            self,
            contract: Contract,
            args_bound_func: ContractFunction,
            gas_limit: Optional[int] = None,
            gas_price_suggestion: Optional[GasPriceSuggestion] = None,
            asset_deltas: Optional[List[AssetDelta]] = None,
            notes: str = "",
    ) -> BlockchainTransaction:
        """Createa a signed tranaction and set up tx broadcast parameters.

        :param args_bound_func:
            A Solidity function with its arguments bound to the function instance.

        :param gas_limit:
            Max gas limit per this transaction.

            The transaction will fail if the gas limit is exceeded.

            If set to `None` then it is up to the signed to figure it out
            based on the function hints.

        :param gas_price_suggestion:
            What gas price will be used.

            Support old-style and London style transactions.

        :param asset_deltas:
            Expected assets inbound and outbound.

        :return:
            Prepared BlockchainTransaction instance.

            This transaction object can be stored in the persistent state.
        """

    @abstractmethod
    def get_token_delivery_address(self) -> str:
        """Get the target address for ERC-20 token delivery.

        Where do Uniswap should send the tokens after a swap.
        """

    @abstractmethod
    def get_erc_20_balance_address(self) -> str:
        """Get the address that holds ERC-20 supply"""

    @abstractmethod
    def get_gas_wallet_address(self) -> str:
        """Get the address that holds native token for gas fees"""

    @abstractmethod
    def get_gas_wallet_balance(self) -> Decimal:
        """Get the balance of the native currency (ETH, BNB, MATIC) of the wallet.

        Useful to check if you have enough cryptocurrency for the gas fees.
        """


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
        super().__init__(web3)
        self.hot_wallet = hot_wallet

    def init(self):
        self.hot_wallet.sync_nonce(self.web3)

    def get_token_delivery_address(self) -> str:
        """Get the target address for ERC-20 approve()"""
        return self.hot_wallet.address

    def get_erc_20_balance_address(self) -> str:
        """Get the address that holds ERC-20 supply"""
        return self.hot_wallet.address

    def get_gas_wallet_address(self) -> str:
        """Get the address that holds native token for gas fees"""
        return self.hot_wallet.address

    def get_gas_wallet_balance(self) -> Decimal:
        """Get the balance of the native currency (ETH, BNB, MATIC) of the wallet.

        Useful to check if you have enough cryptocurrency for the gas fees.
        """
        return self.hot_wallet.get_native_currency_balance(self.web3)

    def sign_transaction(
        self,
        contract: Contract,
        args_bound_func: ContractFunction,
        gas_limit: Optional[int] = None,
        gas_price_suggestion: Optional[GasPriceSuggestion] = None,
        asset_deltas: Optional[List[AssetDelta]] = None,
        notes: str = "",
    ) -> BlockchainTransaction:
        """Sign a transaction with the hot wallet private key.

        See also :py:meth:`sign_transaction_data`.
        """

        assert isinstance(contract, Contract), f"Expected Contract, got {contract}"
        assert isinstance(args_bound_func, ContractFunction), f"Expected ContractFunction, got {args_bound_func}"

        if gas_price_suggestion is None:
            gas_price_suggestion = self.fetch_gas_price_suggestion()

        if gas_limit is None:
            gas_limit = 500_000

        tx = args_bound_func.build_transaction({
            "chainId": self.chain_id,
            "from": Web3.to_checksum_address(self.hot_wallet.address),
            "gas": gas_limit,
        })

        apply_gas(tx, gas_price_suggestion)

        import ipdb ; ipdb.set_trace()
        signed_tx = self.hot_wallet.sign_transaction_with_new_nonce(tx)

        logger.info(
            "Signed transactions using gas fee method %s for %s, tx's nonce is %d, gas limit %d",
            gas_price_suggestion,
            args_bound_func.fn_name,
            signed_tx.nonce,
            gas_limit,
        )

        signed_bytes = signed_tx.rawTransaction.hex()

        if asset_deltas is None:
            asset_deltas = []

        return BlockchainTransaction(
            chain_id=self.chain_id,
            from_address=self.hot_wallet.address,
            contract_address=args_bound_func.address,
            function_selector=args_bound_func.fn_name,
            transaction_args=args_bound_func.args,
            args=args_bound_func.args,
            wrapped_args=None,
            signed_bytes=signed_bytes,
            signed_tx_object=encode_pickle_over_json(signed_tx),
            tx_hash=signed_tx.hash.hex(),
            nonce=signed_tx.nonce,
            details=tx,
            asset_deltas=[JSONAssetDelta.from_asset_delta(a) for a in asset_deltas],
            notes=notes,
        )
