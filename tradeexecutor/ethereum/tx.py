"""Transaction helpers."""
import datetime

from eth_account.datastructures import SignedTransaction
from hexbytes import HexBytes
from web3 import Web3
from web3.contract import Contract, ContractFunction

from eth_defi.gas import GasPriceSuggestion, apply_gas
from eth_defi.hotwallet import HotWallet
from eth_defi.txmonitor import broadcast_transactions
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction


#: How many gas units we assume ERC-20 approval takes
APPROVE_GAS_LIMIT = 100_000


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

        tx = args_bound_func.buildTransaction({
            "chainId": self.chain_id,
            "from": self.hot_wallet.address,
            "gas": gas_limit,
        })

        apply_gas(tx, self.gas_fees)

        signed_tx = self.hot_wallet.sign_transaction_with_new_nonce(tx)
        signed_bytes = signed_tx.rawTransaction.hex()

        return BlockchainTransaction(
            chain_id=self.chain_id,
            contract_address=args_bound_func.address,
            function_selector=args_bound_func.fn_name,
            args=args_bound_func.args,
            signed_bytes=signed_bytes,
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
        # ).buildTransaction({
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
        signed_tx = TransactionBuilder.get_signed_transaction(tx)
        tx.broadcasted_at = datetime.datetime.utcnow()
        return broadcast_transactions(self.web3, [signed_tx])[0]

    @staticmethod
    def get_signed_transaction(tx: "BlockchainTransaction") -> SignedTransaction:
        """Convert expanded info to low-level Web3 transaction object."""

        # TODO: Make hash, r, s, v filled up as well
        return SignedTransaction(rawTransaction=tx.signed_bytes, hash=None, r=0, s=0, v=0)




