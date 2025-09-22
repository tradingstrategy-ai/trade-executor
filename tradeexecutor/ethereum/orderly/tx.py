"""Orderly's vault transaction construction."""

import logging
from pprint import pformat
from typing import List, Optional
from decimal import Decimal

from web3.contract.contract import Contract, ContractFunction

from eth_defi.abi import present_solidity_args
from eth_defi.gas import GasPriceSuggestion, apply_gas
from eth_defi.hotwallet import HotWallet
from eth_defi.tx import AssetDelta

from tradeexecutor.ethereum.orderly.orderly_vault import OrderlyVault
from tradeexecutor.ethereum.tx import TransactionBuilder, HotWalletTransactionBuilder
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction, BlockchainTransactionType, JSONAssetDelta
from tradeexecutor.state.pickle_over_json import encode_pickle_over_json

logger = logging.getLogger(__name__)


class OrderlyTransactionBuilder(TransactionBuilder):
    """Build swap transactions with Orderly vault operations."""

    def __init__(
        self,
        vault: OrderlyVault,
        hot_wallet: HotWallet,
        broker_id: str,
        orderly_account_id: str,
        extra_gas=200_000,
    ):
        """
        :param vault:
            Orderly vault instance
        :param hot_wallet:
            Hot wallet instance used for this execution
        :param broker_id:
            Orderly broker ID for transactions
        :param orderly_account_id:
            Orderly account ID for transactions
        :param extra_gas:
            How many extra gas units we add on the top of gas estimations
        """
        super().__init__(vault.web3)
        self.vault = vault
        self.hot_wallet = hot_wallet
        self.broker_id = broker_id
        self.orderly_account_id = orderly_account_id
        self.extra_gas = extra_gas

    def init(self):
        self.hot_wallet.sync_nonce(self.web3)

    def get_token_delivery_address(self) -> str:
        """Get the target address for ERC-20 approve()"""
        return self.vault.address

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
        """Create a signed transaction for Orderly vault operations."""

        assert isinstance(contract, Contract), f"Expected Contract, got {contract}"
        assert isinstance(args_bound_func, ContractFunction), f"Expected ContractFunction, got {args_bound_func}"
        assert asset_deltas is not None, f"{args_bound_func.fn_name}() - cannot make Orderly trades without asset_deltas set. Set to asset_deltas=[] for approve()"
        assert gas_limit is not None, f"Gas limit set to None for call: {args_bound_func} - OrderlyTransactionBuilder needs explicit gas limit on all txs"

        logger.info("Orderly tx for <%s>.%s(%s), gas limit %d, deltas %s",
                    args_bound_func.address,
                    args_bound_func.fn_name,
                    present_solidity_args(args_bound_func.args),
                    gas_limit,
                    asset_deltas)

        assert contract.address == args_bound_func.address, f"Contract address {contract.address} does not match bound function address {args_bound_func.address}"

        gas_price_suggestion = gas_price_suggestion or self.fetch_gas_price_suggestion()

        tx_data = args_bound_func.build_transaction({
            "gas": gas_limit + self.extra_gas,
            "from": self.hot_wallet.address,
            "chainId": self.chain_id,
        })
        tx_data.update(gas_price_suggestion.get_tx_gas_params())

        if "maxFeePerGas" in tx_data and "gasPrice" in tx_data:
            # We can have only one
            # https://ethereum.stackexchange.com/questions/121361/web3py-issue-on-avalanche-when-using-maxpriorityfeepergas-and-maxfeepergas
            del tx_data["gasPrice"]

        signed_tx = self.hot_wallet.sign_transaction_with_new_nonce(tx_data)
        signed_bytes = signed_tx.rawTransaction.hex()

        # Needed for get_swap_transactions() hack
        tx_data["function"] = args_bound_func.fn_name

        return BlockchainTransaction(
            type=BlockchainTransactionType.orderly_vault,
            chain_id=self.chain_id,
            from_address=self.hot_wallet.address,
            contract_address=self.vault.address,
            function_selector=args_bound_func.fn_name,
            transaction_args=args_bound_func.args,
            args=args_bound_func.args,
            signed_bytes=signed_bytes,
            signed_tx_object=encode_pickle_over_json(signed_tx),
            tx_hash=signed_tx.hash.hex(),
            nonce=signed_tx.nonce,
            details=tx_data,
            asset_deltas=[JSONAssetDelta.from_asset_delta(a) for a in asset_deltas],
            other={
                "extra_gas": self.extra_gas,
                "broker_id": self.broker_id,
                "orderly_account_id": self.orderly_account_id,
            },
            notes=notes,
        )