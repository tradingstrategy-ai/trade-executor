"""Velvet's vault transaction construction."""

import logging
from typing import List, Optional
from decimal import Decimal

from web3.contract.contract import Contract, ContractFunction

from eth_defi.gas import GasPriceSuggestion
from eth_defi.hotwallet import HotWallet

from eth_defi.tx import AssetDelta
from eth_defi.velvet import VelvetVault
from tradeexecutor.ethereum.tx import TransactionBuilder
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction, BlockchainTransactionType, JSONAssetDelta
from tradeexecutor.state.pickle_over_json import encode_pickle_over_json

logger = logging.getLogger(__name__)


class VelvetTransactionBuilder(TransactionBuilder):

    def __init__(
        self,
        vault: VelvetVault,
        hot_wallet: HotWallet,
    ):
        super().__init__(vault.web3)
        self.vault = vault

    @property
    def hot_wallet(self) -> HotWallet:
        """Get the underlying web3 connection."""
        return self.vault_controlled_wallet.hot_wallet

    def init(self):
        self.hot_wallet.sync_nonce(self.web3)

    def get_token_delivery_address(self) -> str:
        """Get the target address for ERC-20 approve()"""
        return self.vault.vault_address

    def get_erc_20_balance_address(self) -> str:
        """Get the address that holds ERC-20 supply"""
        return self.vault.vault_address

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
        tx_data: dict,
        notes: str = "",
    ) -> BlockchainTransaction:

        signed_tx = self.hot_wallet.sign_transaction_with_new_nonce(tx_data)
        signed_bytes = signed_tx.rawTransaction

        return BlockchainTransaction(
            type=BlockchainTransactionType.enzyme_vault,
            chain_id=self.chain_id,
            from_address=self.hot_wallet.address,
            contract_address=tx_data["address"],
            function_selector=None,
            transaction_args=None,
            args=None,
            wrapped_args=None,
            wrapped_function_selector=None,
            signed_bytes=signed_bytes,
            signed_tx_object=encode_pickle_over_json(signed_tx),
            tx_hash=signed_tx.hash.hex(),
            nonce=signed_tx.nonce,
            details=tx_data,
            notes=notes,
        )
