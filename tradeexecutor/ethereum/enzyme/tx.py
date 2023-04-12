"""Enzyme's vault transaction construction."""

import logging
from typing import List, Optional
from decimal import Decimal

from web3.contract.contract import Contract, ContractFunction

from eth_defi.enzyme.vault import Vault
from eth_defi.enzyme.vault_controlled_wallet import VaultControlledWallet, EnzymeVaultTransaction
from eth_defi.gas import GasPriceSuggestion, apply_gas
from eth_defi.hotwallet import HotWallet

from eth_defi.tx import AssetDelta
from tradeexecutor.ethereum.tx import TransactionBuilder
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction, BlockchainTransactionType, JSONAssetDelta

logger = logging.getLogger(__name__)


class EnzymeTransactionBuilder(TransactionBuilder):
    """Create transactions that are executed by Enzyme's vaults.

    Creates trackable transactions. TransactionHelper is initialised
    at the start of the each cycle.

    Transaction builder can prepare multiple transactions in one batch.
    For all tranactions, we use the previously prepared gas price information.
    """

    def __init__(self,
                 hot_wallet: HotWallet,
                 vault: Vault,
                 ):
        super().__init__(vault.web3)
        self.vault_controlled_wallet = VaultControlledWallet(vault, hot_wallet)

    @property
    def vault(self) -> Vault:
        """Get the underlying web3 connection."""
        return self.vault_controlled_wallet.vault

    @property
    def hot_wallet(self) -> HotWallet:
        """Get the underlying web3 connection."""
        return self.vault_controlled_wallet.hot_wallet

    def init(self):
        self.hot_wallet.sync_nonce(self.web3)

    def get_token_delivery_address(self) -> str:
        """Get the target address for ERC-20 approve()"""
        assert self.vault.generic_adapter is not None, "GenericAdapter smart contract information for Enzyme vault has not been passed. Cannot make transactions."
        return self.vault.generic_adapter.address

    def get_erc_20_balance_address(self) -> str:
        """Get the address that holds ERC-20 supply"""
        return self.vault.vault.address

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
    ) -> BlockchainTransaction:
        """Createa a signed tranaction and set up tx broadcast parameters.

        :param args_bound_func:
            Web3 function thingy
        :param gas_limit:
            Max gas per this transaction

        :return:
            Prepared BlockchainTransaction instance
        """

        assert isinstance(contract, Contract), f"Expected Contract, got {contract}"
        assert isinstance(args_bound_func, ContractFunction), f"Expected ContractFunction, got {args_bound_func}"

        assert asset_deltas is not None, f"{args_bound_func.fn_name}() - cannot make Enzyme trades without asset_deltas set. Set to asset_deltas=[] for approve()"

        if gas_limit is None:
            gas_limit = 2_500_000

        logger.info("Enzyme tx for %s.%s(%s), gas limit %d, deltas %s",
                    contract.address,
                    args_bound_func.fn_name,
                    ", ".join([str(a) for a in args_bound_func.args]),
                    gas_limit,
                    asset_deltas)

        enzyme_tx = EnzymeVaultTransaction(
            contract,
            args_bound_func,
            gas_limit,
            asset_deltas=asset_deltas,
        )

        gas_price_suggestion = gas_price_suggestion or self.fetch_gas_price_suggestion()
        gas_data = gas_price_suggestion.get_tx_gas_params()

        signed_tx, execute_calls_bound_func = self.vault_controlled_wallet.sign_transaction_with_new_nonce(enzyme_tx, gas_data)
        signed_bytes = signed_tx.rawTransaction.hex()



        return BlockchainTransaction(
            type=BlockchainTransactionType.enzyme_vault,
            chain_id=self.chain_id,
            from_address=self.hot_wallet.address,
            contract_address=self.vault.comptroller.address,
            function_selector=execute_calls_bound_func.fn_name,
            args=execute_calls_bound_func.args,
            signed_bytes=signed_bytes,
            tx_hash=signed_tx.hash.hex(),
            nonce=signed_tx.nonce,
            details=enzyme_tx.as_json_friendly_dict(),
            asset_deltas=[JSONAssetDelta.from_asset_delta(a) for a in asset_deltas],
        )