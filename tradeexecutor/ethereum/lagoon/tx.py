"""Velvet's vault transaction construction."""

import logging
from pprint import pformat
from typing import List, Optional
from decimal import Decimal

from web3.contract.contract import Contract, ContractFunction

from eth_defi.gas import GasPriceSuggestion, apply_gas
from eth_defi.hotwallet import HotWallet
from eth_defi.lagoon.vault import LagoonVault

from eth_defi.tx import AssetDelta
from eth_defi.velvet import VelvetVault
from tradeexecutor.ethereum.tx import TransactionBuilder, HotWalletTransactionBuilder
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction, BlockchainTransactionType, JSONAssetDelta
from tradeexecutor.state.pickle_over_json import encode_pickle_over_json

logger = logging.getLogger(__name__)


class LagoonTransactionBuilder(TransactionBuilder):
    """Build swap transactions with TradingStrategyModuleV0.prepareCall()."""

    def __init__(
        self,
        vault: LagoonVault,
        hot_wallet: HotWallet,
        extra_gnosis_gas=200_000,
    ):
        """
        :param extra_gnosis_gas:
            How many extra gas units we add on the top of gas estimations because of Gnosis Safe
        """
        super().__init__(vault.web3)
        self.vault = vault
        self.hot_wallet = hot_wallet
        self.extra_gnosis_gas = extra_gnosis_gas

    def init(self):
        self.hot_wallet.sync_nonce(self.web3)

    def get_token_delivery_address(self) -> str:
        """Get the target address for ERC-20 approve()"""
        return self.vault.safe_address

    def get_erc_20_balance_address(self) -> str:
        """Get the address that holds ERC-20 supply"""
        return self.vault.safe_address

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
        """Create a signed tranaction using TradingStrategyModuleV0.prepareCall()."""

        assert isinstance(contract, Contract), f"Expected Contract, got {contract}"
        assert isinstance(args_bound_func, ContractFunction), f"Expected ContractFunction, got {args_bound_func}"
        assert asset_deltas is not None, f"{args_bound_func.fn_name}() - cannot make Enzyme trades without asset_deltas set. Set to asset_deltas=[] for approve()"
        assert gas_limit is not None, f"Gas limit set to None for call: {args_bound_func} - LagoonTransactionBuilder needs explicit gas limit on all txs"

        def present(a):
            if type(a) == bytes:
                return "0x" + a.hex()
            return str(a)

        logger.info("Lagoon tx for %s.%s(%s), gas limit %d, deltas %s",
                    contract.address,
                    args_bound_func.fn_name,
                    ", ".join([present(a) for a in args_bound_func.args]),
                    gas_limit,
                    asset_deltas)

        gas_price_suggestion = gas_price_suggestion or self.fetch_gas_price_suggestion()

        bound_prepare_call = self.vault.transact_via_trading_strategy_module(
            args_bound_func
        )
        tx_data = bound_prepare_call.build_transaction({
            "gas": gas_limit + self.extra_gnosis_gas,
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
            type=BlockchainTransactionType.lagoon_vault,
            chain_id=self.chain_id,
            from_address=self.hot_wallet.address,
            contract_address=self.vault.trading_strategy_module_address,
            function_selector=bound_prepare_call.fn_name,
            transaction_args=bound_prepare_call.args,
            args=args_bound_func.args,
            wrapped_args=args_bound_func.args,
            wrapped_function_selector=args_bound_func.fn_name,
            signed_bytes=signed_bytes,
            signed_tx_object=encode_pickle_over_json(signed_tx),
            tx_hash=signed_tx.hash.hex(),
            nonce=signed_tx.nonce,
            details=tx_data,
            # Lagoon/TradingStrategyModuleV0 does not support guard level slippage tolerance
            asset_deltas=[JSONAssetDelta.from_asset_delta(a) for a in asset_deltas],
            other={"extra_gnosis_gas": self.extra_gnosis_gas},
            notes=notes,
        )
