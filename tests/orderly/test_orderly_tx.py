"""Test Orderly transaction builder."""

import os
from decimal import Decimal

import pytest
from web3 import Web3
from web3.contract.contract import Contract

from eth_defi.gas import estimate_gas_fees
from eth_defi.hotwallet import HotWallet
from eth_defi.orderly.vault import OrderlyVault
from eth_defi.token import TokenDetails
from eth_defi.tx import AssetDelta

from tradeexecutor.ethereum.orderly.tx import OrderlyTransactionBuilder
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction, BlockchainTransactionType


JSON_RPC_ARBITRUM_SEPOLIA = os.environ.get("JSON_RPC_ARBITRUM_SEPOLIA")
pytestmark = pytest.mark.skipif(not JSON_RPC_ARBITRUM_SEPOLIA, reason="No JSON_RPC_ARBITRUM_SEPOLIA environment variable")


def test_orderly_tx_builder_comprehensive(
    orderly_vault: OrderlyVault,
    asset_manager: HotWallet,
    broker_id: str,
    orderly_account_id: str,
    web3: Web3,
):
    """Test comprehensive OrderlyTransactionBuilder functionality including initialization, addresses, gas operations, and nonce management."""

    # Test initialization
    tx_builder = OrderlyTransactionBuilder(
        vault=orderly_vault,
        hot_wallet=asset_manager,
        broker_id=broker_id,
        orderly_account_id=orderly_account_id,
    )

    assert tx_builder.vault == orderly_vault
    assert tx_builder.hot_wallet == asset_manager
    assert tx_builder.broker_id == broker_id
    assert tx_builder.orderly_account_id == orderly_account_id
    assert tx_builder.extra_gas == 200_000

    # Test address management
    assert tx_builder.get_token_delivery_address() == orderly_vault.address
    assert tx_builder.get_erc_20_balance_address() == asset_manager.address
    assert tx_builder.get_gas_wallet_address() == asset_manager.address

    # Test gas balance retrieval
    balance = tx_builder.get_gas_wallet_balance()
    assert balance > 0
    assert isinstance(balance, Decimal)

    # Test nonce synchronization
    tx_builder.init()
    assert tx_builder.hot_wallet.current_nonce >= 0

    # Test gas price suggestions
    gas_suggestion = tx_builder.fetch_gas_price_suggestion()
    assert gas_suggestion is not None
    assert hasattr(gas_suggestion, 'get_tx_gas_params')


@pytest.mark.skip(reason="Requires actual contract interaction")
def test_orderly_tx_builder_sign_transaction(
    web3: Web3,
    orderly_tx_builder: OrderlyTransactionBuilder,
    usdc_token: TokenDetails,
):
    """Test signing a transaction with OrderlyTransactionBuilder."""

    # Create a mock approve transaction
    contract = usdc_token.contract
    args_bound_func = contract.functions.approve(
        orderly_tx_builder.vault.address,
        100 * 10**6  # 100 USDC
    )

    gas_price_suggestion = estimate_gas_fees(web3)

    # Create asset delta for the transaction
    asset_deltas = [
        AssetDelta(
            asset=usdc_token.address,
            raw_amount=100 * 10**6,
        )
    ]

    # Sign the transaction
    tx = orderly_tx_builder.sign_transaction(
        contract=contract,
        args_bound_func=args_bound_func,
        gas_limit=200_000,
        gas_price_suggestion=gas_price_suggestion,
        asset_deltas=asset_deltas,
        notes="Test approve transaction",
    )

    # Verify transaction structure
    assert isinstance(tx, BlockchainTransaction)
    assert tx.type == BlockchainTransactionType.orderly_vault
    assert tx.chain_id == web3.eth.chain_id
    assert tx.from_address == orderly_tx_builder.hot_wallet.address
    assert tx.contract_address == orderly_tx_builder.vault.address
    assert tx.function_selector == "approve"
    assert tx.notes == "Test approve transaction"

    # Check that orderly-specific data is included
    assert "broker_id" in tx.other
    assert tx.other["broker_id"] == orderly_tx_builder.broker_id
    assert "orderly_account_id" in tx.other
    assert tx.other["orderly_account_id"] == orderly_tx_builder.orderly_account_id
    assert "extra_gas" in tx.other
    assert tx.other["extra_gas"] == orderly_tx_builder.extra_gas