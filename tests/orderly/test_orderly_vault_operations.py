"""Test Orderly vault operations integration with trade executor."""

import os
import pytest
from decimal import Decimal
from eth_typing import HexAddress
from web3 import Web3

from eth_defi.hotwallet import HotWallet
from eth_defi.orderly.vault import OrderlyVault, deposit, withdraw
from eth_defi.token import TokenDetails
from eth_defi.trace import assert_transaction_success_with_explanation

from tradeexecutor.ethereum.orderly.tx import OrderlyTransactionBuilder


JSON_RPC_ARBITRUM_SEPOLIA = os.environ.get("JSON_RPC_ARBITRUM_SEPOLIA")
pytestmark = pytest.mark.skipif(not JSON_RPC_ARBITRUM_SEPOLIA, reason="No JSON_RPC_ARBITRUM_SEPOLIA environment variable")


@pytest.mark.skip(reason="Requires proper vault deployment and USDC balance")
def test_orderly_deposit_integration(
    web3: Web3,
    orderly_vault: OrderlyVault,
    hot_wallet: HotWallet,
    usdc_token: TokenDetails,
    broker_id: str,
    orderly_account_id: HexAddress,
    orderly_tx_builder: OrderlyTransactionBuilder,
):
    """Test depositing USDC to Orderly vault using transaction builder."""
    
    initial_balance = usdc_token.fetch_balance_of(hot_wallet.address)
    assert initial_balance > Decimal(100), "Need at least 100 USDC for test"
    
    deposit_amount = 100 * 10**6  # 100 USDC in raw units
    
    # Get deposit functions from web3-ethereum-defi
    approve_fn, get_deposit_fee_fn, deposit_fn = deposit(
        vault=orderly_vault,
        token=usdc_token.contract,
        amount=deposit_amount,
        depositor_address=hot_wallet.address,
        orderly_account_id=orderly_account_id,
        broker_id=broker_id,
        token_id="USDC",
    )
    
    # Step 1: Approve USDC spending
    tx_data = approve_fn.build_transaction({
        "from": hot_wallet.address,
        "gas": 200_000
    })
    signed_tx = hot_wallet.sign_transaction_with_new_nonce(tx_data)
    tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
    assert_transaction_success_with_explanation(web3, tx_hash)
    
    # Step 2: Get deposit fee
    deposit_fee = get_deposit_fee_fn.call()
    
    # Step 3: Make the deposit
    tx_data = deposit_fn.build_transaction({
        "from": hot_wallet.address,
        "value": deposit_fee,
        "gas": 500_000
    })
    signed_tx = hot_wallet.sign_transaction_with_new_nonce(tx_data)
    tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
    assert_transaction_success_with_explanation(web3, tx_hash)
    
    # Verify USDC balance decreased
    final_balance = usdc_token.fetch_balance_of(hot_wallet.address)
    assert final_balance == initial_balance - Decimal(100)


@pytest.mark.skip(reason="Requires proper vault deployment and existing deposit")
def test_orderly_withdraw_integration(
    web3: Web3,
    orderly_vault: OrderlyVault,
    hot_wallet: HotWallet,
    usdc_token: TokenDetails,
    broker_id: str,
    orderly_account_id: HexAddress,
):
    """Test withdrawing USDC from Orderly vault."""
    
    initial_balance = usdc_token.fetch_balance_of(hot_wallet.address)
    withdraw_amount = 50 * 10**6  # 50 USDC in raw units
    
    # Get withdraw functions from web3-ethereum-defi
    approve_fn, get_withdraw_fee_fn, withdraw_fn = withdraw(
        vault=orderly_vault,
        token=usdc_token.contract,
        amount=withdraw_amount,
        wallet_address=hot_wallet.address,
        orderly_account_id=orderly_account_id,
        broker_id=broker_id,
        token_id="USDC",
    )
    
    # Step 1: Approve (if needed)
    tx_data = approve_fn.build_transaction({
        "from": hot_wallet.address,
        "gas": 200_000
    })
    signed_tx = hot_wallet.sign_transaction_with_new_nonce(tx_data)
    tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
    assert_transaction_success_with_explanation(web3, tx_hash)
    
    # Step 2: Get withdraw fee (should be same as deposit fee function)
    withdraw_fee = get_withdraw_fee_fn.call()
    
    # Step 3: Make the withdrawal
    tx_data = withdraw_fn.build_transaction({
        "from": hot_wallet.address,
        "value": withdraw_fee,
        "gas": 500_000
    })
    signed_tx = hot_wallet.sign_transaction_with_new_nonce(tx_data)
    tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
    assert_transaction_success_with_explanation(web3, tx_hash)
    
    # Verify USDC balance increased
    final_balance = usdc_token.fetch_balance_of(hot_wallet.address)
    assert final_balance == initial_balance + Decimal(50)


def test_orderly_vault_and_tx_builder_configuration(
    orderly_tx_builder: OrderlyTransactionBuilder,
    orderly_vault: OrderlyVault,
    broker_id: str,
    orderly_account_id: str,
):
    """Test that vault and transaction builder are properly configured with correct parameters and contract access."""

    # Test transaction builder parameters
    assert orderly_tx_builder.broker_id == broker_id
    assert orderly_tx_builder.orderly_account_id == orderly_account_id

    # These are required for Orderly deposits/withdrawals
    assert orderly_tx_builder.broker_id is not None
    assert orderly_tx_builder.orderly_account_id is not None
    assert len(orderly_tx_builder.orderly_account_id) == 66  # 0x + 64 hex chars

    # Test vault contract access
    assert orderly_vault.contract is not None
    assert orderly_vault.address is not None
    assert orderly_vault.web3 is not None

    # Check that required contract methods exist (these should be available from ABI)
    contract_functions = dir(orderly_vault.contract.functions)

    # Note: Actual function names depend on the Orderly vault ABI
    # This is a basic check that the contract interface is loaded
    assert len(contract_functions) > 0

    # Verify vault address is consistent
    assert orderly_tx_builder.vault == orderly_vault