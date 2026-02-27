"""Bridge USDC from Base Sepolia → Arbitrum Sepolia via CCTP.

Quick utility to fund the deployer wallet on Arb Sepolia for cross-chain vault testing.
Uses low-level CCTP V2 transfer functions (no vault needed).

Usage:
    source .local-test.env && poetry run python scripts/lagoon/bridge-usdc-to-arb-sepolia.py
"""

import logging
import os
import sys
import time

from eth_account import Account
from web3 import Web3

from eth_defi.cctp.attestation import fetch_attestation
from eth_defi.cctp.constants import TESTNET_CHAIN_ID_TO_CCTP_DOMAIN
from eth_defi.cctp.receive import prepare_receive_message
from eth_defi.cctp.transfer import prepare_approve_for_burn, prepare_deposit_for_burn
from eth_defi.hotwallet import HotWallet
from eth_defi.token import USDC_NATIVE_TOKEN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

AMOUNT_USDC = 9  # Bridge 9 USDC
BASE_SEPOLIA_CHAIN_ID = 84532
ARB_SEPOLIA_CHAIN_ID = 421614


def main():
    private_key = os.environ.get("LAGOON_MULTCHAIN_TEST_PRIVATE_KEY") or os.environ.get("PRIVATE_KEY")
    base_rpc = os.environ.get("JSON_RPC_BASE_SEPOLIA")
    arb_rpc = os.environ.get("JSON_RPC_ARBITRUM_SEPOLIA")

    if not all([private_key, base_rpc, arb_rpc]):
        print("Need LAGOON_MULTCHAIN_TEST_PRIVATE_KEY (or PRIVATE_KEY), JSON_RPC_BASE_SEPOLIA, JSON_RPC_ARBITRUM_SEPOLIA")
        sys.exit(1)

    base_web3 = Web3(Web3.HTTPProvider(base_rpc))
    arb_web3 = Web3(Web3.HTTPProvider(arb_rpc))

    account = Account.from_key(private_key)
    hot_wallet = HotWallet(account)
    hot_wallet.sync_nonce(base_web3)

    deployer = account.address
    amount_raw = AMOUNT_USDC * 10**6

    # Check USDC balance on Base Sepolia
    usdc_addr = USDC_NATIVE_TOKEN.get(BASE_SEPOLIA_CHAIN_ID)
    assert usdc_addr, f"No USDC address for chain {BASE_SEPOLIA_CHAIN_ID}"
    from eth_defi.abi import get_deployed_contract
    usdc = get_deployed_contract(base_web3, "ERC20MockDecimals.json", usdc_addr)
    balance = usdc.functions.balanceOf(deployer).call()
    logger.info("USDC balance on Base Sepolia: %.2f", balance / 1e6)
    assert balance >= amount_raw, f"Need {AMOUNT_USDC} USDC but have {balance / 1e6:.2f}"

    # Step 1: Approve USDC to TokenMessenger on Base Sepolia
    logger.info("Approving %d USDC to TokenMessenger on Base Sepolia...", AMOUNT_USDC)
    approve_fn = prepare_approve_for_burn(base_web3, amount_raw)
    signed_tx = hot_wallet.sign_bound_call_with_new_nonce(
        approve_fn, tx_params={"gas": 200_000}, web3=base_web3, fill_gas_price=True,
    )
    tx_hash = base_web3.eth.send_raw_transaction(signed_tx.rawTransaction)
    receipt = base_web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    assert receipt["status"] == 1, f"Approve failed: {tx_hash.hex()}"
    logger.info("Approve confirmed: %s", tx_hash.hex())

    # Step 2: depositForBurn — send USDC to deployer on Arb Sepolia
    logger.info("Burning %d USDC on Base Sepolia for Arb Sepolia...", AMOUNT_USDC)
    burn_fn = prepare_deposit_for_burn(
        base_web3,
        amount=amount_raw,
        destination_chain_id=ARB_SEPOLIA_CHAIN_ID,
        mint_recipient=deployer,
    )
    signed_tx = hot_wallet.sign_bound_call_with_new_nonce(
        burn_fn, tx_params={"gas": 500_000}, web3=base_web3, fill_gas_price=True,
    )
    burn_tx_hash = base_web3.eth.send_raw_transaction(signed_tx.rawTransaction)
    receipt = base_web3.eth.wait_for_transaction_receipt(burn_tx_hash, timeout=120)
    assert receipt["status"] == 1, f"Burn failed: {burn_tx_hash.hex()}"
    logger.info("Burn confirmed: %s", burn_tx_hash.hex())

    # Step 3: Wait for attestation from Circle Iris API
    source_domain = TESTNET_CHAIN_ID_TO_CCTP_DOMAIN[BASE_SEPOLIA_CHAIN_ID]
    logger.info("Waiting for CCTP attestation (this may take 15-20 minutes on testnet)...")

    from eth_defi.cctp.constants import IRIS_API_SANDBOX_URL
    attestation_result = fetch_attestation(
        source_domain=source_domain,
        transaction_hash=burn_tx_hash.hex(),
        timeout=1800.0,  # 30 min timeout
        poll_interval=10.0,
        api_base_url=IRIS_API_SANDBOX_URL,
    )
    logger.info("Attestation received! Status: %s", attestation_result.status)

    # Step 4: Receive on Arb Sepolia
    logger.info("Receiving USDC on Arb Sepolia...")
    arb_wallet = HotWallet(account)
    arb_wallet.sync_nonce(arb_web3)

    receive_fn = prepare_receive_message(arb_web3, attestation_result.message, attestation_result.attestation)
    signed_tx = arb_wallet.sign_bound_call_with_new_nonce(
        receive_fn, tx_params={"gas": 500_000}, web3=arb_web3, fill_gas_price=True,
    )
    receive_tx_hash = arb_web3.eth.send_raw_transaction(signed_tx.rawTransaction)
    receipt = arb_web3.eth.wait_for_transaction_receipt(receive_tx_hash, timeout=120)
    assert receipt["status"] == 1, f"Receive failed: {receive_tx_hash.hex()}"
    logger.info("Receive confirmed: %s", receive_tx_hash.hex())

    # Check final balance
    arb_usdc_addr = USDC_NATIVE_TOKEN.get(ARB_SEPOLIA_CHAIN_ID)
    arb_usdc = get_deployed_contract(arb_web3, "ERC20MockDecimals.json", arb_usdc_addr)
    final_balance = arb_usdc.functions.balanceOf(deployer).call()
    logger.info("Final USDC balance on Arb Sepolia: %.2f", final_balance / 1e6)
    logger.info("Bridge complete!")


if __name__ == "__main__":
    main()
