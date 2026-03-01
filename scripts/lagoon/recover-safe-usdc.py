"""Recover USDC from a previous testnet Gnosis Safe back to the deployer.

After running :py:mod:`manual-trade-executor-multichain`, each test run
deploys a new Gnosis Safe on Arbitrum Sepolia. If the script aborts before
redeeming all shares, USDC may remain locked in the old Safe. This script
transfers that USDC back to the deployer so it can be reused in the next
test run.

The deployer is the sole owner of the Safe (threshold = 1) because the
deploy command defaults to ``multisig_owners = [deployer]`` when no
``MULTISIG_OWNERS`` env var is set.

Prerequisite environment variables
-----------------------------------

``JSON_RPC_ARBITRUM_SEPOLIA``
    Arbitrum Sepolia RPC URL.

``LAGOON_MULTCHAIN_TEST_PRIVATE_KEY``
    Private key of the deployer (must be the sole Safe owner).

Hard-coded addresses
--------------------

``SAFE_ADDRESS``
    The Gnosis Safe from the previous test run. Update this to the address
    shown in the test output or ``vault-record.json``.

Usage
-----

.. code-block:: shell

    source .local-test.env && poetry run python scripts/lagoon/recover-safe-usdc.py
"""

import os
import sys

from eth_account import Account
from web3 import Web3

from eth_defi.hotwallet import HotWallet
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.token import USDC_NATIVE_TOKEN, fetch_erc20_details


# Update this to the Safe address from your previous test run
SAFE_ADDRESS = "0x496e7b9eA5f568F95330396046F83e33451AD75b"
ARBITRUM_SEPOLIA_CHAIN_ID = 421614

SAFE_ABI = [
    {
        "inputs": [
            {"name": "to", "type": "address"},
            {"name": "value", "type": "uint256"},
            {"name": "data", "type": "bytes"},
            {"name": "operation", "type": "uint8"},
            {"name": "safeTxGas", "type": "uint256"},
            {"name": "baseGas", "type": "uint256"},
            {"name": "gasPrice", "type": "uint256"},
            {"name": "gasToken", "type": "address"},
            {"name": "refundReceiver", "type": "address"},
            {"name": "signatures", "type": "bytes"},
        ],
        "name": "execTransaction",
        "outputs": [{"name": "success", "type": "bool"}],
        "stateMutability": "payable",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "nonce",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "getThreshold",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"name": "to", "type": "address"},
            {"name": "value", "type": "uint256"},
            {"name": "data", "type": "bytes"},
            {"name": "operation", "type": "uint8"},
            {"name": "safeTxGas", "type": "uint256"},
            {"name": "baseGas", "type": "uint256"},
            {"name": "gasPrice", "type": "uint256"},
            {"name": "gasToken", "type": "address"},
            {"name": "refundReceiver", "type": "address"},
            {"name": "_nonce", "type": "uint256"},
        ],
        "name": "getTransactionHash",
        "outputs": [{"name": "", "type": "bytes32"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"name": "owner", "type": "address"}],
        "name": "isOwner",
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
]

ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"


def main():
    json_rpc = os.environ.get("JSON_RPC_ARBITRUM_SEPOLIA")
    private_key = os.environ.get("LAGOON_MULTCHAIN_TEST_PRIVATE_KEY")

    assert json_rpc, "JSON_RPC_ARBITRUM_SEPOLIA required"
    assert private_key, "LAGOON_MULTCHAIN_TEST_PRIVATE_KEY required"

    web3 = create_multi_provider_web3(json_rpc)
    deployer = HotWallet.from_private_key(private_key)
    deployer.sync_nonce(web3)

    usdc_address = USDC_NATIVE_TOKEN[ARBITRUM_SEPOLIA_CHAIN_ID]
    usdc = fetch_erc20_details(web3, usdc_address)

    safe_balance = usdc.fetch_balance_of(SAFE_ADDRESS)
    deployer_balance = usdc.fetch_balance_of(deployer.address)

    print(f"Deployer:     {deployer.address}")
    print(f"Safe:         {SAFE_ADDRESS}")
    print(f"Safe USDC:    {safe_balance}")
    print(f"Deployer USDC (before): {deployer_balance}")

    if safe_balance == 0:
        print("Safe has no USDC to recover.")
        return

    # Verify deployer is an owner
    safe = web3.eth.contract(
        address=Web3.to_checksum_address(SAFE_ADDRESS), abi=SAFE_ABI,
    )
    is_owner = safe.functions.isOwner(deployer.address).call()
    threshold = safe.functions.getThreshold().call()
    safe_nonce = safe.functions.nonce().call()
    print(f"Deployer is Safe owner: {is_owner}")
    print(f"Safe threshold: {threshold}")
    print(f"Safe nonce: {safe_nonce}")

    if not is_owner:
        print("ERROR: Deployer is not an owner of this Safe. Cannot recover USDC.")
        sys.exit(1)

    # Build the ERC-20 transfer call data
    raw_amount = usdc.convert_to_raw(safe_balance)
    transfer_data = usdc.contract.functions.transfer(
        deployer.address, raw_amount,
    ).build_transaction({"from": SAFE_ADDRESS})["data"]

    # Get the Safe transaction hash to sign
    tx_hash = safe.functions.getTransactionHash(
        Web3.to_checksum_address(usdc_address),  # to
        0,  # value
        bytes.fromhex(transfer_data[2:]),  # data (strip 0x)
        0,  # operation (Call)
        0,  # safeTxGas
        0,  # baseGas
        0,  # gasPrice
        ZERO_ADDRESS,  # gasToken
        ZERO_ADDRESS,  # refundReceiver
        safe_nonce,  # nonce
    ).call()

    # Sign with the deployer's key.
    # getTransactionHash returns the EIP-712 typed data hash — sign it
    # directly (no Ethereum prefix) and keep v as 27/28.
    signed = Account.unsafe_sign_hash(tx_hash, private_key)
    signature = (
        signed.r.to_bytes(32, "big")
        + signed.s.to_bytes(32, "big")
        + bytes([signed.v])
    )

    print(f"Executing Safe transaction to transfer {safe_balance} USDC to deployer...")

    tx = safe.functions.execTransaction(
        Web3.to_checksum_address(usdc_address),  # to
        0,  # value
        bytes.fromhex(transfer_data[2:]),  # data
        0,  # operation (Call)
        0,  # safeTxGas
        0,  # baseGas
        0,  # gasPrice
        ZERO_ADDRESS,  # gasToken
        ZERO_ADDRESS,  # refundReceiver
        signature,  # signatures
    ).build_transaction({
        "from": deployer.address,
        "nonce": deployer.current_nonce,
        "gas": 200_000,
    })

    signed_tx = web3.eth.account.sign_transaction(tx, private_key)
    tx_hash = web3.eth.send_raw_transaction(signed_tx.raw_transaction)
    print(f"  TX hash: {tx_hash.hex()}")

    receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    print(f"  Status: {'SUCCESS' if receipt['status'] == 1 else 'FAILED'}")
    print(f"  Gas used: {receipt['gasUsed']}")

    deployer_balance_after = usdc.fetch_balance_of(deployer.address)
    print(f"Deployer USDC (after): {deployer_balance_after}")
    print(f"Recovered: {deployer_balance_after - deployer_balance}")


if __name__ == "__main__":
    main()
