"""Standalone diagnostic script for HyperCore EVM escrow activation and deposit.

Tests the EVM escrow flow on mainnet or testnet WITHOUT a Lagoon vault —
uses a plain EOA wallet (HotWallet) to call CoreDepositWallet directly.

This isolates HyperCore escrow behaviour from Lagoon/Safe complexity.

Findings
--------

- **EOA deposits clear immediately on testnet**: When calling
  ``CoreDepositWallet.deposit()`` or ``depositFor()`` directly from an EOA,
  the EVM escrow clears within the first poll iteration (~0s). USDC appears
  in the spot clearinghouse balance instantly with no pending ``evmEscrows``
  entries. This was tested on HyperEVM testnet (chain 998) with a 3 USDC
  deposit on 2026-03-09.

- **Lagoon/Safe deposits time out on testnet**: When the same ``depositFor()``
  call is routed through a Safe multisig via ``transact_via_trading_strategy_module``
  (i.e. ``TradingStrategyModuleV0.performCall()``), the ``coreUserExists``
  precompile never returns ``True`` within the 60s timeout. The ``msg.sender``
  to ``CoreDepositWallet`` is the Safe address (correct), and the ``recipient``
  parameter also points to the Safe address.

- This confirms the escrow issue is **not** a testnet infrastructure problem
  but is specific to the Safe/module execution path. Possible causes include
  guard whitelisting of ``depositFor``, Safe nonce issues, or HyperCore
  treating module-delegated calls differently from direct EOA calls.

Environment variables
---------------------
NETWORK
    ``mainnet`` (default) or ``testnet``
PRIVATE_KEY or HYPERCORE_WRITER_TEST_PRIVATE_KEY
    Wallet private key (must hold HYPE for gas and USDC for deposit)
USDC_AMOUNT
    Amount of USDC to deposit (default: ``3`` — enough for 2 USDC
    activation + 1 USDC margin)
SKIP_DEPOSIT
    Set to ``true`` to skip the deposit step and only run diagnostics

Example
-------
.. code-block:: shell

    source .local-test.env && NETWORK=testnet poetry run python scripts/hyperliquid/test-hypercore-escrow.py
"""

import logging
import os
import sys
import time
from decimal import Decimal

from eth_account import Account
from web3 import Web3

from eth_defi.hotwallet import HotWallet
from eth_defi.hyperliquid.api import fetch_spot_clearinghouse_state
from eth_defi.hyperliquid.core_writer import (
    CORE_DEPOSIT_WALLET,
    SPOT_DEX,
    get_core_deposit_wallet_contract,
)
from eth_defi.hyperliquid.evm_escrow import (
    DEFAULT_ACTIVATION_AMOUNT,
    is_account_activated,
)
from eth_defi.hyperliquid.session import (
    HYPERLIQUID_API_URL,
    HYPERLIQUID_TESTNET_API_URL,
    create_hyperliquid_session,
)
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.token import USDC_NATIVE_TOKEN, fetch_erc20_details

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# RPC endpoints
HYPEREVM_RPC = {
    "mainnet": "https://rpc.hyperliquid.xyz/evm",
    "testnet": "https://rpc.hyperliquid-testnet.xyz/evm",
}

API_URL = {
    "mainnet": HYPERLIQUID_API_URL,
    "testnet": HYPERLIQUID_TESTNET_API_URL,
}


def print_header(title: str):
    width = 60
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_row(label: str, value, indent: int = 2):
    print(f"{' ' * indent}{label:<35} {value}")


def main():
    network = os.environ.get("NETWORK", "mainnet").lower()
    assert network in ("mainnet", "testnet"), f"NETWORK must be mainnet or testnet, got: {network}"

    private_key = os.environ.get("PRIVATE_KEY") or os.environ.get("HYPERCORE_WRITER_TEST_PRIVATE_KEY")
    assert private_key, "Set PRIVATE_KEY or HYPERCORE_WRITER_TEST_PRIVATE_KEY"

    usdc_amount = Decimal(os.environ.get("USDC_AMOUNT", "3"))
    skip_deposit = os.environ.get("SKIP_DEPOSIT", "").lower() in ("true", "1", "yes")

    print_header(f"HyperCore EVM Escrow Diagnostic ({network})")

    # -------------------------------------------------------------------
    # Step 1: Connect to HyperEVM
    # -------------------------------------------------------------------
    print_header("Step 1: Connect to HyperEVM")
    rpc_url = HYPEREVM_RPC[network]
    web3 = create_multi_provider_web3(rpc_url)
    chain_id = web3.eth.chain_id
    print_row("RPC", rpc_url)
    print_row("Chain ID", chain_id)
    print_row("Latest block", f"{web3.eth.block_number:,}")

    # Create wallet
    account = Account.from_key(private_key)
    hot_wallet = HotWallet(account)
    hot_wallet.sync_nonce(web3)
    wallet_address = hot_wallet.address
    print_row("Wallet address", wallet_address)
    print_row("Nonce", hot_wallet.current_nonce)

    # -------------------------------------------------------------------
    # Step 2: Check balances (HYPE gas + USDC)
    # -------------------------------------------------------------------
    print_header("Step 2: Check balances")

    hype_balance = web3.eth.get_balance(wallet_address)
    hype_ether = Decimal(hype_balance) / Decimal(10**18)
    print_row("HYPE (gas)", f"{hype_ether:.6f} HYPE")
    if hype_balance == 0:
        print("  ** WARNING: No HYPE for gas! Deposit will fail. **")

    usdc_address = USDC_NATIVE_TOKEN[chain_id]
    usdc_token = fetch_erc20_details(web3, usdc_address)
    usdc_raw_balance = usdc_token.contract.functions.balanceOf(wallet_address).call()
    usdc_balance = Decimal(usdc_raw_balance) / Decimal(10**usdc_token.decimals)
    print_row("USDC address", usdc_address)
    print_row("USDC balance", f"{usdc_balance:.6f} USDC")

    if usdc_balance < usdc_amount and not skip_deposit:
        print(f"  ** WARNING: USDC balance ({usdc_balance}) < requested deposit ({usdc_amount}) **")

    # -------------------------------------------------------------------
    # Step 3: Check account activation (coreUserExists precompile)
    # -------------------------------------------------------------------
    print_header("Step 3: Check account activation (coreUserExists)")

    activated = is_account_activated(web3, wallet_address)
    print_row("Account activated", activated)

    # -------------------------------------------------------------------
    # Step 4: Check spot clearinghouse state (API)
    # -------------------------------------------------------------------
    print_header("Step 4: Spot clearinghouse state (API)")

    api_url = API_URL[network]
    session = create_hyperliquid_session(api_url=api_url)
    print_row("API URL", api_url)

    spot_state = fetch_spot_clearinghouse_state(session, wallet_address)

    if spot_state.balances:
        print("  Spot balances:")
        for b in spot_state.balances:
            print_row(f"{b.coin} (token {b.token})", f"total={b.total}  hold={b.hold}", indent=4)
    else:
        print("  No spot balances")

    if spot_state.evm_escrows:
        print("  EVM escrows (pending):")
        for e in spot_state.evm_escrows:
            print_row(f"{e.coin} (token {e.token})", f"total={e.total}", indent=4)
    else:
        print("  No EVM escrows")

    # -------------------------------------------------------------------
    # Step 5: Deposit (if not skipping)
    # -------------------------------------------------------------------
    if skip_deposit:
        print_header("Step 5: Deposit SKIPPED (SKIP_DEPOSIT=true)")
    elif usdc_balance < usdc_amount:
        print_header("Step 5: Deposit SKIPPED (insufficient USDC)")
    else:
        raw_amount = int(usdc_amount * Decimal(10**usdc_token.decimals))
        cdw_address = CORE_DEPOSIT_WALLET[chain_id]
        core_deposit_wallet = get_core_deposit_wallet_contract(web3, cdw_address)

        if not activated:
            print_header(f"Step 5a: Activate account (depositFor {usdc_amount} USDC)")
            print_row("CoreDepositWallet", cdw_address)

            # Approve USDC
            print("  Approving USDC...")
            approve_tx = usdc_token.contract.functions.approve(
                Web3.to_checksum_address(cdw_address),
                raw_amount,
            )
            signed_approve = hot_wallet.sign_transaction_with_new_nonce(
                approve_tx.build_transaction({
                    "from": wallet_address,
                    "gas": 200_000,
                    "gasPrice": web3.eth.gas_price,
                    "chainId": chain_id,
                }),
            )
            approve_hash = web3.eth.send_raw_transaction(signed_approve.rawTransaction)
            receipt = web3.eth.wait_for_transaction_receipt(approve_hash)
            print_row("Approve tx", approve_hash.hex())
            print_row("Approve status", "OK" if receipt["status"] == 1 else "FAILED")

            # depositFor (activates account)
            print("  Calling depositFor...")
            deposit_for_tx = core_deposit_wallet.functions.depositFor(
                Web3.to_checksum_address(wallet_address),
                raw_amount,
                SPOT_DEX,
            )
            signed_deposit = hot_wallet.sign_transaction_with_new_nonce(
                deposit_for_tx.build_transaction({
                    "from": wallet_address,
                    "gas": 200_000,
                    "gasPrice": web3.eth.gas_price,
                    "chainId": chain_id,
                }),
            )
            deposit_hash = web3.eth.send_raw_transaction(signed_deposit.rawTransaction)
            receipt = web3.eth.wait_for_transaction_receipt(deposit_hash)
            print_row("depositFor tx", deposit_hash.hex())
            print_row("depositFor status", "OK" if receipt["status"] == 1 else "FAILED")
        else:
            print_header(f"Step 5: Deposit {usdc_amount} USDC (account already activated)")
            print_row("CoreDepositWallet", cdw_address)

            # Approve USDC
            print("  Approving USDC...")
            approve_tx = usdc_token.contract.functions.approve(
                Web3.to_checksum_address(cdw_address),
                raw_amount,
            )
            signed_approve = hot_wallet.sign_transaction_with_new_nonce(
                approve_tx.build_transaction({
                    "from": wallet_address,
                    "gas": 200_000,
                    "gasPrice": web3.eth.gas_price,
                    "chainId": chain_id,
                }),
            )
            approve_hash = web3.eth.send_raw_transaction(signed_approve.rawTransaction)
            receipt = web3.eth.wait_for_transaction_receipt(approve_hash)
            print_row("Approve tx", approve_hash.hex())
            print_row("Approve status", "OK" if receipt["status"] == 1 else "FAILED")

            # deposit (regular deposit for activated account)
            print("  Calling deposit...")
            deposit_tx = core_deposit_wallet.functions.deposit(
                raw_amount,
                SPOT_DEX,
            )
            signed_deposit = hot_wallet.sign_transaction_with_new_nonce(
                deposit_tx.build_transaction({
                    "from": wallet_address,
                    "gas": 200_000,
                    "gasPrice": web3.eth.gas_price,
                    "chainId": chain_id,
                }),
            )
            deposit_hash = web3.eth.send_raw_transaction(signed_deposit.rawTransaction)
            receipt = web3.eth.wait_for_transaction_receipt(deposit_hash)
            print_row("deposit tx", deposit_hash.hex())
            print_row("deposit status", "OK" if receipt["status"] == 1 else "FAILED")

        # -------------------------------------------------------------------
        # Step 6: Poll coreUserExists + escrow state
        # -------------------------------------------------------------------
        print_header("Step 6: Poll activation + escrow clearing")

        timeout = 120.0
        poll_interval = 3.0
        start = time.time()
        iteration = 0

        while time.time() - start < timeout:
            iteration += 1
            elapsed = time.time() - start

            activated_now = is_account_activated(web3, wallet_address)
            spot_state_now = fetch_spot_clearinghouse_state(session, wallet_address)

            escrow_summary = (
                ", ".join(f"{e.coin}={e.total}" for e in spot_state_now.evm_escrows)
                if spot_state_now.evm_escrows
                else "none"
            )
            balance_summary = (
                ", ".join(f"{b.coin}={b.total}" for b in spot_state_now.balances)
                if spot_state_now.balances
                else "none"
            )

            print(
                f"  [{iteration:3d}] {elapsed:6.1f}s  "
                f"activated={activated_now}  "
                f"escrows=[{escrow_summary}]  "
                f"balances=[{balance_summary}]"
            )

            if activated_now and not spot_state_now.evm_escrows:
                print("  Escrow cleared successfully!")
                break

            time.sleep(poll_interval)
        else:
            print(f"  ** TIMEOUT after {timeout}s — escrow did not clear **")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print_header("Summary")

    # Re-check final state
    activated_final = is_account_activated(web3, wallet_address)
    spot_state_final = fetch_spot_clearinghouse_state(session, wallet_address)
    usdc_raw_final = usdc_token.contract.functions.balanceOf(wallet_address).call()
    usdc_balance_final = Decimal(usdc_raw_final) / Decimal(10**usdc_token.decimals)

    print_row("Network", network)
    print_row("Chain ID", chain_id)
    print_row("Wallet", wallet_address)
    print_row("Account activated", activated_final)
    print_row("USDC balance (EVM)", f"{usdc_balance_final:.6f}")

    if spot_state_final.balances:
        for b in spot_state_final.balances:
            print_row(f"Spot {b.coin}", f"total={b.total}  hold={b.hold}")

    if spot_state_final.evm_escrows:
        for e in spot_state_final.evm_escrows:
            print_row(f"Escrow {e.coin}", f"total={e.total}")
        print()
        print("  ** EVM escrow NOT clear — funds still pending **")
    else:
        print_row("EVM escrows", "clear")

    print()


if __name__ == "__main__":
    main()
