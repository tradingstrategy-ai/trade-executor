"""Deposit deployer USDC into an existing Lagoon vault and settle it.

This operator utility uses the normal Lagoon ERC-7540 subscription lifecycle:

1. Approve and request the deployer's USDC deposit.
2. Run ``trade-executor lagoon-settle`` in-process to post NAV, settle the
   investor queue and update executor state.
3. Claim the deployer's Lagoon shares.

The script is intended for a stopped executor and inherits the same environment
as the executor container. It never transfers USDC directly to the Safe.

Example::

    poetry run python scripts/lagoon/deposit-and-settle.py 20

Required environment variables are ``JSON_RPC_ETHEREUM``, ``PRIVATE_KEY``,
``VAULT_ADDRESS``, ``VAULT_ADAPTER_ADDRESS``, ``EXECUTOR_ID``,
``STRATEGY_FILE``, ``STATE_FILE`` and ``ASSET_MANAGEMENT_MODE=lagoon``.
"""

import argparse
import logging
import os
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

from eth_defi.erc_4626.vault_protocol.lagoon.vault import LagoonVault
from eth_defi.hotwallet import HotWallet
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.provider.receipt import wait_for_transaction_receipt_robust
from eth_defi.utils import setup_console_logging
from eth_defi.vault.base import VaultSpec
from web3 import Web3
from web3.contract.contract import ContractFunction

from tradeexecutor.cli.main import app
from tradeexecutor.cli.commands.lagoon_utils import get_lagoon_reserve_baseline
from tradeexecutor.state.state import State


logger = logging.getLogger(__name__)

#: Only Ethereum mainnet vaults are supported by this operator script.
ETHEREUM_CHAIN_ID = 1
#: Lagoon ERC-7540 deposit-request identifier.
LAGOON_DEPOSIT_REQUEST_ID = 0
#: Connect timeout for configured Ethereum JSON-RPC providers.
RPC_CONNECT_TIMEOUT_SECONDS = 3.0
#: Read timeout for configured Ethereum JSON-RPC providers.
RPC_READ_TIMEOUT_SECONDS = 180.0
#: Maximum time to wait for Lagoon settlement to make a deposit claimable.
CLAIMABLE_WAIT_SECONDS = 60
#: Interval between claimable-share reads after Lagoon settlement.
CLAIMABLE_POLL_SECONDS = 5
#: Gas limit for simple Lagoon approval, request and claim transactions.
LAGOON_DEPOSIT_GAS_LIMIT = 1_000_000


class LagoonDepositError(RuntimeError):
    """Raised when a Lagoon deposit cannot be completed safely."""


def _require_environment(name: str) -> str:
    """Read required configuration without printing its value."""
    value = os.environ.get(name)
    if not value:
        raise LagoonDepositError(f"{name} is required")
    return value


def _parse_amount(value: str) -> Decimal:
    """Parse one positive finite denomination-token amount."""
    try:
        amount = Decimal(value)
    except InvalidOperation as error:
        raise LagoonDepositError("Deposit amount must be a decimal number") from error
    if not amount.is_finite() or amount <= 0:
        raise LagoonDepositError("Deposit amount must be positive and finite")
    return amount


def _broadcast_and_wait(
    web3: Web3,
    hot_wallet: HotWallet,
    function: ContractFunction,
    description: str,
) -> None:
    """Broadcast one deployer transaction and wait for robust RPC visibility."""
    hot_wallet.sync_nonce(web3)
    logger.info("Broadcasting: %s", description)
    transaction_hash = hot_wallet.transact_and_broadcast_with_contract(
        function,
        gas_limit=LAGOON_DEPOSIT_GAS_LIMIT,
    )
    receipt = wait_for_transaction_receipt_robust(web3, transaction_hash)
    if receipt["status"] != 1:
        raise LagoonDepositError(f"Transaction failed: {description}")
    logger.info("Confirmed transaction: %s", Web3.to_hex(transaction_hash))


def _run_lagoon_settle() -> None:
    """Run the supported Typer vault sync using the current environment."""
    logger.info("Running: trade-executor lagoon-settle")
    try:
        app(["lagoon-settle"], standalone_mode=False)
    except SystemExit as error:
        if error.code not in (None, 0):
            raise


def _wait_for_claimable_deposit(vault: LagoonVault, depositor: str) -> int:
    """Wait until the settled deposit is claimable."""
    deadline = time.monotonic() + CLAIMABLE_WAIT_SECONDS
    while True:
        claimable_raw = vault.vault_contract.functions.maxDeposit(depositor).call()
        if claimable_raw > 0:
            return claimable_raw
        if time.monotonic() >= deadline:
            raise LagoonDepositError(
                "Lagoon deposit is not claimable after vault sync; inspect settlement logs "
                "and rerun this script without submitting another deposit",
            )
        time.sleep(CLAIMABLE_POLL_SECONDS)


def deposit_and_settle(amount: Decimal) -> None:
    """Move deployer funds through Lagoon and verify Safe/state accounting."""
    if os.environ.get("ASSET_MANAGEMENT_MODE") != "lagoon":
        raise LagoonDepositError("ASSET_MANAGEMENT_MODE must be lagoon")

    rpc_url = _require_environment("JSON_RPC_ETHEREUM")
    private_key = _require_environment("PRIVATE_KEY")
    vault_address = _require_environment("VAULT_ADDRESS")
    module_address = _require_environment("VAULT_ADAPTER_ADDRESS")
    state_path = Path(_require_environment("STATE_FILE"))

    web3 = create_multi_provider_web3(
        rpc_url,
        default_http_timeout=(RPC_CONNECT_TIMEOUT_SECONDS, RPC_READ_TIMEOUT_SECONDS),
    )
    if web3.eth.chain_id != ETHEREUM_CHAIN_ID:
        raise LagoonDepositError("This script requires an Ethereum mainnet RPC")

    hot_wallet = HotWallet.from_private_key(private_key)
    vault = LagoonVault(
        web3,
        VaultSpec(web3.eth.chain_id, vault_address),
        trading_strategy_module_address=module_address,
        default_block_identifier="latest",
        require_denomination_token=True,
    )
    denomination_token = vault.denomination_token
    raw_amount = denomination_token.convert_to_raw(amount)
    if denomination_token.convert_to_decimals(raw_amount) != amount:
        raise LagoonDepositError(
            f"Deposit amount has more than {denomination_token.decimals} decimal places",
        )

    state = State.read_json_file(state_path)
    state.check_if_clean()

    deployer = hot_wallet.address
    deployer_balance_before = denomination_token.fetch_balance_of(deployer)
    safe_balance_before = denomination_token.fetch_balance_of(vault.safe_address)
    share_balance_before = vault.share_token.fetch_balance_of(deployer)

    try:
        reserve = get_lagoon_reserve_baseline(state, denomination_token, safe_balance_before)
    except ValueError as error:
        raise LagoonDepositError(str(error)) from error
    if reserve is None:
        logger.info("No executor reserve yet; accepting the zero-Safe activation baseline")

    pending_raw = vault.vault_contract.functions.pendingDepositRequest(
        LAGOON_DEPOSIT_REQUEST_ID,
        deployer,
    ).call()
    claimable_raw = vault.vault_contract.functions.maxDeposit(deployer).call()

    logger.info("Deployer: %s", deployer)
    logger.info("Vault: %s", vault.address)
    logger.info("Safe: %s", vault.safe_address)
    logger.info("Requested deposit: %s %s", amount, denomination_token.symbol)
    logger.info("Deployer balance: %s %s", deployer_balance_before, denomination_token.symbol)
    logger.info("Safe balance: %s %s", safe_balance_before, denomination_token.symbol)

    requested_now = False
    settled_this_run = claimable_raw == 0
    if claimable_raw > 0:
        if claimable_raw != raw_amount:
            claimable = denomination_token.convert_to_decimals(claimable_raw)
            raise LagoonDepositError(
                f"Deployer already has a {claimable} {denomination_token.symbol} "
                f"claimable Lagoon deposit; requested amount was {amount}",
            )
        logger.info("Resuming an already settled %s %s deposit", amount, denomination_token.symbol)
    elif pending_raw > 0:
        if pending_raw != raw_amount:
            pending = denomination_token.convert_to_decimals(pending_raw)
            raise LagoonDepositError(
                f"Deployer already has a {pending} {denomination_token.symbol} "
                f"pending Lagoon deposit; requested amount was {amount}",
            )
        logger.info("Resuming an existing %s %s deposit request", amount, denomination_token.symbol)
    else:
        if deployer_balance_before < amount:
            raise LagoonDepositError(
                f"Deployer has {deployer_balance_before} {denomination_token.symbol}, "
                f"but the requested deposit is {amount}",
            )
        _broadcast_and_wait(
            web3,
            hot_wallet,
            denomination_token.approve(vault.address, amount),
            f"Approve {amount} {denomination_token.symbol} for Lagoon",
        )
        _broadcast_and_wait(
            web3,
            hot_wallet,
            vault.request_deposit(deployer, raw_amount),
            f"Request {amount} {denomination_token.symbol} Lagoon deposit",
        )
        requested_now = True

    if settled_this_run:
        _run_lagoon_settle()
        claimable_raw = _wait_for_claimable_deposit(vault, deployer)

    if claimable_raw != raw_amount:
        claimable = denomination_token.convert_to_decimals(claimable_raw)
        raise LagoonDepositError(
            f"Claimable Lagoon deposit is {claimable} {denomination_token.symbol}, expected {amount}",
        )
    _broadcast_and_wait(
        web3,
        hot_wallet,
        vault.finalise_deposit(deployer, raw_amount=claimable_raw),
        "Claim Lagoon vault shares",
    )

    deployer_balance_after = denomination_token.fetch_balance_of(deployer)
    safe_balance_after = denomination_token.fetch_balance_of(vault.safe_address)
    share_balance_after = vault.share_token.fetch_balance_of(deployer)
    pending_after_raw = vault.vault_contract.functions.pendingDepositRequest(
        LAGOON_DEPOSIT_REQUEST_ID,
        deployer,
    ).call()
    claimable_after_raw = vault.vault_contract.functions.maxDeposit(deployer).call()
    final_state = State.read_json_file(state_path)
    final_reserve = final_state.portfolio.get_default_reserve_position()

    if requested_now and deployer_balance_after != deployer_balance_before - amount:
        raise LagoonDepositError("Deployer USDC debit does not match the requested deposit")
    if settled_this_run and safe_balance_after < safe_balance_before + amount:
        raise LagoonDepositError("Lagoon Safe did not receive the requested deposit")
    if share_balance_after <= share_balance_before:
        raise LagoonDepositError("Deployer Lagoon share balance did not increase")
    if pending_after_raw != 0 or claimable_after_raw != 0:
        raise LagoonDepositError("Deployer Lagoon deposit was not fully claimed")
    if final_reserve.quantity != safe_balance_after:
        raise LagoonDepositError(
            f"Executor reserve is {final_reserve.quantity} {denomination_token.symbol}, "
            f"but the Safe holds {safe_balance_after}",
        )
    if final_state.portfolio.get_net_asset_value() < 0:
        raise LagoonDepositError("Executor state has negative net asset value")

    logger.info("Deposit complete")
    logger.info("Deployer balance: %s %s", deployer_balance_after, denomination_token.symbol)
    logger.info("Safe and executor reserve: %s %s", safe_balance_after, denomination_token.symbol)
    logger.info("Deployer shares: %s %s", share_balance_after, vault.share_token.symbol)
    logger.info("All ok")


def main() -> None:
    """Parse the requested amount and run the Lagoon deposit lifecycle."""
    parser = argparse.ArgumentParser(
        description="Deposit deployer funds into Lagoon, settle the vault and claim shares.",
    )
    parser.add_argument(
        "amount",
        help="Human-readable denomination-token amount, for example 20 for 20 USDC.",
    )
    arguments = parser.parse_args()
    setup_console_logging("INFO")
    deposit_and_settle(_parse_amount(arguments.amount))


if __name__ == "__main__":
    main()
