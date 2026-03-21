"""lagoon-first-deposit command.

Performs the initial deposit into a Lagoon vault:

1. Checks the asset manager wallet has sufficient denomination token (e.g. USDC)
2. Approves and deposits into the vault (buys shares)
3. Settles the vault on-chain (updates NAV, processes deposit queue)
4. Updates the state file with the new reserves
5. Reports final vault balance and share balance
"""

import logging
from decimal import Decimal
from pathlib import Path
from typing import Optional

from eth_defi.erc_4626.vault_protocol.lagoon.testing import fund_lagoon_vault
from typer import Option

from tradeexecutor.cli.bootstrap import prepare_cache_and_token_cache, prepare_executor_id
from tradeexecutor.cli.commands import shared_options
from tradeexecutor.cli.commands.app import app
from tradeexecutor.cli.commands.lagoon_utils import (
    create_lagoon_command_context,
    ensure_state_store_exists,
    sync_reserve_balance_to_state,
)
from tradeexecutor.cli.log import setup_logging

logger = logging.getLogger(__name__)


@app.command()
def lagoon_first_deposit(
    id: str = shared_options.id,

    strategy_file: Path = shared_options.strategy_file,
    state_file: Optional[Path] = shared_options.state_file,
    cache_path: Optional[Path] = shared_options.cache_path,

    log_level: str = shared_options.log_level,
    json_rpc_binance: Optional[str] = shared_options.json_rpc_binance,
    json_rpc_polygon: Optional[str] = shared_options.json_rpc_polygon,
    json_rpc_avalanche: Optional[str] = shared_options.json_rpc_avalanche,
    json_rpc_ethereum: Optional[str] = shared_options.json_rpc_ethereum,
    json_rpc_base: Optional[str] = shared_options.json_rpc_base,
    json_rpc_arbitrum: Optional[str] = shared_options.json_rpc_arbitrum,
    json_rpc_anvil: Optional[str] = shared_options.json_rpc_anvil,
    json_rpc_arbitrum_sepolia: Optional[str] = shared_options.json_rpc_arbitrum_sepolia,
    json_rpc_base_sepolia: Optional[str] = shared_options.json_rpc_base_sepolia,
    json_rpc_hyperliquid: Optional[str] = shared_options.json_rpc_hyperliquid,
    json_rpc_hyperliquid_testnet: Optional[str] = shared_options.json_rpc_hyperliquid_testnet,
    private_key: str = shared_options.private_key,

    vault_address: Optional[str] = shared_options.vault_address,
    vault_adapter_address: Optional[str] = shared_options.vault_adapter_address,

    unit_testing: bool = shared_options.unit_testing,
    simulate: bool = shared_options.simulate,

    deposit_amount: float = Option(..., envvar="DEPOSIT_AMOUNT", help="Amount of denomination token (e.g. USDC) to deposit into the vault."),
):
    """Make the first deposit into a Lagoon vault.

    - Checks the asset manager wallet has sufficient denomination token
    - Approves and deposits into the vault (buys shares)
    - Settles the vault and updates the state file with the new reserves
    - Reports final vault balance and depositor share balance
    """
    id = prepare_executor_id(id, strategy_file)

    logger = setup_logging(log_level=log_level)

    assert private_key, "PRIVATE_KEY is required"
    assert vault_address, "VAULT_ADDRESS is required"
    assert vault_adapter_address, "VAULT_ADAPTER_ADDRESS is required"
    assert deposit_amount > 0, f"DEPOSIT_AMOUNT must be positive, got {deposit_amount}"

    cache_path, token_cache = prepare_cache_and_token_cache(
        id,
        cache_path,
        unit_testing=unit_testing,
    )

    context = create_lagoon_command_context(
        private_key=private_key,
        vault_address=vault_address,
        token_cache=token_cache,
        simulate=simulate,
        json_rpc_binance=json_rpc_binance,
        json_rpc_polygon=json_rpc_polygon,
        json_rpc_avalanche=json_rpc_avalanche,
        json_rpc_ethereum=json_rpc_ethereum,
        json_rpc_base=json_rpc_base,
        json_rpc_anvil=json_rpc_anvil,
        json_rpc_arbitrum=json_rpc_arbitrum,
        json_rpc_arbitrum_sepolia=json_rpc_arbitrum_sepolia,
        json_rpc_base_sepolia=json_rpc_base_sepolia,
        json_rpc_hyperliquid=json_rpc_hyperliquid,
        json_rpc_hyperliquid_testnet=json_rpc_hyperliquid_testnet,
    )
    web3config = context.web3config
    web3 = context.web3
    hot_wallet = context.hot_wallet
    vault = context.vault

    denomination_token = vault.denomination_token
    amount = Decimal(str(deposit_amount))

    # Pre-flight checks — verify state file exists before depositing
    if not simulate:
        state_path, store = ensure_state_store_exists(id, state_file)
    else:
        state_path, store = None, None

    eth_balance = web3.eth.get_balance(hot_wallet.address)
    eth_human = eth_balance / 10**18
    logger.info("Asset manager: %s", hot_wallet.address)
    logger.info("  ETH balance: %.6f", eth_human)
    assert eth_human >= 0.001, f"Asset manager has {eth_human:.6f} ETH, need at least 0.001 for gas"

    token_balance = denomination_token.fetch_balance_of(hot_wallet.address)
    logger.info("  %s balance: %s", denomination_token.symbol, token_balance)
    assert token_balance >= amount, (
        f"Asset manager has {token_balance} {denomination_token.symbol} "
        f"but needs {amount} for the deposit"
    )

    # Perform the deposit (approve, request deposit, settle, claim shares)
    logger.info("Depositing %s %s into vault %s", amount, denomination_token.symbol, vault_address)

    fund_lagoon_vault(
        web3=web3,
        vault_address=vault_address,
        asset_manager=hot_wallet.address,
        test_account_with_balance=hot_wallet.address,
        trading_strategy_module_address=vault_adapter_address,
        amount=amount,
        hot_wallet=hot_wallet,
        token_cache=token_cache,
    )

    # Report on-chain balances after deposit
    safe_balance = denomination_token.fetch_balance_of(vault.safe_address)
    share_balance = vault.share_token.fetch_balance_of(hot_wallet.address)
    logger.info("Deposit complete")
    logger.info("  Vault Safe %s balance: %s", denomination_token.symbol, safe_balance)
    logger.info("  Depositor share balance: %s %s", share_balance, vault.share_token.symbol)

    # Update the state file with the new reserve balance.
    # fund_lagoon_vault() already settled the vault on-chain,
    # so we directly update the state's reserve position from the
    # on-chain Safe balance rather than running the full settle flow
    # (which would require constructing the full trading universe).
    if simulate:
        logger.info("Simulation mode — skipping state file update")
        logger.info("  Vault Safe %s balance: %s", denomination_token.symbol, safe_balance)
        logger.info("  Depositor shares: %s %s", share_balance, vault.share_token.symbol)
    else:
        state, reserve_position = sync_reserve_balance_to_state(store, denomination_token, safe_balance)

        # Final state check
        reserve_value = reserve_position.get_value()
        reserve_symbol = reserve_position.asset.token_symbol

        logger.info("State file updated: %s", state_path)
        logger.info("  Reserve balance: %s %s", reserve_value, reserve_symbol)
        logger.info("  Vault Safe %s balance: %s", denomination_token.symbol, safe_balance)
        logger.info("  Depositor shares: %s %s", share_balance, vault.share_token.symbol)

        assert reserve_value > 0, f"Reserve balance is 0 after deposit — something went wrong"

    logger.info("All ok")
