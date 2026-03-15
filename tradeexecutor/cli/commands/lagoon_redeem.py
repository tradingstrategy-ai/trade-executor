"""lagoon-redeem command.

Redeems all vault shares held by the asset manager from a Lagoon vault:

1. Checks the asset manager wallet holds vault shares
2. Approves and requests redemption of all shares (Phase 1 - ERC-7540)
3. Posts new valuation and settles the vault (Phase 2)
4. Finalises the redemption to claim denomination tokens (Phase 3)
5. Updates the state file with the new reserve balance
6. Reports final balances
"""

import logging
from pathlib import Path
from typing import Optional

from eth_defi.erc_4626.vault_protocol.lagoon.testing import redeem_vault_shares
from eth_defi.trace import assert_transaction_success_with_explanation

from tradeexecutor.cli.bootstrap import prepare_executor_id
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
def lagoon_redeem(
    id: str = shared_options.id,

    strategy_file: Path = shared_options.strategy_file,
    state_file: Optional[Path] = shared_options.state_file,

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

    simulate: bool = shared_options.simulate,
):
    """Redeem all vault shares from a Lagoon vault for the asset manager.

    - Performs the full ERC-7540 three-phase async redemption
    - Approves and requests redemption of all shares (Phase 1)
    - Posts new valuation and settles the vault (Phase 2)
    - Finalises the redemption to claim denomination tokens back (Phase 3)
    - Updates the state file reserves to reflect the new Safe balance
    """
    id = prepare_executor_id(id, strategy_file)

    logger = setup_logging(log_level=log_level)

    assert private_key, "PRIVATE_KEY is required"
    assert vault_address, "VAULT_ADDRESS is required"
    assert vault_adapter_address, "VAULT_ADAPTER_ADDRESS is required"

    context = create_lagoon_command_context(
        private_key=private_key,
        vault_address=vault_address,
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

    # Required for settle_via_trading_strategy_module()
    vault.trading_strategy_module_address = vault_adapter_address

    denomination_token = vault.denomination_token
    share_token = vault.share_token

    # Pre-flight checks
    eth_balance = web3.eth.get_balance(hot_wallet.address)
    eth_human = eth_balance / 10**18
    logger.info("Asset manager: %s", hot_wallet.address)
    logger.info("  ETH balance: %.6f", eth_human)
    assert eth_human >= 0.001, f"Asset manager has {eth_human:.6f} ETH, need at least 0.001 for gas"

    share_balance = share_token.fetch_balance_of(hot_wallet.address)
    logger.info("  Share balance: %s %s", share_balance, share_token.symbol)
    assert share_balance > 0, f"Asset manager has no vault shares to redeem"

    usdc_before = denomination_token.fetch_balance_of(hot_wallet.address)
    logger.info("  %s balance before: %s", denomination_token.symbol, usdc_before)

    # Phase 1: Request redemption (approve shares + requestRedeem)
    logger.info("Phase 1: Requesting redemption of %s %s shares", share_balance, share_token.symbol)
    redeem_vault_shares(
        web3=web3,
        vault_address=vault_address,
        redeemer=hot_wallet.address,
        hot_wallet=hot_wallet,
    )

    # Phase 2: Settle (post valuation + settleDeposit which also settles redeems)
    logger.info("Phase 2: Settling vault")
    nav = vault.fetch_nav()
    logger.info("  Current NAV: %s %s", nav, denomination_token.symbol)

    hot_wallet.sync_nonce(web3)
    tx_hash = hot_wallet.transact_and_broadcast_with_contract(
        vault.post_new_valuation(nav), gas_limit=1_000_000,
    )
    assert_transaction_success_with_explanation(web3, tx_hash)

    hot_wallet.sync_nonce(web3)
    tx_hash = hot_wallet.transact_and_broadcast_with_contract(
        vault.settle_via_trading_strategy_module(nav), gas_limit=1_000_000,
    )
    assert_transaction_success_with_explanation(web3, tx_hash)

    # Phase 3: Claim redeemed denomination tokens
    logger.info("Phase 3: Finalising redemption")
    hot_wallet.sync_nonce(web3)
    tx_hash = hot_wallet.transact_and_broadcast_with_contract(
        vault.finalise_redeem(hot_wallet.address), gas_limit=1_000_000,
    )
    assert_transaction_success_with_explanation(web3, tx_hash)

    # Report on-chain balances after redemption
    safe_balance = denomination_token.fetch_balance_of(vault.safe_address)
    final_share_balance = share_token.fetch_balance_of(hot_wallet.address)
    usdc_after = denomination_token.fetch_balance_of(hot_wallet.address)
    redeemed_amount = usdc_after - usdc_before

    logger.info("Redemption complete")
    logger.info("  Vault Safe %s balance: %s", denomination_token.symbol, safe_balance)
    logger.info("  Redeemer share balance: %s %s", final_share_balance, share_token.symbol)
    logger.info("  Redeemer %s balance: %s (redeemed %s)", denomination_token.symbol, usdc_after, redeemed_amount)

    # Update the state file with the new reserve balance
    if simulate:
        logger.info("Simulation mode — skipping state file update")
        logger.info("  Vault Safe %s balance: %s", denomination_token.symbol, safe_balance)
        logger.info("  Redeemer shares: %s %s", final_share_balance, share_token.symbol)
    else:
        state_path, store = ensure_state_store_exists(id, state_file)
        state, reserve_position = sync_reserve_balance_to_state(store, denomination_token, safe_balance)

        reserve_value = reserve_position.get_value()
        reserve_symbol = reserve_position.asset.token_symbol

        logger.info("State file updated: %s", state_path)
        logger.info("  Reserve balance: %s %s", reserve_value, reserve_symbol)
        logger.info("  Vault Safe %s balance: %s", denomination_token.symbol, safe_balance)
        logger.info("  Redeemer shares: %s %s", final_share_balance, share_token.symbol)

    logger.info("All ok")
