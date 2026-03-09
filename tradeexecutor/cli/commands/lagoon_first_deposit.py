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

from eth_defi.compat import native_datetime_utc_now
from eth_defi.erc_4626.classification import create_vault_instance
from eth_defi.erc_4626.core import ERC4626Feature
from eth_defi.erc_4626.vault_protocol.lagoon.testing import fund_lagoon_vault
from eth_defi.erc_4626.vault_protocol.lagoon.vault import LagoonVault
from eth_defi.hotwallet import HotWallet
from typer import Option

from tradeexecutor.cli.bootstrap import (create_state_store,
                                         create_web3_config,
                                         prepare_executor_id)
from tradeexecutor.cli.commands import shared_options
from tradeexecutor.cli.commands.app import app
from tradeexecutor.cli.log import setup_logging
from tradeexecutor.ethereum.token import translate_token_details

logger = logging.getLogger(__name__)


@app.command()
def lagoon_first_deposit(
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
    private_key: str = shared_options.private_key,

    vault_address: Optional[str] = shared_options.vault_address,
    vault_adapter_address: Optional[str] = shared_options.vault_adapter_address,

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

    web3config = create_web3_config(
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
        simulate=simulate,
    )

    if not web3config.has_any_connection():
        raise RuntimeError("first-deposit requires that you pass a JSON-RPC connection to one of the networks")

    if len(web3config.connections) > 1:
        web3config.choose_single_chain(default_chain_id=next(iter(web3config.connections.keys())))
    else:
        web3config.choose_single_chain()

    web3 = web3config.get_default()

    # Set up hot wallet and check balances
    hot_wallet = HotWallet.from_private_key(private_key)
    hot_wallet.sync_nonce(web3)

    # Load vault to get denomination token details
    vault = create_vault_instance(
        web3,
        vault_address,
        features={ERC4626Feature.lagoon_like},
        default_block_identifier="latest",
        require_denomination_token=True,
    )
    assert isinstance(vault, LagoonVault), f"Not a Lagoon vault: {vault}"

    denomination_token = vault.denomination_token
    amount = Decimal(str(deposit_amount))

    # Pre-flight checks — verify state file exists before depositing
    if not simulate:
        if not state_file:
            state_file = f"state/{id}.json"
        store = create_state_store(Path(state_file))
        assert not store.is_pristine(), (
            f"State file does not exist: {state_file}. "
            f"Run 'init' first to create the state file before depositing."
        )

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
        # store was already created and verified in pre-flight checks above
        state = store.load()

        ts = native_datetime_utc_now()
        if len(state.portfolio.reserves) == 0:
            reserve_asset = translate_token_details(denomination_token)
            state.portfolio.initialise_reserves(reserve_asset, reserve_token_price=1.0)
        reserve_position = state.portfolio.get_default_reserve_position()
        reserve_position.quantity = safe_balance
        reserve_position.reserve_token_price = 1.0
        reserve_position.last_pricing_at = ts
        reserve_position.last_sync_at = ts

        store.sync(state)

        # Final state check
        reserve_value = reserve_position.get_value()
        reserve_symbol = reserve_position.asset.token_symbol

        logger.info("State file updated: %s", state_file)
        logger.info("  Reserve balance: %s %s", reserve_value, reserve_symbol)
        logger.info("  Vault Safe %s balance: %s", denomination_token.symbol, safe_balance)
        logger.info("  Depositor shares: %s %s", share_balance, vault.share_token.symbol)

        assert reserve_value > 0, f"Reserve balance is 0 after deposit — something went wrong"

    logger.info("All ok")
