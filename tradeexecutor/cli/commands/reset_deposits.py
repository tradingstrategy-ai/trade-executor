
"""Reset account balances.

"""
from pathlib import Path
from typing import Optional

from eth_defi.hotwallet import HotWallet

from .app import app
from ..backup import backup_state
from ..bootstrap import prepare_executor_id, create_web3_config, create_sync_model
from ..log import setup_logging
from ...strategy.execution_model import AssetManagementMode
from . import shared_options
from tradeexecutor.strategy.account_correction import correct_accounts as _correct_accounts


@app.command()
def reset_deposits(
    id: str = shared_options.id,
    name: str = shared_options.name,

    strategy_file: Path = shared_options.strategy_file,
    state_file: Optional[Path] = shared_options.state_file,
    private_key: Optional[str] = shared_options.private_key,
    log_level: str = shared_options.log_level,

    asset_management_mode: AssetManagementMode = shared_options.asset_management_mode,
    vault_address: Optional[str] = shared_options.vault_address,
    vault_deployment_block_number: Optional[int] = shared_options.vault_deployment_block_number,

    json_rpc_binance: Optional[str] = shared_options.json_rpc_binance,
    json_rpc_polygon: Optional[str] = shared_options.json_rpc_polygon,
    json_rpc_avalanche: Optional[str] = shared_options.json_rpc_avalanche,
    json_rpc_ethereum: Optional[str] = shared_options.json_rpc_ethereum,
    json_rpc_base: Optional[str] = shared_options.json_rpc_base,
    json_rpc_arbitrum: Optional[str] = shared_options.json_rpc_arbitrum,
    json_rpc_anvil: Optional[str] = shared_options.json_rpc_anvil,

):
    """Reset account balances.

    Resets account balances from on-chain data. This includes reserve and spot market balances.
    Does not lose trade history, but lose any unprocessed deposit and redemption events.
    """

    id = prepare_executor_id(id, strategy_file)

    logger = setup_logging(log_level)

    logger.warning("This command is deprecated - please use correct-accounts")

    web3config = create_web3_config(
        gas_price_method=None,
        json_rpc_binance=json_rpc_binance,
        json_rpc_polygon=json_rpc_polygon,
        json_rpc_avalanche=json_rpc_avalanche,
        json_rpc_ethereum=json_rpc_ethereum, json_rpc_base=json_rpc_base, 
        json_rpc_arbitrum=json_rpc_arbitrum,
        json_rpc_anvil=json_rpc_anvil,
    )

    assert web3config, "No RPC endpoints given. A working JSON-RPC connection is needed for check-wallet"

    # Check that we are connected to the chain strategy assumes
    web3config.choose_single_chain()

    if private_key is not None:
        hot_wallet = HotWallet.from_private_key(private_key)
    else:
        hot_wallet = None

    web3 = web3config.get_default()

    sync_model = create_sync_model(
        asset_management_mode,
        web3,
        hot_wallet,
        vault_address,
    )

    logger.info("RPC details")

    # Check the chain is online
    logger.info(f"  Chain id is {web3.eth.chain_id:,}")
    logger.info(f"  Latest block is {web3.eth.block_number:,}")

    # Check balances
    logger.info("Balance details")
    logger.info("  Hot wallet is %s", hot_wallet.address)

    vault_address =  sync_model.get_vault_address()
    start_block = None
    if vault_address:
        logger.info("  Vault is %s", vault_address)
        if vault_deployment_block_number:
            start_block = vault_deployment_block_number
            logger.info("  Vault deployment block number is %d", start_block)

    logger.info("Syncing initial strategy chain state: %s", name)
    logger.info(f"Vault deployment block number hint is {start_block or 0:,}.")

    store = backup_state(id, state_file)
    state = store.load()

    # Skip all deposit/redemption events between last scanend block and current block
    sync_model.reset_deposits(state)
    logger.info(f"Deposits and redemptions skipped until block {state.sync.treasury.last_block_scanned:,}")

    store.sync(state)

    logger.info("All done: State deployment info is %s", state.sync.deployment)
    logger.info("Please run correct-accounts next")
