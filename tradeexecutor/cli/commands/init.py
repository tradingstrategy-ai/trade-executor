"""init command.

Quick local dev example:

.. code-block:: shell

    # Set up JSON_RPC_POLYGON
    source env/local-test.env

    # Set up hto wallet private key
    export PRIVATE_KEY=...

    poetry run trade-executor init \
        --id=vault-init-test \
        --vault-address=0x6E321256BE0ABd2726A234E8dBFc4d3caf255AE0


"""

from pathlib import Path
from typing import Optional

from typer import Option

from eth_defi.hotwallet import HotWallet

from .app import app
from ..bootstrap import prepare_executor_id, create_web3_config, create_sync_model, create_state_store
from ..log import setup_logging
from ...strategy.execution_model import AssetManagementMode
from . import shared_options


@app.command()
@shared_options.with_json_rpc_options(preset="default")
def init(
    id: str = shared_options.id,
    name: str = shared_options.name,

    strategy_file: Path = shared_options.strategy_file,
    state_file: Optional[Path] = shared_options.state_file,
    private_key: Optional[str] = shared_options.private_key,
    log_level: str = shared_options.log_level,

    asset_management_mode: AssetManagementMode = shared_options.asset_management_mode,
    vault_address: Optional[str] = shared_options.vault_address,
    vault_deployment_block_number: Optional[int] = shared_options.vault_deployment_block_number,
    vault_adapter_address: Optional[str] = shared_options.vault_adapter_address,

    rpc_kwargs: dict | None = None,

):
    """Initialise a strategy.

    A strategy initialisation will create its state file.
    It will also connect to a blockchain and check the vault smart contract is ready.

    Vault deployment is still handled separate.
    """

    global logger

    id = prepare_executor_id(id, strategy_file)

    logger = setup_logging(log_level)

    web3config = create_web3_config(
        gas_price_method=None,
        **rpc_kwargs,
    )

    assert web3config, "No RPC endpoints given. A working JSON-RPC connection is needed for check-wallet"

    # Set default chain, allowing multiple connections for multichain strategies
    if len(web3config.connections) == 1:
        web3config.choose_single_chain()
    else:
        default_chain_id = next(iter(web3config.connections.keys()))
        web3config.set_default_chain(default_chain_id)

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
        vault_adapter_address=vault_adapter_address,
        init=True,
    )

    logger.info("RPC details")

    # Check the chain is online
    logger.info(f"  Chain id is {web3.eth.chain_id:,}")
    logger.info(f"  Latest block is {web3.eth.block_number:,}")

    # Check balances
    logger.info("Balance details")
    logger.info("  Hot wallet is %s", hot_wallet.address)

    vault_address =  sync_model.get_key_address()
    start_block = None
    if vault_address:
        logger.info("  Vault is %s", vault_address)
        if vault_deployment_block_number:
            start_block = vault_deployment_block_number
            logger.info("  Vault deployment block number is %d", start_block)

    if not state_file:
        state_file = f"state/{id}.json"

    store = create_state_store(Path(state_file))
    assert store.is_pristine(), f"State file already exists: {state_file}"

    state = store.create(name)

    logger.info("Syncing initial strategy chain state.")
    logger.info("For Enzyme vaults this may take a long time as the sync will go through all the blocks in the chain.")
    if not start_block:
        logger.warning("To speed up process use --vault_deployment_block_number hint as a command line argument.")
    logger.info(f"Vault deployment block number hint is {start_block or 0:,}.")
    sync_model.sync_initial(state, start_block=start_block)

    store.sync(state)

    web3config.close()

    logger.info("All done: State deployment info is %s", state.sync.deployment)
