"""iniy command"""

import datetime
from pathlib import Path
from typing import Optional

import typer
from tradingstrategy.chain import ChainId
from web3 import Web3

from eth_defi.balances import fetch_erc20_balances_by_token_list
from eth_defi.gas import GasPriceMethod
from eth_defi.hotwallet import HotWallet
from eth_defi.token import fetch_erc20_details
from tradingstrategy.client import Client
from .app import app
from .shared_options import private_key_option, strategy_file_option
from ..bootstrap import prepare_executor_id, prepare_cache, create_web3_config, create_trade_execution_model, create_sync_model, create_state_store
from ..log import setup_logging
from ...strategy.approval import UncheckedApprovalModel
from ...strategy.bootstrap import make_factory_from_strategy_mod
from ...strategy.description import StrategyExecutionDescription
from ...strategy.execution_context import ExecutionContext, ExecutionMode
from ...strategy.execution_model import AssetManagementMode
from ...strategy.run_state import RunState
from ...strategy.strategy_module import read_strategy_module
from ...strategy.trading_strategy_universe import TradingStrategyUniverseModel
from ...strategy.universe_model import UniverseOptions
from ...utils.fullname import get_object_full_name
from ...utils.timer import timed_task
from . import shared_options


@app.command()
def init(
    id: str = shared_options.id,
    strategy_file: Path = shared_options.strategy_file,
    state_file: Optional[Path] = shared_options.state_file,
    private_key: Optional[str] = shared_options.private_key,
    log_level: str = shared_options.log_level,

    chain: ChainId = shared_options.chain,
    asset_management_mode: AssetManagementMode = shared_options.asset_management_mode,
    vault_address: Optional[str] = shared_options.vault_address,

    json_rpc_binance: Optional[str] = shared_options.json_rpc_binance,
    json_rpc_polygon: Optional[str] = shared_options.json_rpc_polygon,
    json_rpc_avalanche: Optional[str] = shared_options.json_rpc_avalanche,
    json_rpc_ethereum: Optional[str] = shared_options.json_rpc_ethereum,
    json_rpc_arbitrum: Optional[str] = shared_options.json_rpc_arbitrum,

):
    """Initialise a strategy.

    A strategy initialisation will create its state file.
    It will also connect to a blockchain and check the vault smart contract is ready.

    Vault deployment is still handled separate.
    """

    # To run this from command line with .env file you can do
    # set -o allexport ; source ~/pancake-eth-usd-sma-final.env ; set +o allexport ;  trade-executor check-wallet

    global logger

    id = prepare_executor_id(id, strategy_file)

    logger = setup_logging(log_level)

    web3config = create_web3_config(
        gas_price_method=None,
        json_rpc_binance=json_rpc_binance,
        json_rpc_polygon=json_rpc_polygon,
        json_rpc_avalanche=json_rpc_avalanche,
        json_rpc_ethereum=json_rpc_ethereum,
        json_rpc_arbitrum=json_rpc_arbitrum,
    )

    assert web3config, "No RPC endpoints given. A working JSON-RPC connection is needed for check-wallet"

    # Check that we are connected to the chain strategy assumes
    web3config.set_default_chain(chain)
    web3config.check_default_chain_id()

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

    if not state_file:
        state_file = f"state/{id}.json"

    store = create_state_store(Path(state_file))
    assert store.is_pristine(), f"State file already exists: {state_file}"

    state = store.create(strategy_file.name)

    logger.info("Creating initial sync")
    sync_model.sync_initial(state)

    store.sync(state)

    web3config.close()
