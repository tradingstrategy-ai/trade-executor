"""blacklist command.

To run with Docker:

.. code-block:: bash

    docker compose run base-ath blacklist

"""
import datetime
from pathlib import Path
from typing import Optional

from typer import Option

from eth_defi.hotwallet import HotWallet
from tabulate import tabulate

from .app import app
from ..blacklist import display_blacklist
from ..bootstrap import prepare_executor_id, create_web3_config, create_sync_model, create_state_store, create_client
from ..log import setup_logging
from ...strategy.bootstrap import make_factory_from_strategy_mod
from ...strategy.description import StrategyExecutionDescription
from ...strategy.execution_context import ExecutionContext, ExecutionMode
from ...strategy.execution_model import AssetManagementMode
from . import shared_options
from ...strategy.run_state import RunState
from ...strategy.strategy_module import StrategyModuleInformation, read_strategy_module
from ...strategy.trading_strategy_universe import TradingStrategyUniverseModel
from ...strategy.universe_model import UniverseOptions


@app.command()
def blacklist(
    id: str = shared_options.id,

    strategy_file: Path = shared_options.strategy_file,
    state_file: Optional[Path] = shared_options.state_file,
    private_key: Optional[str] = shared_options.private_key,
    log_level: str = shared_options.log_level,

    trading_strategy_api_key: str = shared_options.trading_strategy_api_key,
    cache_path: Optional[Path] = shared_options.cache_path,

    asset_management_mode: AssetManagementMode = shared_options.asset_management_mode,
    vault_address: Optional[str] = shared_options.vault_address,
    vault_adapter_address: Optional[str] = shared_options.vault_adapter_address,
    vault_deployment_block_number: Optional[int] = shared_options.vault_deployment_block_number,

    json_rpc_binance: Optional[str] = shared_options.json_rpc_binance,
    json_rpc_polygon: Optional[str] = shared_options.json_rpc_polygon,
    json_rpc_avalanche: Optional[str] = shared_options.json_rpc_avalanche,
    json_rpc_ethereum: Optional[str] = shared_options.json_rpc_ethereum,
    json_rpc_base: Optional[str] = shared_options.json_rpc_base,
    json_rpc_arbitrum: Optional[str] = shared_options.json_rpc_arbitrum,
    json_rpc_anvil: Optional[str] = shared_options.json_rpc_anvil,

    # Test functionality
    test_evm_uniswap_v2_router: Optional[str] = shared_options.test_evm_uniswap_v2_router,
    test_evm_uniswap_v2_factory: Optional[str] = shared_options.test_evm_uniswap_v2_factory,
    test_evm_uniswap_v2_init_code_hash: Optional[str] = shared_options.test_evm_uniswap_v2_init_code_hash,
    unit_testing: bool = shared_options.unit_testing,

    add_token: str = Option(None, envvar="ADD_TOKEN", help="Add a token by an address to the blacklist"),
    remove_token: str = Option(None, envvar="REMOVE_TOKEN", help="Remove a token by an address from the blacklist"),
):
    """Manage blacklist state.

    - Show currently blacklisted tokens (do not trade them)
    - Add and remove entries
    """

    global logger

    id = prepare_executor_id(id, strategy_file)

    logger = setup_logging(log_level)

    web3config = create_web3_config(
        gas_price_method=None,
        json_rpc_binance=json_rpc_binance,
        json_rpc_polygon=json_rpc_polygon,
        json_rpc_avalanche=json_rpc_avalanche,
        json_rpc_ethereum=json_rpc_ethereum, json_rpc_base=json_rpc_base, 
        json_rpc_arbitrum=json_rpc_arbitrum,
        json_rpc_anvil=json_rpc_anvil,
        unit_testing=unit_testing,
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
        vault_adapter_address,
    )

    if not state_file:
        state_file = f"state/{id}.json"

    state_file = Path(state_file)
    store = create_state_store(state_file)
    assert not store.is_pristine(), f"State does not exists yet: {state_file}"

    state = store.load()

    mod: StrategyModuleInformation = read_strategy_module(strategy_file)

    client, routing_model = create_client(
        mod=mod,
        web3config=web3config,
        trading_strategy_api_key=trading_strategy_api_key,
        cache_path=cache_path,
        test_evm_uniswap_v2_factory=test_evm_uniswap_v2_factory,
        test_evm_uniswap_v2_router=test_evm_uniswap_v2_router,
        test_evm_uniswap_v2_init_code_hash=test_evm_uniswap_v2_init_code_hash,
        clear_caches=False,
    )
    assert client is not None, "You need to give details for TradingStrategy.ai client"

    execution_context = ExecutionContext(mode=ExecutionMode.one_off)

    strategy_factory = make_factory_from_strategy_mod(mod)

    run_description: StrategyExecutionDescription = strategy_factory(
        execution_model=None,
        execution_context=execution_context,
        sync_model=None,
        valuation_model_factory=None,
        pricing_model_factory=None,
        approval_model=None,
        client=client,
        run_state=RunState(),
        timed_task_context_manager=execution_context.timed_task_context_manager,
    )

    universe_model: TradingStrategyUniverseModel = run_description.universe_model
    strategy_universe = universe_model.construct_universe(
        datetime.datetime.utcnow(),
        execution_context.mode,
        UniverseOptions(history_period=mod.get_live_trading_history_period()),
    )

    if add_token:
        asset = strategy_universe.get_asset_by_address(add_token)
        logger.info("Blacklisting asset %s", asset)
        state.blacklist_asset(asset)

    if remove_token:
        assert len(state.blacklisted_assets) > 0, "No blacklisted assets in the state"
        asset = strategy_universe.get_asset_by_address(remove_token)
        logger.info("Unblacklisting asset %s", asset)
        state.unblacklist_asset(asset)

    df = display_blacklist(state, strategy_universe)
    if len(df) > 0:
        output = tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=True)
        logger.info("Blacklisted assets:\n%s", output)
    else:
        logger.info("No blacklisted assets found.")

    if add_token or remove_token:
        logger.info("Saving state to %s", state_file)
        store.sync(state)

