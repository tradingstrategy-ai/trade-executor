"""console command."""
import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional

import typer

from IPython import embed
import pandas as pd

from eth_defi.hotwallet import HotWallet
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.timebucket import TimeBucket
from . import shared_options

from .app import app
from ..bootstrap import prepare_executor_id, prepare_cache, create_web3_config, create_trade_execution_model, \
    create_state_store
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
from ...utils.timer import timed_task


def launch_console(bindings: dict):
    """Start IPython session"""

    print('')
    print('Following classes and objects are available:')
    for var, val in bindings.items():
        line = "{key:30}: {value}".format(
            key=var,
            value=str(val).replace('\n', ' ').replace('\r', ' ')
        )
        print(line)
    print('')

    embed(user_ns=bindings, colors="Linux")


@app.command()
def console(
    id: str = typer.Option(None, envvar="EXECUTOR_ID", help="Executor id used when programmatically referring to this instance. If not given, take the base of --strategy-file."),

    # State
    state_file: Optional[Path] = typer.Option(None, envvar="STATE_FILE", help="JSON file where we serialise the execution state. If not given defaults to state/{executor-id}.json"),

    strategy_file: Path = typer.Option(..., envvar="STRATEGY_FILE"),
    private_key: str = typer.Option(None, envvar="PRIVATE_KEY"),
    trading_strategy_api_key: str = typer.Option(None, envvar="TRADING_STRATEGY_API_KEY", help="Trading Strategy API key"),
    cache_path: Optional[Path] = typer.Option("cache/", envvar="CACHE_PATH", help="Where to store downloaded datasets"),

    # Get minimum gas balance from the env
    minimum_gas_balance: Optional[float] = typer.Option(0.1, envvar="MINUMUM_GAS_BALANCE", help="What is the minimum balance of gas token you need to have in your wallet. If the balance falls below this, abort by crashing and do not attempt to create transactions. Expressed in the native token e.g. ETH."),

    # Web3 connection options
    json_rpc_binance: Optional[str] = shared_options.json_rpc_binance,
    json_rpc_polygon: Optional[str] = shared_options.json_rpc_polygon,
    json_rpc_ethereum: Optional[str] = shared_options.json_rpc_ethereum,
    json_rpc_avalanche: Optional[str] = shared_options.json_rpc_avalanche,
    json_rpc_arbitrum: Optional[str] = shared_options.json_rpc_arbitrum,
    json_rpc_anvil: Optional[str] = shared_options.json_rpc_anvil,

    # Live trading or backtest
    asset_management_mode: AssetManagementMode = shared_options.asset_management_mode,
    vault_address: Optional[str] = shared_options.vault_address,
    vault_adapter_address: Optional[str] = shared_options.vault_adapter_address,

    log_level: str = typer.Option(None, envvar="LOG_LEVEL", help="The Python default logging level. The defaults are 'info' is live execution, 'warning' if backtesting. Set 'disabled' in testing."),

    unit_testing: bool = typer.Option(False, "--unit-testing", envvar="UNIT_TESTING", help="The trade executor is called under the unit testing mode. No caches are purged."),
):
    """Open interactive IPython console to explore state.

    Open an interactive Python prompt where you can inspect and debug the current trade
    executor state.

    Strategy, state and execution state are loaded to the memory for debugging.
    """

    global logger

    id = prepare_executor_id(id, strategy_file)

    logger = setup_logging(log_level)

    mod = read_strategy_module(strategy_file)

    cache_path = prepare_cache(id, cache_path)

    client = Client.create_live_client(trading_strategy_api_key, cache_path=cache_path)

    execution_context = ExecutionContext(
        mode=ExecutionMode.preflight_check,
        timed_task_context_manager=timed_task
    )

    web3config = create_web3_config(
        json_rpc_binance=json_rpc_binance,
        json_rpc_polygon=json_rpc_polygon,
        json_rpc_avalanche=json_rpc_avalanche,
        json_rpc_ethereum=json_rpc_ethereum,
        json_rpc_anvil=json_rpc_anvil,
        json_rpc_arbitrum=json_rpc_arbitrum,
    )

    assert web3config, "No RPC endpoints given. A working JSON-RPC connection is needed for check-wallet"

    # Check that we are connected to the chain strategy assumes
    web3config.set_default_chain(mod.chain_id)
    web3config.check_default_chain_id()

    execution_model, sync_model, valuation_model_factory, pricing_model_factory = create_trade_execution_model(
        asset_management_mode=AssetManagementMode.hot_wallet,
        private_key=private_key,
        web3config=web3config,
        confirmation_timeout=datetime.timedelta(seconds=60),
        confirmation_block_count=6,
        max_slippage=0.01,
        min_balance_threshold=minimum_gas_balance,
        vault_address=vault_address,
        vault_adapter_address=vault_adapter_address,
    )

    hot_wallet = HotWallet.from_private_key(private_key)

    # Set up the strategy engine
    factory = make_factory_from_strategy_mod(mod)
    run_description: StrategyExecutionDescription = factory(
        execution_model=execution_model,
        execution_context=execution_context,
        timed_task_context_manager=execution_context.timed_task_context_manager,
        sync_model=sync_model,
        valuation_model_factory=valuation_model_factory,
        pricing_model_factory=pricing_model_factory,
        approval_model=UncheckedApprovalModel(),
        client=client,
        run_state=RunState(),
    )

    # We construct the trading universe to know what's our reserve asset
    universe_model: TradingStrategyUniverseModel = run_description.universe_model
    ts = datetime.datetime.utcnow()
    universe = universe_model.construct_universe(
        ts,
        ExecutionMode.preflight_check,
        UniverseOptions())

    # Get all tokens from the universe
    reserve_assets = universe.reserve_assets
    web3 = web3config.get_default()

    logger.info("RPC details")

    # Check the chain is online
    logger.info(f"  Chain id is {web3.eth.chain_id:,}")
    logger.info(f"  Latest block is {web3.eth.block_number:,}")

    # Check balances
    logger.info("Balance details")
    logger.info("  Hot wallet is %s", hot_wallet.address)
    gas_balance = web3.eth.get_balance(hot_wallet.address) / 10**18
    logger.info("  We have %f tokens for gas left", gas_balance)

    if not state_file:
        state_file = f"state/{id}.json"

    store = create_state_store(Path(state_file))

    if store.is_pristine():
        state = store.create()
    else:
        state = store.load()

    logger.info("State details")
    logger.info("  Number of positions: %s", len(list(state.portfolio.get_all_positions())))
    logger.info("  Number of trades: %s", len(list(state.portfolio.get_all_trades())))

    runner = run_description.runner
    routing_state, pricing_model, valuation_method = runner.setup_routing(universe)

    # Set up the default objects
    # availalbe in the interactive session
    bindings = {
        "web3": web3,
        "client": client,
        "state": state,
        "universe": universe,
        "store": store,
        "routing_state": routing_state,
        "pricing_model": pricing_model,
        "valuation_method": valuation_method,
        "pd": pd,
        "cache_path": cache_path,
        "datetime": datetime,
        "Decimal": Decimal,
        "ExecutionMode": ExecutionMode,
        "ChainId": ChainId,
        "TimeBucket": TimeBucket,
    }

    if not unit_testing:
        launch_console(bindings)
