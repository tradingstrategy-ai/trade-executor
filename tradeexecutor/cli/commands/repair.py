"""repair command.

This command does not have automatic test coverage,
as it is pretty hard to drive to the test point.
"""

import datetime
from pathlib import Path
from typing import Optional

import typer

from tradingstrategy.client import Client
from .app import app
from ..bootstrap import prepare_executor_id, prepare_cache, create_web3_config, create_state_store, \
    create_trade_execution_model
from ..log import setup_logging
from ...state.repair import repair_trades
from ...state.state import UncleanState
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


@app.command()
def repair(
    id: str = typer.Option(None, envvar="EXECUTOR_ID", help="Executor id used when programmatically referring to this instance. If not given, take the base of --strategy-file."),
    log_level: str = typer.Option(None, envvar="LOG_LEVEL", help="The Python default logging level. The defaults are 'info' is live execution, 'warning' if backtesting. Set 'disabled' in testing."),

    strategy_file: Path = typer.Option(..., envvar="STRATEGY_FILE"),
    private_key: str = typer.Option(None, envvar="PRIVATE_KEY", help="Trade executor private key."),
    trading_strategy_api_key: str = typer.Option(None, envvar="TRADING_STRATEGY_API_KEY", help="Trading Strategy API key"),
    state_file: Optional[Path] = typer.Option(None, envvar="STATE_FILE", help="JSON file where we serialise the execution state. If not given defaults to state/{executor-id}.json"),
    # cache_path: Optional[Path] = typer.Option("cache/", envvar="CACHE_PATH", help="Where to store downloaded datasets"),

    # Web3 connection options
    json_rpc_binance: str = typer.Option(None, envvar="JSON_RPC_BINANCE", help="BNB Chain JSON-RPC node URL we connect to"),
    json_rpc_polygon: str = typer.Option(None, envvar="JSON_RPC_POLYGON", help="Polygon JSON-RPC node URL we connect to"),
    json_rpc_ethereum: str = typer.Option(None, envvar="JSON_RPC_ETHEREUM", help="Ethereum JSON-RPC node URL we connect to"),
    json_rpc_avalanche: str = typer.Option(None, envvar="JSON_RPC_AVALANCHE", help="Avalanche C-chain JSON-RPC node URL we connect to"),
):
    """Repair broken state.

    Attempt to repair a broken strategy execution state. This may include

    - Confirming trades that failed to broadcast

    """
    global logger

    id = prepare_executor_id(id, strategy_file)

    logger = setup_logging(log_level=log_level)

    mod = read_strategy_module(strategy_file)

    # cache_path = prepare_cache(id, cache_path)

    # client = Client.create_live_client(trading_strategy_api_key, cache_path=cache_path)

    execution_context = ExecutionContext(
        mode=ExecutionMode.preflight_check,
        timed_task_context_manager=timed_task
    )

    # web3config = create_web3_config(
    #     json_rpc_binance=json_rpc_binance,
    #     json_rpc_polygon=json_rpc_polygon,
    #     json_rpc_avalanche=json_rpc_avalanche,
    #     json_rpc_ethereum=json_rpc_ethereum,
    #     gas_price_method=None,
    # )
    #
    # assert web3config, "No RPC endpoints given. A working JSON-RPC connection is needed for check-wallet"
    #
    # # Check that we are connected to the chain strategy assumes
    # web3config.set_default_chain(mod.chain_id)
    # # web3config.check_default_chain_id()
    #
    # execution_model, sync_method, valuation_model_factory, pricing_model_factory = create_trade_execution_model(
    #     execution_type=TradeExecutionType.uniswap_v2_hot_wallet,
    #     private_key=private_key,
    #     web3config=web3config,
    #     confirmation_timeout=datetime.timedelta(seconds=60),
    #     confirmation_block_count=6,
    #     max_slippage=2.50,
    #     min_balance_threshold=0,
    # )

    if not state_file:
        state_file = f"state/{id}.json"
    store = create_state_store(Path(state_file))
    assert not store.is_pristine(), f"State file not found {state_file}"
    state = store.load()

    # Set up the strategy engine
    # factory = make_factory_from_strategy_mod(mod)
    # run_description: StrategyExecutionDescription = factory(
    #     execution_model=execution_model,
    #     execution_context=execution_context,
    #     timed_task_context_manager=execution_context.timed_task_context_manager,
    #     sync_method=sync_method,
    #     valuation_model_factory=valuation_model_factory,
    #     pricing_model_factory=pricing_model_factory,
    #     approval_model=UncheckedApprovalModel(),
    #     client=client,
    #     run_state=RunState(),
    # )

    # TODO: Current repair logic does not ened the price data ATM
    #
    # We construct the trading universe to know what's our reserve asset
    # universe_model: TradingStrategyUniverseModel = run_description.universe_model
    # ts = datetime.datetime.utcnow()
    # universe = universe_model.construct_universe(
    #     ts,
    #     ExecutionMode.preflight_check,
    #     UniverseOptions())
    #
    # runner = run_description.runner

    report = repair_trades(
        state,
        attempt_repair=True,
        interactive=True,
    )

    store.sync(state)

    print(f"Repair report: {report}")
