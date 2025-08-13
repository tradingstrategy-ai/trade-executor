"""console command.

- Open interactive IPython session within the trade-executor

- Can be used as a part of Docker image

To start a console in shell with `docker-compose.yml` set up:

.. code-block:: shell

     docker-compose run $YOUR_CONTAINER_NAME console




"""
import datetime
import itertools
import multiprocessing
from decimal import Decimal
from pathlib import Path
from typing import Optional

import typer

from IPython import embed
import pandas as pd
from eth.vm.logic.block import timestamp
from typer import Option

from eth_defi.gas import GasPriceMethod
from eth_defi.hotwallet import HotWallet
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.timebucket import TimeBucket
from . import shared_options

from .app import app
from ..bootstrap import prepare_executor_id, prepare_cache, create_web3_config, create_execution_and_sync_model, \
    create_state_store, create_client, configure_default_chain
from ..log import setup_logging
from ..version_info import VersionInfo
from ...analysis.pair import display_strategy_universe
from ...monkeypatch.web3 import construct_sign_and_send_raw_middleware
from ...state.state import State
from ...statistics.in_memory_statistics import refresh_run_state
from ...strategy.approval import UncheckedApprovalModel
from ...strategy.bootstrap import make_factory_from_strategy_mod
from ...strategy.cycle import CycleDuration
from ...strategy.description import StrategyExecutionDescription
from ...strategy.execution_context import ExecutionContext, ExecutionMode
from ...strategy.execution_model import AssetManagementMode
from ...strategy.pandas_trader.indicator import DiskIndicatorStorage, MemoryIndicatorStorage, prepare_indicators, calculate_indicators
from ...strategy.pandas_trader.strategy_input import StrategyInputIndicators
from ...strategy.parameters import dump_parameters
from ...strategy.run_state import RunState
from ...strategy.strategy_module import read_strategy_module
from ...strategy.trading_strategy_universe import TradingStrategyUniverseModel
from ...strategy.universe_model import UniverseOptions
from ...utils.cpu import get_safe_max_workers_count
from ...utils.timer import timed_task


def launch_console(bindings: dict):
    """Start IPython session.

    Assume line length of 130.

    :param eval:
        Run this Python snippet and exit.
    """

    print('')
    print('Following classes and objects are added to the interactive interpreter without import:')
    for var, val in bindings.items():
        str_value = str(val)
        str_value = str_value[0:100] + "..." if len(str_value) > 60 else str_value
        line = "{key:25}: {value}".format(
            key=var,
            value=str_value.replace('\n', ' ').replace('\r', ' ')
        )
        print(line)
    print('')

    embed(user_ns=bindings, colors="Linux")


@app.command()
def console(
    id: str = typer.Option(None, envvar="EXECUTOR_ID", help="Executor id used when programmatically referring to this instance. If not given, take the base of --strategy-file."),

    # State
    state_file: Optional[Path] = shared_options.state_file,

    strategy_file: Path = shared_options.strategy_file,
    private_key: str = shared_options.private_key,
    trading_strategy_api_key: str = shared_options.trading_strategy_api_key,
    cache_path: Optional[Path] = shared_options.cache_path,

    # Web3 connection options
    json_rpc_binance: Optional[str] = shared_options.json_rpc_binance,
    json_rpc_polygon: Optional[str] = shared_options.json_rpc_polygon,
    json_rpc_ethereum: Optional[str] = shared_options.json_rpc_ethereum,
    json_rpc_avalanche: Optional[str] = shared_options.json_rpc_avalanche,
    json_rpc_base: Optional[str] = shared_options.json_rpc_base,
    json_rpc_arbitrum: Optional[str] = shared_options.json_rpc_arbitrum,
    json_rpc_anvil: Optional[str] = shared_options.json_rpc_anvil,

    # Live trading or backtest
    asset_management_mode: AssetManagementMode = shared_options.asset_management_mode,
    vault_address: Optional[str] = shared_options.vault_address,
    vault_adapter_address: Optional[str] = shared_options.vault_adapter_address,
    vault_payment_forwarder_address: Optional[str] = shared_options.vault_payment_forwarder,

    log_level: str = shared_options.log_level,

    unit_testing: bool = shared_options.unit_testing,
    max_workers: Optional[int] = shared_options.max_workers,

    gas_price_method: Optional[GasPriceMethod] = shared_options.gas_price_method,
    simulate: bool = shared_options.simulate,

    code: Optional[str] = Option(None, envvar="CODE", help="Run this Python snipped and exit"),

):
    """Open interactive IPython console to explore state.

    Open an interactive Python prompt where you can inspect and debug the current trade
    executor state.

    Strategy, state and execution state are loaded to the memory for debugging.

    Assumes you have a strategy deployed as a Docker container,
    environment variabels and such are set up, then you want to diagnose
    or modify the strategy environment after it has been taken offline.

    For an example console command you can read the hot wallet balance by:

        >>> web3.eth.get_balance(hot_wallet.address) / 10**18

    """

    global logger

    id = prepare_executor_id(id, strategy_file)

    logger = setup_logging(log_level)

    version_info = VersionInfo.read_docker_version()
    logger.info(f"Docker image version: {version_info.tag}")
    logger.info(f"Commit hash: {version_info.commit_hash}")
    logger.info(f"Commit message: {version_info.commit_message}")
    logger.info("")

    mod = read_strategy_module(strategy_file)

    cache_path = prepare_cache(id, cache_path)

    execution_context = ExecutionContext(
        mode=ExecutionMode.one_off,
        timed_task_context_manager=timed_task,
        engine_version=mod.trading_strategy_engine_version,
    )

    web3config = create_web3_config(
        json_rpc_binance=json_rpc_binance,
        json_rpc_polygon=json_rpc_polygon,
        json_rpc_avalanche=json_rpc_avalanche,
        json_rpc_ethereum=json_rpc_ethereum,
        json_rpc_base=json_rpc_base,
        json_rpc_anvil=json_rpc_anvil,
        json_rpc_arbitrum=json_rpc_arbitrum,
        gas_price_method=gas_price_method,
        unit_testing=unit_testing,
        simulate=simulate,
    )

    assert web3config, "No RPC endpoints given. A working JSON-RPC connection is needed for check-wallet"

    hot_wallet = HotWallet.from_private_key(private_key)

    # Check that we are connected to the chain strategy assumes
    configure_default_chain(
        web3config,
        mod,
    )

    if hot_wallet:
        # Add to Python console singing
        web3config.add_hot_wallet_signing(hot_wallet)

    run_state = RunState()
    run_state.version = VersionInfo.read_docker_version()
    run_state.executor_id = id

    client, routing_model = create_client(
        mod=mod,
        web3config=web3config,
        trading_strategy_api_key=trading_strategy_api_key,
        cache_path=cache_path,
        clear_caches=False,
        test_evm_uniswap_v2_factory=None,
        test_evm_uniswap_v2_router=None,
        test_evm_uniswap_v2_init_code_hash=None,
        asset_management_mode=asset_management_mode,
    )
    assert client is not None, "You need to give details for TradingStrategy.ai client"

    execution_model, sync_model, valuation_model_factory, pricing_model_factory = create_execution_and_sync_model(
        asset_management_mode=asset_management_mode,
        private_key=private_key,
        web3config=web3config,
        confirmation_timeout=datetime.timedelta(seconds=5*60),
        confirmation_block_count=5,
        max_slippage=0.02,
        min_gas_balance=0,
        vault_address=vault_address,
        vault_adapter_address=vault_adapter_address,
        vault_payment_forwarder_address=vault_payment_forwarder_address,
        routing_hint=mod.trade_routing,
    )

    logger.info("Valuation model factory is %s, pricing model factory is %s", valuation_model_factory, pricing_model_factory)

    parameters = mod.parameters
    if parameters:
        universe_options = UniverseOptions.from_strategy_parameters_class(parameters, execution_context)
        dump_parameters(parameters)
    else:
        universe_options = UniverseOptions()

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
        routing_model=routing_model,
        run_state=RunState(),
    )

    run_state.source_code = run_description.source_code

    # We construct the trading universe to know what's our reserve asset
    universe_model: TradingStrategyUniverseModel = run_description.universe_model

    cycle_duration: CycleDuration = None
    if mod and mod.parameters:
        cycle_duration = mod.parameters.get("cycle_duration")

    if cycle_duration:
        # We need to found universe timestamp to its previous cycle when we have data
        cycle_timestamp = cycle_duration.round_down(datetime.datetime.utcnow())
    else:
        # Legacy path
        cycle_timestamp = datetime.datetime.utcnow()

    universe = universe_model.construct_universe(
        cycle_timestamp,
        ExecutionMode.preflight_check,
        universe_options,
        execution_model=execution_model,
        strategy_parameters=mod.parameters,
    )

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 140):
        universe_df = display_strategy_universe(universe)
        logger.info("Universe is:\n%s", str(universe_df))

    # Get all tokens from the universe
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
        state = store.create(id)
    else:
        state = store.load()

    logger.info("State details")
    logger.info("  Number of positions: %s", len(list(state.portfolio.get_all_positions())))
    logger.info("  Number of trades: %s", len(list(state.portfolio.get_all_trades())))

    runner = run_description.runner
    routing_state, pricing_model, valuation_model = runner.setup_routing(universe)

    # TODO: Make construction of routing model cleaner
    if routing_model is None:
        routing_model = runner.routing_model

    # Expose the previous backtest run to in the console as well
    backtest_result = Path(f"state/{id}-backtest.json")
    if backtest_result.exists():
        backtested_state = State.read_json_file(backtest_result)
    else:
        backtested_state = None

    cycle_duration = run_description.cycle_duration
    refresh_run_state(
        run_state,
        state,
        execution_context,
        visualisation=True,
        universe=universe,
        sync_model=sync_model,
        backtested_state=backtested_state,
        cycle_duration=cycle_duration,
    )

    if mod.create_indicators:
        # If the strategy uses indicators calculate and expose them to the console
        #

        if max_workers is None:
            max_workers = get_safe_max_workers_count()

        print(f"Creating real-time indicators for the strategy module for timestamp {cycle_timestamp}")
        print(f"No indicator cache used")

        # TODO: MemoryIndicatorStorage does not support yet parallel indicator calculations
        max_workers = 1

        assert mod.parameters, "You need to have Parameters class if create_indicators is specified"
        indicator_storage = MemoryIndicatorStorage(universe_key=universe.get_cache_key())
        indicator_set = prepare_indicators(
            mod.create_indicators,
            mod.parameters,
            universe,
            execution_context,
            timestamp=cycle_timestamp,
        )
        indicators_needed = set(indicator_set.generate_combinations(universe))
        indicator_result_map = calculate_indicators(
            universe,
            indicator_storage,
            indicator_set,
            execution_context=execution_context,
            remaining=indicators_needed,
            all_combinations=indicators_needed,
            max_workers=max_workers,
            strategy_cycle_timestamp=cycle_timestamp,
        )
        indicators = StrategyInputIndicators(
            strategy_universe=universe,
            available_indicators=indicator_set,
            indicator_results=indicator_result_map,
            timestamp=pd.Timestamp(cycle_timestamp),
        )
    else:
        indicator_storage = indicator_set = indicator_result_map = indicators = None

    # Expose Vault smart contract proxy class
    vault = getattr(sync_model, "vault", None)

    # Set up the default objects in Python eval context
    # and available in the interactive session
    bindings = {
        "cycle_timestamp": cycle_timestamp,
        "web3": web3,
        "client": client,
        "state": state,
        "strategy_universe": universe,
        "store": store,
        "hot_wallet": hot_wallet,
        "routing_state": routing_state,
        "pricing_model": pricing_model,
        "valuation_model": valuation_model,
        "routing_model": routing_model,
        "execution_model": execution_model,
        "runner": runner,
        "sync_model": sync_model,
        "pd": pd,
        "cache_path": cache_path.absolute(),
        "datetime": datetime,
        "Decimal": Decimal,
        "ExecutionMode": ExecutionMode,
        "ChainId": ChainId,
        "TimeBucket": TimeBucket,
        "strategy_module": mod,
        "run_state": run_state,
        "backtested_state": backtested_state,
        "parameters": parameters,
        "indicator_storage": indicator_storage,
        "indicator_set": indicator_set,
        "indicator_result_map": indicator_result_map,
        "indicators": indicators,
        "vault": vault,
    }

    # Expose pairs to console as well
    for pair in itertools.islice(universe.iterate_pairs(), 5):
        name = pair.get_ticker().lower().replace("-", "_")
        bindings[name] = pair

    if code:
        logger.info(f"Executing Python code:\n{code}")
        exec(code, bindings)
        print("All olk")
    elif not unit_testing:
        logger.info("Launching interactive console session. Type `exit` to exit the console.")
        launch_console(bindings)
