"""close-all command"""

import datetime
from pathlib import Path
from typing import Optional

from .app import app
from ..bootstrap import prepare_executor_id, prepare_cache, create_web3_config, create_state_store, \
    create_execution_and_sync_model, create_client
from ..log import setup_logging
from ...strategy.approval import UncheckedApprovalModel
from ...strategy.bootstrap import make_factory_from_strategy_mod
from ...strategy.description import StrategyExecutionDescription
from ...strategy.execution_context import ExecutionContext, ExecutionMode
from ...strategy.execution_model import AssetManagementMode
from ...strategy.run_state import RunState
from ...strategy.strategy_module import read_strategy_module, StrategyModuleInformation
from ...strategy.trading_strategy_universe import TradingStrategyUniverseModel
from ...strategy.universe_model import UniverseOptions
from ...utils.timer import timed_task
from tradeexecutor.cli.commands import shared_options
from tradeexecutor.cli.close_all import close_all as _close_all


@app.command()
def close_all(
    id: str = shared_options.id,

    strategy_file: Path = shared_options.strategy_file,
    trading_strategy_api_key: str = shared_options.trading_strategy_api_key,
    state_file: Optional[Path] = shared_options.state_file,
    cache_path: Optional[Path] = shared_options.cache_path,

    log_level: str = shared_options.log_level,
    json_rpc_binance: Optional[str] = shared_options.json_rpc_binance,
    json_rpc_polygon: Optional[str] = shared_options.json_rpc_polygon,
    json_rpc_avalanche: Optional[str] = shared_options.json_rpc_avalanche,
    json_rpc_ethereum: Optional[str] = shared_options.json_rpc_ethereum,
    json_rpc_arbitrum: Optional[str] = shared_options.json_rpc_arbitrum,
    json_rpc_anvil: Optional[str] = shared_options.json_rpc_anvil,
    private_key: str = shared_options.private_key,

    asset_management_mode: AssetManagementMode = shared_options.asset_management_mode,
    vault_address: Optional[str] = shared_options.vault_address,
    vault_adapter_address: Optional[str] = shared_options.vault_adapter_address,
    vault_payment_forwarder_address: Optional[str] = shared_options.vault_payment_forwarder,
    min_gas_balance: Optional[float] = shared_options.min_gas_balance,
    max_slippage: float = shared_options.max_slippage,

    confirmation_block_count: int = shared_options.confirmation_block_count,

    # Test EVM backend when running e2e tests
    test_evm_uniswap_v2_router: Optional[str] = shared_options.test_evm_uniswap_v2_router,
    test_evm_uniswap_v2_factory: Optional[str] = shared_options.test_evm_uniswap_v2_factory,
    test_evm_uniswap_v2_init_code_hash: Optional[str] = shared_options.test_evm_uniswap_v2_init_code_hash,

    unit_testing: bool = shared_options.unit_testing,
    simulate: bool = shared_options.simulate,
):
    """Close all open positions.

    - Syncs the latest reserve deposits and redemptions

    - Closes any position that's open currently
    """
    id = prepare_executor_id(id, strategy_file)

    logger = setup_logging(log_level=log_level)

    mod: StrategyModuleInformation = read_strategy_module(strategy_file)

    cache_path = prepare_cache(id, cache_path)

    web3config = create_web3_config(
        json_rpc_binance=json_rpc_binance,
        json_rpc_polygon=json_rpc_polygon,
        json_rpc_avalanche=json_rpc_avalanche,
        json_rpc_ethereum=json_rpc_ethereum,
        json_rpc_anvil=json_rpc_anvil,
        json_rpc_arbitrum=json_rpc_arbitrum,
        simulate=simulate,
    )

    if not web3config.has_any_connection():
        raise RuntimeError("Vault deploy requires that you pass JSON-RPC connection to one of the networks")

    web3config.choose_single_chain()

    execution_model, sync_model, valuation_model_factory, pricing_model_factory = create_execution_and_sync_model(
        asset_management_mode=asset_management_mode,
        private_key=private_key,
        web3config=web3config,
        confirmation_timeout=datetime.timedelta(seconds=60),
        confirmation_block_count=confirmation_block_count,
        min_gas_balance=min_gas_balance,
        max_slippage=max_slippage,
        vault_address=vault_address,
        vault_adapter_address=vault_adapter_address,
        vault_payment_forwarder_address=vault_payment_forwarder_address,
        routing_hint=mod.trade_routing,
    )

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

    if not state_file:
        state_file = f"state/{id}.json"

    store = create_state_store(Path(state_file))

    assert not store.is_pristine(), f"Strategy state file does not exist: {state_file}"
    state = store.load()

    if simulate:
        def break_sync(x):
            raise NotImplementedError("Cannot save state when simulating")
        store.sync = break_sync

        logger.info("Simulating test trades")

    interactive = not (unit_testing or simulate)

    execution_context = ExecutionContext(
        mode=ExecutionMode.one_off,
        timed_task_context_manager=timed_task,
        engine_version=mod.trading_strategy_engine_version,
    )

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

    universe_options = mod.get_universe_options(execution_context.mode)

    # We construct the trading universe to know what's our reserve asset
    universe_model: TradingStrategyUniverseModel = run_description.universe_model
    ts = datetime.datetime.utcnow()
    universe = universe_model.construct_universe(
        ts,
        execution_context.mode,
        universe_options
    )

    runner = run_description.runner
    routing_state, pricing_model, valuation_method = runner.setup_routing(universe)

    _close_all(
        web3config.get_default(),
        execution_model,
        pricing_model,
        sync_model,
        state,
        universe,
        runner.routing_model,
        routing_state,
        slippage_tolerance=mod.parameters.get("slippage_tolerance") or 0.01,
        interactive=interactive,
    )

    # Store the test trade data in the strategy history
    if not simulate:
        store.sync(state)

    logger.info("All ok")
