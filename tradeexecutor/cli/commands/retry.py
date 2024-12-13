"""retry command"""

import datetime
from pathlib import Path
from typing import Optional


from tradeexecutor.cli.bootstrap import (
    prepare_executor_id, prepare_cache, create_web3_config, create_state_store,
    create_execution_and_sync_model, create_client,
)
from tradeexecutor.cli.log import setup_logging
from tradeexecutor.ethereum.rebroadcast import rebroadcast_all
from tradeexecutor.state.retry import retry_trades
from tradeexecutor.strategy.approval import UncheckedApprovalModel
from tradeexecutor.strategy.bootstrap import make_factory_from_strategy_mod
from tradeexecutor.strategy.description import StrategyExecutionDescription
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.execution_model import AssetManagementMode
from tradeexecutor.strategy.run_state import RunState
from tradeexecutor.strategy.strategy_module import read_strategy_module, StrategyModuleInformation
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverseModel
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradeexecutor.utils.timer import timed_task
from tradeexecutor.cli.commands import shared_options

from .app import app


@app.command()
def retry(
    id: str = shared_options.id,
    name: Optional[str] = shared_options.name,
    strategy_file: Path = shared_options.strategy_file,
    trading_strategy_api_key: str = shared_options.trading_strategy_api_key,
    state_file: Optional[Path] = shared_options.state_file,
    cache_path: Optional[Path] = shared_options.cache_path,

    log_level: str = shared_options.log_level,
    json_rpc_binance: Optional[str] = shared_options.json_rpc_binance,
    json_rpc_polygon: Optional[str] = shared_options.json_rpc_polygon,
    json_rpc_avalanche: Optional[str] = shared_options.json_rpc_avalanche,
    json_rpc_ethereum: Optional[str] = shared_options.json_rpc_ethereum,
    json_rpc_base: Optional[str] = shared_options.json_rpc_base,
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
):
    """Retry a failed trade.

    Attempt to rebroadcast failed trades and unfreeze the position
    """

    id = prepare_executor_id(id, strategy_file)

    logger = setup_logging(log_level=log_level)

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
        json_rpc_ethereum=json_rpc_ethereum, json_rpc_base=json_rpc_base, 
        json_rpc_anvil=json_rpc_anvil,
        json_rpc_arbitrum=json_rpc_arbitrum,
        unit_testing=unit_testing,
    )

    assert web3config, "No RPC endpoints given. A working JSON-RPC connection is needed for check-wallet"

    # Check that we are connected to the chain strategy assumes
    web3config.set_default_chain(mod.get_default_chain_id())

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

    assert not store.is_pristine(), f"Cannot repair prisnite strategy: {state_file}"
    state = store.load()

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

    # We construct the trading universe to know what's our reserve asset
    universe_model: TradingStrategyUniverseModel = run_description.universe_model
    ts = datetime.datetime.utcnow()
    universe = universe_model.construct_universe(
        ts,
        ExecutionMode.preflight_check,
        UniverseOptions()
    )

    runner = run_description.runner
    routing_model = runner.routing_model
    routing_state, pricing_model, valuation_method = runner.setup_routing(universe)

    # Then retry trades
    sync_model.resync_nonce()

    report = retry_trades(
        state=state,
        execution_model=execution_model,
        routing_model=routing_model,
        routing_state=routing_state,
        attempt_repair=True,
        interactive=True,
    )

    store.sync(state)

    logger.info(f"Repair report:\n{report}")

