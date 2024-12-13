"""repair command.

This command does not have automatic test coverage,
as it is pretty hard to drive to the test point.
"""
import datetime
from pathlib import Path
from typing import Optional
from typer import Option

from . import shared_options
from .app import app
from ..bootstrap import prepare_executor_id, create_state_store, create_execution_and_sync_model, prepare_cache, create_web3_config, create_client
from ..log import setup_logging
from ...ethereum.rebroadcast import rebroadcast_all
from ...state.repair import repair_trades, repair_tx_not_generated, repair_zero_quantity
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
    auto_approve: bool = Option(False, envvar="AUTO_APPROVE", help="Approve the repair trade without asking for confirmation"),
):
    """Repair broken state.

    Attempt to repair a broken strategy execution state.

    The actions may include

    - If a trade failed and blockchain tx has been marked as reverted,
      undo this trade in the internal accounting

    - If a tranaction has been broadcasted, and can be now confirmed,
      confirm this trade

    - If a transaction has not been broadcasted, try to redo the trade
      by recreating the blockchain transaction objects with a new nonce
      and broadcast them

    See also check-accounts and correct-accounts commands.
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

    universe_options = mod.get_universe_options(execution_context.mode)

    # We construct the trading universe to know what's our reserve asset
    universe_model: TradingStrategyUniverseModel = run_description.universe_model
    ts = datetime.datetime.utcnow()
    universe = universe_model.construct_universe(
        ts,
        ExecutionMode.preflight_check,
        universe_options
    )

    runner = run_description.runner
    routing_model = runner.routing_model
    routing_state, pricing_model, valuation_method = runner.setup_routing(universe)

    #
    # First trades that have txs missing
    #
    repair_tx_not_generated(state, interactive=(not auto_approve))

    #
    # Second fix txs that have unresolved state
    #

    # Get the latest nonce from the chain
    sync_model.resync_nonce()

    trades, txs = rebroadcast_all(
        web3config.get_default(),
        state,
        execution_model,
        routing_model,
        routing_state
    )
    logger.info("Rebroadcast fixed %d trades", len(trades))
    for t in trades:
        logger.info("Trade %s", t)
        for tx in t.blockchain_transactions:
            logger.info("Transaction %s", tx)

    store.sync(state)

    #
    # Repair positions that failed to open, have 0 quanttiy
    #
    repair_zero_quantity(
        state,
        interactive=(not auto_approve),
    )

    #
    # Then resolve trade status based whether
    # its txs have succeeded or not
    #

    report = repair_trades(
        state,
        attempt_repair=True,
        interactive=(not auto_approve),
    )

    store.sync(state)

    logger.info(f"Repair report:\n{report}")
