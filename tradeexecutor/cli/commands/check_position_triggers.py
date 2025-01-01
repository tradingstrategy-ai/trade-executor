"""Command to check position triggers, mainly used in testing at the moment."""
import datetime
from pathlib import Path
from typing import Optional

from eth_defi.hotwallet import HotWallet

from tradeexecutor.strategy.account_correction import check_accounts as _check_accounts
from .app import app
from ..bootstrap import prepare_executor_id, create_web3_config, create_state_store, create_client, create_execution_and_sync_model
from ..log import setup_logging
from ...strategy.approval import UncheckedApprovalModel
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
def check_position_triggers(
    id: str = shared_options.id,

    strategy_file: Path = shared_options.strategy_file,
    state_file: Optional[Path] = shared_options.state_file,
    private_key: Optional[str] = shared_options.private_key,
    log_level: str = shared_options.log_level,

    trading_strategy_api_key: str = shared_options.trading_strategy_api_key,
    cache_path: Optional[Path] = shared_options.cache_path,

    # Get minimum gas balance from the env
    min_gas_balance: Optional[float] = shared_options.min_gas_balance,

    asset_management_mode: AssetManagementMode = shared_options.asset_management_mode,
    vault_address: Optional[str] = shared_options.vault_address,
    vault_adapter_address: Optional[str] = shared_options.vault_adapter_address,
    vault_payment_forwarder_address: Optional[str] = shared_options.vault_payment_forwarder,
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
):
    """Check that state internal ledger matches on chain balances.

    - Print out differences between actual on-chain balances and expected state balances
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

    mod = read_strategy_module(strategy_file)

    execution_model, sync_model, valuation_model_factory, pricing_model_factory = create_execution_and_sync_model(
        asset_management_mode=asset_management_mode,
        private_key=private_key,
        web3config=web3config,
        confirmation_timeout=datetime.timedelta(seconds=60),
        confirmation_block_count=6,
        max_slippage=0.01,
        min_gas_balance=min_gas_balance,
        vault_address=vault_address,
        vault_adapter_address=vault_adapter_address,
        vault_payment_forwarder_address=vault_payment_forwarder_address,
        routing_hint=mod.trade_routing,
    )

    logger.info("RPC details")

    # Check the chain is online
    logger.info(f"  Chain id is {web3.eth.chain_id:,}")
    logger.info(f"  Latest block is {web3.eth.block_number:,}")

    # Check balances
    logger.info("Balance details")
    logger.info("  Hot wallet is %s", hot_wallet.address)

    vault_address =  sync_model.get_key_address()
    if vault_address:
        logger.info("  Vault is %s", vault_address)
        if vault_deployment_block_number:
            start_block = vault_deployment_block_number
            logger.info("  Vault deployment block number is %d", start_block)

    if not state_file:
        state_file = f"state/{id}.json"

    state_file = Path(state_file)
    store = create_state_store(state_file)
    assert not store.is_pristine(), f"State does not exists yet: {state_file}"

    state = store.load()

    # mod: StrategyModuleInformation = read_strategy_module(strategy_file)

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

    execution_context = ExecutionContext(
        mode=ExecutionMode.unit_testing_trading,
        engine_version=mod.trading_strategy_engine_version,
    )

    strategy_factory = make_factory_from_strategy_mod(mod)

    run_description: StrategyExecutionDescription = strategy_factory(
        execution_model=execution_model,
        execution_context=execution_context,
        sync_model=sync_model,
        valuation_model_factory=valuation_model_factory,
        pricing_model_factory=pricing_model_factory,
        approval_model=UncheckedApprovalModel(),
        client=client,
        run_state=RunState(),
        timed_task_context_manager=execution_context.timed_task_context_manager,
    )

    universe_model: TradingStrategyUniverseModel = run_description.universe_model
    universe = universe_model.construct_universe(
        datetime.datetime.utcnow(),
        execution_context.mode,
        UniverseOptions(history_period=mod.get_live_trading_history_period()),
    )

    logger.info("Universe contains %d pairs", universe.data_universe.pairs.get_count())
    logger.info("Reserve assets are: %s", universe.reserve_assets)

    runner = run_description.runner
    routing_state, pricing_model, valuation_method = runner.setup_routing(universe)

    ts = datetime.datetime.utcnow()

    trades = runner.check_position_triggers(
        ts,
        state,
        universe,
        pricing_model,
        routing_state,
        long_short_metrics_latest=None,
    )
