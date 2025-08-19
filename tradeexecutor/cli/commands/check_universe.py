"""check-universe command"""

import datetime
import logging
import secrets
from pathlib import Path
from typing import Optional

import pandas as pd

from tradingstrategy.chain import ChainId
from .app import app
from .shared_options import unit_testing
from ..bootstrap import prepare_executor_id, prepare_cache, create_web3_config, create_client, create_execution_and_sync_model
from ..log import setup_logging
from ...analysis.pair import display_strategy_universe
from ...strategy.approval import UncheckedApprovalModel
from ...strategy.bootstrap import import_strategy_file, make_factory_from_strategy_mod
from ...strategy.cycle import CycleDuration, snap_to_previous_tick
from ...strategy.description import StrategyExecutionDescription
from ...strategy.execution_context import console_command_execution_context, ExecutionContext, ExecutionMode
from ...strategy.execution_model import AssetManagementMode
from ...strategy.pandas_trader.indicator import calculate_and_load_indicators_inline, MemoryIndicatorStorage
from ...strategy.run_state import RunState
from ...strategy.strategy_module import StrategyModuleInformation, read_strategy_module
from ...strategy.trading_strategy_universe import TradingStrategyUniverseModel
from ...utils.cpu import get_safe_max_workers_count
from . import shared_options
from ...utils.timer import timed_task


@app.command()
def check_universe(
    id: str = shared_options.id,
    strategy_file: Path = shared_options.strategy_file,
    trading_strategy_api_key: str = shared_options.trading_strategy_api_key,
    cache_path: Optional[Path] = shared_options.cache_path,
    max_data_delay_minutes: int = shared_options.max_data_delay_minutes,
    log_level: str = shared_options.log_level,
    max_workers: int | None = shared_options.max_workers,

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
):
    """Checks that the trading universe is healthy.

    Check that create_trading_universe() and create_indicators() functions in the strategy module work,
    and will display all available trading pairs.
    """

    global logger

    id = prepare_executor_id(id, strategy_file)

    logger = setup_logging(log_level)
    if log_level != "disabled":
        assert logger.level <= logging.INFO, "Log level must be at least INFO to get output from this command"

    logger.info("Loading strategy file %s", strategy_file)

    mod: StrategyModuleInformation = read_strategy_module(strategy_file)

    cache_path = prepare_cache(id, cache_path)

    execution_context = ExecutionContext(
        mode=ExecutionMode.preflight_check,
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

    if not web3config.has_any_connection():
        # Only revelvant if create_trading_universe() uses web3 connection
        web3config.default_chain_id = mod.chain_id or ChainId.ethereum
    else:
        web3config.choose_single_chain()

    # create_trading_universe() which needs to access Lagoon
    if asset_management_mode is None:
        asset_management_mode = AssetManagementMode.hot_wallet

    # create_trading_universe() which needs to access Lagoon
    if private_key is None:
        private_key = "0x" + secrets.token_hex(32)

    if web3config.has_any_connection():
        execution_model, sync_model, valuation_model_factory, pricing_model_factory = create_execution_and_sync_model(
            asset_management_mode=asset_management_mode,
            private_key=private_key,
            web3config=web3config,
            min_gas_balance=0,
            max_slippage=99,
            vault_address=vault_address,
            vault_adapter_address=vault_adapter_address,
            vault_payment_forwarder_address=vault_payment_forwarder_address,
            routing_hint=mod.trade_routing,
            confirmation_block_count=0,  # Not used
            confirmation_timeout=datetime.timedelta(seconds=60),  # Not used
        )
    else:
        execution_model = sync_model = valuation_model_factory = pricing_model_factory = None

    client, routing_model = create_client(
        mod=mod,
        web3config=web3config,
        trading_strategy_api_key=trading_strategy_api_key,
        cache_path=cache_path,
        clear_caches=False,
        asset_management_mode=asset_management_mode,
        test_evm_uniswap_v2_factory=None,  # Not used
        test_evm_uniswap_v2_router=None, # Not used
        test_evm_uniswap_v2_init_code_hash=None,  # Not used
    )
    assert client is not None, "You need to give details for TradingStrategy.ai client"

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
        routing_model=routing_model,  # None unless test EVM
        run_state=RunState(),
    )

    universe_options = mod.get_universe_options()

    # We construct the trading universe to know what's our reserve asset
    universe_model: TradingStrategyUniverseModel = run_description.universe_model
    ts = datetime.datetime.utcnow()
    universe = universe_model.construct_universe(
        ts,
        ExecutionMode.preflight_check,
        universe_options,
        strategy_parameters=mod.parameters,
        execution_model=execution_model,
    )

    if max_data_delay_minutes:
        max_data_delay = datetime.timedelta(minutes=max_data_delay_minutes)
        logger.info(f"Maximum price feed delay is {max_data_delay}")
    else:
        logger.info(f"Maximum price feed delay is not set")
        max_data_delay = None

    latest_candle_at = universe_model.check_data_age(ts, universe, max_data_delay)
    ago = datetime.datetime.utcnow() - latest_candle_at
    logger.info("Latest OHCLV candle is at: %s, %s ago", latest_candle_at, ago)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 140):
        universe_df = display_strategy_universe(universe)
        # universe_df = pd.DataFrame(universe.get_universe_data())
        logger.info("Universe is:\n%s", str(universe_df))

    # Disable excessive logging for the following section
    logging.getLogger("tradeexecutor.strategy.pandas_trader.indicator").setLevel(logging.WARNING)

    # Poke create_indicators() if the strategy module defines one
    create_indicators = run_description.runner.create_indicators
    if create_indicators:
        parameters = run_description.runner.parameters
        cycle_duration: CycleDuration = parameters["cycle_duration"]
        clock = datetime.datetime.utcnow()
        strategy_cycle_timestamp = snap_to_previous_tick(
            clock,
            cycle_duration,
        )

        logger.info("Checking create_indicators(), using strategy cycle timestamp %s", strategy_cycle_timestamp)
        calculate_and_load_indicators_inline(
            create_indicators=create_indicators,
            strategy_universe=universe,
            parameters=parameters,
            execution_context=execution_context,
            storage=MemoryIndicatorStorage(universe.get_cache_key()),
            strategy_cycle_timestamp=strategy_cycle_timestamp,
            max_workers=max_workers or get_safe_max_workers_count,
        )
    else:
        logger.info("Strategy module lacks create_indicators()")

    logger.info("All ok")


