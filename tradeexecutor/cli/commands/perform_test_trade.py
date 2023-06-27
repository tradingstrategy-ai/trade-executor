"""perform-test-trade command"""

import datetime
import logging
from pathlib import Path
from typing import Optional
import re
import typer

from .app import app
from ..bootstrap import prepare_executor_id, prepare_cache, create_web3_config, create_state_store, \
    create_execution_and_sync_model, create_client
from ..log import setup_logging
from ..testtrade import make_test_trade
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
from tradingstrategy.chain import ChainId


@app.command()
def perform_test_trade(
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

    # for multipair strategies
    pair: Optional[str] = shared_options.pair,
):
    """Perform a small test swap.

    Tests that the private wallet and the exchange can trade by making 1 USD trade using
    the routing configuration from the strategy.

    The trade will be recorded on the state as a position.
    """

    if pair:
        pair = parse_pair_data(pair)

    id = prepare_executor_id(id, strategy_file)

    logger = setup_logging(log_level=log_level)

    mod: StrategyModuleInformation = read_strategy_module(strategy_file)

    cache_path = prepare_cache(id, cache_path)

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

    if store.is_pristine():
        assert name, "Strategy state file has not been createad. You must pass strategy name to create."
        state = store.create(name)
    else:
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
        UniverseOptions())

    runner = run_description.runner
    routing_state, pricing_model, valuation_method = runner.setup_routing(universe)

    make_test_trade(
        web3config.get_default(),
        execution_model,
        pricing_model,
        sync_model,
        state,
        universe,
        runner.routing_model,
        routing_state,
        pair=pair,
    )

    # Store the test trade data in the strategy history
    store.sync(state)

    logger.info("All ok")


def parse_pair_data(s: str):
    """Extract pair data from string.
    
    :param s:
        String in the format of: [(chain_id, exchange_slug, base_token, quote_token, fee)])], 
        
        where rate is optional.

    :raises ValueError:
        If the string is not in the correct format.
    
    :return:
        Tuple of (chain_id, exchange_slug, base_token, quote_token, fee)"""
    
    try: 
        # Extract the tuple
        tuple_str = re.search(r'\((.*?)\)', s)[1]

        # Split elements and remove leading/trailing whitespaces
        elements = [e.strip() for e in tuple_str.split(',')]

        assert 4 <= len(elements) <= 5, f'Invalid pair data: {s}. Tuple must have 4 or 5 elements. Must be in the format of: (chain_id, exchange_slug, base_token, quote_token, fee), where fee is optional'

        # Process elements
        chain_id = getattr(ChainId, elements[0].split('.')[-1])
        exchange_slug = elements[1].strip('"')
        base_token = elements[2].strip('"')
        quote_token = elements[3].strip('"')
        fee = float(elements[4]) if len(elements) > 4 else None

    except:
        raise ValueError(f'Invalid pair data: {s}. Tuple must be in the format of: (chain_id, exchange_slug, base_token, quote_token, fee), where fee is optional')

    return (chain_id, exchange_slug, base_token, quote_token, fee)