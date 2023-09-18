"""perform-test-trade command"""

import datetime
import logging
from pathlib import Path
from typing import Optional
import re
import typer
from typer import Option

from .app import app
from ..bootstrap import prepare_executor_id, prepare_cache, create_web3_config, create_state_store, \
    create_execution_and_sync_model, create_client, create_generic_execution_and_sync_model, create_generic_client
from ..log import setup_logging
from ..testtrade import make_test_trade
from ...strategy.approval import UncheckedApprovalModel
from ...strategy.bootstrap import make_factory_from_strategy_mod, make_generic_factory_from_strategy_mod
from ...strategy.description import StrategyExecutionDescription
from ...strategy.execution_context import ExecutionContext, ExecutionMode
from ...strategy.execution_model import AssetManagementMode
from ...strategy.run_state import RunState
from ...strategy.strategy_module import read_strategy_module, StrategyModuleInformation
from ...strategy.trading_strategy_universe import TradingStrategyUniverseModel
from ...strategy.universe_model import UniverseOptions
from ...strategy.generic_pricing_model import get_pricing_model_for_pair
from ...utils.timer import timed_task
from tradeexecutor.cli.commands import shared_options
from tradeexecutor.testing.evm_uniswap_testing_data import deserialize_uniswap_test_data_list
from tradingstrategy.chain import ChainId
from tradingstrategy.pair import DEXPair


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
    test_evm_uniswap_data: Optional[str] = shared_options.test_evm_uniswap_data,

    # for multipair strategies
    pair: Optional[str] = shared_options.pair,
    all_pairs: bool = shared_options.all_pairs,

    buy_only: bool = Option(None, "--buy-only", envvar="BUY_ONLY", help="Only perform the buy side of the test trade - leave position open.")
):
    """Perform a small test swap.

    Tests that the private wallet and the exchange can trade by making 1 USD trade using
    the routing configuration from the strategy.

    The trade will be recorded on the state as a position.
    """

    if pair:
        assert not all_pairs, "Cannot specify both --pair and --all-pairs"
        pair = parse_pair_data(pair)

    if test_evm_uniswap_data is not None:
        test_evm_uniswap_data = deserialize_uniswap_test_data_list(test_evm_uniswap_data)

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

    # assert len(mod.trade_routing) == 1, "Test trade only works with single routing strategies for now"

    confirmation_timeout = datetime.timedelta(seconds=60)

    generic_routing_data = None
    if len(mod.trade_routing) > 1:
        generic_routing_data, sync_model = create_generic_execution_and_sync_model(
                asset_management_mode=asset_management_mode,
                private_key=private_key,
                web3config=web3config,
                confirmation_timeout=confirmation_timeout,
                confirmation_block_count=confirmation_block_count,
                max_slippage=max_slippage,
                min_gas_balance=min_gas_balance,
                vault_address=vault_address,
                vault_adapter_address=vault_adapter_address,
                vault_payment_forwarder_address=vault_payment_forwarder_address,
                routing_hints=mod.trade_routing,
                execution_context=execution_context,
                reserve_currency=mod.reserve_currency,
            )
    else:
        execution_model, sync_model, valuation_model_factory, pricing_model_factory = \
            create_execution_and_sync_model(
                asset_management_mode=asset_management_mode,
                private_key=private_key,
                web3config=web3config,
                confirmation_timeout=confirmation_timeout,
                confirmation_block_count=confirmation_block_count,
                min_gas_balance=min_gas_balance,
                max_slippage=max_slippage,
                vault_address=vault_address,
                vault_adapter_address=vault_adapter_address,
                vault_payment_forwarder_address=vault_payment_forwarder_address,
                routing_hint=mod.trade_routing[0],
            )

    clear_caches = False
    if generic_routing_data:
        assert test_evm_uniswap_data, "test_evm_uniswap_data is needed for generic routing data"

        client = create_generic_client(
            mod=mod,
            web3config=web3config,
            trading_strategy_api_key=trading_strategy_api_key,
            cache_path=cache_path,
            test_evm_uniswap_data=test_evm_uniswap_data,
            generic_routing_data=generic_routing_data,
            clear_caches=clear_caches,
        )

    else:
        client, routing_model = create_client(
            mod=mod,
            web3config=web3config,
            trading_strategy_api_key=trading_strategy_api_key,
            cache_path=cache_path,
            test_evm_uniswap_v2_factory=test_evm_uniswap_v2_factory,
            test_evm_uniswap_v2_router=test_evm_uniswap_v2_router,
            test_evm_uniswap_v2_init_code_hash=test_evm_uniswap_v2_init_code_hash,
            clear_caches=clear_caches,
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
    if generic_routing_data:
        generic_factory = make_generic_factory_from_strategy_mod(mod)
        run_description = generic_factory(
            execution_context=execution_context,
            timed_task_context_manager=execution_context.timed_task_context_manager,
            sync_model=sync_model,
            approval_model=UncheckedApprovalModel(),
            client=client,
            run_state=RunState(),
            generic_routing_data=generic_routing_data,
        )
    else:
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

    if generic_routing_data:
        generic_execution_data = runner.setup_generic_routing(universe)
        pricing_models = [item["pricing_model"] for item in generic_execution_data]

        for _pair in universe.universe.pairs.iterate_pairs():

            pricing_model_to_compare = get_pricing_model_for_pair(_pair, pricing_models, set_routing_hint=False) 

            # for item in generic_routing_data:
            #     routing_model = item["routing_model"]
            #     execution_model = item["execution_model"]
            
            routing_state, routing_model, execution_model, pricing_model = None, None, None, None
            
            for item in generic_execution_data:
                if item["pricing_model"] == pricing_model_to_compare:
                    assert item["routing_model"] == pricing_model_to_compare.routing_model, "routing_model mismatch"
                    routing_state = item["routing_state"]
                    routing_model = item["routing_model"]
                    pricing_model = item["pricing_model"]
                    break

            assert routing_state, "routing_state not found"

            for my_item in generic_routing_data:
                if my_item["routing_model"] == routing_model:
                    execution_model = my_item["execution_model"]
                    break

            assert execution_model, "execution_model not found"

            p = _get_pair_identifier_from_dex_pair(_pair)

            if p == pair or all_pairs:
                make_test_trade(
                    web3config.get_default(),
                    execution_model,
                    pricing_model,
                    sync_model,
                    state,
                    universe,
                    routing_model,
                    routing_state,
                    pair=p,
                    buy_only=buy_only,
                )

    else:

        routing_state, pricing_model, valuation_method = runner.setup_routing(universe)

        if all_pairs:

            for _pair in universe.universe.pairs.iterate_pairs():

                p = _get_pair_identifier_from_dex_pair(_pair)

                make_test_trade(
                    web3config.get_default(),
                    execution_model,
                    pricing_model,
                    sync_model,
                    state,
                    universe,
                    runner.routing_model,
                    routing_state,
                    pair=p,
                    buy_only=buy_only,
                )
        else:

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
                buy_only=buy_only,
            )

    # Store the test trade data in the strategy history
    store.sync(state)

    logger.info("All ok")

def _get_pair_identifier_from_dex_pair(_pair) -> tuple[int, str, str, str, float]:
    """Gets the pair identifier from a DEX pair.
    
    :param _pair:
        DEX pair.
    """
    p = [
        _pair.chain_id,
        _pair.exchange_slug,
        _pair.base_token_symbol,
        _pair.quote_token_symbol,
    ]
    if _pair.fee:
        p.append(_pair.fee/10_000)
    return p


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

        if len(elements) not in {4, 5}:
            raise ValueError()

        # Process elements
        chain_id = getattr(ChainId, elements[0].split('.')[-1])
        exchange_slug = elements[1].strip('"')
        base_token = elements[2].strip('"')
        quote_token = elements[3].strip('"')
        fee = float(elements[4]) if len(elements) > 4 else None

    except:
        raise ValueError(f'Invalid pair data: {s}. Tuple must be in the format of: (chain_id, exchange_slug, base_token, quote_token, fee), where fee is optional')

    return (chain_id, exchange_slug, base_token, quote_token, fee)


def get_string_identifier_from_pair(pair: DEXPair) -> str:
    """Construct pair identifier from pair data.
    
    :param pair:
        Pair data as DEXPair.

    :return:
        Pair identifier string."""
    
    assert isinstance(pair, DEXPair), 'Pair must be of type DEXPair'

    return f'({pair.chain_id.name}, "{pair.exchange_slug}", "{pair.base_token_symbol}", "{pair.quote_token_symbol}", {pair.fee/10_000})'