"""perform-test-trade command"""

import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional
from typer import Option

from .app import app
from .pair_mapping import parse_pair_data, construct_identifier_from_pair
from ..bootstrap import prepare_executor_id, prepare_cache, create_web3_config, create_state_store, \
    create_execution_and_sync_model, create_client
from ..log import setup_logging
from ..slippage import configure_max_slippage_tolerance
from ..testtrade import make_test_trade
from ...ethereum.routing_state import OutOfBalance
from ...ethereum.velvet.execution import VelvetExecution
from ...ethereum.velvet.vault import VelvetVaultSyncModel
from ...ethereum.velvet.velvet_enso_routing import VelvetEnsoRouting
from ...strategy.approval import UncheckedApprovalModel
from ...strategy.bootstrap import make_factory_from_strategy_mod
from ...strategy.description import StrategyExecutionDescription
from ...strategy.execution_context import ExecutionContext, ExecutionMode
from ...strategy.execution_model import AssetManagementMode
from ...strategy.generic.generic_pricing_model import GenericPricing
from ...strategy.run_state import RunState
from ...strategy.strategy_module import read_strategy_module, StrategyModuleInformation
from ...strategy.trading_strategy_universe import TradingStrategyUniverseModel
from ...utils.timer import timed_task
from tradeexecutor.cli.commands import shared_options


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
    confirmation_timeout: int = shared_options.confirmation_timeout,

    # Test EVM backend when running e2e tests
    test_evm_uniswap_v2_router: Optional[str] = shared_options.test_evm_uniswap_v2_router,
    test_evm_uniswap_v2_factory: Optional[str] = shared_options.test_evm_uniswap_v2_factory,
    test_evm_uniswap_v2_init_code_hash: Optional[str] = shared_options.test_evm_uniswap_v2_init_code_hash,
    unit_testing: bool = shared_options.unit_testing,

    # for multipair strategies
    pair: Optional[str] = shared_options.pair,
    all_pairs: bool = shared_options.all_pairs,

    buy_only: bool = Option(False, "--buy-only/--no-buy-only", help="Only perform the buy side of the test trade - leave position open."),
    test_short: bool = Option(True, "--test-short/--no-test-short", help="Perform test short trades as well."),
    test_credit_supply: bool = Option(True, "--test-credit-supply/--no-test-credit-supply", help="Perform test credit supply trades as well."),
    amount: float = Option(1.0, envvar="AMOUNT", help="The USD value of the test trade"),
    simulate: bool = shared_options.simulate,
):
    """Perform a small test swap.

    Tests that the private wallet and the exchange can trade by making 1 USD trade using
    the routing configuration from the strategy.

    The trade will be recorded on the state as a position.
    """

    if pair:
        assert not all_pairs, "Cannot specify both --pair and --all-pairs"
        pair = parse_pair_data(pair)

    id = prepare_executor_id(id, strategy_file)

    logger = setup_logging(log_level=log_level)

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
        simulate=simulate,
    )

    if not web3config.has_any_connection():
        raise RuntimeError("perform-test-trade requires that you pass JSON-RPC connection to one of the networks")

    web3config.choose_single_chain()

    max_slippage = configure_max_slippage_tolerance(max_slippage, mod)

    execution_model, sync_model, valuation_model_factory, pricing_model_factory = create_execution_and_sync_model(
        asset_management_mode=asset_management_mode,
        private_key=private_key,
        web3config=web3config,
        confirmation_timeout=datetime.timedelta(seconds=confirmation_timeout),
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
        asset_management_mode=asset_management_mode,
    )
    assert client is not None, "You need to give details for TradingStrategy.ai client"

    if not state_file:
        state_file = f"state/{id}.json"

    store = create_state_store(Path(state_file))

    if store.is_pristine():
        assert name, "Strategy state file has not been created. You must pass strategy name to create."
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
        universe_options
    )

    runner = run_description.runner
    routing_state, pricing_model, valuation_method = runner.setup_routing(universe)

    if simulate:
        logger.info("Simulating test trades")
    else:
        logger.info("Performing real trades")

    # Trip wire for Velvet integration, as Velvet needs its special Enso path
    if asset_management_mode == AssetManagementMode.velvet:
        assert isinstance(execution_model, VelvetExecution), f"Got: {execution_model}"
        assert isinstance(sync_model, VelvetVaultSyncModel), f"Got: {sync_model}"
        assert isinstance(pricing_model, GenericPricing), f"Got: {pricing_model}"
        # TODO: Clean up this parameter passing in some point
        assert isinstance(runner.routing_model, VelvetEnsoRouting), f"Got: {runner.routing_model}"
        assert routing_model is None, f"Got: {routing_model}"

    try:
        if all_pairs:
            logger.info("Testing all pairs, we have %d pairs", universe.get_pair_count())

            if not simulate:
                assert universe.get_pair_count() < 10, "Too many pairs to test"

            for pair in universe.data_universe.pairs.iterate_pairs():

                logger.info("Making test trade for %s", pair)

                _p = construct_identifier_from_pair(pair)
                p = parse_pair_data(_p)

                make_test_trade(
                    web3config.get_default(),
                    execution_model,
                    pricing_model,
                    sync_model,
                    state,
                    universe,
                    runner.routing_model,
                    routing_state,
                    max_slippage=max_slippage,
                    pair=p,
                    buy_only=buy_only,
                    test_short=test_short,
                    test_credit_supply=test_credit_supply,
                    amount=Decimal(amount),
                )
        else:

            logger.info("Single pair test trade for %s", pair)
            make_test_trade(
                web3config.get_default(),
                execution_model,
                pricing_model,
                sync_model,
                state,
                universe,
                runner.routing_model,
                routing_state,
                max_slippage=max_slippage,
                pair=pair,
                buy_only=buy_only,
                test_short=test_short,
                test_credit_supply=test_credit_supply,
                amount=Decimal(amount),
            )
    except OutOfBalance as e:
        raise RuntimeError(f"Failed to a test trade, as we run out of balance. Make sure vault has enough stablecoins deposited for the test trades.") from e

    # Store the test trade data in the strategy history,
    # but only if we did not run the simulation
    if not simulate:
        logger.info("Storing the state data from test trades")
        store.sync(state)
    else:
        logger.info("Simulation, no state changes")

    logger.info("All ok")


