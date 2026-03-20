"""trade-ui command.

Interactive TUI for selecting a trading pair and performing a test trade.

Displays the strategy's trading universe with balances, lets the user
pick a pair, amount and trade mode, then executes the test trade.

Quick test run using a test strategy (``--simulate`` launches an Anvil
fork internally, no need to start one manually):

.. code-block:: shell

    trade-executor trade-ui \
        --strategy-file=strategies/test_only/hyper-ai-tui-test.py \
        --private-key=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80 \
        --asset-management-mode=hot_wallet \
        --json-rpc-hyperliquid=$JSON_RPC_HYPERLIQUID \
        --trading-strategy-api-key=$TRADING_STRATEGY_API_KEY \
        --simulate

"""

import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional

from typer import Option
from eth_defi.compat import native_datetime_utc_now
from tradingstrategy.chain import ChainId

from .app import app
from ..bootstrap import (
    prepare_executor_id,
    prepare_cache,
    create_web3_config,
    create_state_store,
    create_execution_and_sync_model,
    create_client,
    configure_default_chain,
)
from ..log import setup_logging
from ..slippage import configure_max_slippage_tolerance
from ..testtrade import make_test_trade
from ..trade_ui_tui import display_pair_selection_ui
from ...ethereum.routing_state import OutOfBalance
from ...ethereum.web3config import SUPPORTED_CHAINS, _get_chain_slug, filter_rpc_kwargs_by_chain
from ...strategy.approval import UncheckedApprovalModel
from ...strategy.bootstrap import make_factory_from_strategy_mod
from ...strategy.description import StrategyExecutionDescription
from ...strategy.execution_context import ExecutionContext, ExecutionMode
from ...strategy.execution_model import AssetManagementMode
from ...strategy.run_state import RunState
from ...strategy.strategy_module import read_strategy_module, StrategyModuleInformation
from ...strategy.trading_strategy_universe import TradingStrategyUniverseModel
from ...utils.timer import timed_task
from tradeexecutor.cli.commands import shared_options


@app.command()
def trade_ui(
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
    json_rpc_arbitrum_sepolia: Optional[str] = shared_options.json_rpc_arbitrum_sepolia,
    json_rpc_base_sepolia: Optional[str] = shared_options.json_rpc_base_sepolia,
    json_rpc_hyperliquid: Optional[str] = shared_options.json_rpc_hyperliquid,
    json_rpc_hyperliquid_testnet: Optional[str] = shared_options.json_rpc_hyperliquid_testnet,
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

    test_short: bool = Option(True, "--test-short/--no-test-short", help="Perform test short trades as well."),
    simulate: bool = shared_options.simulate,
):
    """Interactive test trade selection.

    Browse the strategy's trading universe and interactively select
    a pair, amount and trade mode to perform a test trade.
    """

    id = prepare_executor_id(id, strategy_file)

    logger = setup_logging(log_level=log_level)

    mod: StrategyModuleInformation = read_strategy_module(strategy_file)

    cache_path = prepare_cache(id, cache_path)

    execution_context = ExecutionContext(
        mode=ExecutionMode.preflight_check,
        timed_task_context_manager=timed_task,
        engine_version=mod.trading_strategy_engine_version,
    )

    rpc_kwargs = dict(
        json_rpc_binance=json_rpc_binance,
        json_rpc_polygon=json_rpc_polygon,
        json_rpc_avalanche=json_rpc_avalanche,
        json_rpc_ethereum=json_rpc_ethereum,
        json_rpc_base=json_rpc_base,
        json_rpc_anvil=json_rpc_anvil,
        json_rpc_arbitrum=json_rpc_arbitrum,
        json_rpc_arbitrum_sepolia=json_rpc_arbitrum_sepolia,
        json_rpc_base_sepolia=json_rpc_base_sepolia,
        json_rpc_hyperliquid=json_rpc_hyperliquid,
        json_rpc_hyperliquid_testnet=json_rpc_hyperliquid_testnet,
    )

    # In simulate mode, only fork the chain the strategy needs
    # to avoid launching unnecessary Anvil instances
    if simulate:
        chain_id = mod.get_default_chain_id()
        if chain_id and chain_id in SUPPORTED_CHAINS:
            chain_slug = _get_chain_slug(chain_id)
            logger.info("Simulate mode: restricting Anvil fork to %s", chain_slug)
            rpc_kwargs = filter_rpc_kwargs_by_chain(chain_slug, **rpc_kwargs)

    web3config = create_web3_config(
        **rpc_kwargs,
        unit_testing=unit_testing,
        simulate=simulate,
    )

    if not web3config.has_any_connection():
        raise RuntimeError("trade-ui requires that you pass JSON-RPC connection to one of the networks")

    configure_default_chain(web3config, mod)

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
        if not name:
            name = id
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

    universe_options = mod.get_universe_options()

    universe_model: TradingStrategyUniverseModel = run_description.universe_model
    ts = native_datetime_utc_now()
    universe = universe_model.construct_universe(
        ts,
        ExecutionMode.preflight_check,
        universe_options,
        strategy_parameters=mod.parameters,
        execution_model=execution_model,
    )

    runner = run_description.runner
    routing_state, pricing_model, valuation_method = runner.setup_routing(universe)

    # Sync treasury so balances are up to date for the TUI display
    web3 = web3config.get_default()
    execution_model.initialize()

    balance_updates = sync_model.sync_treasury(
        ts,
        state,
        list(universe.reserve_assets),
        post_valuation=True,
    )

    # Gather balance info for the TUI header
    hot_wallet = sync_model.get_hot_wallet()
    gas_balance = hot_wallet.get_native_currency_balance(web3)

    if len(state.portfolio.reserves) > 0:
        reserve_position = state.portfolio.get_default_reserve_position()
        reserve_balance = float(reserve_position.get_value())
        reserve_symbol = reserve_position.asset.token_symbol
    else:
        reserve_balance = 0.0
        reserve_symbol = universe.get_reserve_asset().token_symbol

    # Detect Hyperliquid chain
    chain_id = ChainId(web3.eth.chain_id)
    is_hyperliquid = chain_id in (ChainId.hyperliquid, ChainId.hyperliquid_testnet)

    # Collect pairs for the TUI
    pairs = list(universe.iterate_pairs())
    assert len(pairs) > 0, "Trading universe has no pairs"

    # Display TUI and get user selections
    if unit_testing:
        # Skip TUI in unit test mode — auto-select the first pair
        selected_pair = pairs[0]
        amount = Decimal("1.0")
        trade_mode = "open" if is_hyperliquid else "open_close"
        logger.info("Unit testing mode: auto-selected pair %s, amount %s, mode %s", selected_pair, amount, trade_mode)
    else:
        selected_pair, amount, trade_mode = display_pair_selection_ui(
            pairs=pairs,
            strategy_universe=universe,
            reserve_balance=reserve_balance,
            reserve_symbol=reserve_symbol,
            gas_balance=float(gas_balance),
            state=state,
            is_hyperliquid=is_hyperliquid,
            pricing_model=pricing_model,
        )

    # Map trade mode to make_test_trade parameters
    buy_only = trade_mode == "open"
    close_only = trade_mode == "close"

    if simulate:
        logger.info("Simulating test trade")
    else:
        logger.info("Performing real test trade")

    logger.info("Selected pair: %s, amount: %s, mode: %s", selected_pair, amount, trade_mode)

    if unit_testing:
        logger.info("Unit testing mode: skipping actual trade execution")
        return

    try:
        make_test_trade(
            web3,
            execution_model,
            pricing_model,
            sync_model,
            state,
            universe,
            runner.routing_model,
            routing_state,
            max_slippage=max_slippage,
            pair=selected_pair,
            buy_only=buy_only,
            close_only=close_only,
            test_short=test_short,
            amount=amount,
        )
    except OutOfBalance as e:
        raise RuntimeError(
            "Failed to do a test trade, as we ran out of balance. "
            "Make sure vault has enough stablecoins deposited for the test trades."
        ) from e

    if not simulate:
        logger.info("Storing the state data from test trades")
        store.sync(state)
    else:
        logger.info("Simulation, no state changes")

    logger.info("All ok")
