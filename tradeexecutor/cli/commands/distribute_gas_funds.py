"""distribute-gas-funds command.

Distribute native gas tokens to hot wallets across all chains
in the strategy's trading universe using LI.FI cross-chain bridging.
"""

import datetime
import logging
import secrets
from decimal import Decimal
from pathlib import Path
from typing import Optional

from typer import Option

from eth_defi.chain import get_chain_name
from eth_defi.hotwallet import HotWallet
from eth_defi.lifi.constants import DEFAULT_MIN_GAS_USD, DEFAULT_TOP_UP_GAS_USD, LIFI_NATIVE_TOKEN_ADDRESS
from eth_defi.lifi.crosschain import prepare_crosschain_swaps, execute_crosschain_swaps
from eth_defi.lifi.top_up import (
    TopUpConfig,
    display_balances,
    display_proposed_swaps,
    display_results,
    verify_and_display_final,
)
from eth_defi.token import USDC_NATIVE_TOKEN

from tradingstrategy.chain import ChainId

from .app import app
from .shared_options import unit_testing
from . import shared_options
from ..bootstrap import prepare_executor_id, prepare_cache, create_web3_config, create_client, create_execution_and_sync_model, configure_default_chain
from ..log import setup_logging
from ...strategy.approval import UncheckedApprovalModel
from ...strategy.bootstrap import make_factory_from_strategy_mod
from ...strategy.description import StrategyExecutionDescription
from ...strategy.execution_context import ExecutionContext, ExecutionMode
from ...strategy.execution_model import AssetManagementMode
from ...strategy.run_state import RunState
from ...strategy.strategy_module import StrategyModuleInformation, read_strategy_module
from ...strategy.trading_strategy_universe import TradingStrategyUniverseModel
from ...utils.timer import timed_task

from eth_defi.compat import native_datetime_utc_now


logger = logging.getLogger(__name__)


@app.command()
@shared_options.with_json_rpc_options()
def distribute_gas_funds(
    id: str = shared_options.id,
    strategy_file: Path = shared_options.strategy_file,
    trading_strategy_api_key: str = shared_options.trading_strategy_api_key,
    cache_path: Optional[Path] = shared_options.cache_path,
    log_level: str = shared_options.log_level,
    private_key: str = shared_options.private_key,

    rpc_kwargs: dict | None = None,

    asset_management_mode: AssetManagementMode = shared_options.asset_management_mode,
    vault_address: Optional[str] = shared_options.vault_address,
    vault_adapter_address: Optional[str] = shared_options.vault_adapter_address,
    vault_payment_forwarder_address: Optional[str] = shared_options.vault_payment_forwarder,

    min_gas_usd: float = Option(
        float(DEFAULT_MIN_GAS_USD),
        envvar="MIN_GAS_USD",
        help="Minimum gas balance in USD per chain. Chains below this threshold are topped up.",
    ),
    top_up_gas_usd: float = Option(
        float(DEFAULT_TOP_UP_GAS_USD),
        envvar="TOP_UP_GAS_USD",
        help="Amount to bridge in USD when topping up a chain.",
    ),
    dry_run: bool = Option(
        False,
        envvar="DRY_RUN",
        help="Only display balances and proposed swaps without executing.",
    ),
    source_token: str = Option(
        "native",
        envvar="SOURCE_TOKEN",
        help="Source token to bridge from: 'native' (gas token) or 'usdc'.",
    ),
):
    """Distribute native gas tokens to hot wallets on all strategy universe chains.

    Reads the strategy's trading universe to discover which chains are used,
    determines the source chain from the reserve asset, and bridges gas tokens
    from the source chain to any target chain running low via LI.FI.

    The source chain is automatically detected from the reserve asset's chain.
    Target chains are all other chains in the universe.
    """

    global logger

    if private_key is None:
        raise ValueError("PRIVATE_KEY is required for distribute-gas-funds")

    id = prepare_executor_id(id, strategy_file)
    logger = setup_logging(log_level)

    logger.info("Loading strategy file %s", strategy_file)
    mod: StrategyModuleInformation = read_strategy_module(strategy_file)
    cache_path = prepare_cache(id, cache_path)

    execution_context = ExecutionContext(
        mode=ExecutionMode.preflight_check,
        timed_task_context_manager=timed_task,
        engine_version=mod.trading_strategy_engine_version,
    )

    web3config = create_web3_config(
        unit_testing=unit_testing,
        **rpc_kwargs,
    )

    assert web3config.has_any_connection(), \
        "At least one JSON-RPC connection is required. Set JSON_RPC_* environment variables."

    # Set the default chain (used by create_execution_and_sync_model)
    # but do NOT call choose_single_chain() — we need all connections.
    configure_default_chain(web3config, mod)

    if asset_management_mode is None:
        asset_management_mode = AssetManagementMode.hot_wallet

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
            confirmation_block_count=0,
            confirmation_timeout=datetime.timedelta(seconds=60),
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
        test_evm_uniswap_v2_factory=None,
        test_evm_uniswap_v2_router=None,
        test_evm_uniswap_v2_init_code_hash=None,
    )

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

    # Determine source chain from reserve asset
    reserve_asset = universe.get_reserve_asset()
    source_chain_id = reserve_asset.chain_id
    source_chain_name = get_chain_name(source_chain_id)

    # Determine target chains (all universe chains except source)
    universe_chains = universe.data_universe.chains
    target_chain_ids = [
        c.value for c in universe_chains
        if c.value != source_chain_id
    ]

    if not target_chain_ids:
        print(f"Only one chain in universe ({source_chain_name}). Nothing to distribute.")
        return

    logger.info(
        "Source chain: %s (ID: %s), target chains: %s",
        source_chain_name,
        source_chain_id,
        [get_chain_name(c) for c in target_chain_ids],
    )

    # Resolve source token address
    source_token_choice = source_token.lower().strip()
    if source_token_choice == "native":
        source_token_address = LIFI_NATIVE_TOKEN_ADDRESS
    elif source_token_choice == "usdc":
        usdc_address = USDC_NATIVE_TOKEN.get(source_chain_id)
        if not usdc_address:
            raise ValueError(
                f"No USDC address configured for source chain {source_chain_name} "
                f"(chain_id={source_chain_id}). "
                f"Supported chains: {list(USDC_NATIVE_TOKEN.keys())}"
            )
        source_token_address = usdc_address
    else:
        raise ValueError(f"Unknown --source-token value: {source_token_choice!r}. Use 'native' or 'usdc'.")

    # Build Web3 connections
    source_web3 = web3config.get_connection(ChainId(source_chain_id))
    target_web3s = {}
    for chain_id in target_chain_ids:
        target_web3s[chain_id] = web3config.get_connection(ChainId(chain_id))

    # Create hot wallet
    wallet = HotWallet.from_private_key(private_key)

    # Build TopUpConfig for display functions
    config = TopUpConfig(
        private_key=private_key,
        source_chain_id=source_chain_id,
        source_chain_name=source_chain_name,
        target_chain_ids=target_chain_ids,
        min_gas_usd=Decimal(str(min_gas_usd)),
        top_up_usd=Decimal(str(top_up_gas_usd)),
        dry_run=dry_run,
        source_token_choice=source_token_choice,
        source_token_address=source_token_address,
    )

    # Display current balances
    display_balances(source_web3, target_web3s, wallet, config)

    # Prepare swaps
    print(f"\nPreparing cross-chain swaps (source token: {source_token_choice})...")
    swaps = prepare_crosschain_swaps(
        wallet=wallet,
        source_web3=source_web3,
        target_web3s=target_web3s,
        min_gas_usd=config.min_gas_usd,
        top_up_usd=config.top_up_usd,
        source_token_address=source_token_address,
    )

    if not swaps:
        print("\nAll chains have sufficient gas. Nothing to do.")
        return

    display_proposed_swaps(swaps)

    if dry_run:
        print("\nDry run mode - not executing any swaps.")
        return

    # Ask for confirmation
    print()
    response = input("Execute these swaps? [y/N]: ").strip().lower()
    if response != "y":
        print("Aborted.")
        return

    # Execute swaps
    print("\nExecuting swaps...")
    wallet.sync_nonce(source_web3)

    results = execute_crosschain_swaps(
        wallet=wallet,
        source_web3=source_web3,
        swaps=swaps,
    )

    display_results(results)

    # Verify and display final balances
    verify_and_display_final(results, source_web3, target_web3s, wallet, config)
