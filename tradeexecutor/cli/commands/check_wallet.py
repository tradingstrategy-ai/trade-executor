"""check-wallet command"""

import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from web3 import Web3

from eth_defi.balances import fetch_erc20_balances_by_token_list
from eth_defi.gas import GasPriceMethod
from eth_defi.hotwallet import HotWallet
from eth_defi.token import fetch_erc20_details
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from . import shared_options
from .app import app
from ..bootstrap import prepare_executor_id, prepare_cache, create_web3_config, create_execution_and_sync_model
from ..log import setup_logging
from ...ethereum.enzyme.vault import EnzymeVaultSyncModel
from ...strategy.approval import UncheckedApprovalModel
from ...strategy.bootstrap import make_factory_from_strategy_mod
from ...strategy.description import StrategyExecutionDescription
from ...strategy.execution_context import ExecutionContext, ExecutionMode, standalone_backtest_execution_context
from ...strategy.execution_model import AssetManagementMode
from ...strategy.run_state import RunState
from ...strategy.strategy_module import read_strategy_module
from ...strategy.trading_strategy_universe import TradingStrategyUniverseModel
from ...strategy.universe_model import UniverseOptions
from ...utils.fullname import get_object_full_name
from ...utils.timer import timed_task


@app.command()
def check_wallet(
    id: str = shared_options.id,

    strategy_file: Path = shared_options.strategy_file,
    private_key: str = shared_options.private_key,
    trading_strategy_api_key: str = shared_options.trading_strategy_api_key,
    cache_path: Optional[Path] = shared_options.cache_path,

    # Get minimum gas balance from the env
    min_gas_balance: Optional[float] = shared_options.min_gas_balance,

    # Live trading or backtest
    asset_management_mode: AssetManagementMode = shared_options.asset_management_mode,
    vault_address: Optional[str] = shared_options.vault_address,
    vault_adapter_address: Optional[str] = shared_options.vault_adapter_address,
    vault_payment_forwarder_address: Optional[str] = shared_options.vault_payment_forwarder,

    # Web3 connection options
    json_rpc_binance: Optional[str] = shared_options.json_rpc_binance,
    json_rpc_polygon: Optional[str] = shared_options.json_rpc_polygon,
    json_rpc_ethereum: Optional[str] = shared_options.json_rpc_ethereum,
    json_rpc_avalanche: Optional[str] = shared_options.json_rpc_avalanche,
    json_rpc_arbitrum: Optional[str] = shared_options.json_rpc_arbitrum,
    json_rpc_anvil: Optional[str] = shared_options.json_rpc_anvil,

    log_level: Optional[str] = shared_options.log_level,

    # Debugging and unit testing
    unit_testing: bool = shared_options.unit_testing,
    unit_test_force_anvil: bool = typer.Option(bool, envvar="UNIT_TEST_FORCE_ANVIL", help="Use Anvil backend regardless of what chain strategy module suggests"),
):
    """Print out the token balances of the hot wallet.

    Check that our hot wallet has cash deposits and gas token deposits.
    """

    # To run this from command line with .env file you can do
    # set -o allexport ; source ~/pancake-eth-usd-sma-final.env ; set +o allexport ;  trade-executor check-wallet

    global logger

    id = prepare_executor_id(id, strategy_file)

    logger = setup_logging(log_level)

    mod = read_strategy_module(strategy_file)

    cache_path = prepare_cache(id, cache_path)

    client = Client.create_live_client(
        trading_strategy_api_key,
        cache_path=cache_path,
        settings_path=None,
    )

    web3config = create_web3_config(
        json_rpc_binance=json_rpc_binance,
        json_rpc_polygon=json_rpc_polygon,
        json_rpc_avalanche=json_rpc_avalanche,
        json_rpc_ethereum=json_rpc_ethereum,
        json_rpc_anvil=json_rpc_anvil,
        json_rpc_arbitrum=json_rpc_arbitrum,
        unit_testing=unit_testing,
    )
    assert web3config.has_chain_configured(), "No RPC endpoints given. A working JSON-RPC connection is needed for running this command. Check your JSON-RPC configuration."

    universe = mod.create_trading_universe(
        pd.Timestamp.utcnow(),
        client,
        standalone_backtest_execution_context,
        mod.get_universe_options(),
    )
    execution_context = ExecutionContext(
        mode=ExecutionMode.preflight_check,
        timed_task_context_manager=timed_task,
        engine_version=mod.trading_strategy_engine_version,
    )

    # Check that we are connected to the chain strategy assumes
    if unit_test_force_anvil:
        web3config.set_default_chain(ChainId.anvil)
    else:
        web3config.set_default_chain(mod.get_default_chain_id())

    web3config.check_default_chain_id()

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

    assert asset_management_mode in (AssetManagementMode.hot_wallet, AssetManagementMode.enzyme), F"Cannot perform check wallet for non-real modes"
    assert sync_model, f"sync_model not set up"
    assert sync_model.get_hot_wallet(), f"sync_model {sync_model} lacks hot wallet"

    hot_wallet = HotWallet.from_private_key(private_key)

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
        run_state=RunState(),
    )

    # Get all tokens from the universe
    reserve_assets = universe.reserve_assets
    web3 = web3config.get_default()
    tokens = [Web3.to_checksum_address(a.address) for a in reserve_assets]

    logger.info("RPC details")

    # Check the chain is online
    logger.info(f"  Chain id is {web3.eth.chain_id:,}")
    logger.info(f"  Latest block is {web3.eth.block_number:,}")

    tx_builder = sync_model

    # Check balances
    reserve_address = sync_model.get_vault_address() or sync_model.get_hot_wallet().address
    logger.info("Balance details")
    logger.info("  Hot wallet is %s", sync_model.get_hot_wallet().address)
    gas_balance = web3.eth.get_balance(hot_wallet.address) / 10**18
    if isinstance(sync_model, EnzymeVaultSyncModel):
        logger.info("  Vault address is %s", sync_model.get_vault_address())
    logger.info("  We have %f tokens left for gas", gas_balance)
    logger.info("  The gas error limit is %f tokens", min_gas_balance)

    for asset in reserve_assets:
        logger.info("  Reserve asset: %s (%s)", asset.token_symbol, asset.address)

    balances = fetch_erc20_balances_by_token_list(web3, reserve_address, tokens)

    for address, balance in balances.items():
        details = fetch_erc20_details(web3, address)
        logger.info("  Balance of %s (%s): %s %s", details.name, details.address, details.convert_to_decimals(balance), details.symbol)

    # Check that the routing looks sane
    # E.g. there is no mismatch between strategy reserve token, wallet and pair universe
    runner = run_description.runner
    routing_state, pricing_model, valuation_method = runner.setup_routing(universe)
    routing_model = runner.routing_model

    logger.info("Execution details")
    logger.info("  Execution model is %s", get_object_full_name(execution_model))
    logger.info("  Routing model is %s", get_object_full_name(routing_model))
    logger.info("  Token pricing model is %s", get_object_full_name(pricing_model))
    logger.info("  Position valuation model is %s", get_object_full_name(valuation_method))
    logger.info("  Sync model is %s", get_object_full_name(sync_model))

    # Check we have enough gas
    execution_model.preflight_check()

    # Check our routes
    routing_model.perform_preflight_checks_and_logging(universe.data_universe.pairs)

    web3config.close()

    logger.info("All ok")
