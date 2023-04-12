"""check-wallet command"""

import datetime
from pathlib import Path
from typing import Optional

import typer
from web3 import Web3

from eth_defi.balances import fetch_erc20_balances_by_token_list
from eth_defi.gas import GasPriceMethod
from eth_defi.hotwallet import HotWallet
from eth_defi.token import fetch_erc20_details
from tradingstrategy.client import Client
from . import shared_options
from .app import app
from ..bootstrap import prepare_executor_id, prepare_cache, create_web3_config, create_trade_execution_model
from ..log import setup_logging
from ...strategy.approval import UncheckedApprovalModel
from ...strategy.bootstrap import make_factory_from_strategy_mod
from ...strategy.description import StrategyExecutionDescription
from ...strategy.execution_context import ExecutionContext, ExecutionMode
from ...strategy.execution_model import AssetManagementMode
from ...strategy.run_state import RunState
from ...strategy.strategy_module import read_strategy_module
from ...strategy.trading_strategy_universe import TradingStrategyUniverseModel
from ...strategy.universe_model import UniverseOptions
from ...utils.fullname import get_object_full_name
from ...utils.timer import timed_task


@app.command()
def check_wallet(
    id: str = typer.Option(None, envvar="EXECUTOR_ID", help="Executor id used when programmatically referring to this instance. If not given, take the base of --strategy-file."),

    strategy_file: Path = typer.Option(..., envvar="STRATEGY_FILE"),
    private_key: str = typer.Option(None, envvar="PRIVATE_KEY"),
    trading_strategy_api_key: str = typer.Option(None, envvar="TRADING_STRATEGY_API_KEY", help="Trading Strategy API key"),
    cache_path: Optional[Path] = typer.Option("cache/", envvar="CACHE_PATH", help="Where to store downloaded datasets"),

    # Get minimum gas balance from the env
    minimum_gas_balance: Optional[float] = typer.Option(0.1, envvar="MINUMUM_GAS_BALANCE", help="What is the minimum balance of gas token you need to have in your wallet. If the balance falls below this, abort by crashing and do not attempt to create transactions. Expressed in the native token e.g. ETH."),

    # Live trading or backtest
    asset_management_mode: AssetManagementMode = shared_options.asset_management_mode,
    vault_address: Optional[str] = shared_options.vault_address,
    vault_adapter_address: Optional[str] = shared_options.vault_adapter_address,


    # Web3 connection options
    json_rpc_binance: Optional[str] = shared_options.json_rpc_binance,
    json_rpc_polygon: Optional[str] = shared_options.json_rpc_polygon,
    json_rpc_ethereum: Optional[str] = shared_options.json_rpc_ethereum,
    json_rpc_avalanche: Optional[str] = shared_options.json_rpc_avalanche,
    json_rpc_arbitrum: Optional[str] = shared_options.json_rpc_arbitrum,
    json_rpc_anvil: Optional[str] = shared_options.json_rpc_anvil,

    log_level: str = typer.Option(None, envvar="LOG_LEVEL", help="The Python default logging level. The defaults are 'info' is live execution, 'warning' if backtesting. Set 'disabled' in testing."),
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

    client = Client.create_live_client(trading_strategy_api_key, cache_path=cache_path)

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
    assert web3config, "No RPC endpoints given. A working JSON-RPC connection is needed for check-wallet"

    # Check that we are connected to the chain strategy assumes
    web3config.set_default_chain(mod.chain_id)
    web3config.check_default_chain_id()

    execution_model, sync_model, valuation_model_factory, pricing_model_factory = create_trade_execution_model(
        asset_management_mode=AssetManagementMode.hot_wallet,
        private_key=private_key,
        web3config=web3config,
        confirmation_timeout=datetime.timedelta(seconds=60),
        confirmation_block_count=6,
        max_slippage=0.01,
        min_balance_threshold=minimum_gas_balance,
        vault_address=vault_address,
        vault_adapter_address=vault_adapter_address,
    )

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

    # We construct the trading universe to know what's our reserve asset
    universe_model: TradingStrategyUniverseModel = run_description.universe_model
    ts = datetime.datetime.utcnow()
    universe = universe_model.construct_universe(
        ts,
        ExecutionMode.preflight_check,
        UniverseOptions())

    # Get all tokens from the universe
    reserve_assets = universe.reserve_assets
    web3 = web3config.get_default()
    tokens = [Web3.to_checksum_address(a.address) for a in reserve_assets]

    logger.info("RPC details")

    # Check the chain is online
    logger.info(f"  Chain id is {web3.eth.chain_id:,}")
    logger.info(f"  Latest block is {web3.eth.block_number:,}")

    # Check balances
    logger.info("Balance details")
    logger.info("  Hot wallet is %s", hot_wallet.address)
    gas_balance = web3.eth.get_balance(hot_wallet.address) / 10**18
    logger.info("  We have %f tokens for gas left", gas_balance)
    logger.info("  The gas error limit is %f tokens", minimum_gas_balance)
    balances = fetch_erc20_balances_by_token_list(web3, hot_wallet.address, tokens)

    for asset in reserve_assets:
        logger.info("Reserve asset: %s", asset.token_symbol)

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

    # Check we have enough gas
    execution_model.preflight_check()

    # Check our routes
    routing_model.perform_preflight_checks_and_logging(universe.universe.pairs)

    web3config.close()


