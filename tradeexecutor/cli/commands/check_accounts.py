import sys
from pathlib import Path
from typing import Optional

import typer
import pandas as pd
from eth_defi.compat import native_datetime_utc_now
from eth_defi.hotwallet import HotWallet
from tabulate import tabulate

from tradeexecutor.exchange_account.lighter import (
    create_lighter_account_value_func,
    get_lighter_trade_hashes,
    is_lighter_exchange_account_pair,
    validate_lighter_exchange_account_pairs,
)
from tradeexecutor.exchange_account.lighter_cleanup import check_lighter_vault_accounts
from tradeexecutor.ethereum.lighter.transfer_verification import reconcile_verified_lighter_transfers
from tradeexecutor.exchange_account.lighter_operator import load_lighter_signers
from tradeexecutor.strategy.account_correction import check_accounts as _check_accounts
from tradeexecutor.cli.account_check import configure_account_check, pending_account_trades, pending_account_assets, check_pool_accounts

from ...ethereum.lagoon.vault import LagoonVaultSyncModel
from ...state.state import UncleanState
from ...state.trade import TradeExecution
from ...strategy.bootstrap import make_factory_from_strategy_mod
from ...strategy.description import StrategyExecutionDescription
from ...strategy.execution_context import ExecutionContext, ExecutionMode
from ...strategy.execution_model import AssetManagementMode
from ...strategy.run_state import RunState
from ...strategy.strategy_module import read_strategy_module
from ...strategy.trading_strategy_universe import TradingStrategyUniverseModel
from ...strategy.universe_model import UniverseOptions
from ..bootstrap import (
    create_client,
    create_state_store,
    create_sync_model,
    create_web3_config,
    prepare_executor_id,
    configure_default_chain,
    resolve_deployment_file,
)
from ..log import setup_logging
from . import shared_options
from .app import app


#: Logged, or raised with ``--raise-on-unclean``, when the books differ from the chain.
#: ``scripts/lagoon/manual-lighter-subaccounts-cli.sh`` requires this message, not only a
#: non-zero exit, before it accepts that the command reported a change made outside the executor.
ACCOUNTS_DO_NOT_MATCH = "Accounts do not match"


@app.command()
@shared_options.with_json_rpc_options()
def check_accounts(
    id: str = shared_options.id,

    strategy_file: Path = shared_options.strategy_file,
    state_file: Optional[Path] = shared_options.state_file,
    private_key: Optional[str] = shared_options.private_key,
    log_level: str = shared_options.log_level,

    trading_strategy_api_key: str = shared_options.trading_strategy_api_key,
    vault_pro_api_key: str = shared_options.vault_pro_api_key,
    cache_path: Optional[Path] = shared_options.cache_path,

    asset_management_mode: AssetManagementMode = shared_options.asset_management_mode,
    vault_address: Optional[str] = shared_options.vault_address,
    vault_adapter_address: Optional[str] = shared_options.vault_adapter_address,
    vault_deployment_block_number: Optional[int] = shared_options.vault_deployment_block_number,

    rpc_kwargs: dict | None = None,

    # Test functionality
    test_evm_uniswap_v2_router: Optional[str] = shared_options.test_evm_uniswap_v2_router,
    test_evm_uniswap_v2_factory: Optional[str] = shared_options.test_evm_uniswap_v2_factory,
    test_evm_uniswap_v2_init_code_hash: Optional[str] = shared_options.test_evm_uniswap_v2_init_code_hash,
    unit_testing: bool = shared_options.unit_testing,
    raise_on_unclean: bool = typer.Option(False, is_flag=False, envvar="RAISE_ON_UNCLEAN", help="Raise an exception if unclean. Unit test option."),
    lighter_operator_record_file: Path | None = shared_options.lighter_operator_record_file,
    lighter_sub_account_operator_record_files: str | None = shared_options.lighter_sub_account_operator_record_files,
):
    """Check that the state ledger matches onchain and external balances.

    - Print differences between observed balances and expected state balances
    """

    global logger

    id = prepare_executor_id(id, strategy_file)

    logger = setup_logging(log_level)

    web3config = create_web3_config(
        gas_price_method=None,
        unit_testing=unit_testing,
        **rpc_kwargs,
    )

    assert web3config, "No RPC endpoints given. A working JSON-RPC connection is needed for check-wallet"

    # Check that we are connected to the chain strategy assumes
    mod = read_strategy_module(strategy_file)
    configure_default_chain(web3config, mod)

    if private_key is not None:
        hot_wallet = HotWallet.from_private_key(private_key)
    else:
        hot_wallet = None

    web3 = web3config.get_default()

    sync_model = create_sync_model(
        asset_management_mode,
        web3,
        hot_wallet,
        vault_address,
        vault_adapter_address,
    )

    logger.info("RPC details")

    # Check the chain is online
    logger.info(f"  Chain id is {web3.eth.chain_id:,}")
    logger.info(f"  Latest block is {web3.eth.block_number:,}")

    # Check balances
    logger.info("Balance details")
    if hot_wallet is not None:
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
    configure_account_check(sync_model, web3config, resolve_deployment_file(id, state_file), state)
    routed_pending = pending_account_trades(state)
    excluded_assets = pending_account_assets(state, routed_pending)
    verified_transfers = reconcile_verified_lighter_transfers(
        state,
        web3,
        mutate=False,
        signers=load_lighter_signers(lighter_operator_record_file, lighter_sub_account_operator_record_files),
    )
    if verified_transfers:
        logger.info(
            "Found %d verified but unreconciled Lighter transfer(s); run correct-accounts or repair",
            len(verified_transfers),
        )

    # Report pending Lighter transfers instead of refusing the state, so the
    # operator can inspect them; any other unfinished trade still refuses it.
    def is_pending_lighter_transfer(trade: TradeExecution) -> bool:
        return trade.is_exchange_account_transfer_pending() and is_lighter_exchange_account_pair(trade.pair)

    state.check_if_clean(ignore=lambda trade: is_pending_lighter_transfer(trade) or trade in routed_pending)
    for trade in routed_pending:
        logger.error("Pending trade #%d (%s); affected assets unchecked; recover before correction", trade.trade_id, trade.get_status().value)
    for asset in excluded_assets:
        logger.warning("Unchecked pending asset: chain %s %s (%s)", asset.chain_id, asset.token_symbol, asset.address)
    pending = [trade for position in state.portfolio.get_open_and_frozen_positions() for trade in position.trades.values() if is_pending_lighter_transfer(trade)]
    for trade in pending:
        logger.error(
            "Lighter transfer trade #%d on %s is pending (%s), %s; run correct-accounts or repair",
            trade.trade_id,
            trade.pair.get_ticker(),
            trade.get_status().value,
            ", ".join(get_lighter_trade_hashes(trade)) or "no recorded hash",
        )

    client, routing_model = create_client(
        mod=mod,
        web3config=web3config,
        trading_strategy_api_key=trading_strategy_api_key,
        vault_pro_api_key=vault_pro_api_key,
        cache_path=cache_path,
        test_evm_uniswap_v2_factory=test_evm_uniswap_v2_factory,
        test_evm_uniswap_v2_router=test_evm_uniswap_v2_router,
        test_evm_uniswap_v2_init_code_hash=test_evm_uniswap_v2_init_code_hash,
        clear_caches=False,
    )
    execution_context = ExecutionContext(mode=ExecutionMode.one_off)

    strategy_factory = make_factory_from_strategy_mod(mod)

    run_description: StrategyExecutionDescription = strategy_factory(
        execution_model=None,
        execution_context=execution_context,
        sync_model=None,
        valuation_model_factory=None,
        pricing_model_factory=None,
        approval_model=None,
        client=client,
        run_state=RunState(),
        timed_task_context_manager=execution_context.timed_task_context_manager,
    )

    universe_model: TradingStrategyUniverseModel = run_description.universe_model
    universe = universe_model.construct_universe(
        native_datetime_utc_now(),
        execution_context.mode,
        UniverseOptions(history_period=mod.get_live_trading_history_period()),
        strategy_parameters=mod.parameters,
        state=state,
    )

    logger.info("Universe contains %d pairs", universe.data_universe.pairs.get_count())
    logger.info("Reserve assets are: %s", universe.reserve_assets)
    configure_account_check(sync_model, web3config, resolve_deployment_file(id, state_file), state, universe)
    logger.info("Coverage: recorded holdings, configured cash and small-universe tokens; arbitrary untracked tokens are not discovered")

    assert len(universe.reserve_assets) == 1, "Need exactly one reserve asset"

    # Set initial reserves,
    # in order to run the tests
    # TODO: Have this / treasury sync as a separate CLI command later
    if unit_testing:
        if len(state.portfolio.reserves) == 0:
            logger.info("Initialising reserves for the unit test: %s", universe.reserve_assets[0])
            state.portfolio.initialise_reserves(universe.reserve_assets[0])

    lighter_pairs = validate_lighter_exchange_account_pairs(
        position.pair
        for position in state.portfolio.get_open_and_frozen_positions()
    )
    exchange_account_value_func = (
        create_lighter_account_value_func()
        if lighter_pairs
        else None
    )

    clean, df = _check_accounts(
        universe.data_universe.pairs,
        universe.reserve_assets,
        state,
        sync_model,
        exchange_account_value_func=exchange_account_value_func,
        check_clean=not (pending or routed_pending),
        excluded_assets=excluded_assets,
    )
    pool_clean, pool_df = check_pool_accounts(state, sync_model, universe)
    # API observations have no EVM block. Keep integer block/chain identifiers
    # as objects so the missing API cells do not display heights as floats.
    df = pd.concat([df.astype(object), pool_df.astype(object)], ignore_index=False)
    df = df.where(pd.notna(df), "")
    clean = clean and not pending and not routed_pending and pool_clean

    # Every Lighter account a Lagoon Safe owns must be declared and tracked, or NAV misses it.
    if isinstance(sync_model, LagoonVaultSyncModel) and check_lighter_vault_accounts(sync_model.vault.safe_address, state, universe):
        clean = False

    output = tabulate(
        df,
        headers='keys',
        tablefmt='rounded_outline',
        disable_numparse=True,
    )

    if clean:
        logger.info("All accounts match:\n%s", output)
        if not raise_on_unclean:
            sys.exit(0)
        else:
            logger.info("Unit test exit - nothing to be do ne")
    else:
        if raise_on_unclean:
            raise UncleanState(ACCOUNTS_DO_NOT_MATCH + ":\n%s", output)
        else:
            logger.warning(ACCOUNTS_DO_NOT_MATCH + ":\n%s", output)
            sys.exit(1)
