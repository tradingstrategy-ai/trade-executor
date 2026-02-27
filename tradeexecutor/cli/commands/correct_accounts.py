"""Correct accounting errors in the internal state.

"""
import datetime
import sys
import time
from _decimal import Decimal
from pathlib import Path
from typing import Optional

import typer
from tabulate import tabulate
from typer import Option

from eth_defi.compat import native_datetime_utc_now
from eth_defi.hotwallet import HotWallet
from eth_defi.provider.broken_provider import get_almost_latest_block_number

from tradeexecutor.exchange_account.derive import DeriveNetwork
from tradeexecutor.exchange_account.utils import create_exchange_account_value_func
from tradeexecutor.strategy.account_correction import correct_accounts as _correct_accounts, check_accounts, UnknownTokenPositionFix, check_state_internal_coherence
from .app import app
from ..bootstrap import prepare_executor_id, create_web3_config, create_sync_model, create_client, backup_state, create_execution_and_sync_model
from ..double_position import check_double_position
from ..log import setup_logging
from ...ethereum.enzyme.tx import EnzymeTransactionBuilder
from ...ethereum.enzyme.vault import EnzymeVaultSyncModel
from ...ethereum.hot_wallet_sync_model import HotWalletSyncModel
from ...ethereum.tx import HotWalletTransactionBuilder
from ...state.state import UncleanState
from ...strategy.bootstrap import make_factory_from_strategy_mod
from ...strategy.account_correction import calculate_account_corrections
from ...strategy.description import StrategyExecutionDescription
from ...strategy.execution_context import ExecutionContext, ExecutionMode
from ...strategy.default_routing_options import TradeRouting
from ...strategy.execution_model import AssetManagementMode
from . import shared_options
from ...strategy.run_state import RunState
from ...strategy.strategy_module import StrategyModuleInformation, read_strategy_module
from ...strategy.trading_strategy_universe import TradingStrategyUniverseModel
from ...strategy.universe_model import UniverseOptions
from ...utils.blockchain import get_block_timestamp
from eth_defi.compat import native_datetime_utc_now


@app.command()
def correct_accounts(
    id: str = shared_options.id,

    strategy_file: Path = shared_options.strategy_file,
    state_file: Optional[Path] = shared_options.state_file,
    private_key: Optional[str] = shared_options.private_key,
    log_level: str = shared_options.log_level,

    trading_strategy_api_key: str = shared_options.trading_strategy_api_key,
    cache_path: Optional[Path] = shared_options.cache_path,

    asset_management_mode: AssetManagementMode = shared_options.asset_management_mode,
    vault_address: Optional[str] = shared_options.vault_address,
    vault_adapter_address: Optional[str] = shared_options.vault_adapter_address,
    vault_payment_forwarder: Optional[str] = shared_options.vault_payment_forwarder,
    vault_deployment_block_number: Optional[int] = shared_options.vault_deployment_block_number,


    json_rpc_binance: Optional[str] = shared_options.json_rpc_binance,
    json_rpc_polygon: Optional[str] = shared_options.json_rpc_polygon,
    json_rpc_avalanche: Optional[str] = shared_options.json_rpc_avalanche,
    json_rpc_ethereum: Optional[str] = shared_options.json_rpc_ethereum,
    json_rpc_base: Optional[str] = shared_options.json_rpc_base,
    json_rpc_arbitrum: Optional[str] = shared_options.json_rpc_arbitrum,
    json_rpc_anvil: Optional[str] = shared_options.json_rpc_anvil,
    json_rpc_derive: Optional[str] = shared_options.json_rpc_derive,

    unknown_token_receiver: Optional[str] = Option(None, "--unknown-token-receiver", envvar="UNKNOWN_TOKEN_RECEIVER", help="The Ethereum address that will receive any token that cannot be associated with an open position. For Enzyme vault based strategies this address defauts to the executor hot wallet."),

    # Test functionality
    test_evm_uniswap_v2_router: Optional[str] = shared_options.test_evm_uniswap_v2_router,
    test_evm_uniswap_v2_factory: Optional[str] = shared_options.test_evm_uniswap_v2_factory,
    test_evm_uniswap_v2_init_code_hash: Optional[str] = shared_options.test_evm_uniswap_v2_init_code_hash,
    unit_testing: bool = shared_options.unit_testing,

    chain_settle_wait_seconds: float = Option(15.0, "--chain-settle-wait-seconds", envvar="CHAIN_SETTLE_WAIT_SECONDS", help="How long we wait after the account correction to see if our broadcasted transactions fixed the issue."),
    skip_save: bool = Option(False, "--skip-save", is_flag=False, envvar="SKIP_SAVE", help="Do not update state file after the account correction. Only used in testing."),
    skip_interest: bool = Option(False, "--skip-interest", envvar="SKIP_INTEREST", help="Do not do interest distribution. If an position balance is fixed down due to redemption, this is useful."),
    process_redemption: bool = Option(False, "--process-redemption", envvar="PROCESS_REDEMPTION", help="Attempt to process deposit and redemption requests before correcting accounts."),
    process_redemption_end_block_hint: int = Option(None, "--process-redemption-end-block-hint", envvar="PROCESS_REDEMPTION_END_BLOCK_HINT", help="Used in integration testing."),
    transfer_away: bool = Option(False, "--transfer-away", envvar="TRANSFER_AWAY", help="For tokens without assigned position, scoop them to the hot wallet instead of trying to construct a new position"),
    raise_on_unclean: bool = typer.Option(False, is_flag=True, envvar="RAISE_ON_UNCLEAN", help="Raise an exception if unclean. Unit test option."),

    # Derive exchange account options
    derive_owner_private_key: Optional[str] = Option(None, envvar="DERIVE_OWNER_PRIVATE_KEY", help="Derive owner wallet private key"),
    derive_session_private_key: Optional[str] = Option(None, envvar="DERIVE_SESSION_PRIVATE_KEY", help="Derive session key private key"),
    derive_wallet_address: Optional[str] = Option(None, envvar="DERIVE_WALLET_ADDRESS", help="Derive wallet address (auto-derived from owner key if not provided). For Lagoon vault deployments, set this to the Safe multisig address."),
    derive_network: DeriveNetwork = Option(DeriveNetwork.mainnet, envvar="DERIVE_NETWORK", help="Derive network: mainnet or testnet"),

    # CCXT exchange account options
    ccxt_exchange_id: Optional[str] = Option(None, envvar="CCXT_EXCHANGE_ID", help="CCXT exchange identifier (e.g. aster, binance, bybit)"),
    ccxt_options: Optional[str] = Option(None, envvar="CCXT_OPTIONS", help="CCXT exchange constructor options as JSON string"),
    ccxt_sandbox: bool = Option(False, envvar="CCXT_SANDBOX", help="Use CCXT exchange sandbox/testnet mode"),
):
    """Correct accounting errors in the internal ledger of the trade executor.

    Trade executor tracks all non-controlled flow of assets with events.
    This includes deposits and redemptions and interest events.
    Under misbehavior, tracked asset amounts in the internal ledger
    might drift off from the actual on-chain balances. Such
    misbehavior may be caused e.g. misbehaving blockchain nodes.

    This command will fix any accounting divergences between a vault and a strategy state.
    The strategy must not have any open positions to be reinitialised, because those open
    positions cannot carry over with the current event based tracking logic.

    This command is interactive and you need to confirm any changes applied to the state.

    An old state file is automatically backed up.
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
        json_rpc_derive=json_rpc_derive,
        unit_testing=unit_testing,
    )

    assert web3config, "No RPC endpoints given. A working JSON-RPC connection is needed for check-wallet"

    # Set default chain but allow multiple connections for multichain strategies
    if len(web3config.connections) == 1:
        web3config.choose_single_chain()
    else:
        default_chain_id = next(iter(web3config.connections.keys()))
        web3config.set_default_chain(default_chain_id)
        logger.info(
            "Multichain mode: default chain %s, %d chain(s) connected",
            default_chain_id.name,
            len(web3config.connections),
        )

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

    # Log all connected chains
    for chain_id, conn in web3config.connections.items():
        logger.info(f"  Chain {chain_id.name} (id {conn.eth.chain_id:,})")
        logger.info(f"    Latest block is {conn.eth.block_number:,}")

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

    store, state = backup_state(state_file, unit_testing=unit_testing)

    mod: StrategyModuleInformation = read_strategy_module(strategy_file)

    slippage_tolerance = 0.013
    if mod:
        if mod.parameters:
            slippage_tolerance = mod.parameters.get("slippage_tolerance", 0.013)

    logger.info("Using slippage tolerance: %f", slippage_tolerance)

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

    execution_context = ExecutionContext(
        mode=ExecutionMode.one_off,
        engine_version=mod.trading_strategy_engine_version,
    )
    strategy_factory = make_factory_from_strategy_mod(mod)

    execution_model, sync_model, valuation_model_factory, pricing_model_factory = create_execution_and_sync_model(
        asset_management_mode=asset_management_mode,
        private_key=private_key,
        web3config=web3config,
        confirmation_timeout=datetime.timedelta(seconds=60),
        vault_address=vault_address,
        vault_adapter_address=vault_adapter_address,
        routing_hint=mod.trade_routing,
        confirmation_block_count=0,
        max_slippage=slippage_tolerance,
        min_gas_balance=Decimal(0),
        vault_payment_forwarder_address=vault_payment_forwarder,
    )

    run_description: StrategyExecutionDescription = strategy_factory(
        execution_model=execution_model,
        execution_context=execution_context,
        sync_model=sync_model,
        valuation_model_factory=valuation_model_factory,
        pricing_model_factory=pricing_model_factory,
        client=client,
        run_state=RunState(),
        timed_task_context_manager=execution_context.timed_task_context_manager,
        approval_model=None,
    )

    universe_model: TradingStrategyUniverseModel = run_description.universe_model
    universe = universe_model.construct_universe(
        native_datetime_utc_now(),
        execution_context.mode,
        UniverseOptions(history_period=mod.get_live_trading_history_period()),
        execution_model=run_description.runner.execution_model,
        strategy_parameters=mod.parameters,
    )

    runner = run_description.runner
    if mod.is_version_greater_or_equal_than(0, 5, 0):
        # Skip routing setup for strategies that don't do on-chain trading
        if mod.trade_routing == TradeRouting.ignore:
            routing_state = pricing_model = valuation_method = None
            logger.info("Routing setup skipped - strategy uses TradeRouting.ignore")
        else:
            routing_state, pricing_model, valuation_method = runner.setup_routing(universe)
    else:
        # Legacy unit test compatibility
        routing_state = pricing_model = valuation_method = None

    logger.info("Engine version: %s", mod.trading_strategy_engine_version)
    logger.info("Universe contains %d pairs", universe.data_universe.pairs.get_count())
    logger.info("Reserve assets are: %s", universe.reserve_assets)
    logger.info("Pricing model is: %s", pricing_model)

    assert len(universe.reserve_assets) == 1, "Need exactly one reserve asset"

    if not state.portfolio.reserves:
        # Running correct-account on clean init()
        # Need to add reserves now, because we have missed the original deposit event
        logger.info("Reserve configuration not detected, adding %s", universe.reserve_assets)
        assert len(universe.reserve_assets) > 0
        reserve_asset = universe.reserve_assets[0]
        if not state.portfolio.reserves:
            state.portfolio.initialise_reserves(reserve_asset)

    if not state.portfolio.get_default_reserve_position().reserve_token_price:
        # Fix USDC stablecoin price to be 1.0
        state.portfolio.get_default_reserve_position().reserve_token_price = 1.0

    logger.info("Reserves are %s", state.portfolio.reserves)

    # Set initial reserves,
    # in order to run the tests
    # TODO: Have this / treasury sync as a separate CLI command later
    if unit_testing:
        if len(state.portfolio.reserves) == 0:
            logger.info("Initialising reserves for the unit test: %s", universe.reserve_assets[0])
            state.portfolio.initialise_reserves(universe.reserve_assets[0])

    double_positions = check_double_position(state, printer=logger.info)
    if double_positions:
        logger.info("Double positions detected. You should *not* proceed with accounting correction,")
        logger.info("because we cannot correct onchain token balance across multiple positions.")
        logger.info("Manually remove duplicates with close-position command first.")
        raise RuntimeError("Crash for safety")

    # Auto-create CCTP bridge positions from universe before corrections
    # (so newly created positions are included in the on-chain balance check)
    from tradeexecutor.strategy.account_correction import create_missing_cctp_bridge_positions

    logger.info("Checking for missing CCTP bridge positions in universe...")
    created_bridge_trades = create_missing_cctp_bridge_positions(
        strategy_universe=universe,
        state=state,
        strategy_cycle_at=native_datetime_utc_now(),
    )
    if created_bridge_trades:
        logger.info("Auto-created %d CCTP bridge position(s)", len(created_bridge_trades))
        for trade in created_bridge_trades:
            logger.info("  Created bridge position for %s", trade.pair)
    else:
        logger.info("No missing CCTP bridge positions")

    block_number = get_almost_latest_block_number(web3)
    logger.info(f"Correcting accounts at block {block_number:,}")

    block_timestamp = get_block_timestamp(web3, block_number)

    # Skip on-chain corrections when all positions are exchange account positions
    # (their balances are synced via exchange API, not on-chain balance checks)
    has_onchain_positions = any(
        not p.pair.is_exchange_account()
        for p in state.portfolio.get_open_and_frozen_positions()
    )

    if not has_onchain_positions:
        corrections = []
        logger.info("On-chain account correction skipped - no on-chain positions")
    else:
        corrections = calculate_account_corrections(
            universe.data_universe.pairs,
            universe.reserve_assets,
            state,
            sync_model,
            block_identifier=block_number,
        )
        corrections = list(corrections)

    if len(corrections) == 0:
        logger.info("No account corrections found")

    check_state_internal_coherence(state)

    tx_builder = sync_model.create_transaction_builder()

    # Set the default token dump address
    if not unknown_token_receiver:
        if isinstance(sync_model, EnzymeVaultSyncModel):
            unknown_token_receiver = sync_model.get_hot_wallet().address

    # TODO: No longer needed as unknown tokens should be mapped to a new opened spot position
    #
    # if not unknown_token_receiver:
    #    raise RuntimeError(f"unknown_token_receiver missing and cannot deduct from the config. Please give one on the command line.")

    assert hot_wallet is not None
    hot_wallet.sync_nonce(web3)
    logger.info("Hot wallet nonce is %d", hot_wallet.current_nonce)

    if not skip_interest:
        credit_positions = [p for p in state.portfolio.get_open_and_frozen_positions() if p.is_credit_supply()]
        if len(credit_positions) > 0:
            # Sync missing credit
            try:
                logger.info("Credit positions detected, syncing interest before applying accounting checks")
                for p in credit_positions:
                    logger.info(" - Position: %s", p)
                balances_updates = sync_model.sync_interests(
                    timestamp=native_datetime_utc_now(),
                    state=state,
                    universe=universe,
                    pricing_model=pricing_model,
                )
                for bu in balances_updates:
                    logger.info("  - Balance update: %s", bu)
            except Exception as e:
                logger.info("correct-accounts: could not sync interest %s", e)
                raise
    else:
        logger.info("Interest distribution skipped")

    # Auto-create missing exchange account positions first
    # (so newly created positions are included in the sync below)
    if universe:
        from tradeexecutor.strategy.account_correction import create_missing_exchange_account_positions

        logger.info("Checking for missing exchange account positions in universe...")
        created_trades = create_missing_exchange_account_positions(
            strategy_universe=universe,
            state=state,
            strategy_cycle_at=native_datetime_utc_now(),
        )

        if created_trades:
            logger.info("Auto-created %d exchange account position(s)", len(created_trades))
            for trade in created_trades:
                logger.info("  Created position for %s", trade.pair)
        else:
            logger.info("No missing exchange account positions")

    # Collect ALL exchange account positions (existing + newly created)
    exchange_account_positions = [
        p for p in state.portfolio.get_open_and_frozen_positions()
        if p.is_exchange_account()
    ]

    # Sync all exchange account positions with actual exchange API values
    if exchange_account_positions:
        logger.info("Found %d exchange account position(s)", len(exchange_account_positions))
        from tradeexecutor.exchange_account.sync_model import ExchangeAccountSyncModel

        account_value_func = create_exchange_account_value_func(
            exchange_account_positions,
            derive_owner_private_key,
            derive_session_private_key,
            derive_wallet_address,
            derive_network,
            ccxt_exchange_id,
            ccxt_options,
            ccxt_sandbox,
            logger,
        )

        if account_value_func:
            exchange_sync_model = ExchangeAccountSyncModel(account_value_func)
            exchange_events = exchange_sync_model.sync_positions(
                timestamp=native_datetime_utc_now(),
                state=state,
                strategy_universe=universe,
                pricing_model=pricing_model,
            )
            logger.info("Exchange account sync: %d balance update(s)", len(exchange_events))
            for evt in exchange_events:
                logger.info("  Position %d: %s (change: %s)", evt.position_id, evt.notes, evt.quantity)

    if process_redemption:
        timestamp = native_datetime_utc_now()
        reserve_assets = list(universe.reserve_assets)

        if process_redemption_end_block_hint:
            # Passed by unit tests so we are not going to scan the whole chain until today (wall clock time)
            end_block = process_redemption_end_block_hint
        else:
            end_block = execution_model.get_safe_latest_block()

        logger.info(
            "Processing deposits/redemptions, timestamp set to %s, reserves are %s, end block is %d",
            timestamp,
            reserve_assets,
            end_block,
        )

        sync_model.sync_treasury(
            strategy_cycle_ts=timestamp,
            state=state,
            end_block=end_block,
            post_valuation=True,
        )
    else:
        logger.info("Deposit/redemption distribution skipped")

    if asset_management_mode.is_vault():
        tx_builder.hot_wallet.sync_nonce(web3)

    balance_updates = _correct_accounts(
        state,
        corrections,
        strategy_cycle_included_at=None,
        interactive=not unit_testing,
        tx_builder=tx_builder,
        unknown_token_receiver=unknown_token_receiver,  # Send any unknown tokens to the hot wallet of the trade-executor
        block_identifier=block_number,
        block_timestamp=block_timestamp,
        strategy_universe=universe,
        pricing_model=pricing_model,
        token_fix_method=UnknownTokenPositionFix.transfer_away if transfer_away else UnknownTokenPositionFix.open_missing_position,
    )
    balance_updates = list(balance_updates)  # Side effect: this will force execution of all actions stuck in the iterator
    logger.info(f"We did {len(corrections)} accounting corrections, of which {len(balance_updates)} internal state balance updates, new block height is {block_number:,} at {block_timestamp}")

    if not skip_save:
        logger.info("Saving state to %s", store.path)
        store.sync(state)
    else:
        logger.info("Saving the fixed state skipped")

    web3config.close()

    # Shortcut here
    if unit_testing:
        chain_settle_wait_seconds = 0

    if len(corrections) > 0 and chain_settle_wait_seconds:
        logger.info("Waiting %f seconds to see before reading back new results from on-chain", chain_settle_wait_seconds)
        time.sleep(chain_settle_wait_seconds)

    block_number = get_almost_latest_block_number(web3)

    # Skip final on-chain account check for strategies without on-chain positions
    has_onchain_positions_final = any(
        not p.pair.is_exchange_account()
        for p in state.portfolio.get_open_and_frozen_positions()
    )
    if not has_onchain_positions_final:
        logger.info("Final account check skipped - no on-chain positions")
        logger.info("All ok")
        sys.exit(0)

    clean, df = check_accounts(
        universe.data_universe.pairs,
        universe.reserve_assets,
        state,
        sync_model,
        block_identifier=block_number,
    )

    output = tabulate(df, headers='keys', tablefmt='rounded_outline')

    # Append exchange account positions to the summary
    exchange_positions = [
        p for p in state.portfolio.get_open_and_frozen_positions()
        if p.is_exchange_account()
    ]
    if exchange_positions:
        rows = []
        for p in exchange_positions:
            protocol = p.pair.get_exchange_account_protocol() or "unknown"
            quantity = p.get_quantity()
            rows.append([
                p.pair.get_ticker(),
                protocol,
                f"{quantity:,.2f}",
                p.position_id,
            ])
        exchange_output = tabulate(
            rows,
            headers=["Position", "Protocol", "Value (USD)", "Position ID"],
            tablefmt="rounded_outline",
        )
        output += f"\n\nExchange account positions:\n{exchange_output}"

    if clean:
        logger.info(f"Accounts after the correction match for block {block_number:,}:\n%s", output)
        if not raise_on_unclean:
            logger.info("All ok")
            sys.exit(0)
        else:
            logger.info("Unit test exit - nothing to be done")
    else:
        logger.error("Accounts still broken after the correction")
        logger.info("\n" + output)
        if not raise_on_unclean:
            sys.exit(1)
        raise UncleanState(output)

