"""Test selected external vaults from a dedicated Lagoon executor."""

import datetime
import logging
import signal
from collections import defaultdict, deque
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

from eth_defi.compat import native_datetime_utc_now
from eth_defi.vault.base import VaultSpec
from tabulate import tabulate
from tradingstrategy.client import Client
from tradingstrategy.chain import ChainId
from typer import Option

from tradeexecutor.cli.bootstrap import (
    check_universe_chains_have_gas,
    check_universe_chains_have_rpc,
    check_universe_contracts_resolve,
    create_execution_and_sync_model,
    create_state_store,
    create_web3_config,
    prepare_cache_and_token_cache,
    prepare_executor_id,
    resolve_deployment_file,
)
from tradeexecutor.cli.commands import shared_options
from tradeexecutor.cli.commands.app import app
from tradeexecutor.cli.log import setup_logging
from tradeexecutor.cli.testtrade import perform_test_trade
from tradeexecutor.cli.vault_test_trade import (
    build_vault_test_universe,
    close_simulated_positions,
    create_vault_test_diagnostic_pair,
    create_vault_test_attempt_metadata,
    filter_rpc_kwargs_for_vault_specs,
    get_latest_vault_position,
    get_vault_test_status,
    get_vault_trade_position,
    load_lagoon_deployment,
    merge_simulated_attempt,
    parse_vault_ids,
    SIMULATED_LAGOON_PRIVATE_KEY,
    stamp_position_vault_test_attempt,
)
from tradeexecutor.cli.vault_test_trade_tui import (
    VaultChoice,
    display_vault_test_trade_ui,
)
from tradeexecutor.cli.vault_test_trade_simulation import (
    SIMULATED_VAULT_ATTEMPT_TIMEOUT,
    SimulatedVaultAttemptTimeout,
    is_simulated_infrastructure_failure,
    queue_simulated_infrastructure_retry,
    raise_simulated_vault_attempt_timeout,
    restore_simulated_snapshots,
    start_simulated_vault_runtime_with_replacement,
    take_simulated_snapshots,
)
from tradeexecutor.ethereum.routing_state import OutOfBalance
from tradeexecutor.ethereum.cctp.retry import check_and_retry_cctp_in_transit
from tradeexecutor.ethereum.token import translate_token_details
from tradeexecutor.state.trade import TradeFlag, TradeStatus
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.execution_model import (
    AssetManagementMode,
    ExecutionHaltableIssue,
)
from tradeexecutor.strategy.generic.generic_pricing_model import GenericPricing
from tradeexecutor.strategy.generic.generic_valuation import GenericValuation
from tradeexecutor.strategy.valuation import revalue_state
from tradeexecutor.strategy.trading_strategy_universe import (
    load_vault_universe_with_metadata,
)

logger = logging.getLogger(__name__)


def _close_web3config_safely(web3config) -> None:
    """Close RPC and Anvil resources without masking the command result."""

    try:
        web3config.close()
    except Exception:
        logger.exception("One or more Web3 connections did not close cleanly")


def _create_client(trading_strategy_api_key: str, cache_path: Path) -> Client:
    """Create the live Trading Strategy client without a strategy module."""

    if not trading_strategy_api_key:
        raise RuntimeError(
            "TRADING_STRATEGY_API_KEY is required to download the vault universe"
        )
    return Client.create_live_client(
        trading_strategy_api_key, cache_path=cache_path, settings_path=None
    )


def _initialise_state(state, sync_model, reserve_asset) -> None:
    """Initialise a pristine Lagoon state from its on-chain reserve."""

    sync_model.sync_initial(state, reserve_asset=reserve_asset, reserve_token_price=1.0)
    sync_model.sync_treasury(native_datetime_utc_now(), state, [reserve_asset])


def _setup_action_models(execution_model, universe):
    """Create generic routing, pricing and valuation for one action universe."""

    routing_model = execution_model.create_default_routing_model(universe)
    pricing_model = GenericPricing(routing_model.pair_configurator)
    valuation_model = GenericValuation(routing_model.pair_configurator)
    routing_state = routing_model.create_routing_state(
        universe,
        execution_model.get_routing_state_details(),
    )
    return routing_model, pricing_model, valuation_model, routing_state


def _should_leave_deposit_open(
    *,
    operation: str,
    is_async: bool,
    redemption_available: bool,
    manual: bool,
) -> bool:
    """Decide whether a deposit action must stop before redemption."""

    return operation == "deposit" and (manual or is_async or not redemption_available)


def _record_attempt_result(
    state,
    pair,
    vault_spec: VaultSpec,
    *,
    simulated: bool,
    result: str,
    detail: str | None = None,
    source_position_id: int | None = None,
):
    """Create a closed normal position for a pre-trade diagnostic result."""

    reserve = state.portfolio.get_default_reserve_position().asset
    now = native_datetime_utc_now()
    position = state.portfolio.open_new_position(
        now,
        pair,
        assumed_price=1.0,
        reserve_currency=reserve,
        reserve_currency_price=1.0,
    )
    position.simulated = simulated
    attempt = create_vault_test_attempt_metadata(vault_spec, simulated=simulated)
    attempt["result"] = result
    if detail:
        attempt["detail"] = detail
    if source_position_id is not None:
        attempt["source_position_id"] = source_position_id
    position.other_data["vault_test_attempt"] = attempt
    state.portfolio.close_position(position, now)
    return position


def _handle_simulated_infrastructure_failure(
    *,
    error: BaseException,
    spec: VaultSpec,
    pending_specs: deque,
    restart_counts: dict[str, int],
    state,
    store,
    rows: list[dict],
    mode: str,
    reserve_asset,
    vault,
    pair,
    previous,
) -> None:
    """Queue one fresh-runtime attempt or record the terminal infrastructure failure."""

    if queue_simulated_infrastructure_retry(spec, pending_specs, restart_counts):
        logger.warning(
            "Vault simulation infrastructure failed for %s; rerunning with a new Anvil generation: %s",
            spec.as_string_id(),
            error,
        )
        return

    logger.error(
        "Vault simulation infrastructure failed again for %s after Anvil replacement",
        spec.as_string_id(),
    )
    pair = pair or create_vault_test_diagnostic_pair(spec, reserve_asset, vault)
    detail = f"Anvil infrastructure failed after replacement: {error}"
    _record_attempt_result(
        state,
        pair,
        spec,
        simulated=True,
        result="infrastructure_failed",
        detail=detail,
        source_position_id=previous.position_id if previous else None,
    )
    store.sync(state)
    rows.append(_create_result_row(vault, spec, state, mode, detail))


def _create_result_row(
    vault, vault_spec: VaultSpec, state, mode: str, detail: str | None = None
) -> dict:
    position = get_latest_vault_position(state, vault_spec)
    attempt = position.other_data.get("vault_test_attempt", {}) if position else {}
    detail = detail or attempt.get("detail")
    if detail:
        detail = " ".join(str(detail).split())
        if len(detail) > 160:
            detail = detail[:157] + "..."
    return {
        "vault id": vault_spec.as_string_id(),
        "vault": getattr(vault, "name", vault_spec.vault_address),
        "chain": getattr(
            getattr(vault, "chain_id", None),
            "get_name",
            lambda: str(vault_spec.chain_id),
        )(),
        "protocol": getattr(vault, "protocol_name", "unknown"),
        "mode": mode,
        "status": get_vault_test_status(position),
        "position": position.position_id if position else None,
        "detail": detail,
    }


@app.command()
@shared_options.with_json_rpc_options()
def vault_test_trade(
    id: str = Option(
        ..., "--id", envvar="EXECUTOR_ID", help="Dedicated Lagoon test executor id."
    ),
    name: str | None = shared_options.name,
    trading_strategy_api_key: str | None = shared_options.trading_strategy_api_key,
    state_file: Path | None = shared_options.state_file,
    cache_path: Path | None = shared_options.cache_path,
    log_level: str | None = shared_options.log_level,
    rpc_kwargs: dict | None = None,
    private_key: str | None = shared_options.private_key,
    asset_management_mode: AssetManagementMode
    | None = shared_options.asset_management_mode,
    min_gas_balance: float | None = shared_options.min_gas_balance,
    max_slippage: float | None = shared_options.max_slippage,
    confirmation_block_count: int = shared_options.confirmation_block_count,
    confirmation_timeout: int = shared_options.confirmation_timeout,
    unit_testing: bool = shared_options.unit_testing,
    amount: float = Option(1.0, "--amount", envvar="AMOUNT", min=0.0000001),
    vault_id: str | None = Option(
        None,
        "--vault-id",
        envvar="VAULT_ID",
        help="Comma-separated chain-address vault ids.",
    ),
    auto_simulated: bool = Option(
        False, "--auto-simulated", help="Run fork-only simulations for VAULT_ID."
    ),
    auto_real: bool = Option(
        False, "--auto-real", help="Run real deposits/redemptions for VAULT_ID."
    ),
    rerun: bool = Option(
        False, "--rerun", help="Create a new attempt after a terminal result."
    ),
):
    """Test selected external vaults using one deployed Lagoon executor.

    With no automatic mode, starts the Textual operator interface.
    """

    if auto_simulated and auto_real:
        raise RuntimeError("--auto-simulated and --auto-real are mutually exclusive")
    if rerun and not (auto_simulated or auto_real):
        raise RuntimeError("--rerun requires --auto-simulated or --auto-real")
    if asset_management_mode != AssetManagementMode.lagoon:
        raise RuntimeError("vault-test-trade requires ASSET_MANAGEMENT_MODE=lagoon")

    id = prepare_executor_id(id, None)
    logger = setup_logging(log_level=log_level)
    vault_specs = parse_vault_ids(vault_id) if auto_simulated or auto_real else []
    resolved_state_file = state_file or Path(f"state/{id}.json")

    cache_path, token_cache = prepare_cache_and_token_cache(
        id, cache_path, unit_testing=unit_testing
    )
    client = _create_client(trading_strategy_api_key, cache_path)
    vault_universe = load_vault_universe_with_metadata(client)

    deployment_file = resolve_deployment_file(id, resolved_state_file)
    simulated_runtime = None
    simulated_runtime_kwargs = None
    max_slippage = max_slippage or 0.005
    amount = Decimal(str(amount))
    if auto_simulated:
        rpc_kwargs = filter_rpc_kwargs_for_vault_specs(rpc_kwargs, vault_specs)
        private_key = private_key or SIMULATED_LAGOON_PRIVATE_KEY
        simulated_runtime_kwargs = dict(
            executor_id=id,
            rpc_kwargs=rpc_kwargs,
            unit_testing=unit_testing,
            vault_specs=vault_specs,
            vault_universe=vault_universe,
            private_key=private_key,
            amount=amount,
            asset_management_mode=asset_management_mode,
            confirmation_timeout=confirmation_timeout,
            confirmation_block_count=confirmation_block_count,
            min_gas_balance=min_gas_balance,
            max_slippage=max_slippage,
            token_cache=token_cache,
        )
        simulated_runtime = start_simulated_vault_runtime_with_replacement(
            generation=1,
            **simulated_runtime_kwargs,
        )
        web3config = simulated_runtime.web3config
        deployment = simulated_runtime.deployment
        deployment_file = simulated_runtime.deployment_file
        execution_model = simulated_runtime.execution_model
        sync_model = simulated_runtime.sync_model
        reserve_asset = simulated_runtime.reserve_asset
    else:
        deployment = load_lagoon_deployment(deployment_file)
        primary_chain_id = deployment.primary_chain_id
        web3config = create_web3_config(
            **rpc_kwargs,
            unit_testing=unit_testing,
            simulate=False,
        )
        if not web3config.has_any_connection():
            raise RuntimeError("vault-test-trade requires JSON-RPC connections")
        web3config.set_default_chain(primary_chain_id)
        web3config.check_default_chain_id()
        execution_model, sync_model, _, _ = create_execution_and_sync_model(
            asset_management_mode=asset_management_mode,
            private_key=private_key,
            web3config=web3config,
            confirmation_timeout=datetime.timedelta(seconds=confirmation_timeout),
            confirmation_block_count=confirmation_block_count,
            min_gas_balance=min_gas_balance,
            max_slippage=max_slippage,
            vault_address=deployment.vault_address,
            vault_adapter_address=deployment.module_address,
            vault_payment_forwarder_address=None,
            token_cache=token_cache,
            deployment_file=deployment_file,
        )
        reserve_asset = translate_token_details(sync_model.vault.denomination_token)
    store = create_state_store(resolved_state_file)
    if store.is_pristine():
        state = store.create(name or id)
        _initialise_state(state, sync_model, reserve_asset)
        store.sync(state)
    else:
        state = store.load()

    real_execution = not auto_simulated
    if real_execution:
        execution_model.initialize()
        resolved_bridges = check_and_retry_cctp_in_transit(
            state=state,
            execution_model=execution_model,
            web3config=web3config,
        )
        resolved_vaults = execution_model.resolve_pending_vault_settlements(
            state=state,
            ts=native_datetime_utc_now(),
        )
        if resolved_bridges or resolved_vaults:
            logger.info(
                "Resolved %d CCTP transfer(s) and %d vault settlement(s) before vault testing",
                len(resolved_bridges),
                len(resolved_vaults),
            )
        # Retry helpers append signed transactions before broadcasting them.
        # Persist even when nothing fully resolved, so a failed broadcast can
        # be resumed idempotently without reusing its nonce.
        store.sync(state)

    manual_action = None
    if not (auto_simulated or auto_real):
        choices = sorted(
            [
                VaultChoice(
                    vault_spec=VaultSpec(vault.chain_id.value, vault.vault_address),
                    name=vault.name or vault.vault_address,
                    chain=vault.chain_id.get_name(),
                    protocol=vault.protocol_name or "unknown",
                )
                for vault in vault_universe.iterate_vaults()
            ],
            key=lambda choice: (choice.name.lower(), choice.vault_spec.as_string_id()),
        )
        manual_action = display_vault_test_trade_ui(choices=choices, state=state)
        if manual_action is None:
            _close_web3config_safely(web3config)
            return
        vault_specs = [manual_action.vault_spec]

    mode = "simulated" if auto_simulated else "real"
    rows: list[dict] = []
    pending_specs = deque(vault_specs)
    infrastructure_restart_counts: dict[str, int] = defaultdict(int)
    restart_requested: BaseException | None = None

    try:
        while pending_specs:
            if restart_requested is not None:
                assert auto_simulated
                assert simulated_runtime is not None
                assert simulated_runtime_kwargs is not None
                failed_generation = simulated_runtime.generation
                logger.warning(
                    "Replacing simulated vault runtime generation %d after infrastructure failure: %s",
                    failed_generation,
                    restart_requested,
                )
                simulated_runtime.close()
                simulated_runtime = start_simulated_vault_runtime_with_replacement(
                    generation=failed_generation + 1,
                    **simulated_runtime_kwargs,
                )
                web3config = simulated_runtime.web3config
                deployment = simulated_runtime.deployment
                deployment_file = simulated_runtime.deployment_file
                execution_model = simulated_runtime.execution_model
                sync_model = simulated_runtime.sync_model
                restarted_reserve_asset = simulated_runtime.reserve_asset
                assert restarted_reserve_asset == reserve_asset, (
                    f"Simulation reserve changed across Anvil generations: "
                    f"{reserve_asset} != {restarted_reserve_asset}"
                )
                restart_requested = None

            spec = pending_specs.popleft()
            vault = vault_universe.get_by_vault_spec(
                (spec.chain_id, spec.vault_address)
            )
            if vault is None:
                pair = create_vault_test_diagnostic_pair(spec, reserve_asset)
                _record_attempt_result(
                    state,
                    pair,
                    spec,
                    simulated=auto_simulated,
                    result="failed",
                    detail="Vault not in downloaded universe",
                )
                store.sync(state)
                rows.append(
                    _create_result_row(
                        None, spec, state, mode, "Vault not in downloaded universe"
                    )
                )
                continue

            pair = None
            previous = None
            fork_snapshots = {}
            previous_alarm_handler = None
            alarm_installed = False
            alarm_armed = False
            attempt_infrastructure_failure = None
            try:
                if auto_simulated:
                    previous_alarm_handler = signal.signal(
                        signal.SIGALRM, raise_simulated_vault_attempt_timeout
                    )
                    alarm_installed = True
                    signal.alarm(SIMULATED_VAULT_ATTEMPT_TIMEOUT)
                    alarm_armed = True
                    fork_snapshots = take_simulated_snapshots(
                        web3config, deployment, spec
                    )
                if (
                    spec.chain_id != deployment.primary_chain_id.value
                    and ChainId(spec.chain_id) not in deployment.satellite_modules
                ):
                    raise RuntimeError(
                        f"Vault {spec.as_string_id()} is on a satellite chain without a deployed Lagoon module"
                    )
                execution_context = ExecutionContext(mode=ExecutionMode.preflight_check)
                universe = build_vault_test_universe(
                    client=client,
                    vault_universe=vault_universe,
                    vault_spec=spec,
                    reserve_asset=reserve_asset,
                    primary_chain_id=deployment.primary_chain_id,
                    execution_context=execution_context,
                )
                pair = universe.get_pair_by_smart_contract(spec.vault_address)

                # Match perform-test-trade preflights before any irreversible
                # bridge burn or vault transaction is constructed.
                check_universe_chains_have_rpc(web3config, universe)
                wallet_address = execution_model.tx_builder.get_gas_wallet_address()
                min_gas = getattr(execution_model, "min_balance_threshold", 0)
                check_universe_chains_have_gas(
                    web3config, universe, wallet_address, min_gas
                )
                check_universe_contracts_resolve(web3config, universe, execution_model)
                routing_model, pricing_model, valuation_model, routing_state = (
                    _setup_action_models(execution_model, universe)
                )

                previous = get_latest_vault_position(state, spec)
                previous_status = get_vault_test_status(previous)
                open_trade_position = get_vault_trade_position(
                    state, spec, open_only=True, simulated=False
                )
                bridge_position = state.portfolio.get_bridge_position_for_chain(
                    spec.chain_id
                )
                bridge_attempt = (
                    bridge_position.other_data.get("vault_test_attempt", {})
                    if bridge_position
                    else {}
                )
                bridge_in_transit = bool(
                    bridge_position
                    and bridge_attempt.get("vault_id") == spec.as_string_id()
                    and any(
                        trade.get_status() == TradeStatus.cctp_in_transit
                        for trade in bridge_position.trades.values()
                    )
                )
                if bridge_in_transit:
                    rows.append(
                        _create_result_row(
                            vault,
                            spec,
                            state,
                            mode,
                            "CCTP transfer is still in transit",
                        )
                    )
                    continue

                if manual_action is not None:
                    operation = manual_action.action
                    if operation == "deposit" and open_trade_position is not None:
                        rows.append(
                            _create_result_row(
                                vault,
                                spec,
                                state,
                                mode,
                                "Deposit is already open; select it for redemption",
                            )
                        )
                        continue
                    if operation == "redeem" and open_trade_position is None:
                        rows.append(
                            _create_result_row(
                                vault,
                                spec,
                                state,
                                mode,
                                "No open deposit is available for redemption",
                            )
                        )
                        continue
                    if operation == "redeem" and get_vault_test_status(
                        open_trade_position
                    ) in {"deposit pending", "redemption pending"}:
                        rows.append(
                            _create_result_row(
                                vault,
                                spec,
                                state,
                                mode,
                                "Pending request must settle before redemption",
                            )
                        )
                        continue
                elif open_trade_position is not None and get_vault_test_status(
                    open_trade_position
                ) in {"deposit pending", "redemption pending"}:
                    rows.append(
                        _create_result_row(
                            vault,
                            spec,
                            state,
                            mode,
                            "Pending request is not retried automatically",
                        )
                    )
                    continue
                elif previous_status in {"deposit pending", "redemption pending"}:
                    rows.append(
                        _create_result_row(
                            vault,
                            spec,
                            state,
                            mode,
                            "Pending request is not retried automatically",
                        )
                    )
                    continue
                elif (
                    previous is not None
                    and previous.other_data.get("vault_test_attempt", {}).get("phase")
                    == "bridge_out_pending"
                ):
                    operation = "deposit"
                elif (
                    previous is not None
                    and previous.other_data.get("vault_test_attempt", {}).get("phase")
                    == "bridge_back_pending"
                ):
                    if bridge_position is not None:
                        operation = "redeem"
                    else:
                        stamp_position_vault_test_attempt(
                            previous,
                            spec,
                            simulated=False,
                            phase="complete",
                            result="success",
                        )
                        store.sync(state)
                        rows.append(_create_result_row(vault, spec, state, mode))
                        continue
                elif open_trade_position is not None:
                    operation = "redeem"
                elif (
                    previous is not None
                    and previous.other_data.get("vault_test_attempt", {}).get("result")
                    and not rerun
                ):
                    rows.append(
                        _create_result_row(
                            vault,
                            spec,
                            state,
                            mode,
                            "Existing terminal result; use --rerun to retest",
                        )
                    )
                    continue
                elif (
                    previous is not None
                    and previous.other_data.get("vault_test_attempt", {}).get("phase")
                    == "redemption_requested"
                    and bridge_position is not None
                ):
                    operation = "redeem"
                elif previous is None:
                    operation = "deposit"
                elif rerun:
                    operation = "deposit"
                else:
                    rows.append(
                        _create_result_row(
                            vault,
                            spec,
                            state,
                            mode,
                            "Existing terminal result; use --rerun to retest",
                        )
                    )
                    continue

                if auto_simulated and pair.is_async_vault() and operation == "redeem":
                    _record_attempt_result(
                        state,
                        pair,
                        spec,
                        simulated=True,
                        result="simulation_unsupported_async",
                        detail="Simulation never requests async redemption",
                        source_position_id=open_trade_position.position_id
                        if open_trade_position
                        else None,
                    )
                    store.sync(state)
                    rows.append(
                        _create_result_row(
                            vault,
                            spec,
                            state,
                            mode,
                            "simulation_unsupported_async: simulation never requests async redemption",
                        )
                    )
                    continue
                if (
                    operation == "deposit"
                    and pricing_model.can_deposit(native_datetime_utc_now(), pair)
                    is False
                ):
                    _record_attempt_result(
                        state,
                        pair,
                        spec,
                        simulated=auto_simulated,
                        result="deposit_closed",
                    )
                    store.sync(state)
                    rows.append(_create_result_row(vault, spec, state, mode))
                    continue
                if (
                    operation == "redeem"
                    and pricing_model.can_redeem(native_datetime_utc_now(), pair)
                    is False
                ):
                    rows.append(
                        _create_result_row(
                            vault,
                            spec,
                            state,
                            mode,
                            "Redemption is not currently available",
                        )
                    )
                    continue
                resuming_bridge_out = bool(
                    previous
                    and previous.other_data.get("vault_test_attempt", {}).get("phase")
                    == "bridge_out_pending"
                    and bridge_position
                    and bridge_position.get_available_bridge_capital() > 0
                )
                if (
                    operation == "deposit"
                    and not resuming_bridge_out
                    and state.portfolio.get_default_reserve_position().get_value() <= 0
                ):
                    rows.append(
                        _create_result_row(
                            vault,
                            spec,
                            state,
                            mode,
                            "No cash remains for another deposit",
                        )
                    )
                    break

                if operation == "deposit":
                    # A live redemption capacity check before the deposit sees
                    # zero shares and can incorrectly report unavailable.  For
                    # deciding whether an instant vault should complete the
                    # same-run round trip, use the universe's venue gate; the
                    # post-deposit redemption itself remains authoritative.
                    redemption_available = pair.can_redeem()
                else:
                    redemption_available = pricing_model.can_redeem(
                        native_datetime_utc_now(), pair
                    )

                if real_execution:
                    revalue_state(state, native_datetime_utc_now(), valuation_model)
                    buy_only = _should_leave_deposit_open(
                        operation=operation,
                        is_async=pair.is_async_vault(),
                        redemption_available=redemption_available,
                        manual=manual_action is not None,
                    )
                    perform_test_trade(
                        web3=web3config.get_default(),
                        execution_model=execution_model,
                        pricing_model=pricing_model,
                        sync_model=sync_model,
                        state=state,
                        universe=universe,
                        routing_model=routing_model,
                        routing_state=routing_state,
                        max_slippage=max_slippage,
                        amount=amount,
                        pair=pair,
                        buy_only=buy_only,
                        close_only=operation == "redeem",
                        web3config=web3config,
                        test_short=False,
                    )
                    target_position = get_vault_trade_position(
                        state, spec, simulated=False
                    )
                    bridge_position = state.portfolio.get_bridge_position_for_chain(
                        spec.chain_id
                    )
                    in_transit_trade = (
                        next(
                            (
                                trade
                                for trade in bridge_position.trades.values()
                                if trade.get_status() == TradeStatus.cctp_in_transit
                            ),
                            None,
                        )
                        if bridge_position
                        else None
                    )
                    if in_transit_trade is not None:
                        phase = (
                            "bridge_back_pending"
                            if in_transit_trade.is_sell()
                            else "bridge_out_pending"
                        )
                        stamp_position_vault_test_attempt(
                            bridge_position, spec, simulated=False, phase=phase
                        )
                        if (
                            phase == "bridge_back_pending"
                            and target_position is not None
                        ):
                            stamp_position_vault_test_attempt(
                                target_position, spec, simulated=False, phase=phase
                            )
                    elif target_position is not None:
                        phase = (
                            "deposit_requested"
                            if operation == "deposit"
                            else "redemption_requested"
                        )
                        stamp_position_vault_test_attempt(
                            target_position, spec, simulated=False, phase=phase
                        )
                    store.sync(state)
                else:
                    original_position_ids = {
                        position.position_id
                        for position in state.portfolio.get_all_positions()
                    }
                    original_trade_ids = {
                        trade.trade_id for trade in state.portfolio.get_all_trades()
                    }
                    fork_state = deepcopy(state)
                    is_async = pair.is_async_vault()
                    perform_test_trade(
                        web3=web3config.get_default(),
                        execution_model=execution_model,
                        pricing_model=pricing_model,
                        sync_model=sync_model,
                        state=fork_state,
                        universe=universe,
                        routing_model=routing_model,
                        routing_state=routing_state,
                        max_slippage=max_slippage,
                        amount=amount,
                        pair=pair,
                        buy_only=_should_leave_deposit_open(
                            operation=operation,
                            is_async=is_async,
                            redemption_available=redemption_available,
                            manual=False,
                        ),
                        close_only=operation == "redeem",
                        web3config=web3config,
                        trade_flags={TradeFlag.simulated},
                        test_short=False,
                        force_async_settlement_on_anvil=False,
                    )
                    # The wall-clock guard exists only for external adapter and
                    # RPC work.  Never let SIGALRM interrupt state merging or an
                    # atomic state-store write after the attempt has returned.
                    signal.alarm(0)
                    alarm_armed = False
                    created_position_ids = {
                        position.position_id
                        for position in fork_state.portfolio.get_all_positions()
                        if position.position_id not in original_position_ids
                    }
                    close_simulated_positions(
                        fork_state,
                        vault_spec=spec,
                        position_ids=created_position_ids,
                        result=(
                            "simulation_unsupported_async"
                            if is_async
                            else "redemption_unavailable"
                            if not redemption_available
                            else None
                        ),
                    )
                    merge_simulated_attempt(
                        source_state=fork_state,
                        target_state=state,
                        original_position_ids=original_position_ids,
                        original_trade_ids=original_trade_ids,
                    )
                    if not created_position_ids:
                        _record_attempt_result(
                            state,
                            pair,
                            spec,
                            simulated=True,
                            result="success_simulated",
                            source_position_id=previous.position_id
                            if previous
                            else None,
                        )
                    store.sync(state)

                rows.append(_create_result_row(vault, spec, state, mode))
            except SimulatedVaultAttemptTimeout as e:
                attempt_infrastructure_failure = e
                restart_requested = e
                _handle_simulated_infrastructure_failure(
                    error=e,
                    spec=spec,
                    pending_specs=pending_specs,
                    restart_counts=infrastructure_restart_counts,
                    state=state,
                    store=store,
                    rows=rows,
                    mode=mode,
                    reserve_asset=reserve_asset,
                    vault=vault,
                    pair=pair,
                    previous=previous,
                )
                continue
            except ExecutionHaltableIssue as e:
                if auto_simulated and is_simulated_infrastructure_failure(e):
                    attempt_infrastructure_failure = e
                    restart_requested = e
                    _handle_simulated_infrastructure_failure(
                        error=e,
                        spec=spec,
                        pending_specs=pending_specs,
                        restart_counts=infrastructure_restart_counts,
                        state=state,
                        store=store,
                        rows=rows,
                        mode=mode,
                        reserve_asset=reserve_asset,
                        vault=vault,
                        pair=pair,
                        previous=previous,
                    )
                    continue

                bridge_position = state.portfolio.get_bridge_position_for_chain(
                    spec.chain_id
                )
                in_transit_trade = (
                    next(
                        (
                            trade
                            for trade in bridge_position.trades.values()
                            if trade.get_status() == TradeStatus.cctp_in_transit
                        ),
                        None,
                    )
                    if bridge_position
                    else None
                )
                if in_transit_trade is not None:
                    phase = (
                        "bridge_back_pending"
                        if in_transit_trade.is_sell()
                        else "bridge_out_pending"
                    )
                    stamp_position_vault_test_attempt(
                        bridge_position, spec, simulated=False, phase=phase
                    )
                    if phase == "bridge_back_pending":
                        target_position = get_vault_trade_position(
                            state, spec, simulated=False
                        )
                        if target_position is not None:
                            stamp_position_vault_test_attempt(
                                target_position, spec, simulated=False, phase=phase
                            )
                    store.sync(state)
                    rows.append(
                        _create_result_row(
                            vault, spec, state, mode, "CCTP transfer is in transit"
                        )
                    )
                    continue

                logger.exception("Vault test halted for %s", spec.as_string_id())
                if pair is None:
                    pair = create_vault_test_diagnostic_pair(spec, reserve_asset, vault)
                _record_attempt_result(
                    state,
                    pair,
                    spec,
                    simulated=auto_simulated,
                    result="failed",
                    detail=str(e),
                    source_position_id=previous.position_id if previous else None,
                )
                store.sync(state)
                rows.append(_create_result_row(vault, spec, state, mode, str(e)))
            except OutOfBalance as e:
                if pair is None:
                    pair = create_vault_test_diagnostic_pair(spec, reserve_asset, vault)
                _record_attempt_result(
                    state,
                    pair,
                    spec,
                    simulated=auto_simulated,
                    result="failed",
                    detail=f"Insufficient cash: {e}",
                    source_position_id=previous.position_id if previous else None,
                )
                store.sync(state)
                rows.append(
                    _create_result_row(
                        vault, spec, state, mode, f"Insufficient cash: {e}"
                    )
                )
                break
            except Exception as e:
                if auto_simulated and is_simulated_infrastructure_failure(e):
                    attempt_infrastructure_failure = e
                    restart_requested = e
                    _handle_simulated_infrastructure_failure(
                        error=e,
                        spec=spec,
                        pending_specs=pending_specs,
                        restart_counts=infrastructure_restart_counts,
                        state=state,
                        store=store,
                        rows=rows,
                        mode=mode,
                        reserve_asset=reserve_asset,
                        vault=vault,
                        pair=pair,
                        previous=previous,
                    )
                    continue

                logger.exception("Vault test failed for %s", spec.as_string_id())
                if pair is None:
                    pair = create_vault_test_diagnostic_pair(spec, reserve_asset, vault)
                _record_attempt_result(
                    state,
                    pair,
                    spec,
                    simulated=auto_simulated,
                    result="failed",
                    detail=str(e),
                    source_position_id=previous.position_id if previous else None,
                )
                store.sync(state)
                rows.append(_create_result_row(vault, spec, state, mode, str(e)))
            finally:
                if alarm_armed:
                    signal.alarm(0)
                if alarm_installed:
                    signal.signal(signal.SIGALRM, previous_alarm_handler)
                if fork_snapshots and attempt_infrastructure_failure is None:
                    try:
                        restore_simulated_snapshots(web3config, fork_snapshots)
                        # The next attempt starts on the source chain. Reset the
                        # single in-memory signer nonce against that chain after
                        # its fork snapshot was restored. Destination routing
                        # creates and synchronises its chain-specific wallet at
                        # the point of use.
                        hot_wallet = execution_model.tx_builder.hot_wallet
                        hot_wallet.current_nonce = None
                        hot_wallet.sync_nonce(web3config.get_default())
                    except BaseException as cleanup_error:
                        if auto_simulated and is_simulated_infrastructure_failure(
                            cleanup_error
                        ):
                            restart_requested = cleanup_error
                            logger.warning(
                                "Discarding Anvil generation after snapshot restoration failed for %s: %s",
                                spec.as_string_id(),
                                cleanup_error,
                            )
                        else:
                            raise

        print(tabulate(rows, headers="keys", tablefmt="rounded_outline"))
    finally:
        if simulated_runtime is not None:
            simulated_runtime.close()
        else:
            _close_web3config_safely(web3config)
