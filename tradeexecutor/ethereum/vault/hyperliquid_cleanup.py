"""Hyperliquid closed-position clean-up helpers.

This module contains operator tooling for one specific recovery path:

1. A Hypercore vault close was attempted from a Lagoon Safe
2. The withdrawal trade failed in state
3. In reality, the capital has already left the Hypercore vault
4. The USDC is stranded on HyperCore ``perp`` and/or ``spot``
5. We need to recover the stranded USDC back to the Safe on HyperEVM
6. We then repair and correct the strategy state so accounting matches reality

Safety model
------------

This tool is intentionally narrower than the live Hypercore routing logic.
It must not try to close genuinely open live vault positions.

The clean-up flow therefore only ever performs these actions:

- ``transferUsdClass(perp -> spot)``
- ``spotSend(spot -> EVM Safe)``

It never performs ``vaultTransfer(vault -> perp)``.

If the failed close candidate still has meaningful live vault equity on
Hyperliquid, this module aborts and tells the operator to review the
position manually.
"""

import datetime
import logging
import os
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from eth_defi.compat import native_datetime_utc_now
from eth_defi.erc_4626.vault_protocol.lagoon.vault import LagoonVault
from eth_defi.hotwallet import HotWallet
from eth_defi.hyperliquid.api import (HyperliquidSession, UserVaultEquity,
                                      fetch_perp_clearinghouse_state,
                                      fetch_spot_clearinghouse_state,
                                      fetch_user_vault_equities)
from eth_defi.hyperliquid.core_writer import (
    build_hypercore_send_asset_to_evm_call,
    build_hypercore_transfer_usd_class_call)
from eth_defi.hyperliquid.session import (HYPERLIQUID_API_URL,
                                          HYPERLIQUID_TESTNET_API_URL,
                                          create_hyperliquid_session)
from eth_defi.provider.broken_provider import get_almost_latest_block_number
from eth_defi.token import TokenDetails
from eth_defi.trace import assert_transaction_success_with_explanation
from tabulate import tabulate
from web3 import Web3

from tradeexecutor.cli.bootstrap import (backup_state, create_client,
                                         create_execution_and_sync_model,
                                         create_state_store, create_sync_model,
                                         create_web3_config, prepare_cache,
                                         prepare_executor_id)
from tradeexecutor.cli.log import setup_logging
from tradeexecutor.ethereum.lagoon.vault import LagoonVaultSyncModel
from tradeexecutor.state.repair import repair_trades
from tradeexecutor.state.state import State
from tradeexecutor.state.store import JSONFileStore
from tradeexecutor.strategy.account_correction import (
    UnknownTokenPositionFix, calculate_account_corrections, check_accounts,
    check_state_internal_coherence, correct_accounts)
from tradeexecutor.strategy.bootstrap import make_factory_from_strategy_mod
from tradeexecutor.strategy.description import StrategyExecutionDescription
from tradeexecutor.strategy.execution_context import (ExecutionContext,
                                                      ExecutionMode)
from tradeexecutor.strategy.execution_model import AssetManagementMode
from tradeexecutor.strategy.run_state import RunState
from tradeexecutor.strategy.strategy_module import (StrategyModuleInformation,
                                                    read_strategy_module)
from tradeexecutor.strategy.trading_strategy_universe import \
    TradingStrategyUniverseModel
from tradeexecutor.strategy.universe_model import UniverseOptions

logger = logging.getLogger(__name__)

BALANCE_TOLERANCE = Decimal("0.02")
RESIDUAL_VAULT_EQUITY_THRESHOLD = Decimal("0.10")
POLL_INTERVAL = 2.0
BALANCE_TIMEOUT = 60.0
SAFE_GAS_LIMIT = 650_000


@dataclass(slots=True)
class HyperliquidCleanupContext:
    """Resolved execution context for one clean-up run."""

    state_file: Path
    strategy_file: Path
    store: JSONFileStore
    state: State
    web3: Web3
    hot_wallet: HotWallet
    lagoon_vault: LagoonVault
    sync_model: LagoonVaultSyncModel
    session: HyperliquidSession
    reserve_token: TokenDetails
    trading_strategy_api_key: str
    json_rpc_hyperliquid: str
    cache_path: Path | None
    unit_testing: bool


@dataclass(slots=True)
class HyperliquidCleanupSnapshot:
    """Live Safe balances observed from HyperEVM and Hyperliquid."""

    safe_address: str
    evm_usdc_balance: Decimal
    spot_total_usdc: Decimal
    spot_free_usdc: Decimal
    perp_withdrawable: Decimal
    perp_account_value: Decimal
    perp_position_count: int
    vault_equities: dict[str, Decimal]


@dataclass(slots=True)
class HyperliquidStatePositionSnapshot:
    """State-side summary for one Hypercore vault position."""

    position_id: int
    state_status: str
    quantity: Decimal
    vault_address: str
    vault_name: str
    failed_trade_ids: list[int]
    stranded_metadata: dict | None


@dataclass(slots=True)
class HyperliquidStateSnapshot:
    """Relevant state-side balances and positions."""

    reserve_quantity: Decimal
    positions: list[HyperliquidStatePositionSnapshot]


@dataclass(slots=True)
class HyperliquidCleanupComparisonRow:
    """State versus live-reality comparison row."""

    position_id: int
    state_status: str
    vault_name: str
    vault_address: str
    state_quantity: Decimal
    live_vault_equity: Decimal
    reserve_quantity: Decimal
    spot_free_usdc: Decimal
    perp_withdrawable: Decimal
    perp_position_count: int
    failed_trade_ids: list[int]
    classification: str


@dataclass(slots=True)
class HyperliquidCleanupAction:
    """One Safe-side stranded-balance recovery action."""

    action_kind: str
    amount: Decimal
    reason: str


@dataclass(slots=True)
class HyperliquidCleanupReport:
    """Final clean-up report."""

    reality_rows: list[dict[str, str]]
    comparison_rows: list[HyperliquidCleanupComparisonRow]
    planned_actions: list[HyperliquidCleanupAction]
    executed_action_kinds: list[str]
    accounts_clean: bool
    state_saved: bool


@dataclass(slots=True)
class HyperliquidAccountingContext:
    """Objects needed to run Python-side account correction."""

    pair_universe: object
    reserve_assets: list
    sync_model: LagoonVaultSyncModel
    tx_builder: object
    strategy_universe: object
    pricing_model: object | None


def _is_within_tolerance(left: Decimal, right: Decimal) -> bool:
    """Check whether two balances are close enough."""
    return abs(left - right) <= BALANCE_TOLERANCE


def _position_vault_address(position) -> str:
    """Get the Hypercore vault address for a state position."""
    address = position.pair.pool_address or position.pair.base.address
    return Web3.to_checksum_address(address)


def _get_spot_usdc_balances(spot_state) -> tuple[Decimal, Decimal]:
    """Extract total and free spot USDC from HyperCore state."""
    for balance in spot_state.balances:
        if balance.coin == "USDC":
            return balance.total, balance.total - balance.hold
    return Decimal(0), Decimal(0)


def _fetch_live_cleanup_snapshot(
    context: HyperliquidCleanupContext,
) -> HyperliquidCleanupSnapshot:
    """Read live Safe balances from HyperEVM and Hyperliquid."""
    safe_address = context.lagoon_vault.safe_address
    spot_state = fetch_spot_clearinghouse_state(context.session, user=safe_address)
    perp_state = fetch_perp_clearinghouse_state(context.session, user=safe_address)
    vault_equities = {
        Web3.to_checksum_address(entry.vault_address): entry.equity
        for entry in fetch_user_vault_equities(context.session, user=safe_address)
    }
    spot_total_usdc, spot_free_usdc = _get_spot_usdc_balances(spot_state)
    return HyperliquidCleanupSnapshot(
        safe_address=safe_address,
        evm_usdc_balance=context.reserve_token.fetch_balance_of(safe_address),
        spot_total_usdc=spot_total_usdc,
        spot_free_usdc=spot_free_usdc,
        perp_withdrawable=perp_state.withdrawable,
        perp_account_value=perp_state.margin_summary.account_value,
        perp_position_count=len(perp_state.asset_positions),
        vault_equities=vault_equities,
    )


def _build_state_snapshot(state: State) -> HyperliquidStateSnapshot:
    """Read the relevant Hypercore positions from state."""
    reserve_quantity = Decimal(0)
    if state.portfolio.reserves:
        reserve_quantity = state.portfolio.get_default_reserve_position().quantity

    positions: list[HyperliquidStatePositionSnapshot] = []
    for position in state.portfolio.get_open_and_frozen_positions():
        if not position.is_vault():
            continue
        if position.pair.other_data.get("vault_protocol") != "hyperliquid":
            continue

        failed_trade_ids = [
            trade.trade_id
            for trade in position.trades.values()
            if trade.is_sell() and trade.is_failed()
        ]
        stranded_metadata = None
        for trade in position.trades.values():
            data = getattr(trade, "other_data", None) or {}
            if "hypercore_stranded_usdc" in data:
                stranded_metadata = data["hypercore_stranded_usdc"]
                break

        positions.append(
            HyperliquidStatePositionSnapshot(
                position_id=position.position_id,
                state_status="frozen" if position.is_frozen() else "open",
                quantity=position.get_quantity(),
                vault_address=_position_vault_address(position),
                vault_name=position.pair.other_data.get("vault_name", position.pair.get_ticker()),
                failed_trade_ids=failed_trade_ids,
                stranded_metadata=stranded_metadata,
            )
        )

    return HyperliquidStateSnapshot(
        reserve_quantity=reserve_quantity,
        positions=positions,
    )


def _classify_cleanup_row(
    position: HyperliquidStatePositionSnapshot,
    live_snapshot: HyperliquidCleanupSnapshot,
) -> str:
    """Classify one state position against live reality."""
    live_vault_equity = live_snapshot.vault_equities.get(position.vault_address, Decimal(0))
    stranded_balance = live_snapshot.spot_free_usdc + live_snapshot.perp_withdrawable
    is_failed_close_candidate = position.state_status == "frozen" or bool(position.failed_trade_ids)
    residual_vault_equity = live_vault_equity <= RESIDUAL_VAULT_EQUITY_THRESHOLD
    stranded_balance_dominates = stranded_balance > live_vault_equity + BALANCE_TOLERANCE
    quantity_mismatch_is_large = position.quantity > live_vault_equity + BALANCE_TOLERANCE

    if live_snapshot.perp_position_count > 0:
        return "manual_review_required_active_perp_positions"

    if (
        position.state_status == "open"
        and not is_failed_close_candidate
        and residual_vault_equity
        and stranded_balance > BALANCE_TOLERANCE
        and stranded_balance_dominates
        and quantity_mismatch_is_large
    ):
        return "open_in_state_residual_vault_equity_stranded"

    if not is_failed_close_candidate:
        return "manual_review_required"

    if live_vault_equity > RESIDUAL_VAULT_EQUITY_THRESHOLD:
        return "live_vault_open_no_action"

    if live_snapshot.spot_free_usdc > BALANCE_TOLERANCE and live_snapshot.perp_withdrawable > BALANCE_TOLERANCE:
        return "closed_in_reality_spot_and_perp_stranded"

    if live_snapshot.perp_withdrawable > BALANCE_TOLERANCE:
        return "closed_in_reality_perp_stranded"

    if live_snapshot.spot_free_usdc > BALANCE_TOLERANCE:
        return "closed_in_reality_spot_stranded"

    return "nothing_to_recover"


def _compare_state_to_reality(
    state_snapshot: HyperliquidStateSnapshot,
    live_snapshot: HyperliquidCleanupSnapshot,
) -> list[HyperliquidCleanupComparisonRow]:
    """Build state-versus-reality comparison rows."""
    rows: list[HyperliquidCleanupComparisonRow] = []
    for position in state_snapshot.positions:
        classification = _classify_cleanup_row(position, live_snapshot)
        rows.append(
            HyperliquidCleanupComparisonRow(
                position_id=position.position_id,
                state_status=position.state_status,
                vault_name=position.vault_name,
                vault_address=position.vault_address,
                state_quantity=position.quantity,
                live_vault_equity=live_snapshot.vault_equities.get(position.vault_address, Decimal(0)),
                reserve_quantity=state_snapshot.reserve_quantity,
                spot_free_usdc=live_snapshot.spot_free_usdc,
                perp_withdrawable=live_snapshot.perp_withdrawable,
                perp_position_count=live_snapshot.perp_position_count,
                failed_trade_ids=position.failed_trade_ids,
                classification=classification,
            )
        )
    return rows


def _plan_cleanup_actions(
    comparison_rows: list[HyperliquidCleanupComparisonRow],
    live_snapshot: HyperliquidCleanupSnapshot,
) -> list[HyperliquidCleanupAction]:
    """Create a safe action plan from current live balances."""
    for row in comparison_rows:
        if row.classification == "live_vault_open_no_action":
            raise RuntimeError(
                f"Refusing clean-up for position {row.position_id}: live vault equity is still "
                f"{row.live_vault_equity}, which is above the residual threshold "
                f"{RESIDUAL_VAULT_EQUITY_THRESHOLD}. This looks like a genuinely open real vault position."
            )
        if row.classification == "manual_review_required_active_perp_positions":
            raise RuntimeError(
                f"Refusing clean-up for position {row.position_id}: manual review required because "
                f"the Safe still has {row.perp_position_count} active HyperCore perp position(s)."
            )

    if not comparison_rows:
        return []

    actionable = {
        "closed_in_reality_spot_and_perp_stranded",
        "closed_in_reality_perp_stranded",
        "closed_in_reality_spot_stranded",
        "open_in_state_residual_vault_equity_stranded",
        "nothing_to_recover",
    }
    if not any(row.classification in actionable for row in comparison_rows):
        raise RuntimeError(
            "No recognised stranded-balance failed-close Hypercore position was found for clean-up."
        )

    actions: list[HyperliquidCleanupAction] = []
    if live_snapshot.perp_withdrawable > BALANCE_TOLERANCE:
        actions.append(
            HyperliquidCleanupAction(
                action_kind="perp_to_spot",
                amount=live_snapshot.perp_withdrawable,
                reason="Recover stranded HyperCore perp USDC back to HyperCore spot",
            )
        )

    total_spot_to_recover = live_snapshot.spot_free_usdc
    if live_snapshot.perp_withdrawable > BALANCE_TOLERANCE:
        total_spot_to_recover += live_snapshot.perp_withdrawable

    if total_spot_to_recover > BALANCE_TOLERANCE:
        actions.append(
            HyperliquidCleanupAction(
                action_kind="spot_to_evm",
                amount=total_spot_to_recover,
                reason="Recover stranded HyperCore spot USDC back to the Safe on HyperEVM",
            )
        )

    return actions


def _print_reality_table(live_snapshot: HyperliquidCleanupSnapshot) -> list[dict[str, str]]:
    """Render the live balance table."""
    rows = [
        {
            "Safe": live_snapshot.safe_address,
            "EVM USDC": f"{live_snapshot.evm_usdc_balance:.6f}",
            "Spot total": f"{live_snapshot.spot_total_usdc:.6f}",
            "Spot free": f"{live_snapshot.spot_free_usdc:.6f}",
            "Perp withdrawable": f"{live_snapshot.perp_withdrawable:.6f}",
            "Perp account": f"{live_snapshot.perp_account_value:.6f}",
            "Perp positions": live_snapshot.perp_position_count,
            "Vault equity total": f"{sum(live_snapshot.vault_equities.values(), start=Decimal(0)):.6f}",
        }
    ]
    print("\nReality")
    print(tabulate(rows, headers="keys", tablefmt="simple"))

    if live_snapshot.vault_equities:
        vault_rows = [
            {
                "Vault": address,
                "Equity": f"{equity:.6f}",
            }
            for address, equity in live_snapshot.vault_equities.items()
        ]
        print("\nReality vault balances")
        print(tabulate(vault_rows, headers="keys", tablefmt="simple"))

    return rows


def _print_state_comparison_table(
    comparison_rows: list[HyperliquidCleanupComparisonRow],
) -> None:
    """Render the state-versus-reality table."""
    rows = [
        {
            "Position": row.position_id,
            "State": row.state_status,
            "Vault": row.vault_name,
            "State qty": f"{row.state_quantity:.6f}",
            "Live vault eq": f"{row.live_vault_equity:.6f}",
            "Reserve": f"{row.reserve_quantity:.6f}",
            "Spot free": f"{row.spot_free_usdc:.6f}",
            "Perp withdr": f"{row.perp_withdrawable:.6f}",
            "Perp pos": row.perp_position_count,
            "Failed trades": ",".join(str(t) for t in row.failed_trade_ids) or "-",
            "Classification": row.classification,
        }
        for row in comparison_rows
    ]
    print("\nState vs reality")
    print(tabulate(rows, headers="keys", tablefmt="simple"))
    if any(row.classification == "open_in_state_residual_vault_equity_stranded" for row in comparison_rows):
        print(
            "\nDetected an open-in-state, effectively closed-in-reality recovery case: "
            "the live vault equity is only residual, there are no active perp positions, "
            "and the meaningful capital is stranded in HyperCore spot/perp."
        )


def _print_action_table(actions: list[HyperliquidCleanupAction]) -> None:
    """Render planned recovery actions."""
    rows = [
        {
            "Action": action.action_kind,
            "Amount": f"{action.amount:.6f}",
            "Reason": action.reason,
        }
        for action in actions
    ]
    print("\nPlanned actions")
    if rows:
        print(tabulate(rows, headers="keys", tablefmt="simple"))
    else:
        print("No Safe-side recovery transactions are needed; proceeding to state repair only.")


def _confirm_cleanup(auto_approve: bool) -> None:
    """Ask for operator confirmation before any mutation."""
    if auto_approve:
        return

    confirmation = input("Execute Hyperliquid clean-up actions and state repair [y/n] ").strip().lower()
    if confirmation != "y":
        raise RuntimeError("Operator aborted Hyperliquid clean-up")


def _wait_for_spot_free_balance(
    session: HyperliquidSession,
    user: str,
    expected_balance: Decimal,
    timeout: float = BALANCE_TIMEOUT,
    poll_interval: float = POLL_INTERVAL,
) -> Decimal:
    """Wait until free spot USDC reaches the expected balance."""
    deadline = time.time() + timeout
    while True:
        spot_state = fetch_spot_clearinghouse_state(session, user=user)
        _spot_total, spot_free = _get_spot_usdc_balances(spot_state)
        if _is_within_tolerance(spot_free, expected_balance):
            return spot_free
        if time.time() >= deadline:
            raise AssertionError(
                f"Timed out waiting for HyperCore free spot USDC {expected_balance} for {user}, "
                f"last observed balance was {spot_free}"
            )
        time.sleep(poll_interval)


def _wait_for_evm_usdc_balance(
    token: TokenDetails,
    address: str,
    expected_balance: Decimal,
    timeout: float = BALANCE_TIMEOUT,
    poll_interval: float = POLL_INTERVAL,
) -> Decimal:
    """Wait until EVM USDC reaches the expected balance."""
    deadline = time.time() + timeout
    while True:
        balance = token.fetch_balance_of(address)
        if _is_within_tolerance(balance, expected_balance):
            return balance
        if time.time() >= deadline:
            raise AssertionError(
                f"Timed out waiting for HyperEVM USDC {expected_balance} for {address}, "
                f"last observed balance was {balance}"
            )
        time.sleep(poll_interval)


def _broadcast_bound_call(
    web3: Web3,
    hot_wallet: HotWallet,
    bound_func,
    gas_limit: int = SAFE_GAS_LIMIT,
) -> str:
    """Broadcast a single Safe/module transaction and assert success."""
    tx_hash = hot_wallet.transact_and_broadcast_with_contract(bound_func, gas_limit=gas_limit)
    assert_transaction_success_with_explanation(web3, tx_hash)
    return tx_hash.hex()


def _execute_perp_to_spot(
    context: HyperliquidCleanupContext,
    amount: Decimal,
) -> str:
    """Recover stranded USDC from HyperCore perp back to spot."""
    safe_address = context.lagoon_vault.safe_address
    live_snapshot = _fetch_live_cleanup_snapshot(context)
    assert live_snapshot.perp_withdrawable + BALANCE_TOLERANCE >= amount, (
        f"Before transferUsdClass(perp->spot), Safe {safe_address} perp withdrawable balance is "
        f"{live_snapshot.perp_withdrawable}, expected at least {amount}"
    )

    expected_spot_free = live_snapshot.spot_free_usdc + amount
    fn = build_hypercore_transfer_usd_class_call(
        context.lagoon_vault,
        hypercore_usdc_amount=context.reserve_token.convert_to_raw(amount),
        to_perp=False,
    )
    tx_hash = _broadcast_bound_call(context.web3, context.hot_wallet, fn)
    _wait_for_spot_free_balance(context.session, safe_address, expected_spot_free)
    return tx_hash


def _execute_spot_to_evm(
    context: HyperliquidCleanupContext,
    amount: Decimal,
) -> str:
    """Recover stranded USDC from HyperCore spot back to HyperEVM."""
    safe_address = context.lagoon_vault.safe_address
    live_snapshot = _fetch_live_cleanup_snapshot(context)
    assert live_snapshot.spot_free_usdc + BALANCE_TOLERANCE >= amount, (
        f"Before spotSend(spot->EVM), Safe {safe_address} free spot USDC balance is "
        f"{live_snapshot.spot_free_usdc}, expected at least {amount}"
    )

    expected_evm_balance = live_snapshot.evm_usdc_balance + amount
    fn = build_hypercore_send_asset_to_evm_call(
        context.lagoon_vault,
        evm_usdc_amount=context.reserve_token.convert_to_raw(amount),
    )
    tx_hash = _broadcast_bound_call(context.web3, context.hot_wallet, fn)
    _wait_for_evm_usdc_balance(context.reserve_token, safe_address, expected_evm_balance)
    return tx_hash


def _execute_cleanup_actions(
    context: HyperliquidCleanupContext,
    actions: list[HyperliquidCleanupAction],
) -> list[str]:
    """Execute Safe-side recovery actions in order."""
    executed: list[str] = []
    for action in actions:
        if action.action_kind == "perp_to_spot":
            _execute_perp_to_spot(context, action.amount)
        elif action.action_kind == "spot_to_evm":
            _execute_spot_to_evm(context, action.amount)
        else:
            raise NotImplementedError(f"Unsupported Hyperliquid clean-up action: {action.action_kind}")
        executed.append(action.action_kind)
    return executed


def _build_accounting_correction_context(
    context: HyperliquidCleanupContext,
) -> HyperliquidAccountingContext:
    """Construct the objects needed for Python-side account correction."""
    mod: StrategyModuleInformation = read_strategy_module(context.strategy_file)
    executor_id = prepare_executor_id(None, context.strategy_file)
    cache_path = prepare_cache(executor_id, context.cache_path, unit_testing=context.unit_testing)

    if context.web3.eth.chain_id == 998:
        web3config = create_web3_config(
            None, None, None, None, None, None, None,
            json_rpc_hyperliquid_testnet=context.json_rpc_hyperliquid,
            unit_testing=context.unit_testing,
        )
    else:
        web3config = create_web3_config(
            None, None, None, None, None, None, None,
            json_rpc_hyperliquid=context.json_rpc_hyperliquid,
            unit_testing=context.unit_testing,
        )
    web3config.choose_single_chain()

    execution_model, sync_model, valuation_model_factory, pricing_model_factory = create_execution_and_sync_model(
        asset_management_mode=AssetManagementMode.lagoon,
        private_key=context.hot_wallet.private_key.hex(),
        web3config=web3config,
        confirmation_timeout=datetime.timedelta(seconds=60),
        confirmation_block_count=0,
        max_slippage=0.013,
        min_gas_balance=Decimal(0),
        vault_address=context.lagoon_vault.address,
        vault_adapter_address=context.lagoon_vault.trading_strategy_module_address,
        routing_hint=mod.trade_routing,
    )

    client, _routing_model = create_client(
        mod=mod,
        web3config=web3config,
        trading_strategy_api_key=context.trading_strategy_api_key,
        cache_path=cache_path,
        test_evm_uniswap_v2_factory=None,
        test_evm_uniswap_v2_router=None,
        test_evm_uniswap_v2_init_code_hash=None,
        clear_caches=False,
        asset_management_mode=AssetManagementMode.lagoon,
    )
    assert client is not None, "TRADING_STRATEGY_API_KEY is required to build the strategy universe for clean-up"

    execution_context = ExecutionContext(
        mode=ExecutionMode.one_off,
        engine_version=mod.trading_strategy_engine_version,
    )
    strategy_factory = make_factory_from_strategy_mod(mod)
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
    strategy_universe = universe_model.construct_universe(
        native_datetime_utc_now(),
        execution_context.mode,
        UniverseOptions(history_period=mod.get_live_trading_history_period()),
        execution_model=run_description.runner.execution_model,
        strategy_parameters=mod.parameters,
    )

    runner = run_description.runner
    pricing_model = None
    if mod.is_version_greater_or_equal_than(0, 5, 0) and mod.trade_routing is not None:
        _routing_state, pricing_model, _valuation_method = runner.setup_routing(strategy_universe)

    return HyperliquidAccountingContext(
        pair_universe=strategy_universe.data_universe.pairs,
        reserve_assets=list(strategy_universe.reserve_assets),
        sync_model=sync_model,
        tx_builder=sync_model.create_transaction_builder(),
        strategy_universe=strategy_universe,
        pricing_model=pricing_model,
    )


def _repair_and_correct_state(
    context: HyperliquidCleanupContext,
) -> tuple[bool, bool]:
    """Repair failed trades, correct accounts, and save when clean."""
    store, state = backup_state(context.state_file, unit_testing=context.unit_testing)
    repair_trades(
        state,
        attempt_repair=True,
        interactive=False,
    )

    accounting_context = _build_accounting_correction_context(context)
    block_identifier = get_almost_latest_block_number(accounting_context.sync_model.web3)
    corrections = list(
        calculate_account_corrections(
            accounting_context.pair_universe,
            accounting_context.reserve_assets,
            state,
            accounting_context.sync_model,
            relative_epsilon=None,
            all_balances=False,
            block_identifier=block_identifier,
        )
    )

    list(
        correct_accounts(
            state,
            corrections,
            strategy_cycle_included_at=native_datetime_utc_now(),
            tx_builder=accounting_context.tx_builder,
            interactive=False,
            unknown_token_receiver=None,
            block_identifier=block_identifier,
            block_timestamp=None,
            token_fix_method=UnknownTokenPositionFix.open_missing_position,
            strategy_universe=accounting_context.strategy_universe,
            pricing_model=accounting_context.pricing_model,
        )
    )

    check_state_internal_coherence(state)
    accounts_clean, dataframe = check_accounts(
        accounting_context.pair_universe,
        accounting_context.reserve_assets,
        state,
        accounting_context.sync_model,
        block_identifier=block_identifier,
    )
    if not accounts_clean:
        logger.error("Final account check is still unclean:\n%s", dataframe)
        return False, False

    store.sync(state)
    return True, True


def _resolve_hyperliquid_api_url(network: str | None, chain_id: int) -> str:
    """Choose Hyperliquid API URL from env hints."""
    if network == "testnet" or chain_id == 998:
        return HYPERLIQUID_TESTNET_API_URL
    return HYPERLIQUID_API_URL


def load_cleanup_context(
    *,
    state_file: Path,
    strategy_file: Path,
    private_key: str,
    json_rpc_hyperliquid: str,
    vault_address: str,
    vault_adapter_address: str,
    trading_strategy_api_key: str,
    network: str = "mainnet",
    cache_path: Path | None = None,
    unit_testing: bool = False,
    log_level: str = "info",
) -> HyperliquidCleanupContext:
    """Create the clean-up execution context from explicit parameters."""
    setup_logging(log_level)

    hot_wallet = HotWallet.from_private_key(private_key)
    web3config = create_web3_config(
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        json_rpc_hyperliquid=json_rpc_hyperliquid,
        unit_testing=unit_testing,
    )
    web3config.choose_single_chain()
    web3 = web3config.get_default()
    hot_wallet.sync_nonce(web3)

    sync_model = create_sync_model(
        AssetManagementMode.lagoon,
        web3,
        hot_wallet,
        vault_address=vault_address,
        vault_adapter_address=vault_adapter_address,
        unit_testing=unit_testing,
    )
    assert isinstance(sync_model, LagoonVaultSyncModel)
    store = create_state_store(state_file)
    assert isinstance(store, JSONFileStore)
    state = store.load()
    lagoon_vault = sync_model.vault
    reserve_token = lagoon_vault.underlying_token
    session = create_hyperliquid_session(
        api_url=_resolve_hyperliquid_api_url(network, web3.eth.chain_id),
    )

    return HyperliquidCleanupContext(
        state_file=state_file,
        strategy_file=strategy_file,
        store=store,
        state=state,
        web3=web3,
        hot_wallet=hot_wallet,
        lagoon_vault=lagoon_vault,
        sync_model=sync_model,
        session=session,
        reserve_token=reserve_token,
        trading_strategy_api_key=trading_strategy_api_key,
        json_rpc_hyperliquid=json_rpc_hyperliquid,
        cache_path=cache_path,
        unit_testing=unit_testing,
    )


def run_hyperliquid_cleanup(
    *,
    state_file: Path,
    strategy_file: Path,
    private_key: str,
    json_rpc_hyperliquid: str,
    vault_address: str,
    vault_adapter_address: str,
    trading_strategy_api_key: str,
    network: str = "mainnet",
    auto_approve: bool = False,
    cache_path: Path | None = None,
    unit_testing: bool = False,
    log_level: str = "info",
) -> HyperliquidCleanupReport:
    """Run the closed-position Hyperliquid clean-up flow end to end.

    Example how to call this from the trade-executor console using the
    preloaded console bindings::

        import os
        from pathlib import Path
        from tradeexecutor.ethereum.vault.hyperliquid_cleanup import run_hyperliquid_cleanup

        run_hyperliquid_cleanup(
            state_file=Path(store.path),
            strategy_file=Path(os.environ["STRATEGY_FILE"]),
            private_key=os.environ["PRIVATE_KEY"],
            json_rpc_hyperliquid=os.environ["JSON_RPC_HYPERLIQUID"],
            vault_address=vault.address,
            vault_adapter_address=vault.trading_strategy_module_address,
            trading_strategy_api_key=os.environ["TRADING_STRATEGY_API_KEY"],
            network=os.environ.get("NETWORK", "mainnet"),
            auto_approve=False,
        )
    """
    context = load_cleanup_context(
        state_file=state_file,
        strategy_file=strategy_file,
        private_key=private_key,
        json_rpc_hyperliquid=json_rpc_hyperliquid,
        vault_address=vault_address,
        vault_adapter_address=vault_adapter_address,
        trading_strategy_api_key=trading_strategy_api_key,
        network=network,
        cache_path=cache_path,
        unit_testing=unit_testing,
        log_level=log_level,
    )
    live_snapshot = _fetch_live_cleanup_snapshot(context)
    state_snapshot = _build_state_snapshot(context.state)
    comparison_rows = _compare_state_to_reality(state_snapshot, live_snapshot)
    reality_rows = _print_reality_table(live_snapshot)
    _print_state_comparison_table(comparison_rows)
    planned_actions = _plan_cleanup_actions(comparison_rows, live_snapshot)
    _print_action_table(planned_actions)
    _confirm_cleanup(auto_approve=auto_approve)
    executed_action_kinds = _execute_cleanup_actions(context, planned_actions)
    accounts_clean, state_saved = _repair_and_correct_state(context)
    return HyperliquidCleanupReport(
        reality_rows=reality_rows,
        comparison_rows=comparison_rows,
        planned_actions=planned_actions,
        executed_action_kinds=executed_action_kinds,
        accounts_clean=accounts_clean,
        state_saved=state_saved,
    )


def run_hyperliquid_cleanup_from_environment() -> HyperliquidCleanupReport:
    """Run clean-up using standard trade-executor environment variables.

    Example how to call from console::

        from tradeexecutor.ethereum.vault.hyperliquid_cleanup import run_hyperliquid_cleanup_from_environment
        run_hyperliquid_cleanup_from_environment()
    """
    state_file = Path(os.environ["STATE_FILE"])
    strategy_file = Path(os.environ["STRATEGY_FILE"])
    private_key = os.environ["PRIVATE_KEY"]
    json_rpc_hyperliquid = os.environ["JSON_RPC_HYPERLIQUID"]
    vault_address = os.environ["VAULT_ADDRESS"]
    vault_adapter_address = os.environ["VAULT_ADAPTER_ADDRESS"]
    trading_strategy_api_key = os.environ.get("TRADING_STRATEGY_API_KEY", "")
    network = os.environ.get("NETWORK", "mainnet")
    auto_approve = os.environ.get("AUTO_APPROVE", "").lower() in {"1", "true", "yes"}
    cache_path = Path(os.environ["CACHE_PATH"]) if os.environ.get("CACHE_PATH") else None
    unit_testing = os.environ.get("UNIT_TESTING", "").lower() in {"1", "true", "yes"}
    log_level = os.environ.get("LOG_LEVEL", "info")
    return run_hyperliquid_cleanup(
        state_file=state_file,
        strategy_file=strategy_file,
        private_key=private_key,
        json_rpc_hyperliquid=json_rpc_hyperliquid,
        vault_address=vault_address,
        vault_adapter_address=vault_adapter_address,
        trading_strategy_api_key=trading_strategy_api_key,
        network=network,
        auto_approve=auto_approve,
        cache_path=cache_path,
        unit_testing=unit_testing,
        log_level=log_level,
    )
