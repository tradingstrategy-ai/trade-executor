"""Move USDC manually between a Lagoon Safe and its Lighter account.

This command is an operator diagnostic and recovery tool. It records verified
custody movements in the executor state, but it does not execute strategy
trades or settle Lagoon investor flows.
"""

import asyncio
import logging
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import typer
from eth_defi.compat import native_datetime_utc_now
from eth_defi.erc_4626.vault_protocol.lagoon.vault import LagoonVault
from eth_defi.hotwallet import HotWallet
from eth_defi.lighter.api import (
    LIGHTER_MIN_MAINNET_USDC,
    fetch_lighter_withdrawal_delay,
    wait_for_lighter_collateral,
)
from eth_defi.lighter.constants import (
    LIGHTER_ETHEREUM_DEPLOYMENT_CHAIN_ID,
    LIGHTER_USDC_ETHEREUM,
)
from eth_defi.lighter.lagoon import (
    claim_usdc_to_lagoon_safe_from_lighter,
    deposit_usdc_from_lagoon_safe_into_lighter,
)
from eth_defi.lighter.session import LighterSession, create_lighter_session
from eth_defi.lighter.valuation import (
    LighterEquity,
    fetch_lighter_account_by_index,
    fetch_lighter_total_equity,
)
from eth_defi.token import TokenDetails, fetch_erc20_details
from eth_defi.vault.base import VaultSpec
from typer import Option

from tradeexecutor.cli.bootstrap import backup_state, prepare_executor_id
from tradeexecutor.cli.commands import shared_options
from tradeexecutor.cli.commands.app import app
from tradeexecutor.cli.commands.lagoon_lighter_test_trade import (
    DEFAULT_LIGHTER_DEPOSIT_TIMEOUT,
    DEFAULT_LIGHTER_WITHDRAW_TIMEOUT,
    LighterOperatorRecord,
    LighterWithdrawalRejected,
    load_lighter_operator_record,
    request_lighter_withdrawal,
    validate_public_deployment_record,
    wait_for_lighter_withdrawal_claimable,
)
from tradeexecutor.cli.commands.lagoon_utils import (
    create_hot_wallet,
    create_single_chain_web3_config,
    resolve_state_store,
)
from tradeexecutor.cli.log import setup_logging
from tradeexecutor.exchange_account.lighter import LIGHTER_PROTOCOL
from tradeexecutor.exchange_account.state import (
    ExchangeAccountTransferError,
    create_exchange_account_transfer,
    mark_exchange_account_transfer_broadcasted,
    record_exchange_account_transfer,
)
from tradeexecutor.exchange_account.sync_model import ExchangeAccountSyncModel
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradeexecutor.state.store import JSONFileStore
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.account_correction import is_relative_mismatch
from tradeexecutor.strategy.dust import (
    get_close_epsilon_for_pair,
    get_dust_epsilon_for_asset,
    get_relative_epsilon_for_asset,
    get_relative_epsilon_for_pair,
)
from tradeexecutor.strategy.execution_model import AssetManagementMode
from tradeexecutor.strategy.strategy_module import read_strategy_module


logger = logging.getLogger(__name__)


#: Lighter USDC amounts use six decimal places.
USDC_DECIMALS = Decimal("0.000001")


class LighterMoveFundsError(RuntimeError):
    """Public operator error for a manual Lighter custody movement."""


def _assert_non_negative(value: Decimal, label: str) -> None:
    """Reject invalid public monetary observations before state mutation."""
    if not value.is_finite() or value < 0:
        raise LighterMoveFundsError(f"Invalid {label}: value must be finite and non-negative")


def _parse_usdc_amount(value: str) -> Decimal:
    """Parse one positive native-USDC amount without binary floats."""
    try:
        amount = Decimal(value)
    except (InvalidOperation, ValueError) as error:
        raise LighterMoveFundsError("Amount in USDC must be a decimal value") from error
    if amount <= 0 or not amount.is_finite():
        raise LighterMoveFundsError("Amount in USDC must be positive")
    if amount.quantize(USDC_DECIMALS) != amount:
        raise LighterMoveFundsError("Amount in USDC can have at most six decimal places")
    return amount


def _validate_transfer_funds(
    direction: str,
    amount: Decimal,
    safe_balance: Decimal,
    equity: LighterEquity,
) -> None:
    """Check the latest custody balances can fund the requested movement."""
    if amount < LIGHTER_MIN_MAINNET_USDC:
        raise LighterMoveFundsError(
            f"Lighter transfers must be at least {LIGHTER_MIN_MAINNET_USDC} USDC",
        )
    if direction == "d" and amount > safe_balance:
        raise LighterMoveFundsError("Lagoon Safe has insufficient USDC for this deposit")
    if direction == "w" and amount > equity.available_balance:
        raise LighterMoveFundsError("Lighter available balance is insufficient for this withdrawal")


def _get_lighter_position(state: State, account_index: int) -> TradingPosition:
    """Return the one Lighter exchange-account position for this command."""
    positions = [
        position
        for position in state.portfolio.get_open_and_frozen_positions()
        if position.is_exchange_account()
        and position.pair.get_exchange_account_protocol() == LIGHTER_PROTOCOL
    ]
    if len(positions) != 1:
        raise LighterMoveFundsError(
            f"Expected exactly one Lighter exchange-account position, got {len(positions)}",
        )
    position = positions[0]
    if int(position.pair.get_exchange_account_id()) != account_index:
        raise LighterMoveFundsError("Executor state Lighter account does not match the operator record")
    return position


def _get_pending_transfer(position: TradingPosition) -> TradeExecution | None:
    """Return one unfinished Lighter custody withdrawal, if present."""
    pending = [
        trade
        for trade in position.trades.values()
        if trade.is_external_account_transfer_pending()
    ]
    if len(pending) > 1:
        raise LighterMoveFundsError("State has multiple unfinished Lighter custody transfers")
    return pending[0] if pending else None


def _assert_clean_custody_baseline(
    *,
    position: TradingPosition,
    equity: LighterEquity,
) -> None:
    """Require that accounting is reconciled before a new manual movement."""
    tracked_account = position.get_quantity()
    if is_relative_mismatch(
        equity.get_total(),
        tracked_account,
        get_relative_epsilon_for_pair(position.pair),
        get_close_epsilon_for_pair(position.pair),
    ):
        raise LighterMoveFundsError(
            "Lighter equity differs from executor state; run correct-accounts before moving funds",
        )


def _assert_reserve_matches_safe(state: State, reserve_asset: AssetIdentifier, safe_balance: Decimal) -> None:
    """Require that the recorded Lagoon reserve matches physical Safe USDC."""
    reserve = state.portfolio.get_reserve_position(reserve_asset)
    if is_relative_mismatch(
        safe_balance,
        reserve.quantity,
        get_relative_epsilon_for_asset(reserve_asset),
        get_dust_epsilon_for_asset(reserve_asset),
    ):
        raise LighterMoveFundsError(
            "Safe USDC differs from executor state; run correct-accounts before moving funds",
        )


def _get_position_metrics(account: dict[str, Any]) -> tuple[Decimal, Decimal]:
    """Return allocated margin and gross notional from Lighter position records."""
    allocated_margin = Decimal(0)
    gross_notional = Decimal(0)
    for position in account.get("positions") or []:
        allocated_margin += Decimal(str(position.get("allocated_margin", 0)))
        gross_notional += abs(Decimal(str(position.get("position_value", 0))))
    return allocated_margin, gross_notional


def _log_balances(
    *,
    state: State,
    reserve_asset: AssetIdentifier,
    safe_balance: Decimal,
    equity: LighterEquity,
    account: dict[str, Any],
    position: TradingPosition,
    heading: str,
) -> None:
    """Log the Safe and Lighter balance breakdown needed by an operator."""
    allocated_margin, gross_notional = _get_position_metrics(account)
    tracked_reserve = state.portfolio.get_reserve_position(reserve_asset).quantity
    logger.info("%s", heading)
    logger.info("  Safe USDC: %s (tracked reserve: %s)", safe_balance, tracked_reserve)
    logger.info("  Lighter collateral: %s", equity.collateral)
    logger.info("  Lighter total equity: %s", equity.get_total())
    logger.info("  Lighter available balance: %s", equity.available_balance)
    logger.info("  Lighter initial margin requirement: %s", equity.initial_margin_requirement)
    logger.info("  Lighter maintenance margin requirement: %s", equity.maintenance_margin_requirement)
    logger.info("  Lighter allocated margin: %s", allocated_margin)
    logger.info("  Lighter gross position notional: %s", gross_notional)
    logger.info("  Lighter unrealised PnL: %s", equity.unrealised_pnl)
    logger.info("  Lighter position records: %d", equity.position_count)
    logger.info("  Tracked Lighter account value: %s", position.get_quantity())


def _sync_lighter_position(
    *,
    state: State,
    equity: LighterEquity,
) -> None:
    """Record residual Lighter PnL after a verified custody movement."""
    sync_model = ExchangeAccountSyncModel(
        lambda _pair, **_kwargs: equity.get_total(),
    )
    sync_model.sync_positions(
        timestamp=native_datetime_utc_now(),
        state=state,
        strategy_universe=None,
        pricing_model=None,
    )


def _create_transfer_metadata(
    *,
    direction: str,
    operator: LighterOperatorRecord,
    amount: Decimal,
    safe_before: Decimal,
    equity_before: LighterEquity,
) -> dict[str, str | int]:
    """Build public-only metadata persisted with the synthetic trade."""
    return {
        "direction": direction,
        "protocol": LIGHTER_PROTOCOL,
        "account_index": operator.account_index,
        "requested_amount": str(amount),
        "safe_balance_before": str(safe_before),
        "lighter_equity_before": str(equity_before.get_total()),
        "lighter_collateral_before": str(equity_before.collateral),
        "requested_at": str(int(time.time())),
    }


def _append_transfer_metadata(trade: TradeExecution, **values: str | int) -> None:
    """Add public recovery metadata to an existing manual transfer trade."""
    metadata = trade.other_data or {}
    metadata.update(values)
    trade.other_data = metadata


def _get_transfer_amount(trade: TradeExecution) -> Decimal:
    """Read and validate a pending transfer amount from public metadata."""
    metadata = trade.other_data or {}
    try:
        return _parse_usdc_amount(str(metadata["requested_amount"]))
    except KeyError as error:
        raise LighterMoveFundsError("Unfinished Lighter transfer has no requested amount") from error


def _get_transfer_requested_at(trade: TradeExecution) -> int:
    """Read the public request time used to locate the Lighter history row."""
    metadata = trade.other_data or {}
    try:
        return int(metadata["requested_at"])
    except (KeyError, TypeError, ValueError) as error:
        raise LighterMoveFundsError("Unfinished Lighter transfer has no request time") from error


def _complete_transfer(
    *,
    state: State,
    store: JSONFileStore,
    position: TradingPosition,
    reserve_asset: AssetIdentifier,
    trade: TradeExecution,
    amount: Decimal,
    safe_before: Decimal,
    safe_after: Decimal,
    equity_after: LighterEquity,
) -> None:
    """Verify and persist one finished custody movement and residual PnL."""
    _assert_non_negative(safe_after, "Safe USDC balance")
    _assert_non_negative(equity_after.get_total(), "Lighter equity")
    expected_safe_change = -amount if trade.is_buy() else amount
    if safe_after - safe_before != expected_safe_change:
        raise LighterMoveFundsError("Safe USDC did not change by the verified Lighter transfer amount")
    _append_transfer_metadata(
        trade,
        safe_balance_after=str(safe_after),
        lighter_equity_after=str(equity_after.get_total()),
        lighter_collateral_after=str(equity_after.collateral),
        received_amount=str(amount),
    )
    record_exchange_account_transfer(
        state=state,
        position=position,
        trade=trade,
        reserve_currency=reserve_asset,
        amount=amount,
        executed_at=native_datetime_utc_now(),
    )
    _sync_lighter_position(state=state, equity=equity_after)
    _assert_reserve_matches_safe(state, reserve_asset, safe_after)
    if position.get_quantity() != equity_after.get_total():
        raise LighterMoveFundsError("Lighter valuation did not match the verified account equity")
    store.sync(state)


def _resume_withdrawal(
    *,
    state: State,
    store: JSONFileStore,
    position: TradingPosition,
    reserve_asset: AssetIdentifier,
    operator: LighterOperatorRecord,
    vault: LagoonVault,
    usdc: TokenDetails,
    hot_wallet: HotWallet,
    session: LighterSession,
    trade: TradeExecution,
    withdraw_timeout: int,
) -> None:
    """Resume one persisted secure withdrawal without resubmitting it."""
    metadata = trade.other_data or {}
    if metadata.get("direction") != "withdraw":
        raise LighterMoveFundsError("Only a pending Lighter withdrawal can be resumed")
    amount = _get_transfer_amount(trade)
    try:
        safe_before = Decimal(str(metadata["safe_balance_before"]))
    except (KeyError, InvalidOperation, ValueError) as error:
        raise LighterMoveFundsError("Unfinished Lighter transfer has no Safe balance baseline") from error
    safe_now = usdc.fetch_balance_of(vault.safe_address)
    if safe_now - safe_before == amount:
        logger.info("Detected an already claimed Lighter withdrawal; completing accounting")
        mark_exchange_account_transfer_broadcasted(trade, native_datetime_utc_now())
        equity_after = fetch_lighter_total_equity(session, operator.account_index)
        _complete_transfer(
            state=state,
            store=store,
            position=position,
            reserve_asset=reserve_asset,
            trade=trade,
            amount=amount,
            safe_before=safe_before,
            safe_after=safe_now,
            equity_after=equity_after,
        )
        return

    logger.info("Resuming secure Lighter withdrawal of %s USDC", amount)
    claimable = asyncio.run(
        wait_for_lighter_withdrawal_claimable(
            operator,
            amount,
            withdraw_timeout,
            _get_transfer_requested_at(trade),
        ),
    )
    if claimable != amount:
        raise LighterMoveFundsError("Lighter claimable withdrawal amount differs from the saved request")
    mark_exchange_account_transfer_broadcasted(trade, native_datetime_utc_now())
    store.sync(state)
    hot_wallet.sync_nonce(vault.web3)
    claim_tx_hash = claim_usdc_to_lagoon_safe_from_lighter(
        vault.web3,
        hot_wallet,
        vault=vault,
        usdc=usdc,
        claimable_usdc=claimable,
    )
    safe_after = usdc.fetch_balance_of(vault.safe_address)
    _append_transfer_metadata(trade, claim_tx_hash=claim_tx_hash, claimed_safe_credit=str(safe_after - safe_now))
    store.sync(state)
    equity_after = fetch_lighter_total_equity(session, operator.account_index)
    _complete_transfer(
        state=state,
        store=store,
        position=position,
        reserve_asset=reserve_asset,
        trade=trade,
        amount=amount,
        safe_before=safe_before,
        safe_after=safe_after,
        equity_after=equity_after,
    )


@app.command()
@shared_options.with_json_rpc_options()
def lighter_move_funds(
    id: str = shared_options.id,
    strategy_file: Path = shared_options.strategy_file,
    state_file: Path | None = shared_options.state_file,
    log_level: str | None = shared_options.log_level,
    rpc_kwargs: dict | None = None,
    private_key: str | None = shared_options.private_key,
    asset_management_mode: AssetManagementMode | None = shared_options.asset_management_mode,
    vault_address: str | None = shared_options.vault_address,
    vault_adapter_address: str | None = shared_options.vault_adapter_address,
    unit_testing: bool = shared_options.unit_testing,
    simulate: bool = shared_options.simulate,
    lighter_operator_record_file: Path | None = Option(
        None,
        envvar="LIGHTER_OPERATOR_RECORD_FILE",
        help="Mode-0600 JSON record emitted by lagoon-deploy-vault.",
    ),
    lighter_account_index: int | None = Option(
        None,
        envvar="LIGHTER_ACCOUNT_INDEX",
        help="Optional public account index to cross-check the operator record.",
    ),
    lighter_deposit_timeout: int = Option(
        DEFAULT_LIGHTER_DEPOSIT_TIMEOUT,
        envvar="LIGHTER_DEPOSIT_TIMEOUT",
    ),
    lighter_withdraw_timeout: int = Option(
        DEFAULT_LIGHTER_WITHDRAW_TIMEOUT,
        envvar="LIGHTER_WITHDRAW_TIMEOUT",
    ),
) -> None:
    """Move USDC between a Lagoon Safe and Lighter for diagnostics and recovery."""
    if simulate:
        raise LighterMoveFundsError("lighter-move-funds does not support SIMULATE=true")
    if asset_management_mode != AssetManagementMode.lagoon:
        raise LighterMoveFundsError("ASSET_MANAGEMENT_MODE must be lagoon")
    if not private_key or not vault_address or not vault_adapter_address:
        raise LighterMoveFundsError(
            "PRIVATE_KEY, VAULT_ADDRESS and VAULT_ADAPTER_ADDRESS are required",
        )
    if lighter_operator_record_file is None:
        raise LighterMoveFundsError("LIGHTER_OPERATOR_RECORD_FILE is required")
    if min(lighter_deposit_timeout, lighter_withdraw_timeout) <= 0:
        raise LighterMoveFundsError("Lighter timeout values must be positive")

    executor_id = prepare_executor_id(id, strategy_file)
    setup_logging(log_level=log_level)
    resolved_state_file, initial_store = resolve_state_store(executor_id, state_file)
    if initial_store.is_pristine():
        raise LighterMoveFundsError(f"State file does not exist: {resolved_state_file}")
    operator = load_lighter_operator_record(lighter_operator_record_file)
    validate_public_deployment_record(
        executor_id=executor_id,
        state_file=resolved_state_file,
        operator=operator,
    )
    if lighter_account_index is not None and lighter_account_index != operator.account_index:
        raise LighterMoveFundsError("LIGHTER_ACCOUNT_INDEX does not match the Lighter operator record")

    module = read_strategy_module(strategy_file)
    web3config = create_single_chain_web3_config(mod=module, **(rpc_kwargs or {}))
    session = create_lighter_session()
    try:
        web3 = web3config.get_default()
        if web3.eth.chain_id != LIGHTER_ETHEREUM_DEPLOYMENT_CHAIN_ID:
            raise LighterMoveFundsError("lighter-move-funds is Ethereum mainnet only")
        hot_wallet = create_hot_wallet(web3, private_key)
        vault = LagoonVault(
            web3,
            VaultSpec(web3.eth.chain_id, vault_address),
            trading_strategy_module_address=vault_adapter_address,
            default_block_identifier="latest",
            require_denomination_token=True,
        )
        if vault.safe_address.lower() != operator.safe_address.lower():
            raise LighterMoveFundsError("Lighter operator record does not match the deployed Safe")
        if operator.vault_address.lower() != vault_address.lower() or operator.module_address.lower() != vault_adapter_address.lower():
            raise LighterMoveFundsError("Lighter operator record does not match the configured vault")
        usdc = fetch_erc20_details(web3, LIGHTER_USDC_ETHEREUM, chain_id=web3.eth.chain_id)
        if vault.denomination_token.address.lower() != usdc.address.lower():
            raise LighterMoveFundsError("Lagoon vault denomination token must be native Ethereum USDC")
        reserve_asset = AssetIdentifier(
            chain_id=web3.eth.chain_id,
            address=usdc.address,
            token_symbol=usdc.symbol,
            decimals=usdc.decimals,
        )

        state = initial_store.load()
        position = _get_lighter_position(state, operator.account_index)
        pending_trade = _get_pending_transfer(position)
        safe_balance = usdc.fetch_balance_of(vault.safe_address)
        equity = fetch_lighter_total_equity(session, operator.account_index)
        account = fetch_lighter_account_by_index(session, operator.account_index)
        _assert_non_negative(safe_balance, "Safe USDC balance")
        _assert_non_negative(equity.get_total(), "Lighter equity")
        _log_balances(
            state=state,
            reserve_asset=reserve_asset,
            safe_balance=safe_balance,
            equity=equity,
            account=account,
            position=position,
            heading="Current custody balances",
        )

        if pending_trade is not None:
            resume_store, resume_state = backup_state(
                resolved_state_file,
                backup_suffix="lighter-move-funds",
                unit_testing=unit_testing,
            )
            position = _get_lighter_position(resume_state, operator.account_index)
            pending_trade = _get_pending_transfer(position)
            assert pending_trade is not None
            _resume_withdrawal(
                state=resume_state,
                store=resume_store,
                position=position,
                reserve_asset=reserve_asset,
                operator=operator,
                vault=vault,
                usdc=usdc,
                hot_wallet=hot_wallet,
                session=session,
                trade=pending_trade,
                withdraw_timeout=lighter_withdraw_timeout,
            )
        else:
            state.check_if_clean()
            _assert_reserve_matches_safe(state, reserve_asset, safe_balance)
            _assert_clean_custody_baseline(
                position=position,
                equity=equity,
            )
            direction = typer.prompt("Move direction [d/w]").strip().lower()
            if direction not in {"d", "w"}:
                raise LighterMoveFundsError("Move direction must be d or w")
            amount = _parse_usdc_amount(typer.prompt("Amount in USDC"))
            _validate_transfer_funds(direction, amount, safe_balance, equity)
            direction_label = "Safe -> Lighter" if direction == "d" else "Lighter -> Safe"
            if not typer.confirm(
                f"Confirm {direction_label} transfer of {amount} USDC for Lighter account {operator.account_index}?",
                default=False,
            ):
                raise typer.Abort()

            store, state = backup_state(resolved_state_file, backup_suffix="lighter-move-funds", unit_testing=unit_testing)
            position = _get_lighter_position(state, operator.account_index)
            safe_balance = usdc.fetch_balance_of(vault.safe_address)
            equity = fetch_lighter_total_equity(session, operator.account_index)
            state.check_if_clean()
            _assert_reserve_matches_safe(state, reserve_asset, safe_balance)
            _assert_clean_custody_baseline(
                position=position,
                equity=equity,
            )
            _validate_transfer_funds(direction, amount, safe_balance, equity)

            metadata = _create_transfer_metadata(
                direction="deposit" if direction == "d" else "withdraw",
                operator=operator,
                amount=amount,
                safe_before=safe_balance,
                equity_before=equity,
            )
            if direction == "d":
                logger.info("Depositing %s USDC from the Lagoon Safe to Lighter", amount)
                deposit_tx_hash = deposit_usdc_from_lagoon_safe_into_lighter(
                    web3,
                    hot_wallet,
                    vault=vault,
                    usdc=usdc,
                    deposit_usdc=amount,
                )
                wait_for_lighter_collateral(
                    session,
                    operator.account_index,
                    equity.collateral + amount,
                    timeout=lighter_deposit_timeout,
                )
                safe_after = usdc.fetch_balance_of(vault.safe_address)
                equity_after = fetch_lighter_total_equity(session, operator.account_index)
                trade = create_exchange_account_transfer(
                    state=state,
                    position=position,
                    strategy_cycle_at=native_datetime_utc_now(),
                    reserve_currency=reserve_asset,
                    amount=amount,
                    deposit=True,
                    notes="Manual lighter-move-funds deposit",
                    metadata=metadata,
                )
                _append_transfer_metadata(trade, deposit_tx_hash=deposit_tx_hash)
                mark_exchange_account_transfer_broadcasted(trade, native_datetime_utc_now())
                _complete_transfer(
                    state=state,
                    store=store,
                    position=position,
                    reserve_asset=reserve_asset,
                    trade=trade,
                    amount=amount,
                    safe_before=safe_balance,
                    safe_after=safe_after,
                    equity_after=equity_after,
                )
            else:
                try:
                    withdrawal_delay = fetch_lighter_withdrawal_delay(session)
                except Exception as error:  # noqa: BLE001 - informational API read must not leak SDK data
                    logger.warning("Could not read Lighter withdrawalDelay (%s)", type(error).__name__)
                else:
                    logger.info("Lighter secure withdrawal delay before request: %d seconds", withdrawal_delay)
                logger.info("Requesting secure withdrawal of %s USDC from Lighter", amount)
                trade = create_exchange_account_transfer(
                    state=state,
                    position=position,
                    strategy_cycle_at=native_datetime_utc_now(),
                    reserve_currency=reserve_asset,
                    amount=amount,
                    deposit=False,
                    notes="Manual lighter-move-funds withdrawal",
                    metadata=metadata,
                )
                store.sync(state)
                try:
                    request_tx_hash = asyncio.run(request_lighter_withdrawal(operator, amount))
                except LighterWithdrawalRejected:
                    failed_at = native_datetime_utc_now()
                    trade.started_at = failed_at
                    trade.mark_failed(failed_at)
                    _append_transfer_metadata(trade, outcome="withdrawal_rejected")
                    store.sync(state)
                    raise
                _append_transfer_metadata(trade, request_tx_hash=request_tx_hash)
                mark_exchange_account_transfer_broadcasted(trade, native_datetime_utc_now())
                store.sync(state)
                _resume_withdrawal(
                    state=state,
                    store=store,
                    position=position,
                    reserve_asset=reserve_asset,
                    operator=operator,
                    vault=vault,
                    usdc=usdc,
                    hot_wallet=hot_wallet,
                    session=session,
                    trade=trade,
                    withdraw_timeout=lighter_withdraw_timeout,
                )

        final_state = initial_store.load()
        final_position = _get_lighter_position(final_state, operator.account_index)
        final_safe = usdc.fetch_balance_of(vault.safe_address)
        final_equity = fetch_lighter_total_equity(session, operator.account_index)
        final_account = fetch_lighter_account_by_index(session, operator.account_index)
        _log_balances(
            state=final_state,
            reserve_asset=reserve_asset,
            safe_balance=final_safe,
            equity=final_equity,
            account=final_account,
            position=final_position,
            heading="Final custody balances",
        )
        _assert_reserve_matches_safe(final_state, reserve_asset, final_safe)
        _assert_clean_custody_baseline(
            position=final_position,
            equity=final_equity,
        )
        logger.info("All ok")
    except ExchangeAccountTransferError as error:
        raise LighterMoveFundsError(str(error)) from None
    finally:
        session.close()
        operator.clear_private_key()
        web3config.close()
