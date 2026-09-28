"""Correct accounting errors in the internal state.

"""
import datetime
import logging
import sys
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Optional

import typer
from tabulate import tabulate
from typer import Option

from eth_defi.compat import native_datetime_utc_now
from eth_defi.hotwallet import HotWallet
from eth_defi.hyperliquid.session import (
    HYPERLIQUID_API_URL,
    HYPERLIQUID_TESTNET_API_URL,
    create_hyperliquid_session,
)
from eth_defi.provider.broken_provider import get_almost_latest_block_number
from web3 import Web3

from tradeexecutor.ethereum.vault import hypercore_transit_recovery, hypercore_vault
from tradeexecutor.ethereum.vault.hypercore_transit_recovery import (
    BALANCE_TOLERANCE,
    HYPERCORE_TRANSIT_RECOVERY_DUST_USDC,
    HypercoreTransitBalanceSnapshot,
)
from tradeexecutor.exchange_account.derive import DeriveNetwork
from tradeexecutor.exchange_account.lighter import LIGHTER_PROTOCOL
from tradeexecutor.ethereum.lighter.transfer_verification import reconcile_verified_lighter_transfers
from tradeexecutor.exchange_account.sync_model import ExchangeAccountSyncModel
from tradeexecutor.exchange_account.utils import create_exchange_account_value_func
from tradeexecutor.strategy.account_correction import (
    check_accounts,
    correct_accounts as _correct_accounts,
    create_missing_exchange_account_positions,
    preflight_state_for_account_correction,
    UnknownTokenPositionFix,
)
from .app import app
from ..bootstrap import prepare_executor_id, create_web3_config, create_sync_model, create_client, backup_state, create_execution_and_sync_model, resolve_deployment_file, configure_default_chain, create_state_store
from ..double_position import check_double_position
from ..log import setup_logging
from ...ethereum.enzyme.tx import EnzymeTransactionBuilder
from ...ethereum.enzyme.vault import EnzymeVaultSyncModel
from ...ethereum.hot_wallet_sync_model import HotWalletSyncModel
from ...ethereum.lagoon.vault import LagoonVaultSyncModel
from ...ethereum.tx import HotWalletTransactionBuilder
from ...ethereum.vault.hypercore_small_position_cleanup import (
    discover_hypercore_small_positions,
    get_hypercore_minimum_allocation,
    run_hypercore_small_position_cleanup,
)
from ...state.repair import close_hypercore_dust_positions
from ...state.position import TradingPosition
from ...state.state import State, UncleanState
from ...state.trade import (
    HYPERCORE_ACCOUNTING_RECONCILIATION_REQUIRED_KEY,
    HYPERCORE_DEPOSIT_CAPITAL_AT_RISK_KEY,
    HYPERCORE_STRANDED_USDC_KEY,
    TradeExecution,
    TradeStatus,
    TradeType,
    has_unresolved_hypercore_accounting,
)
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


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class InterruptedHypercoreDeposit:
    """Observed custody for one interrupted opening deposit, before any transfer.

    ``correct_accounts()`` uses this to reject ambiguous custody rather than
    assuming that a successful phase-1 EVM receipt proves vault settlement.

    :param trade: Started buy whose reserve remains allocated at risk.
    :param position: Empty target vault position to close only after recovery.
    :param snapshot: Current Safe and HyperCore spot/perp balances.
    :param vault_equity: Current target-vault equity, required to be zero.
    :param already_returned: Whether cash reached the Safe before this run.
    """

    trade: TradeExecution
    position: TradingPosition
    snapshot: HypercoreTransitBalanceSnapshot
    vault_equity: Decimal
    already_returned: bool


def _inspect_interrupted_hypercore_deposit(
    state: State,
    sync_model: LagoonVaultSyncModel,
    web3: Web3,
) -> InterruptedHypercoreDeposit | None:
    """Classify the single at-risk HyperCore opening buy against live custody.

    Called after ``repair`` has cleared unrelated no-transaction trades and
    before ``correct-accounts`` can run small-position cleanup or broadcast a
    transit transfer. The phase-3 revert is not in the state file; an operator
    must verify that receipt separately before authorising the real run.

    :param state: Persisted executor state, after no-transaction repair.
    :param sync_model: Lagoon Safe whose balances are being corrected.
    :param web3: HyperEVM connection used to verify recorded phase-1 receipts.
    :return: Verified incident custody, or ``None`` without an at-risk trade.
    """
    trades = [trade for trade in state.portfolio.get_all_trades() if has_unresolved_hypercore_accounting(trade)]
    if not trades:
        return None
    if len(trades) != 1:
        raise RuntimeError(f"Expected one at-risk HyperCore trade, found {[trade.trade_id for trade in trades]}")
    trade = trades[0]
    position = state.portfolio.get_position_by_id(trade.position_id)
    if not (
        isinstance(sync_model, LagoonVaultSyncModel)
        and trade.pair.is_hyperliquid_vault()
        and trade.is_buy()
        and trade.get_status() == TradeStatus.started
        and trade.blockchain_transactions
        and position.position_id in state.portfolio.open_positions
        and position.get_quantity() == 0
    ):
        raise RuntimeError(f"HyperCore trade #{trade.trade_id} is not a zero-fill started Lagoon vault deposit")

    for tx in trade.blockchain_transactions:
        receipt = web3.eth.get_transaction_receipt(tx.tx_hash)
        if receipt["status"] != 1:
            raise RuntimeError(f"Stored phase-1 transaction {tx.tx_hash} did not succeed; stop for operator review")

    safe_address = sync_model.get_token_storage_address()
    marker = trade.other_data.get(HYPERCORE_DEPOSIT_CAPITAL_AT_RISK_KEY)
    if not isinstance(marker, dict) or trade.reserve_currency_allocated is None:
        raise RuntimeError(f"HyperCore trade #{trade.trade_id} lacks the saved deposit amount needed for reconciliation")
    if marker.get("safe_address", "").lower() != safe_address.lower():
        raise RuntimeError(f"HyperCore trade #{trade.trade_id} marker belongs to a different Safe")
    amount = Decimal(marker["amount_human"])
    if abs(amount - trade.reserve_currency_allocated) > BALANCE_TOLERANCE:
        raise RuntimeError(f"HyperCore trade #{trade.trade_id} reserve allocation disagrees with its at-risk marker")

    api_url = HYPERLIQUID_TESTNET_API_URL if web3.eth.chain_id == 998 else HYPERLIQUID_API_URL
    session = create_hyperliquid_session(api_url=api_url)
    snapshot = hypercore_transit_recovery.fetch_hypercore_transit_balances(
        session=session,
        safe_address=safe_address,
        reserve_token=sync_model.vault.underlying_token,
    )
    vault_equity = hypercore_vault.create_hypercore_vault_value_func(
        session=session,
        safe_address=safe_address,
        bypass_cache=True,
    )(trade.pair)
    if snapshot.perp_position_count or vault_equity != 0:
        raise RuntimeError(
            f"HyperCore trade #{trade.trade_id} has active perp positions or target vault equity {vault_equity}; "
            "automatic transit recovery would be unsafe"
        )
    if snapshot.spot_total_usdc > HYPERCORE_TRANSIT_RECOVERY_DUST_USDC + BALANCE_TOLERANCE:
        raise RuntimeError(f"Safe {safe_address} also has material HyperCore spot USDC; stop for operator review")

    saved_safe = state.portfolio.get_default_reserve_position().quantity
    safe_surplus = snapshot.evm_usdc_balance - saved_safe
    stranded = (
        abs(safe_surplus) <= BALANCE_TOLERANCE
        and abs(snapshot.perp_withdrawable - amount) <= HYPERCORE_TRANSIT_RECOVERY_DUST_USDC
    )
    already_returned = (
        snapshot.perp_withdrawable <= HYPERCORE_TRANSIT_RECOVERY_DUST_USDC + BALANCE_TOLERANCE
        and amount - Decimal("0.50") <= safe_surplus <= amount + BALANCE_TOLERANCE
    )
    if not (stranded or already_returned):
        raise RuntimeError(
            f"HyperCore trade #{trade.trade_id} custody is ambiguous: saved Safe {saved_safe}, "
            f"live Safe {snapshot.evm_usdc_balance}, perp {snapshot.perp_withdrawable}, "
            f"spot {snapshot.spot_free_usdc}, vault {vault_equity}; stop for operator review"
        )
    if stranded and not hypercore_transit_recovery.plan_hypercore_transit_recovery_actions(snapshot):
        raise RuntimeError(f"HyperCore trade #{trade.trade_id} has no transferable recovery path")
    return InterruptedHypercoreDeposit(trade, position, snapshot, vault_equity, already_returned)


def _sync_hypercore_vault_positions(
    *,
    asset_management_mode: AssetManagementMode,
    universe,
    sync_model,
    web3,
    state,
) -> bool:
    """Auto-create and mark Hypercore vault positions from the Hyperliquid API.

    :return:
        True if any phantom positions were closed (zero-proceeds repair).
        The caller should ensure the reserve balance correction runs
        afterwards, because USDC from untracked withdrawals may be
        sitting in the Safe without being reflected in state reserves.
    """

    if not asset_management_mode.is_vault() or universe is None:
        return False

    from tradeexecutor.strategy.trading_strategy_universe import translate_trading_pair

    has_vault_pairs = any(
        translate_trading_pair(p).is_hyperliquid_vault()
        for p in universe.data_universe.pairs.iterate_pairs()
    )
    if not has_vault_pairs:
        return False

    safe_address = sync_model.get_token_storage_address()
    chain_id = web3.eth.chain_id

    from tradeexecutor.strategy.account_correction import create_missing_vault_positions
    from tradeexecutor.state.valuation import ValuationUpdate

    is_testnet = chain_id == 998
    api_url = HYPERLIQUID_TESTNET_API_URL if is_testnet else HYPERLIQUID_API_URL
    hl_session = create_hyperliquid_session(api_url=api_url)

    vault_value_func = hypercore_vault.create_hypercore_vault_value_func(
        session=hl_session,
        safe_address=safe_address,
        is_testnet=is_testnet,
    )

    logger.info("Checking for missing Hypercore vault positions (Safe: %s)...", safe_address)
    vault_created_trades = create_missing_vault_positions(
        strategy_universe=universe,
        state=state,
        strategy_cycle_at=native_datetime_utc_now(),
        vault_value_func=vault_value_func,
    )
    if vault_created_trades:
        logger.info("Auto-created %d Hypercore vault position(s)", len(vault_created_trades))
        for trade in vault_created_trades:
            logger.info("  Created vault position for %s", trade.pair)

    vault_positions = [
        p for p in state.portfolio.get_open_positions()
        if p.pair.is_hyperliquid_vault()
    ]
    closed_phantom = False
    for position in vault_positions:
        try:
            current_equity = vault_value_func(position.pair)
        except Exception as e:
            logger.error("Failed to get vault equity for position %d: %s", position.position_id, e)
            continue

        quantity = position.get_quantity()
        old_value = position.get_value()
        diff = float(current_equity) - old_value
        vault_name = position.pair.get_vault_name() or "unknown"
        vault_addr = position.pair.pool_address or "?"

        # Detect phantom positions: the Hyperliquid API reports zero equity
        # but the state still tracks positive deposited USDC.
        #
        # This happens when a Hypercore vault withdrawal completed on
        # Hyperliquid but the executor failed to confirm it (e.g. timeout
        # or restart during the 3-phase settlement in _settle_withdrawal).
        # The repair command zeroes out the unconfirmed trade, leaving a
        # phantom position with quantity > 0 but no on-chain equity.
        #
        # It can also happen when a vault's trading losses wipe out the
        # follower's deposit entirely — the userVaultEquities API omits
        # zero-equity vaults, so the value function returns Decimal(0).
        #
        # We cannot distinguish these two cases here. The safe approach is
        # to close the position as a zero-proceeds loss. If USDC actually
        # returned to the Safe (untracked withdrawal), the generic reserve
        # balance correction later in correct-accounts will detect the
        # surplus and credit reserves.
        #
        # This check must run before the diff==0 early return below,
        # because a previous valuator tick may have already set
        # last_token_price=0.0, making old_value=0 and diff=0.
        if float(current_equity) == 0 and float(quantity) > 0:
            logger.warning(
                "Vault position %d (%s %s): Hyperliquid API reports zero equity "
                "but state has %s USDC deposited. "
                "Closing as zero-proceeds loss (phantom position). "
                "If USDC was returned to the Safe, the reserve correction will credit it.",
                position.position_id,
                vault_name,
                vault_addr,
                quantity,
            )

            valued_at = native_datetime_utc_now()
            reserve_asset = state.portfolio.get_default_reserve_position().asset
            _position_ref, correction_trade, _created = state.portfolio.create_trade(
                strategy_cycle_at=valued_at,
                pair=position.pair,
                quantity=-quantity,
                reserve=None,
                assumed_price=1.0,
                trade_type=TradeType.repair,
                reserve_currency=reserve_asset,
                reserve_currency_price=1.0,
                position=position,
                notes=(
                    f"correct-accounts: closing phantom Hypercore vault position. "
                    f"Hyperliquid API reports zero equity but state has {quantity} USDC deposited. "
                    f"This is either an untracked withdrawal or a total vault trading loss. "
                    f"Any returned USDC will be reconciled by the reserve balance correction."
                ),
            )
            # Use trade.mark_success() directly (not state.mark_trade_success())
            # to bypass return_capital_to_reserves(). We don't know whether USDC
            # actually returned to the Safe; the generic reserve correction
            # handles that separately based on actual on-chain balances.
            correction_trade.mark_success(
                executed_at=valued_at,
                executed_price=0.0,
                executed_quantity=-quantity,
                executed_reserve=Decimal(0),
                lp_fees=0,
                native_token_price=0.0,
                force=True,
            )
            # Accounting correction: the onchain balance is gone, so the book-close is the
            # intended outcome and the offsetting correction is recorded separately.
            state.portfolio.close_position(position, valued_at, allow_value_destruction=True)

            logger.info(
                "Closed phantom vault position %d with repair trade %d",
                position.position_id,
                correction_trade.trade_id,
            )
            closed_phantom = True
            continue

        if diff == 0:
            logger.debug(
                "Vault position %d (%s): no change (equity=%.2f)",
                position.position_id,
                position.pair.get_vault_name() or position.pair.pool_address,
                current_equity,
            )
            continue

        if quantity <= 0:
            logger.warning(
                "Vault position %d (%s) has equity %.2f but no positive state quantity (%s)",
                position.position_id,
                position.pair.get_vault_name() or position.pair.pool_address,
                current_equity,
                quantity,
            )
            continue

        logger.info(
            "Vault position %d (%s %s): marking equity %.2f -> %.2f (diff=%.2f)",
            position.position_id,
            vault_name,
            vault_addr,
            old_value,
            current_equity,
            diff,
        )

        valued_at = native_datetime_utc_now()
        old_price = position.last_token_price
        new_price = float(current_equity / quantity)
        new_value = position.revalue_base_asset(valued_at, new_price)
        position.valuation_updates.append(
            ValuationUpdate(
                created_at=valued_at,
                position_id=position.position_id,
                valued_at=valued_at,
                old_value=old_value,
                new_value=new_value,
                old_price=old_price,
                new_price=new_price,
                quantity=quantity,
            )
        )

    return closed_phantom


def _has_hypercore_vault_positions(state) -> bool:
    """Check whether the strategy tracks any HyperCore vault position.

    This deliberately includes open, frozen, and closed positions. Transit USDC
    belongs to the Safe rather than a position, so only looking at closed
    positions could skip the default #1486 recovery for a strategy which has
    started following HyperCore vaults but has not closed one yet. The planner
    separately reads the live perp account and refuses recovery when it finds
    an actual active perp position.
    """
    return any(
        position.pair.is_hyperliquid_vault()
        for position_collection in (
            getattr(state.portfolio, "open_positions", {}),
            getattr(state.portfolio, "frozen_positions", {}),
            getattr(state.portfolio, "closed_positions", {}),
        )
        for position in position_collection.values()
    )


def _recover_hypercore_transit_balances(
    *,
    asset_management_mode: AssetManagementMode,
    sync_model,
    web3,
    hot_wallet: HotWallet | None,
    state,
    skip_hypercore_transit_recovery: bool,
    dry_run: bool,
) -> list[str]:
    """Plan or recover Safe-level HyperCore spot/perp USDC before correction.

    HyperAI trade #1486 (Citadel, 2026-07-30; PR #1593) is the reason this runs before
    generic account correction. Its combined CoreWriter request received a
    successful HyperEVM receipt, moved 48.884068 USDC from spot to perp, and
    silently did not move that USDC from perp to the vault. Generic accounting
    cannot see HyperCore's internal USDC classes, so it must never decide that
    the stranded balance is Safe cash.

    The live snapshot is authoritative: recovery is Safe-level rather than
    trade-level because an older state file can predate the failure marker.
    Recovery only returns unencumbered spot/perp USDC to HyperEVM, and the
    lower-level planner refuses active perp positions. ``dry_run`` deliberately
    follows the same snapshot and planning path without requiring a signer or
    broadcasting. The normal planner is enabled by default for every eligible
    HyperCore vault strategy and returns the transferable excess which it just
    moved from perp to spot; it leaves only HyperCore's mandatory bridge-fee
    margin when the pre-existing spot balance cannot cover it. This prevents
    the #1486 amount being reduced by unrelated pre-existing spot dust.
    """
    if not asset_management_mode.is_vault():
        return []

    if not _has_hypercore_vault_positions(state):
        return []

    if skip_hypercore_transit_recovery:
        logger.info(
            "Skipping HyperCore transit recovery although HyperCore vault positions exist"
        )
        return []

    if not isinstance(sync_model, LagoonVaultSyncModel):
        raise RuntimeError(
            "HyperCore vault positions exist, but automatic HyperCore transit recovery "
            f"is only implemented for Lagoon vaults. Got sync model {type(sync_model)}."
        )

    is_testnet = web3.eth.chain_id == 998
    api_url = HYPERLIQUID_TESTNET_API_URL if is_testnet else HYPERLIQUID_API_URL
    session = create_hyperliquid_session(api_url=api_url)
    safe_address = sync_model.get_token_storage_address()
    reserve_token = sync_model.vault.underlying_token

    logger.info(
        "HyperCore vault positions detected; checking Safe-level HyperCore transit balances for %s",
        safe_address,
    )
    snapshot = hypercore_transit_recovery.fetch_hypercore_transit_balances(
        session=session,
        safe_address=safe_address,
        reserve_token=reserve_token,
    )
    actions = hypercore_transit_recovery.plan_hypercore_transit_recovery_actions(snapshot)
    if not actions:
        logger.info("No HyperCore spot/perp transit balances need recovery")
        return []

    for action in actions:
        logger.info(
            "HyperCore transit recovery planned: %s %.6f USDC (%s)",
            action.action_kind,
            action.amount,
            action.reason,
        )

    if dry_run:
        # This is intentionally after the live snapshot and action planner.
        # Before this change, correct-accounts --dry-run skipped the exact
        # perp->spot->EVM plan that would have made the #1486 recovery visible.
        logger.warning(
            "Dry run: HyperCore transit recovery would broadcast %s for Safe %s; "
            "no transaction is signed, broadcast, or persisted.",
            ", ".join(action.action_kind for action in actions),
            safe_address,
        )
        return [action.action_kind for action in actions]

    if hot_wallet is None:
        raise RuntimeError(
            "HyperCore vault positions exist and HyperCore transit recovery is needed, "
            "but no hot wallet/private key is configured for broadcasting Safe transactions."
        )

    hot_wallet.sync_nonce(web3)
    executed_action_kinds = hypercore_transit_recovery.execute_hypercore_transit_recovery_actions(
        web3=web3,
        hot_wallet=hot_wallet,
        lagoon_vault=sync_model.vault,
        session=session,
        reserve_token=reserve_token,
        actions=actions,
    )
    logger.info(
        "HyperCore transit recovery completed: %s",
        ", ".join(executed_action_kinds),
    )
    return executed_action_kinds


@app.command()
@shared_options.with_json_rpc_options()
def correct_accounts(
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
    vault_payment_forwarder: Optional[str] = shared_options.vault_payment_forwarder,
    vault_deployment_block_number: Optional[int] = shared_options.vault_deployment_block_number,


    rpc_kwargs: dict | None = None,

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
    skip_hypercore_transit_recovery: bool = Option(False, "--skip-hypercore-transit-recovery", envvar="SKIP_HYPERCORE_TRANSIT_RECOVERY", help="Skip Safe-level HyperCore spot/perp USDC recovery before account correction."),
    cleanup_hypercore_small_positions: bool = Option(True, "--cleanup-hypercore-small-positions/--no-cleanup-hypercore-small-positions", envvar="CLEANUP_HYPERCORE_SMALL_POSITIONS", help="Redeem open HyperCore vault positions below the strategy minimum allocation before correcting accounts."),
    dry_run: bool = Option(False, "--dry-run", envvar="DRY_RUN", help="Read live balances and print any Safe-level HyperCore perp->spot->EVM recovery plan without signing, broadcasting, persisting state, or applying accounting corrections."),
    consume_partial_hypercore_slot: bool = Option(False, "--consume-partial-hypercore-slot", help="After verified recovery, mark an interrupted HyperCore decision with already-executed trades as consumed instead of replaying it."),

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
    Use ``--dry-run`` for a read-only rehearsal. It includes the live
    Safe-level HyperCore transit-recovery planner: this reads spot/perp/EVM
    balances and prints any ``perp -> spot -> EVM`` actions, but never signs or
    broadcasts them. This makes a partial CoreWriter settlement visible before
    generic correction changes state. Dry-run never writes or backs up the
    state file.

    For the interrupted 28 September Hyper-AI deposit, first verify the known
    failed vault-deposit receipt and current custody. ``repair`` fixes its
    never-broadcast sibling trades but leaves the at-risk deposit untouched.
    After reviewing this command's dry-run, pass
    ``--consume-partial-hypercore-slot`` on the real run to save the failed
    deposit, actual Safe reserve correction and consumed two-day slot together.
    This option does not retry the rejected vault deposit.

    An old state file is automatically backed up.
    """

    global logger

    id = prepare_executor_id(id, strategy_file)

    logger = setup_logging(log_level)

    web3config = create_web3_config(
        gas_price_method=None,
        **rpc_kwargs,
        unit_testing=unit_testing,
    )

    assert web3config, "No RPC endpoints given. A working JSON-RPC connection is needed for check-wallet"

    # Read strategy module early so we can use its default chain id
    mod: StrategyModuleInformation = read_strategy_module(strategy_file)

    # Set default chain using the strategy module's declared chain,
    # so the correct Web3 instance is used for vault contract calls
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

    # Log all connected chains
    for chain_id, conn in web3config.connections.items():
        logger.info(f"  Chain {chain_id.name} (id {conn.eth.chain_id:,})")
        logger.info(f"    Latest block is {conn.eth.block_number:,}")

    # Check balances
    logger.info("Balance details")
    logger.info("  Hot wallet is %s", hot_wallet.address if hot_wallet else "not configured")

    vault_address =  sync_model.get_key_address()
    if vault_address:
        logger.info("  Vault is %s", vault_address)
        if vault_deployment_block_number:
            start_block = vault_deployment_block_number
            logger.info("  Vault deployment block number is %d", start_block)

    if not state_file:
        state_file = f"state/{id}.json"

    if dry_run:
        store = create_state_store(Path(state_file), simulate=True)
        assert not store.is_pristine(), f"State file does not exist: {state_file}"
        state = store.load()
        logger.warning("Dry run enabled: no transactions will be broadcast and no state will be written")
    else:
        store, state = backup_state(state_file, unit_testing=unit_testing)

    reconciled_transfers = reconcile_verified_lighter_transfers(
        state,
        web3,
        mutate=not dry_run,
    )
    if reconciled_transfers:
        if not dry_run:
            store.sync(state)
        logger.info(
            "%s %d verified Lighter transfer(s)",
            "Found" if dry_run else "Reconciled",
            len(reconciled_transfers),
        )

    # This must precede universe construction, vault synchronisation, and the
    # HyperCore transit hook below.  The hook can broadcast real Safe actions;
    # discovering an unfinished trade afterwards reproduces the #1486 incident
    # where funds were recovered but the command could not complete its state
    # reconciliation.
    at_risk_trades = [
        trade for trade in state.portfolio.get_all_trades()
        if has_unresolved_hypercore_accounting(trade)
    ]
    if at_risk_trades and not dry_run and not consume_partial_hypercore_slot:
        raise RuntimeError(
            "An interrupted HyperCore deposit needs --consume-partial-hypercore-slot "
            "after receipt and custody review; use --dry-run first"
        )
    if at_risk_trades and (skip_save or process_redemption):
        raise RuntimeError("Interrupted HyperCore deposit reconciliation requires an atomic state save and no redemption processing")
    preflight_state_for_account_correction(state, allow_at_risk_hypercore_deposit=bool(at_risk_trades))
    incident = _inspect_interrupted_hypercore_deposit(state, sync_model, web3) if at_risk_trades else None
    if incident:
        slot = state.pending_data_availability_slot
        if slot is None or incident.trade.opened_at != slot:
            raise RuntimeError("At-risk HyperCore deposit is not tied to the pending decision slot")
        sibling_trades = [
            trade for trade in state.portfolio.get_all_trades()
            if trade.opened_at == slot and trade.trade_id != incident.trade.trade_id
        ]
        if not any(trade.trade_type == TradeType.rebalance and trade.is_success() for trade in sibling_trades):
            raise RuntimeError("Pending HyperCore slot has no successful sibling trade to preserve")
        if any(
            trade.get_status() not in (TradeStatus.success, TradeStatus.failed, TradeStatus.repaired, TradeStatus.expired)
            for trade in sibling_trades
        ):
            raise RuntimeError("Pending HyperCore slot still has unfinished sibling trades; run repair first")
        logger.warning(
            "Interrupted HyperCore deposit #%d, position #%d: Safe %s, perp %s, spot %s, "
            "vault equity %s, slot %s; %s. Not safe to restart yet.",
            incident.trade.trade_id,
            incident.position.position_id,
            incident.snapshot.evm_usdc_balance,
            incident.snapshot.perp_withdrawable,
            incident.snapshot.spot_free_usdc,
            incident.vault_equity,
            slot,
            "funds already returned to Safe" if incident.already_returned else "transit recovery required",
        )

    slippage_tolerance = 0.013
    if mod:
        if mod.parameters:
            slippage_tolerance = mod.parameters.get("slippage_tolerance", 0.013)

    logger.info("Using slippage tolerance: %f", slippage_tolerance)

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
        asset_management_mode=asset_management_mode,
    )
    # A code-defined universe, such as the static Lighter external-account
    # monitor, does not download market data and therefore deliberately has no
    # Trading Strategy API client.  Keep supporting normal data-backed
    # strategies when a key is configured, but do not make an unused client a
    # prerequisite for account reconciliation.

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
        confirmation_block_count=1,
        max_slippage=slippage_tolerance,
        min_gas_balance=Decimal(0),
        vault_payment_forwarder_address=vault_payment_forwarder,
        deployment_file=resolve_deployment_file(id, state_file),
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
    routing_state = pricing_model = valuation_method = None

    def ensure_routing_setup() -> None:
        """Initialise routing lazily when downstream code truly needs it."""

        nonlocal routing_state, pricing_model, valuation_method

        if pricing_model is not None or routing_state is not None or valuation_method is not None:
            return

        if not mod.is_version_greater_or_equal_than(0, 5, 0):
            logger.info("Routing setup skipped - legacy strategy compatibility mode")
            return

        if mod.trade_routing == TradeRouting.ignore:
            logger.info("Routing setup skipped - strategy uses TradeRouting.ignore")
            return

        routing_state, pricing_model, valuation_method = runner.setup_routing(universe)
        logger.info("Lazy routing model initialised")

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

    if not skip_interest:
        credit_positions = [p for p in state.portfolio.get_open_and_frozen_positions() if p.is_credit_supply()]
        if len(credit_positions) > 0:
            # Sync missing credit
            try:
                logger.info("Credit positions detected, syncing interest before applying accounting checks")
                for p in credit_positions:
                    logger.info(" - Position: %s", p)
                ensure_routing_setup()
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

    exchange_account_value_func = None

    # Sync all exchange account positions with actual exchange API values
    if exchange_account_positions:
        logger.info("Found %d exchange account position(s)", len(exchange_account_positions))
        exchange_account_value_func = create_exchange_account_value_func(
            exchange_account_positions,
            derive_owner_private_key,
            derive_session_private_key,
            derive_wallet_address,
            derive_network,
            ccxt_exchange_id,
            ccxt_options,
            ccxt_sandbox,
            logger,
            web3=web3config.get_default() if web3config.has_any_connection() else None,
            execution_model=execution_model,
        )

        if exchange_account_value_func:
            exchange_sync_model = ExchangeAccountSyncModel(exchange_account_value_func, web3=web3)
            ensure_routing_setup()
            exchange_events = exchange_sync_model.sync_positions(
                timestamp=native_datetime_utc_now(),
                state=state,
                strategy_universe=universe,
                pricing_model=pricing_model,
            )
            logger.info("Exchange account sync: %d balance update(s)", len(exchange_events))
            for evt in exchange_events:
                logger.info("  Position %d: %s (change: %s)", evt.position_id, evt.notes, evt.quantity)

    closed_phantom_positions = False
    if not incident:
        closed_phantom_positions = _sync_hypercore_vault_positions(
            asset_management_mode=asset_management_mode,
            universe=universe,
            sync_model=sync_model,
            web3=web3,
            state=state,
        )

    if incident:
        logger.info("Skipping unrelated HyperCore vault synchronisation and small-position cleanup during deposit recovery")
    if not incident and cleanup_hypercore_small_positions and asset_management_mode.is_vault():
        minimum_allocation = get_hypercore_minimum_allocation(mod.parameters)
        if minimum_allocation is None:
            logger.info(
                "HyperCore small-position cleanup skipped: strategy does not define a minimum allocation parameter"
            )
        elif not isinstance(sync_model, LagoonVaultSyncModel):
            logger.info(
                "HyperCore small-position cleanup skipped: it is only implemented for Lagoon vault strategies"
            )
        else:
            cleanup_candidates = discover_hypercore_small_positions(
                state,
                minimum_allocation,
            )
            if not cleanup_candidates:
                logger.info("HyperCore small-position cleanup found no eligible positions")
            else:
                is_testnet = web3.eth.chain_id == 998
                api_url = HYPERLIQUID_TESTNET_API_URL if is_testnet else HYPERLIQUID_API_URL
                session = create_hyperliquid_session(api_url=api_url)
                ensure_routing_setup()
                if routing_state is not None:
                    cleanup_report = run_hypercore_small_position_cleanup(
                        state=state,
                        timestamp=native_datetime_utc_now(),
                        candidates=cleanup_candidates,
                        execution_model=execution_model,
                        routing_model=runner.routing_model,
                        routing_state=routing_state,
                        session=session,
                        safe_address=sync_model.get_token_storage_address(),
                        store=store,
                        dry_run=dry_run,
                    )
                    if dry_run:
                        assert cleanup_report.simulated_state is not None
                        state = cleanup_report.simulated_state
                        logger.info(
                            "Dry run: HyperCore small-position cleanup found %d candidate(s)",
                            len(cleanup_report.candidates),
                        )
                    else:
                        logger.info(
                            "HyperCore small-position cleanup processed %d trade(s) across %d candidate(s); "
                            "%d position(s) were closed",
                            len(cleanup_report.executed_trades),
                            len(cleanup_report.candidates),
                            len(cleanup_report.closed_position_ids),
                        )
                else:
                    logger.warning(
                        "HyperCore small-position cleanup skipped: routing is unavailable for this strategy"
                    )

    closed_dust_trades = [] if incident else close_hypercore_dust_positions(state.portfolio)
    if closed_dust_trades:
        logger.info(
            "Auto-closed %d Hypercore dust position(s) before duplicate and accounting checks",
            len(closed_dust_trades),
        )

    # The initial preflight protects external work performed during command
    # setup. Recheck immediately before the transit hook as a defence against
    # any position-discovery or cleanup step above introducing an unfinished
    # trade. This preserves the #1486 invariant: a transit recovery is never
    # the first irreversible action after a coherence failure.
    preflight_state_for_account_correction(state, allow_at_risk_hypercore_deposit=incident is not None)

    if incident and incident.already_returned:
        logger.info("HyperCore transit funds are already back in the Safe; skipping transfer legs")
    else:
        if incident and skip_hypercore_transit_recovery:
            raise RuntimeError("Cannot skip transit recovery for an at-risk HyperCore deposit")
        _recover_hypercore_transit_balances(
            asset_management_mode=asset_management_mode,
            sync_model=sync_model,
            web3=web3,
            hot_wallet=hot_wallet,
            state=state,
            skip_hypercore_transit_recovery=skip_hypercore_transit_recovery,
            dry_run=dry_run,
        )

    if incident and not dry_run:
        returned = _inspect_interrupted_hypercore_deposit(state, sync_model, web3)
        if returned is None or not returned.already_returned:
            raise RuntimeError("HyperCore deposit recovery did not return the expected USDC to the Safe")
        recovered = returned.snapshot.evm_usdc_balance - incident.snapshot.evm_usdc_balance
        incident.trade.mark_failed(native_datetime_utc_now())
        incident.trade.add_note(
            f"correct-accounts: phase-3 vault deposit rejected; phase-1 receipts "
            f"{[tx.tx_hash for tx in incident.trade.blockchain_transactions]}; "
            f"Safe USDC {incident.snapshot.evm_usdc_balance} -> {returned.snapshot.evm_usdc_balance}, "
            f"recovered {recovered}, remaining perp {returned.snapshot.perp_withdrawable}, "
            f"spot {returned.snapshot.spot_free_usdc}, vault equity {returned.vault_equity}. "
            "Phase-3 revert verified separately by operator."
        )
        incident.position.add_notes_message("Opening HyperCore deposit failed before vault settlement; no vault equity acquired")

    if process_redemption and not dry_run:
        timestamp = native_datetime_utc_now()
        reserve_assets = list(universe.reserve_assets)

        if process_redemption_end_block_hint:
            # Passed by unit tests so we are not going to scan the whole chain until today (wall clock time)
            end_block = process_redemption_end_block_hint
        else:
            end_block = execution_model.get_safe_latest_block()

        logger.info(
            "Processing deposits/redemptions before correcting accounts, timestamp set to %s, reserves are %s, end block is %d",
            timestamp,
            reserve_assets,
            end_block,
        )

        # A completed Lagoon settlement changes the Safe balance.  Reconcile it
        # before deriving corrections so its investor flow is not mistaken for
        # an unexplained wallet-balance difference.
        sync_model.sync_treasury(
            strategy_cycle_ts=timestamp,
            state=state,
            end_block=end_block,
            post_valuation=True,
        )
    elif process_redemption:
        logger.info("Deposit/redemption distribution skipped for dry run")
    else:
        logger.info("Deposit/redemption distribution skipped")

    block_number = get_almost_latest_block_number(web3)
    logger.info(f"Correcting accounts at block {block_number:,}")

    block_timestamp = get_block_timestamp(web3, block_number)

    # Skip on-chain corrections when all positions are exchange account positions
    # (their balances are synced via exchange API, not on-chain balance checks).
    # A Lagoon Lighter vault is the narrow exception: its Safe reserve remains
    # on-chain even when its sole trading position is the synthetic account.
    # A closed phantom Hypercore vault can similarly leave USDC in the Safe.
    has_lagoon_lighter_position = (
        asset_management_mode == AssetManagementMode.lagoon
        and any(
            position.pair.get_exchange_account_protocol() == LIGHTER_PROTOCOL
            for position in state.portfolio.get_open_and_frozen_positions()
            if position.is_exchange_account()
        )
    )
    has_onchain_positions = (
        any(
            not p.pair.is_exchange_account()
            for p in state.portfolio.get_open_and_frozen_positions()
        )
        or closed_phantom_positions
        or has_lagoon_lighter_position
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
    if incident and not dry_run:
        reserve_corrections = [correction for correction in corrections if correction.reserve_asset]
        if len(corrections) != 1 or len(reserve_corrections) != 1 or reserve_corrections[0].actual_amount <= reserve_corrections[0].expected_amount:
            raise RuntimeError("Interrupted deposit requires exactly one positive Safe reserve correction; refusing unrelated changes")

    if dry_run:
        logger.info("Dry run found %d accounting correction(s)", len(corrections))
        for correction in corrections:
            logger.info("  Would apply: %s", correction)
        logger.info("Dry run complete: no transactions broadcast and no state written")
        if incident:
            logger.warning("Interrupted HyperCore deposit #%d is not safe to restart; slot %s remains pending", incident.trade.trade_id, state.pending_data_availability_slot)
        web3config.close()
        return

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

    closed_dust_trades = [] if incident else close_hypercore_dust_positions(state.portfolio)
    if closed_dust_trades:
        logger.info(
            "Auto-closed %d Hypercore dust position(s) after accounting corrections",
            len(closed_dust_trades),
        )

    if not skip_save and not incident:
        logger.info("Saving state to %s", store.path)
        store.sync(state)
    else:
        logger.info("Saving the fixed state skipped")

    if not incident:
        web3config.close()

    # Shortcut here
    if unit_testing:
        chain_settle_wait_seconds = 0

    if len(corrections) > 0 and chain_settle_wait_seconds:
        logger.info("Waiting %f seconds to see before reading back new results from on-chain", chain_settle_wait_seconds)
        time.sleep(chain_settle_wait_seconds)

    block_number = get_almost_latest_block_number(web3)

    # Lagoon Lighter needs a final Safe reserve check even though Lighter itself
    # is represented by a synthetic exchange-account position.
    has_onchain_positions_final = any(
        not p.pair.is_exchange_account()
        for p in state.portfolio.get_open_and_frozen_positions()
    ) or has_lagoon_lighter_position
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
        exchange_account_value_func=exchange_account_value_func,
    )

    if incident:
        if not clean:
            raise UncleanState("Final account check failed; at-risk marker and pending slot remain on disk")
        trade = incident.trade
        position = incident.position
        if position.get_quantity() != 0 or position.position_id not in state.portfolio.open_positions:
            raise UncleanState("Interrupted vault position is no longer an empty open position")
        state.portfolio.close_position(position, native_datetime_utc_now())
        trade.reserve_currency_allocated = Decimal(0)
        for key in (
            HYPERCORE_DEPOSIT_CAPITAL_AT_RISK_KEY,
            HYPERCORE_STRANDED_USDC_KEY,
            HYPERCORE_ACCOUNTING_RECONCILIATION_REQUIRED_KEY,
            "retain_reserve_allocation_on_failure",
        ):
            trade.other_data.pop(key, None)
        slot = state.pending_data_availability_slot
        if any(
            t.get_status() not in (TradeStatus.success, TradeStatus.failed, TradeStatus.repaired, TradeStatus.expired)
            for t in state.portfolio.get_all_trades() if t.opened_at == slot
        ):
            raise UncleanState("Cannot consume partial HyperCore slot while a same-slot trade remains unfinished")
        state.last_cycle_at = slot
        state.pending_data_availability_slot = None
        state.cycle += 1
        trade.add_note(f"Partially executed HyperCore decision slot {slot} consumed; do not replay successful sibling trades")
        state.check_if_clean()
        clean, df = check_accounts(
            universe.data_universe.pairs,
            universe.reserve_assets,
            state,
            sync_model,
            block_identifier=get_almost_latest_block_number(web3),
            exchange_account_value_func=exchange_account_value_func,
        )
        if not clean:
            raise UncleanState("Post-resolution account check failed; state was not saved")
        store.sync(state)
        web3config.close()

    output = tabulate(
        df,
        headers='keys',
        tablefmt='rounded_outline',
        disable_numparse=True,
    )

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
