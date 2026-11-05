"""lagoon-redeem command.

Redeems all vault shares held by the asset manager from a Lagoon vault:

1. Resumes existing claims and reads the asset manager's remaining shares
2. Approves and requests redemption of all shares (Phase 1 - ERC-7540)
3. Posts new valuation and settles the vault (Phase 2)
4. Finalises the redemption to claim denomination tokens (Phase 3)
5. Updates the state file with the new reserve balance
6. Reports final balances
"""

import logging
import time
from decimal import Decimal
from pathlib import Path
from typing import Optional

from eth_defi.erc_4626.vault_protocol.lagoon.testing import redeem_vault_shares
from eth_defi.lighter.session import create_lighter_session
from eth_defi.lighter.valuation import fetch_lighter_total_equity
from eth_defi.provider.receipt import wait_for_transaction_receipt_robust
from eth_defi.token import USDC_NATIVE_TOKEN, fetch_erc20_details
from tradingstrategy.chain import ChainId
from typer import Option

from tradeexecutor.cli.bootstrap import prepare_cache_and_token_cache, prepare_executor_id, read_lighter_pool_account, resolve_deployment_file
from tradeexecutor.cli.commands import shared_options
from tradeexecutor.cli.commands.app import app
from tradeexecutor.cli.state_lock import with_state_lock
from tradeexecutor.cli.commands.lagoon_utils import (
    create_lagoon_command_context,
    ensure_state_store_exists,
    sync_reserve_balance_to_state,
)
from tradeexecutor.cli.log import setup_logging
from tradeexecutor.exchange_account.sub_account import get_lighter_account_resolution
from tradeexecutor.ethereum.lighter.idle_collateral import refresh_lighter_idle_collateral
from tradeexecutor.exchange_account.lighter import create_executor_lighter_session
from tradeexecutor.state.trade import TradeStatus
from tradeexecutor.state.state import State
from tradeexecutor.strategy.strategy_module import read_strategy_module

logger = logging.getLogger(__name__)


#: Small Lighter balances are not liquid enough to hold up a full Lagoon redemption.
LIGHTER_REDEEM_IGNORE_THRESHOLD_USDC = Decimal(5)


def _check_drained_pool_state(state: State) -> None:
    """Refuse a terminal cash-only NAV while holdings or submissions remain.

    Check held quantities as well as equity: a stale zero price cannot establish
    that a position was exited. Synthetic bridge ledgers use their net equity;
    historical transfers and allocations need not each net to zero.
    A submitted trade also remains unresolved
    before its receipt creates a vault ticket or bridge-in-transit status.
    Historical failed, repaired and expired trades do not block a drained state.
    """
    if any((not p.pair.is_cctp_bridge() and p.get_quantity() != 0) or p.get_equity() >= 0.000001 for p in state.portfolio.get_open_and_frozen_positions()) or any(
        t.is_vault_settlement_in_flight()
        or t.get_status() in {TradeStatus.started, TradeStatus.broadcasted, TradeStatus.cctp_in_transit}
        for t in state.portfolio.get_all_trades()
    ):
        raise RuntimeError("Close all positions and reconcile submissions before final redemption")


def _get_redemption_nav(vault, safe_balance: Decimal, lighter_account_indices: list[int] | None, *, pool_account_drained: bool = False) -> Decimal:
    """Return the NAV to post for a redemption settlement.

    A Lighter exchange-account vault can retain small balances outside its Safe, such
    as the activation deposit or sub-account dust below Lighter's withdrawal
    minimum. Lagoon can only pay a redemption from the Safe, so exclude
    Lighter balances that together stay below
    :data:`LIGHTER_REDEEM_IGNORE_THRESHOLD_USDC` from this terminal
    redemption. Larger external balances must be withdrawn first. Dedicated
    pool deployments must instead prove every external position and collateral
    balance is drained; only then is their terminal NAV the Safe cash balance.
    """
    if pool_account_drained:
        return safe_balance
    nav = vault.fetch_nav()
    if not lighter_account_indices:
        return nav

    with create_lighter_session() as session:
        lighter_equity = sum(
            (fetch_lighter_total_equity(session, index).get_total() for index in lighter_account_indices),
            Decimal(0),
        )

    if lighter_equity < LIGHTER_REDEEM_IGNORE_THRESHOLD_USDC:
        logger.warning(
            "Ignoring small Lighter equity of %s USDC for redemption settlement; "
            "posting Safe balance %s USDC instead",
            lighter_equity,
            safe_balance,
        )
        return safe_balance

    return nav


def _broadcast_and_wait(web3, hot_wallet, func, gas_limit=1_000_000):
    """Broadcast a transaction and wait until all read RPC providers see it.

    Production multi-provider setups (e.g. Arbitrum with sequencer + read RPCs)
    have propagation delays between write and read providers. Without waiting
    for all providers, state reads immediately after a transaction (like
    ``balanceOf``) may hit a stale provider and return zero.
    """
    tx_hash = hot_wallet.transact_and_broadcast_with_contract(func, gas_limit=gas_limit)
    receipt = wait_for_transaction_receipt_robust(
        web3, tx_hash,
        confirmation_block_count=2,
        extra_sleep=2.0,
    )
    assert receipt["status"] == 1, (
        f"Transaction {tx_hash.hex()} failed with status 0. "
        f"Block: {receipt.get('blockNumber')}, gas used: {receipt.get('gasUsed')}"
    )


def _poll_and_finalise_redeem(vault, hot_wallet, web3, share_token, max_attempts=12, poll_delay=5):
    """Poll maxRedeem after settlement and finalise once claimable.

    Avoids the stale-read problem where ``finalise_redeem()`` does a hidden
    ``maxRedeem()`` call immediately after settlement, which can return 0 due
    to RPC lag on live chains.
    """
    max_redeemable_raw = 0
    for attempt in range(1, max_attempts + 1):
        max_redeemable_raw = vault.vault_contract.functions.maxRedeem(hot_wallet.address).call()
        if max_redeemable_raw > 0:
            break
        logger.info("  Waiting for settlement to propagate (attempt %d/%d)...", attempt, max_attempts)
        time.sleep(poll_delay)

    assert max_redeemable_raw > 0, (
        f"Settlement completed but maxRedeem is still 0 for {hot_wallet.address} after polling"
    )

    redeemable_human = share_token.convert_to_decimals(max_redeemable_raw)
    logger.info("  Finalising redemption of %s %s", redeemable_human, share_token.symbol)
    hot_wallet.sync_nonce(web3)
    # Wait for all read RPCs to see the receipt before reading balances
    _broadcast_and_wait(
        web3, hot_wallet,
        vault.finalise_redeem(hot_wallet.address, raw_amount=max_redeemable_raw),
    )


def _claim_leftover_deposits(vault, hot_wallet, web3):
    """Claim any leftover deposits from a previous interrupted run.

    When a deposit was settled but the deposit was never finalised,
    ``maxDeposit(owner)`` returns the claimable amount and shares sit in the
    vault contract waiting to be moved to the hot wallet.

    Uses the 3-argument ERC-7540 ``deposit(assets, receiver, controller)``
    instead of the 2-argument ERC-4626 ``deposit(assets, receiver)``, because
    Lagoon vaults require the controller parameter to identify the original
    deposit request.
    """
    denomination_token = vault.denomination_token
    max_deposit_raw = vault.vault_contract.functions.maxDeposit(hot_wallet.address).call()
    if max_deposit_raw > 0:
        deposit_human = denomination_token.convert_to_decimals(max_deposit_raw)
        logger.info(
            "Found %s %s settled but unclaimed deposit — claiming shares now",
            deposit_human, denomination_token.symbol,
        )
        hot_wallet.sync_nonce(web3)
        # ERC-7540 requires deposit(assets, receiver, controller) to claim
        # async deposits. The 2-arg ERC-4626 deposit() does not work here.
        bound_func = vault.vault_contract.functions.deposit(
            max_deposit_raw,
            hot_wallet.address,
            hot_wallet.address,
        )
        # Wait for all read RPCs to sync before reading the share balance
        _broadcast_and_wait(web3, hot_wallet, bound_func)
        share_balance = vault.share_token.fetch_balance_of(hot_wallet.address)
        logger.info("  Claimed deposit — share balance is now %s %s", share_balance, vault.share_token.symbol)


def _claim_leftover_redemptions(vault, hot_wallet, web3, share_token, lighter_account_indices: list[int] | None, *, pool_account_drained: bool = False) -> bool:
    """Claim any leftover redemptions from a previous interrupted run.

    Handles two states:

    - **State A** (settled but unclaimed): ``maxRedeem() > 0`` — finalise
      immediately to claim denomination tokens.
    - **State B** (pending unsettled): ``pendingRedeemRequest() > 0`` — post
      valuation, settle, poll ``maxRedeem``, then finalise.

    :return:
        ``True`` if this invocation claimed a previous redemption.
    """
    redemption_claimed = False

    # State A: settled but unclaimed
    max_redeemable_raw = vault.vault_contract.functions.maxRedeem(hot_wallet.address).call()
    if max_redeemable_raw > 0:
        redeemable_human = share_token.convert_to_decimals(max_redeemable_raw)
        logger.info(
            "Found %s %s settled but unclaimed shares from a previous redemption — claiming now",
            redeemable_human, share_token.symbol,
        )
        hot_wallet.sync_nonce(web3)
        # Wait for all read RPCs to sync so subsequent balance reads are fresh
        _broadcast_and_wait(
            web3, hot_wallet,
            vault.finalise_redeem(hot_wallet.address, raw_amount=max_redeemable_raw),
        )
        logger.info("  Claimed previous redemption")
        redemption_claimed = True

    # Some Lagoon versions retain the controller's old request amount after a
    # claim. Check the global queue even after a restart following that claim.
    if vault.get_flow_manager().fetch_pending_redemption("latest") == 0:
        return redemption_claimed

    # State B: pending unsettled
    pending_redeem_raw = vault.vault_contract.functions.pendingRedeemRequest(
        0, hot_wallet.address
    ).call()
    if pending_redeem_raw > 0:
        pending_human = share_token.convert_to_decimals(pending_redeem_raw)
        logger.info(
            "Found %s %s pending unsettled redemption — settling and claiming now",
            pending_human, share_token.symbol,
        )
        safe_balance = vault.denomination_token.fetch_balance_of(vault.safe_address)
        nav = _get_redemption_nav(vault, safe_balance, lighter_account_indices, pool_account_drained=pool_account_drained)
        hot_wallet.sync_nonce(web3)
        # Wait for all read RPCs to see valuation before settling
        _broadcast_and_wait(web3, hot_wallet, vault.post_new_valuation(nav))

        hot_wallet.sync_nonce(web3)
        # Wait for all read RPCs to see settlement before polling maxRedeem
        _broadcast_and_wait(web3, hot_wallet, vault.settle_via_trading_strategy_module(nav))

        _poll_and_finalise_redeem(vault, hot_wallet, web3, share_token)
        logger.info("  Settled and claimed previous pending redemption")
        redemption_claimed = True

    return redemption_claimed


@app.command()
@shared_options.with_json_rpc_options()
@with_state_lock(read_only=lambda options: options["simulate"])
def lagoon_redeem(
    id: str = shared_options.id,

    strategy_file: Path = shared_options.strategy_file,
    state_file: Optional[Path] = shared_options.state_file,
    cache_path: Optional[Path] = shared_options.cache_path,

    log_level: str = shared_options.log_level,
    rpc_kwargs: dict | None = None,
    private_key: str = shared_options.private_key,

    vault_address: Optional[str] = shared_options.vault_address,
    vault_adapter_address: Optional[str] = shared_options.vault_adapter_address,

    lighter_account_index: int | None = Option(
        None,
        envvar="LIGHTER_ACCOUNT_INDEX",
        help="Lighter master account index of a deployment whose state does not record its accounts; otherwise every account in the state is valued and this must be one of them.",
    ),

    unit_testing: bool = shared_options.unit_testing,
    simulate: bool = shared_options.simulate,
):
    """Redeem all vault shares from a Lagoon vault for the asset manager.

    - Performs the full ERC-7540 three-phase async redemption
    - Approves and requests redemption of all shares (Phase 1)
    - Posts new valuation and settles the vault (Phase 2)
    - Finalises the redemption to claim denomination tokens back (Phase 3)
    - Updates the state file reserves to reflect the new Safe balance
    """
    id = prepare_executor_id(id, strategy_file)

    logger = setup_logging(log_level=log_level)

    assert private_key, "PRIVATE_KEY is required"
    assert vault_address, "VAULT_ADDRESS is required"
    assert vault_adapter_address, "VAULT_ADAPTER_ADDRESS is required"

    mod = read_strategy_module(strategy_file)
    # The state records which Lighter accounts the deployment owns; a single
    # account deployment without that record names its master explicitly.
    state_file, store = ensure_state_store_exists(id, state_file, simulate=simulate)
    state = store.load()
    pool_account = read_lighter_pool_account(resolve_deployment_file(id, state_file))
    if pool_account is None and (
        state.portfolio.lighter_idle_collateral
        or any(p.pair.is_lighter_pool() for p in state.portfolio.get_all_positions())
    ):
        raise RuntimeError("Lighter pool redemption requires public satellite account metadata")
    resolved_accounts = get_lighter_account_resolution(state)
    if resolved_accounts:
        lighter_account_indices = list(resolved_accounts.values())
        if lighter_account_index is not None and lighter_account_index not in lighter_account_indices:
            raise ValueError(f"LIGHTER_ACCOUNT_INDEX {lighter_account_index} is not one of the deployment's accounts {lighter_account_indices}")
    elif lighter_account_index is not None:
        lighter_account_indices = [lighter_account_index]
    else:
        lighter_account_indices = None

    cache_path, token_cache = prepare_cache_and_token_cache(
        id,
        cache_path,
        unit_testing=unit_testing,
    )

    context = create_lagoon_command_context(
        mod=mod,
        private_key=private_key,
        vault_address=vault_address,
        token_cache=token_cache,
        simulate=simulate,
        **rpc_kwargs,
    )
    web3config = context.web3config
    web3 = context.web3
    hot_wallet = context.hot_wallet
    vault = context.vault

    if pool_account is not None:
        account_index, safe_address = pool_account
        if safe_address.lower() != vault.safe_address.lower():
            raise RuntimeError("Lighter satellite metadata belongs to another Safe")
        # This terminal command posts NAV directly. Require all external
        # capital back before using Safe cash as NAV; never apply the legacy
        # exchange-account dust policy to pool collateral or pool shares.
        with create_executor_lighter_session() as session:
            refresh_lighter_idle_collateral(state, session, account_index, safe_address)
        if state.portfolio.get_lighter_idle_collateral_value() >= 0.000001:
            raise RuntimeError("Return idle Lighter pool collateral through a pool purchase and exit before final redemption")
        _check_drained_pool_state(state)
        satellite_usdc = fetch_erc20_details(web3config.get_connection(ChainId.ethereum), USDC_NATIVE_TOKEN[1])
        if satellite_usdc.fetch_raw_balance_of(safe_address) > 0:
            raise RuntimeError("Reclaim Ethereum satellite USDC before final redemption")
        if not simulate:
            store.sync(state)

    # Required for settle_via_trading_strategy_module()
    vault.trading_strategy_module_address = vault_adapter_address

    denomination_token = vault.denomination_token
    share_token = vault.share_token

    # Pre-flight checks
    eth_balance = web3.eth.get_balance(hot_wallet.address)
    eth_human = eth_balance / 10**18
    logger.info("Asset manager: %s", hot_wallet.address)
    logger.info("  ETH balance: %.6f", eth_human)
    assert eth_human >= 0.001, f"Asset manager has {eth_human:.6f} ETH, need at least 0.001 for gas"

    usdc_before = denomination_token.fetch_balance_of(hot_wallet.address)
    logger.info("  %s balance before: %s", denomination_token.symbol, usdc_before)

    _claim_leftover_deposits(vault, hot_wallet, web3)
    resumed_redemption = _claim_leftover_redemptions(
        vault,
        hot_wallet,
        web3,
        share_token,
        lighter_account_indices,
        pool_account_drained=pool_account is not None,
    )

    share_balance = share_token.fetch_balance_of(hot_wallet.address)
    logger.info("  Share balance: %s %s", share_balance, share_token.symbol)
    if share_balance == 0:
        logger.info("No vault shares remain; skipping a new redemption request (claimed previous redemption: %s)", resumed_redemption)
    else:
        # Phase 1: Request redemption (approve shares + requestRedeem)
        logger.info("Phase 1: Requesting redemption of %s %s shares", share_balance, share_token.symbol)
        redeem_vault_shares(
            web3=web3,
            vault_address=vault_address,
            redeemer=hot_wallet.address,
            hot_wallet=hot_wallet,
            token_cache=token_cache,
        )

        # Phase 2: Settle (post valuation + settleDeposit which also settles redeems)
        logger.info("Phase 2: Settling vault")
        safe_balance = denomination_token.fetch_balance_of(vault.safe_address)
        nav = _get_redemption_nav(vault, safe_balance, lighter_account_indices, pool_account_drained=pool_account is not None)
        logger.info("  Redemption settlement NAV: %s %s", nav, denomination_token.symbol)

        hot_wallet.sync_nonce(web3)
        # Wait for all read RPCs to see valuation before settling
        _broadcast_and_wait(web3, hot_wallet, vault.post_new_valuation(nav))

        hot_wallet.sync_nonce(web3)
        # Wait for all read RPCs to see settlement before polling maxRedeem
        _broadcast_and_wait(web3, hot_wallet, vault.settle_via_trading_strategy_module(nav))

        # Phase 3: Claim redeemed denomination tokens
        logger.info("Phase 3: Finalising redemption")
        _poll_and_finalise_redeem(vault, hot_wallet, web3, share_token)

    # Report on-chain balances after redemption
    safe_balance = denomination_token.fetch_balance_of(vault.safe_address)
    final_share_balance = share_token.fetch_balance_of(hot_wallet.address)
    usdc_after = denomination_token.fetch_balance_of(hot_wallet.address)
    redeemed_amount = usdc_after - usdc_before

    logger.info("Redemption complete")
    logger.info("  Vault Safe %s balance: %s", denomination_token.symbol, safe_balance)
    logger.info("  Redeemer share balance: %s %s", final_share_balance, share_token.symbol)
    logger.info("  Redeemer %s balance: %s (redeemed %s)", denomination_token.symbol, usdc_after, redeemed_amount)

    # Update the state file with the new reserve balance
    if simulate:
        logger.info("Simulation mode — skipping state file update")
        logger.info("  Vault Safe %s balance: %s", denomination_token.symbol, safe_balance)
        logger.info("  Redeemer shares: %s %s", final_share_balance, share_token.symbol)
    else:
        state, reserve_position = sync_reserve_balance_to_state(store, denomination_token, safe_balance)

        reserve_value = reserve_position.get_value()
        reserve_symbol = reserve_position.asset.token_symbol

        logger.info("State file updated: %s", state_file)
        logger.info("  Reserve balance: %s %s", reserve_value, reserve_symbol)
        logger.info("  Vault Safe %s balance: %s", denomination_token.symbol, safe_balance)
        logger.info("  Redeemer shares: %s %s", final_share_balance, share_token.symbol)

    logger.info("All ok")
