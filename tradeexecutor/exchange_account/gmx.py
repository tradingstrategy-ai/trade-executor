"""GMX-specific exchange account support.

Provides the exchange account pair creation and account value function
for GMX perpetuals positions traded through a Lagoon vault.

GMX positions are on-chain (unlike Derive which uses an off-chain API),
so the account value function reads position data directly via
:py:func:`~eth_defi.gmx.valuation.fetch_gmx_total_equity`.

Transfer from Safe to GMX is managed by an external FreqTrade instance
(https://github.com/tradingstrategy-ai/gmx-ccxt-freqtrade),
not by the trade executor's execution pipeline.  The trade executor only
tracks the aggregate value of GMX positions via the account value function
and creates a bookkeeping exchange account position to represent them.

.. todo::

    Devise a mechanism to ensure communication between FreqTrade trades
    and NAV syncing.  Currently, when FreqTrade opens or closes GMX
    positions, USDC moves in/out of the Safe without the trade executor
    knowing.  The Lagoon sync model works around this by reconciling
    reserves from the on-chain Safe balance before calculating NAV,
    but a proper coordination protocol (e.g. webhook, shared state,
    or event-driven notification) would be more robust.
"""

import datetime
import logging
from decimal import Decimal
from typing import Callable

from eth_defi.compat import native_datetime_utc_now
from eth_defi.gmx.config import GMXConfig
from eth_defi.gmx.contracts import get_contract_addresses
from eth_defi.gmx.contracts import get_exchange_router_contract
from eth_defi.gmx.funding import (
    build_claim_funding_fees_call,
    extract_claimed_funding_fees,
    fetch_claimable_funding_fees,
)
from eth_defi.gmx.lagoon.approvals import UNLIMITED, approve_gmx_collateral_via_vault
from eth_defi.timestamp import get_block_timestamp
from eth_defi.gmx.valuation import fetch_gmx_total_equity
from eth_defi.token import fetch_erc20_details
from eth_defi.tx import AssetDelta
from hexbytes import HexBytes

from tradeexecutor.state.balance_update import (
    BalanceUpdate,
    BalanceUpdateCause,
    BalanceUpdatePositionType,
)
from tradeexecutor.state.identifier import (
    AssetIdentifier,
    TradingPairIdentifier,
    TradingPairKind,
)
from tradeexecutor.state.state import State
from tradeexecutor.state.store import StateStore
from tradeexecutor.state.sync import BalanceEventRef

logger = logging.getLogger(__name__)


GMX_FUNDING_CLAIM_INTERVAL = datetime.timedelta(days=1)
"""How often a live executor checks for claimable GMX funding."""

GMX_FUNDING_CLAIM_MINIMUM = Decimal("0.01")
"""Minimum reserve-token amount worth broadcasting a claim for."""

GMX_FUNDING_CLAIM_GAS_LIMIT = 4_000_000
"""Outer ExchangeRouter multicall gas limit, before Lagoon's Safe overhead."""

GMX_FUNDING_LAST_CHECKED_AT = "gmx_funding_last_checked_at"
GMX_FUNDING_LAST_CLAIM_TX = "gmx_funding_last_claim_tx"


def create_gmx_exchange_account_pair(
    quote: AssetIdentifier,
    is_testnet: bool = False,
    claim_funding_fees: bool = False,
) -> TradingPairIdentifier:
    """Create a TradingPairIdentifier for a GMX exchange account.

    Builds the pair with correct ``kind``, ``exchange_name``, and ``other_data``
    fields needed by the sync, pricing, and valuation pipeline.

    The base asset is a synthetic ``GMX-ACCOUNT`` token whose address is set
    to the GMX ExchangeRouter contract. ``pool_address`` and ``exchange_address``
    also point to the ExchangeRouter — the Safe address (which holds GMX
    positions) is resolved at runtime from the execution model's transaction
    builder, not stored in the pair.

    Example::

        from tradeexecutor.exchange_account.gmx import create_gmx_exchange_account_pair

        pair = create_gmx_exchange_account_pair(quote=usdc)

    :param quote:
        Reserve / quote asset (e.g. ``USDC``).
    :param is_testnet:
        Whether this targets a testnet deployment.
    :param claim_funding_fees:
        Opt this strategy into the low-frequency trader funding claim task.
        Disabled by default so existing GMX strategies never gain a new
        transaction side effect.
    :return:
        Fully configured exchange account pair.
    """
    chain = "arbitrum_sepolia" if is_testnet else "arbitrum"
    addresses = get_contract_addresses(chain)

    base = AssetIdentifier(
        chain_id=quote.chain_id,
        address=addresses.exchangerouter,
        token_symbol="GMX-ACCOUNT",
        decimals=6,
    )

    return TradingPairIdentifier(
        base=base,
        quote=quote,
        pool_address=addresses.exchangerouter,
        exchange_address=addresses.exchangerouter,
        internal_id=1,
        internal_exchange_id=1,
        fee=0.0,
        kind=TradingPairKind.exchange_account,
        exchange_name="GMX",
        other_data={
            "exchange_protocol": "gmx",
            "exchange_is_testnet": is_testnet,
            "claim_funding_fees": claim_funding_fees,
        },
    )


def is_gmx_funding_claim_enabled(strategy_universe) -> bool:
    """Return whether the current strategy universe opts into funding claims."""
    iterate_pairs = getattr(strategy_universe, "iterate_pairs", None)
    if iterate_pairs is None:
        return False
    return any(
        pair.is_exchange_account()
        and pair.get_exchange_account_protocol() == "gmx"
        and pair.other_data.get("claim_funding_fees", False)
        for pair in iterate_pairs()
    )


def _get_gmx_exchange_account_position(state: State):
    """Find the open GMX bookkeeping position used for claim throttling."""
    for position in state.portfolio.get_open_positions():
        if (
            position.is_exchange_account()
            and position.pair.get_exchange_account_protocol() == "gmx"
        ):
            return position
    return None


def _record_gmx_funding_claim(
    timestamp: datetime.datetime,
    state: State,
    *,
    web3,
    safe_address: str,
    tx_hash: str,
    receipt: dict,
    raw_amount: int,
    claimed_market_count: int,
) -> BalanceUpdate:
    """Apply an exact mined GMX funding claim to reserve accounting."""
    reserve_position = state.portfolio.get_default_reserve_position()
    amount = reserve_position.asset.convert_to_decimal(raw_amount)
    old_balance = reserve_position.quantity
    event_id = state.portfolio.next_balance_update_id
    state.portfolio.next_balance_update_id += 1
    block_number = receipt["blockNumber"]

    event = BalanceUpdate(
        balance_update_id=event_id,
        position_type=BalanceUpdatePositionType.reserve,
        cause=BalanceUpdateCause.interest,
        asset=reserve_position.asset,
        block_mined_at=get_block_timestamp(web3, block_number),
        strategy_cycle_included_at=timestamp,
        chain_id=reserve_position.asset.chain_id,
        quantity=amount,
        old_balance=old_balance,
        usd_value=float(amount),
        owner_address=safe_address,
        tx_hash=tx_hash,
        log_index=None,
        position_id=None,
        notes="Claimed GMX trader funding fees into strategy reserves",
        block_number=block_number,
        other_data={
            "protocol": "gmx",
            "event": "FundingFeesClaimed",
            "raw_amount": raw_amount,
            "market_count": claimed_market_count,
        },
    )
    reserve_position.quantity += amount
    reserve_position.balance_updates[event.balance_update_id] = event
    state.sync.accounting.balance_update_refs.append(
        BalanceEventRef.from_balance_update_event(event)
    )
    state.sync.accounting.last_updated_at = native_datetime_utc_now()
    return event


def claim_gmx_funding_fees_if_due(
    timestamp: datetime.datetime,
    state: State,
    strategy_universe,
    execution_model,
    store: StateStore,
    *,
    interval: datetime.timedelta = GMX_FUNDING_CLAIM_INTERVAL,
    minimum_amount: Decimal = GMX_FUNDING_CLAIM_MINIMUM,
) -> list[BalanceUpdate]:
    """Claim GMX funding during the existing live-position refresh task.

    Only the strategy reserve token is claimed.  The current Lagoon accounting
    model is single-reserve, so claiming WBTC, WETH or other funding rewards
    would introduce assets that reserve reconciliation cannot represent.

    The last successful check is stored on the GMX bookkeeping position.  This
    avoids a state-schema migration and keeps the hourly valuation scheduler
    from turning into an hourly transaction source.
    """
    if not is_gmx_funding_claim_enabled(strategy_universe):
        return []

    position = _get_gmx_exchange_account_position(state)
    if position is None:
        logger.debug(
            "GMX funding claim enabled, but no GMX exchange-account position exists"
        )
        return []

    last_checked_raw = position.other_data.get(GMX_FUNDING_LAST_CHECKED_AT)
    if last_checked_raw:
        last_checked_at = datetime.datetime.fromisoformat(last_checked_raw)
        if timestamp - last_checked_at < interval:
            return []

    if getattr(execution_model, "disable_broadcast", False):
        logger.info(
            "GMX funding claim skipped because transaction broadcasting is disabled"
        )
        return []

    tx_builder = execution_model.tx_builder
    web3 = execution_model.web3
    safe_address = tx_builder.get_token_delivery_address()
    reserve_position = state.portfolio.get_default_reserve_position()
    reserve_asset = reserve_position.asset

    claims = fetch_claimable_funding_fees(
        web3,
        safe_address,
        token_addresses=[reserve_asset.checksum_address],
    )
    position.other_data[GMX_FUNDING_LAST_CHECKED_AT] = timestamp.isoformat()

    raw_amount = sum(claim.amount for claim in claims)
    if raw_amount < reserve_asset.convert_to_raw_amount(minimum_amount):
        logger.info(
            "No GMX funding claim broadcast: %s %s is below the %s minimum",
            reserve_asset.convert_to_decimal(raw_amount),
            reserve_asset.token_symbol,
            minimum_amount,
        )
        store.sync(state)
        return []

    logger.info(
        "Claiming %s %s of GMX trader funding across %d market/token tuples",
        reserve_asset.convert_to_decimal(raw_amount),
        reserve_asset.token_symbol,
        len(claims),
    )
    bound_call = build_claim_funding_fees_call(web3, claims, safe_address)
    exchange_router = get_exchange_router_contract(web3, GMXConfig(web3).chain)
    tx = tx_builder.sign_transaction(
        exchange_router,
        bound_call,
        gas_limit=GMX_FUNDING_CLAIM_GAS_LIMIT,
        asset_deltas=[AssetDelta(reserve_asset.checksum_address, raw_amount)],
        notes="Claim GMX trader funding fees",
    )
    tx_builder.broadcast_and_wait_transactions_to_complete(
        web3,
        [tx],
        confirmation_block_count=execution_model.confirmation_block_count,
        max_timeout=execution_model.confirmation_timeout,
        revert_reasons=True,
    )
    receipt = web3.eth.get_transaction_receipt(tx.tx_hash)
    if receipt["status"] != 1:
        raise RuntimeError(f"GMX funding claim transaction reverted: {tx.tx_hash}")

    claimed = extract_claimed_funding_fees(
        web3,
        receipt,
        account=safe_address,
        receiver=safe_address,
        token_addresses=[reserve_asset.checksum_address],
    )
    actual_raw_amount = sum(claim.amount for claim in claimed)
    if actual_raw_amount == 0:
        raise RuntimeError(
            f"Successful GMX funding claim emitted no matching events: {tx.tx_hash}"
        )

    event = _record_gmx_funding_claim(
        timestamp,
        state,
        web3=web3,
        safe_address=safe_address,
        tx_hash=tx.tx_hash,
        receipt=receipt,
        raw_amount=actual_raw_amount,
        claimed_market_count=len(claimed),
    )
    position.other_data[GMX_FUNDING_LAST_CLAIM_TX] = tx.tx_hash
    store.sync(state)
    return [event]


def has_gmx_exchange_account_pairs(strategy_universe) -> bool:
    """Detect whether a strategy trades GMX through an exchange account pair.

    The strategy universe itself drives GMX feature discovery — no environment
    variable or CLI flag is needed, so a misconfigured deployment cannot
    silently disable GMX-specific accounting. Used to auto-wire the GMX value
    functions in
    :py:class:`~tradeexecutor.ethereum.ethereum_protocol_adapters.EthereumPairConfigurator`.

    Why GMX needs special handling: GMX positions are opened and closed by an
    external FreqTrade instance that moves USDC between the Lagoon Safe and GMX
    without the trade executor knowing. The tracked reserve balance goes stale
    until treasury settlement reconciles it from on-chain, and any statistics
    point recorded in between combines stale reserve cash with a fresh GMX
    position valuation, producing spurious NAV spikes on the public equity
    curve. Full analysis:
    https://github.com/tradingstrategy-ai/trade-executor/pull/1558#issuecomment-4925733588

    :param strategy_universe:
        Strategy universe. Any object without ``iterate_pairs()``
        (e.g. a non-trading universe stub) counts as not using GMX.
    :return:
        True if the universe contains at least one GMX exchange account pair.
    """
    iterate_pairs = getattr(strategy_universe, "iterate_pairs", None)
    if iterate_pairs is None:
        return False
    for pair in iterate_pairs():
        if pair.is_exchange_account() and pair.get_exchange_account_protocol() == "gmx":
            return True
    return False


def create_gmx_account_value_func(
    execution_model=None,
    *,
    web3=None,
    safe_address: str | None = None,
) -> Callable[..., Decimal]:
    """Create GMX-specific account value function.

    The returned function queries the GMX Reader contract for all open
    positions held by the Safe address, and returns the total equity
    (collateral + unrealised PnL) across all positions via
    :py:func:`~eth_defi.gmx.valuation.fetch_gmx_total_equity`.

    Free USDC sitting in the Safe is tracked separately by Lagoon
    treasury sync, so this function only returns the value locked
    in GMX positions to avoid double counting (``reserve_tokens=[]``).

    The returned function accepts an optional ``block_identifier`` kwarg
    to pin reads to a specific block.  When omitted, ``"latest"`` is used.

    USDC flow and NAV implications
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    When GMX positions are opened, USDC is **transferred** from the Safe
    to the GMX OrderVault via ``sendTokens()`` in an ``ExchangeRouter.multicall()``.
    This reduces the Safe's USDC balance but happens outside the trade engine,
    so ``reserve_position.quantity`` in the portfolio becomes stale.

    The total vault NAV is: ``Safe USDC (reserves) + GMX position value (this function)``.

    To prevent double-counting (stale reserves + real position value),
    ``LagoonVaultSyncModel.sync_treasury()`` reconciles reserves from the
    actual on-chain Safe balance before calculating NAV.  See
    ``README-GMX-Lagoon.md`` for the full security analysis and token flow.

    Can be called with either an *execution_model* (used by the runner)
    or explicit *web3* + *safe_address* (used by CLI commands like
    ``correct-accounts``).

    :param execution_model:
        The execution model (e.g. ``LagoonExecution``) that provides
        Web3 and the transaction builder with the Safe address.
    :param web3:
        Explicit Web3 instance (used when no execution model is available).
    :param safe_address:
        Explicit Safe address (used when no execution model is available).
    :return:
        Function that takes a TradingPairIdentifier (and optional
        ``block_identifier`` kwarg) and returns the total GMX position
        equity in USD.
    """
    if execution_model is None:
        assert web3 is not None and safe_address is not None, (
            "Either execution_model or both web3 and safe_address must be provided"
        )
        _web3 = web3
        _safe_address = safe_address
    else:
        _web3 = None
        _safe_address = None

    def get_gmx_account_value(pair: TradingPairIdentifier, **kwargs) -> Decimal:
        """Get GMX account value for the given pair.

        Uses :py:func:`~eth_defi.gmx.valuation.fetch_gmx_total_equity`
        to read all open positions from the GMX Reader contract.

        :param pair:
            Exchange account trading pair with GMX metadata in other_data.
        :param kwargs:
            Optional ``block_identifier`` to pin reads to a specific block.
        :return:
            Total position equity in USD, or ``Decimal(0)`` if no
            open positions.
        """
        assert pair.is_exchange_account(), f"Not an exchange account pair: {pair}"
        assert pair.get_exchange_account_protocol() == "gmx", (
            f"Not a GMX pair: {pair.get_exchange_account_protocol()}"
        )

        # TODO: Switch to per-block GMX oracle prices when available,
        # so valuations are fully reproducible from block number alone.
        block_id = kwargs.get("block_identifier", "latest")

        if _web3 is not None:
            web3 = _web3
            safe_address = _safe_address
        else:
            web3 = execution_model.web3
            safe_address = execution_model.tx_builder.get_token_delivery_address()

        try:
            result = fetch_gmx_total_equity(
                web3=web3,
                account=safe_address,
                reserve_tokens=[],
                block_identifier=block_id,
            )
            logger.debug(
                "GMX account %s total equity: $%s (block=%s)",
                safe_address,
                result.positions,
                block_id,
            )
            return result.positions

        except Exception as e:
            logger.error(
                "Failed to get GMX account value for %s: %s",
                safe_address,
                e,
            )
            raise

    return get_gmx_account_value


def approve_gmx_trading(
    vault,
    hot_wallet,
    collateral_address: str | None = None,
) -> HexBytes:
    """Approve GMX collateral token to trade via the Lagoon vault.

    Convenience function intended to be called from the trade executor
    console (``trade-executor console``) to initialise GMX trading.
    Approves the collateral token (e.g. USDC) for the GMX
    SyntheticsRouter through the vault's Safe, because
    ``sendTokens()`` in the multicall does ``transferFrom`` via the
    SyntheticsRouter on every trade.

    The approval uses an unlimited amount (``2**256 - 1``).
    The Guard contract ensures that tokens can only flow to
    the pre-configured OrderVault and that order proceeds always
    return to the Safe, so an unlimited allowance does not increase
    the attack surface.

    For the full security analysis of the approval and trading flow,
    see `README-GMX-Lagoon.md <https://github.com/tradingstrategy-ai/web3-ethereum-defi/blob/master/eth_defi/gmx/README-GMX-Lagoon.md>`_
    (section *"Pre-requisite: USDC approval"*).

    Example usage from the console::

        from tradeexecutor.exchange_account.gmx import approve_gmx_trading
        approve_gmx_trading(vault, hot_wallet)

    :param vault:
        ``LagoonVault`` instance (available as ``vault`` in the console).
    :param hot_wallet:
        ``HotWallet`` of the asset manager (available as ``hot_wallet``
        in the console).
    :param collateral_address:
        ERC-20 address of the collateral token to approve.
        Defaults to the vault's underlying token (denomination asset).
    :return:
        Transaction hash of the collateral approval.
    """
    web3 = vault.web3

    # Resolve collateral token
    if collateral_address:
        collateral_token = fetch_erc20_details(web3, collateral_address)
    else:
        collateral_token = vault.underlying_token
    logger.info(
        "Using collateral token: %s (%s)",
        collateral_token.symbol,
        collateral_token.address,
    )

    # Approve collateral for GMX SyntheticsRouter
    logger.info("Approving collateral token for GMX SyntheticsRouter...")
    collateral_tx = approve_gmx_collateral_via_vault(
        vault=vault,
        asset_manager=hot_wallet,
        collateral_token=collateral_token,
        amount=UNLIMITED,
    )
    logger.info("Collateral approval tx: %s", web3.to_hex(collateral_tx))

    logger.info("GMX trading approval complete")
    return collateral_tx


def create_gmx_vault_valuation_func(
    web3,
    safe_address: str,
    reserve_asset: "AssetIdentifier",
) -> Callable[..., float]:
    """Create a GMX-specific vault NAV calculation function.

    For GMX strategies where an external FreqTrade instance
    (https://github.com/tradingstrategy-ai/gmx-ccxt-freqtrade)
    moves USDC between the Safe and GMX, the portfolio's
    ``reserve_position.quantity`` can be stale.  This function
    uses :py:func:`~eth_defi.gmx.valuation.fetch_gmx_total_equity`
    to read both the on-chain Safe USDC balance and GMX position
    equity in a single call at a consistent block.

    The returned callable is passed to
    :py:class:`~tradeexecutor.ethereum.lagoon.vault.LagoonVaultSyncModel`
    as ``calculate_valuation_func``.

    Token flow when FreqTrade opens a GMX position::

        Safe USDC ──sendTokens──▶ OrderVault ──keeper──▶ GMX position

    After this transfer the Safe's USDC balance drops, but the
    portfolio reserves are not updated until the next settlement.
    This function computes NAV as::

        NAV = on-chain Safe USDC + GMX position equity

    instead of relying on the potentially stale reserve quantity.

    Both the reserve value and GMX position value are still kept
    accurate in the portfolio state for historical tracing:

    - **GMX position**: updated by ``ExchangeAccountValuator`` during
      ``revalue_state()`` (before ``sync_treasury()`` runs)
    - **Reserves**: reconciled from on-chain in ``sync_treasury()``

    :param web3:
        Web3 instance connected to the vault's chain.
    :param safe_address:
        Gnosis Safe address that holds the vault's assets.
    :param reserve_asset:
        The reserve currency asset (e.g. USDC).
    :return:
        Callable that takes ``State`` and optional ``block_number`` kwarg,
        returns NAV as float.
    """
    from tradeexecutor.state.state import State

    def calculate_nav(state: State, *, block_number=None) -> float:
        reserve_token = fetch_erc20_details(
            web3,
            reserve_asset.address,
            chain_id=reserve_asset.chain_id,
        )
        # TODO: Switch to per-block GMX oracle prices when available,
        # so valuations are fully reproducible from block number alone.
        result = fetch_gmx_total_equity(
            web3=web3,
            account=safe_address,
            reserve_tokens=[reserve_token],
            block_identifier=block_number or "latest",
        )
        nav = float(result.get_total())
        logger.info(
            "GMX vault valuation: reserves=%s, positions=%s, NAV=%f (block=%s)",
            result.reserves,
            result.positions,
            nav,
            block_number,
        )
        return nav

    return calculate_nav
