"""Helpers for managing exchange account positions in state.

Exchange account positions (Derive, Hyperliquid, etc.) are created directly
on the state object, bypassing PositionManager and the normal execution pipeline.

Trades are immediately spoofed (marked success) so they never reach routing
or execution.
"""

import datetime
from decimal import Decimal

from eth_defi.compat import native_datetime_utc_now

from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeType


def open_exchange_account_position(
    state: State,
    strategy_cycle_at: datetime.datetime,
    pair: TradingPairIdentifier,
    reserve_currency: AssetIdentifier,
    reserve_amount: Decimal = Decimal(0),
    notes: str | None = None,
) -> list[TradeExecution]:
    """Open an exchange account position by creating a spoofed trade.

    What are exchange accounts?
    ---------------------------

    Exchange accounts represent capital allocated to external trading accounts
    on centralised exchanges (Derive, Hyperliquid) or CCXT-supported exchanges
    (Binance, Bybit, etc.). Unlike on-chain DEX positions, they:

    - Track value via external exchange APIs (not blockchain)
    - Bypass routing and execution completely
    - Use spoofed trades marked success immediately
    - Generate balance updates when account value changes

    How they work
    -------------

    Exchange account positions follow a 4-step lifecycle:

    1. **Position creation**: Call this function with a zero or nominal reserve amount
    2. **Trade spoofing**: Trade is marked success with ``force=True``, bypassing execution
    3. **API sync**: ``ExchangeAccountSyncModel`` queries the exchange API for real account value
    4. **Balance updates**: ``BalanceUpdate`` events track PnL and value changes over time

    The position quantity represents the USD-denominated value of the external account
    (including collateral, open positions, and unrealised PnL).

    Usage in strategies
    -------------------

    In your strategy's ``decide_trades()`` function:

    .. code-block:: python

        def decide_trades(input: StrategyInput) -> list[TradeExecution]:
            state = input.state
            timestamp = input.timestamp

            # Check if exchange account position already exists
            for pos in state.portfolio.open_positions.values():
                if pos.pair.is_exchange_account():
                    return []  # Position exists, nothing to do

            # First cycle: create exchange account position
            open_exchange_account_position(
                state=state,
                strategy_cycle_at=timestamp.to_pydatetime(),
                pair=EXCHANGE_ACCOUNT_PAIR,
                reserve_currency=USDC,
                reserve_amount=Decimal(0),
                notes="Initial exchange account position",
            )

            # CRITICAL: Always return empty list!
            # Exchange account trades must never reach execution
            return []

    The position value will be updated when:

    - The strategy runner calls ``sync_positions()`` during the ``sync_portfolio`` phase
    - The ``correct-accounts`` CLI command is run to sync balances

    Auto-creation by correct-accounts
    ----------------------------------

    If a strategy defines exchange account pairs in ``create_trading_universe()``
    but hasn't created positions yet, the ``correct-accounts`` CLI command will
    automatically create them using this function. This makes setup easier by
    ensuring all exchange account pairs have corresponding positions.

    Spoofed trades and execution bypass
    ------------------------------------

    Exchange account trades are "spoofed" - they're created directly in state
    and immediately marked as successfully executed using ``force=True``. This
    ensures they never reach the routing or execution pipeline:

    - No on-chain transactions are broadcasted
    - No router contracts are called
    - No gas is consumed
    - Asserts in routing/execution will crash if exchange account trades reach them

    This is by design: exchange account operations (deposits, withdrawals, trades)
    happen externally via the exchange's API, not on-chain.

    Reserve deduction and NAV accounting
    -------------------------------------

    When this function is called with ``reserve_amount > 0``, it deducts that
    amount from the portfolio reserves via ``adjust_reserves()``.  This mirrors
    the normal trade flow where ``start_execution()`` calls
    ``move_capital_from_reserves_to_spot_trade()`` — a step that exchange
    account trades bypass entirely.

    For on-chain exchange accounts like GMX, USDC is **transferred** from the
    Lagoon vault's Safe to the exchange (e.g. via ``sendTokens()`` to the GMX
    OrderVault in a multicall).  The exchange account value function
    (e.g. ``create_gmx_account_value_func``) deliberately excludes free USDC
    in the Safe to avoid double counting, because the Lagoon treasury sync
    tracks the Safe's USDC balance separately as the reserve component of NAV.

    If reserves are not properly deducted, the portfolio NAV double-counts the
    transferred USDC — once in stale reserves and once in the exchange account
    position — inflating the NAV and mispricing vault deposits.

    For positions auto-created by ``correct-accounts`` (which passes
    ``reserve_amount=Decimal(1)`` as a placeholder), the Lagoon sync model
    additionally reconciles ``reserve_position.quantity`` from the on-chain
    Safe balance before calculating NAV.  See
    ``LagoonVaultSyncModel.sync_treasury()`` and ``README-GMX-Lagoon.md``
    for the full token flow.

    :param state:
        Current strategy state where the position will be created.

    :param strategy_cycle_at:
        The timestamp of the current strategy cycle.

    :param pair:
        The exchange account trading pair identifier. Must have
        ``pair.kind == TradingPairKind.exchange_account``.

    :param reserve_currency:
        The reserve currency asset (e.g. USDC). Used for accounting.

    :param reserve_amount:
        Nominal reserve amount for the position.
        Defaults to 0 — for on-chain exchange accounts like GMX,
        no capital is locked externally at position creation time.
        The sync process will update this to the actual account value
        from the exchange API.

    :param notes:
        Optional human-readable notes for the trade (e.g. "Initial position",
        "Auto-created by correct-accounts").

    :return:
        List containing the single spoofed trade. This is returned for logging
        purposes only. **Do NOT return this from ``decide_trades()``** - always
        return ``[]`` instead to prevent the trades from reaching execution.

    :raise AssertionError:
        If the pair is not an exchange account pair.

    Supported Exchange Accounts
    ---------------------------

    The following exchange account integrations are available:

    **Derive** (Lyra L2 perpetuals exchange):
        - Implementation: ``tradeexecutor.exchange_account.derive``
        - Documentation: ``tradeexecutor/exchange_account/README-Derive.md``
        - Requires: ``DERIVE_SESSION_PRIVATE_KEY``, ``DERIVE_OWNER_PRIVATE_KEY`` or ``DERIVE_WALLET_ADDRESS``
        - Subaccount support: Yes (via ``exchange_subaccount_id`` in ``other_data``)
        - Example: Vega strategy at ``strategies/vega.py``

    **CCXT** (Centralised exchanges via CCXT library):
        - Implementation: ``tradeexecutor.exchange_account.ccxt_exchange``
        - Documentation: ``tradeexecutor/exchange_account/README-CCXT.md``
        - Supported exchanges: Binance, Bybit, OKX, Hyperliquid (via ``ccxt.hyperliquid``), Aster, and others
        - Requires: Exchange-specific API credentials in environment variables
        - Account ID support: Yes (via ``ccxt_account_id`` in ``other_data``)

    **Hyperliquid** (native perpetuals DEX):
        - Use CCXT integration with ``ccxt_exchange_id="hyperliquid"``
        - Documentation: ``tradeexecutor/exchange_account/README-CCXT.md``
        - Supports spot and perps trading

    See Also
    --------
    - ``ExchangeAccountSyncModel`` for syncing position values
    - ``ExchangeAccountPricingModel`` for pricing (always 1:1 with USD)
    - ``ExchangeAccountValuator`` for revaluing positions via API
    - ``create_derive_value_func_from_credentials`` for Derive setup (in ``utils.py``)
    - ``create_ccxt_exchange_account_value_func`` for CCXT setup (in ``utils.py``)
    - ``create_exchange_account_value_func`` for unified setup (in ``utils.py``)
    """
    assert pair.is_exchange_account(), f"Expected exchange account pair, got: {pair}"

    position, trade, created = state.create_trade(
        strategy_cycle_at=strategy_cycle_at,
        pair=pair,
        quantity=None,
        reserve=reserve_amount,
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=reserve_currency,
        reserve_currency_price=1.0,
        notes=notes or f"Open exchange account position for {pair.get_exchange_account_protocol()}",
        pair_fee=0.0,
        lp_fees_estimated=0,
    )

    # Ensure portfolio_value_at_open is set to at least the reserve amount.
    # On a fresh start with no prior positions and no cash,
    # calculate_total_equity() returns 0, which downstream code
    # (get_capital_tied_at_open_pct) treats as invalid/legacy data.
    # For exchange accounts the position IS the portfolio allocation,
    # so the reserve amount is a reasonable minimum.
    if not position.portfolio_value_at_open:
        position.portfolio_value_at_open = float(reserve_amount)

    # Deduct reserve from portfolio, matching the normal trade flow
    # where start_execution() → move_capital_from_reserves_to_spot_trade()
    # deducts reserves.  Exchange account trades bypass start_execution()
    # entirely (they are spoofed via mark_success(force=True) below),
    # so we must deduct explicitly.
    #
    # For positions auto-created by correct-accounts, reserve_amount is
    # Decimal(1) — a placeholder.  The Lagoon sync model additionally
    # reconciles reserves from the on-chain Safe balance before
    # calculating NAV, which catches any remaining discrepancy.
    if reserve_amount > 0 and reserve_currency.get_identifier() in state.portfolio.reserves:
        state.portfolio.adjust_reserves(
            reserve_currency,
            -reserve_amount,
            f"Capital allocated to exchange account {pair.get_exchange_account_protocol()}",
        )

    # Immediately spoof the trade as successfully executed
    trade.mark_success(
        executed_at=native_datetime_utc_now(),
        executed_price=1.0,
        executed_quantity=reserve_amount,
        executed_reserve=reserve_amount,
        lp_fees=0,
        native_token_price=0,
        force=True,
    )

    # Seed share price state for real capital allocations (not placeholders).
    # Placeholder trades ($0 or $1 from correct-accounts) are excluded —
    # their share_price_state is created from the first valuation sync instead.
    if reserve_amount > 1:
        from tradeexecutor.strategy.position_internal_share_price import (
            create_share_price_state,
        )
        position.share_price_state = create_share_price_state(trade)

    return [trade]
