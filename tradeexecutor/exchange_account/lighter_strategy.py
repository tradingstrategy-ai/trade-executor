"""Strategy helper for Lagoon vaults with a Lighter master account and sub-accounts.

The Safe owns one Lighter master account, which acts as the treasury hub, and
any number of sub-accounts traded by external processes. Each account is its
own exchange-account position. :py:func:`decide_lighter_cash_management`
turns the latest balances into custody trades, one kind of movement per
cycle:

1. Safe liquidity: withdraw what pending Lagoon redemptions need, from the
   master first and then from the sub-accounts with the most free collateral.
2. Deposit idle Safe cash to the master.
3. Withdraw overweight sub-accounts to the Safe.
4. Fund underweight sub-accounts from the master.

Every movement is sized from balances observed at decision time, never from
the expected result of another movement in the same cycle, so the plan stays
correct however the runner orders the trades. Rebalancing converges over
cycles: moving capital from one sub-account to another takes a Safe round
trip, because a sub-account to sub-account transfer must be signed with the
sending sub-account's API key and the executor holds only the master key.

See ``tradeexecutor/exchange_account/README-Lighter.md``.
"""

import logging
from collections.abc import Sequence
from decimal import Decimal

from tradeexecutor.exchange_account.cash_manager import (
    ExchangeCashManager,
    ExchangeCashSnapshot,
    SubAccountBalance,
    plan_sub_account_rebalance,
)
from tradeexecutor.exchange_account.lighter import (
    LIGHTER_PROTOCOL,
    get_lighter_master_pair,
    is_lighter_sub_account_pair,
)
from tradeexecutor.exchange_account.state import (
    create_exchange_account_funding,
    create_exchange_account_transfer,
    open_exchange_account_position,
)
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput

logger = logging.getLogger(__name__)


def parse_lighter_sub_account_weights(raw: str) -> tuple[Decimal, ...]:
    """Parse comma-separated sub-account target weights.

    :param raw:
        For example ``"0.5,0.5"``; an empty string means no weights.
    :return:
        Weights in sub-account creation order.
    """
    if not raw.strip():
        return ()
    return tuple(Decimal(part.strip()) for part in raw.split(","))


def get_lighter_account_pairs(pairs: Sequence[TradingPairIdentifier]) -> tuple[TradingPairIdentifier, list[TradingPairIdentifier]]:
    """Split a universe's Lighter pairs into the master and its sub-accounts.

    :param pairs:
        Strategy universe pairs.
    :return:
        Master pair and sub-account pairs in universe order.
    """
    return get_lighter_master_pair(pairs), [pair for pair in pairs if is_lighter_sub_account_pair(pair)]


def _resolve_weights(weights: object, sub_pairs: list[TradingPairIdentifier]) -> dict[int, Decimal]:
    """Map strategy weight parameters to sub-account indices.

    :param weights:
        A mapping keyed by account index, or a sequence in sub-account order.
    :param sub_pairs:
        Sub-account pairs in universe order.
    :return:
        Weight per sub-account index.
    """
    if not weights:
        return {}
    if isinstance(weights, dict):
        return {int(index): Decimal(str(weight)) for index, weight in weights.items()}
    weights = tuple(weights)
    if len(weights) != len(sub_pairs):
        raise ValueError(f"Got {len(weights)} Lighter sub-account weights for {len(sub_pairs)} sub-accounts")
    return {int(pair.get_exchange_account_id()): Decimal(str(weight)) for pair, weight in zip(sub_pairs, weights)}


def decide_lighter_cash_management(input: StrategyInput) -> list[TradeExecution]:
    """Return this cycle's custody trades for a Lighter master and sub-accounts.

    Reads these strategy parameters:

    - ``lighter_safe_cash_buffer_usd``: USDC kept in the Safe
    - ``lighter_free_collateral_buffer_usd``: free collateral kept on the master
    - ``lighter_min_transfer_usd``: smallest movement, at least Lighter's 1 USDC minimum
    - ``lighter_sub_account_weights``: target share per sub-account, a mapping
      keyed by account index or a sequence in sub-account order
    - ``lighter_rebalance_band_usd``: tolerated distance from a target, default zero
    - ``lighter_transfer_fee_usd``: fee budgeted per funding transfer, default
      zero, as Lighter charges nothing for same-owner transfers

    :param input:
        Strategy cycle input.
    :return:
        Trades of at most one movement kind.
    """
    state = input.state
    universe = input.strategy_universe
    portfolio = state.portfolio
    reserve_asset = universe.get_reserve_asset()
    timestamp = input.timestamp.to_pydatetime()
    master_pair, sub_pairs = get_lighter_account_pairs(list(universe.iterate_pairs()))

    # Position initialisation is a state operation, not a custody transfer.
    missing = [pair for pair in [master_pair, *sub_pairs] if portfolio.get_open_position_for_pair(pair) is None]
    for pair in missing:
        open_exchange_account_position(
            state=state,
            strategy_cycle_at=timestamp,
            pair=pair,
            reserve_currency=reserve_asset,
            reserve_amount=Decimal(0),
            notes=f"Initialise Lighter account {pair.get_exchange_account_id()} position",
        )
    if missing or input.execution_context.mode.is_backtesting():
        return []

    if any(trade.is_exchange_account_transfer_pending() for position in portfolio.get_open_and_frozen_positions() for trade in position.trades.values()):
        return []

    parameters = input.parameters
    minimum = Decimal(str(parameters.lighter_min_transfer_usd))
    master_buffer = Decimal(str(parameters.lighter_free_collateral_buffer_usd))
    position_manager = input.get_position_manager()
    master_position = portfolio.get_open_position_for_pair(master_pair)
    sub_positions = {int(pair.get_exchange_account_id()): portfolio.get_open_position_for_pair(pair) for pair in sub_pairs}
    master_free = Decimal(str(position_manager.get_exchange_account_available_balance(master_pair)))
    sub_balances = {
        int(pair.get_exchange_account_id()): SubAccountBalance(
            equity=Decimal(str(sub_positions[int(pair.get_exchange_account_id())].get_quantity())),
            free=Decimal(str(position_manager.get_exchange_account_available_balance(pair))),
        )
        for pair in sub_pairs
    }

    def transfer(position: TradingPosition, amount: Decimal, deposit: bool, reason: str) -> TradeExecution:
        return create_exchange_account_transfer(
            state=state,
            position=position,
            strategy_cycle_at=timestamp,
            reserve_currency=reserve_asset,
            amount=amount,
            deposit=deposit,
            notes=f"Automatic Lighter cash management: {reason}",
            metadata={"direction": "deposit" if deposit else "withdraw", "protocol": LIGHTER_PROTOCOL, "policy": reason},
        )

    # 1. Safe liquidity, and 2. idle Safe cash, decided by the single-account manager
    # over the aggregate free collateral of all accounts.
    snapshot = ExchangeCashSnapshot(
        safe_usdc=Decimal(str(position_manager.get_current_cash())),
        exchange_available_usdc=master_free + sum((balance.free for balance in sub_balances.values()), Decimal(0)),
        pending_deposits_usdc=Decimal(str(state.sync.treasury.pending_deposits or 0)),
        pending_redemptions_usdc=Decimal(str(state.sync.treasury.pending_redemptions or 0)),
    )
    manager = ExchangeCashManager(
        safe_cash_buffer_usdc=Decimal(str(parameters.lighter_safe_cash_buffer_usd)),
        free_collateral_buffer_usdc=master_buffer,
        minimum_transfer_usdc=minimum,
    )
    decision = manager.decide(snapshot)
    if decision.direction == "withdraw":
        trades = []
        remaining = decision.amount_usdc
        sources = [(master_position, max(Decimal(0), master_free - master_buffer))]
        sources += sorted(
            ((sub_positions[index], balance.free) for index, balance in sub_balances.items()),
            key=lambda item: -item[1],
        )
        for position, available in sources:
            amount = min(remaining, available)
            if amount < minimum:
                continue
            trades.append(transfer(position, amount, deposit=False, reason="safe_liquidity"))
            remaining -= amount
            if remaining < minimum:
                break
        return trades
    if decision.direction == "deposit":
        return [transfer(master_position, decision.amount_usdc, deposit=True, reason="idle_safe_cash")]

    # 3. and 4. Sub-account target weights.
    weights = _resolve_weights(parameters.get("lighter_sub_account_weights"), sub_pairs)
    if not sub_pairs or not weights:
        return []
    movements = plan_sub_account_rebalance(
        master_free_usdc=max(Decimal(0), master_free - master_buffer),
        sub_accounts=sub_balances,
        target_weights=weights,
        minimum_transfer_usdc=minimum,
        rebalance_band_usdc=Decimal(str(parameters.get("lighter_rebalance_band_usd") or 0)),
        transfer_fee_usdc=Decimal(str(parameters.get("lighter_transfer_fee_usd") or 0)),
    )
    trades = []
    for index, amount in movements:
        if amount < 0:
            trades.append(transfer(sub_positions[index], -amount, deposit=False, reason="sub_account_overweight"))
        else:
            trades.append(
                create_exchange_account_funding(
                    state=state,
                    source_position=master_position,
                    destination_position=sub_positions[index],
                    strategy_cycle_at=timestamp,
                    reserve_currency=reserve_asset,
                    amount=amount,
                    notes="Automatic Lighter cash management: sub_account_underweight",
                    metadata={"protocol": LIGHTER_PROTOCOL, "policy": "sub_account_underweight"},
                )
            )
    if trades:
        logger.info("Lighter sub-account rebalance: %s", movements)
    return trades
