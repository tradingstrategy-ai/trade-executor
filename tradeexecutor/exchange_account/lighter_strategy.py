"""Strategy helper for Lagoon vaults with a Lighter master account and sub-accounts.

The Safe owns one Lighter master account, which acts as the treasury hub, and
any number of sub-accounts traded by external processes. Each account is its
own exchange-account position. :py:func:`decide_lighter_cash_management`
turns the latest balances into custody trades, one kind of movement per
cycle:

1. Safe liquidity: withdraw what pending Lagoon redemptions need, from the
   master first and then from the sub-accounts with the most free collateral.
2. Deposit idle Safe cash to the master, optionally capped at fixed sleeve needs.
3. Withdraw overweight sub-accounts to the Safe.
4. Fund underweight sub-accounts from the master.

Percentage weights keep the existing two-way rebalance. Fixed dollar minimums
only fund shortfalls and retain sleeve profits. With Safe surplus retention,
unused master cash is withdrawn only after sleeve funding is complete.

Every movement is sized from balances observed at decision time, never from
the expected result of another movement in the same cycle, so the plan stays
correct however the runner orders the trades. Rebalancing converges over
cycles: moving capital from one sub-account to another takes a Safe round
trip, because a sub-account to sub-account transfer must be signed with the
sending sub-account's API key, which by default only the external trader
holds. An executor given a sub-account's record can transfer from it
directly with :py:func:`tradeexecutor.exchange_account.state.create_exchange_account_funding`.

See ``tradeexecutor/exchange_account/README-Lighter.md``.
"""

import logging
from collections.abc import Collection, Iterable, Sequence
from decimal import Decimal

from eth_defi.lighter.api import LIGHTER_MIN_MAINNET_USDC
from eth_defi.lighter.constants import LIGHTER_L1_CONTRACT
from eth_defi.token import USDC_NATIVE_TOKEN
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import Exchange, ExchangeType
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

from tradeexecutor.exchange_account.cash_manager import (
    ExchangeCashManager,
    ExchangeCashSnapshot,
    SubAccountBalance,
    plan_sub_account_rebalance,
    round_down_to_usdc,
)
from tradeexecutor.exchange_account.lighter import (
    create_lighter_exchange_account_pairs,
    get_lighter_account_id,
    get_lighter_master_pair,
    is_lighter_sub_account_pair,
)
from tradeexecutor.exchange_account.state import open_exchange_account_position
from tradeexecutor.exchange_account.sub_account import LIGHTER_MASTER_ACCOUNT_ID, SubAccount, get_lighter_account_resolution, get_sub_account_ids
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.pandas_trader.trading_universe_input import CreateTradingUniverseInput
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, create_pair_universe_from_code

logger = logging.getLogger(__name__)


def resolve_lighter_accounts(input: CreateTradingUniverseInput, sub_accounts: Iterable[SubAccount] | None) -> tuple[int, dict[int, str]]:
    """Map the strategy's declared sub-accounts to the Lighter accounts recorded in the state.

    The strategy declares ``Parameters.lighter_sub_accounts`` as
    :py:class:`~tradeexecutor.exchange_account.sub_account.SubAccount` values.
    ``init`` records which Lighter account each id maps to in the state, so a
    running deployment resolves from ``input.state``. Without a state, at
    deployment or in a universe check, placeholder indices are used; they are
    never persisted.

    :param input:
        Universe construction input carrying the state.
    :param sub_accounts:
        ``Parameters.lighter_sub_accounts`` of the strategy module; passed
        explicitly because not every command gives the input the parameters.
    :return:
        Master account index and sub-account index to id.
    :raise ValueError:
        If a running deployment has not resolved every declared sub-account.
    """
    sub_account_ids = get_sub_account_ids(sub_accounts)
    resolution = get_lighter_account_resolution(input.state) if input.state is not None else {}
    if LIGHTER_MASTER_ACCOUNT_ID in resolution:
        missing = [account_id for account_id in sub_account_ids if account_id not in resolution]
        if missing:
            raise ValueError(f"Sub-accounts {missing} are declared in Parameters but not resolved in the state; deploy them and run init")
        return resolution[LIGHTER_MASTER_ACCOUNT_ID], {resolution[account_id]: account_id for account_id in sub_account_ids}
    if input.state is not None:
        raise ValueError("The state does not record the Lighter accounts of this deployment; run init with the deployment artefact")
    # No state: placeholders keep the universe constructible for deployment and checks.
    return 0, {number: account_id for number, account_id in enumerate(sub_account_ids, 1)}


def create_lighter_account_universe(input: CreateTradingUniverseInput, sub_accounts: Iterable[SubAccount] | None) -> TradingStrategyUniverse:
    """Create a static universe with one exchange-account pair per declared Lighter account.

    See :py:func:`resolve_lighter_accounts` for how the declared sub-accounts
    map to Lighter accounts.

    :param input:
        Universe construction input carrying the state.
    :param sub_accounts:
        ``Parameters.lighter_sub_accounts`` of the strategy module.
    :return:
        Universe with the master account, the sub-accounts and the native
        Ethereum USDC reserve.
    """
    master, resolved = resolve_lighter_accounts(input, sub_accounts)
    usdc = AssetIdentifier(
        chain_id=ChainId.ethereum.value,
        address=USDC_NATIVE_TOKEN[ChainId.ethereum.value],
        token_symbol="USDC",
        decimals=6,
    )
    pairs = create_lighter_exchange_account_pairs(usdc, master, resolved)
    exchange = Exchange(
        chain_id=ChainId.ethereum,
        chain_slug="ethereum",
        exchange_id=1,
        exchange_slug="lighter",
        address=LIGHTER_L1_CONTRACT,
        exchange_type=ExchangeType.lighter,
        pair_count=len(pairs),
    )
    universe = Universe(
        time_bucket=TimeBucket.d1,
        chains={ChainId.ethereum},
        exchanges={exchange},
        pairs=create_pair_universe_from_code(ChainId.ethereum, pairs),
        candles=None,
        liquidity=None,
    )
    return TradingStrategyUniverse(data_universe=universe, reserve_assets=[usdc])


def ensure_lighter_positions(input: StrategyInput) -> bool:
    """Open a zero-quantity position for every Lighter account pair that has none.

    Position initialisation is a state operation, not a custody transfer, so
    a cycle that opens positions must not also move capital.

    :param input:
        Strategy cycle input.
    :return:
        ``True`` when a position was opened in this cycle.
    """
    state = input.state
    universe = input.strategy_universe
    master_pair, sub_pairs = get_lighter_account_pairs(list(universe.iterate_pairs()))
    missing = [pair for pair in [master_pair, *sub_pairs] if state.portfolio.get_open_position_for_pair(pair) is None]
    for pair in missing:
        open_exchange_account_position(
            state=state,
            strategy_cycle_at=input.timestamp.to_pydatetime(),
            pair=pair,
            reserve_currency=universe.get_reserve_asset(),
            reserve_amount=Decimal(0),
            notes=f"Initialise Lighter account {pair.get_exchange_account_id()} position",
        )
    return bool(missing)


def get_lighter_account_pairs(pairs: Sequence[TradingPairIdentifier]) -> tuple[TradingPairIdentifier, list[TradingPairIdentifier]]:
    """Split a universe's Lighter pairs into the master and its sub-accounts.

    :param pairs:
        Strategy universe pairs.
    :return:
        Master pair and sub-account pairs in universe order.
    """
    return get_lighter_master_pair(pairs), [pair for pair in pairs if is_lighter_sub_account_pair(pair)]


def get_sub_account_targets(sub_accounts: Iterable[SubAccount] | None, sub_account_ids: Collection[str]) -> tuple[dict[str, Decimal], dict[str, Decimal]]:
    """Resolve managed targets without including unmanaged accounts in allocation.

    :param sub_accounts:
        ``Parameters.lighter_sub_accounts``.
    :param sub_account_ids:
        Ids of the sub-account pairs in the universe.
    :return:
        Weight and fixed USDC minimum mappings keyed by sub-account id.
    :raise ValueError:
        If a managed sub-account has no pair or the strategy mixes target modes.
    """
    weights = {}
    amounts = {}
    for sub_account in sub_accounts or ():
        if sub_account.weight is None and sub_account.minimum_usd is None:
            continue
        if sub_account.id not in sub_account_ids:
            raise ValueError(f"Sub-account {sub_account.id!r} is not in the strategy universe")
        if sub_account.minimum_usd is not None:
            amounts[sub_account.id] = sub_account.minimum_usd
        else:
            weights[sub_account.id] = Decimal(str(sub_account.weight))
    if weights and amounts:
        raise ValueError("Lighter sub-accounts must use either weights or fixed minimums, not both")
    return weights, amounts


def _policy_trade(trade: TradeExecution, policy: str) -> TradeExecution:
    """Tag a custody trade with the cash-policy rule that created it."""
    trade.other_data["policy"] = policy
    return trade


def _withdraw_for_safe_liquidity(
    position_manager: PositionManager,
    amount: Decimal,
    master_free: Decimal,
    sub_balances: dict[str, SubAccountBalance],
    minimum: Decimal,
) -> list[TradeExecution]:
    """Withdraw ``amount`` to the Safe from the master first, then from the sub-accounts with the most free collateral."""
    sources = [(LIGHTER_MASTER_ACCOUNT_ID, master_free)]
    sources += sorted(((account_id, balance.free) for account_id, balance in sub_balances.items()), key=lambda item: -item[1])
    trades = []
    remaining = amount
    for account_id, available in sources:
        withdrawal = round_down_to_usdc(min(remaining, available))
        if withdrawal < minimum:
            continue
        trades.append(_policy_trade(position_manager.withdraw_from_lighter(account_id, withdrawal, notes="Automatic Lighter cash management: safe_liquidity"), "safe_liquidity"))
        remaining -= withdrawal
        if remaining < minimum:
            break
    return trades


def decide_lighter_cash_management(input: StrategyInput) -> list[TradeExecution]:
    """Return this cycle's custody trades for a Lighter master and sub-accounts.

    Reads these strategy parameters:

    - ``lighter_safe_cash_buffer_usd``: USDC kept in the Safe
    - ``lighter_free_collateral_buffer_usd``: free collateral kept on the master
    - ``lighter_min_transfer_usd``: smallest movement; defaults to Lighter's 1 USDC minimum
    - ``lighter_sub_accounts``: the declared sub-accounts; those with a
      ``weight`` get that share of investable capital, or use ``minimum_usd``
      for fixed USDC equity minimums; allocation modes cannot be mixed
    - ``lighter_rebalance_band_usd``: tolerated distance from a target, default zero
    - ``lighter_transfer_fee_usd``: fee budgeted per funding transfer, default
      zero, as Lighter charges nothing for same-owner transfers
    - ``lighter_keep_surplus_in_safe``: retain cash beyond sleeve funding needs
      in the Safe; defaults to false and requires fixed minimums

    Every movement is one trade made with the
    :py:class:`~tradeexecutor.strategy.pandas_trader.position_manager.PositionManager`
    custody methods and tagged with the rule that created it in
    ``other_data["policy"]``. Weighted sub-accounts are rebalanced through the
    Safe: an overweight account is withdrawn, and proceeds are deposited to
    the master and sent on in later cycles. Fixed minimums only fund
    shortfalls; sleeve profits are retained. Surplus master cash returns to
    the Safe after funding when ``lighter_keep_surplus_in_safe`` is enabled.

    :param input:
        Strategy cycle input.
    :return:
        Trades of at most one movement kind.
    """
    if input.execution_context.mode.is_backtesting() or ensure_lighter_positions(input):
        return []
    position_manager = input.get_position_manager()
    if position_manager.is_lighter_transfer_pending():
        return []

    parameters = input.parameters
    minimum = Decimal(str(parameters.get("lighter_min_transfer_usd") or LIGHTER_MIN_MAINNET_USDC))
    master_buffer = Decimal(str(parameters.lighter_free_collateral_buffer_usd))
    _master_pair, sub_pairs = get_lighter_account_pairs(list(input.strategy_universe.iterate_pairs()))
    sub_account_ids = [get_lighter_account_id(pair) for pair in sub_pairs]
    weights, amounts = get_sub_account_targets(parameters.get("lighter_sub_accounts"), sub_account_ids)
    master_free = position_manager.get_lighter_free_collateral()
    master_spendable = max(Decimal(0), master_free - master_buffer)
    sub_balances = {
        account_id: SubAccountBalance(equity=position_manager.get_lighter_balance(account_id), free=position_manager.get_lighter_free_collateral(account_id))
        for account_id in sub_account_ids
    }
    keep_surplus_in_safe = parameters.get("lighter_keep_surplus_in_safe", False)
    if keep_surplus_in_safe and not amounts:
        raise ValueError("Keeping Lighter surplus in the Safe requires fixed sub-account minimums")
    transfer_fee = Decimal(str(parameters.get("lighter_transfer_fee_usd") or 0))
    rebalance_band = Decimal(str(parameters.get("lighter_rebalance_band_usd") or 0))
    funding_deficits = [target - sub_balances[account_id].equity for account_id, target in amounts.items() if target > sub_balances[account_id].equity]
    funding_need = sum(funding_deficits, Decimal(0)) + transfer_fee * len(funding_deficits)
    active_funding_need = sum((deficit + transfer_fee for deficit in funding_deficits if deficit > rebalance_band), Decimal(0))

    # 1. Safe liquidity, and 2. idle Safe cash, decided by the single-account manager
    # over the aggregate free collateral of all accounts.
    state = input.state
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
        return _withdraw_for_safe_liquidity(position_manager, decision.amount_usdc, master_spendable, sub_balances, minimum)
    if decision.direction == "deposit":
        amount = decision.amount_usdc
        if keep_surplus_in_safe:
            # Deposit only missing sleeve capital; investor surplus stays liquid in the Safe.
            amount = round_down_to_usdc(min(amount, max(Decimal(0), active_funding_need + master_buffer - master_free)))
        if amount >= minimum and amount > 0:
            return [_policy_trade(position_manager.deposit_to_lighter(amount, notes="Automatic Lighter cash management: idle_safe_cash"), "idle_safe_cash")]

    # 3. and 4. Rebalance managed equity targets; unmanaged accounts stay untouched.
    if not weights and not amounts:
        return []
    movements = plan_sub_account_rebalance(
        master_free_usdc=master_spendable,
        sub_accounts={account_id: sub_balances[account_id] for account_id in (weights or amounts)},
        target_weights=weights or None,
        minimum_amounts_usdc=amounts or None,
        minimum_transfer_usdc=minimum,
        rebalance_band_usdc=rebalance_band,
        transfer_fee_usdc=transfer_fee,
    )
    if keep_surplus_in_safe and not movements:
        # Reserve even small sleeve shortfalls so price drift cannot bounce funding
        # between the Safe and master before a sleeve crosses its rebalance band.
        surplus = round_down_to_usdc(max(Decimal(0), master_spendable - funding_need))
        if surplus >= minimum and surplus > rebalance_band:
            return [_policy_trade(position_manager.withdraw_from_lighter(LIGHTER_MASTER_ACCOUNT_ID, surplus, notes="Automatic Lighter cash management: unallocated_cash"), "unallocated_cash")]
    trades = []
    for account_id, amount in movements:
        if amount < 0:
            trades.append(_policy_trade(position_manager.withdraw_from_lighter(account_id, -amount, notes="Automatic Lighter cash management: sub_account_overweight"), "sub_account_overweight"))
        else:
            trades.append(_policy_trade(position_manager.transfer_lighter_funds(LIGHTER_MASTER_ACCOUNT_ID, account_id, amount, notes="Automatic Lighter cash management: sub_account_underweight"), "sub_account_underweight"))
    if trades:
        logger.info("Lighter sub-account rebalance: %s", movements)
    return trades
