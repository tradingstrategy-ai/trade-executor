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
sending sub-account's API key, which by default only the external trader
holds. An executor given a sub-account's record can transfer from it
directly with :py:func:`tradeexecutor.exchange_account.state.create_exchange_account_funding`.

See ``tradeexecutor/exchange_account/README-Lighter.md``.
"""

import logging
from collections.abc import Iterable, Sequence
from decimal import Decimal

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
)
from tradeexecutor.exchange_account.lighter import (
    LIGHTER_PROTOCOL,
    create_lighter_exchange_account_pairs,
    find_lighter_account_pair,
    get_lighter_master_pair,
    is_lighter_sub_account_pair,
)
from tradeexecutor.exchange_account.state import (
    create_exchange_account_funding,
    create_exchange_account_transfer,
    open_exchange_account_position,
)
from tradeexecutor.exchange_account.sub_account import MASTER_ACCOUNT_ID, SubAccount, get_lighter_account_resolution, get_sub_account_ids
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.pandas_trader.trading_universe_input import CreateTradingUniverseInput
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, create_pair_universe_from_code

logger = logging.getLogger(__name__)


def resolve_lighter_accounts(input: CreateTradingUniverseInput, sub_accounts: Iterable[SubAccount] | None, master_account_index: int | None = None) -> tuple[int, dict[int, str]]:
    """Map the strategy's declared sub-accounts to the Lighter accounts recorded in the state.

    The strategy declares ``Parameters.lighter_sub_accounts`` as
    :py:class:`~tradeexecutor.exchange_account.sub_account.SubAccount` values.
    ``init`` records which Lighter account each id maps to in the state, so a
    running deployment resolves from ``input.state``. Without a state, at
    deployment or in a universe check, placeholder indices are used; they are
    never persisted. A vault deployed before sub-accounts existed passes its
    master index explicitly, as ``strategies/lighter-ai.py`` does.

    :param input:
        Universe construction input carrying the state.
    :param sub_accounts:
        ``Parameters.lighter_sub_accounts`` of the strategy module; passed
        explicitly because not every command gives the input the parameters.
    :param master_account_index:
        Master account for a deployment whose state has no resolution.
    :return:
        Master account index and sub-account index to id.
    :raise ValueError:
        If a running deployment has not resolved every declared sub-account.
    """
    sub_account_ids = get_sub_account_ids(sub_accounts)
    resolution = get_lighter_account_resolution(input.state) if input.state is not None else {}
    if MASTER_ACCOUNT_ID in resolution:
        missing = [account_id for account_id in sub_account_ids if account_id not in resolution]
        if missing:
            raise ValueError(f"Sub-accounts {missing} are declared in Parameters but not resolved in the state; deploy them and run init")
        return resolution[MASTER_ACCOUNT_ID], {resolution[account_id]: account_id for account_id in sub_account_ids}
    if master_account_index is not None and not sub_account_ids:
        return master_account_index, {}
    if input.state is not None:
        raise ValueError("The state does not record the Lighter accounts of this deployment; run init with the deployment artefact")
    # No state: placeholders keep the universe constructible for deployment and checks.
    return 0, {number: account_id for number, account_id in enumerate(sub_account_ids, 1)}


def create_lighter_account_universe(input: CreateTradingUniverseInput, sub_accounts: Iterable[SubAccount] | None, master_account_index: int | None = None) -> TradingStrategyUniverse:
    """Create a static universe with one exchange-account pair per declared Lighter account.

    See :py:func:`resolve_lighter_accounts` for how the declared sub-accounts
    map to Lighter accounts.

    :param input:
        Universe construction input carrying the state.
    :param sub_accounts:
        ``Parameters.lighter_sub_accounts`` of the strategy module.
    :param master_account_index:
        Master account for a deployment whose state has no resolution.
    :return:
        Universe with the master account, the sub-accounts and the native
        Ethereum USDC reserve.
    """
    master, resolved = resolve_lighter_accounts(input, sub_accounts, master_account_index)
    usdc = AssetIdentifier(
        chain_id=ChainId.ethereum.value,
        address=USDC_NATIVE_TOKEN[ChainId.ethereum.value],
        token_symbol="USDC",
        decimals=6,
    )
    pairs = create_lighter_exchange_account_pairs(usdc, master_account_index=master, sub_account_indices=resolved)
    exchange = Exchange(
        chain_id=ChainId.ethereum,
        chain_slug="ethereum",
        exchange_id=1,
        exchange_slug="lighter",
        address=LIGHTER_L1_CONTRACT,
        exchange_type=ExchangeType.derive,
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
        ``True`` when every position already existed and the cycle may trade.
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
    return not missing and not input.execution_context.mode.is_backtesting()


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
        A mapping keyed by sub-account id or index, or a sequence in
        sub-account order.
    :param sub_pairs:
        Sub-account pairs in universe order.
    :return:
        Weight per sub-account index.
    """
    if not weights:
        return {}
    if isinstance(weights, dict):
        return {int(find_lighter_account_pair(sub_pairs, account).get_exchange_account_id()): Decimal(str(weight)) for account, weight in weights.items()}
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
      keyed by sub-account id or a sequence in sub-account order
    - ``lighter_rebalance_band_usd``: tolerated distance from a target, default zero
    - ``lighter_transfer_fee_usd``: fee budgeted per funding transfer, default
      zero, as Lighter charges nothing for same-owner transfers

    :param input:
        Strategy cycle input.
    :return:
        Trades of at most one movement kind.
    """
    if not ensure_lighter_positions(input):
        return []
    state = input.state
    universe = input.strategy_universe
    portfolio = state.portfolio
    reserve_asset = universe.get_reserve_asset()
    timestamp = input.timestamp.to_pydatetime()
    master_pair, sub_pairs = get_lighter_account_pairs(list(universe.iterate_pairs()))

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
