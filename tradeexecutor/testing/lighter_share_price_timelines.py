"""Event timelines for the flow-neutral profit of Lighter account positions.

Each Lighter account, the master and every sub-account, is a position whose
profit must not depend on the capital moved in and out of it. This module
holds the timelines that check this as data, a replay that applies their
events to a :py:class:`~tradeexecutor.state.state.State` through the same
state helpers the executor uses, and the invariants checked after every
event. The unit tests and ``scripts/lagoon/manual-lighter-share-price-charts.py``
both replay these timelines, so the charts show exactly what the tests check.

Events are booked without a chain or Lighter:

- deposits and withdrawals as custody transfer trades on the account itself;
- transfers between accounts as funding trades, with Lighter's fee booked on
  the source;
- gains and losses by changing the value the stub Lighter reports and running
  :py:class:`~tradeexecutor.exchange_account.sync_model.ExchangeAccountSyncModel`,
  the ``correct-accounts`` valuation path, which stamps balance updates with
  the replay's clock.

See ``docs/plans/lighter-subaccount-flow-neutral-profit.md``.
"""

import dataclasses
import datetime
import enum
import math
from dataclasses import dataclass, field
from decimal import Decimal

from eth_defi.lighter.constants import LIGHTER_L1_CONTRACT
from eth_defi.token import USDC_NATIVE_TOKEN
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import Exchange, ExchangeType
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

from tradeexecutor.exchange_account.lighter import create_lighter_exchange_account_pairs
from tradeexecutor.exchange_account.state import (
    close_exchange_account_position,
    complete_exchange_account_funding,
    complete_exchange_account_transfer,
    create_exchange_account_funding,
    create_exchange_account_transfer,
    open_exchange_account_position,
)
from tradeexecutor.exchange_account.sync_model import ExchangeAccountSyncModel
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.statistics.core import update_statistics
from tradeexecutor.strategy.execution_context import ExecutionMode
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, create_pair_universe_from_code
from tradeexecutor.strategy.position_internal_share_price import calculate_flow_neutral_profit_usd

#: Lighter account index of the master account.
MASTER_ACCOUNT_INDEX = 700

#: Lighter account index of the first sub-account; later ones count down, as Lighter's do.
FIRST_SUB_ACCOUNT_INDEX = 2**48 - 1000

#: USDC held in the Safe before any timeline event; enough for the
#: million-dollar deposits of the tiny-remainders timeline, so the reserve
#: never goes negative in the portfolio charts.
SAFE_USDC = Decimal(10_000_000)

#: Clock of the first event.
TIMELINE_START = datetime.datetime(2026, 1, 1)

#: Simulated time between two events.
EVENT_INTERVAL = datetime.timedelta(hours=1)

#: Absolute tolerance of the invariant checks, in USD, units or price; they
#: also allow a 1e-9 relative difference, for float noise on large amounts.
TOLERANCE = 1e-9

#: Account id of the master account.
MASTER = "master"


class TimelineEventKind(enum.Enum):
    """What happens to an account in one timeline event."""

    #: First valuation of an account opened with a placeholder trade; its value is capital, not profit.
    seed = "seed"
    #: Safe → account.
    deposit = "deposit"
    #: Account → Safe; the Safe can receive less than the account is debited.
    withdraw = "withdraw"
    #: Transfer to another account of the vault, with Lighter's fee on the source.
    fund = "fund"
    #: Lighter reports a new account value: a gain or a loss.
    revalue = "revalue"
    #: A further real allocation with ``open_exchange_account_position()``.
    allocate = "allocate"
    #: The account is closed and its residual written off.
    close = "close"


#: Events that move capital and must not move the return or the USD profit, apart from fees.
FLOW_EVENTS = frozenset({TimelineEventKind.deposit, TimelineEventKind.withdraw, TimelineEventKind.fund, TimelineEventKind.allocate})


@dataclass(slots=True, frozen=True)
class TimelineEvent:
    """One event of a timeline."""

    #: What happens.
    kind: TimelineEventKind

    #: Account the event happens to; the source of a transfer.
    account: str

    #: USDC moved: deposited, debited by a withdrawal, transferred or allocated.
    #: For ``seed`` and ``revalue``, the account value Lighter reports afterwards.
    amount: Decimal = Decimal(0)

    #: Destination account of a transfer.
    destination: str | None = None

    #: For ``fund``, Lighter's fee charged to the source on top of the amount.
    #: For ``withdraw``, how much less than the debit the Safe receives.
    fee: Decimal = Decimal(0)


@dataclass(slots=True, frozen=True)
class AccountExpectation:
    """Expected figures of one account after the last event of a timeline."""

    #: Internal share supply, or ``None`` to skip.
    units: float | None

    #: Internal share price, or ``None`` to skip.
    share_price: float | None

    #: :py:meth:`TradingPosition.get_total_profit_percent`.
    return_pct: float

    #: :py:meth:`TradingPosition.get_total_profit_usd`.
    profit_usd: float

    #: Whether the position is closed.
    closed: bool = False


@dataclass(slots=True, frozen=True)
class Timeline:
    """A named sequence of events with the figures it must end at."""

    #: Short id, also used as the sub-account id in charts.
    timeline_id: str

    #: What the timeline exercises.
    description: str

    #: Accounts the timeline uses, the master or sub-account ids.
    accounts: tuple[str, ...]

    #: Events, in order.
    events: tuple[TimelineEvent, ...]

    #: Expected figures per account after the last event.
    expected: dict[str, AccountExpectation] = field(default_factory=dict)


def _event(kind: TimelineEventKind, account: str, amount: str | int = 0, destination: str | None = None, fee: str | int = 0) -> TimelineEvent:
    """Build an event from literal amounts."""
    return TimelineEvent(kind=kind, account=account, amount=Decimal(str(amount)), destination=destination, fee=Decimal(str(fee)))


_K = TimelineEventKind

#: The timelines of the plan's state unit tests. Expected figures were computed
#: with the share-price functions and the flow-neutral identity
#: ``value + withdrawn - invested``.
TIMELINES: tuple[Timeline, ...] = (
    Timeline(
        timeline_id="gain-before-inflow",
        description="A deposit after a gain mints shares at the higher price and leaves the return unchanged",
        accounts=("gain-before-inflow",),
        events=(
            _event(_K.deposit, "gain-before-inflow", 10),
            _event(_K.revalue, "gain-before-inflow", 11),
            _event(_K.deposit, "gain-before-inflow", 11),
            _event(_K.revalue, "gain-before-inflow", "24.2"),
        ),
        expected={"gain-before-inflow": AccountExpectation(units=20, share_price=1.21, return_pct=0.21, profit_usd=3.2)},
    ),
    Timeline(
        timeline_id="inflow-before-loss",
        description="A loss after two deposits, then a partial withdrawal that changes neither figure",
        accounts=("inflow-before-loss",),
        events=(
            _event(_K.deposit, "inflow-before-loss", 10),
            _event(_K.deposit, "inflow-before-loss", 10),
            _event(_K.revalue, "inflow-before-loss", 18),
            _event(_K.withdraw, "inflow-before-loss", 9),
        ),
        expected={"inflow-before-loss": AccountExpectation(units=10, share_price=0.9, return_pct=-0.1, profit_usd=-2)},
    ),
    Timeline(
        timeline_id="exit-and-reentry",
        description="A full withdrawal and a new deposit continue the same share-price series",
        accounts=("exit-and-reentry",),
        events=(
            _event(_K.deposit, "exit-and-reentry", 10),
            _event(_K.revalue, "exit-and-reentry", 9),
            _event(_K.withdraw, "exit-and-reentry", 9),
            _event(_K.deposit, "exit-and-reentry", 9),
            _event(_K.revalue, "exit-and-reentry", "9.9"),
        ),
        expected={"exit-and-reentry": AccountExpectation(units=10, share_price=0.99, return_pct=-0.01, profit_usd=-0.1)},
    ),
    Timeline(
        timeline_id="funding-chain",
        description="Master funds a sub-account, which funds another, each transfer paying a 0.5 USDC fee",
        accounts=(MASTER, "chain-trend", "chain-arb"),
        events=(
            _event(_K.deposit, MASTER, 10),
            _event(_K.fund, MASTER, 4, destination="chain-trend", fee="0.5"),
            _event(_K.revalue, "chain-trend", "4.4"),
            _event(_K.fund, "chain-trend", "2.2", destination="chain-arb", fee="0.5"),
            _event(_K.revalue, "chain-arb", "1.98"),
            _event(_K.withdraw, MASTER, "5.5"),
            _event(_K.withdraw, "chain-trend", "1.7"),
            _event(_K.withdraw, "chain-arb", "1.98"),
        ),
        expected={
            MASTER: AccountExpectation(units=0, share_price=0.95, return_pct=-0.05, profit_usd=-0.5),
            "chain-trend": AccountExpectation(units=0, share_price=0.975, return_pct=-0.025, profit_usd=-0.1),
            "chain-arb": AccountExpectation(units=0, share_price=0.9, return_pct=-0.1, profit_usd=-0.22),
        },
    ),
    Timeline(
        timeline_id="valuation-seeded",
        description="An account opened with a placeholder gets its capital from the first valuation",
        accounts=("valuation-seeded",),
        events=(
            _event(_K.seed, "valuation-seeded", 5),
            _event(_K.deposit, "valuation-seeded", 5),
            _event(_K.revalue, "valuation-seeded", 11),
        ),
        expected={"valuation-seeded": AccountExpectation(units=10, share_price=1.1, return_pct=0.1, profit_usd=1)},
    ),
    Timeline(
        timeline_id="withdrawal-fee",
        description="A withdrawal credited below its debit is a USD loss; the return excludes it",
        accounts=("withdrawal-fee",),
        events=(
            _event(_K.deposit, "withdrawal-fee", 10),
            _event(_K.withdraw, "withdrawal-fee", 10, fee="0.1"),
        ),
        expected={"withdrawal-fee": AccountExpectation(units=0, share_price=1, return_pct=0, profit_usd=-0.1)},
    ),
    Timeline(
        timeline_id="write-off-and-close",
        description="A residual written off on close; the closed return is money-weighted, not the zeroed share price",
        accounts=("write-off-and-close",),
        events=(
            _event(_K.deposit, "write-off-and-close", 10),
            _event(_K.revalue, "write-off-and-close", "10.4"),
            _event(_K.withdraw, "write-off-and-close", 10),
            _event(_K.close, "write-off-and-close"),
        ),
        expected={"write-off-and-close": AccountExpectation(units=None, share_price=None, return_pct=0, profit_usd=0, closed=True)},
    ),
    Timeline(
        timeline_id="total-loss-and-refund",
        description="A deposit into an account valued at zero starts a new share-price series instead of crashing",
        accounts=("total-loss-and-refund",),
        events=(
            _event(_K.deposit, "total-loss-and-refund", 10),
            _event(_K.revalue, "total-loss-and-refund", 0),
            _event(_K.deposit, "total-loss-and-refund", 5),
            _event(_K.revalue, "total-loss-and-refund", "5.5"),
        ),
        expected={"total-loss-and-refund": AccountExpectation(units=5, share_price=1.1, return_pct=0.1, profit_usd=-9.5)},
    ),
    Timeline(
        timeline_id="repeated-allocation",
        description="Further allocations, of any size, mint into the existing share-price state instead of replacing it",
        accounts=("repeated-allocation",),
        events=(
            _event(_K.seed, "repeated-allocation", 5),
            _event(_K.allocate, "repeated-allocation", 5),
            _event(_K.allocate, "repeated-allocation", 1),
        ),
        expected={"repeated-allocation": AccountExpectation(units=11, share_price=1, return_pct=0, profit_usd=0)},
    ),
    Timeline(
        timeline_id="tiny-remainders",
        description="Real balances of 0.0005 USDC left on million-dollar positions are kept, not rounded to zero",
        accounts=("small-exit", "small-value"),
        events=(
            _event(_K.deposit, "small-exit", 1_000_000),
            _event(_K.withdraw, "small-exit", "999999.9995"),
            _event(_K.deposit, "small-value", 1_000_000),
            _event(_K.revalue, "small-value", "0.0005"),
            _event(_K.deposit, "small-value", 1),
        ),
        expected={
            "small-exit": AccountExpectation(units=0.0005, share_price=1, return_pct=0, profit_usd=0),
            # The deposit mints at the near-zero price instead of starting a new
            # series, which would read about +0.05%. Shares are not compared: the
            # float value of 1,000,000 - 999,999.9995 is off by about 1e-7
            # relative, and minting at that price magnifies it.
            "small-value": AccountExpectation(units=None, share_price=5e-10, return_pct=-0.9999999995, profit_usd=-999_999.9995),
        },
    ),
)


class TimelineInvariantError(AssertionError):
    """An account broke a flow-neutrality invariant or ended at an unexpected figure."""


@dataclass(slots=True)
class AccountSnapshot:
    """Figures of one account, taken before and after each event."""

    #: Whether the position has a share-price state.
    tracked: bool
    #: Internal share supply.
    units: float
    #: Internal share price as last updated.
    share_price: float
    #: Booked account value.
    value: float
    #: Flow-neutral USD profit, or ``None`` while unknown.
    profit_usd: float | None


class TimelineReplay:
    """Apply timeline events to a fresh state with a master and sub-account positions.

    Every account is opened with a placeholder trade, as ``correct-accounts``
    does. The replay keeps the value the stub Lighter reports for each
    account and moves it with every flow, so a valuation books a balance
    update only for a ``seed`` or ``revalue`` event. It also keeps a ledger of
    the gains, losses and fees booked on each account and of the capital
    that came from or went back to the Safe, for invariant 4.

    :param account_ids:
        Accounts to open: the master and sub-account ids. The master is
        always opened.
    :param allow_unknown_withdrawn:
        Accept accounts whose withdrawn total is unknown, as in a state saved
        before it was tracked, and skip invariants 2 and 4 for them. In a
        fresh replay an unknown total is a failure, so the checks cannot pass
        without checking anything.
    """

    def __init__(self, account_ids: list[str] | tuple[str, ...], allow_unknown_withdrawn: bool = False):
        self.allow_unknown_withdrawn = allow_unknown_withdrawn
        self.usdc = AssetIdentifier(chain_id=1, address=USDC_NATIVE_TOKEN[1], token_symbol="USDC", decimals=6)
        self.state = State()
        self.state.portfolio.initialise_reserves(self.usdc, reserve_token_price=1.0)
        self.state.portfolio.get_reserve_position(self.usdc).quantity = SAFE_USDC
        self.clock = TIMELINE_START

        sub_account_ids = [account for account in dict.fromkeys(account_ids) if account != MASTER]
        sub_accounts = {FIRST_SUB_ACCOUNT_INDEX - offset: account for offset, account in enumerate(sub_account_ids)}
        pairs = create_lighter_exchange_account_pairs(self.usdc, MASTER_ACCOUNT_INDEX, sub_accounts)
        names = [MASTER, *sub_account_ids]
        #: Pair of each account.
        self.pairs: dict[str, TradingPairIdentifier] = dict(zip(names, pairs))
        self._names_by_account_index = {str(pair.get_exchange_account_id()): name for name, pair in self.pairs.items()}
        for pair in self.pairs.values():
            open_exchange_account_position(state=self.state, strategy_cycle_at=self.clock, pair=pair, reserve_currency=self.usdc, reserve_amount=Decimal(0))
        #: Position of each account; a closed account keeps its closed position.
        self.positions: dict[str, TradingPosition] = {
            self._names_by_account_index[str(position.pair.get_exchange_account_id())]: position
            for position in self.state.portfolio.open_positions.values()
        }

        #: Value the stub Lighter reports for each account.
        self.lighter_values: dict[str, Decimal] = {name: Decimal(0) for name in names}
        #: Gains minus losses booked on each account.
        self.gains: dict[str, float] = {name: 0.0 for name in names}
        #: Fees each account paid.
        self.fees: dict[str, float] = {name: 0.0 for name in names}
        #: Capital from the Safe, including seeds and allocations, minus what the Safe received back.
        self.net_deposits = 0.0
        #: Events applied so far, with the clock of each.
        self.applied: list[tuple[datetime.datetime, TimelineEvent]] = []

        self._sync_model = ExchangeAccountSyncModel(self._fetch_account_value)
        update_statistics(self.clock, self.state.stats, self.state.portfolio, ExecutionMode.unit_testing)

    def create_strategy_universe(self) -> TradingStrategyUniverse:
        """Create the strategy universe of the replayed accounts, as a live Lighter strategy has.

        Mirrors :py:func:`tradeexecutor.exchange_account.lighter_strategy.create_lighter_account_universe`,
        which resolves the accounts from deployment records a replay does not
        have: one exchange-account pair per account and no candles. Chart
        functions read the pairs from it.

        :return:
            Universe with the replay's pairs and its USDC reserve.
        """
        pairs = list(self.pairs.values())
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
        return TradingStrategyUniverse(data_universe=universe, reserve_assets=[self.usdc])

    def _fetch_account_value(self, pair: TradingPairIdentifier, block_identifier: int | None = None) -> Decimal:
        """Stub Lighter equity read: the value the replay tracks for the account.

        :param block_identifier:
            Unused; the signature matches the account-value functions the sync model calls.
        """
        return self.lighter_values[self._names_by_account_index[str(pair.get_exchange_account_id())]]

    def snapshot(self, account: str) -> AccountSnapshot:
        """Take the figures of one account."""
        position = self.positions[account]
        state = position.share_price_state
        return AccountSnapshot(
            tracked=state is not None,
            units=state.total_supply if state else 0.0,
            share_price=state.current_share_price if state else 0.0,
            value=float(position.get_quantity()) if position.is_open() else 0.0,
            profit_usd=calculate_flow_neutral_profit_usd(position),
        )

    def apply(self, event: TimelineEvent) -> None:
        """Apply one event, record statistics, and check the invariants.

        :raise TimelineInvariantError:
            If any account breaks an invariant.
        """
        self.clock += EVENT_INTERVAL
        before = {account: self.snapshot(account) for account in self.positions}
        position = self.positions[event.account]
        match event.kind:
            case TimelineEventKind.seed | TimelineEventKind.revalue:
                previous = self.lighter_values[event.account]
                self.lighter_values[event.account] = event.amount
                if event.kind == TimelineEventKind.seed:
                    self.net_deposits += float(event.amount - previous)
                else:
                    self.gains[event.account] += float(event.amount - previous)
                self._sync_model.sync_positions(self.clock, self.state, None, None)
            case TimelineEventKind.deposit:
                trade = create_exchange_account_transfer(state=self.state, position=position, strategy_cycle_at=self.clock, reserve_currency=self.usdc, amount=event.amount, deposit=True, notes="Timeline deposit", metadata={})
                self._broadcast(trade)
                complete_exchange_account_transfer(state=self.state, position=position, trade=trade, executed_amount=event.amount, executed_reserve=event.amount, executed_at=self.clock)
                self.lighter_values[event.account] += event.amount
                self.net_deposits += float(event.amount)
            case TimelineEventKind.withdraw:
                credit = event.amount - event.fee
                trade = create_exchange_account_transfer(state=self.state, position=position, strategy_cycle_at=self.clock, reserve_currency=self.usdc, amount=event.amount, deposit=False, notes="Timeline withdrawal", metadata={})
                self._broadcast(trade)
                complete_exchange_account_transfer(state=self.state, position=position, trade=trade, executed_amount=-event.amount, executed_reserve=credit, executed_at=self.clock)
                self.lighter_values[event.account] -= event.amount
                self.fees[event.account] += float(event.fee)
                self.net_deposits -= float(credit)
            case TimelineEventKind.fund:
                destination = self.positions[event.destination]
                trade = create_exchange_account_funding(state=self.state, source_position=position, destination_position=destination, strategy_cycle_at=self.clock, reserve_currency=self.usdc, amount=event.amount, notes="Timeline transfer")
                trade.other_data["lighter_transfer_tx_hash"] = f"0xtimeline{len(self.applied)}"
                self._broadcast(trade)
                complete_exchange_account_funding(state=self.state, trade=trade, executed_amount=event.amount, fee=event.fee, executed_at=self.clock)
                self.lighter_values[event.account] -= event.amount + event.fee
                self.lighter_values[event.destination] += event.amount
                self.fees[event.account] += float(event.fee)
            case TimelineEventKind.allocate:
                trades = open_exchange_account_position(state=self.state, strategy_cycle_at=self.clock, pair=self.pairs[event.account], reserve_currency=self.usdc, reserve_amount=event.amount)
                # The spoofed allocation trade is stamped with the wall clock,
                # right for a live executor; put it on the replay's clock.
                for trade in trades:
                    trade.executed_at = self.clock
                if position.share_price_state is not None:
                    position.share_price_state = dataclasses.replace(position.share_price_state, last_updated_at=self.clock)
                self.lighter_values[event.account] += event.amount
                self.net_deposits += float(event.amount)
            case TimelineEventKind.close:
                # The residual is written off: a loss on the books.
                self.gains[event.account] -= float(position.get_quantity())
                close_exchange_account_position(state=self.state, position=position, executed_at=self.clock, notes="Timeline close")
                self.lighter_values[event.account] = Decimal(0)
        self.applied.append((self.clock, event))
        update_statistics(self.clock, self.state.stats, self.state.portfolio, ExecutionMode.unit_testing)
        self.check_invariants(event, before)

    def replay(self, timeline: Timeline) -> None:
        """Apply every event of a timeline."""
        for event in timeline.events:
            self.apply(event)

    def _broadcast(self, trade: TradeExecution) -> None:
        """Start and broadcast a custody trade, as the router does before completing it."""
        self.state.start_execution(self.clock, trade)
        trade.mark_broadcasted(self.clock)

    def check_invariants(self, event: TimelineEvent, before: dict[str, AccountSnapshot]) -> None:
        """Check the four invariants after an event.

        1. A flow does not move the return: the share price stays, except a
           transfer source's, which falls by its fee over its value.
        2. A flow does not move the USD profit, except by the fee it pays.
        3. A flow mints or burns ``amount / share price`` units; a valuation
           mints and burns none.
        4. Each account's USD profit equals its booked gains minus its fees;
           over all accounts, the change in value minus net deposits.
           Accounts whose withdrawn total is unknown, from an old state, are
           not checked for 2 and 4, and fail the check unless the replay
           allows them.

        :raise TimelineInvariantError:
            Naming the event, the account and the figures.
        """
        after = {account: self.snapshot(account) for account in self.positions}
        where = f"event {len(self.applied)} ({event.kind.value} {event.amount} on {event.account})"

        def fail(account: str, message: str) -> None:
            raise TimelineInvariantError(f"{where}, account {account}: {message}")

        def close_to(a: float, b: float) -> bool:
            return math.isclose(a, b, rel_tol=1e-9, abs_tol=TOLERANCE)

        for account, snapshot in after.items():
            if snapshot.tracked and snapshot.profit_usd is None and not self.allow_unknown_withdrawn:
                fail(account, "the withdrawn total is unknown in a fresh replay")

        if event.kind in FLOW_EVENTS:
            moves: dict[str, tuple[float, float]] = {}
            if event.kind == TimelineEventKind.fund:
                moves[event.account] = (-float(event.amount), float(event.fee))
                moves[event.destination] = (float(event.amount), 0.0)
            elif event.kind == TimelineEventKind.withdraw:
                moves[event.account] = (-float(event.amount), float(event.fee))
            else:
                moves[event.account] = (float(event.amount), 0.0)
            for account, (amount, fee) in moves.items():
                old, new = before[account], after[account]
                if not old.tracked:
                    # The first flow creates the share-price state at the initial price.
                    if not close_to(new.units, amount):
                        fail(account, f"first flow minted {new.units} units for {amount} USDC")
                    continue
                if old.share_price <= 0 and amount > 0:
                    # A deposit after a total loss starts a new series at the initial price.
                    if not close_to(new.units, amount) or not close_to(new.share_price, 1.0):
                        fail(account, f"deposit after a total loss gave {new.units} units at {new.share_price}")
                    continue
                expected_price = old.share_price * (old.value - fee) / old.value if fee and amount < 0 and event.kind == TimelineEventKind.fund else old.share_price
                if not close_to(new.share_price, expected_price):
                    fail(account, f"share price moved from {old.share_price} to {new.share_price}, expected {expected_price}")
                if not close_to(new.units - old.units, amount / expected_price):
                    fail(account, f"units moved by {new.units - old.units}, expected {amount / expected_price}")
                if old.profit_usd is not None and new.profit_usd is not None and not close_to(new.profit_usd, old.profit_usd - fee):
                    fail(account, f"USD profit moved from {old.profit_usd} to {new.profit_usd}, expected only the fee {fee}")
        elif event.kind in (TimelineEventKind.revalue, TimelineEventKind.seed):
            for account in self.positions:
                old, new = before[account], after[account]
                if old.tracked and not close_to(new.units, old.units):
                    fail(account, f"a valuation moved units from {old.units} to {new.units}")

        for account in self.positions:
            profit = after[account].profit_usd
            if profit is not None and not close_to(profit, self.gains[account] - self.fees[account]):
                fail(account, f"USD profit {profit} differs from gains {self.gains[account]} minus fees {self.fees[account]}")
        if any(snapshot.tracked and snapshot.profit_usd is None for snapshot in after.values()):
            # An old state's unknown withdrawn total makes the vault-wide sum unknown too.
            return
        total_profit = sum(snapshot.profit_usd or 0.0 for snapshot in after.values())
        total_value = sum(snapshot.value for snapshot in after.values())
        if not close_to(total_profit, total_value - self.net_deposits):
            fail("all", f"summed USD profit {total_profit} differs from value {total_value} minus net deposits {self.net_deposits}")

    def check_expectations(self, timeline: Timeline) -> None:
        """Check each account's figures after the timeline's last event.

        :raise TimelineInvariantError:
            Naming the account, the figure and both values.
        """
        for account, expected in timeline.expected.items():
            position = self.positions[account]
            state = position.share_price_state
            observed = {
                "closed": position.is_closed(),
                "return_pct": position.get_total_profit_percent(),
                "profit_usd": position.get_total_profit_usd(),
            }
            if expected.units is not None:
                observed["units"] = state.total_supply
            if expected.share_price is not None:
                observed["share_price"] = position.get_share_price_profit().current_share_price
            for name, value in observed.items():
                wanted = getattr(expected, name)
                if (value != wanted) if isinstance(wanted, bool) else not math.isclose(value, wanted, rel_tol=1e-9, abs_tol=1e-6):
                    raise TimelineInvariantError(f"{timeline.timeline_id}, account {account}: {name} is {value}, expected {wanted}")


def replay_timeline(timeline: Timeline) -> TimelineReplay:
    """Replay one timeline in a fresh state and check its expected figures.

    :return:
        The replay, with its state, for further inspection.
    :raise TimelineInvariantError:
        If an invariant breaks or a figure differs from the expectation.
    """
    replay = TimelineReplay(timeline.accounts)
    replay.replay(timeline)
    replay.check_expectations(timeline)
    return replay
