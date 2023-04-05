"""Trade executor state.

The whole application date can be dumped and loaded as JSON.

Any datetime must be naive, without timezone, and is assumed to be UTC.
"""
from dataclasses import dataclass, field
import datetime
from decimal import Decimal
from typing import List, Callable, Tuple, Set, Optional

import pandas as pd
from dataclasses_json import dataclass_json

from .sync import Sync
from .identifier import AssetIdentifier, TradingPairIdentifier
from .portfolio import Portfolio
from .position import TradingPosition
from .reserve import ReservePosition
from .statistics import Statistics
from .trade import TradeExecution, TradeStatus, TradeType
from .types import USDollarAmount, BPS, USDollarPrice
from .uptime import Uptime
from .visualisation import Visualisation

from tradeexecutor.strategy.trade_pricing import TradePricing


class UncleanState(Exception):
    """State containst trades that need manual intervention."""


@dataclass_json
@dataclass
class State:
    """The current state of the trading strategy execution.

    It tells the current and past state of a single trading strategy execution:
    positions, their trades and related valuations, metrics and such data.

    This class is the root object of the serialisable state tree
    for a trading strategy.

    - Can be serialised as :term:`JSON`

    - Contains one :py:class:`Portfolio` object that contains
      all positions, trades and underlying blockchain transactions

    - Contains one :py:class:`Visualisation` object
      that contains run-time calculated and stored visualisation  about the portfolio

    - Contains one :py:class:`Statistics` object
      that contains run-time calculated and stored metrics about the portfolio

    Uses of this class include

    - Backtest fills in the state when simulating the trades

    - The live execution environment keeps its internal state
      on a disk as a serialised :py:class:`State` object

    - Analysis and performance metrics read the state

    - The web frontend reads the state

    """

    #: When this state was created
    #:
    #: Same as when the strategy was launched
    created_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)

    #: When this state was saved
    #:
    #: UTC timestamp.
    #: Set by by :py:meth:`tradeexecutor.state.store.StateStore.sync`
    last_updated_at: Optional[datetime.datetime] = None

    #: The next cycle.
    #:
    #: How many strategy thinking and execution
    #: cycles we have completed successfully.
    #:
    #: Starts with 1 (no cycles completed)
    #:
    cycle: int = 1

    #: The name of this strategy.
    #: Can be unset.
    #: Set when the state is created.
    name: Optional[str] = None

    #: Portfolio of this strategy.
    #: Currently only one portfolio per strategy.
    portfolio: Portfolio = field(default_factory=Portfolio)

    #: Portfolio and position performance records over time.
    stats: Statistics = field(default_factory=Statistics)

    #: Assets that the strategy is not allowed to touch,
    #: or have failed to trade in the past, resulting to a frozen position.
    #: Besides this internal black list, the executor can have other blacklists
    #: based on the trading universe and these are not part of the state.
    #: The main motivation of this list is to avoid assets that caused a freeze in the future.
    #: Key is Ethereum address, lowercased.
    asset_blacklist: Set[str] = field(default_factory=set)

    #: Strategy visualisation and debug messages
    #: to show how the strategy is thinking.
    visualisation: Visualisation = field(default_factory=Visualisation)

    #: Trade execution uptime and success statistcis]
    #:
    #: Contains statistics about trade execution having to manage
    #: to run its internal functions.
    uptime: Uptime = field(default_factory=Uptime)

    sync: Sync = field(default_factory=Sync)

    def __repr__(self):
        return f"<State for {self.name}>"

    def is_empty(self) -> bool:
        """This state has no open or past trades or reserves."""
        return self.portfolio.is_empty()

    def is_good_pair(self, pair: TradingPairIdentifier) -> bool:
        """Check if the trading pair is blacklisted."""
        assert isinstance(pair, TradingPairIdentifier), f"Expected TradingPairIdentifier, got {type(pair)}: {pair}"
        return (pair.base.get_identifier() not in self.asset_blacklist) and (pair.quote.get_identifier() not in self.asset_blacklist)

    def create_trade(self,
                     strategy_cycle_at: datetime.datetime,
                     pair: TradingPairIdentifier,
                     quantity: Optional[Decimal],
                     reserve: Optional[Decimal],
                     assumed_price: USDollarPrice,
                     trade_type: TradeType,
                     reserve_currency: AssetIdentifier,
                     reserve_currency_price: USDollarPrice,
                     notes: Optional[str] = None,
                     pair_fee: Optional[float] = None,
                     lp_fees_estimated: Optional[USDollarAmount] = None,
                     planned_mid_price: Optional[USDollarPrice] = None,
                     price_structure: Optional[TradePricing] = None,
                     position: Optional[TradingPosition] = None,
                     slippage_tolerance: Optional[float] = None,
                     ) -> Tuple[TradingPosition, TradeExecution, bool]:
        """Creates a request for a new trade.

        If there is no open position, marks a position open.

        Trade can be opened by knowing how much you want to buy (quantity) or how much cash you have to buy (reserve).

        - To open a spot buy, fill in `reseve` amount you wish to use for the buy

        - To open a spot sell, fill in `quoantity` amount you wish to use for the buy,
          as a negative number
          
        :param strategy_cycle_at:
            The strategy cycle timestamp for which this trade was executed.

        :param trade_id:
            Trade id allocated by the portfolio

        :param quantity:
            How many units this trade does.

            Positive for buys, negative for sells in the spot market.

        :param assumed_price:
            The planned execution price.

            This is the price we expect to pay per `quantity` unit after the execution.
            This is the mid price + any LP fees included.

        :param trade_type:
            What kind of a trade is this.

        :param reserve_currency:
            Which portfolio reserve we use for this trade.

         :param reserve_currency_price:
            If the quote token is not USD, then the exchange rate between USD and quote token we assume we have.

            Actual exchange rate may depend on the execution.

        :param notes:
            Any human-readable remarks we want to tell about this trade.

        :param pair_fee:
            The fee tier from the trading pair / overriden fee.

        :param lp_fees_estimated:
            HOw much we estimate to pay in LP fees (dollar)

        :param planned_mid_price:
            What was the mid-price of the trading pair when we started to plan this trade.

        :param reserve:
            How many reserve units this trade produces/consumes.

            I.e. dollar amount for buys/sells.

        :param price_structure:
            The full planned price structure for this trade.

            The state of the market at the time of planning the trade,
            and what fees we assumed we are going to get.

        :param position:
            Override the position for the trade.

            - Use for close trades (you need to explicitly tell which position to close
              as there might be two positions with the same pair)

            - Use for repair trades.

        :param notes:
            Human-readable string to show on the trade.

        :param slippage_tolerance:
            Slippage tolerance for this trade.

            See :py:attr:`tradeexecutor.state.trade.TradeExecution.slippage_tolerance` for details.

        :return:
            Tuple of entries

            - Trade position (old/new)

            - New trade

            - True if a a new position was opened

        """

        assert isinstance(strategy_cycle_at, datetime.datetime)
        assert not isinstance(strategy_cycle_at, pd.Timestamp)
        
        if pair_fee:
            assert type(pair_fee) == float
        
        if price_structure is not None:
            assert isinstance(price_structure, TradePricing)

        if quantity is not None:
            assert reserve is None, "Quantity and reserve both cannot be given at the same time"

        position, trade, created = self.portfolio.create_trade(
            strategy_cycle_at,
            pair,
            quantity,
            reserve,
            assumed_price,
            trade_type,
            reserve_currency,
            reserve_currency_price,
            pair_fee=pair_fee,
            lp_fees_estimated=lp_fees_estimated,
            planned_mid_price=planned_mid_price,
            price_structure=price_structure,
            position=position,
            slippage_tolerance=slippage_tolerance,
            notes=notes,
        )

        return position, trade, created

    def start_execution(self, ts: datetime.datetime, trade: TradeExecution, txid: str, nonce: int):
        """Update our balances and mark the trade execution as started.

        Called before a transaction is broadcasted.
        """

        assert trade.get_status() == TradeStatus.planned

        position = self.portfolio.find_position_for_trade(trade)
        assert position, f"Trade does not belong to an open position {trade}"

        self.portfolio.check_for_nonce_reuse(nonce)

        if trade.is_buy():
            self.portfolio.move_capital_from_reserves_to_trade(trade)

        trade.started_at = ts

        # TODO: Legacy attributes that need to go away
        trade.txid = txid
        trade.nonce = nonce

    def mark_broadcasted(self, broadcasted_at: datetime.datetime, trade: TradeExecution):
        """"""
        assert trade.get_status() == TradeStatus.started
        trade.broadcasted_at = broadcasted_at

    def mark_trade_success(self,
                           executed_at: datetime.datetime,
                           trade: TradeExecution,
                           executed_price: USDollarPrice,
                           executed_amount: Decimal,
                           executed_reserve: Decimal,
                           lp_fees: USDollarAmount,
                           native_token_price: USDollarPrice):
        """"""

        position = self.portfolio.find_position_for_trade(trade)

        if trade.is_buy():
            assert executed_amount and executed_amount > 0, f"Executed amount was {executed_amount}"
        else:
            assert executed_reserve > 0, f"Executed reserve must be positive for sell, got amount:{executed_amount}, reserve:{executed_reserve}"
            assert executed_amount < 0, f"Executed amount must be negative for sell, got amount:{executed_amount}, reserve:{executed_reserve}"

        trade.mark_success(executed_at, executed_price, executed_amount, executed_reserve, lp_fees, native_token_price)

        if trade.is_sell():
            self.portfolio.return_capital_to_reserves(trade)

        if position.can_be_closed():
            # Move position to closed
            position.closed_at = executed_at
            del self.portfolio.open_positions[position.position_id]
            self.portfolio.closed_positions[position.position_id] = position

    def mark_trade_failed(self, failed_at: datetime.datetime, trade: TradeExecution):
        """Unroll the allocated capital."""
        trade.mark_failed(failed_at)
        # Return unused reserves back to accounting
        if trade.is_buy():
            self.portfolio.adjust_reserves(trade.reserve_currency, trade.reserve_currency_allocated)

    def update_reserves(self, new_reserves: List[ReservePosition]):
        self.portfolio.update_reserves(new_reserves)

    def revalue_positions(self, ts: datetime.datetime, valuation_method: Callable):
        """Revalue all open positions in the portfolio.

        Reserves are not revalued.
        """
        self.portfolio.revalue_positions(ts, valuation_method)

    def blacklist_asset(self, asset: AssetIdentifier):
        """Add a asset to the blacklist."""
        address = asset.get_identifier()
        self.asset_blacklist.add(address)

    def perform_integrity_check(self):
        """Check that we are not reusing any trade or position ids and counters are correct.

        :raise: Assertion error in the case internal data structures are damaged
        """

        position_ids = set()
        trade_ids = set()

        for p in self.portfolio.get_all_positions():
            assert p.position_id not in position_ids, f"Position id reuse {p.position_id}"
            position_ids.add(p.position_id)
            for t in p.trades.values():
                assert t.trade_id not in trade_ids, f"Trade id reuse {p.trade_id}"
                trade_ids.add(t.trade_id)

        max_position_id = max(position_ids) if position_ids else 0
        assert max_position_id + 1 == self.portfolio.next_position_id, f"Position id tracking lost. Max {max_position_id}, next {self.portfolio.next_position_id}"

        max_trade_id = max(trade_ids) if trade_ids else 0
        assert max_trade_id + 1 == self.portfolio.next_trade_id, f"Trade id tracking lost. Max {max_trade_id}, next {self.portfolio.next_trade_id}"

        # Check that all stats have a matching position
        for pos_stat_id in self.stats.positions.keys():
            assert pos_stat_id in position_ids, f"Stats had position id {pos_stat_id} for which actual trades are missing"

    def start_trades(self, ts: datetime.datetime, trades: List[TradeExecution], max_slippage: float=0.01, underflow_check=False):
        """Mark trades ready to go.

        Update any internal accounting of capital allocation from reseves to trades.

        Sets the execution model specific parameters like `max_slippage` on the trades.

        :param max_slippage:
            The slippage allowed for this trade before it fails in execution.
            0.01 is 1%.

        :param underflow_check:
            If true warn us if we do not have enough reserves to perform the trades.
            This does not consider new reserves released from the closed positions
            in this cycle.
        """

        for t in trades:
            if t.is_buy():
                self.portfolio.move_capital_from_reserves_to_trade(t, underflow_check=underflow_check)

            t.started_at = ts
            t.planned_max_slippage = max_slippage

    def check_if_clean(self):
        """Check that the state data is intact.

        Check for the issues that could be caused e.g. trade-executor unclean shutdown
        or a blockchain node crash.

        One of a typical issue would be

        - A trade that failed to execute

        - A trade that was broadcasted, but we did not get a confirmation back in time,
          causing the trade executor to crash

        Call this when you restart a trade execution to ensure
        the old state is intact. For any unfinished trades,
        run a repair command or manually repair the database.

        :raise UncleanState:
            In the case we detect unclean stuff
        """

        for p in self.portfolio.open_positions.values():
            t: TradeExecution
            for t in p.trades.values():
                if t.is_unfinished():
                    raise UncleanState(f"Position {p}, trade {t} is unfinished")


