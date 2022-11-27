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

from .identifier import AssetIdentifier, TradingPairIdentifier
from .portfolio import Portfolio
from .position import TradingPosition
from .reserve import ReservePosition
from .statistics import Statistics
from .trade import TradeExecution, TradeStatus, TradeType
from .types import USDollarAmount
from .visualisation import Visualisation


@dataclass_json
@dataclass
class State:
    """The current state of the trading strategy execution."""

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
                     ts: datetime.datetime,
                     pair: TradingPairIdentifier,
                     quantity: Optional[Decimal],
                     reserve: Optional[Decimal],
                     assumed_price: USDollarAmount,
                     trade_type: TradeType,
                     reserve_currency: AssetIdentifier,
                     reserve_currency_price: USDollarAmount,
                     notes: Optional[str] = None,
                     ) -> Tuple[TradingPosition, TradeExecution, bool]:
        """Creates a request for a new trade.

        If there is no open position, marks a position open.

        Trade can be opened by knowing how much you want to buy (quantity) or how much cash you have to buy (reserve).

        - To open a spot buy, fill in `reseve` amount you wish to use for the buy

        - To open a spot sell, fill in `quoantity` amount you wish to use for the buy,
          as a negative number

        :return: Tuple position, trade, was a new position created
        """

        assert isinstance(ts, datetime.datetime)
        assert not isinstance(ts, pd.Timestamp)

        if quantity is not None:
            assert reserve is None, "Quantity and reserve both cannot be given at the same time"

        position, trade, created = self.portfolio.create_trade(
            ts,
            pair,
            quantity,
            reserve,
            assumed_price,
            trade_type,
            reserve_currency,
            reserve_currency_price)
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

        trade.txid = txid
        trade.nonce = nonce

    def mark_broadcasted(self, broadcasted_at: datetime.datetime, trade: TradeExecution):
        """"""
        assert trade.get_status() == TradeStatus.started
        trade.broadcasted_at = broadcasted_at

    def mark_trade_success(self, executed_at: datetime.datetime, trade: TradeExecution, executed_price: USDollarAmount, executed_amount: Decimal, executed_reserve: Decimal, lp_fees: USDollarAmount, native_token_price: USDollarAmount):
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
