"""Trade executor state.

The whole application date can be dumped and loaded as JSON.

Any datetime must be naive, without timezone, and is assumed to be UTC.
"""
import json
from dataclasses import dataclass, field
import datetime
import logging
from decimal import Decimal
from pathlib import Path
from typing import List, Callable, Tuple, Set, Optional

import pandas as pd
from dataclasses_json import dataclass_json
from dataclasses_json.core import _ExtendedEncoder
from qstrader.asset.asset import Asset

from .other_data import OtherData
from .sync import Sync
from .identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind
from .portfolio import Portfolio
from .position import TradingPosition
from .reserve import ReservePosition
from .statistics import Statistics
from .trade import TradeExecution, TradeStatus, TradeType, TradeFlag
from .types import USDollarAmount, BPS, USDollarPrice
from .uptime import Uptime
from .visualisation import Visualisation

from tradeexecutor.utils.summarydataframe import as_duration, format_value
from tradeexecutor.strategy.trade_pricing import TradePricing
from ..strategy.cycle import CycleDuration
from ..utils.accuracy import ZERO_DECIMAL
from tradeexecutor.strategy.lending_protocol_leverage import (
    create_short_loan,
    update_short_loan,
    create_credit_supply_loan,
    update_credit_supply_loan,
)

logger = logging.getLogger(__name__)


class UncleanState(Exception):
    """State containst trades that need manual intervention."""


@dataclass_json
@dataclass(slots=True)
class BacktestData:
    """Miscellaneous data needed to store only for the backtest state."""

    #: The start of backtest period
    start_at: datetime. datetime

    #: The end of backtest period
    end_at: datetime.datetime

    #: What has the decision cycle duration
    #:
    decision_cycle_duration: CycleDuration

    #: When the strategy was ready to make its first trade decision.
    #:
    #: This timestamp marks there was enough trade history to correctly complete decide_trades().
    #: Must be manually set with :py:attr:`mark_ready` in decide trades..
    #: If available then benchmark curves and strategy equity curves can be correctly aligned,
    #: otherwise there might be misaligment (strategy sits on cash until enough history is available).
    #:
    ready_at: Optional[datetime.datetime] = None

    def mark_ready(self, timestamp: datetime.datetime | pd.Timestamp):
        """Mark that the strategy has enough data to decide its first trade.

        - See :py:attr:`ready_at` for more information

        - Can be called multiple times, only the first time counts

        - Interest positions may be ignored for mark_ready()
          (decide_trades() can do interest positions before ready state has been reached)
        """

        if self.ready_at is not None:
            # Already set
            return

        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.to_pydatetime()

        self.ready_at = timestamp


@dataclass_json
@dataclass(slots=True)
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

    #: Legacy: Do not use.
    #:
    #: See :py:meth:`blacklist_asset`
    #:
    asset_blacklist: Set[str] = field(default_factory=set)

    #: Maintain set of blacklisted asset identifiers
    blacklisted_assets: Set[AssetIdentifier] = field(default_factory=set)

    #: Strategy visualisation and debug messages
    #: to show how the strategy is thinking.
    visualisation: Visualisation = field(default_factory=Visualisation)

    #: Trade execution uptime and success statistcis]
    #:
    #: Contains statistics about trade execution having to manage
    #: to run its internal functions.
    uptime: Uptime = field(default_factory=Uptime)

    sync: Sync = field(default_factory=Sync)

    #: Backtest data related to this backtest result
    #:
    #: Data that is relevant only for backtest results,
    #: not live trading.
    backtest_data: BacktestData | None = None

    #: Misc. backtesting variables settable by users
    other_data: Optional[OtherData] = field(default_factory=OtherData)

    def __repr__(self):
        return f"<State for {self.name}>"

    def is_empty(self) -> bool:
        """This state has no open or past trades or reserves."""
        return self.portfolio.is_empty()

    def is_good_pair(self, pair: TradingPairIdentifier) -> bool:
        """Check if the trading pair is blacklisted."""
        assert isinstance(pair, TradingPairIdentifier), f"Expected TradingPairIdentifier, got {type(pair)}: {pair}"

        if pair.base.address in self.asset_blacklist:
            # Legacy state ecompatiblity
            return True

        return (pair.base not in self.blacklisted_assets) and (pair.quote not in self.blacklisted_assets)

    def mark_ready(self, timestamp: datetime.datetime | pd.Timestamp):
        """Mark that the strategy has enough (backtest) data to decide the first trade.

        This marks the difference between the backtesting data availability period,
        and actual tradeable period when all indicators have enough data.

        - See :py:attr:`BacktestData.ready_at` for more information

        - Can be called multiple times, only the first time counts

        - See :py:meth:`get_trading_time_range` for reading

        - If you do not call this method in `decide_trades()`, nothing bad happens,
          but backtest benchmark indices (buy and hold BTC, etc.) might have biases results
          to a direction or another
        """
        if self.backtest_data:
            self.backtest_data.mark_ready(timestamp)

    def get_strategy_time_range(self) -> Tuple[datetime.datetime, datetime.datetime]:
        """Get the time range for which the strategy should have data.

        - If this is a backtest, return backtesting range

        - If this is a live execution, return created - last updated

        - See also :py:meth:`get_trading_time_range`
        """
        if self.backtest_data and self.backtest_data.start_at:
            return self.backtest_data.start_at, self.backtest_data.end_at
        else:
            return self.created_at, self.last_updated_at

    def get_trading_time_range(self) -> Tuple[datetime.datetime, datetime.datetime]:
        """Get the time range when the strategy could have made trades.

        - If live trading same as :py:meth:`get_strategy_time_range`

        - See :py:meth:`mark_ready` for setting

        - See :py:func:`get_strategy_time_range` to get the full data availability range of a backtest
        """
        start_at, end_at = self.get_strategy_time_range()

        # We can do this backtest period normalisation only if mark_ready() has been called
        if self.backtest_data and self.backtest_data.ready_at:
            return self.backtest_data.ready_at, end_at

        return start_at, end_at

    def get_strategy_duration(self) -> datetime.timedelta | None:
        """Get the age of the strategy execution. If backtest, return backtest range, if live, return created - last updated
        
        See :py:meth:`get_strategy_time_range` for details.
        
        :returns: Age of the strategy execution, or None if the age cannot be calculated.
        """
        strategy_start, strategy_end  = self.get_strategy_time_range()
        if strategy_start and strategy_end:
            return strategy_end - strategy_start
        return None

    def get_formatted_strategy_duration(self) -> str:
        """Get the age of the strategy execution in human-readable format.
        
        See :py:meth:`get_strategy_duration` for details.
        
        :returns: Age of the strategy execution in human-readable format.
        """
        age = self.get_strategy_duration()
        return "Unknown" if age is None else format_value(as_duration(age))

    def create_trade(
        self,
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
        leverage: Optional[float] = None,
        closing: Optional[bool] = False,
        planned_collateral_consumption: Optional[Decimal] = None,
        planned_collateral_allocation: Optional[Decimal] = None,
        flags: Optional[Set[TradeFlag]] = None,
        pending=False,
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

            For short positions, negative quantity means increase the position of this much,
            positive quantity means decrease the position.

            Any fees have been already reduced away from this quantity,
            as :py:class:`PriceModel` gives the planned price that includes
            fees.

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

        :param closing:
            This trade should close the position entirely.

            A flag used with leveraged positions.

        :param pending:
            Do not generate a new open position.

            Used when adding take profit triggers to market limit position.

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
            # We assume we give either reserve (how much cash we spent) or
            # quantity (how much token we spent).
            # However for leveraged position we give both, because quantity/reserve
            # gives the final loan health rate.
            if not pair.kind.is_credit_based():
                assert reserve is None, "Quantity and reserve both cannot be given at the same time for a spot market pair"

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
            leverage=leverage,
            closing=closing,
            planned_collateral_consumption=planned_collateral_consumption,
            planned_collateral_allocation=planned_collateral_allocation,
            flags=flags,
            pending=pending,
        )

        return position, trade, created

    def trade_short(
            self,
            strategy_cycle_at: datetime.datetime,
            pair: TradingPairIdentifier,
            borrowed_asset_price: USDollarPrice,
            trade_type: TradeType,
            reserve_currency: AssetIdentifier,
            collateral_asset_price: USDollarPrice,
            borrowed_quantity: Optional[Decimal] = None,
            collateral_quantity: Optional[Decimal] = None,
            notes: Optional[str] = None,
            pair_fee: Optional[float] = None,
            lp_fees_estimated: Optional[USDollarAmount] = None,
            planned_mid_price: Optional[USDollarPrice] = None,
            price_structure: Optional[TradePricing] = None,
            position: Optional[TradingPosition] = None,
            slippage_tolerance: Optional[float] = None,
            closing: Optional[bool] = False,
            planned_collateral_consumption: Optional[Decimal] = None,
            planned_collateral_allocation: Optional[Decimal] = None,
            flags: Optional[Set[TradeFlag]] = None,
        ) -> Tuple[TradingPosition, TradeExecution, bool]:
        """Creates a trade for a short position.

        - Opens, increases or decreases short position size.

        For argument and return value documentation see :py:meth:`create_trade`.

        See also :py:meth:`supply_credit`.

        :param borrowed_quantity:
            How much we are going to borrow and increase/decrease our exposure.

            Our short position size in the target token.

            - Negative for increasing short position size

            - Positive for reducing the short position size

            See ``test_short_increase_size`` and ``test_short_decrease_size`` for an example.

        :param collateral_quantity:
            How much reserve currency we are going to use as a collateral for loans.

            This is moved from cash reserves to lending protocol deposit.

            - Always positive when opening

            - Can be zero and in this case,
              the shorted token is bought or sold and this
              will affect the underlying loan health factor

            For releasing collateral see ``planned_collateral_allocation`` argument.
            See ``test_short_decrease_size`` for an example.

        :param borrowed_asset_price:
            What is the assumed price of the token we are going to borrow.

            We estimate fees and value selling it short.

        :param closing:
            This trade should close any remaining exposure and return the collateral after the trade.

            If set, norrowed quantity and collateral quantity
            are automatically calculated.

        :param planned_collateral_consumption:
            See :py:attr:`tradeexecutor.state.trade.TradeExecution.planned_collateral_consumption`.


        :param planned_collateral_allocation:
            See :py:attr:`tradeexecutor.state.trade.TradeExecution.planned_collateral_allocation`.

        :return:
            Trading position, trade execution and created flag.

            :py:attr:`TradeExecution.planned_loan` is set.

            After the trade succeeds, :py:attr:`TradingPosition.loan`
            is set.

            If the trade does not succeed loan data remains unchanged.
        """
        assert pair.kind.is_shorting()
        assert pair.quote.underlying.is_stablecoin(), "Shorts accept only stablecoin collateral"

        assert pair.quote.underlying == self.portfolio.get_default_reserve_position().asset, f"Collateral is not our reserve"

        assert reserve_currency == pair.quote.underlying

        if not closing:
            assert borrowed_quantity is not None, "borrowed_quantity must be always set"
            assert collateral_quantity is not None, "collateral_quantity must be always set. Set to zero if you do not want to have change to the amount of collateral"

        return self.create_trade(
            strategy_cycle_at=strategy_cycle_at,
            pair=pair,
            quantity=borrowed_quantity,
            reserve=collateral_quantity,
            assumed_price=borrowed_asset_price,
            trade_type=trade_type,
            reserve_currency=reserve_currency,
            reserve_currency_price=collateral_asset_price,
            notes=notes,
            pair_fee=pair_fee,
            lp_fees_estimated=lp_fees_estimated,
            planned_mid_price=planned_mid_price,
            price_structure=price_structure,
            position=position,
            slippage_tolerance=slippage_tolerance,
            closing=closing,
            planned_collateral_consumption=planned_collateral_consumption,
            planned_collateral_allocation=planned_collateral_allocation,
            flags=flags,
        )

    def supply_credit(
        self,
        strategy_cycle_at: datetime.datetime,
        pair: TradingPairIdentifier,
        trade_type: TradeType,
        reserve_currency: AssetIdentifier,
        collateral_asset_price: USDollarPrice,
        collateral_quantity: Optional[Decimal] = None,
        notes: Optional[str] = None,
        pair_fee: Optional[float] = None,
        lp_fees_estimated: Optional[USDollarAmount] = None,
        planned_mid_price: Optional[USDollarPrice] = None,
        price_structure: Optional[TradePricing] = None,
        position: Optional[TradingPosition] = None,
        slippage_tolerance: Optional[float] = None,
        closing: Optional[bool] = False,
        flags: Optional[Set[TradeFlag]] = None,
    ) -> Tuple[TradingPosition, TradeExecution, bool]:
        """Create or adjust credit supply position.

        Credit supply position trades are modelled as following

        - You BUY aToken using the reserve. Like buying aUSDC
          with USDC.

        - You SELL aToken and get back reserve + interest,
          with the trade size reserve + interest

        - Reserve is USDC, always positive

        - The modelling is different from trade_short/trade_long

        - See also :py:meth:`trade_short`

        :param collateral_quantity:
            Positive for supplying credit, negative of recalling reserves.
        """

        assert pair.kind == TradingPairKind.credit_supply

        planned_collateral_allocation = None
        if collateral_quantity < 0:
            # Moving collateral back to reserves
            reserve = abs(collateral_quantity)
        else:
            reserve = collateral_quantity

        quantity = collateral_quantity

        # For credit supply, slippage tolerance should be 0
        # as all USDC is converted to aPolUSDC in the supply operation
        # and None is lost for the slippage
        if slippage_tolerance is None:
            slippage_tolerance = 0.0

        position, trade, created = self.create_trade(
            strategy_cycle_at=strategy_cycle_at,
            pair=pair,
            quantity=quantity,
            assumed_price=1.0,
            reserve=reserve,
            trade_type=trade_type,
            reserve_currency=reserve_currency,
            reserve_currency_price=collateral_asset_price,
            notes=notes,
            pair_fee=pair_fee,
            lp_fees_estimated=lp_fees_estimated,
            planned_mid_price=planned_mid_price,
            price_structure=price_structure,
            position=position,
            slippage_tolerance=slippage_tolerance,
            closing=closing,
            planned_collateral_allocation=planned_collateral_allocation,
            flags=flags,
        )
        return position, trade, created

    def start_execution(
        self,
        ts: datetime.datetime,
        trade: TradeExecution,
        txid: str | None = None,
        nonce: int | None = None,
        underflow_check=False,
        triggered=False,
        ):
        """Update our balances and mark the trade execution as started.

        - Called before a transaction is broadcasted.

        - Updates internal accounting and moves capital from the reserve account locked on a trade

        See also :py:meth:`start_execution_all`.

        :param trade:
            Trade to

        :param underflow_check:
            Raise exception if we have not enough cash to allocate.

            The check disabled by default. Might be legit for portfolio strats that need to sell old assets before buying new assets.

        :param txid:
            Legacy. Do not use.

        :param nonce:
            Legacy. Do not use.

        :param triggered:
            True if this execution is from stop loss trigger checks, otherwise from decision trades cycle.

        """

        assert trade.get_status() == TradeStatus.planned, f"start_execution(): received a trade with status {trade.get_status()}: {trade}"

        if not triggered:
            assert TradeFlag.triggered not in trade.flags, f"Got a trigger trade for execution: {trade}, {trade.flags}.\n" \
                                                           f"This is not needed: The trade will be automatically executed when the trigger hits.\n" \
                                                           f"You do not need to return triggered trades from decide_trades"

            position = self.portfolio.find_position_for_trade(trade)
        else:
            # Consider market limit opens
            position = self.portfolio.find_position_for_trade(trade, pending=True)

        assert position, f"Trade does not belong to an open position {trade}"

        # Legacy check
        if nonce is not None:
            self.portfolio.check_for_nonce_reuse(nonce)

        # Allocate reserve capital for this trade.
        # Reserve capital cannot be double spent until the trades are execured.
        if trade.is_spot():
            if trade.is_buy():
                # Spot trade reserves can go to negative before execution,
                # because reservs will be there after we have executed some sell trades first
                self.portfolio.move_capital_from_reserves_to_spot_trade(trade, underflow_check=underflow_check)
        elif trade.is_leverage():
            self.portfolio.move_capital_from_reserves_to_spot_trade(trade, underflow_check=underflow_check)
        elif trade.is_credit_supply():
            if trade.is_buy():
                self.portfolio.move_capital_from_reserves_to_spot_trade(trade, underflow_check=underflow_check)
        else:
            raise NotImplementedError()

        trade.started_at = ts

        logger.info("Trade %s started at %s", trade.get_short_label(), ts)

        # TODO: Legacy attributes that need to go away
        if txid is not None:
            trade.txid = txid

        if nonce is not None:
            trade.nonce = nonce

    def mark_broadcasted(self, broadcasted_at: datetime.datetime, trade: TradeExecution):
        """"""
        assert trade.get_status() == TradeStatus.started
        trade.broadcasted_at = broadcasted_at

    def mark_trade_success(
        self,
        executed_at: datetime.datetime,
        trade: TradeExecution,
        executed_price: USDollarPrice,
        executed_amount: Decimal,
        executed_reserve: Decimal,
        lp_fees: USDollarAmount,
        native_token_price: USDollarPrice,
        cost_of_gas: float | None = None ,
        executed_collateral_consumption: Optional[Decimal] = None,
        executed_collateral_allocation: Optional[Decimal] = None,
        force: bool = False,
    ):
        """After trade has been successfully executed, update the state of our internal ledged to reflect this.

        - Trade is marked as successfully complete

        - All position sizes are updated to match executed values (instead of planned values)

        - Mark any LP and gas fees from the trade

        - If this was the final trade of the position, mark the position closed
        """

        position = self.portfolio.find_position_for_trade(trade, pending=True)

        if trade.is_spot():
            if trade.is_buy():
                assert executed_amount and executed_amount > 0, f"Executed amount was {executed_amount}"
            else:
                assert executed_reserve > 0, f"Executed reserve must be positive for sell, got amount:{executed_amount}, reserve:{executed_reserve}"
                assert executed_amount < 0, f"Executed amount must be negative for sell, got amount:{executed_amount}, reserve:{executed_reserve}"

        trade.mark_success(
            executed_at,
            executed_price,
            executed_amount,
            executed_reserve,
            lp_fees,
            native_token_price,
            cost_of_gas=cost_of_gas,
            executed_collateral_consumption=executed_collateral_consumption,
            executed_collateral_allocation=executed_collateral_allocation,
            force=force,
        )

        # The loan status of the position is reflected back to be
        # whatever is on chain after the execution
        if trade.planned_loan_update:
            if not trade.executed_loan_update:
                if position.is_short():
                    if not position.loan:
                        trade.executed_loan_update = create_short_loan(
                            position,
                            trade,
                            executed_at,
                            mode="execute",
                        )
                    else:
                        trade.executed_loan_update = update_short_loan(
                            position.loan.clone(),
                            position,
                            trade,
                            mode="execute",
                            close_position=TradeFlag.close in trade.flags,
                        )
                elif position.is_credit_supply():
                    if not position.loan:
                        trade.executed_loan_update = create_credit_supply_loan(
                            position,
                            trade,
                            executed_at,
                            mode="execute",
                        )
                    else:
                        trade.executed_loan_update = update_credit_supply_loan(
                            loan=position.loan.clone(),
                            position=position,
                            trade=trade,
                            timestamp=executed_at,
                            mode="execute",
                        )

            position.loan = trade.executed_loan_update

        if trade.is_spot() and trade.is_sell():
            self.portfolio.return_capital_to_reserves(trade)
        elif trade.is_leverage():

            # Release any collateral and move it back to the wallet
            if executed_collateral_allocation:
                assert trade.pair.quote.underlying
                self.portfolio.adjust_reserves(
                    trade.pair.quote.underlying,
                    -executed_collateral_allocation,
                    reason=f"Collateral allocation for leveraged position #{position.position_id}, trade #{trade.trade_id}"
                )

        elif trade.is_credit_supply():
            if trade.is_sell():
                self.portfolio.adjust_reserves(
                    trade.pair.quote,
                    executed_reserve,
                    reason=f"Returned cash from the credit position #{position.position_id}"
                )

        if trade.is_long():
            raise NotImplementedError()

        if position.is_pending():
            # Position has now executed trades.
            # It can be no longer pending.
            logger.info("Position moving from pending -> open: %s", position)
            self.portfolio.open_positions[position.position_id] = position
            del self.portfolio.pending_positions[position.position_id]
            position.pending_since_at = None

        if position.can_be_closed():

            logger.info("Marking position to closed: %s", position)
            self.portfolio.close_position(position, executed_at)

            if position.loan:

                trade.claimed_interest = position.loan.claim_interest()

                # Mark that the trade claimed any interest
                # that was available on the collateral
                if trade.is_leverage():
                    pass

                    # Claimed interest is already include in the collateral release
                    # self.portfolio.adjust_reserves(
                    #    trade.pair.quote.get_pricing_asset(),
                    #    trade.claimed_interest,
                    #    reason=f"Claimed interest on position #{position.position_id}"
                    #)

                # Mark that the trade paid any remaining interest
                # on the debt
                if position.loan.borrowed:
                    # TODO: Add planned interest payments
                    trade.paid_interest = position.loan.repay_interest()

        else:
            logger.info(
                "Position #%d still open after a trade: %s, quantity: %s, quantity w/planning: %s",
                position.position_id,
                trade.get_short_label(),
                position.get_quantity(),
                position.get_quantity(planned=True),
            )

    def mark_trade_failed(self, failed_at: datetime.datetime, trade: TradeExecution):
        """Unroll the allocated capital."""
        trade.mark_failed(failed_at)
        # Return unused reserves back to accounting
        if trade.is_buy():
            self.portfolio.adjust_reserves(
                trade.reserve_currency,
                trade.reserve_currency_allocated,
                f"Trade failed, allocated reserve was not used:\n{trade}"
            )

    def update_reserves(self, new_reserves: List[ReservePosition]):
        self.portfolio.update_reserves(new_reserves)

    def revalue_positions(self, ts: datetime.datetime, valuation_method: Callable):
        """Revalue all open positions in the portfolio.

        Reserves are not revalued.
        """
        raise RuntimeError(f"Removed. Use valuation.revalue_state()")

    def blacklist_asset(self, asset: AssetIdentifier):
        """Add a asset to the blacklist.

        See :py:meth:`is_good_pair`.
        """
        logger.info("Blacklisted: %s", asset)
        self.blacklisted_assets.add(asset)
        self.asset_blacklist.add(asset.address)  # Legacy compatibility

    def perform_integrity_check(self):
        """Check that we are not reusing any trade or position ids and counters are correct.

        :raise: Assertion error in the case internal data structures are damaged
        """

        position_ids = set()
        trade_ids = set()

        for p in self.portfolio.get_all_positions(pending=True):
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

    def start_execution_all(
        self,
        ts: datetime.datetime,
        trades: List[TradeExecution],
        max_slippage: float=None,
        underflow_check=False,
        rebroadcast=False,
        triggered=False,
    ):
        """Mark a bunch of trades ready to go.

        Update any internal accounting of capital allocation from reseves to trades.

        Sets the execution model specific parameters like `max_slippage` on the trades.

        See also :py:meth:`start_execution`

        :param ts:
            Strategy cycle timestamp

        :param trades:
            List of trades to prepare

        :param max_slippage:

            Legacy. Do not use.

            The slippage allowed for this trade before it fails in execution.
            0.01 is 1%.

        :param underflow_check:

            Legacy. Do not use.

            If true warn us if we do not have enough reserves to perform the trades.
            This does not consider new reserves released from the closed positions
            in this cycle.
        """

        for t in trades:

            self.start_execution(ts, t, triggered=triggered)

            if max_slippage is not None:
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
                    tx_hashes = ", ".join([str(tx.tx_hash) for tx in t.blockchain_transactions])
                    raise UncleanState(f"Position {p}, trade {t} is unfinished\nTransactions are: {tx_hashes}")

    def to_json_safe(self) -> str:
        """Serialise to JSON format with helpful validation and error messages.

        Extra validation adds performance overhead.

        :return:
            The full strategy execution state as JSON string.

            The strategy can be saved on a disk, resumed,
            or server to the web frontend using this JSON blob.
        """

        # TODO: Avoid circular imports, refactor modules
        from tradeexecutor.state.validator import validate_nested_state_dict

        # Fix timedelta handling
        from tradeexecutor.monkeypatch.dataclasses_json import patch_dataclasses_json

        patch_dataclasses_json()

        # Insert special validation logic here to have
        # friendly error messages for the JSON serialisation errors
        data = self.to_dict(encode_json=False)
        validate_nested_state_dict(data)

        txt = json.dumps(data, cls=_ExtendedEncoder)
        return txt

    def write_json_file(self, path: Path | str):
        """Write JSON to a file.

        - Validates state before writing it out

        - Work around any serialisation quirks
        """

        if type(path) == str:
            path = Path(path)

        assert isinstance(path, Path)
        txt = self.to_json_safe()
        with path.open("wt") as out:
            out.write(txt)

    def get_strategy_start_and_end(self) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        """Get the time range for which the strategy should have data.

        TODO: Clean up this. In the future, each strategy must record its start (launched at)
        and end (last run) dates on the state itself.
        """

        if not self.stats.portfolio:
            logger.warning("No portfolio statistics, this is required for the time range")

            # Backwards compatible with legacy unit testing
            trades = list(self.portfolio.get_all_trades())
            if trades:
                return pd.Timestamp(trades[0].opened_at), pd.Timestamp(trades[-1].executed_at)

            return None, None

        start_at = pd.Timestamp(self.stats.portfolio[0].calculated_at)
        end_at = pd.Timestamp(self.stats.portfolio[-1].calculated_at)

        return start_at, end_at

    @staticmethod
    def read_json_file(path: Path | str) -> "State":
        """Read state from the JSON file.

        - Deal with all serialisation quirks
        """

        if type(path) == str:
            path = Path(path)

        assert isinstance(path, Path), f"Expected Path, got {path.__class__}"

        with open(path, "rt") as inp:
            return State.read_json_blob(inp.read())

    @staticmethod
    def read_json_blob(text: str) -> "State":
        """Parse state from JSON blob.

        - Deal with all serialisation quirks
        """

        assert isinstance(text, str)

        # Run in any monkey-patches we need for JSON decoding
        from tradeexecutor.monkeypatch.dataclasses_json import patch_dataclasses_json

        patch_dataclasses_json()
        return State.from_json(text)
