"""Portfolio state management."""

import logging
import datetime
import copy
import warnings
from dataclasses import dataclass, field
from decimal import Decimal
from itertools import chain
from typing import Dict, Iterable, Optional, Tuple, List, Set

from dataclasses_json import dataclass_json

from tradingstrategy.types import PrimaryKey
from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier, AssetFriendlyId
from tradeexecutor.state.loan import Loan
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.reserve import ReservePosition
from tradeexecutor.state.trade import TradeType, TradeFlag
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.types import USDollarAmount, BPS, USDollarPrice
from tradeexecutor.strategy.dust import get_dust_epsilon_for_pair
from tradeexecutor.strategy.trade_pricing import TradePricing


logger = logging.getLogger(__name__)


class NotEnoughMoney(Exception):
    """We try to allocate reserve for a buy trade, but do not have cash."""


class MultipleOpenPositionsWithAsset(Exception):
    """Multiple open spot positiosn for a single base asset.

    Cannot determine which one to return.
    """


class NotSinglePair(Exception):
    """Raised when there is zero or two plus pairs when single expected.

    See :py:meth:`Portfolio.get_single_pair`.
    """


class TooSmallTrade(Exception):
    """The trade amount is smaller than our dust epsilon.

    If the position quantity is too small, we cannot distinguish it from a closed position.
    """


@dataclass_json
@dataclass
class Portfolio:
    """Represents a trading portfolio on DEX markets.

    Multiple trading pair issue: We do not identify the actual assets we have purchased,
    but the trading pairs that we used to purchase them. Thus, each position marks
    "openness" in a certain trading pair. For example open position of WBNB-BUSD
    means we have bought X BNB tokens through WBNB-BUSD trading pair.
    But because this is DEX market, we could enter and exit through WBNB-USDT,
    WBNB-ETH, etc. and our position is not really tied to a trading pair.
    However, because all TradFi trading view the world through trading pairs,
    we keep the tradition here.
    """

    #: Each position gets it unique running counter id.
    next_position_id: int = field(default=1)

    #: Each trade gets it unique id as a running counter.
    #: Trade ids are unique across different positions.
    next_trade_id: int = field(default=1)

    #: Each balance update event gets it unique id as a running counter.
    next_balance_update_id: int = field(default=1)

    #: Currently open trading positions
    open_positions: Dict[int, TradingPosition] = field(default_factory=dict)

    #: Currently held reserve assets
    #:
    #: Token -> reserve position mapping.
    #:
    #: For migration code, see :py:class:`ReservePosition`.
    #:
    #: Set by :py:meth:`initialise_reserves`.
    #:
    reserves: Dict[AssetFriendlyId, ReservePosition] = field(default_factory=dict)

    #: Trades completed in the past
    closed_positions: Dict[int, TradingPosition] = field(default_factory=dict)

    #: Positions that have failed sells, or otherwise immovable and need manual clean up.
    #: Failure reasons could include
    #: - blockchain halted
    #: - ERC-20 token tax fees
    #: - rug pull token - transfer disabled
    frozen_positions: Dict[int, TradingPosition] = field(default_factory=dict)

    #: Mark positions that we cannot value as zero
    #:
    #: This is a backtesting issue workaround flag for disappearing markets.
    #: E.g. MKR-USDC liquidity disappears here https://tradingstrategy.ai/trading-view/ethereum/uniswap-v3/mkr-usdc-fee-5#7d
    #:
    #: TODO: Not supported yet.
    #:
    revalue_failures_as_zero: bool = False

    #: Positions which have not been opened yet, but are waiting the trade order trigger to happen.
    #:
    #: - When a trigger happens, the
    #:
    #: Will be pruned when these order expire.
    #:
    pending_positions: Dict[int, TradingPosition] = field(default_factory=dict)

    #: Positions from :py:attr:`pending_positions` that never triggered
    #:
    #: Stored for diagnostics.
    #:
    expired_positions: Dict[int, TradingPosition] = field(default_factory=dict)

    def __repr__(self):
        reserve_asset, _ = self.get_default_reserve_asset()
        reserve_position = self.get_reserve_position(reserve_asset)
        return f"<Portfolio with positions open:{len(self.open_positions)} frozen:{len(self.frozen_positions)} closed:{len(self.closed_positions)} and reserve {reserve_position.quantity} {reserve_position.asset.token_symbol}>"

    def is_empty(self):
        """This portfolio has no open or past trades or any reserves."""
        return len(self.open_positions) == 0 and len(self.reserves) == 0 and len(self.closed_positions) == 0

    def get_position_by_id(self, position_id: int) -> TradingPosition:
        """Get any position open/closed/frozen by id.

        Always assume the position for a `position_id` exists.

        :param position_id:
            Internal running counter id for the position inside
            this portfolio.

        :return:
            Always returns

        :throw:
            Fails with py:class:`AssertionError` if there is no such position.
        """

        assert position_id

        p1 = self.open_positions.get(position_id)
        p2 = self.closed_positions.get(position_id)
        if p2:
            # Sanity check we do not have the same position in multiple tables
            assert not p1
        p3 = self.frozen_positions.get(position_id)
        if p3:
            # Sanity check we do not have the same position in multiple tables
            assert not (p1 or p2)

        assert p1 or p2 or p3, f"Did not have position with id {position_id}"

        return p1 or p2 or p3

    def get_trade_by_id(self, trade_id: int) -> Optional[TradeExecution]:
        """Look up any trade in all positions.

        .. note ::]

            Slow lookup. Only designed for testing.

        :return:
            Found trade or
        """
        for p in self.get_all_positions():
            t = p.trades.get(trade_id)
            if t is not None:
                return t

        return None

    def get_trade_by_tx_hash(self, tx_hash: str) -> TradeExecution | None:
        """Find a trade that contains a particular transaction.

        :param tx_hash:
            Ethereum transaction hash

        :return:
            None if the portfolio does not contain such a trade
        """
        for t in self.get_all_trades():
            for b in t.blockchain_transactions:
                if b.tx_hash == tx_hash:
                    return t

        return None

    def get_all_positions(self, pending=False) -> Iterable[TradingPosition]:
        """Get open, closed and frozen, positions.

        :param pending:
            Include hypotethical market limit positions.
        """
        if pending:
            return chain(
                self.open_positions.values(),
                self.closed_positions.values(),
                self.frozen_positions.values(),
                self.pending_positions.values(),
            )
        else:
            return chain(self.open_positions.values(), self.closed_positions.values(), self.frozen_positions.values())

    def get_open_loans(self) -> Iterable[Loan]:
        """Get loans across all positions."""
        for p in self.get_open_and_frozen_positions():
            if p.loan:
                yield p.loan

    def get_open_and_frozen_positions(self) -> Iterable[TradingPosition]:
        """Get open and frozen, positions.

        These are all the positions where we have capital tied at the moment.
        """
        return chain(self.open_positions.values(), self.frozen_positions.values())

    def get_open_positions(self) -> Iterable[TradingPosition]:
        """Get currently open positions."""
        return self.open_positions.values()

    def get_unfrozen_positions(self) -> Iterable[TradingPosition]:
        """Get positions that have been repaired."""
        positions = chain(self.open_positions.values(), self.closed_positions.values())
        for p in positions:
            if p.is_unfrozen():
                yield p

    def get_executed_positions(self) -> Iterable[TradingPosition]:
        """Get all positions with already executed trades.

        Ignore positions that are still pending - they have only planned trades.
        """
        p: TradingPosition
        for p in self.open_positions.values():
            if p.has_executed_trades():
                yield p

    def get_open_position_for_pair(self, pair: TradingPairIdentifier) -> Optional[TradingPosition]:
        """Get Open position for a trading pair."""
        assert isinstance(pair, TradingPairIdentifier)
        pos: TradingPosition
        for pos in self.open_positions.values():
            if pos.pair.get_identifier() == pair.get_identifier():
                return pos
        return None

    def get_pending_position_for_pair(self, pair: TradingPairIdentifier) -> Optional[TradingPosition]:
        """Get pending position for a trading pair.

        - Used to check if we already have market limit ready for a pair
        """
        assert isinstance(pair, TradingPairIdentifier)
        pos: TradingPosition
        for pos in self.pending_positions.values():
            if pos.pair.get_identifier() == pair.get_identifier():
                return pos
        return None
    
    def get_closed_positions_for_pair(
        self,
        pair: TradingPairIdentifier,
        include_test_position: bool = False,
    ) -> list[TradingPosition]:
        """Get closed position for a trading pair."""

        return [
            p 
            for p in self.closed_positions.values()
            if p.pair == pair and (include_test_position or not p.is_test())
        ]
    

    def get_open_position_for_asset(self, asset: AssetIdentifier) -> Optional[TradingPosition]:
        """Get open position for a trading pair.

        - Check all open positions where asset is a base token

        :return:
            Tingle open position or None

        :raise MultipleOpenPositionsWithAsset:

            If more than one position is open

        """
        assert isinstance(asset, AssetIdentifier)

        matches = []

        pos: TradingPosition
        for pos in self.open_positions.values():
            if pos.pair.base == asset:
                matches.append(pos)

        if len(matches) > 1:
            raise MultipleOpenPositionsWithAsset(f"Querying asset: {asset} - found multipe open positions: {matches}")

        if len(matches) == 1:
            return matches[0]

        return None

    def get_open_quantities_by_position_id(self) -> Dict[str, Decimal]:
        """Return the current ownerships.

        Keyed by position id -> quantity.
        """
        return {p.get_identifier(): p.get_quantity() for p in self.open_positions.values()}

    def get_open_quantities_by_internal_id(self) -> Dict[int, Decimal]:
        """Return the current holdings in different trading pairs.

        Keyed by trading pair internal id -> quantity.
        """
        result = {}
        for p in self.open_positions.values():
            assert p.pair.internal_id, f"Did not have internal id for pair {p.pair}"
            result[p.pair.internal_id] = p.get_quantity()
        return result

    def open_new_position(self,
                          ts: datetime.datetime,
                          pair: TradingPairIdentifier,
                          assumed_price: USDollarPrice,
                          reserve_currency: AssetIdentifier,
                          reserve_currency_price: USDollarPrice) -> TradingPosition:
        """Opens a new trading position.

        - Marks the position opened.

        - Does not add any trades yet

        - Marks the current value of the portfolio at the trade opening time,
          as we need to use this for the risk calculations

        """

        portfolio_value = self.get_total_equity()

        p = TradingPosition(
            position_id=self.next_position_id,
            opened_at=ts,
            pair=pair,
            last_pricing_at=ts,
            last_token_price=assumed_price,
            last_reserve_price=reserve_currency_price,
            reserve_currency=reserve_currency,
            portfolio_value_at_open=portfolio_value,

        )
        self.open_positions[p.position_id] = p
        self.next_position_id += 1
        return p

    def get_position_by_trading_pair(
        self,
        pair: TradingPairIdentifier,
        pending=False,
    ) -> Optional[TradingPosition]:
        """Get open position by a trading pair smart contract address identifier.

        - Get the first open position for a trading pair

        - Optioonally check

        - Frozen positions not included

        See also

        - :py:func:`get_pending_position_for_pair`

        - :py:func:`get_open_position_for_pair`

        :param pending:
            Check also pending positions that wait market limit open and are not yet triggered

        """

        if pending:
            pending_position = self.get_pending_position_for_pair(pair)
            if pending_position:
                return pending_position

        # https://stackoverflow.com/a/2364277/315168
        return next((p for p in self.open_positions.values() if p.pair == pair), None)


    def get_existing_open_position_by_trading_pair(self, pair: TradingPairIdentifier) -> Optional[TradingPosition]:
        """Get a position by a trading pair smart contract address identifier.

        The position must have already executed trades (cannot be planned position(.
        """
        assert isinstance(pair, TradingPairIdentifier), f"Got {pair}"
        for p in self.open_positions.values():
            if p.has_executed_trades():
                if p.pair.pool_address == pair.pool_address:
                    return p
        return None

    def get_positions_closed_at(self, ts: datetime.datetime) -> Iterable[TradingPosition]:
        """Get positions that were closed at a specific timestamp.

        Useful to display closed positions after the rebalance.
        """
        for p in self.closed_positions.values():
            if p.closed_at == ts:
                yield p

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
        pair_fee: Optional[BPS] = None,
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
        """Create a trade.

        Trade can be opened by knowing how much you want to buy (quantity) or how much cash you have to buy (reserve).

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

            Use for repair trades.

        :param slippage_tolerance:
            Slippage tolerance for this trade.

            See :py:attr:`tradeexecutor.state.trade.TradeExecution.slippage_tolerance` for details.

        :param pending:
            Do not generate a new open position.

            Used when adding take profit triggers to market limit position.

        :return:
            Tuple of entries

            - Trade position (old/new)

            - New trade

            - True if a a new position was opened
        """

        if price_structure is not None:
            assert isinstance(price_structure, TradePricing)

        # Done in State.create_trade()
        # if quantity is not None:
        #    assert reserve is None, "Quantity and reserve both cannot be given at the same time"

        if position is None:
            # Open a new position
            position = self.get_position_by_trading_pair(pair)

        portfolio_value = self.get_total_equity()

        if position is None:
            # Initialise new position data structure
            position = self.open_new_position(
                strategy_cycle_at,
                pair,
                assumed_price,
                reserve_currency,
                reserve_currency_price)
            position.portfolio_value_at_open = portfolio_value
            created = True
        else:
            # Trade counts against old position,
            # - inc/dec size
            # - repair trades
            created = False

        assert position.pair == pair

        trade = position.open_trade(
            strategy_cycle_at,
            self.next_trade_id,
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
            slippage_tolerance=slippage_tolerance,
            portfolio_value_at_creation=portfolio_value,
            leverage=leverage,
            closing=closing,
            planned_collateral_consumption=planned_collateral_consumption,
            planned_collateral_allocation=planned_collateral_allocation,
            flags=flags,
        )

        # Update notes
        trade.add_note(notes)
        position.add_notes_message(notes)

        # Check we accidentally do not reuse trade id somehow

        self.next_trade_id += 1

        # Do not allow open positions that are so small
        # we cannot track
        dust_epsilon = get_dust_epsilon_for_pair(trade.pair)
        if trade.planned_quantity != 0:
            if abs(trade.planned_quantity) <= dust_epsilon:
                raise TooSmallTrade(f"Trade cannot be this small\n"
                                    f"Quantity {trade.planned_quantity}, dust epsilon {dust_epsilon}\n"
                                    f"Trade: {trade}\n"
                                    f"Pair: {trade.pair}")

        return position, trade, created

    def get_cash(self) -> USDollarAmount:
        """Get how much reserve stablecoins we have."""
        return sum([r.get_value() for r in self.reserves.values()])

    def get_current_cash(self):
        """Alias for get_cash()

        TODO: Deprecate
        """
        return self.get_cash()

    def get_position_equity_and_loan_nav(self, include_interest=True) -> USDollarAmount:
        """Get the equity tied tot the current trading positions.

        TODO: Rename this function - also deals with loans not just equity

        - Includes open positions
        - Does not include frozen positions
        """

        # Any trading positions we have one
        spot_values = sum([p.get_equity() for p in self.open_positions.values() if not p.is_loan_based()])

        # Minus any outstanding loans we have
        leveraged_values = self.get_all_loan_nav(include_interest)

        return spot_values + leveraged_values

    def get_all_loan_nav(self,
                         include_interest=True,
                         include_trading_fees=True,
                         ) -> USDollarAmount:
        """Get net asset value we can theoretically free from leveraged positions.

        :param include_interest:
            Include accumulated interest in the calculations

        :param include_trading_fees:
            Include trading fees in the calculations

        """
        return sum([p.get_loan_based_nav(include_interest, include_trading_fees) for p in self.open_positions.values() if p.is_loan_based()])

    def get_frozen_position_equity(self) -> USDollarAmount:
        """Get the value of trading positions that are frozen currently."""
        return sum([p.get_value() for p in self.frozen_positions.values()])

    def get_live_position_equity(self) -> USDollarAmount:
        """Get the value of current trading positions plus unexecuted trades."""
        return sum([p.get_value() for p in self.open_positions.values()])

    def get_total_equity(self) -> USDollarAmount:
        """Get the value of the portfolio based on the latest pricing.

        This includes

        - Equity Value of the positions

        - ...and cash in the hand

        But not

        - Leverage/loan based positions (equity is in collateral)

        See also :py:meth:`get_theoretical_value`

        """

        # Any trading positions we have one
        # spot_values = sum([p.get_equity() for p in self.open_positions.values() if not p.is_leverage()])
        return self.get_position_equity_and_loan_nav() + self.get_cash()

    def get_net_asset_value(self, include_interest=True) -> USDollarAmount:
        """Calculate portfolio value if every position would be closed now.

        This includes

        - Cash

        - Equity hold in spot positions

        - Net asset value hold in leveraged positions

        TODO: Net asset value calculation does not account for fees
        paid to close a short position.
        """
        return self.get_position_equity_and_loan_nav(include_interest) + self.get_cash()

    def get_unrealised_profit_usd(self) -> USDollarAmount:
        """Get the profit of currently open positions.

        - This profit includes spot market equity i.e. holding tokens

        See also :py:meth:`get_unrealised_profit_in_leveraged_positions`.
        """
        return sum([p.get_unrealised_profit_usd() for p in self.open_positions.values()])

    def get_unrealised_profit_in_leveraged_positions(self) -> USDollarAmount:
        """Get the profit of currently open margiend positions.

        - This profit is not included in the portfolio total equity

        See also :py:meth:`get_unrealised_profit_usd`.
        """
        return sum([p.get_unrealised_profit_usd() for p in self.open_positions.values() if p.is_leverage()])

    def get_realised_profit_in_leveraged_positions(self) -> USDollarAmount:
        """Get the profit of currently open margiend positions.

        - This profit is not included in the portfolio total equity

        See also :py:meth:`get_unrealised_profit_usd`.
        """
        return sum([p.get_realised_profit_usd() for p in self.open_positions.values() if p.is_leverage()])

    def get_closed_profit_usd(self) -> USDollarAmount:
        """Get the value of the portfolio based on the latest pricing."""
        return sum([p.get_total_profit_usd() for p in self.closed_positions.values()])

    def find_position_for_trade(self, trade, pending=False) -> Optional[TradingPosition]:
        """Find a position that a trade belongs for.

        :param pending:
            Include pending positions (not yet trading for market limit)
        """

        if pending:
            pending_position = self.pending_positions.get(trade.position_id)
            if pending_position:
                return pending_position

        return self.get_position_by_id(trade.position_id)

    def get_reserve_position(self, asset: AssetIdentifier) -> ReservePosition:
        """Get reserves for a certain reserve asset.

        :raise KeyError:
            If we do not have reserves for the asset
        """

        # Legacy data support.
        # All new reserves are encoded as chain id + address
        if asset.address in self.reserves:
            return self.reserves[asset.address]

        # The modern code path
        return self.reserves[asset.get_identifier()]

    def get_default_reserve_position(self) -> ReservePosition:
        """Get the default reserve position.

        Assume portfolio has only one reserve asset.

        :raise AssertionError:
            If there is not exactly one reserve position
        """
        positions = list(self.reserves.values())
        assert len(positions) == 1, f"Had {len(positions)} reserve position"
        return positions[0]

    def get_reserve_assets(self) -> List[AssetIdentifier]:
        """Get reserves assets.

        Reserve assets are registered with the state when it is initialised.

        :return:
            If the state is not properly initialised, the reserve asset list is empty.
        """
        return [r.asset for r in self.reserves.values()]

    def get_equity_for_pair(self, pair: TradingPairIdentifier) -> Decimal:
        """Return how much equity allocation we have in a certain trading pair."""
        position = self.get_position_by_trading_pair(pair)
        if position is None:
            return Decimal(0)
        return position.get_quantity_old()

    def close_position(
        self,
        position: TradingPosition,
        executed_at: datetime.datetime,
    ):
        """Move a position from open positions to closed ones.

        See also :py:meth:`TradingPosition.can_be_closed`.

        :param position:
            Trading position where the trades and balance updates quantity equals to zero

        :param executed_at:
            Wall clock time

        """

        assert position.position_id in self.open_positions, f"Not in open positions: {position}"

        # Move position to closed
        logger.info("Marking position to closed: %s at %s", position, executed_at)
        position.closed_at = executed_at
        del self.open_positions[position.position_id]
        self.closed_positions[position.position_id] = position

        assert position.is_closed()

    def adjust_reserves(
        self,
        asset: AssetIdentifier,
        amount: Decimal,
        reason: str = None,
    ):
        """Add or remove assets from the cash reserves.

        For internal accounting of the portfolio state.

        :param asset:
            Reserve asset

        :param amount:
            Negative to reduce portfolio reserves, positive to increase

        :param reason
            Human-readable loggable reason why this happened
        """

        assert asset is not None, "Asset missing"

        assert isinstance(amount, Decimal), f"Expected Decimal. Got amount {amount.__class__}: {amount}"

        if amount == 0:
            # This action does not need reserve adjustment
            return

        reserve = self.get_reserve_position(asset)
        assert reserve, f"No reserves available for {asset}"
        assert reserve.quantity is not None, f"Reserve quantity not set for {asset} in portfolio {self}"

        # TODO: On paper reserves can go negative.
        # because we might execute sell trade that increase our capital
        # before executing buy trades
        # assert reserve.quantity + amount >= 0, f"Reserves went to negative with new amount {amount}, current reserves {reserve.quantity}"

        logger.info(
            "Adjusting reserves for %s: %+f, reason: %s",
            asset.token_symbol,
            amount,
            reason or "<not given>",
        )

        reserve.quantity += amount

    def move_capital_from_reserves_to_spot_trade(
            self,
            trade: TradeExecution,
            underflow_check=True):
        """Allocate capital from reserves to trade instance.

        Total equity of the porfolio stays the same.
        """

        if trade.is_spot():
            assert trade.is_buy()

        reserve = trade.get_planned_reserve()
        try:
            position = self.get_reserve_position(trade.reserve_currency)
            available = position.quantity
        except KeyError as e:
            raise RuntimeError(f"Reserve missing for {trade.reserve_currency}") from e

        # Sanity check on price calculatins
        if trade.is_spot():
            assert abs(float(reserve) - trade.get_planned_value()) < 0.01, f"Trade {trade}: Planned value {trade.get_planned_value()}, but wants to allocate reserve currency for {reserve}"

        if underflow_check:
            if available < reserve:
                raise NotEnoughMoney(f"Not enough reserves. We have {available}, trade {trade} wants {reserve}")

        trade.reserve_currency_allocated = reserve

        self.adjust_reserves(
            trade.reserve_currency,
            -reserve,
            f"Moving USD from reserves to the trade {trade}"
        )

    def return_capital_to_reserves(
            self,
            trade: TradeExecution,
            underflow_check=True):
        """Return capital to reserves after a spot sell or collateral returned.

        """
        if trade.is_spot():
            assert trade.is_sell()

        assert trade.executed_reserve > 0

        self.adjust_reserves(
            trade.reserve_currency,
            trade.executed_reserve,
            f"Returning USD to reserves from trade::\n{trade}"
        )

    def has_unexecuted_trades(self) -> bool:
        """Do we have any trades that have capital allocated, but not executed yet."""
        return any([p.has_unexecuted_trades() for p in self.open_positions])

    def update_reserves(self, new_reserves: List[ReservePosition]):
        """Update current reserves.

        Overrides current amounts of reserves.

        E.g. in the case users have deposited more capital.
        """

        assert not self.has_unexecuted_trades(), "Updating reserves while there are trades in progress will mess up internal account"

        for r in new_reserves:
            self.reserves[r.get_identifier()] = r

    def check_for_nonce_reuse(self, nonce: int):
        """A helper assert to see we are not generating invalid transactions somewhere.

        :raise: AssertionError
        """
        for p in self.get_all_positions():
            for t in p.trades.values():
                for tx in t.blockchain_transactions:
                    assert tx.nonce != nonce, f"Nonce {nonce} is already being used by trade {t} with txinfo {t.tx_info}"

    def get_default_reserve_asset(self) -> Tuple[Optional[AssetIdentifier], Optional[USDollarPrice]]:
        """Gets the default reserve currency associated with this state.

        For strategies that use only one reserve currency.
        This is the first in the reserve currency list.

        See also

        - :py:meth:`get_default_reserve_position`

        :return:
            Tuple (Reserve currency asset, its latest US dollar exchanage rate)

            Return (None, 1.0) if the strategy has not seen any deposits yet.
        """

        # Legacy path
        if len(self.reserves) == 0:
            return None, 1.0

        res_pos = next(iter(self.reserves.values()))
        return res_pos.asset, res_pos.reserve_token_price

    def get_all_trades(self) -> Iterable[TradeExecution]:
        """Iterate through all trades: completed, failed and in progress"""
        pos: TradingPosition
        for pos in self.get_all_positions():
            for t in pos.trades.values():
                yield t

    def get_first_and_last_executed_trade(self) -> Tuple[Optional[TradeExecution], Optional[TradeExecution]]:
        """Get first and last trades overall."""

        first = last = None

        for t in self.get_all_trades():
            if first is None:
                first = t
            if last is None:
                last = t
            if t.executed_at and first.executed_at and (t.executed_at < first.executed_at):
                first = t
            if t.executed_at and last.executed_at and (t.executed_at > last.executed_at):
                last = t
        return first, last

    def get_trading_history_duration(self) -> Optional[datetime.timedelta]:
        """How long this portfolio has trading history.

        Calculated as the difference between first and last executed trade.

        :return:
            None if there has been any trades.

            Zero seconds period if there is only a single trade.
        """
        first, last = self.get_first_and_last_executed_trade()
        
        if first and last:
            if first.executed_at and last.executed_at:
                return last.executed_at - first.executed_at

        return None

    def get_initial_deposit(self) -> Optional[USDollarAmount]:
        """Deprecated.

        See :py:meth:`get_initial_cash`
        """
        warnings.warn('This function is deprecated. Use get_initial_cash() instead', DeprecationWarning, stacklevel=2)
        return self.get_initial_cash()

    def get_initial_cash(self) -> Optional[USDollarAmount]:
        """How much we invested at the beginning of a backtest.

        .. note::

            Only applicable to the backtest. Will fail
            for live strategies.

        - Assumes we track the performance against the US dollar

        - Assume there has been only one deposit event

        - This deposit happened at the start of the backtest

        TODO: Shoud not be used, as we have new `SyncModel` instance
        for backtesters. Code will be removed.
        """

        if len(self.reserves) == 0:
            return 0

        assert len(self.reserves) == 1, f"Reserve assets are not defined for this state, cannot get initial deposit\n" \
                                        f"State is {self}\n" \
                                        f"Reserves are {self.reserves}\n"
        reserve = next(iter(self.reserves.values()))
        if reserve.initial_deposit:
            return float(reserve.initial_deposit) * reserve.initial_deposit_reserve_token_price
        return None

    def get_all_traded_pairs(self) -> Iterable[TradingPairIdentifier]:
        """Get all pairs for which we have or had positions."""
        already_iterated_pairs = set()
        for p in self.get_all_positions():
            if p.pair not in already_iterated_pairs:
                already_iterated_pairs.add(p.pair)
                yield p.pair

    def initialise_reserves(self, asset: AssetIdentifier):
        """Create the initial reserve currency list.

        Currently we assume there can be only one reserve currency.
        """
        assert len(self.reserves) == 0, "Reserves already initialised"
        self.reserves[asset.address] = ReservePosition(
            asset=asset,
            quantity=Decimal(0),
            last_sync_at=None,
            reserve_token_price=None,
            last_pricing_at=None,
        )

    def get_single_pair(self) -> TradingPairIdentifier:
        """Return the only trading pair a single pair strategy has been trading.

        :raise NotSinglePair:

            This may happen when

            - Strategy has not traded yet

            - Strategy has traded multiple pairs
        """
        pairs = {p for p in self.get_all_traded_pairs()}
        if len(pairs) != 1:
            raise NotSinglePair(f"We have {len(pairs)} trading pairs, one expected")

        (pair,) = pairs
        return pair

    def allocate_balance_update_id(self) -> PrimaryKey:
        """Get a new balance update event id."""
        self.next_balance_update_id += 1
        return self.next_balance_update_id - 1

    def correct_open_position_balance(
            self,
            position: TradingPosition,
            expected_amount: Decimal,
            actual_amount: Decimal,
            strategy_cycle_ts: datetime.datetime,
            block_number: int,
            balance_update_id: int,
            ) -> TradeExecution:
        """Create an accounting entry trade that correct the balance."""

        raise NotImplementedError("This method is currently not being used, as the trading positions take account of direct quantity updates in get_quantity()")
        #
        # trade_id = self.next_trade_id
        # self.next_trade_id += 1
        #
        # assumed_price = position.last_token_price
        #
        # correction_amount = actual_amount - expected_amount
        #
        # trade = position.open_trade(
        #     strategy_cycle_ts,
        #     trade_id,
        #     quantity=correction_amount,
        #     reserve=None,
        #     assumed_price=assumed_price,
        #     trade_type=TradeType.accounting_correction,
        #     reserve_currency=position.reserve_currency,
        #     reserve_currency_price=position.last_reserve_price,
        # )
        #
        # trade.balance_update_id = balance_update_id
        # trade.notes = f"Accounting correction based on the actual on-chain balances.\n" \
        #               f"The internal ledger balance was  {expected_amount} {position.pair.base.token_symbol}\n" \
        #               f"On-chain balance was {actual_amount} {position.pair.base.token_symbol} at block {block_number:,}\n" \
        #               f"Balance was updated {correction_amount} {position.pair.base.token_symbol}\n"
        #
        # trade.mark_success(
        #     datetime.datetime.utcnow(),
        #     trade.planned_price,
        #     trade.planned_quantity,
        #     trade.planned_reserve,
        #     lp_fees=0,
        #     native_token_price=position.last_reserve_price,
        #     force=True,
        # )
        #
        # return trade

    def get_current_credit_positions(self) -> List[TradingPosition]:
        """Return currently open credit positions."""
        credit_positions = [p for p in self.get_open_and_frozen_positions() if p.is_credit_supply()]
        return credit_positions
    
    def get_leverage_positions(self) -> List[TradingPosition]:
        """Return currently open credit positions."""
        return [p for p in self.get_open_and_frozen_positions() if p.is_leverage()]

    def get_current_interest_positions(self) -> List[TradingPosition]:
        """Get lis of all positions for which we need to sync the on-chain interest"""
        return self.get_current_credit_positions() + self.get_leverage_positions()

    def get_borrowed(self) -> USDollarAmount:
        return sum([p.get_borrowed() for p in self.get_open_and_frozen_positions()])

    def get_loan_net_asset_value(self) -> USDollarAmount:
        """What is our Net Asset Value (NAV) across all open loan positions."""
        return sum(l.get_net_asset_value() for l in self.get_open_loans())

    def has_trading_capital(self, threshold_usd=0.15) -> bool:
        """Does this strategy have non-zero deposits and total equity?

        Check the reserves.

        - If we have zero deposits, do not attempt to trade

        - The actual amount is a bit above zero to account for rounding errors

        :return:
            If we have any capital to trade
        """
        return self.get_total_equity() >= threshold_usd
    
    def get_total_claimed_interest(self) -> USDollarAmount:
        """Get the total interest claimed from the positions."""
        return sum(p.get_claimed_interest() for p in self.get_all_positions())
    
    def get_total_repaid_interest(self) -> USDollarAmount:
        """Get the total interest repaid from the positions."""
        return sum(p.get_repaid_interest() for p in self.get_all_positions())
