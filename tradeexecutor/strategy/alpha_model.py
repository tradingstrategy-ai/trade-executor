"""Alpha model and portfolio construction model related logic."""
import datetime
import heapq
import logging
from _decimal import Decimal

from dataclasses import dataclass, field
from io import StringIO
from types import NoneType
from typing import Optional, Dict, Iterable, List

import pandas as pd
import numpy as np
from dataclasses_json import dataclass_json

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.trade import TradeExecution, TradeType
from tradeexecutor.state.types import PairInternalId, USDollarAmount, Percent

from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.weighting import weight_by_1_slash_n, check_normalised_weights, normalise_weights


logger = logging.getLogger(__name__)


@dataclass_json
@dataclass(slots=True)
class TradingPairSignal:
    """Present one asset in alpha model weighting.

    - The life cycle of the instance is one strategy cycle and it is part of
      :py:class:`AlphaModel`

    - Asset is represented as a trading pair, as that is how we internally present assets

    - We capture all the calculations and intermediate values for a single asset
      in one instance (row) per each trading strategy cycle, making
      investigations for alpha model strategies easy

    - Required variables (pair, signal) are =input from `decide_trades()` function in a strategy

    - Optional variables are calculated and filled in the various phases of alpha model processing,
      as the model moves from abstract weightings to actual trade execution and dollar amounts

    - When we need to close old positions, we automatically generate :py:attr:`old_weight`
      and negative :py:attr:`position_adjust` for them

    - Data here is serialisable for visualisation a a part of the strategy state visualisation
      and also for console logging diagnostics

    .. note ::

        Currently only longs are supported.

    """

    #: For which pair is this alpha weight
    #:
    #:
    pair: TradingPairIdentifier

    #: Raw signal
    #:
    #: E.g. raw value of the momentum.
    #:
    #: Can be any number between ]-inf, inf[
    #:
    #: Set zero for pairs that are discarded, e.g. due to risk assessment.
    #:
    signal: float

    #: Stop loss for this position
    #:
    #: Used for the risk management.
    #:
    #: 0.98 means 2% stop loss over mid price at open.
    #:
    #: Set to `None` to disable stop loss.
    stop_loss: Optional[Percent] = None

    #: Take profit for this position
    #:
    #: Used for the risk management.
    #:
    #: 1.02 means 2% take profit over mid price at open.
    #:
    #: Set to `None` to disable stop loss.
    take_profit: Optional[Percent] = None

    #: Trailing stop loss for this position
    #:
    #: See :py:attr:`tradeexecutor.state.position.TradingPosition.trailing_stop_loss_pct` for details.
    #:
    trailing_stop_loss: Optional[Percent] = None

    #: Raw portfolio weight
    #:
    #: Each raw signal is assigned to a weight based on some methodology,
    #: e.g. 1/N where the highest signal gets 50% of portfolio weight.
    raw_weight: Percent = 0.0

    #: Weight 0...1 so that all portfolio weights sum to 1
    normalised_weight: Percent = 0.0

    #: Old weight of this pair from the previous cycle.
    #:
    #: If this asset was part of the portfolio at previous :term:`strategy cycle`
    #: then this is the value of the previous cycle weight.
    #: The old weight is always normalised.
    #:
    #: This can be dynamically calculated from the :py:class:`tradeexecutor.state.portfolio.Portfolio` state.
    old_weight: Percent = 0.0

    #: Old US Dollar value of this value from the previous cycle.
    #:
    #: If this asset was part of the portfolio at previous :term:`strategy cycle`
    #: then this is the value of the previous cycle weight.
    #:
    old_value: USDollarAmount = 0.0

    #: How many dollars we plan to invest on trading pair.
    #:
    #: Calculated by portfolio total investment equity * normalised weight * price.
    position_target: USDollarAmount = 0.0

    #: How much we are going to increase/decrease the position on this strategy cycle.
    #:
    #: Used when the position increases and we need to know how
    #: many dollars we need to spend to buy more.
    #:
    #: If this is a positive, then we need to make a buy trade for this amount to
    #: reach out target position for this cycle. If negative then we need
    #: to decrease our position.
    position_adjust_usd: USDollarAmount = 0.0

    #: How much we are going to increase/decrease the position on this strategy cycle.
    #:
    #: Used when the position decreases and we need to know
    #: how many units of asset we need to sell to get to the :py:attr:`position_target`.
    #:
    #: At the momeny always negative and available only when decreasing a position.
    #:
    #: Note that this value is not used when closing position (weight=0),
    #: due to rounding and epsilon errors.
    #:
    position_adjust_quantity: float = 0.0

    #: Trading position that is controlled by this signal.
    #:
    #: Query with :py:meth:`tradeexecutor.state.portfolio.Portfolio.get_position_by_id`
    #:
    #: After open, any position will live until it is fully closed.
    #: After that a new position will be opened.
    position_id: Optional[int] = None

    #: No rebalancing trades was executed for this position adjust.
    #:
    #: This is because the resulting trade is under the minimum trade threshold.
    position_adjust_ignored: bool = False

    #: What was the profit of the position of this signal.
    #:
    #: Record the historical profit as the part of the signal model.
    #: Makes building alpha model visualisation easier later,
    #: so that we can show the profitability of the position of the signal.
    #:
    #: Calculate the position profit before any trades were executed.
    profit_before_trades: USDollarAmount = 0

    #: What was the profit of the position of this signal.
    #:
    #: Record the historical profit as the part of the signal model.
    #: Makes building alpha model visualisation easier later,
    #: so that we can show the profitability of the position of the signal.
    #:
    #: Calculate the position profit before any trades were executed.
    profit_before_trades_pct: Percent = 0

    def __post_init__(self):
        assert isinstance(self.pair, TradingPairIdentifier)
        if type(self.signal) != float:
            # Convert from numpy.float64
            self.signal = float(self.signal)

    def __repr__(self):
        return f"Pair: {self.pair.get_ticker()} old weight: {self.old_weight:.4f} old value: {self.old_value:,} raw signal:{self.signal:.4f} normalised weight: {self.normalised_weight:.4f} new value: {self.position_target:,} adjust: {self.position_adjust_usd:,}"

    def has_trades(self) -> bool:
        """Did/should this signal cause any trades to be executed.

        - We have trades if we need to rebalance (old weight != new weight)

        - Even if the weight does not change we might still rebalance because the prices change

        - Some adjustments might be too small and then we just ignore any trades
          and have :py:attr:position_adjust_ignored` flag set
        """
        return (self.normalised_weight or self.old_weight) and not self.position_adjust_ignored


@dataclass_json
@dataclass(slots=True)
class AlphaModel:
    """Capture alpha model state for one strategy cycle.

    - A helper class for portfolio construction models and such

    - Converts portfolio weightings to rebalancing trades

    - Supports stop loss and passing through other trade execution parameters

    - Each :term:`strategy cycle` creates its own
      :py:class:`AlphaModel` instance in `decide_trades()` function of the strategy

    - Stores the intermediate results of the calculationsn between raw
      weights and the final investment amount

    - We are serializable as JSON, so we can pass the calculations
      as data around in :py:attr:`tradeexecutor.state.visualisation.Visualisation.calculations`
      and then later visualise alph model progress over time with other analytic
      diagrams

    """

    #: Timestamp of the strategy cycle for which this alpha model was calculated
    #:
    timestamp: Optional[datetime.datetime] = None

    #: Calculated signals for all trading pairs.
    #:
    #: Pair internal id -> trading signal data.
    #:
    #: For all trading pairs in the model.
    #:
    #: Set by :py:meth:`set_signal`
    #:
    raw_signals: Dict[PairInternalId, TradingPairSignal] = field(default_factory=dict)

    #: The chosen top signals.
    #:
    #: Pair internal id -> trading signal data.
    #:
    #: For signals chosen for the rebalance, e.g. top 5 long signals.
    #:
    #:
    #: Set by :py:meth:`select_top_signals`
    #:
    signals: Dict[PairInternalId, TradingPairSignal] = field(default_factory=dict)

    #: How much we can afford to invest on this cycle
    investable_equity: Optional[USDollarAmount] = 0.0

    #: Determine the lower threshold for a position weight.
    #:
    #: Clean up "dust" by explicitly closing positions if they fall too small.
    #:
    #: If position weight is less than 0.5% always close it
    close_position_weight_epsilon: Percent = 0.005

    #: Allow set_signal() to override stop loss set for the position earlier
    #:
    override_stop_loss = False

    def __post_init__(self):
        if self.timestamp is not None:
            if isinstance(self.timestamp, pd.Timestamp):
                # need to make serializable
                self.timestamp = self.timestamp.to_pydatetime()
            assert isinstance(self.timestamp, datetime.datetime)

    def iterate_signals(self) -> Iterable[TradingPairSignal]:
        """Iterate over all recorded signals."""
        yield from self.signals.values()

    def get_signal_by_pair_id(self, pair_id: PairInternalId) -> Optional[TradingPairSignal]:
        """Get a trading pair signal instance for one pair.

        Use integer id lookup.
        """
        return self.signals.get(pair_id)

    def get_signal_by_pair(self, pair: TradingPairIdentifier) -> Optional[TradingPairSignal]:
        """Get a trading pair signal instance for one pair.

        Use verbose :py:class:`TradingPairIdentifier` lookup.
        """
        return self.get_signal_by_pair_id(pair.internal_id)

    def get_signals_sorted_by_weight(self) -> Iterable[TradingPairSignal]:
        """Get the signals sorted by the weight.

        Return the highest weight first.
        """
        return sorted(self.signals.values(), key=lambda s: s.raw_weight, reverse=True)

    def get_debug_print(self) -> str:
        """Present the alpha model in a format suitable for the console."""
        buf = StringIO()
        print(f"Alpha model for {self.timestamp}, for USD {self.investable_equity:,} investments", file=buf)
        for idx, signal in enumerate(self.get_signals_sorted_by_weight(), start=1):
            print(f"   Signal #{idx} {signal}", file=buf)
        return buf.getvalue()

    def set_signal(
            self,
            pair: TradingPairIdentifier,
            alpha: float | np.float32,
            stop_loss: Percent | NoneType = None,
            take_profit: Percent | NoneType = None,
            trailing_stop_loss: Percent | NoneType = None,
            ):
        """Set trading pair alpha to a value.

        If called repeatatle for the same trading pair,
        remember the last value.

        :param pair:
            Trading pair

        :param alpha:
            How much alpha signal this trading pair carries.

            Set to zero to have the pair excluded out after a risk assessment

        :param stop_loss:
            Stop loss threshold for this pair.

            As the percentage of the position value.

            `0.98` means 2% stop loss.

        :param take_profit:
            Stop loss threshold for this pair.

            As the percentage of the position value.

            `1.02` means 2% take profit.

        :param trailing_stop_loss:
            Trailing stop loss threshold for this pair.

            As the percentage of the position value.

            `0.98` means 2% trailing stop loss.
        """

        # Don't let Numpy values beyond this point, as
        # they cause havoc in serisaliation
        if isinstance(alpha, np.float32):
            alpha = float(alpha)

        if alpha == 0:
            # Delete so that the pair so that it does not get any further computations
            if pair.internal_id in self.raw_signals:
                del self.raw_signals[pair.internal_id]

        else:
            signal = TradingPairSignal(
                pair=pair,
                signal=alpha,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trailing_stop_loss=trailing_stop_loss,
            )
            self.raw_signals[pair.internal_id] = signal

    def set_old_weight(
            self,
            pair: TradingPairIdentifier,
            old_weight: float,
            old_value: USDollarAmount,
            ):
        """Set the weights for the8 current portfolio trading positions before rebalance."""
        if pair.internal_id in self.signals:
            self.signals[pair.internal_id].old_weight = old_weight
            self.signals[pair.internal_id].old_value = old_value
        else:
            self.signals[pair.internal_id] = TradingPairSignal(
                pair=pair,
                signal=0,
                old_weight=old_weight,
                old_value=old_value,
            )

    def select_top_signals(self,
                           count: int,
                           threshold=0.0,
                           ):
        """Chooses top long signals.

        Sets :py:attr:`signals` attribute of the model

        Example:

        .. code-block:: python

            alpha_model.select_top_signals(
                count=5,  # Pick top 5 trading pairs
                threshold=0.01,  # Need at least 1% signal certainty to be eligible
            )

        :param count:
            How many signals to pick

        :param threshold:
            If the raw signal value is lower than this threshold then don't pick the signal.

            Inclusive.

            `0.01 = 1%` signal strenght.
        """
        filtered_signals = [s for s in self.raw_signals.values() if s.signal >= threshold]
        top_signals = heapq.nlargest(count, filtered_signals, key=lambda s: s.raw_weight)
        self.signals = {s.pair.internal_id: s for s in top_signals}

    def normalise_weights(self):
        raw_weights = {s.pair.internal_id: s.raw_weight for s in self.signals.values()}
        normalised = normalise_weights(raw_weights)
        for pair_id, normal_weight in normalised.items():
            self.signals[pair_id].normalised_weight = normal_weight

    def assign_weights(self, method=weight_by_1_slash_n):
        """Convert raw signals to their portfolio weight counterparts.

        Update :py:attr:`TradingPairSignal.raw_weight` attribute
        to our target trading pairs.

        :param method:
            What method we use to convert a trading signal to a portfolio weights
        """
        raw_signals = {s.pair.internal_id: s.signal for s in self.signals.values()}
        weights = method(raw_signals)
        for pair_id, raw_weight in weights.items():
            self.signals[pair_id].raw_weight = raw_weight

    def update_old_weights(self, portfolio: Portfolio):
        """Update the old weights of the last strategy cycle to the alpha model.

        - Update % of portfolio weight of an asset

        - Update USD portfolio value of an asset
        """
        total = portfolio.get_open_position_equity()
        for position in portfolio.open_positions.values():
            value = position.get_value()
            weight = value  / total
            self.set_old_weight(
                position.pair,
                weight,
                value,
            )

    def calculate_weight_diffs(self) -> Dict[PairInternalId, float]:
        """Calculate how much % asset weight has changed between strategy cycles.

        :return:
            Pair id, weight delta dict
        """

        new_weights = {s.pair.internal_id: s.normalised_weight for s in self.signals.values()}
        existing_weights = {s.pair.internal_id: s.old_weight for s in self.signals.values()}

        # Check that both inputs are sane
        check_normalised_weights(new_weights)
        check_normalised_weights(existing_weights)

        diffs = {}
        for id, new_weight in new_weights.items():
            diffs[id] = new_weight - existing_weights.get(id, 0)

        # Refill gaps of old assets that did not appear
        # in the new portfolio
        for id, old_weight in existing_weights.items():
            if id not in diffs:
                # Sell all
                diffs[id] = -old_weight

        return diffs

    def calculate_target_positions(self, position_manager: PositionManager, investable_equity: USDollarAmount):
        """Calculate individual dollar amount for each position based on its normalised weight."""
        # dollar_values = {pair_id: weight * investable_equity for pair_id, weight in diffs.items()}

        self.investable_equity = investable_equity

        for s in self.iterate_signals():
            s.position_target = s.normalised_weight * investable_equity
            s.position_adjust_usd = s.position_target - s.old_value

            if s.position_adjust_usd < 0:
                # Decreasing positions by selling the token
                # A lot of options here how to go about this.
                # We might get some minor position size skew here because fees not included
                # for these transactions
                s.position_adjust_quantity = position_manager.estimate_asset_quantity(s.pair, s.position_adjust_usd)
                assert type(s.position_adjust_quantity) == float

    def generate_rebalance_trades_and_triggers(
            self,
            position_manager: PositionManager,
            min_trade_threshold: USDollarAmount = 10.0,
    ) -> List[TradeExecution]:
        """Generate the trades that will rebalance the portfolio.

        This will generate

        - Sells for the existing assets

        - Buys for new assets or assets where we want to increase our position

        - Set up take profit/stop loss triggers for positions

        :param position_manager:
            Portfolio of our existing holdings

        :param min_trade_threshold:
            Threshold for too small trades.

            If the notional value of a rebalance trade is smaller than this
            USD amount don't make a trade, but keep whatever
            position we currently we have.

            This is to prevent doing too small trades due to fuzziness in the valuations
            and calculations.

        :return:
            List of trades we need to execute to reach the target portfolio.
            The sells are sorted always before buys.
        """
        # Generate trades
        trades: List[TradeExecution] = []

        for signal in self.iterate_signals():
            pair = signal.pair

            dollar_diff = signal.position_adjust_usd
            quantity_diff = signal.position_adjust_quantity
            value = signal.position_target

            # Do backtesting record keeping
            position = position_manager.get_current_position_for_pair(pair)
            if position:
                signal.profit_before_trades = position.get_total_profit_usd()
                signal.profit_before_trades_pct = position.get_total_profit_percent()
            else:
                signal.profit_before_trades = 0

            logger.info("Rebalancing %s, old weight: %f, new weight: %f, diff: %f USD",
                        pair,
                        signal.old_weight,
                        signal.normalised_weight,
                        dollar_diff)

            if abs(dollar_diff) < min_trade_threshold:
                logger.info("Not doing anything, value %f below trade threshold %f", value, min_trade_threshold)
                signal.position_adjust_ignored = True
            else:

                if signal.normalised_weight < self.close_position_weight_epsilon:
                    # Explicit close to avoid rounding issues
                    position = position_manager.get_current_position_for_pair(signal.pair)
                    if position:
                        trade = position_manager.close_position(
                            position,
                            TradeType.rebalance,
                        )
                        position_rebalance_trades = [trade]
                else:
                    # Increase or decrease the position.
                    # Open new position if needed.
                    position_rebalance_trades = position_manager.adjust_position(
                        pair,
                        dollar_diff,
                        quantity_diff,
                        signal.normalised_weight,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        trailing_stop_loss=signal.trailing_stop_loss,
                        override_stop_loss=self.override_stop_loss,
                    )

                assert len(position_rebalance_trades) == 1, "Assuming always on trade for rebalance"

                # Connect trading signal to its position
                first_trade = position_rebalance_trades[0]
                assert first_trade.position_id
                signal.position_id = first_trade.position_id

                logger.info("Adjusting holdings for %s: %s", pair, position_rebalance_trades[0])
                trades += position_rebalance_trades

        trades.sort(key=lambda t: t.get_execution_sort_position())

        # Return all rebalance trades
        return trades
