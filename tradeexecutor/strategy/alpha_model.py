"""Alpha model and portfolio construction model related logic."""
import datetime
import heapq
import logging

from dataclasses import dataclass
from io import StringIO
from typing import Optional, Dict, Iterable, List

import pandas as pd
import numpy as np
from dataclasses_json import dataclass_json

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.types import PairInternalId, USDollarAmount

from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.weighting import weight_by_1_slash_n, check_normalised_weights, normalise_weights


logger = logging.getLogger(__name__)


@dataclass_json
@dataclass(slots=True)
class TradingPairSignal:
    """Present one asset in alpha model weighting.

    - Asset is represented as a trading pair, as that is how we internally present assets

    - We capture all the calculations and intermediate values for a single asset
      in one instance (row) per each trading strategy cycle, making
      investigations for alpha model strategies easy

    - Required variables (pair, signal) are =input from `decide_trades()` function in a strategy

    - Optional variables are calculated and filled in the various phases of alpha model processing,
      as the model moves from abstract weightings to actual trade execution and dollar amounts

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
    #: E.g. raw value of the momentum
    signal: float

    #: Stop loss for this position
    #:
    #: Used for the risk management.
    #:
    #: 0.98 means 2% stop loss.
    #:
    #: Set to 0 to disable stop loss.
    stop_loss: float = 0.0

    #: Raw portfolio weight
    #:
    #: Each raw signal is assigned to a weight based on some methodology,
    #: e.g. 1/N where the highest signal gets 50% of portfolio weight.
    raw_weight: float = 0.0

    #: Weight 0...1 so that all portfolio weights sum to 1
    normalised_weight: float = 0.0

    #: Old weight of this pair from the previous cycle.
    #:
    #: If this asset was part of the portfolio at previous :term:`strategy cycle`
    #: then this is the value of the previous cycle weight.
    #: The old weight is always normalised.
    #:
    #: This can be dynamically calculated from the :py:class:`tradeexecutor.state.portfolio.Portfolio` state.
    old_weight: float = 0.0

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
    #: If this is a positive, then we need to make a buy trade for this amount to
    #: reach out target position for this cycle. If negative then we need
    #: to decrease our position.
    position_adjust: USDollarAmount = 0.0

    def __repr__(self):
        return f"Pair: {self.pair.get_ticker()} old weight: {self.old_weight:.4f} old value: {self.old_value:,} new weight: {self.normalised_weight:.4f} new value: {self.position_target:,}"


@dataclass_json
@dataclass(slots=True)
class AlphaModel:
    """Capture alpha model state.

    - A helper class for portfolio construction models and such

    - Converts portfolio weightings to rebalancing trades

    - Supports stop loss and passing through other trade execution parameters

    - Each :term:`strategy cycle` creates its own
      :py:class:`AlphaModel` instance in `decide_trades()` function of the strategy

    - Stores the intermediate results of the calculationsn between raw
      weights and the final investment amount

    - We are serializable as JSON, so we can pass the calculations
      as data around in :py:attr:`tradeexecutor.state.visualisation.Visualisation.calculations`
      and then later visualise alph model progress over time

    """

    #: Timestamp of the strategy cycle for which this alpha model was calculated
    #:
    timestamp: Optional[datetime.datetime]

    #: Pair internal id -> trading signal data
    signals: Dict[PairInternalId, TradingPairSignal]

    #: How much we can afford to invest on this cycle
    investable_equity: Optional[USDollarAmount] = 0.0

    def __init__(self, timestamp: Optional[datetime.datetime | pd.Timestamp] = None):
        self.signals = dict()

        if timestamp is not None:
            if isinstance(timestamp, pd.Timestamp):
                # need to make serializable
                timestamp = timestamp.to_pydatetime()
            assert isinstance(timestamp, datetime.datetime)

        self.timestamp = timestamp

    def iterate_signals(self) -> Iterable[TradingPairSignal]:
        """Iterate over all recorded signals."""
        yield from self.signals.values()

    def get_signal_by_pair_id(self, pair_id: PairInternalId) -> Optional[TradingPairSignal]:
        """Get a trading pair signal instance for one pair."""
        return self.signals.get(pair_id)

    def get_signals_sorted_by_weight(self) -> Iterable[TradingPairSignal]:
        """Get the signals sorted by the weight.

        Return the highest weight first.
        """
        return sorted(self.signals.values(), key=lambda s: s.raw_weight, reverse=True)

    def get_debug_print(self) -> str:
        """Present the alpha model in a format suitable for the console."""
        buf = StringIO()
        print(f"Alpha model for {self.timestamp}, for USD {self.investable_equity:,} investments", file=buf)
        for idx, signal in enumerate(self.get_signals_sorted_by_weight()):
            print(f"   #{idx} {signal}", file=buf)
        return buf.getvalue()

    def set_signal(
            self,
            pair: TradingPairIdentifier,
            alpha: float | np.float32,
            stop_loss: float = 0,
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
        """

        # Don't let Numpy values beyond this point, as
        # they cause havoc in serisaliation
        if isinstance(alpha, np.float32):
            alpha = float(alpha)

        if alpha == 0:
            # Delete so that the pair so that it does not get any further computations
            if pair.internal_id in self.signals:
                del self.signals[pair.internal_id]

        else:
            signal = TradingPairSignal(
                pair=pair,
                signal=alpha,
                stop_loss=stop_loss,
            )
            self.signals[pair.internal_id] = signal

    def set_old_weight(
            self,
            pair: TradingPairIdentifier,
            old_weight: float,
            old_value: USDollarAmount,
            ):
        """Set the weights for the8 current portfolio trading positions before rebalance."""
        if pair.internal_id in self.signals:
            self.signals[pair.internal_id].old_weight = old_weight
        else:
            self.signals[pair.internal_id] = TradingPairSignal(
                pair=pair,
                signal=0,
                old_weight=old_weight,
                old_value=old_value,
            )

    def select_top_signals(self, count: int):
        """Chooses top long signals.

        Modifies :py:attr:`weights` in-place.
        """
        top_signals = heapq.nlargest(count, self.signals.values(), key=lambda s: s.raw_weight)
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
        """Update the old weights of the last strategy cycle to the alpha model."""
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

    def calculate_target_positions(self, investable_equity: USDollarAmount):
        """Calculate individual dollar amount for each position based on its normalised weight."""
        # dollar_values = {pair_id: weight * investable_equity for pair_id, weight in diffs.items()}

        self.investable_equity = investable_equity

        for s in self.iterate_signals():
            s.position_target = s.normalised_weight * investable_equity
            s.position_adjust = s.position_target - s.old_value

    def generate_adjustment_trades_and_update_stop_losses(
            self,
            position_manager: PositionManager,
            min_trade_threshold: USDollarAmount = 10.0,
    ) -> List[TradeExecution]:
        """Generate the trades that will rebalance the portfolio.

        This will generate

        - Sells for the existing assets

        - Buys for new assetes or assets where we want to increase our position

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

            dollar_diff = signal.position_adjust
            value = signal.position_target

            logger.info("Rebalancing %s, old weight: %f, new weight: %f, diff: %f USD",
                        pair,
                        signal.old_weight,
                        signal.normalised_weight,
                        dollar_diff)

            if abs(dollar_diff) < min_trade_threshold:
                logger.info("Not doing anything, value %f below trade threshold %f", value, min_trade_threshold)
            else:
                position_rebalance_trades = position_manager.adjust_position(
                    pair,
                    dollar_diff,
                    signal.normalised_weight,
                    signal.stop_loss,
                )

                assert len(position_rebalance_trades) == 1, "Assuming always on trade for rebalacne"
                logger.info("Adjusting holdings for %s: %s", pair, position_rebalance_trades[0])
                trades += position_rebalance_trades

        trades.sort(key=lambda t: t.get_execution_sort_position())

        # Return all rebalance trades
        return trades
