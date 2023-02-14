"""Alpha model and portfolio construction model related logic."""
import heapq
import logging

from dataclasses import dataclass
from typing import Optional, TypeAlias, Dict, Iterable, List

from dataclasses_json import dataclass_json

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.types import PairInternalId, USDollarAmount
from tradeexecutor.strategy.weights import normalise_weights

from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.weighting import weight_by_1_slash_n, check_normalised_weights


logger = logging.getLogger(__name__)


@dataclass_json
@dataclass(slots=True)
class TradingPairSignal:
    """Present one asset in alpha model weighting.

    - Required variables are needed as an input from `decide_trades()` function in a strategy

    - Optional variables are calculated in the various phases of alpha model processing

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
    stop_loss: float = 0

    #: Raw portfolio weight
    #:
    #: Each raw signal is assigned to a weight based on some methodology,
    #: e.g. 1/N where the highest signal gets 50% of portfolio weight.
    raw_weight: Optional[float] = 0

    #: Weight 0...1 so that all portfolio weights sum to 1
    normalised_weight: Optional[float] = 0

    #: Old weight of this pair from the previous cycle.
    #:
    #: If this asset was part of the portfolio at previous :term:`strategy cycle`
    #: then this is the value of the previous cycle weight.
    #: The old weight is always normalised.
    #:
    #: This can be dynamically calculated from the :py:class:`tradeexecutor.state.portfolio.Portfolio` state.
    old_weight: Optional[float] = None

    #: Old US Dollar value of this value from the previous cycle.
    #:
    #: If this asset was part of the portfolio at previous :term:`strategy cycle`
    #: then this is the value of the previous cycle weight.
    #:
    old_value: Optional[USDollarAmount] = None

    #: How many dollars we plan to invest on trading pair.
    #:
    #: Calculated by portfolio total investment equity * normalised weight * price.
    position_target: Optional[USDollarAmount] = None

    #: How much we are going to increase/decrease the position on this strategy cycle.
    #:
    #: If this is a positive, then we need to make a buy trade for this amount to
    #: reach out target position for this cycle. If negative then we need
    #: to decrease our position.
    position_adjust: Optional[USDollarAmount] = None



#: Map of different weights of trading pairs for alpha model.
#:
#: If there is no entry present assume its weight is zero.
#:
AlphaSignals: TypeAlias = Dict[PairInternalId, TradingPairSignal]



@dataclass_json
@dataclass(slots=True)
class AlphaModel:
    """Capture alpha model in a debuggable object.

    - Supports stop loss and passing through other trade execution parameters

    - Each :term:`strategy cycle` creates its own
      alpha model instance

    - Stores the intermediate results of the calculationsn between raw
      weights and the final investment amount

    - We are serializable as JSON, so we can pass
      alpha model as it debug data around in :py:attr:`tradeexecutor.state.visualisation.Visualisation.calculations`.

    """

    #: Pair internal id -> trading signal data
    signals: AlphaSignals

    def __init__(self):
        self.signals = dict()

        #: Pairs we have touched on this cycle
        self.pairs = set()

    def iterate_signals(self) -> Iterable[TradingPairSignal]:
        """Iterate over all recorded signals."""
        yield from self.signals.values()

    def get_signal_by_pair_id(self, pair_id: PairInternalId) -> Optional[TradingPairSignal]:
        """Get a trading pair signal instance for one pair."""
        return self.signals.get(pair_id)

    def set_signal(
            self,
            pair: TradingPairIdentifier,
            alpha: float,
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

        if alpha == 0:
            # Delete so that the pair so that it does not get any further computations
            if pair.internal_id in self.weight:
                del self.signals[pair.internal_id]

        else:
            signal = TradingPairSignal(
                pair=pair,
                signal=alpha,
                stop_loss=stop_loss,
            )
            self.signals[pair.internal_id] = signal

        if pair not in self.pairs:
            self.pairs.add(pair)

    def set_old_weight(
            self,
            pair_id: PairInternalId,
            old_weight: float,
            old_value: USDollarAmount,
            ):
        """Set the weights for the8 current portfolio trading positions before rebalance."""
        for pair, w in weights.items():
            if pair.internal_id in self.signals:
                self.signals[pair.internal_id].old_weight = old_weight
            else:
                self.signals[pair.internal_id] = TradingPairSignal(
                    pair=pair,
                    signal=0,
                    old_weight=old_weight
                )

    def select_top_signals(self, count: int):
        """Chooses top long signals.

        Modifies :py:attr:`weights` in-place.
        """
        top_signals = heapq.nlargest(count, self.values(), key=lambda s: s.raw_weight)
        self.signals = {s.pair.internal_id: s for s in top_signals}

    def normalise_weights(self):
        raw_weights = {s.pair.internal_id: s.raw_weight for s in self.signals.values()}
        normalised = normalise_weights(raw_weights)
        for pair_id, normal_weight in normalised.items():
            self.signals[pair_id].normalised_weight = normal_weight

    def assign_weights(self, method=weight_by_1_slash_n):
        """Convert raw signals to their portfolio weight counterparts.

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
                position.pair.internal_id,
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

    def calculate_target_positions_and_adjusts(self, investable_equity: USDollarAmount):
        """Calculate individual dollar amount for each position based on its normalised weight."""
        # dollar_values = {pair_id: weight * investable_equity for pair_id, weight in diffs.items()}
        for s in self.iterate_signals():
            s.position_target_value = s.normalised_weight * investable_equity
            s.position_adjust = s.position_target_value - s.old_value

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
            pair = position_manager.get_trading_pair(signal.pair.pair_id)
            signal = self.get_signal_by_pair_id(signal.pair.pair_id)

            dollar_diff = signal.position_adjust

            logger.info("Rebalancing %s, old weight: %f, new weight: %f, diff: %f USD",
                        pair,
                        signal.old_weight
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
                ),

                assert len(position_rebalance_trades) == 1, "Assuming always on trade for rebalacne"
                logger.info("Adjusting holdings for %s: %s", pair, position_rebalance_trades[0])
                trades += position_rebalance_trades

        trades.sort(key=lambda t: t.get_execution_sort_position())

        # Return all rebalance trades
        return trades




