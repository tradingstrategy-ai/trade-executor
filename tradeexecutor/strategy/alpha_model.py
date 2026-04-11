"""Alpha model and portfolio construction model related logic."""
import datetime
import enum
import heapq
import logging
from collections import Counter
from dataclasses import dataclass, field
from io import StringIO
from types import NoneType
from typing import Callable, Dict, Iterable, List, Literal, Optional

import numpy as np
import pandas as pd
from dataclasses_json import dataclass_json
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.size_risk import SizeRisk
from tradeexecutor.state.trade import TradeExecution, TradeType
from tradeexecutor.state.types import (LeverageMultiplier, PairInternalId,
                                       Percent, USDollarAmount)
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.pandas_trader.position_manager import \
    PositionManager
from tradeexecutor.strategy.redemption import (
    RedemptionCheckResult,
    RedemptionCheckStage,
)
from tradeexecutor.strategy.size_risk_model import SizeRiskModel
from tradeexecutor.strategy.weighting import (Signal, check_normalised_weights,
                                              clip_to_normalised,
                                              normalise_weights,
                                              weight_by_1_slash_n)
from tradingstrategy.types import PrimaryKey

logger = logging.getLogger(__name__)


#: Use in-process running counter id to debug signal
_signal_id_counter = 0

def _get_next_id():
    global _signal_id_counter
    _signal_id_counter += 1
    return _signal_id_counter


def _get_pair_protocol(pair: "TradingPairIdentifier") -> str | None:
    """Resolve the protocol slug for a trading pair.

    - Vault pairs use ``pair.get_vault_protocol()``
    - Exchange account pairs use ``pair.get_exchange_account_protocol()``
    - Returns None for pairs without a named protocol
    """
    if pair.is_vault():
        return pair.get_vault_protocol()
    if pair.is_exchange_account():
        return pair.get_exchange_account_protocol()
    return None


class TradingPairSignalFlags(enum.Enum):
    """Diagnostics flags set on trading signal to understand better the decision making process."""

    #: This signal was capped by the lit liquidity pool risk.
    capped_by_pool_size = "capped_by_pool_size"

    #: This signal was capped by the concentration risk.
    capped_by_concentration = "capped_by_concentration"

    #: No trades were made because the maximum difference in old and new portfolio is not meaningful.
    #: Set on every signak.
    max_adjust_too_small = "max_adjust_too_small"

    #: This pair was not adjusted because the trade to rebalance would be too small dollar wise
    individual_trade_size_too_small = "individual_trade_size_too_small"

    #: Position was not opened/was closed because its weight % in the portfolio is too small
    close_position_weight_limit = "close_position_weight_limit"

    #: This signal's weight was reduced because its chain exceeded the per-chain allocation cap.
    capped_by_chain_allocation = "capped_by_chain_allocation"

    #: This signal's weight was reduced because its protocol exceeded the per-protocol allocation cap.
    #: Risk of having too much allocation into a single vault protocol.
    capped_by_protocol_allocation = "capped_by_protocol_allocation"

    #: This signal led to closing the position (signal went to zero)
    closed = "closed"

    #: This signal was carried forward because the current position cannot be redeemed now.
    cannot_redeem = "cannot_redeem"


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

    """

    #: For which pair is this alpha weight.
    #:
    #: Always the spot pair, the determines the asset price.
    #: For lending protocol leveraged trading this is the underlying trading pair.
    #:
    #: See also :py:attr`leveraged_pair`.
    #:
    pair: TradingPairIdentifier

    #: Raw signal.
    #:
    #: E.g. raw value of the momentum.
    #:
    #: Negative signal indicates short.
    #:
    #: Can be any number between ]-inf, inf[
    #:
    #: Set zero for pairs that are discarded, e.g. due to risk assessment.
    #:
    signal: Signal

    #: Running counter signal ids
    #:
    #: - Useful for internal debugging onyl
    #: - Signal ids are not stable - only for single process debugging
    #:
    signal_id: int = field(default_factory=_get_next_id)

    #: Stop loss for this position.
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
    #: Represents USD allocated to this position.
    #:
    #: Each raw signal is assigned to a weight based on some methodology,
    #: e.g. 1/N where the highest signal gets 50% of portfolio weight.
    #:
    #: Negative signals have positive weight.
    #:
    raw_weight: Percent = 0.0

    #: Weight 0...1 so that all portfolio weights sum to 1
    #:
    #: Represents USD allocated to this position.
    #:
    #: Negative signals have positive weight.
    #:
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

    #: Which trading pair this signal was using before.
    #:
    #: Allows us to switch between spot, leveraged long, leveraged short.
    #:
    old_pair: TradingPairIdentifier | None = None

    #: How many dollars we plan to invest on trading pair.
    #:
    #: Calculated by portfolio total investment equity * normalised weight * price.
    #:
    #: Initially set to None. Can be set either by :py:meth:`AlphaModel.normalise_weights`
    #: or `AlphaModel.calculate_target_positions` depending on the risk model configuration.
    #:
    position_target: USDollarAmount | None = None

    #: How much we are going to increase/decrease the position on this strategy cycle.
    #:
    #: Used when the position increases and we need to know how
    #: many dollars we need to spend to buy more.
    #:
    #: If this is a positive, then we need to make a buy trade for this amount to
    #: reach out target position for this cycle. If negative then we need
    #: to decrease our position.
    #:
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
    position_id: Optional[PrimaryKey] = None

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

    #: For leveraged and spot positions, the pair we use to construct the position.
    #:
    #: This is the leveraged pair derived from :py:attr:`pair`.
    #: Can be leveraged long, leveraged shor or directly the underlying
    #: spot pair.
    #:
    #: This information is not available until the trades have been calculated
    #: in :py:meth:`AlphaModel.generate_rebalance_trades_and_triggers`.
    #:
    #: For spot pairs, this is the pair itself.
    #:
    synthetic_pair: TradingPairIdentifier | None = None

    #: Keep the current marked position value pinned for this cycle.
    carry_forward_position: bool = False

    #: How much leverage we dare to take with this signal
    #:
    #: Unset for spot.
    #:
    leverage: LeverageMultiplier | None = None

    #: Information about the position size risk calculations.
    #:
    position_size_risk: SizeRisk | None = None

    #: Information about the rebalancing trade size risk calculations.
    #:
    trade_size_risk: SizeRisk | None = None

    #: Other data.
    #:
    #: Strategies can use this dict to store any custom
    #: attributes related to this signal.
    #:
    #: All data should be JSON serialisable.
    #:
    other_data: dict = field(default_factory=dict)

    #: Debug flags for this signal, see :py:fuc:`format_signals`
    flags: set[TradingPairSignalFlags] = field(default_factory=set)

    #: Structured redemption diagnostics collected for this cycle.
    redemption_check_results: list[RedemptionCheckResult] = field(default_factory=list)

    def __post_init__(self):
        assert isinstance(self.pair, TradingPairIdentifier)
        if type(self.signal) != float:
            # Convert from numpy.float64
            self.signal = float(self.signal)

        assert self.pair.is_spot() or self.pair.is_vault(), "Signals must be identified by their spot pairs"

        if self.leverage:
            assert type(self.leverage) == float
            assert self.leverage > 0

    def __repr__(self):
        return f"Signal #{self.signal_id} pair:{self.pair.get_ticker()} old weight:{self.old_weight:.4f} old value:{self.old_value:,} raw signal:{self.signal:.4f} normalised weight:{self.normalised_weight:.4f} new value:{self.position_target or 0:,} adjust:{self.position_adjust_usd:,}"

    def set_redemption_check_result(self, result: RedemptionCheckResult) -> None:
        """Store the latest redemption check result for a stage."""
        for idx, existing in enumerate(self.redemption_check_results):
            if existing.stage == result.stage:
                self.redemption_check_results[idx] = result
                return

        self.redemption_check_results.append(result)

    def has_trades(self) -> bool:
        """Did/should this signal cause any trades to be executed.

        - We have trades if we need to rebalance (old weight != new weight)

        - Even if the weight does not change we might still rebalance because the prices change

        - Some adjustments might be too small and then we just ignore any trades
          and have :py:attr:position_adjust_ignored` flag set
        """
        return (self.normalised_weight or self.old_weight) and not self.position_adjust_ignored

    def is_short(self) -> bool:
        """Is the underlying trading activity for this signal to short the asset.

        See also py:attr:`leveraged_pair`.
        """

        assert self.synthetic_pair, "Trades have not been generated yet"

        if not self.synthetic_pair:
            return False

        return self.synthetic_pair.is_short()

    def is_spot(self) -> bool:
        """Is the underlying trading activity for this signal buy spot asset.

        See also py:attr:`is_short`.
        """

        assert self.synthetic_pair, "Trades have not been generated yet"
        return self.synthetic_pair.is_spot()

    def is_new(self) -> bool:
        """The asset did not have any trades (long/short) open on the previous cycle."""
        return self.old_weight == 0

    def is_closing(self) -> bool:
        return self.normalised_weight == 0

    def is_flipping(self) -> bool:
        """On this cycle, are we flipping between long and short.

        - Closing the position to zero does not count as flipping

        - If there was no signal on the previous signal,
          it's not flipping either

        :return:
            True if the pair is going to flip
        """
        if self.normalised_weight == 0:
            return False

        if self.old_pair is None:
            return False

        if self.signal < 0:
            return self.old_pair.is_long() or self.old_pair.is_spot()
        elif self.signal > 0:
            return self.old_pair.is_short()
        else:
            return False

    def get_flip_label(self) -> str:
        """Get flip label"""

        if self.old_pair is None:
            if self.signal > 0:
                return "none -> spot"
            elif self.signal < 0:
                return "none -> short"
            elif self.signal == 0:
                return "spot -> close"
            else:
                return "no flip"

        elif self.old_pair.is_spot() or self.old_pair.is_vault():
            if self.signal < 0:
                return "spot -> short"
            elif self.signal == 0:
                return "spot -> close"
            else:
                return "no flip"

        elif self.old_pair.is_short():
            if self.signal > 0:
                return "short -> spot"
            elif self.signal == 0:
                return "short -> close"
            else:
                return "no flip"

        else:
            raise AssertionError(f"Unsupported")

    def get_tvl(self) -> USDollarAmount:
        """What was TVL used for this signal.

        TVL data we use in calculations in :py:meth:`AlphaModel._normalise_weights_size_risk`.

        Expose for debugging
        """
        if self.position_size_risk:
            return self.position_size_risk.tvl or 0
        return 0


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
    #: Set by :py:meth:`select_top_signals`
    #:
    signals: Dict[PairInternalId, TradingPairSignal] = field(default_factory=dict)

    #: How much we can afford to invest on this cycle.
    #:
    #: See also :py:attr:`accepted_investable_equity`
    #:
    investable_equity: Optional[USDollarAmount] = None

    #: How much we can decide to invest, after calculating position size risk.
    #:
    #: Filled by :py:meth:`normalise_weights` and size risk model is used.
    #:
    #: See also :py:attr:`investable_equity`.
    #:
    #:
    accepted_investable_equity: Optional[USDollarAmount] = None

    #: How much money we left on a table because of the size risk on the positions
    #:
    #: Applies to lit pool size, not to concentration risk.
    #:
    size_risk_discarded_value: Optional[USDollarAmount] = None

    #: The largest position adjust in this cycle.
    #:
    #: What is the largest USD value any individual position would change in this cycle.
    #:
    #: Diagnostics output value.
    #:
    #: If this is below :py:attr:`position_adjust_threshold`, no any rebalance was made.
    #:
    max_position_adjust_usd: Optional[USDollarAmount] = None

    #: What as the position adjust threshold in this cycle.
    #:
    #: Diagnostics output value.
    #:
    #: If this is above :py:attr:`max_position_adjust_usd`, no any rebalance was made.
    position_adjust_threshold_usd: Optional[USDollarAmount] = None

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

    def get_signals_sorted_by_weight(self, reverse=True) -> Iterable[TradingPairSignal]:
        """Get the signals sorted by the weight.

        Return the highest weight first.
        """
        return sorted(self.signals.values(), key=lambda s: s.raw_weight, reverse=reverse)

    def get_debug_print(self) -> str:
        """Present the alpha model in a format suitable for the console."""
        buf = StringIO()
        print(f"Alpha model for {self.timestamp}, for USD {self.investable_equity:,} equity", file=buf)
        for idx, signal in enumerate(self.get_signals_sorted_by_weight(), start=1):
            print(f"   Signal {signal}", file=buf)
        return buf.getvalue()

    def get_allocated_value(self) -> USDollarAmount:
        """How much we have money allocated on signals."""
        return sum(s.position_target for s in self.signals.values())

    def has_any_signal(self) -> bool:
        """For this cycle, should we try to do any trades.

        Any of the signals have non-zero value

        :return:
            True if alpha model should attempt to do some trades on this timestamp/cycle.

            Trades could be still cancelled (zeroed out) by a risk model.
        """
        return any(s for s in self.signals.values() if s.signal != 0)

    def get_signal_count(self) -> int:
        """How many signals we have generated in this cycle."""
        return len([s for s in self.signals.values() if s.signal != 0])

    def has_any_position(self) -> bool:
        """For this cycle, are we going to do any trades.

        Some signals have :py:attr:`TradingSignal.position_target` set after the risk adjustments.

        :return:
            True if alpha model should attempt to do some trades on this timestamp/cycle.

            Trades could be still cancelled (zeroed out) by a risk model.
        """
        return any(s for s in self.signals.values() if s.position_target != 0)

    def is_rebalance_triggered(self) -> bool:
        """Did the consistency of the portfolio change enough in this cycle to do a rebalance.

        - Individual trades might be still considered individually too small to perform
        """
        assert self.max_position_adjust_usd is not None, "Call generate_rebalance_trades_and_triggers() first"
        if self.max_position_adjust_usd == 0:
            # No volatile signals
            return False
        assert self.position_adjust_threshold_usd, "Call generate_rebalance_trades_and_triggers() first"
        return self.max_position_adjust_usd >= self.position_adjust_threshold_usd

    def set_signal(
            self,
            pair: TradingPairIdentifier,
            alpha: float | np.float32,
            stop_loss: Percent | NoneType = None,
            take_profit: Percent | NoneType = None,
            trailing_stop_loss: Percent | NoneType = None,
            leverage: LeverageMultiplier | NoneType = None,
            ):
        """Set trading pair alpha to a value.

        If called repeatatle for the same trading pair,
        remember the last value.

        :param pair:
            Trading pair.

            Always the underlying spot pair.

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

        :param leverage:
            Position leverage.

            Must be set for short and leveraged long.

            If not set assume spot.
        """

        assert pair.is_spot() or pair.is_vault(), f"Signals are tracked by their spot pairs. got {pair}"

        # Don't let Numpy values beyond this point, as
        # they cause havoc in serialisation
        if isinstance(alpha, np.float32):
            alpha = float(alpha)

        if alpha < 0:
            assert leverage is not None, f"Leverage must be set for short, received signal {alpha} for pair {pair.get_human_description(describe_type=True)}"

        if alpha == 0:
            # Zero signal.
            # Delete the pair from the signal mappings so that the pair so that it does not get any further computations
            if pair.internal_id in self.raw_signals:
                del self.raw_signals[pair.internal_id]

        else:
            signal = TradingPairSignal(
                pair=pair,
                signal=alpha,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trailing_stop_loss=trailing_stop_loss,
                leverage=leverage,
            )
            self.raw_signals[pair.internal_id] = signal

    def set_old_weight(
            self,
            pair: TradingPairIdentifier,
            old_weight: float,
            old_value: USDollarAmount,
            old_synthetic_pair: TradingPairIdentifier,
            ):
        """Set the weights for the8 current portfolio trading positions before rebalance.

        :param pair:
            The spot pair we are trading.
        """

        assert pair is not None
        assert pair.is_spot() or pair.is_vault(), f"Expected spot pair, got {pair}"

        if pair.internal_id in self.signals:
            self.signals[pair.internal_id].old_weight = old_weight
            self.signals[pair.internal_id].old_value = old_value
            self.signals[pair.internal_id].old_pair = old_synthetic_pair
        else:
            self.signals[pair.internal_id] = TradingPairSignal(
                pair=pair,
                signal=0,
                old_weight=old_weight,
                old_value=old_value,
                old_pair=old_synthetic_pair,
                position_target=0.0,
            )

    def select_top_signals(
        self,
        count: int,
        threshold=0.0,
    ):
        """Chooses top signals.

        Choose trading pairs to the next rebalance by their signal strength.

        Sets :py:attr:`signals` attribute of the model

        Example:

        .. code-block:: python

            alpha_model.select_top_signals(
                count=5,  # Pick top 5 trading pairs
                threshold=0.01,  # Need at least 1% signal certainty to be eligible
            )

        :param count:
            How many signals to pick.

        :param threshold:
            If the raw signal value is lower than this threshold then don't pick the signal.

            .. note ::

                It's better to filter signals in your `decide_trades()` functinos
                before calling this, as this allows you have to different
                thresholds for long and short signals.

            Inclusive.

            `0.01 = 1%` signal strenght.
        """
        carry_forward_signals = [s for s in self.raw_signals.values() if s.carry_forward_position]
        filtered_signals = [
            s
            for s in self.raw_signals.values()
            if abs(s.signal) >= threshold and not s.carry_forward_position
        ]
        top_signals = heapq.nlargest(count, filtered_signals, key=lambda s: s.signal)
        self.signals = {s.pair.internal_id: s for s in top_signals}
        for signal in carry_forward_signals:
            self.signals[signal.pair.internal_id] = signal

    def _get_allocatable_signals(self) -> list[TradingPairSignal]:
        """Signals that compete for currently deployable capital."""
        return [s for s in self.signals.values() if not s.carry_forward_position]

    def carry_forward_non_redeemable_positions(
        self,
        position_manager: PositionManager,
    ) -> USDollarAmount:
        """Pin current non-redeemable positions at their marked value for this cycle."""
        locked_position_value = 0.0

        for position in position_manager.get_current_portfolio().open_positions.values():
            redemption_result = self._check_redemption_for_position(
                position_manager,
                position,
                stage=RedemptionCheckStage.carry_forward,
            )
            if redemption_result.can_redeem:
                continue

            current_value = position.get_value()
            signal = self.raw_signals.get(position.pair.internal_id)
            if signal is None:
                self.set_signal(position.pair, current_value)
                signal = self.raw_signals[position.pair.internal_id]
            else:
                signal.signal = current_value

            signal.carry_forward_position = True
            signal.position_target = current_value
            self._mark_signal_cannot_redeem(
                signal,
                redemption_result,
                current_value=current_value,
            )
            locked_position_value += current_value

        return locked_position_value

    def _check_redemption_for_position(
        self,
        position_manager: PositionManager,
        position: TradingPosition,
        *,
        stage: RedemptionCheckStage,
    ) -> RedemptionCheckResult:
        """Run the pricing-model redemption check for a position."""
        return position_manager.pricing_model.check_redemption(
            self.timestamp,
            position.pair,
            stage=stage,
            position=position,
        )

    def _mark_signal_cannot_redeem(
        self,
        signal: TradingPairSignal,
        redemption_result: RedemptionCheckResult,
        *,
        current_value: USDollarAmount | None,
    ) -> None:
        """Attach diagnostics and emit one searchable blocked-redemption log line."""
        signal.flags.add(TradingPairSignalFlags.cannot_redeem)
        signal.set_redemption_check_result(redemption_result)

        logger.info(
            "REDEMPTION_DIAGNOSTIC stage=%s pair_ticker=%s vault_address=%s safe_address=%s reason_code=%s current_value=%s position_recorded_lockup_expires_at=%s user_lockup_expires_at=%s max_withdrawable=%s max_redemption=%s message=%s",
            redemption_result.stage.value,
            redemption_result.pair_ticker,
            redemption_result.vault_address,
            redemption_result.safe_address,
            redemption_result.reason_code.value if redemption_result.reason_code else None,
            current_value,
            redemption_result.position_recorded_lockup_expires_at,
            redemption_result.user_lockup_expires_at,
            redemption_result.max_withdrawable,
            redemption_result.max_redemption,
            redemption_result.message,
        )

    def _normalise_weights_simple(
        self,
        max_weight=1.0,
        max_weight_function: Callable[["TradingPairSignal"], float] | None = None,
    ):
        """Normalises position weights between 0 and 1.

        - Simple approach, do not deal with the US dollar size/liquidity risk
        """
        raw_weights = {
            s.pair.internal_id: s.raw_weight
            for s in self._get_allocatable_signals()
        }
        if len(raw_weights) == 0:
            for s in self.signals.values():
                if s.carry_forward_position:
                    s.normalised_weight = 0.0
            return
        normalised = normalise_weights(raw_weights)
        for pair_id, normal_weight in normalised.items():
            s = self.signals[pair_id]
            effective_max = max_weight
            if max_weight_function is not None:
                per_pair = max_weight_function(s)
                if per_pair is not None:
                    effective_max = per_pair
            s.normalised_weight = min(normal_weight, effective_max)

    def _normalise_weights_size_risk(
        self,
        max_weight=1.0,
        investable_equity: USDollarAmount | None = None,
        size_risk_model: SizeRiskModel | None = None,
        max_weight_function: Callable[["TradingPairSignal"], float] | None = None,
    ):
        """Normalises position weights between 0 and 1.

        - Calculate dollar based position sizes and limit them by liquidity if needed
        """

        assert type(max_weight) == float, f"Got {type(max_weight)} instead of float"
        if investable_equity is not None:
            assert type(investable_equity) == float, f"Got {type(investable_equity)} instead of float"

        raw_weights = {
            s.pair.internal_id: s.raw_weight
            for s in self._get_allocatable_signals()
        }
        if len(raw_weights) == 0:
            self.investable_equity = investable_equity
            self.accepted_investable_equity = 0.0
            self.size_risk_discarded_value = 0.0
            for s in self.signals.values():
                if s.carry_forward_position:
                    s.normalised_weight = 0.0
            return

        # First calculate raw normals
        normalised = normalise_weights(raw_weights)

        # We want to iterate from the largest signal to smallest,
        # as we redistribute equity we cannot allocate in larger positions
        normalised = Counter(normalised)

        # For each signal, check if it exceeds
        # US dollar based size risk bsaed on the current market conditions
        total_accetable_investments = 0
        equity_left = investable_equity

        total_missed_investments = 0

        for pair_id, normal_weight in normalised.most_common():

            # NOTE: Here we might have a conflict between given normal weight
            # and size risk, because size risk may overallocate to a position
            # if all positions are size-risked down
            s = self.signals[pair_id]

            assert s.old_weight is not None, f"TradingSignal.old_weight is not available: {s} - remember to call AlphaModel.update_old_weights()"

            assert s.raw_weight >= 0, "_normalise_weights_size_risk(): short or leverage not implemented"

            try:
                effective_max = max_weight
                if max_weight_function is not None:
                    per_pair = max_weight_function(s)
                    if per_pair is not None:
                        effective_max = per_pair
                concentration_capped_normal_weight = min(normal_weight, effective_max)

                if concentration_capped_normal_weight != normal_weight:
                    s.flags.add(TradingPairSignalFlags.capped_by_concentration)

            except TypeError as e:
                raise TypeError(f"Cannot min({normal_weight}, {effective_max})") from e

            asked_position_size = concentration_capped_normal_weight * equity_left
            size_risk = size_risk_model.get_acceptable_size_for_position(
                self.timestamp,
                s.pair,
                asked_position_size
            )

            if size_risk.capped:
                s.flags.add(TradingPairSignalFlags.capped_by_pool_size)

            total_missed_investments += (size_risk.asked_size - size_risk.accepted_size)

            logger.info(
                "Position size risk, pair: %s, asked: %s, accepted: %s, diagnostics: %s",
                s.pair,
                size_risk.asked_size,
                size_risk.accepted_size,
                size_risk.diagnostics_data,
            )

            s.position_size_risk = size_risk
            s.position_target = size_risk.accepted_size
            total_accetable_investments += size_risk.accepted_size

            # Distribute the remaining equity to other positions
            # in the rebalance if we could not fully allocate this one
            equity_left += (size_risk.asked_size - size_risk.accepted_size)

        # Store our risk adjusted sizes
        self.investable_equity = investable_equity
        self.accepted_investable_equity = total_accetable_investments
        self.size_risk_discarded_value = total_missed_investments

        # Recalculate normals based on size-risk adjusted USD values
        clipped_weights = {}
        for pair_id, normal_weight in normalised.items():
            s = self.signals[pair_id]
            clipped_weights[pair_id] = s.position_target / total_accetable_investments

        # Make sure we sum to 1.0, not over,
        # due to floating point issues
        clipped_weights = clip_to_normalised(clipped_weights)

        # Put clipped weights into the model
        for pair_id, normal_weight in clipped_weights.items():
            s = self.signals[pair_id]
            s.normalised_weight = normal_weight

        # Any remaining signal is set to zero
        for s in self.signals.values():
            if s.position_target is None:
                s.position_target = 0.0

    def _normalise_weights_size_risk_positions(
        self,
        max_positions: int,
        max_weight=1.0,
        investable_equity: USDollarAmount | None = None,
        size_risk_model: SizeRiskModel | None = None,
        epsilon_usd=5.0,
        max_weight_function: Callable[["TradingPairSignal"], float] | None = None,
    ):
        """Normalises position weights between 0 and 1.

        - Calculate dollar based position sizes and limit them by liquidity if needed
        """

        assert type(max_weight) == float, f"Got {type(max_weight)} instead of float"
        if investable_equity is not None:
            assert type(investable_equity) == float, f"Got {type(investable_equity)} instead of float"

        raw_weights = {
            s.pair.internal_id: s.raw_weight
            for s in self._get_allocatable_signals()
        }
        if len(raw_weights) == 0:
            self.investable_equity = investable_equity
            self.accepted_investable_equity = 0.0
            self.size_risk_discarded_value = 0.0
            for s in self.signals.values():
                if s.carry_forward_position:
                    s.normalised_weight = 0.0
            return

        remaining_weights = raw_weights.copy()
        equity_left = investable_equity
        total_accetable_investments = 0
        total_missed_investments = 0
        positions_accepted = 0
        invested = 0
        included_pair_ids = set()

        # logging.getLogger().setLevel(logging.INFO)

        logger.info("Total %d positions to consider for size risk adjustment", len(remaining_weights))

        while equity_left - epsilon_usd > 0 and len(remaining_weights) > 0 and positions_accepted < max_positions:
            # First calculate raw normals
            normalised = normalise_weights(remaining_weights)

            # We want to iterate from the largest signal to smallest,
            # as we redistribute equity we cannot allocate in larger positions
            normalised = Counter(normalised)
            # For each signal, check if it exceeds
            # US dollar based size risk bsaed on the current market conditions
            s: TradingPairSignal
            pair_id, weight = normalised.most_common()[0]

            # NOTE: Here we might have a conflict between given normal weight
            # and size risk, because size risk may overallocate to a position
            # if all positions are size-risked down
            s = self.signals[pair_id]

            assert s.old_weight is not None, f"TradingSignal.old_weight is not available: {s} - remember to call AlphaModel.update_old_weights()"
            assert s.raw_weight >= 0, "_normalise_weights_size_risk(): short or leverage not implemented"

            asked_position_size = equity_left * weight
            effective_max = max_weight
            if max_weight_function is not None:
                per_pair = max_weight_function(s)
                if per_pair is not None:
                    effective_max = per_pair
            max_concentrion_capped_size = effective_max * investable_equity

            if asked_position_size > max_concentrion_capped_size:
                asked_position_size = max_concentrion_capped_size
                s.flags.add(TradingPairSignalFlags.capped_by_concentration)

            size_risk = size_risk_model.get_acceptable_size_for_position(
                self.timestamp,
                s.pair,
                asked_position_size
            )

            if size_risk.capped:
                s.flags.add(TradingPairSignalFlags.capped_by_pool_size)

            total_missed_investments += (size_risk.asked_size - size_risk.accepted_size)

            s.position_size_risk = size_risk
            s.position_target = size_risk.accepted_size
            total_accetable_investments += size_risk.accepted_size

            # Distribute the remaining equity to other positions
            # in the rebalance if we could not fully allocate this one
            equity_left -= size_risk.accepted_size
            del remaining_weights[pair_id]

            logger.info(
                "Position size risk, pair: %s, asked: %s, accepted: %s, diagnostics: %s, equity left: %f, invested: %f",
                s.pair.base.token_symbol,
                size_risk.asked_size,
                size_risk.accepted_size,
                size_risk.diagnostics_data,
                equity_left,
                total_accetable_investments
            )
            positions_accepted += 1
            included_pair_ids.add(pair_id)

        # Store our risk adjusted sizes
        self.investable_equity = investable_equity
        self.accepted_investable_equity = total_accetable_investments
        self.size_risk_discarded_value = total_missed_investments

        logger.info(
            "Positions accepted: %d out of %d requested, total accepted %f out of %f",
            positions_accepted,
            max_positions,
            self.accepted_investable_equity,
            self.investable_equity
        )

        # Recalculate normals based on size-risk adjusted USD values
        clipped_weights = {}
        for pair_id in included_pair_ids:
            s = self.signals[pair_id]
            clipped_weights[pair_id] = s.position_target / total_accetable_investments

        # Make sure we sum to 1.0, not over,
        # due to floating point issues
        clipped_weights = clip_to_normalised(clipped_weights)

        # Put clipped weights into the model
        for pair_id, normal_weight in clipped_weights.items():
            s = self.signals[pair_id]
            s.normalised_weight = normal_weight

        # Any remaining signal is set to zero
        for s in self.signals.values():
            if s.position_target is None:
                s.position_target = 0.0

    def _normalise_weights_waterfall(
        self,
        max_positions: int,
        max_weight=1.0,
        investable_equity: USDollarAmount | None = None,
        size_risk_model: SizeRiskModel | None = None,
        epsilon_usd=5.0,
        max_protocol_weight: float | None = None,
        max_weight_function: Callable[["TradingPairSignal"], float] | None = None,
    ):
        """Normalises position weights between 0 and 1.

        - Max allocate to the first entry, and keep going down until everything is allocated
        """

        assert type(max_weight) == float, f"Got {type(max_weight)} instead of float"
        if investable_equity is not None:
            assert type(investable_equity) == float, f"Got {type(investable_equity)} instead of float"

        raw_weights = {
            s.pair.internal_id: s.raw_weight
            for s in self._get_allocatable_signals()
        }
        if len(raw_weights) == 0:
            self.investable_equity = investable_equity
            self.accepted_investable_equity = 0.0
            self.size_risk_discarded_value = 0.0
            for s in self.signals.values():
                if s.carry_forward_position:
                    s.normalised_weight = 0.0
            return

        remaining_weights = Counter(raw_weights)
        equity_left = investable_equity
        total_accetable_investments = 0
        total_missed_investments = 0
        positions_accepted = 0
        invested = 0
        included_pair_ids = set()

        # Per-protocol allocation tracking
        if max_protocol_weight is not None:
            assert 0 < max_protocol_weight <= 1.0, f"max_protocol_weight must be in (0, 1], got {max_protocol_weight}"
            max_protocol_budget = max_protocol_weight * investable_equity
        else:
            max_protocol_budget = None
        protocol_allocated: dict[str, float] = {}

        # logging.getLogger().setLevel(logging.INFO)

        logger.info("Total %d positions to consider for size risk adjustment", len(remaining_weights))

        while equity_left - epsilon_usd > 0 and len(remaining_weights) > 0 and positions_accepted < max_positions:

            # For each signal, check if it exceeds
            # US dollar based size risk bsaed on the current market conditions
            s: TradingPairSignal
            pair_id, weight = remaining_weights.most_common()[0]

            # NOTE: Here we might have a conflict between given normal weight
            # and size risk, because size risk may overallocate to a position
            # if all positions are size-risked down
            s = self.signals[pair_id]

            assert s.old_weight is not None, f"TradingSignal.old_weight is not available: {s} - remember to call AlphaModel.update_old_weights()"
            assert s.raw_weight >= 0, "_normalise_weights_size_risk(): short or leverage not implemented"

            asked_position_size = equity_left
            effective_max = max_weight
            if max_weight_function is not None:
                per_pair = max_weight_function(s)
                if per_pair is not None:
                    effective_max = per_pair
            max_concentrion_capped_size = effective_max * investable_equity

            if asked_position_size > max_concentrion_capped_size:
                asked_position_size = max_concentrion_capped_size
                s.flags.add(TradingPairSignalFlags.capped_by_concentration)

            # Cap by protocol allocation - risk of having too much allocation into a single vault protocol
            if max_protocol_budget is not None:
                protocol = _get_pair_protocol(s.pair)
                if protocol is not None:
                    already_allocated = protocol_allocated.get(protocol, 0.0)
                    remaining_budget = max_protocol_budget - already_allocated

                    if remaining_budget <= epsilon_usd:
                        # Protocol budget exhausted — skip and redistribute to other protocols
                        logger.info(
                            "Skipping %s: protocol %s budget exhausted (allocated %.2f / %.2f)",
                            s.pair.base.token_symbol,
                            protocol,
                            already_allocated,
                            max_protocol_budget,
                        )
                        s.flags.add(TradingPairSignalFlags.capped_by_protocol_allocation)
                        del remaining_weights[pair_id]
                        continue

                    if asked_position_size > remaining_budget:
                        asked_position_size = remaining_budget
                        s.flags.add(TradingPairSignalFlags.capped_by_protocol_allocation)

            size_risk = size_risk_model.get_acceptable_size_for_position(
                self.timestamp,
                s.pair,
                asked_position_size
            )

            if size_risk.capped:
                s.flags.add(TradingPairSignalFlags.capped_by_pool_size)

            total_missed_investments += (size_risk.asked_size - size_risk.accepted_size)

            s.position_size_risk = size_risk
            s.position_target = size_risk.accepted_size
            total_accetable_investments += size_risk.accepted_size

            # Distribute the remaining equity to other positions
            # in the rebalance if we could not fully allocate this one
            equity_left -= size_risk.accepted_size
            del remaining_weights[pair_id]

            # Update protocol budget tracker
            if max_protocol_budget is not None:
                protocol = _get_pair_protocol(s.pair)
                if protocol is not None:
                    protocol_allocated[protocol] = protocol_allocated.get(protocol, 0.0) + size_risk.accepted_size

            logger.info(
                "Position size risk, pair: %s, asked: %s, accepted: %s, diagnostics: %s, equity left: %f, invested: %f",
                s.pair.base.token_symbol,
                size_risk.asked_size,
                size_risk.accepted_size,
                size_risk.diagnostics_data,
                equity_left,
                total_accetable_investments
            )
            positions_accepted += 1
            included_pair_ids.add(pair_id)

        # Store our risk adjusted sizes
        self.investable_equity = investable_equity
        self.accepted_investable_equity = total_accetable_investments
        self.size_risk_discarded_value = total_missed_investments

        logger.info(
            "Positions accepted: %d out of %d requested, total accepted %f out of %f", 
            positions_accepted, 
            max_positions,
            self.accepted_investable_equity,
            self.investable_equity
        )

        # Recalculate normals based on size-risk adjusted USD values
        clipped_weights = {}
        for pair_id in included_pair_ids:
            s = self.signals[pair_id]
            clipped_weights[pair_id] = s.position_target / total_accetable_investments

        # Make sure we sum to 1.0, not over,
        # due to floating point issues
        clipped_weights = clip_to_normalised(clipped_weights)

        # Put clipped weights into the model
        for pair_id, normal_weight in clipped_weights.items():
            s = self.signals[pair_id]
            s.normalised_weight = normal_weight

        # Any remaining signal is set to zero
        for s in self.signals.values():
            if s.position_target is None:
                s.position_target = 0.0                                

    def normalise_weights(
        self,
        max_weight=1.0,
        investable_equity: USDollarAmount | None = None,
        size_risk_model: SizeRiskModel | None = None,
        max_positions: int | None = None,
        waterfall=False,
        max_protocol_weight: float | None = None,
        max_weight_function: Callable[["TradingPairSignal"], float] | None = None,
    ):
        """Normalise weights to 0...1 scale.

        - Apply different risk-adjustments for the normalised positions sizes,
          if given

        - After normalising, we can allocate the positionts `normalised_weight * portfolio equity`.

        - See also :py:mod:`tradeexecutor.strategy.size_risk_model` to set per-pair
          specific US dollar nominated settings for a position size

        :param max_weight:
            Do not allow equity allocation to exceed this % for a single asset.

            Set to 1.0 to no portfolio concentrated risk considered.

            This may happen if you have a portfolio of max assets of 10,
            but due to market conditions there is signal only for 1-2 pairs.
            `max_weight` caps the asset allocation, preventing too concentrated
            positions.

            When ``max_weight_function`` is given, this value is used as a
            fallback only for signals where the function returns ``None``.

        :param size_risk_model:
            Limit position sizes by the current market conditions.

            E.g. Do not allow large positions that exceed available lit liquidity.

        :param investable_equity:
            Only needed if `size_risk_model` is given.

        :param max_protocol_weight:
            Maximum share of the portfolio that can be allocated to positions
            belonging to a single vault protocol (e.g. ``0.40`` means 40 %).

            Only used with ``waterfall=True``. Caps the risk of having too much
            allocation into a single vault protocol (e.g. Hyperliquid).

            Set to ``None`` to disable (default).

        :param max_weight_function:
            Optional callable that returns the per-pair maximum concentration
            weight.  Receives a :py:class:`TradingPairSignal` and should return
            a ``float`` in (0, 1].  When provided, the returned value overrides
            ``max_weight`` for that signal.  Return ``None`` to fall back to the
            global ``max_weight``.

            Example – cap gold-tier vaults at 33 %, others at 10 %::

                def my_max_weight(signal):
                    addr = signal.pair.pool_address.lower()
                    return tier_map.get(addr, 0.10)

                alpha_model.normalise_weights(
                    max_weight=0.10,
                    max_weight_function=my_max_weight,
                    ...
                )
        """

        if not size_risk_model:
            # Easy path
            self._normalise_weights_simple(max_weight, max_weight_function=max_weight_function)
        else:
            # Thinking harder

            if waterfall:
                self._normalise_weights_waterfall(
                    max_positions=max_positions,
                    max_weight=max_weight,
                    investable_equity=investable_equity,
                    size_risk_model=size_risk_model,
                    max_protocol_weight=max_protocol_weight,
                    max_weight_function=max_weight_function,
                )
            elif max_positions is not None:
                # Method 1: weights normalised after TVL-based adjustment
                self._normalise_weights_size_risk_positions(
                    max_positions=max_positions,
                    max_weight=max_weight,
                    investable_equity=investable_equity,
                    size_risk_model=size_risk_model,
                    max_weight_function=max_weight_function,
                )
            else:
                # Method 2: weights normalised before TVL-based adjustment
                self._normalise_weights_size_risk(
                    max_weight=max_weight,
                    size_risk_model=size_risk_model,
                    investable_equity=investable_equity,
                    max_weight_function=max_weight_function,
                )

        # Risk model zeroed out everything so something is likely wrong
        if self.has_any_signal() and not self.has_any_position():
            logger.warning(
                "normalise_weights() at %s had signal, but refuses to have any position target, all positions zeroed out by risk model",
                self.timestamp,
            )
            for s in self.signals.values():
                logger.warning("Signal %s", s)

    def cap_chain_allocation(
        self,
        max_weight_per_chain: float,
    ) -> None:
        """Cap the total normalised weight allocated to any single blockchain.

        Call after :py:meth:`normalise_weights`. Positions on over-allocated
        chains are proportionally scaled down. Freed weight stays as cash
        (no redistribution to other chains).

        :param max_weight_per_chain:
            Maximum share of the portfolio that can be allocated to positions
            on a single chain. E.g. ``0.33`` means 33 %.
        """
        from collections import defaultdict

        if not self.signals:
            return

        assert 0 < max_weight_per_chain <= 1.0, f"max_weight_per_chain must be in (0, 1], got {max_weight_per_chain}"

        # Group signals by chain
        chain_signals: dict[int, list[TradingPairSignal]] = defaultdict(list)
        for s in self.signals.values():
            chain_signals[s.pair.chain_id].append(s)

        # Sum normalised weights per chain
        chain_weights: dict[int, float] = {}
        for chain_id, signals in chain_signals.items():
            chain_weights[chain_id] = sum(s.normalised_weight for s in signals)

        # Scale down over-allocated chains
        any_capped = False
        for chain_id, chain_weight in chain_weights.items():
            if chain_weight > max_weight_per_chain:
                scale = max_weight_per_chain / chain_weight
                for s in chain_signals[chain_id]:
                    s.normalised_weight *= scale
                    if s.position_target is not None:
                        s.position_target *= scale
                    s.flags.add(TradingPairSignalFlags.capped_by_chain_allocation)

                logger.info(
                    "Chain %s allocation capped: %.2f%% -> %.2f%%",
                    chain_id,
                    chain_weight * 100,
                    max_weight_per_chain * 100,
                )
                any_capped = True

        # Update aggregate bookkeeping
        if any_capped and self.accepted_investable_equity is not None:
            old_accepted = self.accepted_investable_equity
            new_accepted = sum(
                s.position_target for s in self.signals.values()
                if s.position_target is not None
            )
            self.accepted_investable_equity = new_accepted
            discarded = old_accepted - new_accepted
            if self.size_risk_discarded_value is not None:
                self.size_risk_discarded_value += discarded
            else:
                self.size_risk_discarded_value = discarded

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

    def update_old_weights(
        self,
        portfolio: Portfolio,
        portfolio_pairs: list[TradingPairIdentifier] | None=None,
        ignore_credit=True,
    ):
        """Update the old weights of the last strategy cycle to the alpha model.

        - Update % of portfolio weight of an asset

        - Update USD portfolio value of an asset

        :param portfolio_pairs:
            Only consider these pairs part of portifolio trading.

            You can use this to exclude credit positions from the portfolio trading.

        :param ignore_credit:
            Automatically ignore credit/vault yield positions.
        """        
        # open positions we consider part of the alpha model: 
        # - exclude non-vault positions
        # - include only portfolio_pairs if given
        alpha_model_positions = []
        for position in portfolio.open_positions.values():
            if ignore_credit and (position.is_credit_supply() or position.is_vault()):
                continue

            if portfolio_pairs and position.pair not in portfolio_pairs:
                continue

            alpha_model_positions.append(position)

        position_values = [
            (position, position.get_value())
            for position in alpha_model_positions
        ]
        total = sum(value for _, value in position_values)

        if alpha_model_positions:
            assert total > 0, f"Portfolio equity is zero, cannot calculate weights: {total}. At {self.timestamp}, positions: {portfolio.open_positions.values()}"

        for position, value in position_values:
            weight = value / total if total > 0 else 0
            self.set_old_weight(
                position.pair.get_pricing_pair(),
                weight,
                value,
                position.pair,
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

    def calculate_target_positions(
        self,
        position_manager: PositionManager,
        investable_equity: USDollarAmount | None = None,
    ):
        """Calculate individual dollar amount for each position based on its normalised weight.

        - Sets the dollar value of the position

        - Adjusts the existing dollar value of positions

        - Map the signal to a trading pair (spot, synthetic short pair, etc.)

        :parma position_manager:
            Used to genenerate TradeExecution instances

        :param investable_equity:
            How much cash we have if we convert the whole portfolio to cash.

            Only needed to give now if size risk model not used with :py:meth:`normalise_weights`.
        """
        # dollar_values = {pair_id: weight * investable_equity for pair_id, weight in diffs.items()}

        # This can be set by normalise_weights() if size risk model is sued
        if self.investable_equity is None:
            self.investable_equity = investable_equity

        for s in self.iterate_signals():
            if s.carry_forward_position:
                s.position_target = s.old_value
                s.synthetic_pair = self.map_pair_for_signal(position_manager, s)
                s.position_adjust_usd = 0.0
                s.position_adjust_quantity = 0.0
                continue

            # Might have been calculated earlier in normalise_weights() with size risk model
            if s.position_target is None:
                assert investable_equity is not None, \
                    f"signal.position_target not set in AlphaModel.normalised_weights(). You need to give AlphaModel.calculate_target_positions(investable_equity)\n" \
                    f"Signal {s} lacks position_target\n"
                s.position_target = s.normalised_weight * investable_equity

            s.synthetic_pair = self.map_pair_for_signal(position_manager, s)

            if s.is_flipping():
                # When we go between short/long/spot
                # we close the previous position and the
                # adjust the full size of the new position
                s.position_adjust_usd = s.position_target
            else:
                #
                s.position_adjust_usd = s.position_target - s.old_value

                if s.position_adjust_usd < 0:
                    # Decreasing positions by selling the token
                    # A lot of options here how to go about this.
                    # We might get some minor position size skew here because fees not included
                    # for these transactions
                    s.position_adjust_quantity = position_manager.estimate_asset_quantity(s.pair, s.position_adjust_usd)
                    assert type(s.position_adjust_quantity) == float

    def map_pair_for_signal(
        self,
        position_manager: PositionManager,
        signal: TradingPairSignal,
    ) -> TradingPairIdentifier:
        """Figure out if we are going to trade spot, leveraged long, leveraged short."""

        underlying = signal.pair

        strategy_universe = position_manager.strategy_universe
        # Spot
        if signal.signal > 0:
            return underlying
        elif signal.signal < 0:
            return strategy_universe.get_shorting_pair(underlying)
        else:
            return underlying

    def _get_current_position_for_signal(
        self,
        position_manager: PositionManager,
        signal: TradingPairSignal,
    ):
        """Resolve the current position and populate signal profit diagnostics."""
        current_position = None
        if signal.old_pair:
            current_position = position_manager.get_current_position_for_pair(signal.old_pair)
            if current_position:
                signal.profit_before_trades = current_position.get_total_profit_usd()
                signal.profit_before_trades_pct = current_position.get_total_profit_percent()
            else:
                signal.profit_before_trades = 0
        return current_position

    def _should_skip_signal_rebalance(
        self,
        signal: TradingPairSignal,
        position_manager: PositionManager,
        frozen_pairs: set,
        individual_rebalance_min_threshold: USDollarAmount,
        sell_rebalance_min_threshold: USDollarAmount | None,
    ) -> bool:
        """Handle early skip conditions before any rebalance trades are built."""
        if position_manager.is_problematic_pair(signal.pair):
            logger.warning("Skipping blacklisted pair: %s", signal.pair)
            return True

        if signal.pair in frozen_pairs:
            logger.warning("Does not generate trades for a pair with frozen positions: %s", signal.pair)
            return True

        if individual_rebalance_min_threshold:
            trade_size = abs(signal.position_adjust_usd)
            if signal.position_adjust_usd < 0:
                threshold = sell_rebalance_min_threshold or individual_rebalance_min_threshold
            else:
                threshold = individual_rebalance_min_threshold

            if trade_size < threshold:
                logger.info("Individual trade size too small, trade size is %s, our threshold %s", trade_size, individual_rebalance_min_threshold)
                signal.flags.add(TradingPairSignalFlags.individual_trade_size_too_small)
                return True

        return False

    def _generate_signal_rebalance_trades(
        self,
        signal: TradingPairSignal,
        position_manager: PositionManager,
        current_position,
        execution_context: ExecutionContext | None,
    ) -> list[TradeExecution]:
        """Generate the concrete rebalance trades for a single signal."""
        position_rebalance_trades = []
        dollar_diff = signal.position_adjust_usd
        quantity_diff = signal.position_adjust_quantity
        value = signal.position_target
        underlying = signal.pair
        synthetic = signal.synthetic_pair

        redemption_result = None
        if current_position and dollar_diff < 0:
            redemption_result = self._check_redemption_for_position(
                position_manager,
                current_position,
                stage=RedemptionCheckStage.sell_rebalance,
            )

        if current_position and dollar_diff < 0 and redemption_result and not redemption_result.can_redeem:
            logger.info(
                "Skipping sell-side rebalance for %s because the position is not redeemable yet",
                current_position.pair,
            )
            signal.position_adjust_ignored = True
            self._mark_signal_cannot_redeem(
                signal,
                redemption_result,
                current_value=current_position.get_value(),
            )
            return position_rebalance_trades

        if signal.normalised_weight < self.close_position_weight_epsilon:
            if current_position:
                logger.info("Closing the position fully: %s", current_position)
                position_rebalance_trades += position_manager.close_position(
                    current_position,
                    TradeType.rebalance,
                    notes=f"Closing position, because the signal weight is below close position weight threshold: {signal}"
                )
                signal.position_id = current_position.position_id
                signal.flags.add(TradingPairSignalFlags.closed)
            else:
                logger.info("Zero signal, but no position to close")
                signal.position_adjust_ignored = True
            signal.flags.add(TradingPairSignalFlags.close_position_weight_limit)
            return position_rebalance_trades

        if signal.is_flipping():
            logger.info("Alpha model signal flipping for %s: %s, new strength %f", signal.pair.get_pricing_pair().base.token_symbol, signal.get_flip_label(), signal.signal)
            old_position = position_manager.get_current_position_for_pair(signal.old_pair)
            if old_position:
                position_rebalance_trades += position_manager.close_position(
                    old_position,
                    TradeType.rebalance,
                    notes=f"Closing because switching between long/short for {signal}"
                )

        if signal.signal < 0:
            leverage = signal.leverage
            assert type(leverage) == float, f"Signal is short, but does not have a leverage multiplier set {signal}"

            if signal.is_flipping() or signal.is_new():
                position_rebalance_trades += position_manager.open_short(
                    underlying,
                    value=value,
                    leverage=leverage,
                    take_profit_pct=signal.take_profit,
                    stop_loss_pct=signal.stop_loss,
                    trailing_stop_loss_pct=signal.trailing_stop_loss,
                    notes="Rebalance opening a new short for signal {signal}",
                )
            else:
                position_rebalance_trades += position_manager.adjust_short(
                    current_position,
                    new_value=value,
                    notes=f"Rebalance existing short for signal: {signal} value: {value}",
                )
        elif signal.leverage is None:
            logger.info("Adjusting spot position")
            notes = ""
            if execution_context and execution_context.mode.is_live_trading():
                notes = f"Resizing position, trade based on signal: {signal} as {self.timestamp}"

            position_rebalance_trades += position_manager.adjust_position(
                synthetic,
                dollar_diff,
                quantity_diff,
                signal.normalised_weight,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                trailing_stop_loss=signal.trailing_stop_loss,
                override_stop_loss=self.override_stop_loss,
                notes=notes,
            )
        else:
            raise NotImplementedError(f"Leveraged long missing w/leverage {signal.leverage}, {signal.get_flip_label()}: {signal}")

        assert len(position_rebalance_trades) >= 1, "Assuming always on trade for rebalance"
        last_trade = position_rebalance_trades[0]
        assert last_trade.position_id
        signal.position_id = last_trade.position_id
        return position_rebalance_trades

    def generate_rebalance_trades_and_triggers(
        self,
        position_manager: PositionManager,
        min_trade_threshold: USDollarAmount = 10.0,
        individual_rebalance_min_threshold: USDollarAmount = 0.0,
        use_spot_for_long=True,
        invidiual_rebalance_min_threshold=None,
        sell_rebalance_min_threshold=None,
        execution_context: ExecutionContext = None,
    ) -> List[TradeExecution]:
        """Generate the trades that will rebalance the portfolio.

        This will generate

        - Sells for the existing assets

        - Buys for new assets or assets where we want to increase our position

        - Set up take profit/stop loss triggers for positions

        Example:

        .. code-block:: python

            trades = alpha_model.generate_rebalance_trades_and_triggers(
                position_manager,
                min_trade_threshold=parameters.rebalance_threshold_usd,  # Don't bother with trades under XXXX USD
                invidiual_rebalance_min_threshold=parameters.individual_rebalance_min_threshold_usd,
                sell_rebalance_min_threshold=parameters.sell_rebalance_min_threshold_usd,
                execution_context=input.execution_context,
            )

        :param position_manager:
            Portfolio of our existing holdings

        :param min_trade_threshold:
            Threshold for too small trades.

            If the notional value of a rebalance trade is smaller than this
            USD amount don't make a trade, but keep whatever
            position we currently we have.

            This is to prevent doing too small trades due to fuzziness in the valuations
            and calculations.

        :param individual_rebalance_min_threshold:
            If an invividual trade value is smaller than this, skip it.

        :param sell_rebalance_min_threshold:
            If an invividual sell trade value is smaller than this, skip it.

            If not given use ``individual_rebalance_min_threshold``.

            Should be lower value, as we want to make sure we do not skip sells that are supposed to release cash for our buys.

        :param use_spot_for_long:
            If we go long a pair, use spot.

            If set False, use leveraged long.

        :param execution_context:
            Needed to tune down print/log/state file clutter when backtesting thousands of trades.

        :return:
            List of trades we need to execute to reach the target portfolio.
            The sells are sorted always before buys.
        """

        assert use_spot_for_long, "Leveraged long unsupported for now"

        if invidiual_rebalance_min_threshold is not None:
            # Legacy typo fix
            individual_rebalance_min_threshold = invidiual_rebalance_min_threshold

        if sell_rebalance_min_threshold is None:
            sell_rebalance_min_threshold = invidiual_rebalance_min_threshold

        # Generate trades
        trades: List[TradeExecution] = []

        logger.info(
            "Generating alpha model rebalances. Before rebalance we have %d positions open. We got %d signals.",
            len(position_manager.state.portfolio.open_positions),
            len(self.signals),
        )

        #
        # Would the portfolio value change enough to justify the rebalance.
        # We calculate this by taking the highest adjust,
        # then either ignore or pass all trades.
        # We cannot ignore individual trades below threshold value,
        # because otherwise buys will fail if their corresponding sells are not triggered.
        #
        #
        # Example scenario where we need to look rebalances as whole:
        #
        # Alpha model for 2020-02-05 00:00:00, for USD 15,171.51501572989 investments
        #    Signal #1 Signal #226 pair:ETH-USDT old weight:0.6115 old value:9,275.537458605508 raw signal:32.2197 normalised weight:0.6444 new value:9,776.423836112543 adjust:500.8863775070349
        #    Signal #2 Signal #225 pair:BTC-USDT old weight:0.3885 old value:5,893.356489256234 raw signal:17.7803 normalised weight:0.3556 new value:5,395.091179617347 adjust:-498.26530963888763

        max_diff = max((abs(s.position_adjust_usd) for s in self.iterate_signals()), default=0)
        self.max_position_adjust_usd = max_diff
        self.position_adjust_threshold_usd = min_trade_threshold
        if max_diff < min_trade_threshold:
            logger.info(
                "Total adjust difference is %f USD, our threshold is %f USD, ignoring all the trades",
                max_diff,
                min_trade_threshold,
            )
            for s in self.iterate_signals():
                s.position_adjust_ignored = True
                s.flags.add(TradingPairSignalFlags.max_adjust_too_small)
            return []

        frozen_pairs = {p.pair for p in position_manager.state.portfolio.frozen_positions.values()}

        for signal in self.iterate_signals():

            dollar_diff = signal.position_adjust_usd

            underlying = signal.pair
            synthetic = signal.synthetic_pair

            # Do backtesting record keeping, so that
            # it is later easier to display alpha model thinking
            current_position = self._get_current_position_for_signal(position_manager, signal)

            logger.info("Rebalancing %s, trading as %s, signal #%d, old position %s, old weight: %f, new weight: %f, size diff: %f USD",
                        underlying.base.token_symbol,
                        synthetic.base.token_symbol,
                        signal.signal_id,
                        current_position and current_position.pair or "-",
                        signal.old_weight,
                        signal.normalised_weight,
                        dollar_diff)

            if self._should_skip_signal_rebalance(
                signal,
                position_manager,
                frozen_pairs,
                individual_rebalance_min_threshold,
                sell_rebalance_min_threshold,
            ):
                continue

            position_rebalance_trades = self._generate_signal_rebalance_trades(
                signal,
                position_manager,
                current_position,
                execution_context,
            )

            if position_rebalance_trades:
                trade_str = ", ".join(t.get_short_label() for t in position_rebalance_trades)
                logger.info("Rebalance trades generated for signal #%d for %s: %s", signal.signal_id, underlying.get_ticker(), trade_str)
            else:
                logger.info("No trades generated for: %s", underlying.get_ticker())

            # Include size risk info for diagnostics
            for t in position_rebalance_trades:
                t.position_size_risk = signal.position_size_risk

            trades += position_rebalance_trades

        trades.sort(key=lambda t: t.get_execution_sort_position())

        # Return all rebalance trades
        return trades

    def get_flag_diagnostics_data(self) -> dict:
        """Get statistics explanation to add to the report of alpha model thinking.

        - See what diagnostics flags :py:class:`TradingPairSignalFlags` we have set on our generated signals

        :return:
            Dict of [flag, count]
        """

        result = Counter()

        for flag in TradingPairSignalFlags:
            for signal in self.signals.values():
                if flag in signal.flags:
                    result[flag] += 1

        return result



def format_signals(
    alpha_model: AlphaModel,
    signal_type: Literal["chosen", "raw"] = "chosen",
    column_mode: Literal["spot", "leveraged"] = "spot",
    sort_key="Signal",
) -> pd.DataFrame:
    """Debug helper used to develop the strategy.

    Print the signal state to the logging output.

    Example:

    .. code-block:: python

        from tradeexecutor.strategy.alpha_model import format_signals

        alpha_model = state.visualisation.discardable_data["alpha_model"]

        print("Chossen signals")
        df = format_signals(alpha_model)
        display(df)

        print("All signals")
        df = format_signals(alpha_model, signal_type="all")
        display(df)

    :param signal_type:
        Show raw signals or only signals that survived filtering.

    :param column_mode:
        What columns include in the resulting table

    :return:
        DataFrame containing a table for signals on this cycle
    """

    data = []

    match signal_type:
        case "chosen":
            signals = alpha_model.signals.values()
        case "all":
            signals = alpha_model.raw_signals.values()
        case _:
            raise NotImplementedError()

    sorted_signals = sorted([s   for s in signals], key=lambda s: s.pair.base.token_symbol)
    # print(f"{timestamp} cycle signals")
    for s in sorted_signals:
        pair = s.pair
        synthetic_pair = s.synthetic_pair.get_ticker() if s.synthetic_pair else "-"
        asked_size = s.position_size_risk.asked_size if s.position_size_risk else "-"
        flags = ", ".join(f.value for f in s.flags)
        old_pair = s.old_pair.get_ticker() if s.old_pair else "-"

        match column_mode:
            case "leveraged":
                # Debug data for Aave shorts
                data.append({
                    "Pair": pair.get_ticker(),
                    "Signal": s.signal,
                    "Asked size": asked_size,
                    "Accepted size": s.position_target,
                    "Value adjust USD": s.position_adjust_usd,
                    "Norm. weights": s.normalised_weight,
                    "Old weight": s.old_weight,
                    "Flipping": s.get_flip_label(),
                    "Trade as": synthetic_pair,
                    "Old pair": old_pair,
                    "Flags": flags
                })
            case "spot":
                # Normal spot market portfolio construction
                data.append({
                    "Pair": pair.get_ticker(),
                    "Signal": s.signal,
                    "Asked size": asked_size,
                    "Accepted size": s.position_target,
                    "Value adjust USD": s.position_adjust_usd,
                    "Weights (raw)": s.raw_weight,
                    "Weights (norm/cap)": s.normalised_weight,
                    "Old weight": s.old_weight,
                    "Flipping": s.get_flip_label(),
                    "TVL": f"{s.get_tvl():.0f}",
                    "Flags": flags
                })
            case _:
                raise NotImplementedError(f"Unknown column mode {column_mode}")

        #print(f"Pair: {pair.get_ticker()}, signal: {s.signal}")

    df = pd.DataFrame(data)
    if len(df) > 0:
        df = df.sort_values(by=[sort_key])
        df = df.set_index("Pair")
    return df


def calculate_required_new_cash(trades: list[TradeExecution]) -> USDollarAmount:
    """How much cash we need to cover the positions to run the rebalance.

    - Calculate the cash needed to open the positions

    - The cash can come from cash in hand,
      credit supply

    - We ignore: The closing of previous positions,
      as these asset sales will release new cash

    :return:
        The amount of cash needed from cash reserves or credit supplies
        to run the rebalance
    """

    assert all([t.is_spot() for t in trades]), "Shorts not supported yet"
    diff = sum([t.get_value() for t in trades])
    return diff
