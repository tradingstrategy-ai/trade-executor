"""Cash yield management."""
import logging
from functools import cached_property

import dataclasses
from pprint import pformat

from tradeexecutor.state.identifier import TradingPairIdentifier, TradingPairKind
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.types import USDollarAmount, Percent
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager


logger = logging.getLogger(__name__)


@dataclasses.dataclass(slots=True, frozen=True)
class YieldWeightingRule:
    """Describe a rule how to optimise yield for spare cash."""

    #: Either Aave pool or a vault.
    #:
    pair: TradingPairIdentifier

    #: How much 0...1 of portfolio we can have in this option.
    #:
    #: Mostly used to mitigate techinical risk. Do not have too large
    #: positions in vaults that are still unproven.
    #:
    #: The weight is as the weight of the cash only, not as the weight of the portfolio total equity.
    #:
    max_weight: float

    def __repr__(self):
        return f"{self.pair.base.token_symbol}: {self.max_weight:,%}"


@dataclasses.dataclass(slots=True, frozen=True)
class YieldRuleset:

    #: How much total equity is allowed to directional + yield position.
    position_allocation: Percent

    #: Buffer amount.
    #
    # Because we do not know the executed reserve quantity released in sell trades, due to slippage,
    # we need to have some safe margin because we might getting a bit less cash from sells
    # we expect.
    buffer_pct: Percent

    #: How much yield position allocation must change before we start to generate new trades.
    cash_change_tolerance_usd: USDollarAmount

    #: Max weights of each yield position
    weights: list[YieldWeightingRule]

    def validate(self):
        """Check all looks good."""
        assert self.buffer_pct > 0
        assert self.cash_change_tolerance_usd > 0
        assert self.position_allocation > 0
        assert len(self.weights) >= 1
        for w in self.weights:
            assert w.pair
            assert w.max_weight > 0

        last_weight = self.weights[-1]
        assert last_weight.max_weight == 1, f"Last weight slot must get the remaining capital, got: {last_weight}"


@dataclasses.dataclass(slots=True, frozen=True)
class YieldDecisionInput:

    #: Total equity of our portfolio
    total_equity: USDollarAmount

    #: Directional trades decided in this cycle
    directional_trades: list[TradeExecution]


@dataclasses.dataclass(slots=True, frozen=True)
class YieldDecisionResult:

    #: Applied rule we used
    rule: YieldWeightingRule

    #: What is the weight we set for this
    weight: Percent

    #: What is the weight in USD
    amount_usd: USDollarAmount

    @property
    def pair(self) -> TradingPairIdentifier:
        return self.rule.pair


class YieldManager:
    """Generate yield on cash.

    - Park cash to profitable positions outside direactional trading
    - Extract cash when needed
    """

    def __init__(
        self,
        position_manager: PositionManager,
        rules: YieldRuleset,
    ):
        """

        :param position_manager:
            PositionManager instance setup within decide_trades()

        """
        rules.validate()
        self.position_manager = position_manager
        self.rules = rules

    @property
    def portfolio(self) -> Portfolio:
        return self.position_manager.state.portfolio

    @cached_property
    def cash_pair(self) -> TradingPairIdentifier:
        reserve_position = self.portfolio.get_default_reserve_position()
        return reserve_position.get_cash_pair()

    def gather_current_yield_positions(self) -> dict[TradingPairIdentifier, USDollarAmount]:
        """Get map of our non-directional positions.

        - Note that this is called in planning phase,
          so there may be pending positions

        :return:
            List of positions that are used to generate yield.
        """

        positions = {self.cash_pair: self.position_manager.get_current_cash()}
        for weight in self.rules.weights:
            position = self.position_manager.get_current_position_for_pair(weight.pair)
            positions[weight.pair] = position.get_value() if position else 0.0

        return positions

    def generate_rebalance_trades(
        self,
        current_yield_positions: dict[TradingPairIdentifier, USDollarAmount],
        desired_yield_positions: dict[TradingPairIdentifier, YieldDecisionResult],
    ) -> list[TradeExecution]:
        """Create trades to adjust yield positions.

        :param current_yield_positions:
            Where is our cash currently held

        :param desired_yield_positions:
            What are the desired positiosn at the end of this cycle
        """

        trades = []
        for pair, desired_result in desired_yield_positions.items():
            desired_amount = desired_result.amount_usd
            existing_position = current_yield_positions.get(pair)
            if existing_position:
                existing_amount = existing_amount.get_value()
            else:
                existing_amount = 0

            dollar_delta = desired_amount - existing_amount
            if existing_position:
                quantity_delta = dollar_delta * existing_position.get_current_price()
            else:
                quantity_delta = None

            notes = f"Adjusting yield management position to: {desired_amount} USD, previously {existing_amount} USD"

            match pair.kind:
                case TradingPairKind.cash:
                    raise AssertionError("Cash should not be in desired positions")
                case TradingPairKind.vault:
                    trades += self.position_manager.adjust_position(
                        pair=pair,
                        dollar_delta=dollar_delta,
                        quantity_delta=quantity_delta,
                        notes=notes,
                        weight=desired_result.weight,
                    )
                case TradingPairKind.credit_supply:
                    # Aave positions are currently always fully closed and then reopenened due to internal limitaiton
                    if existing_position:
                        trades += self.position_manager.close_position(existing_position, notes=notes)
                    trades += self.position_manager.open_credit_supply_position_for_reserves(
                        lending_reserve_identifier=pair,
                        amount=desired_amount,
                        notes=notes,
                    )
                case _:
                    raise NotImplementedError(f"Unsupported yield manager trading pair: {pair}")

        logger.info("Generated yield rebalancing trades:")
        for idx, trade in enumerate(trades):
            logger.info("#%d: %s", idx, trade)

        return trades

    def calculate_yield_positions(
        self,
        cash_available_for_yield: USDollarAmount,
    ) -> dict[TradingPairIdentifier, YieldDecisionResult]:
        """Calculate cash positions we are allowed to take.

        - Simple first in, first out, fill earier rules to their max weight
        """

        desired_yield_positions: dict[TradingPairIdentifier, YieldDecisionResult] = {}

        left = cash_available_for_yield

        logger.info("Distributing total %f USD to yield positions", left)

        for rule in self.rules.weights:
            if rule.max_weight < 1:
                amount = cash_available_for_yield * rule.max_weight
                left -= amount
            else:
                # Last position (Aave) gets what ever is left
                amount = left

            weight = amount / cash_available_for_yield

            result = YieldDecisionResult(
                rule=rule,
                weight=weight,
                amount_usd=amount,
            )
            desired_yield_positions[rule.pair] = result

        for pair, result in desired_yield_positions.items():
            logger.info(
                "Yield position %s: %f USD (%f %%)",
                pair.base.token_symbol,
                result.amount_usd,
                result.weight * 100,
            )

        total_distributed = sum(result.amount_usd for result in desired_yield_positions.values())
        assert total_distributed <= cash_available_for_yield, f"Total distributed {total_distributed} exceed available cash {cash_available_for_yield} USD."
        return desired_yield_positions

    def calculate_cash_needed_to_cover_directional_trades(
        self,
        input: YieldDecisionInput,
        available_cash: USDollarAmount,
        already_deposited: USDollarAmount,
    ):
        """How much cash we need ton this cycle.

        - We have a strategy that uses Aave for USDC credit yield, or similar yield farming service
        - We need to know how much new cash we need to release

        .. note ::

            Only call this after you have set up all the other trades in this cycle.

        See also :py:meth:`manage_credit_flow`

        :return:
            Positive: This much of cash must be released from credit supplied.
            Negative: This much of cash be deposited to Aave at the end of the cycle.
        """

        trades = input.directional_trades
        buffer_pct = self.rules.buffer_pct
        allocation_pct = self.rules.position_allocation
        state = self.position_manager.state

        assert allocation_pct > 0.30, f"allocation_pct might not be correct: {allocation_pct}"

        cash_needed = 0.0
        cash_released = 0.0

        for t in trades:
            assert t.is_spot(), f"Only spot trades supported in calculate_cash_needed(), got: {t}"

            if t.is_buy():
                cash_needed += float(t.planned_reserve)
            else:
                cash_released += float(t.planned_reserve) * (1 - buffer_pct)

        # Keep this amount always in cash.
        # For Lagoon this will enable instant small redemptions.
        nav = state.portfolio.get_net_asset_value()
        target_reserve = nav * (1 - allocation_pct)

        reserve_diff = target_reserve - available_cash
        trade_cash_diff = cash_needed - cash_released
        total_cash_needed = trade_cash_diff + reserve_diff
        flow = total_cash_needed

        # Construct a sophisticated log/error message
        msg = \
            "calculate_cash_needed(): trades: %d nav: %f flow: %f\n" \
            "total cash needed for buys and reserve: %f, cash consumed in trades: %f, cash released in trades: %f\n" \
            "trade cash diff: %f\n" \
            "deposited in Aave: %f\n" \
            "sell buffer pct: %f\n" \
            "target reserve: %f, cash reserves: %f, reserve diff: %f\n" \
            % (
                len(trades),
                nav,
                flow,
                total_cash_needed,
                cash_needed,
                cash_released,
                trade_cash_diff,
                already_deposited,
                buffer_pct,
                target_reserve,
                available_cash,
                reserve_diff,
            )

        logger.info(msg)

        if flow > 0:
            if flow > already_deposited:
                # This may happen in some situation that we need all reserves we have (all in on volatile positions)
                # so we have no Aave credit left and eating into a reservs a bit.
                # Esp. because we have some margin how much cash we will release for sells.
                logger.info(f"Tries to release {flow} from yield management, but we have only {already_deposited}")
                flow = already_deposited

        return trade_cash_diff

    def calculate_yield_management(self, input: YieldDecisionInput):
        """Calculate trades for the yield management."""

        # 1. Calculate how much we have currently cash in hand and in yield reserves
        #
        current_positions = self.gather_current_yield_positions()
        current_cash_yielding = sum([v for k, v in current_positions.items() if k.kind != TradingPairKind.cash])
        current_cash_in_hand = sum([v for k, v in current_positions.items() if k.kind == TradingPairKind.cash])
        all_cash_like = current_cash_yielding + current_cash_in_hand

        assert all_cash_like > 0, f"No cash-like instruments available for yield management:\n{pformat(current_positions)}"

        logger.info(
            "Current cash in hand: %f USD, cash yielding: %f USD, all cash like %f USD",
            current_cash_in_hand,
            current_cash_yielding,
            all_cash_like,
        )

        # 2. Calculate the amount of cash needed/released from directional trades
        trade_cash_diff = self.calculate_cash_needed_to_cover_directional_trades(
            input,
            already_deposited=current_cash_yielding,
            available_cash=current_cash_in_hand,
        )

        #. 3. Calculate how much cash we can allocate for yield
        always_in_cash = input.total_equity * (1 - self.rules.position_allocation)
        available_for_yield = all_cash_like - trade_cash_diff - always_in_cash

        logger.info(
            "Cash requirements calculated. Needed to cover trades/released from trades: %f USD, needed always cash in hand: %f USD, left for yield: %f USD",
            trade_cash_diff,
            always_in_cash,
            available_for_yield,
        )

        assert available_for_yield > 0, f"Yield positions cannot go negative - something wrong with our calcultions? Available for yield: {available_for_yield}"

        # 4. Calculate new yield positions
        desired_yield_positions = self.calculate_yield_positions(
            available_for_yield,
        )

        # 5. Calculate rebalance trades for yield positions
        trades = self.generate_rebalance_trades(
            current_positions,
            desired_yield_positions,
        )

        return trades









