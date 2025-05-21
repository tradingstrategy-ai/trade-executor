"""Cash yield management."""
import datetime
import logging
from functools import cached_property

import dataclasses
from pprint import pformat

from eth.vm.logic.block import timestamp
from numpy.distutils.conv_template import header
from tabulate import tabulate

from tradeexecutor.state.generic_position import GenericPosition
from tradeexecutor.state.identifier import TradingPairIdentifier, TradingPairKind
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.size_risk import SizeRisk
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.types import USDollarAmount, Percent
from tradeexecutor.strategy.execution_context import ExecutionMode
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.tvl_size_risk import BaseTVLSizeRiskModel

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
    max_concentration: Percent

    #: How much of the total TVL of this vault/reserve we can have in this position.
    #:
    #: E.g. 0.01 means we can only be 1% of the pool.
    #:
    #: If not given, unlimited and we can be the whole pool ourselves.
    #:
    max_pool_participation: Percent | None = None

    def __repr__(self):
        return f"{self.pair.base.token_symbol}: {self.max_concentration:,%}"


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
            assert w.max_concentration > 0

        last_weight = self.weights[-1]
        assert last_weight.max_concentration == 1, f"Last weight slot must get the remaining capital, got: {last_weight}"


@dataclasses.dataclass(slots=True, frozen=True)
class YieldDecisionInput:

    #: Backtesting or live trading
    execution_mode: ExecutionMode

    #: Strategy cycle number
    cycle: int

    #: When we make the decision
    #:
    #: Must be filled for backtesting.
    timestamp: datetime.datetime | None

    #: Total equity of our portfolio
    total_equity: USDollarAmount

    #: Directional trades decided in this cycle
    directional_trades: list[TradeExecution]

    #: The size risk model we use to limit the participation in the pool size
    size_risk_model: BaseTVLSizeRiskModel | None = None


@dataclasses.dataclass(slots=True, frozen=True)
class YieldDecision:

    #: Applied rule we used
    rule: YieldWeightingRule

    #: What is the weight we set for this
    weight: Percent

    #: What is the weight in USD
    amount_usd: USDollarAmount

    #: How much we had in this position prior the calculation
    existing_amount_usd: USDollarAmount | None

    #: Pool participation size risk calculated
    size_risk: SizeRisk | None = None

    @property
    def pair(self) -> TradingPairIdentifier:
        return self.rule.pair



@dataclasses.dataclass(slots=True, frozen=True)
class YieldResult:
    """Track outcome of yield management."""

    trade_cash_diff: USDollarAmount

    available_for_yield: USDollarAmount

    trades: list[TradeExecution]


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

    def gather_current_yield_positions(self) -> dict[TradingPairIdentifier, GenericPosition | None]:
        """Get map of our non-directional positions.

        - Note that this is called in planning phase,
          so there may be pending positions

        :return:
            List of positions that are used to generate yield.
        """

        positions = {self.cash_pair: self.position_manager.state.portfolio.get_default_reserve_position()}
        for weight in self.rules.weights:
            position = self.position_manager.get_current_position_for_pair(weight.pair)
            positions[weight.pair] = position

        return positions

    def generate_rebalance_trades(
        self,
        cycle: int,
        current_yield_positions: dict[TradingPairIdentifier, GenericPosition | None],
        desired_yield_positions: dict[TradingPairIdentifier, YieldDecision],
    ) -> list[TradeExecution]:
        """Create trades to adjust yield positions.

        :param cycle:
            Strategy cycle number for diagnostics

        :param current_yield_positions:
            Where is our cash currently held

        :param desired_yield_positions:
            What are the desired positiosn at the end of this cycle
        """

        trades = []

        trade_output_table: list[dict] = []

        for pair, desired_result in desired_yield_positions.items():
            desired_amount = desired_result.amount_usd
            existing_position = current_yield_positions.get(pair)
            if existing_position:
                existing_amount = existing_position.get_value()
                existing_id = existing_position.position_id
            else:
                existing_amount = 0
                existing_id = None

            dollar_delta = desired_amount - existing_amount
            if existing_position:
                quantity_delta = dollar_delta * existing_position.get_current_price()
            else:
                quantity_delta = None

            if quantity_delta is not None and quantity_delta < 0:
                # Double check for fumbled calculations
                assert existing_position.get_quantity() >= abs(quantity_delta), f"Trying to reduce yield position more than we have. Dollar delta: {dollar_delta}, quantity delta: {quantity_delta}, {existing_position}"

            if abs(dollar_delta) > self.rules.cash_change_tolerance_usd:
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
            else:
                logger.info(
                    "Yield position %s delta %s less than our minimum required change %s",
                    pair,
                    dollar_delta,
                    self.rules.cash_change_tolerance_usd,
                )
                trades = []

            trade_output_table.append({
                "pair": pair.base.token_symbol,
                "existing": existing_amount,
                "existing_pos": existing_id,
                "desired": desired_amount,
                "delta": dollar_delta,
                "trades": "\n".join(str(t) for t in trades),
            })

        trade_output_table_msg = tabulate(
            trade_output_table,
            headers="keys",
            tablefmt="fancy_grid",
        )
        logger.info(
            "Generated yield rebalancing trades at cycle #%d:\n%s",
            cycle,
            trade_output_table_msg,
        )
        return trades

    def calculate_yield_positions(
        self,
        execution_mode: ExecutionMode,
        timestamp: datetime.datetime,
        cycle: int,
        cash_available_for_yield: USDollarAmount,
        current_positions: dict[TradingPairIdentifier, GenericPosition | None],
        size_risk_model: BaseTVLSizeRiskModel | None = None,
        usd_assert_epsilon=0.01,
    ) -> dict[TradingPairIdentifier, YieldDecision]:
        """Calculate cash positions we are allowed to take.

        - Simple first in, first out, fill earier rules to their max weight
        """

        desired_yield_positions: dict[TradingPairIdentifier, YieldDecision] = {}

        left = cash_available_for_yield

        logger.info(
            "Distributing total %f USD to yield positions using %d weighting rules",
            left,
            len(self.rules.weights),
        )

        for rule in self.rules.weights:
            size_risk = None
            if rule.max_concentration < 1:
                amount = cash_available_for_yield * rule.max_concentration

                if size_risk_model:
                    # Limit by pool participation
                    size_risk = size_risk_model.get_acceptable_size_for_position(
                        timestamp,
                        rule.pair,
                        asked_value=amount,
                        check_price=execution_mode.is_backtesting(),
                    )

                    # In backtesting,
                    # size risk can limit the position to zero if the
                    # pool is not yet available
                    amount = size_risk.accepted_size

                    logger.info(
                        "Size risk applied for %s, before %s, after %s",
                        rule.pair,
                        size_risk.asked_size,
                        size_risk.accepted_size,
                    )

                left -= amount
            else:
                # Last position (Aave) gets what ever is left
                amount = left

            if amount > 0:
                weight = amount / cash_available_for_yield

                existing_position = current_positions.get(rule.pair)
                existing_usd = existing_position.get_value() if existing_position else None

                result = YieldDecision(
                    rule=rule,
                    weight=weight,
                    amount_usd=amount,
                    existing_amount_usd=existing_usd,
                    size_risk=size_risk,
                )
                desired_yield_positions[rule.pair] = result

        table = []
        for pair, result in desired_yield_positions.items():
            table.append({
                "Pair": pair.base.token_symbol,
                "Existing amount USD": result.amount_usd,
                "New amount USD": result.amount_usd,
                "Weight %": result.weight * 100,
                "Accepted size risk": result.size_risk.accepted_size if result.size_risk else "-",
                "Size risk TVL": result.size_risk.tvl if result.size_risk else "-",
            })

        table_msg = tabulate(
            table,
            headers="keys",
            tablefmt="rounded_outline"
        )

        logger.info(
            "Desired yield positions for cycle #%d, timestamp %s, cash available %s:\n%s",
            cycle,
            timestamp,
            cash_available_for_yield,
            table_msg,
        )

        total_distributed = sum(result.amount_usd for result in desired_yield_positions.values())
        assert total_distributed <= cash_available_for_yield + usd_assert_epsilon, f"Total distributed {total_distributed} exceed available cash {cash_available_for_yield} USD."
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

    def calculate_yield_management(self, input: YieldDecisionInput) -> YieldResult:
        """Calculate trades for the yield management."""

        # 1. Calculate how much we have currently cash in hand and in yield reserves
        #
        current_positions = self.gather_current_yield_positions()
        current_cash_yielding = sum([position and position.get_value() or 0.0 for k, position in current_positions.items() if k.kind != TradingPairKind.cash])
        current_cash_in_hand = sum([position and position.get_value() or 0.0 for k, position in current_positions.items() if k.kind == TradingPairKind.cash])
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
            execution_mode=input.execution_mode,
            timestamp=input.timestamp,
            cycle=input.cycle,
            cash_available_for_yield=available_for_yield,
            size_risk_model=input.size_risk_model,
            current_positions=current_positions,
        )

        # 5. Calculate rebalance trades for yield positions
        trades = self.generate_rebalance_trades(
            input.cycle,
            current_positions,
            desired_yield_positions,
        )

        return YieldResult(
            trade_cash_diff=trade_cash_diff,
            available_for_yield=available_for_yield,
            trades=trades,
        )
