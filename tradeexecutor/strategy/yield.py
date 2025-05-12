"""Cash yield management."""
import logging
from functools import cached_property

from tradeexecutor.state.identifier import TradingPairIdentifier, TradingPairKind
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.types import USDollarAmount
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager


logger = logging.getLogger(__name__)


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


class YieldManager:
    """Generate yield on cash.

    - Park cash to profitable positions outside direactional trading
    - Extract cash when needed
    """

    def __init__(
        self,
        position_manager: PositionManager, rules: list[YieldWeightingRule],
        cash_change_tolerance_usd: USDollarAmount,
    ):
        """

        :param position_manager:
            PositionManager instance setup within decide_trades()

        :param cash_change_tolerance_usd:
            How much yield position allocation must change before we start to generate new trades.

        """
        self.position_manager = position_manager
        self.rules = rules
        self.cash_change_tolerance_usd = cash_change_tolerance_usd

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
        for rule in self.rules:
            position = self.position_manager.get_current_position_for_pair(rule.pair)
            position[rule.pair] = position.get_value()

        return positions

    def manage_yield_flow(self, flow: USDollarAmount) -> list[TradeExecution]:
        """Create trades to adjust cash/credit supply positions.

        :param flow:
            How much cash yield total amount should change.

            Positive: This much of cash must be released from yield positions to cover opening of directional positions.
            Negative: This much of cash be will be deposited to the yield positions after closing of directional positions.

        """

        assert type(flow) == float, f"Got: {type(flow)} instead of float"


        logger.info(
            "manage_yield_flow(), flow %f, tolerance: %f, rules are: %s",

            flow,
            self.cash_change_tolerance_usd,
            self.rules,
        )

        current_positions = self.gather_current_yield_positions()
        trades = []

        current_cash_yielding = sum([v for k, v in current_positions.items() if k.kind != TradingPairKind.cash])
        current_cash_in_hand = sum([v for k, v in current_positions.items() if k.kind == TradingPairKind.cash])

        logger.info(
            "Current cash in hand: %f USD, cash yielding: %f USD",
            current_cash_in_hand,
            current_cash_yielding,
        )

        if flow == 0:
            logger.info("No credit flow, skipping")
            return []

        if flow > 0:

            # We need to release cash from the reserves.
            # How we do this is
            # - Fully close Aave position at the start of the trades
            # - Deposit money left back at the end of the trades
            # We do this this way so that we claim interest at this kind of cycle,
            # makes accounting easier
            funds_to_deposit_at_cycle_end = current_cash_yielding - flow

            logger.info(
                "Releasing all Aave cash and re-depositing at the end. Flow: %f, Aave deposits at the end: %f, trades: %d",
                flow,
                funds_to_deposit_at_cycle_end,
                len(trades)
            )

            # Don't generate trades if we only see change in the interest.
            # Small interest changes should not trigger rebalance,
            # we only need to trigger if cash is needed for trades.
            if len(trades) == 0 and flow < self.cash_change_tolerance_usd:
                logger.info("No trades and credit supply position within the no action tolerance")
                return []

            trades += self.close_credit_supply_position(
                position,
                notes="Releasing all funds to trade on the cycle"
            )

            trades += self.open_credit_supply_position_for_reserves(
                funds_to_deposit_at_cycle_end,
                flags={TradeFlag.ignore_open},
                notes="Redepositing remaining funds at the end of cycle"
            )
        else:
            # The credit supply position increaes in this cycle
            if position is None:
                logger.info("Creating initial credit supply position")
                assert flow < 0, f"Initial credit flow must be supply, got {flow}"

                trades += self.open_credit_supply_position_for_reserves(
                    -flow,
                    notes="Initial supply"
                )
            else:
                logger.info("Increasing the existing credit supply position")
                # Add funds to the existing credit position
                trades += self.adjust_credit_supply_position(
                    position,
                    delta=-flow,
                    notes="Extra cash, increasing credit supply"
                )

        return trades


