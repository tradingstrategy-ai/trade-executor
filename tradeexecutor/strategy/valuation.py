"""Valuation models for the portfolio.

Valuation models estimate the value of the portfolio.

This is important for

- Investors understanding if they are profit or loss

- Accounting (taxes)

For the simplest case, we take all open positions and estimate their sell
value at the open market.
"""
import logging
import datetime
from abc import abstractmethod, ABC
from typing import Protocol, Tuple, Callable

from tradeexecutor.state.valuation import ValuationUpdate
from tradingstrategy.types import USDollarAmount

from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradeexecutor.strategy.pricing_model import PricingModel


logger = logging.getLogger(__name__)


class InvalidValuationOutput(Exception):
    """Valuation model did not generate proper price value."""


class ValuationModel(ABC):
    """Revalue an open position.

    Each protocol has its own way to value the position:
    how much free cash we will receive if we close the position.

    - Spot positions are valued at their sell price - fees

    - Loan based positions are valued at the loan NAV

    """

    @abstractmethod
    def __call__(
            self,
            ts: datetime.datetime,
            position: TradingPosition
        ) -> ValuationUpdate:
        """Set the new position value and reference price for an asset.

        - The implementation must check if the position protocol is correct
          for this valuation model.

        - The implementation must update :py:attr:`~tradeexecutor.state.position.TradingPosition.valuation_updates`

        - The implementation must set legacy :py:attr:`~tradeexecutor.state.position.TradingPosition.last_token_price`
          and :py:attr:`~tradeexecutor.state.position.TradingPosition.last_pricing_at` variables.

        :param ts:
            When to revalue. Used in backesting. Live strategies may ignore.

        :param position:
            Open position

        :return:
            (revaluation date, position net value) tuple.
            Note that revaluation date may differ from the wantead timestamp if
            there is no data available.
        """


class ValuationModelFactory(Protocol):
    """Creates a valuation method.

    - Valuation method is recreated for each cycle

    - Valuation method takes `PricingModel` as an input

    - Called after the pricing model has been established for the cycle
    """

    def __call__(self, pricing_model: PricingModel) -> ValuationModel:
        pass


def revalue_portfolio(
    valuation_model: ValuationModel,
    ts: datetime.datetime,
    portfolio: Portfolio,
    revalue_frozen=True,
):
    """Revalue all open positions in the portfolio.

    - Reserves are not revalued
    - Credit supply positions are not revalued

    :param ts:
        Timestamp.

        Strategy cycle time if valuation performed in pre-tick.
        otherwise wall clock time.

    :param valuation_model:
        The model we use to reassign values to the positions

    :param revalue_frozen:
        Revalue frozen positions as well
    """

    logger.info("Portfolio revaluation, timestamp %s, %d open positions", ts, len(portfolio.open_positions))

    positions = portfolio.get_open_and_frozen_positions() if revalue_frozen else portfolio.open_positions.values()

    for position in positions:
        try:
            value_update = valuation_model(ts, position)
            assert isinstance(value_update, ValuationUpdate), f"Expected ValuationUpdate, received {value_update.__class__} from {valuation_model.__class__.__name__}"

            logger.info(
                "Re-valued position #%d, kind %s, base asset %s.\nValue movement: %s USD -> %s USD.\nPrice movement: %s USD -> %s USD.\nUsing model %s.\nValuation done for the timestamp: %s",
                position.position_id,
                position.pair.kind.value,
                position.pair.base.token_symbol,
                value_update.old_value,
                value_update.new_value,
                value_update.old_price,
                value_update.new_price,
                valuation_model.__class__.__name__,
                ts,
            )
        except Exception as e:
            raise InvalidValuationOutput(f"Valuation model failed {valuation_model.__class__.__name__} failed for position {position}\nPosition debug data: {position.get_debug_dump()}") from e



def revalue_state(
        state: State,
        ts: datetime.datetime,
        valuation_model: ValuationModel | Callable):
    """Revalue all open positions in the portfolio.

    - Write new valuations for all positions in the state

    - Reserves are not revalued.

    :param ts:
        Strategy timestamp pre-tick, or wall clock time.

    :param valuation_model:
        Model that pulls out new values for positions.

        For legacy tests, this is a callable.
    """

    revalue_portfolio(valuation_model, ts, state.portfolio)

