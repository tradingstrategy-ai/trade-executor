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

from tradingstrategy.types import USDollarAmount

from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradeexecutor.strategy.pricing_model import PricingModel


logger = logging.getLogger(__name__)


class InvalidValuationOutput(Exception):
    """Valuation model did not generate proper price value."""



class ValuationModel(ABC):
    """Revalue a current position.

    TODO: See if this should be moved inside state module, as it is referred by state.revalue_positions.
    
    Used by EthereumPoolRevaluator model (which, in turn, is used by UniswapV2PoolRevaluator and UniswapV3PoolRevaluator)
    """

    def revalue_position(
        self,
        ts: datetime.datetime,
        position: TradingPosition,
    ):
        """Re-value a trading position.

        - Read the spot, collateral and loan values from the pricing model

        - Write the new position valuation data to the state

        - For spot pairs, refresh :py:attr:`TradingPosition.last_token_price` through
          :py:meth:`TradingPosition.revalue_base_asset`.

        - For leveraged positions, refresh both collateral and borrowed asset
        """
        if not position.is_credit_supply():
            old_price = position.last_token_price
            ts, new_price = self(ts, position)
            position.revalue_base_asset(ts, new_price)
            logger.info("Re-valued position base asset %s. Price movement: %f USD -> %f USD",
                        position.pair.base.token_symbol,
                        old_price,
                        new_price,
                        )
            # TODO: Query price for the collateral asset and set it
        else:
            logger.info("No updates to credit supply position pricing: %s", position)

    def revalue_portfolio(
        self,
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

        positions = portfolio.get_open_and_frozen_positions() if revalue_frozen else portfolio.open_positions.values()

        for p in positions:
            try:
                self.revalue_position(ts, p)
            except Exception as e:
                raise InvalidValuationOutput(f"Valuation model failed to output proper price: {self}: {p} -> {e}") from e

    @abstractmethod
    def __call__(self,
                 ts: datetime.datetime,
                 position: TradingPosition) -> Tuple[datetime.datetime, USDollarAmount]:
        """Calculate a new price to an asset.

        TODO: Legacy. Use :py:meth:`revalue_position`

        :param ts:
            When to revalue. Used in backesting. Live strategies may ignore.

        :param position:
            Open position

        :return:
            (revaluation date, position net value) tuple.
            Note that revaluation date may differ from the wantead timestamp if
            there is no data available.
        """
        raise NotImplementedError()


class ValuationModelFactory(Protocol):
    """Creates a valuation method.

    - Valuation method is recreated for each cycle

    - Valuation method takes `PricingModel` as an input

    - Called after the pricing model has been established for the cycle
    """

    def __call__(self, pricing_model: PricingModel) -> ValuationModel:
        pass


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

    if not isinstance(valuation_model, ValuationModel):
        # Legacy call only valuation.
        # Used in legacy tests only.
        for p in state.portfolio.get_open_and_frozen_positions():
            if not p.is_credit_supply():
                ts, price = valuation_model(ts, p.pair)
                p.revalue_base_asset(ts, price)
        return

    valuation_model.revalue_portfolio(
        ts,
        state.portfolio)
