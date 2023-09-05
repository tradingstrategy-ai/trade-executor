"""Valuation models for the portfolio.

Valuation models estimate the value of the portfolio.

This is important for

- Investors understanding if they are profit or loss

- Accounting (taxes)

For the simplest case, we take all open positions and estimate their sell
value at the open market.
"""
import datetime
from abc import abstractmethod
from typing import Protocol, Tuple

from tradingstrategy.types import USDollarAmount

from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradeexecutor.strategy.pricing_model import PricingModel


class InvalidValuationOutput(Exception):
    """Valuation model did not generate proper price value."""



class ValuationModel(Protocol):
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
            spot_pair = position.pair.get_pricing_pair()
            ts, price = self(ts, spot_pair)

            position.revalue_base_asset(ts, price)
            # TODO: Query price for the collateral asset and set it

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
        try:
            for p in portfolio.open_positions.values():
                self.revalue_position(ts, p)

            if revalue_frozen:
                for portfolio in self.frozen_positions.values():
                    self.revalue_position(ts, p)

        except Exception as e:
            raise InvalidValuationOutput(f"Valuation model failed to output proper price: {valuation_model}: {e}") from e

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
        assert isinstance(ts, datetime.datetime)
        pair = position.pair

        assert position.is_long(), "Short not supported"

        quantity = position.get_quantity()
        # Cannot do pricing for zero quantity
        if quantity == 0:
            return ts, 0.0

        price_structure = self.pricing_model.get_sell_price(ts, pair, position)

        return ts, price_structure.price


class ValuationModelFactory(Protocol):
    """Creates a valuation method.

    - Valuation method is recreated for each cycle

    - Valuation method takes `PricingModel` as an input

    - Called after the pricing model has been established for the cycle
    """

    def __call__(self, pricing_model: PricingModel) -> ValuationModel:
        pass


def revalue_state(state: State, ts: datetime.datetime, valuation_model: ValuationModel):
    """Revalue all open positions in the portfolio.

    Reserves are not revalued.
    """
    state.portfolio.revalue_positions(ts, valuation_model)
