import logging
from decimal import Decimal
from typing import Dict, Tuple

import pandas as pd
import numpy as np

from qstrader.portcon.order_sizer.order_sizer import OrderSizer
from tradeexecutor.state.state import State
from tradeexecutor.strategy.pricingmethod import PricingMethod

logger = logging.getLogger(__name__)


class CashBufferedOrderSizer(OrderSizer):
    """
    Creates a target portfolio of quantities for each Asset
    using its provided weight and total equity available in the
    Broker portfolio.

    Includes an optional cash buffer due to the non-fractional amount
    of share/unit sizes. The cash buffer defaults to 5% of the total
    equity, but can be modified.

    Parameters
    ----------

    cash_buffer_percentage : `float`, optional
        The percentage of the portfolio equity to retain in
        cash to avoid generating Orders that exceed account
        equity (assuming no margin available).
    """

    def __init__(
        self,
        state: State,
        pricing_method: PricingMethod,
        cash_buffer_percentage=0.05
    ):
        self.state = state
        self.pricing_method = pricing_method
        self.cash_buffer_percentage = self._check_set_cash_buffer(cash_buffer_percentage)

    def _check_set_cash_buffer(self, cash_buffer_percentage):
        """
        Checks and sets the cash buffer percentage value.

        Parameters
        ----------
        cash_buffer_percentage : `float`
            The percentage of the portfolio equity to retain in
            cash to avoid generating Orders that exceed account
            equity (assuming no margin available).

        Returns
        -------
        `float`
            The cash buffer percentage value.
        """
        if (
            cash_buffer_percentage < 0.0 or cash_buffer_percentage > 1.0
        ):
            raise ValueError(
                'Cash buffer percentage "%s" provided to dollar-weighted '
                'execution algorithm is negative or '
                'exceeds 100%.' % cash_buffer_percentage
            )
        else:
            return cash_buffer_percentage

    def _normalise_weights(self, weights):
        """
        Rescale provided weight values to ensure
        weight vector sums to unity.

        Parameters
        ----------
        weights : `dict{Asset: float}`
            The un-normalised weight vector.

        Returns
        -------
        `dict{Asset: float}`
            The unit sum weight vector.
        """
        if any([weight < 0.0 for weight in weights.values()]):
            raise ValueError(
                'Dollar-weighted cash-buffered order sizing does not support '
                'negative weights. All positions must be long-only.'
            )

        weight_sum = sum(weight for weight in weights.values())

        # If the weights are very close or equal to zero then rescaling
        # is not possible, so simply return weights unscaled
        if np.isclose(weight_sum, 0.0):
            return weights

        return {
            asset: (weight / weight_sum)
            for asset, weight in weights.items()
        }

    def __call__(self, dt: pd.Timestamp, weights: Dict[int, float]) -> Tuple[Dict, Dict]:
        """
        Creates a dollar-weighted cash-buffered target portfolio from the
        provided target weights at a particular timestamp.

        Parameters
        ----------
        dt : `pd.Timestamp`
            The current date-time timestamp.
        weights : `dict{Asset: float}`
            The (potentially unnormalised) target weights.

        """
        total_equity = self.state.portfolio.get_total_equity()

        cash_buffered_total_equity = total_equity * (
            1.0 - self.cash_buffer_percentage
        )

        logger.info(f"Calculating dollar weights for the new portfolio. Total equity {total_equity:,.2f} USD, cash buffered total equity {cash_buffered_total_equity:,.2f} USD")

        # Pre-cost dollar weight
        N = len(weights)
        if N == 0:
            # No forecasts so portfolio remains in cash
            # or is fully liquidated
            return {}

        # Ensure weight vector sums to unity
        normalised_weights = self._normalise_weights(weights)

        target_portfolio = {}
        target_prices = {}

        total_spend = 0

        for asset, weight in sorted(normalised_weights.items()):
            pre_cost_dollar_weight = cash_buffered_total_equity * weight

            # Estimate broker fees for this asset
            est_quantity = 0  # TODO: Needs to be added for IB
            est_costs = 0  # self.broker.fee_model.calc_total_cost(asset, est_quantity, pre_cost_dollar_weight, broker=self.broker)

            # Calculate integral target asset quantity assuming broker costs
            after_cost_dollar_weight = pre_cost_dollar_weight - est_costs

            asset_quantity = 0

            if weight > 0:
                asset_price = self.pricing_method.get_simple_buy_price(dt, asset)

                if after_cost_dollar_weight > 0:
                    if np.isnan(asset_price):
                        raise ValueError(
                            'Asset price for "%s" at timestamp "%s" is Not-a-Number (NaN). '
                            'This can occur if the chosen backtest start date is earlier '
                            'than the first available price for a particular asset. Try '
                            'modifying the backtest start date and re-running.' % (asset, dt)
                        )

                    asset_quantity = after_cost_dollar_weight / asset_price
                    asset_quantity = self.pricing_method.quantize_quantity(asset, asset_quantity)

                # Add to the target portfolio
                target_portfolio[asset] = {"quantity": asset_quantity}
                target_prices[asset] = asset_price
                total_spend += (asset_quantity * Decimal(asset_price)) + est_costs

        logger.info(f"Total new portfolio cost {total_spend:,.2f}")

        return target_portfolio, target_prices
