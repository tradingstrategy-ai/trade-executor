import datetime
import logging
from typing import Optional, Dict, List, Tuple

import pandas as pd

from qstrader import settings
from qstrader.execution.order import Order
from tradeexecutor.client.translations import translate_trading_pair
from tradeexecutor.state.state import State, AssetIdentifier, TradeType, TradeExecution
from tradingstrategy.universe import Universe

from tradeexecutor.strategy.pricingmethod import PricingMethod

logger = logging.getLogger(__name__)


class PortfolioConstructionModel(object):
    """
    Encapsulates the process of generating a target weight vector
    for a universe of assets, based on input from an AlphaModel,
    a RiskModel and a TransactionCostModel.

    The optimisation process itself is delegated to a TargetWeightGenerator
    instance provided an instantiation.

    Parameters
    ----------
    universe : `Universe`
        The Universe on which to construct a portfolio.
    order_sizer : `OrderSizeGenerator`
        Converts target weights into integral positions.
    optimiser : `PortfolioOptimiser`
        The optimisation mechanism for generating the target weights,
    alpha_model : `AlphaModel`, optional
        The optional alpha/forecasting signal model for Assets in the Universe,
    risk_model : `RiskModel`, optional
        The optional risk model for Assets in the Universe.
    cost_model : `TransactionCostModel`, optional
        The optional transaction cost model for Assets in the Universe.
    data_handler : `DataHandler`, optional
        The optional data handler used within portfolio construction.
    """

    def __init__(
        self,
        universe: Universe,
        state: State,
        order_sizer,
        optimiser,
        pricing_method: PricingMethod,
        reserve_currency: AssetIdentifier,
        alpha_model=None,
        risk_model=None,
        cost_model=None
    ):
        self.universe = universe
        self.state = state
        self.order_sizer = order_sizer
        self.optimiser = optimiser
        self.alpha_model = alpha_model
        self.reserve_currency = reserve_currency
        self.risk_model = risk_model
        self.cost_model = cost_model
        self.pricing_method = pricing_method

    def _obtain_full_asset_list(self, dt):
        """
        Create a union of the Assets in the current Universe
        and those in the Broker Portfolio.

        Parameters
        ----------
        dt : `pd.Timestamp`
            The current time used to obtain Universe Assets.

        Returns
        -------
        `list[int]`
            The sorted full list of Asset symbol strings.
        """
        return self.universe.pairs.get_all_pair_ids()

    def _create_zero_target_weight_vector(self, full_assets):
        """
        Create an initial zero target weight vector for all
        assets in both the Broker Portfolio and current Universe.

        Parameters
        ----------
        full_assets : `list[str]`
            The full list of asset symbols.

        Returns
        -------
        `dict{str: float}`
            The zero target weight vector for all Assets.
        """
        return {asset: 0.0 for asset in full_assets}

    def _create_full_asset_weight_vector(self, zero_weights, optimised_weights):
        """
        Ensure any Assets in the Broker Portfolio are sold out if
        they are not specifically referenced on the optimised weights.

        Parameters
        ----------
        zero_weights : `dict{str: float}`
            The full weight list of assets, all with zero weight.
        optimised_weights : `dict{str: float}`
            The weight list for those assets having a non-zero weight.
            Overrides the zero-weights where keys intersect.

        Returns
        -------
        `dict{str: float}`
            The union of the zero-weights and optimised weights, where the
            optimised weights take precedence.
        """
        return {**zero_weights, **optimised_weights}

    def _generate_target_portfolio(self, dt, weights) -> Tuple[dict, dict]:
        """
        """
        return self.order_sizer(dt, weights)

    def _obtain_current_portfolio(self):
        """
        Query the broker for the current account asset quantities and
        return as a portfolio dictionary.

        Returns
        -------
        `dict{str: dict}`
            Current broker account asset quantities in integral units.
        """
        # return self.broker.get_portfolio_as_dict(self.broker_portfolio_id)
        return self.state.portfolio.get_open_position_quantities_as_dict()

    def _generate_rebalance_orders(
        self,
        dt: datetime.datetime,
        target_portfolio,
        current_portfolio,
        target_prices,
        debug_details: Optional[Dict] = None,
    ) -> List[TradeExecution]:
        """
        Creates an incremental list of rebalancing Orders from the provided
        target and current portfolios.

        Parameters
        ----------
        dt : `pd.Timestamp`
            The current time used to populate the Order instances.
        target_portfolio : `dict{str: dict}`
            Target asset quantities in integral units.
        curent_portfolio : `dict{str: dict}`
            Current (broker) asset quantities in integral units.
        target_prices : `dict{str: decimal}`
            Target asset price for 1 unit
        """

        # Set all assets from the target portfolio that
        # aren't in the current portfolio to zero quantity
        # within the current portfolio
        for asset in target_portfolio:
            if asset not in current_portfolio:
                current_portfolio[asset] = {"quantity": 0}

        # Set all assets from the current portfolio that
        # aren't in the target portfolio (and aren't cash) to
        # zero quantity within the target portfolio
        for asset in current_portfolio:
            if type(asset) != str:
                if asset not in target_portfolio:
                    target_portfolio[asset] = {"quantity": 0}

        # Iterate through the asset list and create the difference
        # quantities required for each asset
        rebalance_portfolio = {}
        for asset in target_portfolio.keys():
            target_qty = target_portfolio[asset]["quantity"]
            current_qty = current_portfolio[asset]["quantity"]
            order_qty = target_qty - current_qty
            rebalance_portfolio[asset] = {"quantity": order_qty}

        # Create the rebalancing Order list from the order portfolio
        # only where quantities are non-zero
        #rebalance_orders = [
        #    Order(dt, asset, rebalance_portfolio[asset]["quantity"], debug_details=debug_details)
        #    for asset, asset_dict in sorted(
        #        rebalance_portfolio.items(), key=lambda x: x[0]
        #    )
        #    if rebalance_portfolio[asset]["quantity"] != 0
        #]

        rebalance_trades = []
        for asset, asset_dict in sorted(rebalance_portfolio.items(), key=lambda x: x[0]):
            quantity = rebalance_portfolio[asset]["quantity"]
            if quantity != 0:

                pandas_pair = self.universe.pairs.get_pair_by_id(asset)
                executor_pair = translate_trading_pair(pandas_pair)
                price = target_prices[asset]

                self.state.create_trade(
                    dt,
                    executor_pair,
                    quantity,
                    price,
                    TradeType.rebalance,
                    self.reserve_currency,
                    1.0,  # TODO: Harcoded stablecoin USD exchange rate
                )

        #rebalance_orders = [
        #    Order(dt, asset, rebalance_portfolio[asset]["quantity"], debug_details=debug_details)
        #    for asset, asset_dict in sorted(
        #        rebalance_portfolio.items(), key=lambda x: x[0]
        #    )
        #    if rebalance_portfolio[asset]["quantity"] != 0
        #]

        return rebalance_trades

    def _create_zero_target_weights_vector(self, dt):
        """
        Determine the Asset Universe at the provided date-time and
        use this to generate a weight vector of zero scalar value
        for each Asset.

        Parameters
        ----------
        dt : `pd.Timestamp`
            The date-time used to determine the Asset list.

        Returns
        -------
        `dict{str: float}`
            The zero-weight vector keyed by Asset symbol.
        """
        assets = self.universe.get_assets(dt)
        return {asset: 0.0 for asset in assets}

    def __call__(self, dt: pd.Timestamp, stats=None, debug_details: Optional[Dict] = None) -> List[TradeExecution]:
        """
        Execute the portfolio construction process at a particular
        provided date-time.

        Use the optional alpha model, risk model and cost model instances
        to create a list of desired weights that are then sent to the
        target weight generator instance to be optimised.

        Parameters
        ----------
        dt : `pd.Timestamp`
            The date-time used to for Asset list determination and
            weight generation.
        stats : `dict`, optional
            An optional statistics dictionary to append values to
            throughout the simulation lifetime.

        Returns
        -------
        `list[Order]`
            The list of rebalancing orders to be sent to Execution.
        """

        weights = self.alpha_model(dt, self.universe, self.state, debug_details)

        # If a risk model is present use it to potentially
        # override the alpha model weights
        if self.risk_model:
            weights = self.risk_model(dt, weights)

        # Run the portfolio optimisation
        optimised_weights = self.optimiser(dt, initial_weights=weights)

        # Ensure any Assets in the Broker Portfolio are sold out if
        # they are not specifically referenced on the optimised weights
        full_assets = self._obtain_full_asset_list(dt)
        full_zero_weights = self._create_zero_target_weight_vector(full_assets)
        full_weights = self._create_full_asset_weight_vector(
            full_zero_weights, optimised_weights
        )
        if settings.PRINT_EVENTS:
            logger.debug(
                "(%s) - target weights: %s" % (dt, full_weights)
            )

        # TODO: Improve this with a full statistics logging handler
        if stats is not None:
            alloc_dict = {'Date': dt}
            alloc_dict.update(full_weights)
            stats['target_allocations'].append(alloc_dict)

        # Calculate target portfolio in notional
        target_portfolio, target_prices = self._generate_target_portfolio(dt, full_weights)

        # Obtain current Broker account portfolio
        current_portfolio = self._obtain_current_portfolio()

        # Create rebalance trade Orders
        rebalance_orders = self._generate_rebalance_orders(
            dt, target_portfolio, current_portfolio, target_prices, debug_details
        )
        # TODO: Implement cost model

        return rebalance_orders
