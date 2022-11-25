"""Portfolio construction model translates alpha model to risk managed trades."""
import datetime
import logging
from typing import Optional, Dict, List, Tuple

import pandas as pd

from tradeexecutor.state.state import State, TradeType
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.qstrader.alpha_model import AlphaModel
from tradeexecutor.strategy.qstrader.order_sizer import CashBufferedOrderSizer
from tradingstrategy.universe import Universe

from tradeexecutor.strategy.trading_strategy_universe import translate_trading_pair, TradingStrategyUniverse

logger = logging.getLogger(__name__)


class PortfolioConstructionModel:
    """Portfolio construction model.

    Encapsulates the process of generating a target weight vector
    for a universe of assets, based on input from an AlphaModel,
    a RiskModel and a TransactionCostModel.
    """

    def __init__(
        self,
        universe: Universe,
        state: State,
        order_sizer: CashBufferedOrderSizer,
        optimiser,
        pricing_model: PricingModel,
        reserve_currency: AssetIdentifier,
        alpha_model: AlphaModel,
        risk_model=None,
        cost_model=None
    ):
        assert isinstance(universe, Universe)
        self.universe = universe
        self.state = state
        self.order_sizer = order_sizer
        self.optimiser = optimiser
        self.alpha_model = alpha_model
        self.reserve_currency = reserve_currency
        self.risk_model = risk_model
        self.cost_model = cost_model
        self.pricing_model = pricing_model

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

    def _obtain_current_portfolio(self):
        """
        Query the broker for the current account asset quantities and
        return as a portfolio dictionary.

        Returns
        -------
        `dict{str: dict}`
            Current broker account asset quantities in integral units.
        """
        res = {}
        for id, quantity in self.state.portfolio.get_open_quantities_by_internal_id().items():
            res[id] = {"quantity": quantity}
        return res

    def _generate_rebalance_trades(
        self,
        dt: datetime.datetime,
        target_portfolio,
        current_portfolio,
        target_prices,
        debug_details: dict,
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

        rebalance_trades = []
        new_positions = []

        # Sanity check for the old/new positions with logging
        pos: TradingPosition
        for pos in self.state.portfolio.open_positions.values():
            logger.info("Rebalance, existing position #%d, pool: %s", pos.position_id, pos.pair.pool_address)

        for asset, asset_dict in sorted(rebalance_portfolio.items(), key=lambda x: x[0]):
            quantity = rebalance_portfolio[asset]["quantity"]
            if quantity != 0:

                pandas_pair = self.universe.pairs.get_pair_by_id(asset)
                executor_pair = translate_trading_pair(pandas_pair)

                # For some reason, the price of the asset has disappeared.
                # Do not spend too much time on this, because this code is going
                # to disappear as well.
                price = target_prices.get(asset)
                if price is None:
                    logger.warning(f"Price missing for asset {asset} - prices are {target_prices}")
                    continue

                if isinstance(dt, pd.Timestamp):
                    dt = dt.to_pydatetime()

                position, trade, created = self.state.create_trade(
                    dt,
                    executor_pair,
                    quantity,
                    None,
                    price,
                    TradeType.rebalance,
                    self.reserve_currency,
                    1.0,  # TODO: Harcoded stablecoin USD exchange rate
                )
                logger.info("Created trade, pair:%s, position #%d, trade #%d, new position:%s", position.pair, position.position_id, trade.trade_id, created)
                rebalance_trades.append(trade)
                new_positions.append(position)

        # Sort trades so that sells always go first
        rebalance_trades.sort(key=lambda t: t.get_execution_sort_position())

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

    def get_all_prices(self):
        """Get prices for all asssets."""

    def __call__(self, dt: pd.Timestamp, stats=None, debug_details: Optional[Dict] = None) -> List[TradeExecution]:
        """
        Execute the portfolio construction process at a particular
        provided date-time.

        Use the optional alpha model, risk model and cost model instances
        to create a list of desired weights that are then sent to the
        target weight generator instance to be optimised.
        """

        logger.info("Performing portfolio constructions for %s", dt)

        weights = self.alpha_model(dt, self.universe, self.state, debug_details)

        # Expose internal states to unit tests
        debug_details["alpha_model_weights"] = weights

        logger.info("We have %d alpha model weights", len(weights))

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

        # Calculate target portfolio in notional
        target_portfolio, target_prices = self.order_sizer(dt, weights, debug_details)

        logger.info("We have %d entries in the target portfolio", len(target_portfolio))

        # Obtain current Broker account portfolio
        current_portfolio = self._obtain_current_portfolio()

        # Get prices for existing assets so we have some idea how much they sell for
        for asset_id, asset_data in current_portfolio.items():
            pair = self.pricing_model.get_pair_for_id(asset_id)
            if pair:
                target_prices[asset_id] = self.pricing_model.get_buy_price(dt, pair, None)

        # Expose internal states to unit tests
        debug_details["positions_at_start_of_construction"] = current_portfolio.copy()  # current_portfolio is mutated later

        # Create rebalance trade Orders
        rebalance_trades = self._generate_rebalance_trades(
            dt, target_portfolio, current_portfolio, target_prices, debug_details
        )

        # Expose internal states to unit tests
        debug_details["target_portfolio"] = target_portfolio
        debug_details["target_prices"] = target_prices
        debug_details["rebalance_trades"] = rebalance_trades

        logger.info("Requesting %d rebalance trades", len(rebalance_trades))

        return rebalance_trades
