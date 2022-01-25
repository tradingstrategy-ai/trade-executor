import queue
import logging
from collections import defaultdict
from typing import Dict, Optional

import numpy as np

from qstrader import settings
from qstrader.broker.broker import Broker
from qstrader.broker.fee_model.fee_model import FeeModel
from qstrader.broker.portfolio.portfolio import Portfolio
from qstrader.broker.transaction.transaction import Transaction
from qstrader.broker.fee_model.zero_fee_model import ZeroFeeModel


logger = logging.getLogger(__name__)


class DEXBroker(Broker):
    """A class to handles live trading on DEXes.

    The default commission/fee model is a ZeroFeeModel
    that charges no commission or tax (such as stamp duty).

    Parameters
    ----------
    start_dt : `pd.Timestamp`
        The starting datetime of the account
    exchange : `Exchange`
        Used to determine whether the simulated exchange venue
        is open, in order to determine if orders can be executed.
    data_handler : `DataHandler`
        The data handler used to obtain latest asset prices.
    account_id : `str`, optional
        The account ID for the brokerage account.
    base_currency : `str`, optional
        The currency denomination of the brokerage account.
    initial_funds : `float`, optional
        An initial amount of cash to add to the broker account.
    fee_model : `FeeModel`, optional
        The commission/fee model used to simulate fees/taxes.
        Defaults to the ZeroFeeModel.
    slippage_model : `SlippageModel`, optional
        The model used to simulate trade slippage.
    market_impact_model : `MarketImpactModel`, optional
        The model used to simulate market impact of trading.
    """

    def __init__(
        self,
        data_handler,
        account_id=None,
        base_currency="USD",
        fee_model=ZeroFeeModel(),
        slippage_model=None,
        market_impact_model=None
    ):
        self.data_handler = data_handler
        self.account_id = account_id

        self.base_currency = self._set_base_currency(base_currency)
        self.initial_funds = self._set_initial_funds(initial_funds)
        self.fee_model = self._set_fee_model(fee_model)
        self.slippage_model = None  # TODO: Implement
        self.market_impact_model = None  # TODO: Implement

        # currency name -> blaance
        self.cash_balances: Dict[str, float]
        self.cash_balances = self._set_cash_balances()

        # portfolio id -> Portfolio
        self.portfolios: Dict[str, Portfolio]
        self.portfolios = self._set_initial_portfolios()

        # portfolio id -> Queue of Orders
        self.open_orders: Dict[str, queue.Queue]
        self.open_orders = self._set_initial_open_orders()

    def _set_base_currency(self, base_currency):
        """
        Check and set the base currency from a list of
        allowed currencies. Raise ValueError if the
        currency is currently not supported by QSTrader.

        Parameters
        ----------
        base_currency : `str`
            The base currency string.

        Returns
        -------
        `str`
            The base currency string.
        """
        if base_currency not in settings.SUPPORTED['CURRENCIES']:
            raise ValueError(
                "Currency '%s' is not supported by QSTrader. Could not "
                "set the base currency in the SimulatedBroker "
                "entity." % base_currency
            )
        else:
            return base_currency

    def _set_fee_model(self, fee_model):
        """
        Check and set the FeeModel instance for the broker.
        The class default is no commission (ZeroFeeModel).

        Parameters
        ----------
        fee_model : `FeeModel` (class)
            The commission/fee model class provided to the Broker.

        Returns
        -------
        `FeeModel` (instance)
            The instantiated FeeModel class.
        """
        if issubclass(fee_model.__class__, FeeModel):
            return fee_model
        else:
            raise TypeError(
                "Provided fee model '%s' in SimulatedBroker is not a "
                "FeeModel subclass, so could not create the "
                "Broker entity." % fee_model.__class__
            )

    def _set_initial_portfolios(self) -> Dict[str, Portfolio]:
        """
        Set the appropriate initial portfolios dictionary.

        Returns
        -------
        `dict`
            The empty initial portfolio dictionary.
        """
        return {}

    def _set_initial_open_orders(self) -> Dict[str, queue.Queue]:
        """
        Set the appropriate initial open orders dictionary.

        Returns
        -------
        `dict`
            The empty initial open orders dictionary.
        """
        return {}

    def get_account_cash_balance(self, currency=None):
        """
        Retrieve the cash dictionary of the account, or
        if a currency is provided, the cash value itself.
        Raises a ValueError if the currency is not
        found within the currency cash dictionary.

        Parameters
        ----------
        currency : `str`, optional
            The currency string to obtain the cash balance for.
        """
        if currency is None:
            return self.cash_balances
        if currency not in self.cash_balances.keys():
            raise ValueError(
                "Currency of type '%s' is not found within the "
                "broker cash master accounts. Could not retrieve "
                "cash balance." % currency
            )
        return self.cash_balances[currency]

    def get_account_total_market_value(self):
        """
        Retrieve the total market value of the account, across
        each portfolio.

        Returns
        -------
        `dict`
            The dictionary of each portfolio's total market value.
        """
        tmv_dict = {}
        master_tmv = 0.0
        for portfolio in self.portfolios.values():
            pmv = self.get_portfolio_market_value(
                portfolio.portfolio_id
            )
            tmv_dict[portfolio.portfolio_id] = pmv
            master_tmv += pmv
        tmv_dict["master"] = master_tmv
        return tmv_dict

    def get_account_total_equity(self):
        """
        Retrieve the total equity of the account, across
        each portfolio.

        Returns
        -------
        `dict`
            The dictionary of each portfolio's total equity.
        """
        equity_dict = {}
        master_equity = 0.0
        for portfolio in self.portfolios.values():
            port_equity = self.get_portfolio_total_equity(
                portfolio.portfolio_id
            )
            equity_dict[portfolio.portfolio_id] = port_equity
            master_equity += port_equity
        equity_dict["master"] = master_equity
        return equity_dict

    def create_portfolio(self, portfolio_id, name=None):
        """
        Create a new sub-portfolio with ID 'portfolio_id' and
        an optional name given by 'name'.

        Parameters
        ----------
        portfolio_id : `str`
            The portfolio ID string.
        name : `str`, optional
            The optional name string of the portfolio.
        """
        portfolio_id_str = str(portfolio_id)
        if portfolio_id_str in self.portfolios.keys():
            raise ValueError(
                "Portfolio with ID '%s' already exists. Cannot create "
                "second portfolio with the same ID." % portfolio_id_str
            )
        else:
            p = Portfolio(
                self.current_dt,
                currency=self.base_currency,
                portfolio_id=portfolio_id_str,
                name=name
            )
            self.portfolios[portfolio_id_str] = p
            self.open_orders[portfolio_id_str] = queue.Queue()
            if settings.PRINT_EVENTS:
                print(
                    '(%s) - portfolio creation: Portfolio "%s" created at broker "%s"' % (
                        self.current_dt, portfolio_id_str, self.account_id
                    )
                )

    def list_all_portfolios(self):
        """
        List all of the sub-portfolios associated with this
        broker account in order of portfolio ID.

        Returns
        -------
        `list`
            The list of portfolios associated with the broker account.
        """
        if self.portfolios == {}:
            return []
        return sorted(
            list(self.portfolios.values()),
            key=lambda port: port.portfolio_id
        )

    def subscribe_funds_to_portfolio(self, portfolio_id, amount):
        """
        Subscribe funds to a particular sub-portfolio, assuming
        it exists and the cash amount is positive. Otherwise raise
        a ValueError.

        Parameters
        ----------
        portfolio_id : `str`
            The portfolio ID string.
        amount : `float`
            The amount of cash to subscribe to the portfolio.
        """
        if amount < 0.0:
            raise ValueError(
                "Cannot add negative amount: "
                "%0.2f to a portfolio account." % amount
            )
        if portfolio_id not in self.portfolios.keys():
            raise KeyError(
                "Portfolio with ID '%s' does not exist. Cannot subscribe "
                "funds to a non-existent portfolio." % portfolio_id
            )
        if amount > self.cash_balances[self.base_currency]:
            raise ValueError(
                "Not enough cash in the broker master account to "
                "fund portfolio '%s'. %0.2f subscription amount exceeds "
                "current broker account cash balance of %0.2f." % (
                    portfolio_id, amount,
                    self.cash_balances[self.base_currency]
                )
            )
        self.portfolios[portfolio_id].subscribe_funds(self.current_dt, amount)
        self.cash_balances[self.base_currency] -= amount
        if settings.PRINT_EVENTS:
            print(
                '(%s) - subscription: %0.2f subscribed to portfolio "%s"' % (
                    self.current_dt, amount, portfolio_id
                )
            )

    def withdraw_funds_from_portfolio(self, portfolio_id, amount):
        """
        Withdraw funds from a particular sub-portfolio, assuming
        it exists, the cash amount is positive and there is
        sufficient remaining cash in the sub-portfolio to
        withdraw. Otherwise raise a ValueError.

        Parameters
        ----------
        portfolio_id : `str`
            The portfolio ID string.
        amount : `float`
            The amount of cash to withdraw from the portfolio.
        """
        if amount < 0.0:
            raise ValueError(
                "Cannot withdraw negative amount: "
                "%0.2f from a portfolio account." % amount
            )
        if portfolio_id not in self.portfolios.keys():
            raise KeyError(
                "Portfolio with ID '%s' does not exist. Cannot "
                "withdraw funds from a non-existent "
                "portfolio. " % portfolio_id
            )
        if amount > self.portfolios[portfolio_id].cash:
            raise ValueError(
                "Not enough cash in portfolio '%s' to withdraw "
                "into brokerage master account. Withdrawal "
                "amount %0.2f exceeds current portfolio cash "
                "balance of %0.2f." % (
                    portfolio_id, amount,
                    self.portfolios[portfolio_id].cash
                )
            )
        self.portfolios[portfolio_id].withdraw_funds(
            self.current_dt, amount
        )
        self.cash_balances[self.base_currency] += amount
        if settings.PRINT_EVENTS:
            print(
                '(%s) - withdrawal: %0.2f withdrawn from portfolio "%s"' % (
                    self.current_dt, amount, portfolio_id
                )
            )

    def get_portfolio_cash_balance(self, portfolio_id):
        """
        Retrieve the cash balance of a sub-portfolio, if
        it exists. Otherwise raise a ValueError.

        Parameters
        ----------
        portfolio_id : `str`
            The portfolio ID string.

        Returns
        -------
        `float`
            The cash balance of the portfolio.
        """
        if portfolio_id not in self.portfolios.keys():
            raise ValueError(
                "Portfolio with ID '%s' does not exist. Cannot "
                "retrieve cash balance for non-existent "
                "portfolio." % portfolio_id
            )
        return self.portfolios[portfolio_id].cash

    def get_total_portfolio_cash_balances(self) -> float:
        cash_balances = [p.cash for p in self.portfolios.values()]
        return sum(cash_balances)

    def get_portfolio_total_market_value(self, portfolio_id):
        """
        Returns the current total market value of a Portfolio
        with ID 'portfolio_id'.

        Parameters
        ----------
        portfolio_id : `str`
            The portfolio ID string.

        Returns
        -------
        `float`
            The total market value of the portfolio.
        """
        if portfolio_id not in self.portfolios.keys():
            raise KeyError(
                "Portfolio with ID '%s' does not exist. "
                "Cannot return total market value for a "
                "non-existent portfolio." % portfolio_id
            )
        return self.portfolios[portfolio_id].total_market_value

    def get_portfolio_total_equity(self, portfolio_id):
        """
        Returns the current total equity of a Portfolio
        with ID 'portfolio_id'.

        Parameters
        ----------
        portfolio_id : `str`
            The portfolio ID string.

        Returns
        -------
        `float`
            The total equity of the portfolio.
        """
        if portfolio_id not in self.portfolios.keys():
            raise KeyError(
                "Portfolio with ID '%s' does not exist. "
                "Cannot return total equity for a "
                "non-existent portfolio." % portfolio_id
            )
        return self.portfolios[portfolio_id].total_equity

    def get_portfolio_as_dict(self, portfolio_id):
        """
        Return a particular portfolio with ID 'portolio_id' as
        a dictionary with Asset symbol strings as keys, with various
        attributes as sub-dictionaries.

        Parameters
        ----------
        portfolio_id : `str`
            The portfolio ID string.

        Returns
        -------
        `dict{str}`
            The portfolio representation of Assets as a dictionary.
        """
        if portfolio_id not in self.portfolios.keys():
            raise KeyError(
                "Cannot return portfolio as dictionary since "
                "portfolio with ID '%s' does not exist." % portfolio_id
            )
        return self.portfolios[portfolio_id].portfolio_to_dict()

    def _execute_order(self, dt, portfolio_id, order, debug_details: Dict):
        """
        For a given portfolio ID string, create a Transaction instance from
        the provided Order and ensure the Portfolio is appropriately updated
        with the new information.

        Parameters
        ----------
        dt : `pd.Timestamp`
            The current timestamp.
        portfolio_id : `str`
            The portfolio ID string.
        order : `Order`
            The Order instance to create the Transaction for.
        """
        # Obtain a price for the asset, if no price then
        # raise a ValueError
        price_err_msg = (
            "Could not obtain a latest market price for "
            "Asset with ticker symbol '%s'. Order with ID '%s' was "
            "not executed." % (
                order.asset, order.order_id
            )
        )
        bid_ask = self.data_handler.get_asset_latest_bid_ask_price(dt, order.asset, complain=True)
        if bid_ask == (np.NaN, np.NaN):
            raise ValueError(price_err_msg)

        # Calculate the consideration and total commission
        # based on the commission model
        if order.direction > 0:
            price = bid_ask[1]
        else:
            price = bid_ask[0]

        assert price > 0

        consideration = round(price * order.quantity)
        total_commission = self.fee_model.calc_total_cost(
            order.asset, order.quantity, consideration, self
        )

        # Check that sufficient cash exists to carry out the
        # order, else scale it down
        est_total_cost = consideration + total_commission
        total_cash = self.portfolios[portfolio_id].cash

        scaled_quantity = order.quantity
        if est_total_cost > total_cash:
            if settings.PRINT_EVENTS:
                logger.warning(
                    "WARNING: Estimated transaction size of %0.2f exceeds "
                    "available cash of %0.2f. Transaction will still occur "
                    "with a negative cash balance." % (est_total_cost, total_cash)
                )

        # Create a transaction entity and update the portfolio
        txn = Transaction(
            order.asset, scaled_quantity, self.current_dt,
            price, order.order_id, commission=total_commission,
            debug_details=debug_details,
        )
        self.portfolios[portfolio_id].transact_asset(txn)
        if settings.PRINT_EVENTS:
            cash_after = self.portfolios[portfolio_id].cash
            logger.info("(%s) - executed %s order: %s, qty: %s, price: %0.8f, consideration: %0.2f, commission: %0.2f, total: %0.2f, cash after %0.2f",
                self.current_dt,
                order.get_order_direction(),
                order.asset,
                scaled_quantity,
                price,
                consideration,
                total_commission,
                consideration + total_commission,
                cash_after)

    def submit_order(self, portfolio_id, order, debug_details):
        """
        Execute an Order instance against the sub-portfolio
        with ID 'portfolio_id'. For the SimulatedBroker class
        specifically there are no restrictions on this occuring
        beyond having sufficient cash in the sub-portfolio to
        allow this to occur.

        This does not take into settlement dates, as with most
        brokerage accounts. The cash is taken immediately upon
        entering a long position and returned immediately upon
        closing out the position.

        Parameters
        ----------
        portfolio_id : `str`
            The portfolio ID string.
        order : `Order`
            The Order instance to submit.
        """
        # Check that the portfolio actually exists
        if portfolio_id not in self.portfolios.keys():
            raise KeyError(
                "Portfolio with ID '%s' does not exist. Order with "
                "ID '%s' was not executed." % (
                    portfolio_id, order.order_id
                )
            )
        self.open_orders[portfolio_id].put(order)
        if settings.PRINT_EVENTS:
            logger.info(
                "(%s) - submitted order: %s, qty: %s" % (
                    self.current_dt, order.asset, order.quantity
                )
            )

    def update(self, dt, execute_orders=False, debug_details: Optional[Dict]=None):
        """
        Updates the current SimulatedBroker timestamp.

        Parameters
        ----------
        dt : `pd.Timestamp`
            The current timestamp to update the Broker to.
        """


        self.current_dt = dt

        # Update portfolio asset values
        logger.info("(%s) repricing the assets", dt)
        for portfolio in self.portfolios:
            for asset in self.portfolios[portfolio].pos_handler.positions:
                mid_price = self.data_handler.get_asset_latest_mid_price(
                    dt, asset, complain=True,
                )
                self.portfolios[portfolio].update_market_value_of_asset(
                    asset, mid_price, self.current_dt
                )

        # Try to execute orders
        if execute_orders:
            logger.info("(%s) executing orders", dt)
            # if self.exchange.is_open_at_datetime(self.current_dt):
            if True: # Crypto is open 24/7
                orders = []
                for portfolio in self.portfolios:
                    while not self.open_orders[portfolio].empty():
                        orders.append(
                            (portfolio, self.open_orders[portfolio].get())
                        )

                sorted_orders = sorted(orders, key=lambda x: x[1].direction)
                for portfolio, order in sorted_orders:
                    self._execute_order(dt, portfolio, order, debug_details)

    def to_dict(self) -> dict:
        """Export the current broker state.

        Semi human-readable info of the current broker state.

        Most useful for debugging.
        """

        open_orders_per_portfolio = defaultdict(list)
        for id, queue in self.open_orders.items():
            # https://stackoverflow.com/a/47887265/315168
            for order in list(queue.queue):
                open_orders_per_portfolio[id].append(order.to_dict())

        export = {
            # Dict of currencies and their balances
            "cash_balances": self.cash_balances.copy(),
            "portfolios": {id: portfolio.to_dict() for id, portfolio in self.portfolios.items()},
            "open_orders": open_orders_per_portfolio,
        }
        return export