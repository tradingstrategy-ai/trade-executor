"""Trade routing instructions.

Each trading universe and strategy can have different trade routing set,
based on the exchanges the universe covers.

Here we define the abstract overview of routing.
"""
import abc
from decimal import Decimal
from typing import List

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.trade import TradeExecution


class CannotRouteTrade(Exception):
    """The router does not know who to execute a trade decided by a strategy."""


class RoutingState(abc.ABC):
    """Keep the track record of already done transactions.

    When performing multiple blockchain transactions for multiple trades
    in one cycle, we need to know what approvals and such we have already done.
    """


class RoutingModel(abc.ABC):
    """Trade roouting model base class.

    Nothing done here - check the subclasses.
    """

    @abc.abstractmethod
    def execute_trades(self, state: RoutingState, trades: List[TradeExecution], check_balances=False):
        """Executes the trades decided by a strategy.

        - Decides the best way, or a way, to execute a trade

        - Trade instances are mutated in-place

        :param check_balances:
            Check that the wallet has enough reserves to perform the trades
            before executing them. Because we are selling before buying.
            sometimes we do no know this until the sell tx has been completed.

        :raise CannotExecuteTrade:
            If a trade cannot be executed, e.g. due to an unsupported pair or an exchange,
        """

    @abc.abstractmethod
    def estimate_price(self,
                       state: RoutingState,
                       pair: TradingPairIdentifier,
                       quantity: Decimal,
                       buy_side=True):
        """Estimate the price of an asset.

        - Price impact and fees are included

        Used by the position revaluator to come up with the new prices
        for the assets on every tick.
        """

