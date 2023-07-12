"""Base class for trading and reserve positions."""

from _decimal import Decimal
from abc import abstractmethod

from black.trans import ABC

from tradeexecutor.state.types import USDollarAmount


class GenericPosition(ABC):
    """Base class for trading and reserve positions.

    Implements common method all positions need to implement.

    TODO: How to define generic `balance_updates` mapping.

    See also

    - :py:class:`tradeexecutor.state.state.position.TradingPosition`

    - :py:class:`tradeexecutor.state.state.reserve.ReservePosition`

    """

    @abstractmethod
    def get_quantity(self) -> Decimal:
        """Get the number of tokens held in this position."""

    @abstractmethod
    def calculate_quantity_usd_value(self, quantity: Decimal) -> USDollarAmount:
        """Price token amount in this position in US dollar.

        An estimation. Use whatever latest exchange rate that is appropriate.

        :return:
            Dollar amount
        """

    @abstractmethod
    def get_balance_update_quantity(self) -> Decimal:
        """Get quantity of all balance updates for this position.

        Balance update events are

        - Deposits

        - Redemptions

        - Accounting corrections

        ... but not trades.

        :return:
            What's the total value of non-trade events affecting the balances of this position.
        """
