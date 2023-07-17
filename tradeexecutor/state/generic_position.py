"""Base class for trading and reserve positions."""

from _decimal import Decimal
from abc import abstractmethod, ABC
from typing import Iterable

from tradeexecutor.state.balance_update import BalanceUpdate
from tradeexecutor.state.types import USDollarAmount



class BalanceUpdateEventAlreadyAdded(Exception):
    """Tries to include the same balance update event twice.

    See :py:meth:`GenericPosition.add_balance_update_event`
    """


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

    @abstractmethod
    def get_balance_update_events(self) -> Iterable[BalanceUpdate]:
        """Iterate over all balance update events.

        Balance updates describe external events affecting the balance of this position:
        the update was not triggered by the trade executor itself.

        - Deposits

        - Redemptions

        - Account corrections

        - **Trades** are not included here
        """

    @abstractmethod
    def add_balance_update_event(self, event: BalanceUpdate):
        """Include a new balance update event

        :raise BalanceUpdateEventAlreadyAdded:
            In the case of a duplicate and event id is already used.
        """
