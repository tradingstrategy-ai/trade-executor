"""Trade approval models.

Trade execution can have a separate approval step

- :py:class:`UncheckedApprovalModel`: all trades are automatically executed

- :py:class:`tradeexecutor.cli.approval.CLIApprovalModel`:
  trades need to be approved in a command line text user interface (TUI)

"""

import abc
import enum
from typing import List

from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution


class ApprovalType(enum.Enum):
    """What kind of approval model the trade executor uses."""

    unchecked = "unchecked"

    cli = "cli"


class ApprovalModel(abc.ABC):
    """A model that defines a checkpoint for trades before they are executd.

    Approval can be get in various ways
    - CLI confirmation by human
    - Web interface confirmation by human
    - A third party automated risk management system
    """

    def confirm_trades(self, state: State, trades: List[TradeExecution]) -> List[TradeExecution]:
        """

        This function may wait forever and block the trade execution thread.
        It is up to the approval internals to decide when to timeout and what to od in such situation.

        :param state: Current portfolio management states
        :param trades: Trades suggested bt the strategy
        :return: Filtered list of trades that should be executed
        """


class UncheckedApprovalModel(ApprovalModel):
    """Approval model where all the trades are automatically approved."""

    def confirm_trades(self, state: State, trades: List[TradeExecution]) -> List[TradeExecution]:
        return trades