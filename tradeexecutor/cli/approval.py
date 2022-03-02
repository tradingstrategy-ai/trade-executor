from typing import List

from tradeexecutor.state.state import State, TradeExecution
from tradeexecutor.strategy.approval import ApprovalModel


class CLIApprovalModel(ApprovalModel):
    """Confirm trades in the CLI before they go through.

    The terminal execution of the bot stops until the user confirms the trades.

    If no one is there to press a key then nothing happens.
    """

    def render_trades(self):
        pass

    def confirm_trades(self, state: State, trades: List[TradeExecution]) -> List[TradeExecution]:
        """
        :param state:
        :param trades:
        :return:
        """
        import ipdb ; ipdb.set_trace()


