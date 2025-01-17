"""Approve new trades in the console."""

from typing import List
import textwrap

from prompt_toolkit import print_formatted_text, HTML
from prompt_toolkit.shortcuts import checkboxlist_dialog, message_dialog

from tradeexecutor.state.state import State
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.approval import ApprovalModel


class CLIApprovalModel(ApprovalModel):
    """Confirm trades in the CLI before they go through.

    The terminal execution of the bot stops until the user confirms the trades.

    If no one is there to press a key then nothing happens.
    """

    def render_portfolio(self, portfolio: Portfolio) -> HTML:
        """Render the current portfolio holdings using ANSI formatting.

        https://python-prompt-toolkit.readthedocs.io/en/master/pages/printing_text.html

        :return: promp_toolkit HTML for displaying the portfolio
        """
        equity = portfolio.calculate_total_equity()
        cash = portfolio.get_cash()
        text = textwrap.dedent(f"""
            Total equity <ansigreen>${equity:,.2f}</ansigreen>
            Current cash <ansigreen>${cash:,.2f}</ansigreen>""")
        text += '\n'

        positions: List[TradingPosition] = list(portfolio.get_executed_positions())

        if positions:
            for tp in positions:
                text += f"<b>{tp.get_name()}</b>: <ansigreen>${tp.get_value():,.2f}</ansigreen> at quantity of <ansigreen>{tp.get_quantity()} {tp.get_quantity_unit_name()}</ansigreen>\n"
        else:
            text += f"<ansired>No open positions</ansired>"

        return HTML(text)

    def confirm_trades(self, state: State, trades: List[TradeExecution]) -> List[TradeExecution]:
        """Create a checkbox list to approve trades using prompt_toolkit.

        See https://python-prompt-toolkit.readthedocs.io/en/master/pages/dialogs.html#checkbox-list-dialog

        :param state: The current trade execution state
        :param trades: New trades to be executed
        :return: What trades went through human approval
        """

        # Show the user what we got
        text = self.render_portfolio(state.portfolio)

        new_trades = [
            (t.trade_id, t.get_human_description())
            for t in trades
        ]

        # Did not detect any new trades
        if len(new_trades) == 0:
            message_dialog(
                title='No new trades to execute',
                text=text).run()
            return []

        approvals_dialog = checkboxlist_dialog(
            title="New trades to execute",
            text=text,
            values=new_trades,
        )
        approvals = approvals_dialog.run()

        return [t for t in trades if t.trade_id in approvals]


