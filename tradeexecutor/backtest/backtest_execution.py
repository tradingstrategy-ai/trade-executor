"""Execution model where trade happens directly on Uniswap v2 style exchange."""

import datetime
from decimal import Decimal
from typing import List, Tuple
import logging

from tradeexecutor.backtest.backtest_routing import BacktestRoutingModel, BacktestRoutingState
from tradeexecutor.backtest.simulated_wallet import SimulatedWallet, OutOfSimulatedBalance
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeStatus
from tradeexecutor.strategy.execution_model import ExecutionModel, AutoClosingOrderUnsupported

logger = logging.getLogger(__name__)


class BacktestExecutionFailed(Exception):
    """Something went wrong in the backtest simulation."""


def fix_sell_token_amount(
        current_balance: Decimal,
        order_quantity: Decimal,
        epsilon=Decimal(10**-9)
) -> Tuple[Decimal, bool]:
    """Fix rounding errors that may cause wallet dust overflow.

    TODO: This should be handled other part of the system.

    :return:
        (new amount, was fixed) tuple
    """

    assert isinstance(current_balance, Decimal)
    assert isinstance(order_quantity, Decimal)
    assert order_quantity < 0

    # Not trying to sell more than we have
    if abs(order_quantity) <= current_balance:
        return order_quantity, False

    # We are trying to sell more we have
    diff = abs(current_balance + order_quantity)
    if diff <= epsilon:
        # Fix to be within the epsilon diff
        logger.warning("Fixing token sell amount to be within the epsilon. Wallet balance: %s, sell order quantity: %s, diff: %s",
                       current_balance,
                       order_quantity,
                       diff
                       )
        return -current_balance, True

    logger.warning("Trying to sell more than we have. Wallet balance: %s, sell order quantity: %s, diff: %s, epsilon: %s",
                   current_balance,
                   order_quantity,
                   diff,
                   epsilon
                   )
    return order_quantity, False


class BacktestExecutionModel(ExecutionModel):
    """Simulate trades against historical data."""

    def __init__(self,
                 wallet: SimulatedWallet,
                 max_slippage: float,
                 lp_fees: float=0.0030,
                 stop_loss_data_available=False,
                 ):
        self.wallet = wallet
        self.max_slippage = max_slippage
        self.lp_fees = lp_fees
        self.stop_loss_data_available = stop_loss_data_available

    def is_live_trading(self):
        return False

    def is_stop_loss_supported(self):
        return self.stop_loss_data_available

    def preflight_check(self):
        pass

    def initialize(self):
        """Set up the wallet"""
        logger.info("Initialising backtest execution model")

    def simulate_trade(self,
                       ts: datetime.datetime,
                       state: State,
                       idx: int,
                       trade: TradeExecution) -> Tuple[Decimal, Decimal]:
        """Set backtesting trade state from planned to executed.
        
        Currently, always executes trades "perfectly" i.e. no different slipppage
        that was planned, etc.

        :poram ts:
            Strategy cycle timestamp

        :param state:
            Current backtesting state

        :param idx:
            Index of the trade to be executed on this cycle

        :param trade:
            The actual trade

        :return:
            Executed quantity and executed reserve amounts
        """

        assert trade.get_status() == TradeStatus.started

        # In the backtesting simulation,
        # execution happens always perfectly
        # without any lag
        trade.started_at = trade.opened_at

        state.mark_broadcasted(ts, trade)

        # Check that the trade "executes" against the simulated wallet
        base = trade.pair.base
        quote = trade.pair.quote
        reserve = trade.reserve_currency

        base_balance = self.wallet.get_balance(base.address)
        quote_balance = self.wallet.get_balance(quote.address)
        reserve_balance = self.wallet.get_balance(reserve.address)

        position = state.portfolio.get_existing_open_position_by_trading_pair(trade.pair)

        sell_amount_epsilon_fix = False
        if trade.is_buy():
            executed_reserve = trade.planned_reserve
            executed_quantity = trade.planned_quantity
        else:
            assert position and position.is_open(), f"Tried to execute sell on position that is not open: {trade}"
            executed_quantity, sell_amount_epsilon_fix = fix_sell_token_amount(base_balance, trade.planned_quantity)
            executed_reserve = abs(Decimal(trade.planned_quantity) * Decimal(trade.planned_price))
        try:
            if trade.is_buy():
                self.wallet.update_balance(reserve.address, -executed_reserve)
                self.wallet.update_balance(base.address, executed_quantity)
            else:
                self.wallet.update_balance(base.address, executed_quantity)
                self.wallet.update_balance(reserve.address, executed_reserve)

        except OutOfSimulatedBalance as e:
            # Better error messages to helping out why backtesting failed

            if trade.is_buy():
                # Give a hint to the user
                extra_help_message = f"---\n" \
                                     f"Tip:" \
                                     f"This is a buy trade that failed.\n" \
                                     f"It means that the strategy had less cash to make purchases that it expected.\n" \
                                     f"It may happen during multiple rebalance operations, as the strategy model might not account properly the trading fees when\n" \
                                     f"it estimates the available cash in hand to make buys and sells for rebalancing operations.\n" \
                                     f"Try increasing the strategy cash buffer to see if it solves the problem.\n"
            else:
                extra_help_message = ""

            raise BacktestExecutionFailed(f"\n"
                f"  Trade #{idx} failed on strategy cycle {ts}\n"
                f"  Execution of trade {trade} failed.\n"
                f"  Pair: {trade.pair}.\n"
                f"  Trade type: {trade.trade_type.name}.\n"
                f"  Trade quantity: {trade.planned_quantity}, reserve: {trade.planned_reserve} {trade.reserve_currency}.\n"
                f"  Wallet base balance: {base_balance} {base.token_symbol} ({base.address}).\n"
                f"  Wallet quote balance: {quote_balance} {quote.token_symbol} ({quote.address}).\n"
                f"  Wallet reserve balance: {reserve_balance} {reserve.token_symbol} ({reserve.address}).\n"
                f"  Executed base amount: {executed_quantity} {base.token_symbol} ({base.address})\n"
                f"  Executed reserve amount: {executed_reserve} {reserve.token_symbol} ({reserve.address})\n"
                f"  Planned base amount: {trade.planned_quantity} {base.token_symbol} ({base.address})\n"
                f"  Planned reserve amount: {trade.planned_reserve} {reserve.token_symbol} ({reserve.address})\n"
                f"  Existing position quantity: {position and position.get_quantity() or '-'} {base.token_symbol}\n"
                f"  Sell amount epsilon fix applied: {sell_amount_epsilon_fix}.\n"
                f"  Out of balance: {e}\n"
                f"  {extra_help_message}\n"
            ) from e

        assert abs(executed_quantity) > 0, f"Expected executed_quantity for the trade to be above zero, got executed_quantity:{executed_quantity}, planned_quantity:{trade.planned_quantity}, trade is {trade}"
        assert executed_reserve > 0, f"Expected executed_reserve for the trade to be above zero, got {executed_reserve}"
        return executed_quantity, executed_reserve

    def execute_trades(self,
                       ts: datetime.datetime,
                       state: State,
                       trades: List[TradeExecution],
                       routing_model: BacktestRoutingModel,
                       routing_state: BacktestRoutingState,
                       check_balances=False):
        """Execute the trades on a simulated environment.

        Calculates price impact based on historical data
        and fills the expected historical trade output.

        :param check_balances:
            Raise an error if we run out of balance to perform buys in some point.
        """
        assert isinstance(ts, datetime.datetime)
        assert isinstance(routing_model, BacktestRoutingModel)
        assert isinstance(routing_state, BacktestRoutingState)

        state.start_trades(ts, trades, max_slippage=0)

        routing_model.setup_trades(
            routing_state,
            trades,
            check_balances=check_balances)

        # Check that backtest does not try to execute stop loss / take profit
        # trades when data is not available
        for t in trades:
            position = state.portfolio.open_positions.get(t.position_id)
            if position and position.has_automatic_close():
                # Check that we have stop loss data available
                # for backtesting
                if not self.is_stop_loss_supported():
                    raise AutoClosingOrderUnsupported("Trade was marked with stop loss/take profit even though backtesting trading universe does have price feed for stop loss checks available.")

        for idx, trade in enumerate(trades):

            # 3. Simulate tx broadcast
            executed_quantity, executed_reserve = self.simulate_trade(ts, state, idx, trade)

            # 4. execution is dummy operation where planned execution becomes actual execution
            # Assume we always get the same execution we planned
            executed_price = float(abs(executed_reserve / executed_quantity))

            state.mark_trade_success(
                ts,
                trade,
                executed_price,
                executed_quantity,
                executed_reserve,
                lp_fees=trade.lp_fees_estimated,
                native_token_price=1)

    def get_routing_state_details(self) -> dict:
        return {"wallet": self.wallet}

    def repair_unconfirmed_trades(self, state: State) -> List[TradeExecution]:
        raise NotImplementedError()