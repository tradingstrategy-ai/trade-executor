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
                       state: State, trade:
            TradeExecution) -> Tuple[Decimal, Decimal]:
        """Set backtesting trade state from planned to executed.
        
        Currently, always executes trades "perfectly" i.e. no different slipppage
        that was planned, etc.
        """

        assert trade.get_status() == TradeStatus.started

        state.mark_broadcasted(ts, trade)

        # Check that the trade "executes" against the simulated wallet
        base = trade.pair.base
        quote = trade.pair.quote
        reserve = trade.reserve_currency

        base_balance = self.wallet.get_balance(base.address)
        quote_balance = self.wallet.get_balance(quote.address)
        reserve_balance = self.wallet.get_balance(reserve.address)

        position = state.portfolio.get_existing_open_position_by_trading_pair(trade.pair)

        if trade.is_buy():
            executed_reserve = trade.planned_reserve
            executed_quantity = trade.planned_quantity * Decimal(1 - self.lp_fees)
        else:
            assert position and position.is_open(), f"Tried to execute sell on position that is not open: {trade}"
            executed_quantity = trade.planned_quantity
            executed_reserve = abs(Decimal(trade.planned_quantity) * Decimal(trade.planned_price) * Decimal(1 + self.lp_fees))
        try:
            if trade.is_buy():
                self.wallet.update_balance(reserve.address, -executed_reserve)
                self.wallet.update_balance(base.address, executed_quantity)
            else:
                self.wallet.update_balance(base.address, executed_quantity)
                self.wallet.update_balance(reserve.address, executed_reserve)

        except OutOfSimulatedBalance as e:
            # Better error messages to helping out why backtesting failed
            raise BacktestExecutionFailed(
                f"Execution of trade {trade} failed.\n"
                f"Trade type: {trade.trade_type.name}.\n"
                f"Wallet base balance: {base_balance} {base.token_symbol}.\n"
                f"Wallet quote balance: {quote_balance} {quote.token_symbol}.\n"
                f"Wallet reserve balance: {reserve_balance} {reserve.token_symbol}.\n"
                f"Executed base amount: {executed_quantity} {base.token_symbol}\n"
                f"Executed reserve amount: {executed_reserve} {reserve.token_symbol}\n"
                f"Planned base amount: {trade.planned_quantity} {base.token_symbol}\n"
                f"Planned reserve amount: {trade.planned_reserve} {reserve.token_symbol}\n"
                f"Position quantity: {position and position.get_quantity() or '-'} {base.token_symbol}\n"
                f"Out of balance: {e}\n"
            ) from e

        assert abs(executed_quantity) > 0
        assert executed_reserve > 0
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

        state.start_trades(datetime.datetime.utcnow(), trades, max_slippage=0)

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

        for trade in trades:

            # 3. Simulate tx broadcast
            executed_quantity, executed_reserve = self.simulate_trade(ts, state, trade)

            # 4. execution is dummy operation where planned execution becomes actual execution
            # Assume we always get the same execution we planned
            executed_price = float(abs(executed_reserve / executed_quantity))

            state.mark_trade_success(
                ts,
                trade,
                executed_price,
                executed_quantity,
                executed_reserve,
                lp_fees=0,
                native_token_price=1)

    def get_routing_state_details(self) -> dict:
        return {"wallet": self.wallet}

