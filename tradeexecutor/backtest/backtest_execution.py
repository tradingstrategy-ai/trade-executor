"""Execution model where trade happens directly on Uniswap v2 style exchange."""

import datetime
from decimal import Decimal
from typing import List, Tuple
import logging

from tradeexecutor.backtest.backtest_routing import BacktestRoutingModel, BacktestRoutingState
from tradeexecutor.backtest.simulated_wallet import SimulatedWallet
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeStatus
from tradeexecutor.strategy.execution_model import ExecutionModel


logger = logging.getLogger(__name__)


class BacktestExecutionModel(ExecutionModel):
    """Simulate trades against historical data."""

    def __init__(self, wallet: SimulatedWallet, max_slippage: float, lp_fees: float=0.0030):
        self.wallet = wallet
        self.max_slippage = max_slippage
        self.lp_fees = lp_fees

    def is_live_trading(self):
        return False

    def preflight_check(self):
        pass

    def initialize(self):
        """Set up the wallet"""
        logger.info("Initialising backtest execution model")

    def simulate_trade(self, ts: datetime.datetime, state: State, trade: TradeExecution) -> Tuple[Decimal, Decimal]:

        assert trade.get_status() == TradeStatus.started

        state.mark_broadcasted(ts, trade)

        # Check that the trade "executes" against the simulated wallet
        base = trade.pair.base
        quote = trade.pair.quote
        reserve = trade.reserve_currency

        if trade.is_buy():
            executed_reserve = trade.planned_reserve
            executed_quantity = trade.planned_quantity * Decimal(1 - self.lp_fees)
            self.wallet.update_balance(reserve.address, -executed_reserve)
            self.wallet.update_balance(base.address, executed_quantity)
        else:
            executed_quantity = trade.planned_quantity
            executed_reserve = abs(Decimal(trade.planned_quantity) * Decimal(trade.planned_price) * Decimal(1 + self.lp_fees))
            self.wallet.update_balance(base.address, executed_quantity)
            self.wallet.update_balance(reserve.address, executed_reserve)

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

        for trade in trades:

            # 3. Simulate tx broadcast
            executed_quantity, executed_reserve = self.simulate_trade(ts, state, trade)

            # 4. execution is dummy operation where planned execution becomes actual execution
            # Assume we always get the same execution we planned
            executed_price = float(abs(executed_reserve / executed_quantity))

            state.mark_trade_success(ts, trade, executed_price, executed_quantity, executed_reserve, lp_fees=0, native_token_price=1)

    def get_routing_state_details(self) -> dict:
        return {"wallet": self.wallet}

