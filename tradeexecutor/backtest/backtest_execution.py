"""Execution model where trade happens directly on Uniswap v2 style exchange."""

import datetime
from decimal import Decimal
from typing import List
import logging

from web3 import Web3

from eth_defi.hotwallet import HotWallet
from tradeexecutor.ethereum.execution import broadcast_and_resolve
from tradeexecutor.ethereum.uniswap_v2_routing import UniswapV2SimpleRoutingModel, UniswapV2RoutingState
from tradeexecutor.state.freeze import freeze_position_on_failed_trade
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.execution_model import ExecutionModel


logger = logging.getLogger(__name__)


class BacktestxecutionModel(ExecutionModel):
    """Simulate trades against historical data."""

    def is_live_trading(self):
        return False

    def preflight_check(self):
        pass

    def initialize(self):
        """Set up the wallet"""
        logger.info("Initialising backtest execution model")

    def execute_trades(self,
                       ts: datetime.datetime,
                       state: State,
                       trades: List[TradeExecution],
                       routing_model: UniswapV2SimpleRoutingModel,
                       routing_state: UniswapV2RoutingState,
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

        routing_model.execute_trades(
            routing_state,
            trades,
            check_balances=check_balances)

    def get_routing_state_details(self) -> dict:
        return {}

