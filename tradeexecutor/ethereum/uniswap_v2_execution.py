import datetime
from typing import List

from web3 import Web3

from smart_contracts_for_testing.hotwallet import HotWallet
from smart_contracts_for_testing.uniswap_v2 import UniswapV2Deployment
from tradeexecutor.ethereum.execution import approve_tokens, prepare_swaps, confirm_approvals, broadcast, \
    wait_trades_to_complete, resolve_trades
from tradeexecutor.state.state import TradeExecution, State
from tradeexecutor.strategy.execution import ExecutionModel


class UniswapV2ExecutionModel(ExecutionModel):
    """Run order execution for uniswap v2 style exchanges."""

    def __init__(self, state: State, uniswap: UniswapV2Deployment, hot_wallet: HotWallet):
        self.state = state
        self.web3 = uniswap.web3
        self.uniswap = uniswap
        self.hot_wallet = hot_wallet

    def execute_trades(self, ts: datetime.datetime, trades: List[TradeExecution]):

        assert isinstance(ts, datetime.datetime)

        # 2. Capital allocation
        # Approvals
        approvals = approve_tokens(
            self.web3,
            self.uniswap,
            self.hot_wallet,
            trades
        )

        # 2: prepare
        # Prepare transactions
        prepare_swaps(
            self.web3,
            self.hot_wallet,
            self.uniswap,
            ts,
            self.state,
            trades
        )

        #: 3 broadcast

        # Handle approvals separately for now
        confirm_approvals(self.web3, approvals)

        broadcasted = broadcast(self.web3, ts, trades)
        #assert trade.get_status() == TradeStatus.broadcasted

        # Resolve
        receipts = wait_trades_to_complete(self.web3, trades)
        resolve_trades(
            self.web3,
            self.uniswap,
            ts,
            self.state,
            broadcasted,
            receipts)
