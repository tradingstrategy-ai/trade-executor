from typing import List

from tradeexecutor.ethereum.execution import approve_tokens, prepare_swaps, confirm_approvals, broadcast, \
    wait_trades_to_complete, resolve_trades
from tradeexecutor.state.state import TradeExecution
from tradeexecutor.strategy.execution import ExecutionModel


class UniswapV2ExecutionModel(ExecutionModel):
    """Run order execution for uniswap v2 style exchanges."""

    def __init__(self):
        pass

    def execute_trades(self, trades: List[TradeExecution]):

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
            self.ts,
            self.state,
            trades
        )

        #: 3 broadcast

        # Handle approvals separately for now
        confirm_approvals(self.web3, approvals)

        broadcasted = broadcast(self.web3, self.ts, trades)
        #assert trade.get_status() == TradeStatus.broadcasted

        # Resolve
        receipts = wait_trades_to_complete(self.web3, trades)
        resolve_trades(
            self.web3,
            self.uniswap,
            self.ts,
            self.state,
            broadcasted,
            receipts)
